import base64
import json
import os
import sys
import time
import tempfile
from threading import Lock
from typing import Dict, Optional, Union
import uuid

import numpy as np

from leptonai.photon import Photon, FileParam, HTTPException
from loguru import logger

# Note: instead of importing whisperx in the main file, we import it in the functions that
# actually use whisperx. This enables local users who do not have whisperx installed to
# still create the photon and run remotely.
# import whisperx


class WhisperXBackground(Photon):
    """
    A WhisperX photon that serves the [WhisperX](https://github.com/m-bain/whisperX) model.

    The photon exposes two endpoints: `/run` and `/run_upload` that deals with files/urls
    and uploaded contents respectively. See the docs of each for details.

    Different from the main WhisperX photon, this photon starts background tassks in the
    background, so it handles requests more efficiently. For example, for the main photon,
    if you run a prediction, it will block the server until the prediction is done, giving
    a pretty bad user experience. This photon, on the other hand, will return immediately
    with a task id. The user can then use the task id to query the status of the task, and
    get the result when it is done.
    """

    requirement_dependency = [
        "leptonai",
        "torch",
        "torchaudio",
        "git+https://github.com/m-bain/whisperx.git",
    ]

    system_dependencies = ["ffmpeg"]

    # Parameters for the photon.
    # The photon will need to have a storage folder
    OUTPUT_ROOT = os.environ.get("WHISPERX_OUTPUT_ROOT", "/tmp/whisperx")
    INPUT_FILE_EXTENSION = ".npy"
    OUTPUT_FILE_EXTENSION = ".json"
    OUTPUT_MAXIMUM_AGE = 60 * 60 * 24  # 1 day
    CLEANUP_INTERVAL = 60 * 60  # 1 hour
    LAST_CLEANUP_TIME = time.time()
    SUPPORTED_LANGUAGES = {"en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"}

    def init(self):
        import whisperx

        logger.info("Initializing WhisperXPhoton")

        # 0. Create output root, and launch a thread to clean up old files
        os.makedirs(self.OUTPUT_ROOT, exist_ok=True)

        # 1. Load whisper model
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
        if not self.hf_token:
            logger.error("Please set the environment variable HUGGING_FACE_HUB_TOKEN.")
            sys.exit(1)
        self.device = "cuda"
        compute_type = "float16"
        self._model = whisperx.load_model(
            "large-v2", self.device, compute_type=compute_type
        )

        # 2. load whisper align model
        self._model_a = {}
        self._metadata = {}
        self._model_a["en"], self._metadata["en"] = whisperx.load_align_model(
            language_code="en", device=self.device
        )
        self.align_model_lock = Lock()
        self._diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )

    def _gen_unique_filename(self) -> str:
        return str(uuid.uuid4())

    def _regular_clean_up(self):
        if time.time() - self.LAST_CLEANUP_TIME < self.CLEANUP_INTERVAL:
            return
        logger.info(f"Cleaning up {self.OUTPUT_ROOT} regularly")
        self.LAST_CLEANUP_TIME = time.time()
        for filename in os.listdir(self.OUTPUT_ROOT):
            if filename.endswith(self.OUTPUT_FILE_EXTENSION):
                filepath = os.path.join(self.OUTPUT_ROOT, filename)
                # Checks if files are older than 1 hour. If so, delete them.
                if (
                    os.path.isfile(filepath)
                    and time.time() - os.path.getmtime(filepath)
                    > self.OUTPUT_MAXIMUM_AGE
                ):
                    os.remove(filepath)

    def _run_whisperx(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        transcribe_only: bool = False,
        task_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        The main function that is called by the others.
        """
        import whisperx

        batch_size = 16
        start_time = time.time()

        audio_filename = None
        if isinstance(audio, str):
            # Load the numpy array from the file
            audio_filename = audio
            logger.debug(f"Loading audio from file {audio_filename}.")
            audio = np.load(audio)
        logger.debug(f"started processing audio of length {len(audio)}.")
        if task_id:
            logger.debug(f"task_id: {task_id}")
        result = self._model.transcribe(audio, batch_size=batch_size, language=language)
        if len(result["segments"]) == 0:
            logger.debug("Empty result from whisperx. Directly return empty.")
            return []

        if not transcribe_only:
            with self.align_model_lock:
                if result["language"] not in self._model_a:
                    (
                        self._model_a[result["language"]],
                        self._metadata[result["language"]],
                    ) = whisperx.load_align_model(
                        language_code=result["language"], device=self.device
                    )
            result = whisperx.align(
                result["segments"],
                self._model_a[result["language"]],
                self._metadata[result["language"]],
                audio,
                self.device,
                return_char_alignments=False,
            )
            # When there is no active diarization, the diarize model throws a KeyError.
            # In this case, we simply skip diarization.
            try:
                if (
                    min_speakers
                    and max_speakers
                    and min_speakers <= max_speakers
                    and min_speakers > 0
                ):
                    diarize_segments = self._diarize_model(
                        audio, min_speakers=min_speakers, max_speakers=max_speakers
                    )
                else:
                    # ignore the hint and do diarization.
                    diarize_segments = self._diarize_model(audio)
            except Exception as e:
                logger.error(f"Error in diarization: {e}. Skipping diarization.")
            else:
                result = whisperx.assign_word_speakers(diarize_segments, result)

        if audio_filename:
            os.remove(audio_filename)
        if task_id is None:
            total_time = time.time() - start_time
            logger.debug(
                f"finished processing task {task_id}. Audio len: {audio.size} Total"
                f" time: {total_time} ({audio.size / 16000 / total_time} x realtime)"
            )
            return result["segments"]
        else:
            # return result["segments"]  # segments are now assigned speaker IDs
            output_filepath = os.path.join(
                self.OUTPUT_ROOT, task_id + self.OUTPUT_FILE_EXTENSION
            )
            json.dump(result["segments"], open(output_filepath, "w"))
            self._regular_clean_up()
            total_time = time.time() - start_time
            logger.debug(
                f"finished processing task {task_id}. Audio len: {audio.size} Total"
                f" time: {total_time} ({audio.size / 16000 / total_time} x realtime)"
            )
            return

    @Photon.handler(
        example={
            "input": (
                "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
            ),
            "language": "en",
            "transcribe_only": True,
        }
    )
    def run(
        self,
        input: Union[FileParam, str],
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        transcribe_only: bool = False,
    ) -> Dict:
        """
        Runs transcription, alignment, and diarization for the input.

        - Inputs:
            - input: a url containing the audio file, or a base64-encoded string containing an
                audio file, or a lepton.photon.FileParam, or a local file path if running
                locally.
            - language(optional): the language code for the input. If not provided, the model
                will try to detect the language automatically (note this runs more slowly)
            - min_speakers(optional): the hint for minimum number of speakers for diarization.
            - max_speakers(optional): the hint for maximum number of speakers for diarization.
            - transcribe_only(optional): if True, only transcribe the audio, and skip alignment
                and diarization.

        - Returns:
            - result: if the input audio is less than 60 seconds, we will directly return the
                result.
            - task: a dictionary with key `task_id` and value the task uuid. Use `status(**task)`
                to query the task status, and `get_result(**task)` to get the result when the
                status is "ok".
        """
        import whisperx

        if language is not None and language not in self.SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported language: {language}. Supported languages:"
                f" {self.SUPPORTED_LANGUAGES}",
            )
        if min_speakers is not None and min_speakers < 1:
            raise HTTPException(400, f"min_speakers must be >= 1, got {min_speakers}")
        if max_speakers is not None and max_speakers < 1:
            raise HTTPException(400, f"max_speakers must be >= 1, got {max_speakers}")
        if (
            min_speakers is not None
            and max_speakers is not None
            and min_speakers > max_speakers
        ):
            raise HTTPException(
                400,
                f"min_speakers must be <= max_speakers, got {min_speakers} >"
                f" {max_speakers}",
            )

        try:
            if isinstance(input, FileParam):
                # We write the content at input.file to a temporary file, then call whisperx.load_audio
                # to load the audio from the temporary file.
                with tempfile.NamedTemporaryFile() as f:
                    f.write(input.file.read())
                    f.flush()
                    filename = f.name
                    audio = whisperx.load_audio(filename)
            elif input.startswith("http://") or input.startswith("https://"):
                # This is a url, we can directly pass it to whisperx.load_audio
                audio = whisperx.load_audio(input)
            else:
                # As a fallback option, we will assume that this is a base64 encoded string.
                # We write the content at input to a temporary file, then call whisperx.load_audio
                # to load the audio from the temporary file.
                if input.startswith("data:audio/wav;base64,"):
                    input = input[22:]
                with tempfile.NamedTemporaryFile() as f:
                    decoded_data = base64.b64decode(input)
                    f.write(decoded_data)
                    f.flush()
                    filename = f.name
                    audio = whisperx.load_audio(filename)
        except Exception:
            raise HTTPException(
                400,
                "Invalid input. Please check your input, it should be a FileParam"
                " (python), a url, or a base64 encoded string.",
            )

        SAMPLE_RATE = 16000  # The default sample rate that whisperx uses
        if len(audio) < SAMPLE_RATE * 60:
            # For audio shorter than 1 minute, directly compute and return the result.
            ret = self._run_whisperx(
                audio,
                language,
                min_speakers,
                max_speakers,
                transcribe_only,
            )
            if ret is None:
                raise HTTPException(
                    500, "You hit a programming error - please let us know."
                )
            else:
                return ret
        elif len(audio) > SAMPLE_RATE * 60 * 60:
            # For audio longer than 90 minutes, raise an error.
            raise HTTPException(400, "Audio longer than 60 minutes is not supported.")
        else:
            task_id = self._gen_unique_filename()
            input_filepath = os.path.join(
                self.OUTPUT_ROOT, task_id + self.INPUT_FILE_EXTENSION
            )
            np.save(input_filepath, audio)
            self.add_background_task(
                self._run_whisperx,
                input_filepath,
                language,
                min_speakers,
                max_speakers,
                transcribe_only,
                task_id,
            )
            self._regular_clean_up()
            return {"task_id": task_id}

    @Photon.handler
    def status(self, task_id: str) -> Dict[str, str]:
        """
        Returns the status of the task. It could be "invalid_task_id", "pending", "not_found", or "ok".
        """
        try:
            _ = uuid.UUID(task_id, version=4)
        except ValueError:
            return {"status": "invalid_task_id"}
        input_filepath = os.path.join(
            self.OUTPUT_ROOT, task_id + self.INPUT_FILE_EXTENSION
        )
        output_filepath = os.path.join(
            self.OUTPUT_ROOT, task_id + self.OUTPUT_FILE_EXTENSION
        )
        if not os.path.exists(output_filepath):
            if os.path.exists(input_filepath):
                return {"status": "pending"}
            else:
                return {"status": "not_found"}
        else:
            return {"status": "ok"}

    @Photon.handler
    def get_result(self, task_id: str) -> Dict:
        """
        Gets the result of the whisper x task. If the task is not finished, it will raise a 404 error.
        Use `status(task_id=task_id)` to check if the task is finished.
        """
        if self.status(task_id=task_id)["status"] == "ok":
            output_filepath = os.path.join(
                self.OUTPUT_ROOT, task_id + self.OUTPUT_FILE_EXTENSION
            )
            return json.load(open(output_filepath, "r"))
        else:
            raise HTTPException(status_code=404, detail="result not found")

    def queue_length(self) -> int:
        """
        Returns the current queue length.
        """
        return len(
            [
                f
                for f in os.listdir(self.OUTPUT_ROOT)
                if f.endswith(self.INPUT_FILE_EXTENSION)
            ]
        )


if __name__ == "__main__":
    p = WhisperXBackground()
    p.launch()
