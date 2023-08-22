import json
import os
import sys
import time
from typing import Dict, Optional
import uuid

from leptonai.photon import Photon, FileParam, HTTPException
from loguru import logger
import whisperx


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
    JOB_FILE_EXTENSION = ".input"
    OUTPUT_FILE_EXTENSION = ".json"
    OUTPUT_MAXIMUM_AGE = 60 * 60 * 24  # 1 day
    CLEANUP_INTERVAL = 60 * 60  # 1 hour
    LAST_CLEANUP_TIME = time.time()

    def init(self):
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

        self._language_code = "en"
        # 2. load whisper align model
        model_a, metadata = whisperx.load_align_model(
            language_code=self._language_code, device=self.device
        )
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

    def _run_file_or_url(
        self,
        audio_file_or_url: str,
        output_filename: str,
        batch_size: int = 4,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> None:
        """
        The main function that is called by the others.
        """
        output_filepath = os.path.join(self.OUTPUT_ROOT, output_filename)
        try:
            audio = whisperx.load_audio(audio_file_or_url)
        except Exception as e:
            errormsg = (
                f"Cannot load audio at {audio_file_or_url}. Detailed error message:"
                f" {str(e)}"
            )
            json.dump({"error": errormsg}, open(output_filepath, "w"))
            return

        result = self._model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"])  # before alignment

        if self._language_code != result["language"]:
            self._model_a, self._metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
            self._language_code = result["language"]

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        # print(result["segments"])  # after alignment

        # add min/max number of speakers if known
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

        result = whisperx.assign_word_speakers(diarize_segments, result)
        # return result["segments"]  # segments are now assigned speaker IDs
        json.dump(result["segments"], open(output_filepath, "w"))
        # If it is a local file, remove it
        if audio_file_or_url.startswith(self.OUTPUT_ROOT):
            os.remove(audio_file_or_url)
        return

    @Photon.handler(
        example={
            "filename": (
                "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
            ),
        }
    )
    def run(
        self,
        filename: str,
        batch_size: int = 4,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Runs transcription, alignment, and diarization for the input.

        - Inputs:
            - filename: a url containing the audio file, or a local file path if running
                locally.
            - batch_size(optional): the batch size to run whisperx inference.
            - min_speakers(optional): the hint for minimum number of speakers for diarization.
            - max_speakers(optional): the hint for maximum number of speakers for diarization.

        - Returns:
            - task: a dictionary with key `task_id` and value the task uuid. Use `status(**task)`
                to query the task status, and `get_result(**task)` to get the result when the
                status is "ok".
        """
        unique_name = self._gen_unique_filename()
        self.add_background_task(
            self._run_file_or_url,
            filename,
            unique_name + self.OUTPUT_FILE_EXTENSION,
            batch_size,
            min_speakers,
            max_speakers,
        )
        self._regular_clean_up()
        return {"task_id": unique_name}

    @Photon.handler(
        example={
            "upload_file": (
                "(please use python) FileParam(open('path/to/your/file.wav', 'rb'))"
            ),
        }
    )
    def run_upload(
        self,
        upload_file: FileParam,
        batch_size: int = 4,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Runs transcription, alignment, and diarization for the input.

        Everything is the same as the `/run` path, except that the input is uploaded
        as a file. If you are using the lepton python client, you can achieve so by
            from leptonai.photon import FileParam
            from leptonai.client import Client
            client = Client(PUT_YOUR_SERVER_INFO_HERE)
            client.run_upload(upload_file=FileParam(open("path/to/your/file.wav", "rb")))

        For more details, refer to `/run`.
        """
        unique_name = self._gen_unique_filename()
        task_file = os.path.join(
            self.OUTPUT_ROOT, unique_name + self.JOB_FILE_EXTENSION
        )
        with open(task_file, "wb") as f:
            f.write(upload_file.file.read())
            f.flush()
        self.add_background_task(
            self._run_file_or_url,
            task_file,
            unique_name + self.OUTPUT_FILE_EXTENSION,
            batch_size,
            min_speakers,
            max_speakers,
        )
        self._regular_clean_up()
        return {"task_id": unique_name}

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
            self.OUTPUT_ROOT, task_id + self.JOB_FILE_EXTENSION
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

    @Photon.handler
    def queue_length(self) -> int:
        """
        Returns the current queue length.
        """
        return len(
            [
                f
                for f in os.listdir(self.OUTPUT_ROOT)
                if f.endswith(self.JOB_FILE_EXTENSION)
            ]
        )


if __name__ == "__main__":
    p = WhisperXBackground()
    p.launch()
