import os
import sys
import tempfile
from typing import List, Dict, Optional

import numpy as np

import whisperx

from loguru import logger

from leptonai.photon import Photon, FileParam, HTTPException


class WhisperXPhoton(Photon):
    """
    A WhisperX photon that serves the [WhisperX](https://github.com/m-bain/whisperX) model.

    The photon exposes two endpoints: `/run` and `/run_upload` that deals with files/urls
    and uploaded contents respectively. See the docs of each for details.
    """

    requirement_dependency = [
        "leptonai",
        "torch",
        "torchaudio",
        "git+https://github.com/m-bain/whisperx.git",
    ]

    system_dependencies = ["ffmpeg"]

    def init(self):
        logger.info("Initializing WhisperXPhoton")
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
        if not self.hf_token:
            logger.warning(
                "Please set the environment variable HUGGING_FACE_HUB_TOKEN."
            )
            sys.exit(1)
        self.device = "cuda"
        compute_type = "float16"
        self._model = whisperx.load_model(
            "large-v2", self.device, compute_type=compute_type
        )

        self._language_code = "en"
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=self._language_code, device=self.device
        )
        self._diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )

    def _run_audio(
        self,
        audio: np.ndarray,
        batch_size: int = 4,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """
        The main function that is called by the others.
        """
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
        diarize_segments = self._diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)
        # return result["segments"]  # segments are now assigned speaker IDs
        return result["segments"]

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
    ) -> List[Dict]:
        """
        Runs transcription, alignment, and diarization for the input.

        - Inputs:
            - filename: a url containing the audio file, or a local file path if running
                locally.
            - batch_size(optional): the batch size to run whisperx inference.
            - min_speakers(optional): the hint for minimum number of speakers for diarization.
            - max_speakers(optional): the hint for maximum number of speakers for diarization.

        - Returns:
            - result: a list of dictionary, each containing one classified segment. Each
                segment is a dictionary containing the following keys: `start` and `end`
                specifying the start and end time of the segment in seconds, `text` as
                the recognized text, `words` that contains segmented words and corresponding
                speaker IDs.
            - 404: if the file cannot be loaded.
            - 500: if internal error occurs.
        """
        try:
            audio = whisperx.load_audio(filename)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Cannot load audio at {filename}. Detailed error message: {str(e)}"
                ),
            )
        return self._run_audio(audio, batch_size, min_speakers, max_speakers)

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
    ) -> List[Dict]:
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
        logger.info(f"upload_file: {upload_file}")
        # Whisper at this moment only reads contents from file, so we will have to
        # write it to a temporary file
        tmpfile = tempfile.NamedTemporaryFile()
        with open(tmpfile.name, "wb") as f:
            f.write(upload_file.file.read())
            f.flush()
        logger.info(f"tmpfile: {tmpfile.name}")
        try:
            audio = whisperx.load_audio(tmpfile.name)
        except Exception as e:
            logger.info(f"encountered error. returning 500. Detailed: {e}")
            raise HTTPException(
                status_code=500,
                detail=(
                    "Cannot load audio with uploaded content. Detailed error"
                    f" message: {str(e)}"
                ),
            )
        logger.info("Started running WhisperX")
        ret = self._run_audio(audio, batch_size, min_speakers, max_speakers)
        # remove temporary file
        tmpfile.close()
        return ret


if __name__ == "__main__":
    p = WhisperXPhoton()
    p.launch()
