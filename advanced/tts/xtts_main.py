from io import BytesIO
import os
import subprocess
from tempfile import NamedTemporaryFile
from threading import Lock
import time
from typing import Optional, Union

from loguru import logger

from leptonai.photon import (
    Photon,
    WAVResponse,
    HTTPException,
    FileParam,
    get_file_content,
)


class XTTSSpeaker(Photon):
    """
    A XTTS service that supports multiple models provided by coqui and others.

    To launch this photon and specify the model to use, you can pass in env
    variables during photon launch:
        --env MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v1.1
    """

    requirement_dependency = ["TTS", "deepspeed"]

    system_dependency = ["ffmpeg", "espeak-ng", "libsndfile1-dev"]

    handler_max_concurrency = 4

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v1.1"
    DEFAULT_DECODER = "ne_hifigan"

    def init(self):
        """
        Initialize a default model.
        """

        # By using XTTS you agree to CPML license https://coqui.ai/cpml
        os.environ["COQUI_TOS_AGREED"] = "1"

        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.generic_utils import get_user_data_dir
        from TTS.utils.manage import ModelManager

        logger.info("Loading the xtts model...")
        try:
            self.MODEL_NAME = os.environ.get("MODEL_NAME", self.MODEL_NAME).strip()
            ModelManager().download_model(self.MODEL_NAME)
            model_path = os.path.join(
                get_user_data_dir("tts"), self.MODEL_NAME.replace("/", "--")
            )
            config = XttsConfig()
            config.load_json(os.path.join(model_path, "config.json"))
            self._model = Xtts.init_from_config(config)
            self._model.load_checkpoint(
                config,
                checkpoint_path=os.path.join(model_path, "model.pth"),
                vocab_path=os.path.join(model_path, "vocab.json"),
                eval=True,
                use_deepspeed=torch.cuda.is_available(),
            )
            # The xtts model's main chunk cannot be run in parallel, so we will need
            # to lock protect it.
            self._model_lock = Lock()
            self._supported_languages = self._model.config.languages
            if torch.cuda.is_available():
                self._model.cuda()
            self._languages = config.languages
        except Exception as e:
            raise RuntimeError(f"Cannot load XTTS model {self.MODEL_NAME}") from e

        logger.debug("Model loaded.")

    def _tts(
        self,
        text: str,
        language: str,
        speaker_wav: Optional[str] = None,
        voice_cleanup: Optional[bool] = False,
    ):
        import torch

        if voice_cleanup:
            with NamedTemporaryFile(suffix=".wav", delete=False) as filtered_file:
                lowpass_highpass = "lowpass=8000,highpass=75,"
                trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02"
                shell_command = (
                    f"ffmpeg -y -i {speaker_wav} -af"
                    f" {lowpass_highpass}{trim_silence} {filtered_file.name}".split(" ")
                )
                logger.debug("Running ffmpeg command: " + " ".join(shell_command))
                try:
                    subprocess.run(
                        shell_command,
                        capture_output=False,
                        text=True,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError as e:
                    logger.debug("Failed to run ffmpeg command: " + str(e))
                    logger.debug("Use original file")
                else:
                    # filter succeeded - use filtered file.
                    speaker_wav = filtered_file.name
        # critical part: cannot run in parallel threads.
        with self._model_lock:
            # learn from speaker_wav
            start = time.time()
            logger.debug("Learning from speaker wav...")
            try:
                gpt_cond_latent, diffusion_conditioning, speaker_embedding = (
                    self._model.get_conditioning_latents(audio_path=speaker_wav)
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to learn from speaker wav.",
                ) from e
            learned_time = time.time()
            logger.debug(f"Learned from speaker wav in {learned_time - start} seconds.")
            out = self._model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                diffusion_conditioning,
                decoder=self.DEFAULT_DECODER,
            )
            logger.debug(f"Synthesized speech in {time.time() - learned_time} seconds.")
        if voice_cleanup:
            os.remove(filtered_file.name)  # type: ignore
        return torch.tensor(out["wav"]).unsqueeze(0)

    ##########################################################################
    # Photon handlers that are exposed to the external clients.
    ##########################################################################
    @Photon.handler(
        example={
            "text": "The quick brown fox jumps over the lazy dog.",
        }
    )
    def tts(
        self,
        text: str,
        language: str,
        speaker_wav: Union[str, FileParam],
        voice_cleanup: bool = False,
    ) -> WAVResponse:
        """
        Synthesizes speech from text. Returns the synthesized speech as a WAV
        response. The XTTS model is multi-lingual, so you need to specify the
        language - use language() to show a list of languages available. The
        model carries out voice transfer from the speaker wav file, so you need
        to specify the speaker wav file. The endpoint tries its best to return
        the correct error message if the parameters are not correct, but it may
        not be perfect.
        """
        import torchaudio

        if language not in self._supported_languages:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Language {language} not supported. Supported languages are:"
                    f" {self._supported_languages}"
                ),
            )

        try:
            speaker_wav_file = get_file_content(
                speaker_wav, allow_local_file=False, return_file=True
            )
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read speaker wav file {speaker_wav}.",
            )

        speaker_wav_file_name = speaker_wav_file.name
        wav = self._tts(
            text,
            language,
            speaker_wav=speaker_wav_file_name,
            voice_cleanup=voice_cleanup,
        )
        wav_io = BytesIO()
        torchaudio.save(wav_io, wav, 24000, format="wav")
        wav_io.seek(0)
        return WAVResponse(wav_io)


if __name__ == "__main__":
    p = XTTSSpeaker()
    p.launch()
