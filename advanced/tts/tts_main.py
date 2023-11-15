from io import BytesIO
import os
from threading import Lock
from typing import List, Optional, Union, Dict

from loguru import logger
import torch

from leptonai.photon import (
    Photon,
    WAVResponse,
    HTTPException,
    FileParam,
    get_file_content,
)


class Speaker(Photon):
    """
    A TTS service that supports multiple models provided by coqui and others.

    To launch this photon and specify the model to use, you can pass in env
    variables during photon launch:
        --env MODEL_NAME=tts_models/en/vctk/vits
    And if you want to preload multiple models, you can pass in a comma-separated
    list of models:
        --env PRELOAD_MODELS=tts_models/en/vctk/vits,tts_models/multilingual/multi-dataset/xtts_v1
    """

    requirement_dependency = ["TTS"]

    system_dependency = ["espeak-ng", "libsndfile1-dev"]

    handler_max_concurrency = 4

    MODEL_NAME = "tts_models/en/vctk/vits"
    # Or, you can choose some other models
    # MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v1"

    # If you want to load multiple models at the same time, you can put it here
    # as a comma-separated string. For example:
    # PRELAOD_MODELS = "tts_models/en/vctk/vits,tts_models/multilingual/multi-dataset/xtts_v1"
    # Note that the default model will always be loaded.
    # Note that this might involve some extra memory - use at your own risk.
    PRELOAD_MODELS = ""

    def init(self):
        """
        Initialize a default model.
        """

        # By using XTTS you agree to CPML license https://coqui.ai/cpml
        os.environ["COQUI_TOS_AGREED"] = "1"

        from TTS.api import TTS

        self._models: Dict[Union[str, None], TTS] = {}
        self._model_lock: Dict[Union[str, None], Lock] = {}

        self.MODEL_NAME = os.environ.get("MODEL_NAME", self.MODEL_NAME).strip()

        self.PRELOAD_MODELS = [
            m
            for m in os.environ.get("PRELOAD_MODELS", self.PRELOAD_MODELS).split(",")
            if m
        ]
        if self.MODEL_NAME not in self.PRELOAD_MODELS:
            self.PRELOAD_MODELS.append(self.MODEL_NAME)

        logger.info("Loading the model...")
        for model_name in self.PRELOAD_MODELS:
            self._models[model_name] = self._load_model(model_name)
            self._model_lock[model_name] = Lock()
        self._models[None] = self._models[self.MODEL_NAME]
        self._model_lock[None] = self._model_lock[self.MODEL_NAME]
        logger.debug("Model loaded.")

    def _load_model(self, model_name: str):
        """
        Internal function to load a model. We will assume that the model name
        is already sanity checked.
        """
        from TTS.api import TTS

        use_gpu = torch.cuda.is_available()
        logger.debug(f"Loading model {model_name}... use_gpu: {use_gpu} ")
        try:
            model = TTS(model_name, progress_bar=False, gpu=use_gpu)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}.") from e
        logger.debug(f"Loaded model {model_name}")
        logger.debug(f"Model {model_name} is_multilingual: {model.is_multi_lingual}")
        logger.debug(f"Model {model_name} is_multi_speaker: {model.is_multi_speaker}")
        try:
            # The below one seems to not always work with xtts models.
            if model.is_multi_lingual:
                logger.debug(f"Model {model_name} languages: {model.languages}")
        except AttributeError:
            try:
                # xtts models have a different way of accessing languages.
                logger.debug(
                    f"Model {model_name} languages:"
                    f" {model.synthesizer.tts_model.config.languages}"
                )
            except Exception:
                # If neither of above works, we will just ignore it and not print
                # anything.
                pass
        if model.is_multi_speaker:
            logger.debug(f"Model {model_name} speakers: {model.speakers}")

        return model

    def _tts(
        self,
        text: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_wav: Optional[str] = None,
    ) -> BytesIO:
        if model not in self._models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model} not loaded.",
            )
        logger.info(
            f"Synthesizing '{text}' with language '{language}' and speaker '{speaker}'"
        )
        # Many of the models might not be python thread safe, so we lock it.
        with self._model_lock[model]:
            wav = self._models[model].tts(
                text=text,
                language=language,  # type: ignore
                speaker=speaker,  # type: ignore
                speaker_wav=speaker_wav,
            )
        return wav

    ##########################################################################
    # Photon handlers that are exposed to the external clients.
    ##########################################################################

    @Photon.handler(method="GET")
    def languages(self, model: Optional[str] = None) -> List[str]:
        """
        Returns a list of languages supported by the current model. Empty list
        if no model is loaded, or the model does not support multiple languages.
        """
        if model not in self._models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model} not loaded.",
            )
        if not self._models[model].is_multi_lingual:
            return []
        try:
            return self._models[model].languages
        except AttributeError:
            # xtts models have a different way of accessing languages.
            # if there are further errors, we don't handle them.
            return self._models[model].synthesizer.tts_model.config.languages

    @Photon.handler(method="GET")
    def speakers(self, model: Optional[str] = None) -> List[str]:
        """
        Returns a list of speakers supported by the model. If the model is an
        XTTS model, this will return empty as you will need to use speaker_wav
        to synthesize speech.
        """
        if model not in self._models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model} not loaded.",
            )
        elif not self._models[model].is_multi_speaker:
            return []
        else:
            return self._models[model].speakers

    @Photon.handler(method="GET")
    def models(self) -> List[str]:
        """
        Returns a list of available models.
        """
        return [k for k in self._models.keys() if k]

    @Photon.handler(
        example={
            "text": "The quick brown fox jumps over the lazy dog.",
        }
    )
    def tts(
        self,
        text: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_wav: Union[None, str, FileParam] = None,
    ) -> WAVResponse:
        """
        Synthesizes speech from text. Returns the synthesized speech as a WAV
        response.

        Pass in language if the model is multilingual. Pass in speaker if the model
        is multi-speaker. Pass in speaker_wav if the model is XTTS. The endpoint
        tries its best to return the correct error message if the parameters are
        not correct, but it may not be perfect.
        """
        if model not in self._models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model} not loaded.",
            )
        tts_model = self._models[model]
        if not tts_model.is_multi_lingual and language is not None:
            raise HTTPException(
                status_code=400,
                detail="Model is not multi-lingual, you should not pass in language.",
            )
        if not tts_model.is_multi_speaker and speaker is not None:
            raise HTTPException(
                status_code=400,
                detail="Model is not multi-speaker, you should not pass in speaker.",
            )
        if tts_model.is_multi_lingual and language is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Model is multi-lingual, you should pass in language.              "
                    "       Use GET /languages to get available languages and pass in  "
                    "                       as optional parameters"
                ),
            )
        if tts_model.is_multi_speaker and speaker is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Model is multi-speaker, you should pass in speaker.               "
                    "      Use GET /speakers to get available speakers and pass in as  "
                    "                       optional parameters"
                ),
            )

        try:
            if speaker_wav is not None:
                speaker_wav_file = get_file_content(
                    speaker_wav, allow_local_file=False, return_file=True
                )
                speaker_wav_file_name = speaker_wav_file.name
            else:
                speaker_wav_file_name = None
            wav = self._tts(
                text=text,
                language=language,
                speaker=speaker,
                speaker_wav=speaker_wav_file_name,
            )
            wav_io = BytesIO()
            tts_model.synthesizer.save_wav(wav, wav_io)  # type: ignore
            wav_io.seek(0)
            return WAVResponse(wav_io)
        except HTTPException:
            raise
        except TypeError as e:
            if "expected str, bytes or os.PathLike object, not NoneType" in str(e):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Speaker wav file is not provided. This is necessary when"
                        " running an XTTS model to do voice cloning."
                    ),
                ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to synthesize speech. Details: {e}",
            ) from e


if __name__ == "__main__":
    p = Speaker()
    p.launch()
