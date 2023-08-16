import base64
from io import BytesIO
from typing import List, Optional

import gradio as gr
from loguru import logger
import torch

from leptonai.photon import Photon, WAVResponse, HTTPException
from TTS.api import TTS


DEFAULT_MODEL_NAME = "tts_models/en/vctk/vits"


class Speaker(Photon):
    requirement_dependency = ["gradio", "TTS"]

    system_dependency = ["espeak-ng", "libsndfile1-dev"]

    _model_name: str = ""
    _model: TTS = TTS()

    def init(self):
        """
        Initialize a default model.
        """
        logger.info("Loading the model...")
        default_model = DEFAULT_MODEL_NAME
        self._load_model(default_model)
        # A warmup trick: usually, the first run of the model is slower than
        # subsequent runs due to some internal initialization of many AI libraries.
        # To avoid this, we run the model once with a dummy input.
        logger.info("Model loaded. Warming up...")
        _ = self._tts("hello world")
        logger.info("Warmup done.")

    def _load_model(self, name):
        """
        Internal function to load a model. We will assume that the model name
        is already sanity checked.
        """
        if name != self._model_name:
            try:
                logger.info(f"Loading model {name}")
                if torch.cuda.is_available():
                    logger.info("Using GPU")
                else:
                    logger.info("Using CPU")
                self._model_name = name
                self._model = TTS(
                    name, progress_bar=False, gpu=torch.cuda.is_available()
                )
                logger.info(f"Loaded model {name}")

                logger.info(f"Model has languages {self.languages}")
                logger.info(f"Model has speakers {self.speakers}")
            except Exception as e:
                logger.error(f"Failed to load model {name}. Detailed error: {e}")
                logger.error("Resetting model to default.")
                self._model_name = ""
                self._model = TTS()

        return (
            gr.Dropdown.update(choices=self.languages, visible=bool(self.languages)),
            gr.Dropdown.update(choices=self.speakers, visible=bool(self.speakers)),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def languages(self) -> List[str]:
        try:
            lan = self._model.languages
            return lan if lan else []
        except AttributeError:
            # this is an uninitialized model
            return []

    @property
    def speakers(self) -> List[str]:
        try:
            s = self._model.speakers
            return s if s else []
        except AttributeError:
            # this is an uninitialized model
            return []

    def _tts(
        self, text: str, language: Optional[str] = None, speaker: Optional[str] = None
    ) -> BytesIO:
        logger.info(
            f"Synthesizing '{text}' with language '{language}' and speaker '{speaker}'"
        )
        if not language:
            if self.languages:
                language = self.languages[0]
            else:
                language = None
        if not speaker:
            if self.speakers:
                speaker = self.speakers[0]
            else:
                speaker = None
        logger.info(
            f"Synthesizing '{text}' with language '{language}' and speaker '{speaker}'"
        )
        wav = self._model.tts(
            text=text,
            language=language,  # type: ignore
            speaker=speaker,  # type: ignore
        )
        wav_io = BytesIO()
        self._model.synthesizer.save_wav(wav, wav_io)  # type: ignore
        wav_io.seek(0)
        return wav_io

    ##########################################################################
    # Photon handlers that are exposed to the external clients.
    ##########################################################################

    @Photon.handler()
    def list_languages(self) -> List[str]:
        """
        Returns a list of languages supported by the current model. Empty list
        if no model is loaded, or the model does not support multiple languages.
        """
        return self.languages

    @Photon.handler()
    def list_speakers(self) -> List[str]:
        """
        Returns a list of speakers supported by the current model. Empty list
        if no model is loaded, or the model does not support multiple speakers.
        """
        return self.speakers

    @Photon.handler()
    def get_model_name(self) -> str:
        """
        Returns the name of the current model.
        """
        return self.model_name

    @Photon.handler(
        example={
            "text": "The quick brown fox jumps over the lazy dog.",
            "language": "",
            "speaker": "",
        }
    )
    def tts(
        self, text: str, language: Optional[str] = None, speaker: Optional[str] = None
    ) -> WAVResponse:
        """
        Synthesizes speech from text. Returns the synthesized speech as a WAV
        file.

        Pass in language and speaker if the model is multilingual. Otherwise,
        only pass in text. To check if the current model supports language and
        speaker selections, call list_languages() and list_speakers().
        """
        if not self._model_name:
            raise HTTPException(
                status_code=400,
                detail="No model loaded yet.",
            )
        try:
            wav_io = self._tts(text=text, language=language, speaker=speaker)
            return WAVResponse(wav_io)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to synthesize speech. Detailed error: {e}",
            )

    ##########################################################################
    # Note: we have intentionally commented out this path, because when we run
    # the server with multiple replicas, calling this path will cause the replicas
    # to have different models loaded - only the replica that received the request
    # will have the model loaded. This is not what we want as we would like all
    # replicas to have the same model loaded.
    #
    # If you want to run the server with only one replica, you can uncomment this
    # path.
    #
    # If you want to run the server with multiple replicas, some sort of shared
    # state store is needed to ensure that when a load_model() request is received,
    # the receiving replica will load the model and update a shared state variable
    # to tell other replicas that, next time when it receives a tts() request, it
    # should load the model first. This is beyond the scope of this tutorial so we
    # will not implement it here.
    ##########################################################################
    # @Photon.handler(
    #     example={
    #         "name": "tts_models/multilingual/multi-dataset/bark",
    #     }
    # )
    # def load_model(self, name: str) -> bool:
    #     """
    #     Loads a model. For a list of available models, call list_models().
    #     """
    #     if name not in TTS().list_models():
    #         raise HTTPException(
    #             status_code=400,
    #             detail=f"Model {name} does not exist. For a list of available models, call list_models().",
    #         )
    #     try:
    #         self._load_model(name)
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=500,
    #             detail=f"Failed to load model {name}. Detailed error: {e}",
    #         )
    #     return True

    # @Photon.handler()
    # def list_models(self) -> List[str]:
    #     """
    #     Returns a list of available models.
    #     """
    #     return TTS().list_models()

    ##########################################################################
    # An UI wrapper around gradio.
    ##########################################################################
    @Photon.handler(mount=True)
    def ui(self):
        blocks = gr.Blocks()

        with blocks:
            with gr.Column():
                # The below line is similar to the load_model() path above. If you
                # are running multiple replicas, you should not dynamically load
                # models in this way. See the comment above for more details.
                # If you are running only one replica, you can uncomment the below
                # line so you can dynamically load models.
                # model = gr.Dropdown(choices=TTS().list_models(), label="Model")
                model = gr.Dropdown(choices=[DEFAULT_MODEL_NAME], label="Model")
                language = gr.Dropdown(label="Language", visible=False)
                speaker = gr.Dropdown(label="Speaker", visible=False)
                model.change(
                    self._load_model, inputs=[model], outputs=[language, speaker]
                )

                text = gr.Textbox(label="Text", max_lines=3)

            with gr.Column():
                audio = gr.HTML(label="Audio")

            text.submit(
                fn=lambda *args, **kwargs: f"""<audio src="data:audio/mpeg;base64,{base64.b64encode(self._tts(*args, **kwargs).read()).decode('utf-8')}" controls autoplay></audio>""",
                inputs=[text, language, speaker],
                outputs=[audio],
            )

        return blocks


if __name__ == "__main__":
    p = Speaker()
    p.launch()
