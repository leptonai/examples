import os
from threading import Thread
from queue import Queue

from loguru import logger
from leptonai.photon import Photon, StreamingResponse


class HfStreamLLM(Photon):

    deployment_template = {
        "resource_shape": "gpu.a10.6xlarge",
        "env": {
            "MODEL_PATH": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        },
        "secret": [
            "HUGGING_FACE_HUB_TOKEN",
        ],
    }

    requirement_dependency = [
        "transformers",
    ]

    handler_max_concurrency = 4

    def init(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.environ["MODEL_PATH"]

        self._tok = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

        self._generation_queue = Queue()

        for _ in range(self.handler_max_concurrency):
            Thread(target=self._generate, daemon=True).start()

    def _generate(self):
        while True:
            streamer, args, kwargs = self._generation_queue.get()
            try:
                self._model.generate(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in generation: {e}")
                streamer.text_queue.put(streamer.stop_signal)

    @Photon.handler
    def run(self, text: str, max_new_tokens: int = 100) -> StreamingResponse:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self._tok, skip_prompt=True, timeout=60)
        inputs = self._tok(text, return_tensors="pt").to("cuda")
        self._generation_queue.put_nowait((
            streamer,
            (),
            dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens),
        ))
        return streamer
