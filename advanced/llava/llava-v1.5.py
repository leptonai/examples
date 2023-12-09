import os
from threading import Thread

from leptonai.photon import Photon, StreamingResponse
from loguru import logger


class Llava(Photon):
    health_check_liveness_tcp_port = 8765

    vcs_url = "https://github.com/haotian-liu/LLaVA.git@7775b12"
    requirement_dependency = [
        "-e .",
    ]

    def init(self):
        # Workaround for issue when using triton
        # `"assert '_distutils' in core.__file__, core.__file__"`
        # https://github.com/pyinstaller/pyinstaller/issues/6911#issuecomment-1165604688
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_size = os.environ.get("LLAVA_MODEL_SIZE", "7b").lower()
        if model_size not in ["7b", "13b"]:
            raise ValueError(f"Model size should be either 7b or 13b, got {model_size}")
        model_path = f"liuhaotian/llava-v1.5-{model_size}"
        logger.info(f"Loading model {model_path}")

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
            )
        )

    @Photon.handler(
        example={
            "image": "https://llava-vl.github.io/static/images/view.jpg",
            "prompt": (
                "What are the things I should be cautious about when I visit here?"
            ),
        }
    )
    def run(
        self,
        image: str,
        prompt: str,
        top_p: float = 1.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> StreamingResponse:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from transformers.generation.streamers import TextIteratorStreamer

        from PIL import Image
        import requests
        import torch

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_data = Image.open(requests.get(image, stream=True).raw)
        image_tensor = (
            self.image_processor.preprocess(image_data, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )

        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

        with torch.inference_mode():
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                ),
            )
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[: -len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()
