import base64
from io import BytesIO
import os

from typing import List, Union

from leptonai.photon import Photon, FileParam, HTTPException


# Pretrained models are obtained from https://github.com/mlfoundations/open_flamingo
# and transcribed to the following dictionary.
pretrained_models = {
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b": [
        "ViT-L-14",
        "openai",
        "mosaicml/mpt-1b-redpajama-200b",
        "mosaicml/mpt-1b-redpajama-200b",
        1,
    ],
    "OpenFlamingo-3B-vitl-mpt1b-langinstruct": [
        "ViT-L-14",
        "openai",
        "mosaicml/mpt-1b-redpajama-200b-dolly",
        "mosaicml/mpt-1b-redpajama-200b-dolly",
        1,
    ],
    "openflamingo/OpenFlamingo-4B-vitl-rpj3b": [
        "ViT-L-14",
        "openai",
        "togethercomputer/RedPajama-INCITE-Base-3B-v1",
        "togethercomputer/RedPajama-INCITE-Base-3B-v1",
        2,
    ],
    "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct": [
        "ViT-L-14",
        "openai",
        "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        2,
    ],
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b": [
        "ViT-L-14",
        "openai",
        "mosaicml/mpt-7b",
        "mosaicml/mpt-7b",
        4,
    ],
}


class Flamingo(Photon):
    requirement_dependency = ["open-flamingo", "huggingface-hub", "Pillow", "requests"]

    IMAGE_TOKEN = "<image>"
    END_OF_TEXT_TOKEN = "<|endofchunk|>"
    DEFAULT_MODEL = "openflamingo/OpenFlamingo-3B-vitl-mpt1b"

    def init(self):
        from open_flamingo import create_model_and_transforms
        from huggingface_hub import hf_hub_download
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        model_name = os.environ.get("OPEN_FLAMINGO_MODEL", self.DEFAULT_MODEL)
        try:
            model_spec = pretrained_models[model_name]
        except KeyError:
            raise KeyError(
                f"Model {model_name} not found in pretrained_models. Available models:"
                f" {pretrained_models.keys()}"
            )

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=model_spec[0],
            clip_vision_encoder_pretrained=model_spec[1],
            lang_encoder_path=model_spec[2],
            tokenizer_path=model_spec[3],
            cross_attn_every_n_layers=model_spec[4],
        )

        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt"
        )
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model = self.model.to(self.device)

        self.tokenizer.padding_side = "left"

    def _img_param_to_img(self, param):
        from PIL import Image
        import requests

        if isinstance(param, FileParam):
            content = param.file.read()
        elif isinstance(param, str):
            if param.startswith("http://") or param.startswith("https://"):
                content = requests.get(param).content
            else:
                content = base64.b64decode(param).decode("utf-8")
        else:
            raise TypeError(f"Invalid image type: {type(param)}")

        return Image.open(BytesIO(content))

    @Photon.handler(
        example={
            "demo_images": [
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            ],
            "demo_texts": ["An image of two cats.", "An image of a bathroom sink."],
            "query_image": (
                "http://images.cocodataset.org/test-stuff2017/000000028352.jpg"
            ),
            "query_text": "An image of",
        },
    )
    def run(
        self,
        demo_images: List[Union[FileParam, str]],
        demo_texts: List[str],
        query_image: Union[FileParam, str],
        query_text: str,
        max_new_tokens: int = 32,
        num_beams: int = 3,
    ) -> str:
        import torch

        if len(demo_images) != len(demo_texts):
            raise HTTPException(
                status_code=400,
                detail="The number of demo images and demo texts must be the same.",
            )

        demo_images = [self._img_param_to_img(img) for img in demo_images]
        query_image = self._img_param_to_img(query_image)

        vision_x = [
            self.image_processor(img).unsqueeze(0).to(self.device)
            for img in (demo_images + [query_image])
        ]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        lang_x_text = self.END_OF_TEXT_TOKEN.join(
            f"{self.IMAGE_TOKEN}{text}" for text in (demo_texts + [query_text])
        )
        lang_x = self.tokenizer(
            lang_x_text,
            return_tensors="pt",
        )

        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to(self.device),
            attention_mask=lang_x["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        generated_text = self.tokenizer.decode(generated_text[0])

        if generated_text.startswith(lang_x_text):
            generated_text = generated_text[len(lang_x_text) :]
        if generated_text.endswith(self.END_OF_TEXT_TOKEN):
            generated_text = generated_text[: -len(self.END_OF_TEXT_TOKEN)]

        return generated_text
