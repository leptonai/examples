from io import BytesIO
import os
from typing import Union, List

from leptonai.photon import Photon, FileParam


class IDEFICS(Photon):
    requirement_dependency = [
        "accelerate",
        "Pillow",
        "torch",
        "transformers",
        "protobuf",
    ]

    def init(self):
        import torch
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = os.environ.get("MODEL", "HuggingFaceM4/idefics-9b-instruct")
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    @Photon.handler(
        example={
            "prompts": [
                (
                    "User: Which famous person does the person in the image look like?"
                    " Could you craft an engaging narrative featuring this character"
                    " from the image as the main protagonist?"
                ),
                "https://huggingfacem4-idefics-playground.hf.space/file=/home/user/app/example_images/obama-harry-potter.jpg",
                "<end_of_utterance>",
                "\nAssistant:",
            ]
        }
    )
    def run(
        self,
        prompts: Union[List[Union[str, FileParam]], List[List[Union[str, FileParam]]]],
        eos_token: str = "<end_of_utterance>",
        bad_words: List[str] = ["<image>", "<fake_token_around_image>"],
        max_length: int = 256,
        **kwargs,
    ) -> Union[str, List[str]]:
        from PIL import Image

        if not prompts:
            return []

        input_is_batch = isinstance(prompts[0], list)
        if not input_is_batch:
            prompts = [prompts]

        for prompt in prompts:
            for i, p in enumerate(prompt):
                if isinstance(p, FileParam):
                    prompt[i] = Image.open(BytesIO(p.read())).convert("RGB")

        inputs = self.processor(
            prompts, add_end_of_utterance_token=False, return_tensors="pt"
        ).to(self.device)

        # Generation args
        exit_condition = self.processor.tokenizer(
            eos_token, add_special_tokens=False
        ).input_ids
        bad_words_ids = self.processor.tokenizer(
            bad_words, add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_length=max_length,
            **kwargs,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        if not input_is_batch:
            return generated_text[0]
        else:
            return generated_text
