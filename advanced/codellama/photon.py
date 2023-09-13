import os
from leptonai.photon import Photon
from typing import List, Optional, Union

import torch
from transformers import pipeline


class CodeLlama(Photon):
    requirement_dependency = [
        "git+https://github.com/huggingface/transformers.git@015f8e1",
        "accelerate",
    ]

    def init(self):
        if torch.cuda.is_available():
            device = 0
        else:
            device = -1

        self.pipeline = pipeline(
            "text-generation",
            model=os.environ.get("MODEL", "codellama/CodeLlama-7b-hf"),
            torch_dtype=torch.float16,
            device=device,
        )

    def _get_generated_text(self, res):
        if isinstance(res, str):
            return res
        elif isinstance(res, dict):
            return res["generated_text"]
        elif isinstance(res, list):
            if len(res) == 1:
                return self._get_generated_text(res[0])
            else:
                return [self._get_generated_text(r) for r in res]
        else:
            raise ValueError(
                f"Unsupported result type in _get_generated_text: {type(res)}"
            )

    @Photon.handler(
        "run",
        example={
            "inputs": "import socket\n\ndef ping_exponential_backoff(host: str):",
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.95,
            "temperature": 0.1,
            "max_new_tokens": 256,
        },
    )
    def run_handler(
        self,
        inputs: Union[str, List[str]],
        do_sample: bool = True,
        top_k: int = 10,
        top_p: float = 0.95,
        temperature: Optional[float] = 0.1,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> Union[str, List[str]]:
        res = self.pipeline(
            inputs,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return self._get_generated_text(res)


if __name__ == "__main__":
    p = CodeLlama()
    p.launch()
