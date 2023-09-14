from io import BytesIO
from typing import Union

from leptonai.photon import Photon, FileParam, get_file_content


class VQAPhoton(Photon):
    requirement_dependency = [
        "salesforce-lavis",
        "Pillow",
        "opencv-python!=4.8.0.76",
        "opencv-contrib-python!=4.8.0.76",
    ]

    def _get_img(self, param):
        from PIL import Image

        content = get_file_content(param)
        return Image.open(BytesIO(content)).convert("RGB")

    def init(self):
        import torch
        from lavis.models import load_model_and_preprocess

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Here we choose blip model, for other available models, please refer to:
        #
        # from lavis.models import model_zoo
        # print(model_zoo)
        #
        self.model_and_preprocess = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device
        )

    @Photon.handler(
        example={
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "question": "How many cats?",
        }
    )
    def run(self, image: Union[FileParam, str], question: str) -> str:
        model, vis_processors, txt_processors = self.model_and_preprocess
        image = self._get_img(image)
        image = vis_processors["eval"](image).unsqueeze(0).to(self.device)
        question = txt_processors["eval"](question)
        answers = model.predict_answers(
            samples={"image": image, "text_input": question},
            inference_method="generate",
        )
        return answers[0]
