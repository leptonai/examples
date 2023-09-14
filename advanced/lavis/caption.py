from io import BytesIO
from typing import Union

from leptonai.photon import Photon, FileParam, get_file_content


class CaptionPhoton(Photon):
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
            name="blip_caption",
            model_type="large_coco",
            is_eval=True,
            device=self.device,
        )

    @Photon.handler(
        example={"image": "http://images.cocodataset.org/val2017/000000039769.jpg"}
    )
    def run(self, image: Union[FileParam, str]) -> str:
        model, vis_processors, _ = self.model_and_preprocess

        image = self._get_img(image)
        image = vis_processors["eval"](image).unsqueeze(0).to(self.device)
        captions = model.generate({"image": image})
        return captions[0]
