from io import BytesIO
from typing import Union, Optional, List

from leptonai.photon import Photon, FileParam, get_file_content, HTTPException


class ExtractFeaturesPhoton(Photon):
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

        # Here we choose blip2 model, for other available models, please refer to:
        #
        # from lavis.models import model_zoo
        # print(model_zoo)
        #
        self.model_and_preprocess = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=self.device,
        )

    @Photon.handler(
        examples=[
            {"image": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"text": "a large fountain spewing water into the air"},
            {
                "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                "text": "two cats",
            },
        ]
    )
    def run(
        self, image: Optional[Union[FileParam, str]] = None, text: Optional[str] = None
    ) -> List[float]:
        model, vis_processors, txt_processors = self.model_and_preprocess

        if image is None and text is None:
            raise HTTPException(
                status_code=400, detail="Either image or text should be provided."
            )

        if image is not None:
            image = self._get_img(image)
            image = vis_processors["eval"](image).unsqueeze(0).to(self.device)
        if text is not None:
            text = txt_processors["eval"](text)

        if image is not None and text is None:
            # image embedding
            features = model.extract_features({"image": image}, mode="image")
            return features.image_embeds[0].tolist()
        elif image is None and text is not None:
            # text embedding
            features = model.extract_features({"text_input": [text]}, mode="text")
            return features.text_embeds[0].tolist()
        else:
            # multimodal embedding
            features = model.extract_features({"image": image, "text_input": [text]})
            return features.multimodal_embeds[0].tolist()
