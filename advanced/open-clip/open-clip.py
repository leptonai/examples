import io
import os
import urllib
from typing import List

import open_clip
from PIL import Image
import torch
import validators

from leptonai.photon import Photon, handler, HTTPException
from leptonai.photon.types import lepton_unpickle, is_pickled


DEFAULT_MODEL_NAME = "ViT-B-32-quickgelu"
DEFAULT_PRETRAINED = "laion400m_e32"


class Clip(Photon):
    """
    This photon is used to embed text and image into a vector space using CLIP.
    """

    # Python dependency
    requirement_dependency = [
        "open_clip_torch",
        "Pillow",
        "torch",
        "transformers",
        "validators",
    ]

    def init(self):
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"
        MODEL_NAME = (
            os.environ["MODEL_NAME"]
            if "MODEL_NAME" in os.environ
            else DEFAULT_MODEL_NAME
        )
        PRETRAINED = (
            os.environ["PRETRAINED"]
            if "PRETRAINED" in os.environ
            else DEFAULT_PRETRAINED
        )
        (
            self.CLIP_MODEL,
            _,
            self.CLIP_IMG_PREPROCESS,
        ) = open_clip.create_model_and_transforms(
            model_name=MODEL_NAME, pretrained=PRETRAINED, device=self.DEVICE
        )
        self.TOKENIZER = open_clip.get_tokenizer(MODEL_NAME)

    @handler("embed")
    def embed(self, query: str) -> List[float]:
        if validators.url(query):
            return self.embed_image(query)
        else:
            return self.embed_text(query)

    @handler("embed_text")
    def embed_text(self, query: str) -> List[float]:
        query = self.TOKENIZER([query])
        with torch.no_grad():
            text_features = self.CLIP_MODEL.encode_text(query.to(self.DEVICE))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return list(text_features.cpu().numpy()[0].astype(float))

    def embed_image_local(self, image: Image):
        image = self.CLIP_IMG_PREPROCESS(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            image_features = self.CLIP_MODEL.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return list(image_features.cpu().numpy()[0].astype(float))

    @handler("embed_image")
    def embed_image(self, url: str) -> List[float]:
        # open the imageurl and then read the content into a buffer
        try:
            raw_img = Image.open(io.BytesIO(urllib.request.urlopen(url).read()))
        except:
            raise HTTPException(
                status_code=400, detail=f"Cannot download image at url {url}."
            )
        return self.embed_image_local(raw_img)

    @handler("embed_pickle_image")
    def embed_pickle_image(self, image) -> List[float]:
        print("Is the image passed in pickled ? :", is_pickled(image))
        try:
            raw_img = lepton_unpickle(image)
        except:
            raise HTTPException(
                status_code=400, detail="Cannot read image from bytes."
            )
        return self.embed_image_local(raw_img)