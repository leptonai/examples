import copy
import os
import requests
from threading import Lock
from typing import Union, Any, Dict

from loguru import logger

import layoutparser as lp
from layoutparser.models.detectron2 import catalog
import cv2

from leptonai.photon import (
    Photon,
    FileParam,
    get_file_content,
    PNGResponse,
    HTTPException,
    make_png_response,
)


class LayoutParser(Photon):
    requirement_dependency = [
        "layoutparser",
        "git+https://github.com/facebookresearch/detectron2.git",
        "pytesseract",
    ]

    system_dependency = [
        "tesseract-ocr",
    ]

    # Layout parser ocr right now seems to be thread safe, so we can turn on
    # multithreading to avoid blocking and improve overall IO time.
    handler_max_concurrency = 4

    # The default model config. Specify "MODEL_CONFIG" env variable to
    # override this.
    DEFAULT_MODEL_CONFIG = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"

    # The path to save the model.
    MODEL_SAVE_PATH = "/tmp/layoutparser_lepton_cache"

    # You can specify the language code(s) of the documents to detect to improve
    # accuracy. The supported language and their code can be found at:
    # https://github.com/tesseract-ocr/langdata
    # The supported format is `+` connected string like `"eng+fra"`
    TESSERACT_LANGUAGE = "eng"
    TESSERACT_CONFIGS = {}

    def init(self):
        logger.debug("Loading model...")
        self.model = LayoutParser.safe_load_model(
            os.environ.get("MODEL_CONFIG", self.DEFAULT_MODEL_CONFIG)
        )
        # We are not sure if the underlying layout parser model is thread safe, so we will
        # consider it a black box and use a lock to prevent concurrent access.
        self.model_lock = Lock()
        self.ocr_agent = lp.TesseractAgent(
            languages=os.environ.get("TESSERACT_LANGUAGE", self.TESSERACT_LANGUAGE),
            **self.TESSERACT_CONFIGS,
        )
        logger.debug("Model loaded successfully.")

    @Photon.handler
    def detect(self, image: Union[str, FileParam]) -> Dict[str, Any]:
        """
        Detects the layout of the image, and returns the layout in a dictionary. On the client
        side, if you want to recover the Layout object, you can use the `layoutparser.load_dict`
        functionality.
        """
        cv_image = self._load_image(image)
        with self.model_lock:
            layout = self.model.detect(cv_image)
        return layout.to_dict()

    @Photon.handler
    def draw_detection_box(
        self, image: Union[str, FileParam], box_width: int = 3
    ) -> PNGResponse:
        """
        Returns the detection box of the input image as a PNG image.
        """
        cv_image = self._load_image(image)
        with self.model_lock:
            layout = self.model.detect(cv_image)
        img = lp.draw_box(cv_image, layout, box_width=box_width)
        return make_png_response(img)

    @Photon.handler
    def ocr(
        self,
        image: Union[str, FileParam],
        return_response: bool = False,
        return_only_text: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Carries out Tesseract ocr for the input image. If return_response=True, the full response
        is returned as a dictionary with two keys: `text` containing the text, and `data` containing
        the full response from Tesseract, as a DataFrame converted to a dict. If you want to recover
        the original DataFrame, you can use `pandas.DataFrame.from_dict(result["data"])`.
        """
        cv_image = self._load_image(image)
        res = self.ocr_agent.detect(
            cv_image, return_response=return_response, return_only_text=return_only_text
        )
        print(type(res))
        print(str(res))
        if return_response:
            # The result is a dict with two keys: "text" being the text, and "data" being a DataFrame.
            # We will convert it to a dict with data converted to a dict.
            return {"text": res["text"], "data": res["data"].to_dict()}
        else:
            # The returned result is a string, so we will simply return it.
            return res

    @Photon.handler
    def draw_ocr_result(
        self,
        image: Union[str, FileParam],
        agg_level: int = 4,
        font_size: int = 12,
        with_box_on_text: bool = True,
        text_box_width: int = 1,
    ) -> PNGResponse:
        """
        Returns the OCR result of the input image as a PNG image. Optionally, specify agg_level to
        aggregate the text into blocks. The default agg_level is 4, which means that the text will
        be aggregated in words. Options are 3 (LINE), 2 (PARA), 1 (BLOCK), and 0 (PAGE).
        """
        try:
            agg_level_enum = lp.TesseractFeatureType(agg_level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"agg_level should be an integer between 0 and 4. Got {agg_level}."
                ),
            )
        cv_image = self._load_image(image)
        res = self.ocr_agent.detect(cv_image, return_response=True)
        layout = self.ocr_agent.gather_data(res, agg_level_enum)
        img = lp.draw_text(
            cv_image,
            layout,
            font_size=font_size,
            with_box_on_text=with_box_on_text,
            text_box_width=text_box_width,
        )
        return make_png_response(img)

    @classmethod
    def safe_load_model(cls, config_path: str):
        """
        A helper function to safely load the model to bypass the bug here:
        https://github.com/Layout-Parser/layout-parser/issues/168
        """
        # override storage path
        if not os.path.exists(cls.MODEL_SAVE_PATH):
            os.mkdir(cls.MODEL_SAVE_PATH)
        config_path_split = config_path.split("/")
        dataset_name = config_path_split[-3]
        model_name = config_path_split[-2]
        # get the URLs from the MODEL_CATALOG and the CONFIG_CATALOG
        # (global variables .../layoutparser/models/detectron2/catalog.py)
        model_url = catalog.MODEL_CATALOG[dataset_name][model_name]
        config_url = catalog.CONFIG_CATALOG[dataset_name][model_name]

        config_file_path, model_file_path = None, None

        for url in [model_url, config_url]:
            filename = url.split("/")[-1].split("?")[0]
            save_to_path = f"{cls.MODEL_SAVE_PATH}/" + filename
            if "config" in filename:
                config_file_path = copy.deepcopy(save_to_path)
            if "model_final" in filename:
                model_file_path = copy.deepcopy(save_to_path)

            # skip if file exist in path
            if filename in os.listdir(f"{cls.MODEL_SAVE_PATH}/"):
                continue
            # Download file from URL
            r = requests.get(
                url, stream=True, headers={"user-agent": "Wget/1.16 (linux-gnu)"}
            )
            with open(save_to_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=4096):
                    if chunk:
                        f.write(chunk)

        # load the label map
        label_map = catalog.LABEL_MAP_CATALOG[dataset_name]

        return lp.models.Detectron2LayoutModel(
            config_path=config_file_path,
            model_path=model_file_path,
            label_map=label_map,
        )

    def _load_image(self, image: Union[str, FileParam]):
        """
        Loads the image, and returns the cv.Image object. Throws HTTPError if the image
        cannot be loaded.
        """
        try:
            file_content = get_file_content(
                image, return_file=True, allow_local_file=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot open image with source: {image}. Detailed error message:"
                    f" {str(e)}"
                ),
            )
        try:
            cv_image = cv2.imread(file_content.name)
            cv_image = cv_image[..., ::-1]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot load image with source: {image}. Detailed error message:"
                    f" {str(e)}"
                ),
            )
        return cv_image


if __name__ == "__main__":
    ph = LayoutParser()
    ph.launch()
