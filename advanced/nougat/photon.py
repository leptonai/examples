from io import BytesIO
import os
import traceback
from typing import Union, Optional

from loguru import logger
import torch
from leptonai.photon import (
    Photon,
    FileParam,
    get_file_content,
    HTTPException,
    StreamingResponse,
)


class Nougat(Photon):
    requirement_dependency = [
        "git+https://github.com/facebookresearch/nougat.git@84b3ae1",
        "torch",
        "pypdf",
        "loguru",
        "opencv-python!=4.8.0.76",
    ]

    system_dependency = ["poppler-utils"]

    def init(self):
        from nougat import NougatModel
        from nougat.utils.checkpoint import get_checkpoint

        model_tag = os.environ.get(
            "MODEL_TAG", "0.1.0-small"
        )  # 0.1.0-small or 0.1.0-base
        checkpoint = get_checkpoint(model_tag=model_tag)
        model = NougatModel.from_pretrained(checkpoint)
        if torch.cuda.is_available():
            model = model.to("cuda")
        self.model = model.to(torch.bfloat16).eval()
        self.batch_size = os.environ.get("BATCH_SIZE", 4)

    def iter_batch(self, iterable, batch_size):
        for start in range(0, len(iterable), batch_size):
            yield iterable[start : min(start + batch_size, len(iterable))]

    def gen_pages(self, pdf, start, end):
        from nougat.dataset.rasterize import rasterize_paper
        from PIL import Image
        from nougat.postprocessing import markdown_compatible

        pages = list(range(start - 1, end))
        for batch_pages in self.iter_batch(pages, self.batch_size):
            image_bytes_list = rasterize_paper(pdf, pages=batch_pages, return_pil=True)
            images = [
                self.model.encoder.prepare_input(
                    Image.open(image_bytes), random_padding=False
                )
                for image_bytes in image_bytes_list
            ]
            model_output = self.model.inference(image_tensors=torch.stack(images))
            logger.info(
                f"#input pages: {len(batch_pages)}, #output pages:"
                f" {len(model_output['predictions'])}"
            )
            for page_prediction in model_output["predictions"]:
                content = markdown_compatible(page_prediction)
                yield content

    @Photon.handler
    def run(
        self,
        file: Union[FileParam, str],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> StreamingResponse:
        import pypdf

        try:
            content = get_file_content(file)
            pdf = pypdf.PdfReader(BytesIO(content))
        except Exception:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail="Failed to read PDF file.")

        total_pages = len(pdf.pages)
        start = start or 1
        end = end or total_pages
        logger.info(f"Total pages: {total_pages}, start: {start}, end: {end}")
        if start < 1 or end > total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Page number should be in range [1, {total_pages}]",
            )
        if start > end:
            raise HTTPException(
                status_code=400, detail="Start page number should be less than end."
            )

        return self.gen_pages(pdf, start, end)
