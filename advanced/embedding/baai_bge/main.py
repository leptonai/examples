import os
from typing import List, Union, Tuple

from leptonai.photon import Photon, HTTPException


# Transcribed from https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list
AVAILABLE_MODELS_AND_INSTRUCTIONS = {
    "BAAI/llm-embedder": None,
    "BAAI/bge-reranker-large": None,
    "BAAI/bge-reranker-base": None,
    "BAAI/bge-large-en-v1.5": (
        "Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-base-en-v1.5": (
        "Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-base-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-base-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh": "为这个句子生成表示以用于检索相关文章：",
}


class BGEEmbedding(Photon):
    """
    The BGE embedding model from BAAI.
    """

    requirement_dependency = [
        "FlagEmbedding",
    ]

    # manage the max concurrency of the photon. This is the number of requests
    # that can be handled at the same time.
    handler_max_concurrency = 4

    DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DEFAULT_QUERY_INSTRUCTION = AVAILABLE_MODELS_AND_INSTRUCTIONS[DEFAULT_MODEL_NAME]
    DEFAULT_USE_FP16 = True
    DEFAULT_NORMALIZE_EMBEDDINGS = True

    def init(self):
        from FlagEmbedding import FlagModel

        model_name = os.environ.get("MODEL_NAME", self.DEFAULT_MODEL_NAME)
        if model_name not in AVAILABLE_MODELS_AND_INSTRUCTIONS:
            raise ValueError(
                f"Model name {model_name} not found. Available models:"
                f" {AVAILABLE_MODELS_AND_INSTRUCTIONS.keys()}"
            )
        query_instruction = os.environ.get(
            "QUERY_INSTRUCTION", self.DEFAULT_QUERY_INSTRUCTION
        )
        use_fp16 = os.environ.get("USE_FP16", self.DEFAULT_USE_FP16)
        normalize_embeddings = os.environ.get(
            "NORMALIZE_EMBEDDINGS", self.DEFAULT_NORMALIZE_EMBEDDINGS
        )
        self._model = FlagModel(
            model_name,
            query_instruction_for_retrieval=query_instruction,
            use_fp16=use_fp16,
            normalize_embeddings=normalize_embeddings,
        )

    @Photon.handler
    def encode(self, sentences: Union[str, List[str]]) -> List[float]:
        """
        Encodes the current sentences into embeddings.
        """
        embeddings = self._model.encode(sentences)
        return embeddings.tolist()

    @Photon.handler
    def rank(self, query: str, sentences: List[str]) -> Tuple[List[int], List[float]]:
        """
        Returns a ranked list of indices of the most relevant sentences. This uses
        the inner product of the embeddings to rank the sentences. If the model is
        not initialized as normalize_embeddings=True, this will raise an error. The
        relative similarity scores are also returned.
        """
        if not self._model.normalize_embeddings:
            raise HTTPException(
                status_code=500,
                detail="Model must have normalize_embeddings=True to use rank.",
            )
        embeddings = self._model.encode([query] + sentences)
        query_embedding = embeddings[0]
        sentence_embeddings = embeddings[1:]
        inner_product = query_embedding @ sentence_embeddings.T
        sorted_indices = inner_product.argsort()[::-1]
        return sorted_indices.tolist(), inner_product[sorted_indices].tolist()


if __name__ == "__main__":
    # TODO: change the name of the class "MyPhoton" to the name of your photon
    ph = BGEEmbedding()
    ph.launch(port=8080)
