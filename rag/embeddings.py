"""Free, local embeddings (ONNX on CPU via fastembed). No API key, no cost."""
from __future__ import annotations

from functools import lru_cache

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 0.22 GB, 384 dims, Apache-2.0


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts)]
