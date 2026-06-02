# ============================================================
# Embeddings — sentence-transformer encoding
# ============================================================
from __future__ import annotations

import os
import threading
from typing import List

import numpy as np

MODEL_NAME = os.environ.get("THEPAW_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

_model = None
_model_lock = threading.Lock()


# ===== Model loading (lazy, one-time) =====

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: List[str]) -> np.ndarray:
    """Encode texts to embeddings. Returns (n, dim) array."""
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def encode_one(text: str) -> np.ndarray:
    """Encode single text. Returns (dim,) array."""
    return encode([text])[0]


def is_available() -> bool:
    """Check if vector search dependencies are installed."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False
