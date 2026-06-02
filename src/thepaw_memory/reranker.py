"""MemReranker integration — lazy-loaded CrossEncoder with auto-unload after idle."""
from __future__ import annotations

import time
import threading
from typing import Any, Dict, List

from thepaw_memory.config_registry import cfg
from thepaw_memory.feature_log import feature_log
from thepaw_memory.card_store import card_text as _card_text

_log = lambda event, detail="": feature_log("reranker", event, detail)

_model = None
_lock = threading.RLock()
_refcount = 0
_last_use: float = 0
_unload_timer: threading.Timer | None = None
_load_failed_until: float = 0

IDLE_TIMEOUT = 600  # 10 minutes
LOAD_COOLDOWN = 300  # 5 min cooldown after load failure


def _load_model():
    global _model, _load_failed_until
    if _model is not None:
        return _model
    if time.monotonic() < _load_failed_until:
        return None
    with _lock:
        if _model is not None:
            return _model
        if time.monotonic() < _load_failed_until:
            return None
        path = cfg("reranker.model_path")
        try:
            from sentence_transformers import CrossEncoder
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            _model = CrossEncoder(
                path,
                model_kwargs={"quantization_config": quantization_config, "device_map": "cuda"},
            )
            _model.prompts["memory"] = (
                "You are in a conversation. Select the memory that best helps you "
                "respond — match the topic, mood, and what the speaker seems to need right now."
            )
            _model.default_prompt_name = "memory"
            _log("model_loaded", f"path={path} int8=true")
        except Exception as e:
            _load_failed_until = time.monotonic() + LOAD_COOLDOWN
            _log("model_load_error", f"path={path} cooldown={LOAD_COOLDOWN}s error={e}")
    return _model


def _unload_model():
    global _model, _unload_timer, _refcount
    with _lock:
        if _model is None:
            return
        if _refcount > 0:
            _log("unload_deferred", f"refcount={_refcount}")
            return
        if time.monotonic() - _last_use < IDLE_TIMEOUT - 5:
            return
        try:
            import torch, gc
            del _model
            _model = None
            gc.collect()
            torch.cuda.empty_cache()
            _log("model_unloaded", f"idle>{IDLE_TIMEOUT}s")
        except Exception as e:
            _model = None
            _log("unload_error", str(e)[:200])
        _unload_timer = None


def _schedule_unload():
    global _unload_timer
    if _unload_timer is not None:
        _unload_timer.cancel()
    _unload_timer = threading.Timer(IDLE_TIMEOUT, _unload_model)
    _unload_timer.daemon = True
    _unload_timer.start()


def unload():
    """Manually free VRAM. Call before launching a game, etc."""
    global _last_use
    _last_use = 0
    _unload_model()


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Rerank candidates using MemReranker-4B. Returns top_k cards sorted by relevance."""
    global _last_use, _refcount

    if not candidates:
        return []

    model = _load_model()
    _last_use = time.monotonic()
    _schedule_unload()

    if model is None:
        _log("fallback_score", f"n={len(candidates)}")
        scored = sorted(candidates, key=lambda c: c.get("_energy", 0), reverse=True)
        return scored[:top_k]

    with _lock:
        _refcount += 1
    try:
        docs = [_card_text(c) for c in candidates]
        results = model.rank(query, docs, top_k=min(top_k, len(docs)))
        reranked = []
        for r in results:
            card = {**candidates[r["corpus_id"]], "_rerank_score": r["score"]}
            reranked.append(card)
        _log("rerank_ok",
             f"n={len(candidates)} top_k={top_k} "
             f"scores={[round(r['score'], 4) for r in results]}")
        return reranked
    except Exception as e:
        _log("rerank_error", str(e)[:200])
        scored = sorted(candidates, key=lambda c: c.get("_energy", 0), reverse=True)
        return scored[:top_k]
    finally:
        with _lock:
            _refcount -= 1
