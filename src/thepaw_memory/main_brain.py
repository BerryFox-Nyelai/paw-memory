"""Main-brain seam — routes to the host-injected async LLM callable.

Used by the write path to harvest memories from a conversation (memory_harvest)
and by nightly review's main-brain pass.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from thepaw_memory import _seams


async def call(
    persona_id: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 1500,
    temperature: float = 0.7,
) -> Optional[str]:
    if _seams.main_brain_fn is None:
        raise RuntimeError(
            "thepaw_memory: main-brain LLM not configured. "
            "Pass main_brain=<async fn> to MemoryEngine(...)."
        )
    return await _seams.main_brain_fn(
        persona_id=persona_id, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
