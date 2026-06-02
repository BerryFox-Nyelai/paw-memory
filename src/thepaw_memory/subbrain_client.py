"""Subbrain seam — routes to the host-injected async LLM callable.

Adapts the host's subbrain function to the ``call`` / ``call_safe`` shapes the
memory engine expects, including a ``CallResult`` carrying truncation metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from thepaw_memory import _seams


@dataclass
class CallResult:
    content: str = ""
    finish_reason: str = ""
    truncated: bool = False


def _require():
    if _seams.subbrain_fn is None:
        raise RuntimeError(
            "thepaw_memory: subbrain LLM not configured. "
            "Pass subbrain=<async fn> to MemoryEngine(...)."
        )
    return _seams.subbrain_fn


async def call(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.1,
    persona_id: str = "",
    return_meta: bool = False,
    timeout: int = 0,
):
    fn = _require()
    content = await fn(
        messages=messages, max_tokens=max_tokens,
        temperature=temperature, persona_id=persona_id,
    )
    content = content or ""
    if return_meta:
        return CallResult(content=content)
    return content


async def call_safe(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.1,
    persona_id: str = "",
) -> CallResult:
    fn = _require()
    content = await fn(
        messages=messages, max_tokens=max_tokens,
        temperature=temperature, persona_id=persona_id,
    )
    return CallResult(content=content or "")
