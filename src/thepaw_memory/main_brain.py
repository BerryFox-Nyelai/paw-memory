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
    session_id: str = "",
) -> Optional[str]:
    if _seams.main_brain_fn is None:
        raise RuntimeError(
            "thepaw_memory: main-brain LLM not configured. "
            "Pass main_brain=<async fn> to MemoryEngine(...)."
        )
    # session_id is accepted for parity with the host app (which uses it for
    # provider-side session/prompt caching), but is NOT forwarded to the host's
    # main_brain callback — its documented contract is (persona_id, messages, *,
    # max_tokens, temperature). Hosts that want session-scoped caching can manage
    # it on their side via the session_id they already pass to the context builder.
    return await _seams.main_brain_fn(
        persona_id=persona_id, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )


# In this package `call` already passes the caller's messages through verbatim (the
# host's callback owns any system-context prepending), so there is no separate "raw"
# variant — call_raw is the same seam, provided for import parity with the host app's
# main_brain (memory_review imports call_raw).
call_raw = call
