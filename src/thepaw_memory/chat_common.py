"""Conversation-context seam.

Only ``build_context_messages`` is needed by the memory engine — nightly review
calls the main brain "through the same pipeline as normal chat". If the host
injects its own context builder we use it; otherwise we fall back to a minimal
single-turn message list.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from thepaw_memory import _seams


async def build_context_messages(
    persona_id: str,
    session_id: str = "",
    user_text: str = "",
    **kwargs,
) -> Tuple[List[Dict[str, Any]], Any, Any]:
    if _seams.context_builder is not None:
        return await _seams.context_builder(
            persona_id=persona_id, session_id=session_id, user_text=user_text,
        )
    return [{"role": "user", "content": user_text}], None, None
