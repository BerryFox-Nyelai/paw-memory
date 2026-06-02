"""Session-dedup seam.

The read path drops cards already injected earlier in the same conversation
("draw without replacement"). The host owns conversation state, so it may inject
a lookup; by default we return an empty set (no dedup). Callers can also pass
already-injected ids directly to the retrieve path.
"""
from __future__ import annotations

from typing import Set

from thepaw_memory import _seams


def injected_card_ids_for_session(session_id: str) -> Set[str]:
    if _seams.injected_ids_fn is not None:
        return _seams.injected_ids_fn(session_id)
    return set()
