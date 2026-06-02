"""Host-injected seams. ``MemoryEngine`` populates these at construction.

The memory engine's *write* path needs two LLMs — a "main brain" to harvest
memories from a conversation, and a "subbrain" for summaries / nightly review —
plus, optionally, the host's conversation-context builder and its session-dedup
source. These are injected rather than bundled so the library never carries
provider keys or host-specific chat plumbing.

Expected callable shapes:
    subbrain_fn:      async (messages, *, max_tokens, temperature, persona_id) -> str
    main_brain_fn:    async (persona_id, messages, *, max_tokens, temperature) -> str | None
    context_builder:  async (persona_id, session_id, user_text) -> (messages, _, _)
    injected_ids_fn:  (session_id) -> set[str]
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

SubbrainFn = Callable[..., Awaitable[str]]
MainBrainFn = Callable[..., Awaitable[Optional[str]]]
ContextBuilderFn = Callable[..., Awaitable[Tuple[List[Dict[str, Any]], Any, Any]]]
InjectedIdsFn = Callable[[str], Set[str]]

subbrain_fn: Optional[SubbrainFn] = None
main_brain_fn: Optional[MainBrainFn] = None
context_builder: Optional[ContextBuilderFn] = None
injected_ids_fn: Optional[InjectedIdsFn] = None
