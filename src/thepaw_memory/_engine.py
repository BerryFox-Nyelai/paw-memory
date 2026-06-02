"""MemoryEngine — the public facade over the card-memory pipeline.

Four entry points wrap the underlying modules:
  retrieve / portrait  — read path (per chat turn, no LLM cost)
  ingest               — write path (post-response: harvest cards from evicted msgs)
  review               — nightly maintenance (summaries, dedup, retention)

Construction wires the host-injected seams (LLMs, optional context builder /
session-dedup) and points the data root at ``base_dir`` *before* any core
module is imported, so per-persona stores land under ``base_dir/<persona_id>/``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryEngine:
    def __init__(
        self,
        base_dir,
        *,
        subbrain=None,
        main_brain=None,
        context_builder=None,
        injected_ids_fn=None,
    ):
        # Point the data root at base_dir BEFORE importing any core module:
        # paths.MEMORY_BASE is resolved at import time from this env var.
        base = Path(base_dir).resolve()
        os.environ["THEPAW_MEMORY_BASE"] = str(base)
        os.environ.setdefault("THEPAW_DATA_DIR", str(base.parent))

        from thepaw_memory import _seams
        _seams.subbrain_fn = subbrain
        _seams.main_brain_fn = main_brain
        _seams.context_builder = context_builder
        _seams.injected_ids_fn = injected_ids_fn

        self._base = base

        # Discover personas already on disk under base_dir.
        from thepaw_memory.core.registry import entity_registry
        entity_registry.load()

    def register_persona(
        self,
        persona_id: str,
        *,
        display_name: Optional[str] = None,
        user_display_name: str = "",
        system_prompt: str = "",
    ) -> None:
        """Create a persona partition under base_dir and register it.

        A persona must exist before its store/retrieve/ingest can be used. This
        writes ``<base_dir>/<persona_id>/persona_meta.json`` and refreshes the
        in-memory registry. Idempotent — safe to call on an existing persona.
        """
        import json
        from thepaw_memory.utils import safe_filename
        from thepaw_memory.paths import MEMORY_BASE
        from thepaw_memory.core.registry import entity_registry

        pid = safe_filename(persona_id)
        pdir = MEMORY_BASE / pid
        pdir.mkdir(parents=True, exist_ok=True)
        meta_path = pdir / "persona_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta.update({
            "persona_id": pid,
            "display_name": display_name or meta.get("display_name") or pid,
            "user_display_name": user_display_name or meta.get("user_display_name", ""),
            "system_prompt_template": system_prompt or meta.get("system_prompt_template", ""),
        })
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        entity_registry.load()

    # ---------------------------------------------------------------- READ
    def retrieve(self, persona_id: str, query: str, *, session_id: str = "") -> Dict[str, Any]:
        """Topic-background activation for the current turn.

        Returns the raw result dict plus ``prompt_text`` (ready to inject) and
        ``card_ids`` (for the host's session-dedup bookkeeping).
        """
        from thepaw_memory.card_retrieve import region_activation, format_cards_for_prompt
        result = region_activation(query, persona_id, session_id=session_id)
        result["prompt_text"] = format_cards_for_prompt(result)
        result["card_ids"] = [c.get("id") for c in result.get("cards", []) if c.get("id")]
        return result

    def portrait(self, persona_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        from thepaw_memory.card_retrieve import get_portrait_injection
        return get_portrait_injection(persona_id, user_id)

    # --------------------------------------------------------------- WRITE
    async def ingest(
        self,
        persona_id: str,
        evicted_messages: List[Dict[str, Any]],
        *,
        session_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Harvest memories from messages leaving the context window."""
        from thepaw_memory.memory_harvest import harvest_evicted
        return await harvest_evicted(persona_id, session_id, evicted_messages)

    # ------------------------------------------------------------- NIGHTLY
    async def review(
        self,
        persona_id: str,
        today_cards: Optional[List[Dict[str, Any]]] = None,
        last_review_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        from thepaw_memory.memory_review import NightlyReview
        return await NightlyReview(persona_id).run(today_cards or [], last_review_at)

    # ---------------------------------------------------------- direct store
    def store(self, persona_id: str):
        """Low-level card store for manual add / list / delete."""
        from thepaw_memory.card_store import get_store
        return get_store(persona_id)
