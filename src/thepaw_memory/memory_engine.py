# ============================================================
# Memory Engine — unified card write layer
#
# All card creation goes through engine.ingest().
# ============================================================
from __future__ import annotations

from typing import Any, Dict, List, Optional

from thepaw_memory.config_registry import cfg
from thepaw_memory.card_store import (
    CardStore, get_store, new_card, normalize_level,
)
from thepaw_memory.feature_log import feature_log

_log = lambda event, detail="": feature_log("memory_engine", event, detail)


# ============================================================
# MemoryEngine
# ============================================================

class MemoryEngine:
    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self._store: CardStore = get_store(persona_id)

    # --------------------------------------------------------
    # ingest: unified write entry point
    # --------------------------------------------------------

    async def ingest(
        self,
        content: str,
        source: str = "conversation",
        session_id: Optional[str] = None,
        *,
        level: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        card_id: Optional[str] = None,
        source_time: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        needs_review: Optional[bool] = None,
        card_type: Optional[str] = None,
        retention: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not content or not content.strip():
            raise ValueError("empty content")

        content, fit_flagged = await self._ensure_token_fit(content.strip())
        if fit_flagged:
            needs_review = True

        resolved_level = normalize_level(level if level is not None else 0)

        card = new_card(
            content=content,
            level=resolved_level,
            keywords=keywords or [],
            tags=tags or [],
            card_id=card_id,
            source=source,
            session_id=session_id,
            source_time=source_time,
            subject=subject,
            topic=topic,
            annotations=annotations,
            card_type=card_type,
            retention=retention,
        )
        if needs_review:
            card["needs_review"] = True
        self._store.append(card)
        _log("ingest", f"id={card['id']} level={resolved_level} source={source}")

        # shadow(Phase 1)：把这张卡撒进个人联想地图(共现图)。
        # 只写独立 graph.db、检索侧零依赖；包死异常，绝不影响写路径。
        try:
            from thepaw_memory.memory_graph import on_card_created
            on_card_created(self.persona_id, card)
        except Exception:
            pass

        from thepaw_memory.card_retrieve import invalidate
        invalidate(self.persona_id)

        return dict(card)

    async def _ensure_token_fit(self, content: str) -> tuple:
        """Returns (content, needs_review: bool)."""
        from thepaw_memory.utils import estimate_tokens, clamp_content
        max_tokens = int(cfg("card_store.max_card_tokens", persona_id=self.persona_id))
        if estimate_tokens(content) <= max_tokens:
            return content, False
        _log("card_over_cap", f"tokens={estimate_tokens(content)} cap={max_tokens}")
        from thepaw_memory.subbrain_client import call as sb_call
        condensed = await sb_call(
            [{"role": "user", "content":
              f"以下内容超过了记忆卡片的存储上限，请精简到350字以内，"
              f"保留核心信息和情感细节，不要添加任何解释：\n\n{content}"}],
            max_tokens=1000, temperature=0.2, persona_id=self.persona_id,
        )
        if condensed and estimate_tokens(condensed.strip()) <= max_tokens:
            _log("condense_ok", "subbrain")
            return condensed.strip(), False
        _log("condense_clamped", "subbrain failed, clamping content")
        return clamp_content(content, max_tokens), True

    def get_stats(self) -> Dict[str, Any]:
        active = self._store.all_active()
        by_level: Dict[int, int] = {}
        for c in active:
            lv = c.get("level", 0)
            by_level[lv] = by_level.get(lv, 0) + 1
        return {
            "card_count": len(active),
            "by_level": by_level,
            "has_portrait": bool(self._store.get_portrait("persona")),
        }


# ============================================================
# Singleton cache
# ============================================================

_engines: Dict[str, MemoryEngine] = {}


def get_engine(persona_id: str) -> MemoryEngine:
    if persona_id not in _engines:
        _engines[persona_id] = MemoryEngine(persona_id)
    return _engines[persona_id]
