"""memory_harvest — White-day memory extraction, directly to cards.

V2: Single main-brain call at compression time. The main brain already
experienced the conversation, so it extracts events, rates them, writes
reflections, and flags corrections — all in one pass. Results are written
directly as L0 cards via engine.ingest() — no draft layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from thepaw_memory.config_registry import cfg
from thepaw_memory.feature_log import feature_log

_log = lambda event, detail="": feature_log("memory_harvest", event, detail)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def harvest_evicted(
    persona_id: str,
    session_id: str,
    evicted_messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Extract memories from evicted messages and write them as L0 cards.

    Called by preemptive compression. Accepts raw evicted messages directly
    (no DB query needed). Returns dict with card_ids, corrections, events.
    (The parser also yields `facts`, but nothing consumes them yet.)
    """
    _log("start", f"persona={persona_id} msgs={len(evicted_messages)}")

    messages = [m for m in evicted_messages if not str(m.get("content") or "").startswith("[BFF_ERROR]")]
    if not messages:
        _log("skip_empty")
        return None

    result = await _main_brain_harvest(persona_id, messages, session_id=session_id)
    if not result or not result.get("events"):
        _log("no_events")
        return None

    events = result["events"]
    corrections = result.get("corrections", [])
    _log("extracted", f"events={len(events)} corrections={len(corrections)}")

    source_time = datetime.now().strftime("%Y-%m-%d") + " " + _time_period()

    card_ids = await _ingest_events(
        persona_id, session_id,
        events, corrections, source_time,
    )

    _log("done", f"cards={len(card_ids)} corrections={len(corrections)}")
    return {"card_ids": card_ids, "corrections": len(corrections), "events": events}


_SUMMARY_SEP = "\n---\n"
_SUMMARY_MAX_CHARS = 2000


def format_cards_as_summary(
    events: List[Dict[str, Any]],
    previous_summary: str = "",
) -> str:
    """Format extracted cards as compression summary for session continuity.

    Each card is separated by ``---``. When total exceeds 2000 chars the
    oldest *complete* cards are dropped first — no half-cut text.
    """
    import re as _re

    items: List[str] = []
    if previous_summary:
        items = [chunk.strip() for chunk in previous_summary.split(_SUMMARY_SEP)
                 if chunk.strip()]

    for e in events:
        record = (e.get("record") or "").strip()
        if not record:
            continue
        record = _re.sub(r"===.*", "", record, flags=_re.DOTALL).strip()
        if not record:
            continue
        topic = e.get("topic") or ""
        items.append(f"[{topic}] {record}" if topic else record)

    while items and len(_SUMMARY_SEP.join(items)) > _SUMMARY_MAX_CHARS:
        items.pop(0)

    return _SUMMARY_SEP.join(items)


def _time_period() -> str:
    hour = datetime.now().hour
    if hour < 6:
        return "凌晨"
    elif hour < 12:
        return "上午"
    elif hour < 18:
        return "下午"
    return "晚上"


# ---------------------------------------------------------------------------
# Main brain harvest call
# ---------------------------------------------------------------------------

async def _main_brain_harvest(
    persona_id: str,
    evicted_messages: List[Dict[str, Any]],
    session_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Single main-brain call on evicted messages: extract events + rate + reflect + correct."""
    from thepaw_memory.core.entity import get_persona_name
    persona_name = get_persona_name(persona_id)
    template = cfg("subbrain.prompt_harvest_combined", persona_id=persona_id)

    region_text = ""
    try:
        from thepaw_memory.card_retrieve import region_activation, format_cards_for_prompt
        evicted_text = " ".join(
            str(m.get("content") or "")[:200]
            for m in evicted_messages if m.get("role") == "user"
        )[:1000]
        if evicted_text:
            region_text = format_cards_for_prompt(region_activation(evicted_text, persona_id))
    except Exception as e:
        _log("region_activation_error", str(e)[:200])

    cards_block = ""
    if region_text:
        cards_block = f"以下是当前已有的记忆，供你参考：\n[相关记忆区域]\n{region_text}\n\n"

    prompt = template.format(
        persona_name=persona_name,
        cards_text=cards_block,
    )

    harvest_messages = list(evicted_messages) + [
        {"role": "user", "content": prompt},
    ]

    from thepaw_memory.main_brain import call as main_brain_call
    _log("harvest_call_start", f"persona={persona_id} msgs={len(evicted_messages)}")

    raw_text = await main_brain_call(
        persona_id, harvest_messages,
        max_tokens=4000, temperature=0.3,
        session_id=session_id,
    )
    if not raw_text:
        _log("harvest_call_empty", "main brain returned no text")
        return None

    _log("harvest_call_done", f"len={len(raw_text)}")

    from thepaw_memory.harvest_parser import parse_combined_harvest
    return parse_combined_harvest(raw_text)


# ---------------------------------------------------------------------------
# Direct ingest (V2: no draft layer)
# ---------------------------------------------------------------------------

async def _ingest_events(
    persona_id: str,
    session_id: str,
    events: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    source_time: str = "",
) -> List[str]:
    """Write each event directly as an L0 card via engine.ingest()."""
    from thepaw_memory.memory_engine import get_engine
    engine = get_engine(persona_id)

    card_ids = []
    per_event_corrections = _match_corrections(events, corrections)

    for i, event in enumerate(events):
        record = (event.get("record") or "").strip()
        if not record:
            continue
        try:
            card = await engine.ingest(
                content=record,
                source="harvest",
                session_id=session_id,
                level=0,
                source_time=source_time,
                subject=event.get("subject"),
                topic=event.get("topic"),
                retention=event.get("retention", 8.0),
            )
            card_ids.append(card["id"])
            _log("card_created", f"id={card['id']} subject={event.get('subject')}")
        except Exception as e:
            _log("ingest_error", f"event={i} error={e!s:.200}")

    # Apply corrections to existing cards
    event_corrections = []
    for corrs in per_event_corrections.values():
        event_corrections.extend(corrs)
    if event_corrections:
        await _apply_corrections(persona_id, engine, event_corrections)

    return card_ids



async def _apply_corrections(
    persona_id: str,
    engine,
    corrections: List[Dict[str, Any]],
) -> int:
    """Apply corrections to existing cards by searching for matching content."""
    from thepaw_memory.card_store import get_store
    store = get_store(persona_id)
    applied = 0
    for corr in corrections:
        original = (corr.get("original") or "").strip()
        user_said = (corr.get("user_said") or "").strip()
        if not original or not user_said:
            continue

        target = None
        for card in store.all_active():
            if card.get("is_portrait"):
                continue
            content = card.get("content", "")
            # >=4 (was >10): the parser accepts any non-empty `original`, but the old
            # >10 floor silently dropped valid short corrections (e.g. a 5-char fact).
            # 4 chars stays specific enough for the substring match, and a correction
            # only appends to edit_history (non-destructive), so a loose match is low-harm.
            if len(original) >= 4 and original[:60] in content:
                target = card
                break

        if not target:
            _log("correction_no_target", f"original={original[:50]}")
            continue

        from thepaw_memory.utils import now_iso
        eh = list(target.get("edit_history") or [])
        eh.append({
            "timestamp": now_iso(),
            "old_content": target.get("content", ""),
            "reason": f"correction: {corr.get('nature', 'fix')}",
        })
        new_content = target["content"].replace(original, user_said)
        if new_content == target["content"]:
            new_content = f"{target['content']}\n[纠正] {user_said}"
        store.update(target["id"], content=new_content, edit_history=eh)
        applied += 1
        _log("correction_applied", f"card={target['id']} nature={corr.get('nature')}")

    return applied


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_corrections(
    events: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Match corrections to events by subject/content overlap. Unmatched → event[0]."""
    if not corrections or not events:
        return {0: corrections} if corrections else {}

    result: Dict[int, List[Dict[str, Any]]] = {}
    for corr in corrections:
        original = corr.get("original", "")
        user_said = corr.get("user_said", "")
        corr_text = f"{original} {user_said}"

        best_idx = 0
        best_score = 0
        for i, event in enumerate(events):
            score = 0
            subj = (event.get("subject") or "").strip()
            if subj and subj in corr_text:
                score += 2
            record = event.get("record", "")
            if original and len(original) > 5 and original[:20] in record:
                score += 3
            if score > best_score:
                best_score = score
                best_idx = i

        result.setdefault(best_idx, []).append(corr)

    return result




