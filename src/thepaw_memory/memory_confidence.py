# ============================================================
# Memory Confidence — Round 2: subbrain text-protocol retrieval
#
# Sits between card_retrieve (Round 1) and prompt injection.
# Subbrain sees full conversation + Round 1 candidates, reviews
# via text markers ([保留]/[搜索]), system parses and acts.
# ============================================================
from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from thepaw_memory.config_registry import cfg
from thepaw_memory.feature_log import feature_log

_log = lambda event, detail="": feature_log("memory_confidence", event, detail)


# ============================================================
# Public entry point
# ============================================================

async def review_and_filter(
    user_text: str,
    retrieve_result: Dict[str, Any],
    persona_id: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    frontend_sys: Optional[List[Dict[str, Any]]] = None,
    portraits_prompt: str = "",
) -> Tuple[Dict[str, Any], str]:
    """Round 2: subbrain text-protocol review of Round 1 candidates.

    Receives the same context as the main brain (system prompt, portraits,
    conversation history) so it can judge relevance with full understanding.

    Returns (filtered_result, hint).
    """
    # spreading activation merges all tracks into `activated`
    search_candidates = retrieve_result.get("activated",
        retrieve_result.get("search_candidates", []))

    if not search_candidates:
        return retrieve_result, ""

    all_card_map: Dict[str, Dict[str, Any]] = {}
    for c in search_candidates:
        all_card_map[c.get("id", "")] = c

    exclude_ids = {c.get("id", "") for c in retrieve_result.get("cards", [])}

    try:
        keep_ids = await _text_loop(
            persona_id=persona_id,
            user_text=user_text,
            messages=messages,
            frontend_sys=frontend_sys,
            portraits_prompt=portraits_prompt,
            search_candidates=search_candidates,
            all_card_map=all_card_map,
            exclude_ids=exclude_ids,
        )
        kept = [all_card_map[cid] for cid in keep_ids if cid in all_card_map]

        _log("review_ok", f"input={len(all_card_map)} output={len(kept)} ids={keep_ids}")

        _update_reference_counts(kept, persona_id)

    except Exception as e:
        _log("review_fallback", f"text-loop failed: {e}")
        kept = _rule_fallback(search_candidates)

    new_result = _rebuild_result(retrieve_result, kept)
    return new_result, ""


# ============================================================
# Text-protocol loop
# ============================================================

_RE_KEEP = re.compile(r"\[保留\]\s*(.+)", re.DOTALL)
_RE_SEARCH = re.compile(r"\[搜索\]\s*(.+)")


def _parse_action(text: str, card_map: Dict[str, Dict]) -> Dict[str, Any]:
    """Parse subbrain text response into action dict.

    Returns {"type": "keep", "ids": [...]} or {"type": "search", "keywords": "..."}.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _find_ids(haystack: str) -> List[str]:
        return [cid for cid in card_map if re.search(rf'(?<![a-z0-9_]){re.escape(cid)}(?![a-z0-9_])', haystack)]

    m = _RE_KEEP.search(text)
    if m:
        raw = m.group(1).strip()
        if raw in ("无", "空", "[]", "无相关记忆"):
            return {"type": "keep", "ids": []}
        found = _find_ids(raw)
        return {"type": "keep", "ids": found}

    m = _RE_SEARCH.search(text)
    if m:
        keywords = m.group(1).strip().split("\n")[0].strip()
        if keywords:
            return {"type": "search", "keywords": keywords}

    found = _find_ids(text)
    if found:
        return {"type": "keep", "ids": found}
    stripped = text.strip()
    if stripped in ("无", "无关", "没有", "没有相关记忆", "无相关记忆", "不需要"):
        return {"type": "keep", "ids": []}

    return {"type": "unknown"}


async def _text_loop(
    persona_id: str,
    user_text: str,
    messages: Optional[List[Dict[str, Any]]],
    frontend_sys: Optional[List[Dict[str, Any]]],
    portraits_prompt: str,
    search_candidates: List[Dict],
    all_card_map: Dict[str, Dict],
    exclude_ids: set,
) -> List[str]:
    """Text-protocol conversation with subbrain. Returns keep_ids.

    Subbrain sees recent conversation context (last 5 rounds) + filter
    instruction as system prompt. No persona identity — it's a filter,
    not a roleplay participant.
    """
    from thepaw_memory.subbrain_client import call as sb_call

    candidates_text = _format_candidates(search_candidates)

    from thepaw_memory.core.entity import get_persona_name
    persona_name = get_persona_name(persona_id)
    filter_instruction = cfg("memory_confidence.prompt_text", persona_id=persona_id).format(
        persona_name=persona_name,
    )

    # Recent conversation context (last 5 rounds = 10 messages)
    recent: List[Dict[str, Any]] = []
    for msg in (messages or []):
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            recent.append({"role": role, "content": content})
    recent = recent[-10:]

    context_lines = []
    for msg in recent:
        tag = "用户" if msg["role"] == "user" else persona_name
        context_lines.append(f"[{tag}]: {msg['content']}")
    context_text = "\n\n".join(context_lines)

    sb_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": filter_instruction},
        {"role": "user", "content": (
            f"## 最近的对话\n{context_text}\n\n"
            f"## 用户最新消息\n{user_text}\n\n"
            f"## 候选记忆（共 {len(all_card_map)} 条）\n{candidates_text}"
        )},
    ]

    max_rounds = int(cfg("memory_confidence.max_tool_rounds") or 3)
    for round_i in range(max_rounds):
        resp = await sb_call(
            sb_messages,
            max_tokens=int(cfg("memory_confidence.max_tokens") or 800),
            temperature=0.1, persona_id=persona_id,
        )

        if not resp:
            _log("empty_response", f"round={round_i}")
            break

        resp = resp.strip()
        _log("sb_response", f"round={round_i} len={len(resp)} text={resp[:500]}")
        action = _parse_action(resp, all_card_map)

        if action["type"] == "keep":
            _log("text_keep", f"round={round_i} ids={action['ids']}")
            return action["ids"]

        if action["type"] == "search":
            keywords = action["keywords"]
            _log("text_search", f"round={round_i} keywords={keywords}")

            search_result = _execute_memory_search(
                keywords, persona_id, exclude_ids, all_card_map,
            )

            sb_messages.append({"role": "assistant", "content": resp})
            sb_messages.append({"role": "user", "content": search_result})
            continue

        _log("text_unknown", f"round={round_i} content={resp[:200]}")
        # Can't parse — try to extract IDs as last resort
        found = [cid for cid in all_card_map if re.search(rf'(?<![a-z0-9_]){re.escape(cid)}(?![a-z0-9_])', resp)]
        if found:
            return found
        break

    _log("max_rounds_fallback", f"rounds={max_rounds}")
    return _fallback_keep_ids(search_candidates)


# ============================================================
# Search execution
# ============================================================

def _execute_memory_search(
    keywords: str,
    persona_id: str,
    exclude_ids: set,
    existing_map: Dict[str, Dict],
) -> str:
    """Run retrieve() with subbrain-provided keywords."""
    if not keywords.strip():
        return "请提供搜索关键词。"

    from thepaw_memory.card_retrieve import retrieve
    result = retrieve(query=keywords, persona_id=persona_id,
                      memory_count=10, exclude_ids=exclude_ids)

    new_cards = []
    for c in result.get("activated", result.get("search_candidates", [])):
        cid = c.get("id", "")
        if cid not in existing_map:
            existing_map[cid] = c
            new_cards.append(c)

    if not new_cards:
        return "没有找到新的相关记忆。请用 [保留] 从已有候选中选择，或 [保留] 无。"

    lines = ["补搜到以下记忆，加上之前的候选一起选择："]
    for c in new_cards:
        lines.append(_format_single_card(c))
    lines.append("\n请用 [保留] id1, id2, ... 输出最终选择（从所有候选中选）。")
    return "\n".join(lines)


# ============================================================
# Formatting helpers
# ============================================================


def _format_candidates(search_candidates: List[Dict]) -> str:
    """Format activated candidates for subbrain filtering."""
    lines: List[str] = []
    for c in search_candidates:
        lines.append(
            f"  id={c['id']} (L{c.get('level', '?')}) "
            f"[{c.get('subject', '?')}/{c.get('topic', '?')}] "
            f"{c.get('content', '')}"
        )
    return "\n".join(lines)


def _format_single_card(c: Dict[str, Any]) -> str:
    return (
        f"- id={c.get('id', '?')} (L{c.get('level', '?')}) "
        f"[{c.get('subject', '?')}/{c.get('topic', '?')}] "
        f"{c.get('content', '')}"
    )




# ============================================================
# Fallback and helpers
# ============================================================

def _rule_fallback(search_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score-based fallback when subbrain is unavailable."""
    threshold = cfg("memory_confidence.fallback_score_threshold", persona_id="")
    return [c for c in search_candidates if (c.get("_energy") or c.get("_score", 0)) >= threshold]


def _fallback_keep_ids(search_candidates: List[Dict]) -> List[str]:
    """When text loop fails, return top search candidates by score."""
    memory_count = int(cfg("card_retrieve.default_memory_count") or 15)
    top = search_candidates[:memory_count]
    return [c.get("id", "") for c in top if c.get("id")]


def _update_reference_counts(kept: List[Dict], persona_id: str) -> None:
    if not kept:
        return
    try:
        from thepaw_memory.card_store import get_store
        store = get_store(persona_id)
        store.update_reference([c["id"] for c in kept if c.get("id")])
    except Exception as e:
        _log("ref_update_error", str(e)[:200])


def _rebuild_result(
    original: Dict[str, Any],
    filtered_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    filtered_ids = {c.get("id") for c in filtered_cards}
    new_hits = [h for h in original.get("hits_debug", []) if h.get("id") in filtered_ids]

    return {
        "cards": filtered_cards,
        "search_mode": original.get("search_mode", "?"),
        "total_raw": original.get("total_raw", 0),
        "hits_debug": new_hits,
    }
