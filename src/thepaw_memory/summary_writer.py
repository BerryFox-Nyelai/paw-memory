"""summary_writer — Generate and rewrite node summaries using subbrain.

Checks which nodes on the dictionary map are "thick enough" to deserve a summary,
gathers their cards, and calls the subbrain to write/rewrite a concise summary.
Runs async, meant to be called from nightly review or heartbeat.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from thepaw_memory.config_registry import cfg
from thepaw_memory.feature_log import feature_log

_log = lambda event, detail="": feature_log("summary_writer", event, detail)

SUMMARY_PROMPT_NEW = """下面是我关于「{node}」的{n}条记忆。
请用我（{persona_name}）的第一人称视角，写一段简洁的背景摘要（300字以内）。
摘要必须以「{node}」为主角——写的是「{node}」本身是什么、我对「{node}」的了解和印象。
像自己在心里默默整理记忆那样写，不要用"根据记录"这类话，直接写。

重要：如果翻遍这些记忆，「{node}」都不是主角（只是被顺嘴提到、或者记忆其实在讲别人别的事），就只输出「跳过」两个字。宁可跳过也不要写成别人的介绍。

我的记忆：
{cards_text}"""

SUMMARY_PROMPT_REWRITE = """下面是我之前整理的关于「{node}」的摘要，以及{n}条新记忆。
请用我（{persona_name}）的第一人称视角，把新记忆的内容融入摘要，整块重写一版（300字以内）。
如果新记忆跟旧摘要矛盾，以新的为准，把更正自然融进去，不要标"[纠正]"。
像自己在心里更新对一个人/一件事的理解那样写，直接写。

我之前整理的：
{old_summary}

新记忆：
{cards_text}"""


async def write_summaries(
    persona_id: str,
    min_df: int = 5,
    min_new: int = 3,
) -> int:
    """Check all nodes and write/rewrite summaries where needed. Returns count of summaries written."""
    from thepaw_memory.memory_graph import get_graph
    from thepaw_memory.card_store import get_store

    graph = get_graph(persona_id)
    store = get_store(persona_id)

    candidates = graph.nodes_needing_summary(min_df=min_df, min_new=min_new)
    if not candidates:
        _log("no_candidates", f"persona={persona_id}")
        return 0

    written = 0
    for node_name, df, new_count in candidates:
        try:
            ok = await _write_one(persona_id, node_name, graph, store)
            if ok:
                written += 1
        except Exception as e:
            _log("write_error", f"node={node_name} error={e!s:.200}")

    _log("done", f"persona={persona_id} written={written}/{len(candidates)}")
    return written


async def _write_one(
    persona_id: str,
    node_name: str,
    graph,
    store,
) -> bool:
    """Write or rewrite summary for a single node."""
    from thepaw_memory.subbrain_client import call_safe

    # Find cards for this node
    rows = graph._conn.execute(
        "SELECT card_id FROM card_nodes WHERE node = ? ORDER BY rowid DESC LIMIT 50",
        (node_name,),
    ).fetchall()
    card_ids = [r[0] for r in rows]
    if not card_ids:
        return False

    # Get card contents — index the active set once, then look up by id
    # (the card_nodes ledger can point at superseded/legacy cards; skip those).
    active_by_id = {c["id"]: c for c in store.all_active()}
    cards = [active_by_id[cid] for cid in card_ids if cid in active_by_id]

    if not cards:
        return False

    cards_text = "\n---\n".join(
        f"[{c.get('source_time', '')}] {c.get('content', '')}"
        for c in cards
    )

    # Check if rewrite or new
    existing = graph.get_summary(node_name)

    from thepaw_memory.core.entity import get_persona_name
    persona_name = get_persona_name(persona_id) or persona_id

    if existing and existing.get("content"):
        prompt = SUMMARY_PROMPT_REWRITE.format(
            node=node_name,
            persona_name=persona_name,
            n=len(cards),
            old_summary=existing["content"],
            cards_text=cards_text,
        )
    else:
        prompt = SUMMARY_PROMPT_NEW.format(
            node=node_name,
            persona_name=persona_name,
            n=len(cards),
            cards_text=cards_text,
        )

    messages = [{"role": "user", "content": prompt}]
    result = await call_safe(messages, max_tokens=500, temperature=0.2, persona_id=persona_id)

    content = (result.content or "").strip()
    if not content:
        _log("empty_result", f"node={node_name}")
        return False

    if content.startswith("跳过") or content == "跳过":
        graph.save_summary(node_name, "", len(cards))
        _log("skipped_by_subbrain", f"node={node_name} cards={len(cards)}")
        return False

    graph.save_summary(node_name, content, len(cards))
    _log("summary_written", f"node={node_name} len={len(content)} cards={len(cards)}")
    return True
