"""Parsers for memory harvest prompt outputs (V2: combined extraction)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from thepaw_memory.utils import extract_fields, strip_think


_RETENTION_MAP = {
    "core": -1.0,
    "lasting": 60.0,
    "passing": 8.0,
    "fleeting": 1.0,
}


def _parse_retention(raw_val: str) -> float:
    """Map retention label (core/lasting/passing/fleeting) to numeric months.
    Falls back to 8.0 (passing) for unrecognized values."""
    val = raw_val.strip().lower()
    for label, num in _RETENTION_MAP.items():
        if label in val:
            return num
    try:
        return float(val)
    except (ValueError, TypeError):
        return 8.0


def parse_combined_harvest(raw: str) -> Dict[str, Any]:
    """Parse combined main-brain harvest: 回忆 + 纠正 blocks."""
    raw = strip_think(raw)

    result: Dict[str, Any] = {"events": [], "corrections": []}

    if ("无回忆" in raw or "无事件" in raw) and "===" not in raw:
        return result

    result["events"] = _parse_memory_blocks(raw)
    result["corrections"] = _parse_corrections(raw)
    return result


def _parse_memory_blocks(raw: str) -> List[Dict[str, Any]]:
    """Parse ===回忆 N=== blocks with all fields."""
    # Bound at 纠正/事实 so a trailing correction or stray legacy 事实 block
    # can't bleed into the last 回忆's fields. 事实 is deprecated, but the weak
    # main brain may still emit one out of habit — kept as a defensive boundary.
    cut = raw
    for sep in (r"===\s*纠正\s*\d+\s*===", r"===\s*事实\s*\d+\s*==="):
        cut = re.split(sep, cut)[0]

    blocks = re.split(r"===\s*回忆\s*\d+\s*===", cut)

    field_map = {
        "内容": "record",
        "主体": "subject",
        "话题": "topic",
        "留存度": "retention_raw",
        "情绪强度": "retention_raw",
        "重要性": "_significance_compat",
    }

    events: List[Dict[str, Any]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event: Dict[str, Any] = dict(extract_fields(block, field_map))

        if "retention_raw" in event:
            event["retention"] = _parse_retention(event.pop("retention_raw"))
        elif "_significance_compat" in event:
            # Backward compat: old prompt with 情绪强度+重要性 (0-1 floats)
            try:
                s = float(event.pop("_significance_compat"))
            except (ValueError, TypeError):
                s = 0.5
            if s >= 0.8:
                event["retention"] = 60.0
            elif s >= 0.4:
                event["retention"] = 8.0
            else:
                event["retention"] = 1.0
        else:
            event["retention"] = 8.0
        event.pop("_significance_compat", None)

        if event.get("record"):
            events.append(event)

    return events


def _parse_corrections(raw: str) -> List[Dict[str, Any]]:
    """Parse ===纠正 N=== text output into list of correction dicts."""
    if "无纠正" in raw and "===" not in raw:
        return []

    # Stop before any stray legacy 事实 block (deprecated) so it can't bleed
    # into the last correction's fields.
    cut = re.split(r"===\s*事实\s*\d+\s*===", raw)[0]
    blocks = re.split(r"===\s*纠正\s*\d+\s*===", cut)
    field_map = {
        "原信息": "original",
        "U 的说法": "user_said",
        "U的说法": "user_said",
        "她的说法": "user_said",
        "性质": "nature",
        "涉及楼层": "seq_ref",
    }
    corrections = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        item = extract_fields(block, field_map)
        if item.get("original") or item.get("user_said"):
            corrections.append(item)
    return corrections
