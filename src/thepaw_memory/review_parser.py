"""Parsers for memory review prompt outputs (text separator format)."""
from __future__ import annotations

import re
from typing import Any, Dict

from thepaw_memory.utils import extract_fields, strip_think


def _is_empty(val: str) -> bool:
    return val in ("无", "无。", "—", "-", "")


def _is_yes(val: str) -> bool:
    return val.startswith("是") or val.lower().startswith("yes")


# ---------------------------------------------------------------------------
# Step 11: Staging pool significance
# ---------------------------------------------------------------------------

def parse_staging_judge(raw: str) -> Dict[str, Any]:
    """Parse ===暂存池判断=== into staging significance dict."""
    raw = strip_think(raw)
    if not raw:
        return {"should_rewrite": False}

    parts = re.split(r"===\s*暂存池判断\s*===", raw)
    block = parts[-1] if len(parts) > 1 else parts[0]

    fields = extract_fields(block, {
        "需要吸收": "should_rewrite",
        "理由": "reasoning",
    })

    return {
        "should_rewrite": _is_yes(fields.get("should_rewrite", "否")),
        "reasoning": fields.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Step 11: Stable portrait review
# ---------------------------------------------------------------------------

def parse_stable_review(raw: str) -> Dict[str, Any]:
    """Parse ===长期画像审查=== into stable review dict."""
    raw = strip_think(raw)
    if not raw:
        return {"should_update": False}

    parts = re.split(r"===\s*长期画像审查\s*===", raw)
    block = parts[-1] if len(parts) > 1 else parts[0]

    fields = extract_fields(block, {
        "需要更新": "should_update",
        "新画像": "new_stable",
        "理由": "reasoning",
    })

    should_update = _is_yes(fields.get("should_update", "否"))
    new_stable = fields.get("new_stable", "")
    if _is_empty(new_stable):
        new_stable = ""

    return {
        "should_update": should_update,
        "new_stable": new_stable,
        "reasoning": fields.get("reasoning", ""),
    }
