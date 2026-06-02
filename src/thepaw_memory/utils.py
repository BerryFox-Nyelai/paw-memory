"""Shared utility functions — token estimation, ID generation, text helpers, config."""
from __future__ import annotations

import json
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from thepaw_memory.paths import CFG_PATH, MEMORY_BASE


# --- Token estimation ---

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.encode("utf-8")) // 3)


def clamp_content(content: str, max_tokens: int) -> str:
    if not content or max_tokens <= 0:
        return ""
    max_bytes = max_tokens * 3
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip() + "……[截断]"


def server_now() -> float:
    return time.time()


# --- Budget config ---

def load_prompt_budget() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        return {
            "model_ctx_total": 8192,
            "out_max": 1024,
            "reserve": 384,
            "service_prompt_soft_max": 8000,
            "system_base_max": 1200,
            "system_runtime_max": 200,
            "memory_total_max": 1200,
            "summary_max": 650,
            "recent_min_keep_msgs": 4,
            "recent_min_keep_tokens": 600,
            "estimator": "utf8_div3",
        }
    raw = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
    return raw


def module_cfg(module_prefix: str, key: str, default: str = "") -> str:
    try:
        from thepaw_memory.config_registry import cfg
        return str(cfg(f"{module_prefix}.{key}"))
    except (KeyError, ValueError, ImportError):
        pass
    try:
        return str(load_prompt_budget().get(f"{module_prefix}_{key}", default)).strip()
    except Exception:
        return default


def is_feature_enabled(key: str, default: str = "1") -> bool:
    try:
        from thepaw_memory.config_registry import cfg
        v = cfg(f"features.{key}")
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() not in ("0", "false")
    except (KeyError, ValueError, ImportError):
        pass
    try:
        raw = load_prompt_budget()
        return str(raw.get(key, default)).strip() not in ("0", "false")
    except Exception:
        return True


# --- Text helpers ---

def _safe_session_id(sid: str) -> str:
    sid = (sid or "default").strip()
    sid = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sid)
    return sid or "default"


def safe_filename(name: str, default: str = "default") -> str:
    name = (name or default).strip()
    name = re.sub(r"[^\w.\-一-鿿㐀-䶿]+", "_", name)
    name = name.replace("..", "_").strip("_. ")
    return name or default


def read_key_file(key_file: str) -> str:
    try:
        return Path(key_file).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def fuzzy_similar(a: str, b: str, threshold: float = 0.5) -> bool:
    def bigrams(s):
        s = s.lower()
        return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else {s}
    sa, sb = bigrams(a), bigrams(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= threshold


# --- LLM output parsing ---

def strip_think(raw: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def extract_fields(block: str, field_map: Dict[str, str]) -> Dict[str, str]:
    """Parse ``中文key: value`` lines from a text block into a dict.

    Multi-line values accumulate until the next recognized key. Accepts both
    ASCII ``:`` and full-width ``：`` separators.
    """
    result: Dict[str, str] = {}
    current_key: str | None = None
    current_lines: List[str] = []
    for line in block.split("\n"):
        stripped = line.strip()
        matched = False
        for cn_key, en_key in field_map.items():
            if stripped.startswith(cn_key + ":") or stripped.startswith(cn_key + "："):
                if current_key:
                    result[current_key] = "\n".join(current_lines).strip()
                sep = ":" if ":" in stripped.split(cn_key, 1)[1][:2] else "："
                current_key = en_key
                current_lines = [stripped.split(sep, 1)[-1].strip()]
                matched = True
                break
        if not matched and current_key and stripped:
            current_lines.append(stripped)
    if current_key:
        result[current_key] = "\n".join(current_lines).strip()
    return result


# --- ID and time helpers ---

_ID_CHARS = string.ascii_lowercase + string.digits


def gen_id(prefix: str = "", length: int = 4) -> str:
    import secrets
    body = "".join(secrets.choice(_ID_CHARS) for _ in range(length))
    return f"{prefix}{body}" if prefix else body


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --- JSONL storage helpers ---

def persona_jsonl_path(persona_id: str, filename: str) -> Path:
    pid = safe_filename(persona_id)
    return MEMORY_BASE / pid / filename


def load_jsonl(filepath: Path, exclude_status: str = "") -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        if not filepath.exists():
            return items
        for line in filepath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
    except Exception:
        pass
    return items


def save_jsonl(filepath: Path, items: List[Dict[str, Any]]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    filepath.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def append_jsonl(filepath: Path, item: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        pass
