"""Unified feature logging (memory buffer + persistent JSONL) and trace context."""
from __future__ import annotations

import contextvars
import json
import logging
import secrets
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from thepaw_memory.paths import LOG_DIR

# --- Trace context (async-safe via contextvars) ---

_current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar('trace_id', default='')
_current_parent_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar('parent_trace_id', default='')
_current_persona_id: contextvars.ContextVar[str] = contextvars.ContextVar('persona_id', default='')


def new_trace_id(prefix: str = "t") -> str:
    tid = f"{prefix}_{secrets.token_hex(4)}"
    _current_trace_id.set(tid)
    return tid


def set_trace_context(*, trace_id: str = "", parent_trace_id: str = "", persona_id: str = "") -> None:
    if trace_id:
        _current_trace_id.set(trace_id)
    if parent_trace_id:
        _current_parent_trace_id.set(parent_trace_id)
    if persona_id:
        _current_persona_id.set(persona_id)


def get_trace_id() -> str:
    return _current_trace_id.get()


# --- Persistent JSONL log (lazy: no filesystem side-effects at import) ---

_file_logger: Optional[logging.Logger] = None


def _get_file_logger() -> logging.Logger:
    global _file_logger
    if _file_logger is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("thepaw_memory.jsonl")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fh = RotatingFileHandler(
            str(LOG_DIR / "bff.jsonl"), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        _file_logger = logger
    return _file_logger


# --- Memory buffer ---

_feature_log_buffer: List[Dict[str, Any]] = []
_FEATURE_LOG_MAX = 200


def _write_jsonl(entry: Dict[str, Any]) -> None:
    try:
        _get_file_logger().info(json.dumps(entry, ensure_ascii=False))
    except Exception:
        pass


def feature_log(module: str, event: str, detail: str = "", *, data: Any = None) -> None:
    entry: Dict[str, Any] = {"ts": time.time(), "module": module, "event": event, "detail": detail}
    tid = _current_trace_id.get()
    if tid:
        entry["trace_id"] = tid
    ptid = _current_parent_trace_id.get()
    if ptid:
        entry["parent_trace_id"] = ptid
    pid = _current_persona_id.get()
    if pid:
        entry["persona_id"] = pid
    if data is not None:
        entry["data"] = data
    _feature_log_buffer.append(entry)
    if len(_feature_log_buffer) > _FEATURE_LOG_MAX:
        _feature_log_buffer.pop(0)
    _write_jsonl(entry)
    try:
        from thepaw_memory.workshop_store import insert
        insert(entry)
    except Exception:
        pass


def get_feature_log(module: str = "") -> List[Dict[str, Any]]:
    if module:
        return [e for e in _feature_log_buffer if e.get("module") == module]
    return list(_feature_log_buffer)


def tail_log(n: int = 50, module: str = "", level: str = "") -> List[Dict[str, Any]]:
    if not _LOG_FILE.exists():
        return []
    try:
        lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-max(n * 3, 200):]:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if module and e.get("module") != module:
                    continue
                if level and e.get("level", "") != level:
                    continue
                entries.append(e)
            except Exception:
                continue
        return entries[-n:]
    except Exception:
        return []
