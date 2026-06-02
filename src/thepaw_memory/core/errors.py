import logging
import threading
import time

_logger = logging.getLogger("thepaw_memory.errors")


class ErrorCounter:
    """Simple thread-safe error counter with per-minute windowing."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total = 0
        self._window: list[float] = []
        self._window_sec = 300

    def inc(self):
        now = time.time()
        with self._lock:
            self._total += 1
            self._window.append(now)

    def snapshot(self) -> dict:
        now = time.time()
        cutoff = now - self._window_sec
        with self._lock:
            self._window = [t for t in self._window if t > cutoff]
            return {"total": self._total, "last_5min": len(self._window)}


error_counter = ErrorCounter()


def report_error(
    severity: str,
    context: str,
    error: Exception,
    detail: str = "",
):
    msg = f"[{severity.upper()}] {context}: {error}"
    if detail:
        msg += f" | {detail}"
    if severity == "critical":
        _logger.error(msg)
    else:
        _logger.warning(msg)
    error_counter.inc()
    try:
        from thepaw_memory.feature_log import _write_jsonl
        _write_jsonl({
            "ts": time.time(), "level": severity, "module": "error",
            "event": context, "detail": f"{error}" + (f" | {detail}" if detail else ""),
        })
    except Exception:
        pass
