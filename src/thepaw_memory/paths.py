"""Centralized path constants for the thepaw_memory library.

Two distinct roots:
  - ROOT       = the installed package directory (holds bundled config/).
  - data root  = the host's runtime data directory (cards, indexes, graphs,
                 logs). Set via ``THEPAW_DATA_DIR``, which MemoryEngine derives
                 from its ``base_dir`` argument.

MEMORY_BASE and LOG_DIR are resolved lazily (see ``_LazyDir``) so the engine's
``base_dir`` wins no matter when a core module first imports them. Defaults fall
back to ./data so the library is usable without configuration.
"""
from __future__ import annotations

import os
from pathlib import Path


class _LazyDir:
    """A directory whose location is resolved from an env var on every access.

    MemoryEngine sets ``THEPAW_MEMORY_BASE`` in its constructor. Because hosts
    may import core modules (which bind ``MEMORY_BASE``) before constructing the
    engine, the data root must not be frozen at import time — this proxy re-reads
    the env var each time, so the engine's ``base_dir`` always wins.
    """

    def __init__(self, env_key: str, default_factory):
        self._env_key = env_key
        self._default_factory = default_factory

    def _resolve(self) -> Path:
        v = os.environ.get(self._env_key)
        return Path(v) if v else self._default_factory()

    def __truediv__(self, other):
        return self._resolve() / other

    def __fspath__(self):
        return str(self._resolve())

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __str__(self):
        return str(self._resolve())

    def __repr__(self):
        return f"_LazyDir({self._env_key}={self._resolve()!r})"


# Package dir — bundled config lives at ROOT/config/*.yml
ROOT = Path(os.environ.get("THEPAW_ROOT", str(Path(__file__).resolve().parent)))

# Runtime data root — host-owned, set via MemoryEngine(base_dir=...).
# Re-read on every access so import order never freezes it to the wrong cwd.
def _data_root() -> Path:
    return Path(os.environ.get("THEPAW_DATA_DIR", str(Path.cwd() / "data")))


# Per-persona stores (cards.jsonl / cards.db / FAISS index / graph.db).
MEMORY_BASE = _LazyDir("THEPAW_MEMORY_BASE", lambda: _data_root() / "memory")
# JSONL feature log dir — lazy too, else `import thepaw_memory` would freeze it to cwd.
LOG_DIR = _LazyDir("THEPAW_LOG_DIR", lambda: _data_root() / "logs")

# Bundled config (the V2-tuned prompts + thresholds)
CFG_PATH = ROOT / "config" / "prompt_budget.yml"
PROFILES_DIR = Path(os.environ.get("THEPAW_PROFILES_DIR", str(ROOT / "config" / "profiles")))
