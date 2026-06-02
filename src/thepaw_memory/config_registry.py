"""Unified configuration registry.

Config params: config/registry.yml
Prompt defs:   config/prompts.yml   (same lookup path, separate file)
Per-persona overrides: memory/{persona_id}/config.yml

    from thepaw_memory.config_registry import cfg
    val = cfg("memory_engine.relation_judge_max_tokens")
"""
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_MISSING = object()
_registry: dict = {}
_persona_cache: dict[str, dict] = {}

from thepaw_memory.paths import ROOT, MEMORY_BASE


def _deep_merge(base: dict, overlay: dict) -> dict:
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_registry() -> dict:
    path = ROOT / "config" / "registry.yml"
    if not path.exists():
        log.warning("registry.yml not found: %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prompts_path = ROOT / "config" / "prompts.yml"
    if prompts_path.exists():
        with open(prompts_path, encoding="utf-8") as f:
            prompts = yaml.safe_load(f) or {}
        _deep_merge(data, prompts)
    return data


def _load_persona(pid: str) -> dict:
    if pid in _persona_cache:
        return _persona_cache[pid]
    path = MEMORY_BASE / pid / "config.yml"
    data: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    _persona_cache[pid] = data
    return data


def _walk(data: dict, parts: list[str]) -> Any:
    node = data
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return _MISSING
        node = node[p]
    return node


def cfg(key: str, persona_id: str | None = None) -> Any:
    """Read config value by dot-separated key.
    Lookup: persona override > registry default > KeyError.
    """
    parts = key.split(".")
    if persona_id:
        val = _walk(_load_persona(persona_id), parts)
        if val is not _MISSING:
            return val
    node = _registry
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"Config key not registered: {key}")
        node = node[p]
    if isinstance(node, dict) and "value" in node:
        if not persona_id and node.get("_scope") == "per_persona":
            raise ValueError(f"cfg('{key}') is per_persona — persona_id required")
        return node["value"]
    raise KeyError(f"Config key not registered: {key}")


def validate() -> list[str]:
    """Check all items have non-null values. Returns missing key list."""
    missing: list[str] = []
    _scan(_registry, [], missing)
    for k in missing:
        log.warning("Config: missing value for '%s'", k)
    return missing


def _scan(node: dict, path: list[str], out: list[str]):
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if k.startswith("_"):
            continue
        child = path + [k]
        if isinstance(v, dict) and "_type" in v:
            if v.get("value") is None:
                out.append(".".join(child))
        elif isinstance(v, dict):
            _scan(v, child, out)


def reload():
    """Re-read registry and clear persona cache."""
    global _registry
    _registry = _load_registry()
    _persona_cache.clear()


_registry = _load_registry()


_PROVIDER_URL_DEFAULTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}


def provider_base_url(provider: str) -> str:
    try:
        return cfg(f"provider_urls.{provider}")
    except KeyError:
        return _PROVIDER_URL_DEFAULTS.get(provider, "")
