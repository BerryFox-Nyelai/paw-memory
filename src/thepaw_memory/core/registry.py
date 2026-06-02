"""EntityRegistry — in-memory registry of all system participants.

Loads Persona entities from config/profiles/*.yml + memory/*/persona_meta.json
at startup.  Human entities are registered on first auth.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from thepaw_memory.core.entity import Entity, Identity, Persona, SubbrainRules

from thepaw_memory.paths import PROFILES_DIR as _PROFILES_DIR, MEMORY_BASE as _MEMORY_BASE

_PROVIDER_PROFILE_KEYS = {"provider", "model", "base_url", "api_key_file", "timeout_s"}


class EntityRegistry:
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._profiles: Dict[str, dict] = {}

    # ---- public API ----

    def register(self, entity: Entity) -> None:
        self._entities[entity.id] = entity

    def get(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def list(self, kind: Optional[str] = None) -> List[Entity]:
        if kind is None:
            return list(self._entities.values())
        return [e for e in self._entities.values() if e.kind == kind]

    def get_profile(self, profile_id: str) -> Optional[dict]:
        return self._profiles.get(profile_id)

    # ---- loading ----

    def load(self) -> None:
        self._load_profiles()
        self._load_personas()

    def _load_profiles(self) -> None:
        if not _PROFILES_DIR.is_dir():
            return
        for yml in sorted(_PROFILES_DIR.glob("*.yml")):
            try:
                data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not data.get("provider"):
                continue
            pid = data.get("name", yml.stem)
            self._profiles[pid] = data

    def _load_personas(self) -> None:
        if not _MEMORY_BASE.is_dir():
            return
        for d in sorted(_MEMORY_BASE.iterdir()):
            if not d.is_dir():
                continue
            pid = d.name
            if pid.startswith("_") or pid in ("exports", "summaries", "mem0", "episodic"):
                continue

            meta = self._read_persona_meta(d)
            reflection_cfg = self._read_reflection_config(d)

            display_name = meta.get("display_name", pid)
            user_display_name = meta.get("user_display_name", "")

            user_ids: Dict[str, Identity] = {}
            if user_display_name:
                from thepaw_memory.config_registry import cfg as _cfg
                try:
                    uid = meta.get("user_id") or _cfg("auth.default_user_id")
                except KeyError:
                    uid = meta.get("user_id", "default")
                user_ids[uid] = Identity(display_name=user_display_name)

            persona_provider = meta.get("provider", "")
            persona_model = meta.get("model", "")
            if not persona_provider:
                profile_override = meta.get("profile")
                if profile_override:
                    profile_name = profile_override if isinstance(profile_override, str) else profile_override[0]
                    profile_data = self._profiles.get(profile_name, {})
                    persona_provider = profile_data.get("provider", "local")
                    persona_model = persona_model or profile_data.get("model", "")
                else:
                    persona_provider = "local"

            persona = Persona(
                id=pid,
                kind="persona",
                identity=Identity(display_name=display_name),
                memory_partition=pid,
                provider=persona_provider,
                model=persona_model,
                system_prompt_template=meta.get("system_prompt_template", ""),
                subbrain_rules=SubbrainRules(
                    inner_active=reflection_cfg.get("inner_active", False),
                    proactive=reflection_cfg.get("proactive", False),
                    visibility_default=reflection_cfg.get("visibility_default", "visible"),
                    social_active=reflection_cfg.get("social_active", False),
                ),
                user_identities=user_ids,
            )
            self.register(persona)

    @staticmethod
    def _read_persona_meta(persona_dir: Path) -> dict:
        p = persona_dir / "persona_meta.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    @staticmethod
    def _read_reflection_config(persona_dir: Path) -> dict:
        p = persona_dir / "reflection" / "config.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}


entity_registry = EntityRegistry()
