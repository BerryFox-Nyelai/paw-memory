"""Entity — the universal participant model.

Every participant in the system (human, persona, subbrain, device, service)
is an Entity.  Persona is a specialization with provider/memory bindings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Identity:
    display_name: str = ""
    avatar: str = ""
    greeting: str = ""


@dataclass
class SubbrainRules:
    inner_active: bool = False
    proactive: bool = False
    visibility_default: str = "visible"
    # 是否参与角色之间的社交：打开后才会出现在别人的好友候选名单里。
    social_active: bool = False


@dataclass
class Entity:
    id: str
    kind: str
    identity: Identity = field(default_factory=Identity)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class Persona(Entity):
    provider: str = "local"
    model: str = ""
    memory_partition: str = ""
    system_prompt_template: str = ""
    subbrain_rules: SubbrainRules = field(default_factory=SubbrainRules)
    user_identities: Dict[str, Identity] = field(default_factory=dict)

    def __post_init__(self):
        self.kind = "persona"
        if not self.memory_partition:
            self.memory_partition = self.id


def get_persona_name(persona_id: str) -> str:
    from thepaw_memory.core.registry import entity_registry
    entity_registry.load()
    persona = entity_registry.get(persona_id)
    if isinstance(persona, Persona) and persona.identity.display_name:
        return persona.identity.display_name
    return persona_id


def get_user_name(persona_id: str) -> str:
    from thepaw_memory.core.registry import entity_registry
    entity_registry.load()
    persona = entity_registry.get(persona_id)
    if isinstance(persona, Persona) and persona.user_identities:
        first = next(iter(persona.user_identities.values()), None)
        if first and first.display_name:
            return first.display_name
    return "用户"
