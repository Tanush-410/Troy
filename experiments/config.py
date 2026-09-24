"""Episode configuration and its hash."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from agents.providers.base import ProviderConfig
from policy.log_schema import ControlCondition, DriftCondition
from policy.permissions import Role
from simmart.generator import GeneratorConfig


class EpisodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    episode_id: str
    seed: int
    model: str
    agent_role: Role
    drift_condition: DriftCondition
    control_condition: ControlCondition
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    provider: ProviderConfig | None = None  # None for the scripted agent
    max_turns_per_task: int = 20

    def config_hash(self) -> str:
        """sha256 of everything that determines the episode except its ids."""
        payload = self.model_dump(mode="json", exclude={"run_id", "episode_id"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
