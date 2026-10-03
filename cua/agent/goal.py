"""A discovery goal: the capability's *contract*, written by a person; the *flow*, found by the model.

The contract (id, typed inputs and outputs, sensitivity) is a design decision about what the
calling agent gets, so it isn't left to the LLM. The model is told input *names* only and types
`{{inputs.x}}` placeholders; real values are substituted by the executor at run time.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cua.artifact.schema import CAPABILITY_ID, SEMVER, AppRef, OutputSpec, ParamSpec


class GoalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=CAPABILITY_ID)
    version: str = Field(default="1.0.0", pattern=SEMVER)
    description: str
    goal: str  # natural language; may reference inputs as {{inputs.name}}
    app: AppRef
    entry_route: str
    inputs: list[ParamSpec] = []
    outputs: list[OutputSpec] = []


def load_goal(path: Path) -> GoalSpec:
    return GoalSpec.model_validate(yaml.safe_load(path.read_text()))
