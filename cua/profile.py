"""App profiles: per-vendor-product configuration shared by all its capabilities and tenants."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cua.artifact.schema import IDENTIFIER, Checkpoint, Target

PROFILES_DIR = Path(__file__).resolve().parents[1] / "config" / "apps"


class MissingCredential(RuntimeError):
    pass


class SignOn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    credentials: dict[str, str]  # sign-on capability input name -> environment variable


class StateKind(StrEnum):
    """How replay responds when a known app state appears. Written once per vendor product."""

    INTERSTITIAL = "interstitial"        # recoverable: click its (safe) dismiss control, carry on
    SESSION_EXPIRED = "session_expired"  # recoverable only if nothing irreversible ran: sign on, restart
    APP_ERROR = "app_error"              # hard failure (retryable if nothing irreversible ran)
    BUSINESS = "business"                # a business outcome shared by every capability (e.g. access denied)


class KnownState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=IDENTIFIER)
    kind: StateKind
    description: str
    when: Checkpoint
    dismiss: Target | None = None  # interstitials only: the acknowledge control (must be a safe click)

    @model_validator(mode="after")
    def _dismiss_only_for_interstitials(self) -> KnownState:
        if (self.kind is StateKind.INTERSTITIAL) != (self.dismiss is not None):
            raise ValueError(f"state {self.name!r}: `dismiss` is required for interstitials, and only them")
        return self


class AppProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(pattern=IDENTIFIER)
    sign_on: SignOn | None = None
    states: list[KnownState] = []
    # UI labels and headings of the product that are safe to show in redacted screenshots.
    ui_vocabulary: list[str] = []

    def sign_on_params(self) -> dict[str, str]:
        """Read credentials from the environment. Error messages name the variable, never a value."""
        if self.sign_on is None:
            return {}
        params: dict[str, str] = {}
        for input_name, env_var in self.sign_on.credentials.items():
            value = os.environ.get(env_var)
            if not value:
                raise MissingCredential(f"environment variable {env_var} is not set")
            params[input_name] = value
        return params


def load_profile(product: str, directory: Path = PROFILES_DIR) -> AppProfile:
    path = directory / f"{product}.yaml"
    profile = AppProfile.model_validate(yaml.safe_load(path.read_text()))
    if profile.product != product:
        raise ValueError(f"{path} declares product {profile.product!r}")
    return profile
