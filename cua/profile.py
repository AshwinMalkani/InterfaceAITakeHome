"""App profiles: per-vendor-product configuration shared by all its capabilities and tenants."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cua.artifact.schema import IDENTIFIER

PROFILES_DIR = Path(__file__).resolve().parents[1] / "config" / "apps"


class MissingCredential(RuntimeError):
    pass


class SignOn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    credentials: dict[str, str]  # sign-on capability input name -> environment variable


class AppProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(pattern=IDENTIFIER)
    sign_on: SignOn | None = None

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
