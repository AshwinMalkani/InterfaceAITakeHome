from pathlib import Path

import pytest
from pydantic import ValidationError

from cua.profile import KnownState, MissingCredential, StateKind, load_profile

NOTICE_WHEN = {"kind": "text_present", "text": "SYSTEM NOTICE"}
DISMISS = {"strategies": [{"kind": "role", "role": "button", "name": "Acknowledge"}]}


def test_committed_cu_core_profile_loads() -> None:
    profile = load_profile("cu_core")
    assert {s.kind for s in profile.states} == set(StateKind)
    assert profile.sign_on is not None


def test_interstitial_requires_dismiss_control() -> None:
    with pytest.raises(ValidationError, match="dismiss"):
        KnownState.model_validate(
            {"name": "n", "kind": "interstitial", "description": "d", "when": NOTICE_WHEN}
        )


def test_only_interstitials_may_declare_dismiss() -> None:
    with pytest.raises(ValidationError, match="dismiss"):
        KnownState.model_validate(
            {"name": "n", "kind": "app_error", "description": "d", "when": NOTICE_WHEN, "dismiss": DISMISS}
        )


def test_missing_credentials_name_the_variable_not_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CU_CORE_PASSWORD", raising=False)
    monkeypatch.setenv("CU_CORE_USERNAME", "teller1")
    with pytest.raises(MissingCredential, match="CU_CORE_PASSWORD"):
        load_profile("cu_core").sign_on_params()


def test_profile_must_declare_its_own_product(tmp_path: Path) -> None:
    (tmp_path / "other.yaml").write_text("product: cu_core\n")
    with pytest.raises(ValueError, match="declares product"):
        load_profile("other", tmp_path)
