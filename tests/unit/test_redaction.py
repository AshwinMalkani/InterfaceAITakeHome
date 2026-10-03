from pathlib import Path

import pytest

from cua.artifact.store import CapabilityLibrary
from cua.policy import ApprovalLedger, PolicyViolation, load_policy
from cua.profile import load_profile
from cua.replay.redaction import capability_vocabulary, normalize_label, profile_vocabulary
from cua.runner import replay

LIBRARY = CapabilityLibrary(Path(__file__).resolve().parents[2] / "capabilities")


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [("Member Number:", "member number"), ("  Share\n  Savings ", "share savings"), ("Tax ID :", "tax id")],
)
def test_normalize_label(raw: str, normalized: str) -> None:
    assert normalize_label(raw) == normalized


def test_capability_vocabulary_is_labels_never_values() -> None:
    vocabulary = capability_vocabulary(LIBRARY.get("cu_core.member.get_share_balance"))
    assert {
        "member number",
        "search",
        "member detail",
        "share savings",
        "balance",
        "name",
        "no records found.",
    } <= vocabulary
    assert not any("{{" in word or "inputs." in word for word in vocabulary)


def test_profile_vocabulary_includes_ui_labels_and_known_states() -> None:
    vocabulary = profile_vocabulary(load_profile("cu_core"))
    assert {"tax id", "phone", "system notice", "acknowledge", "your session has expired"} <= vocabulary


def test_unredacted_screenshots_need_a_policy_that_allows_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CU_CORE_USERNAME", "teller1")
    monkeypatch.setenv("CU_CORE_PASSWORD", "x-password")
    with pytest.raises(PolicyViolation, match="unredacted"):
        replay(
            "cu_core.member.get_share_balance",
            {"member_id": "10001"},
            base_url="http://127.0.0.1:1",
            library=LIBRARY,
            profile=load_profile("cu_core"),
            policy=load_policy("cu_core"),
            approvals=ApprovalLedger(tmp_path / "a.json"),
            evidence_root=tmp_path,
            unredacted_screenshots=True,
        )


def test_redact_observation_keeps_vocabulary_and_masks_values() -> None:
    from cua.replay.redaction import redact_observation

    screen = "\n".join(
        [
            "[frame main  /core/member/detail]",
            '  e12   cell "Phone:"',
            '  e13   cell "555-0117"',
            "  e14   textbox (next to 'Member Number')",
            "  e15   select options=['Savings', 'Acct 12345678']",
        ]
    )
    out = redact_observation(screen, frozenset({"phone", "savings"}))
    assert '"Phone:"' in out and "555-0117" not in out
    assert "(next to 'Member Number')" in out  # structure stays readable
    assert "'Savings'" in out and "12345678" not in out
