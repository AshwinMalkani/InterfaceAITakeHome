import copy
from typing import Any

import pytest
from pydantic import ValidationError

from cua.artifact.schema import AllOf, Capability, Risk, Status


def minimal() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": "app.domain.action",
        "version": "1.0.0",
        "description": "d",
        "app": {"product": "app", "versions": ">=1"},
        "entry_route": "/start",
        "inputs": [{"name": "member_id", "description": "id", "sensitivity": "pii"}],
        "outputs": [{"name": "balance", "type": "decimal", "description": "b"}],
        "steps": [
            {
                "id": "s1",
                "intent": "fill",
                "action": {
                    "kind": "fill",
                    "target": {"strategies": [{"kind": "near_text", "anchor": "Id", "control": "textbox"}]},
                    "value": "{{inputs.member_id}}",
                },
                "risk": "safe",
            },
            {
                "id": "s2",
                "intent": "read",
                "action": {
                    "kind": "extract",
                    "target": {"strategies": [{"kind": "table_cell", "row": "Savings", "column": "Balance"}]},
                    "output": "balance",
                    "parse": "currency",
                },
                "risk": "safe",
            },
        ],
        "success": {"kind": "text_present", "text": "Done"},
        "provenance": {"source": "hand_written"},
    }


def build(**changes: Any) -> Capability:
    data = copy.deepcopy(minimal())
    data.update(changes)
    return Capability.model_validate(data)


def invalid(data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Capability.model_validate(data)


def test_minimal_is_valid() -> None:
    cap = build()
    assert cap.status is Status.DRAFT
    assert cap.max_risk is Risk.SAFE


class TestTemplates:
    def test_undeclared_input_rejected(self) -> None:
        data = minimal()
        data["steps"][0]["action"]["value"] = "{{inputs.other}}"
        invalid(data, "undeclared inputs")

    def test_unused_input_rejected(self) -> None:
        data = minimal()
        data["inputs"].append({"name": "spare", "description": "x"})
        invalid(data, "never used")

    @pytest.mark.parametrize("template", ["{{env.SECRET}}", "{{inputs.member_id | upper}}", "{{ 1 + 1 }}"])
    def test_only_plain_input_substitution_allowed(self, template: str) -> None:
        data = minimal()
        data["steps"][0]["action"]["value"] = template
        invalid(data, "unsupported template")

    def test_entry_route_can_be_templated(self) -> None:
        data = minimal()
        data["entry_route"] = "/member/{{inputs.member_id}}"
        data["steps"][0]["action"]["value"] = "constant"
        Capability.model_validate(data)


class TestOutputs:
    def test_extract_into_undeclared_output_rejected(self) -> None:
        data = minimal()
        data["steps"][1]["action"]["output"] = "other"
        invalid(data, "undeclared output")

    def test_declared_output_never_extracted_rejected(self) -> None:
        data = minimal()
        data["outputs"].append({"name": "name", "type": "string", "description": "n"})
        invalid(data, "never extracted")

    def test_parse_must_match_output_type(self) -> None:
        data = minimal()
        data["steps"][1]["action"]["parse"] = "text"
        invalid(data, "yields string")

    def test_output_extracted_twice_rejected(self) -> None:
        data = minimal()
        data["steps"].append({**data["steps"][1], "id": "s3"})
        invalid(data, "duplicate extracted output")


class TestStructure:
    def test_duplicate_step_ids_rejected(self) -> None:
        data = minimal()
        data["steps"][1]["id"] = "s1"
        invalid(data, "duplicate step id")

    def test_unknown_fields_rejected(self) -> None:
        invalid({**minimal(), "stpes": []}, "Extra inputs")

    def test_unknown_locator_kind_rejected(self) -> None:
        data = minimal()
        data["steps"][0]["action"]["target"]["strategies"] = [{"kind": "xpath", "expr": "//x"}]
        invalid(data, "does not match any of the expected tags")

    def test_target_needs_a_strategy(self) -> None:
        data = minimal()
        data["steps"][0]["action"]["target"]["strategies"] = []
        invalid(data, "at least 1")

    def test_read_only_actions_must_be_safe(self) -> None:
        data = minimal()
        data["steps"][1]["risk"] = "irreversible"
        invalid(data, "must be risk 'safe'")

    @pytest.mark.parametrize(
        ("field", "value"), [("id", "NoDots"), ("version", "1.0"), ("schema_version", "2.0")]
    )
    def test_identity_formats(self, field: str, value: str) -> None:
        invalid({**minimal(), field: value}, field)

    def test_nested_checkpoints(self) -> None:
        cap = build(success={"kind": "all_of", "conditions": [
            {"kind": "any_of", "conditions": [{"kind": "text_present", "text": "A"},
                                              {"kind": "url_matches", "pattern": "^/done"}]},
            {"kind": "element_visible", "target": {"frame": "main", "strategies": [
                {"kind": "role", "role": "button", "name": "OK"}]}},
        ]})
        assert isinstance(cap.success, AllOf)

    def test_max_risk_is_highest_step_risk(self) -> None:
        data = minimal()
        data["steps"][0]["risk"] = "irreversible"
        assert Capability.model_validate(data).max_risk is Risk.IRREVERSIBLE


class TestContentHash:
    def test_stable_and_ignores_status_and_provenance(self) -> None:
        base = build()
        assert base.content_hash() == build().content_hash()
        approved = build(status="approved", provenance={"source": "discovery", "model": "m"})
        assert approved.content_hash() == base.content_hash()

    def test_any_behavioural_change_changes_hash(self) -> None:
        data = minimal()
        data["steps"][0]["timeout_ms"] = 20_000
        assert Capability.model_validate(data).content_hash() != build().content_hash()


class TestOutcomes:
    OUTCOME = {"name": "member_not_found", "description": "d", "when": {"kind": "text_present", "text": "x"}}

    def test_v1_0_artifacts_still_load_without_outcomes(self) -> None:
        assert build().outcomes == []

    def test_outcome_after_unknown_step_rejected(self) -> None:
        invalid({**minimal(), "schema_version": "1.1", "outcomes": [{**self.OUTCOME, "after_step": "s9"}]},
                "unknown step")

    def test_duplicate_outcomes_rejected(self) -> None:
        outcomes = [self.OUTCOME, self.OUTCOME]
        invalid({**minimal(), "schema_version": "1.1", "outcomes": outcomes}, "duplicate outcome")

    def test_outcomes_are_part_of_the_content_hash(self) -> None:
        assert build(outcomes=[self.OUTCOME]).content_hash() != build().content_hash()

    def test_schema_upgrade_does_not_change_old_artifacts_hash(self) -> None:
        """A 1.0 artifact must serialize (and hash) identically after the 1.1 upgrade."""
        assert "outcomes" not in build().canonical_dict()
        assert build().content_hash() == build(outcomes=[]).content_hash()
