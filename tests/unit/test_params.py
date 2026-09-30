from decimal import Decimal
from typing import Any

import pytest

from cua.artifact.params import InputError, OutputParseError, parse_output, render, validate_inputs
from cua.artifact.schema import Capability, Parse
from tests.unit.test_schema import minimal


def cap_with_inputs(*inputs: dict[str, Any]) -> Capability:
    data = minimal()
    data["inputs"] = list(inputs)
    data["steps"][0]["action"]["value"] = " ".join(f"{{{{inputs.{i['name']}}}}}" for i in inputs)
    return Capability.model_validate(data)


MEMBER = {"name": "member_id", "description": "id", "pattern": r"\d{5}", "sensitivity": "pii"}


class TestValidateInputs:
    def test_valid(self) -> None:
        assert validate_inputs(cap_with_inputs(MEMBER), {"member_id": " 10001 "}) == {"member_id": "10001"}

    def test_missing_and_blank_are_required(self) -> None:
        cap = cap_with_inputs(MEMBER)
        for raw in ({}, {"member_id": ""}, {"member_id": None}):
            with pytest.raises(InputError, match="required"):
                validate_inputs(cap, raw)

    def test_unknown_input_rejected(self) -> None:
        with pytest.raises(InputError, match="not a declared input"):
            validate_inputs(cap_with_inputs(MEMBER), {"member_id": "10001", "extra": "x"})

    def test_pattern_error_never_echoes_the_value(self) -> None:
        with pytest.raises(InputError) as exc:
            validate_inputs(cap_with_inputs(MEMBER), {"member_id": "123-45-6789"})
        assert "123-45-6789" not in str(exc.value)
        assert exc.value.field == "member_id"

    @pytest.mark.parametrize(
        ("type_", "raw", "canonical"),
        [
            ("integer", 7, "7"),
            ("integer", "007", "7"),
            ("decimal", "10.50", "10.50"),
            ("boolean", "True", "true"),
        ],
    )
    def test_type_coercion(self, type_: str, raw: object, canonical: str) -> None:
        cap = cap_with_inputs({"name": "x", "description": "x", "type": type_})
        assert validate_inputs(cap, {"x": raw}) == {"x": canonical}

    @pytest.mark.parametrize(("type_", "raw"), [("integer", "1.5"), ("integer", True), ("decimal", "ten"),
                                                ("boolean", "yes")])
    def test_type_errors(self, type_: str, raw: object) -> None:
        cap = cap_with_inputs({"name": "x", "description": "x", "type": type_})
        with pytest.raises(InputError, match="must be"):
            validate_inputs(cap, {"x": raw})

    def test_choices(self) -> None:
        kinds = ["certificate", "money_market"]
        cap = cap_with_inputs({"name": "kind", "description": "k", "choices": kinds})
        assert validate_inputs(cap, {"kind": "certificate"}) == {"kind": "certificate"}
        with pytest.raises(InputError, match="one of"):
            validate_inputs(cap, {"kind": "crypto"})


def test_render_substitutes_with_or_without_spaces() -> None:
    assert render("/m/{{inputs.a}}/{{ inputs.b }}", {"a": "1", "b": "2"}) == "/m/1/2"


class TestParseOutput:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("$4,719.56", Decimal("4719.56")), ("  $0.00 ", Decimal("0.00")), ("($12.00)", Decimal("-12.00")),
         ("1204", Decimal("1204"))],
    )
    def test_currency(self, text: str, expected: Decimal) -> None:
        assert parse_output(text, Parse.CURRENCY) == expected

    def test_text_normalizes_whitespace(self) -> None:
        assert parse_output("  Avery \n  Testwood ", Parse.TEXT) == "Avery Testwood"

    def test_integer(self) -> None:
        assert parse_output("1,204", Parse.INTEGER) == 1204

    @pytest.mark.parametrize(("text", "parse"), [("N/A", Parse.CURRENCY), ("12.5", Parse.INTEGER)])
    def test_unparseable(self, text: str, parse: Parse) -> None:
        with pytest.raises(OutputParseError):
            parse_output(text, parse)
