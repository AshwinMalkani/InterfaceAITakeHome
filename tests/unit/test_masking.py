from decimal import Decimal

import pytest
from pydantic import BaseModel

from cua.security.masking import (
    MASKING_ERROR,
    SecretRegistry,
    Sensitivity,
    mask_secrets,
    mask_text,
    safe_mask,
    use_registry,
)


@pytest.fixture
def reg() -> SecretRegistry:
    r = SecretRegistry(include_env=False, key=b"k" * 32)
    r.register("member_id", "48213", Sensitivity.PII)
    r.register("member_name", "Jane Q Testperson", Sensitivity.PII)
    r.register("login_password", "hunter2hunter2", Sensitivity.SECRET)
    return r


class TestKnownValues:
    def test_pii_value_is_replaced_with_correlatable_token(self, reg: SecretRegistry) -> None:
        out = mask_text("searching member 48213 then member 48213", reg)
        assert "48213" not in out
        tokens = [t for t in out.split() if t.startswith("[PII:member_id#")]
        assert len(tokens) == 2 and tokens[0] == tokens[1]

    def test_match_is_case_insensitive(self, reg: SecretRegistry) -> None:
        assert "JANE" not in mask_text("Member: JANE Q TESTPERSON", reg)

    def test_secret_token_has_no_correlation_suffix(self, reg: SecretRegistry) -> None:
        assert mask_text("pw=hunter2hunter2", reg) == "pw=[SECRET:login_password]"

    def test_public_and_short_values_are_not_registered(self) -> None:
        r = SecretRegistry(include_env=False)
        r.register("account_type", "Savings", Sensitivity.PUBLIC)
        r.register("flag", "No", Sensitivity.PII)
        assert mask_text("Savings No", r) == "Savings No"

    def test_tokens_differ_across_registries(self) -> None:
        a, b = SecretRegistry(include_env=False), SecretRegistry(include_env=False)
        for r in (a, b):
            r.register("member_id", "48213", Sensitivity.PII)
        assert mask_text("48213", a) != mask_text("48213", b)

    def test_env_secrets_registered_automatically(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "my-very-secret-value")
        assert mask_text("key my-very-secret-value", SecretRegistry()) == "key [SECRET:anthropic_api_key]"


class TestPatterns:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ssn 123-45-6789", "ssn [SSN]"),
            ("card 4111 1111 1111 1111", "card [CARD:***1111]"),
            ("acct 000123456789", "acct [ACCT:***6789]"),
            ("mail jane@example.com", "mail [EMAIL]"),
            ("call (555) 123-4567", "call [PHONE]"),
            ("key sk-ant-api03-abcdefghijk", "key [SECRET:api_key]"),
            ("Authorization: Bearer abc.def.ghi", "Authorization: [SECRET:bearer_token]"),
        ],
    )
    def test_detectors(self, raw: str, expected: str) -> None:
        assert mask_text(raw, SecretRegistry(include_env=False)) == expected

    def test_non_luhn_16_digits_is_treated_as_account_not_card(self) -> None:
        assert mask_text("1234567812345678", SecretRegistry(include_env=False)) == "[ACCT:***5678]"

    def test_ordinary_text_and_small_numbers_untouched(self) -> None:
        text = "Step 3 took 250 ms; balance $1,204.55 on 2026-09-30"
        assert mask_text(text, SecretRegistry(include_env=False)) == text


class TestStructures:
    def test_sensitive_keys_redacted_regardless_of_value(self, reg: SecretRegistry) -> None:
        out = mask_secrets({"password": "x", "Set-Cookie": "sid=1", "api_key": 7}, reg)
        assert out == {
            "password": "[REDACTED:password]",
            "Set-Cookie": "[REDACTED:Set-Cookie]",
            "api_key": "[REDACTED:api_key]",
        }

    def test_token_counts_are_not_credentials(self, reg: SecretRegistry) -> None:
        out = mask_secrets({"access_token": "abc", "authToken": "def", "input_tokens": 1200,
                            "cache_read_tokens": 900}, reg)
        assert out == {"access_token": "[REDACTED:access_token]", "authToken": "[REDACTED:authToken]",
                       "input_tokens": 1200, "cache_read_tokens": 900}

    def test_nested_dicts_lists_and_models(self, reg: SecretRegistry) -> None:
        class Result(BaseModel):
            outputs: dict[str, str]

        data = {"results": [Result(outputs={"name": "Jane Q Testperson"})], "id": ("48213",)}
        out = mask_secrets(data, reg)
        assert "Jane" not in str(out) and "48213" not in str(out)

    def test_input_not_mutated(self, reg: SecretRegistry) -> None:
        data = {"note": "member 48213"}
        mask_secrets(data, reg)
        assert data == {"note": "member 48213"}

    def test_sensitive_numbers_masked_innocent_numbers_kept(self, reg: SecretRegistry) -> None:
        out = mask_secrets({"acct": 123456789012, "member": 48213, "ms": 250, "ok": True}, reg)
        assert out["acct"] == "[ACCT:***9012]"
        assert out["member"].startswith("[PII:member_id#")
        assert out["ms"] == 250 and out["ok"] is True

    def test_other_types_are_stringified(self, reg: SecretRegistry) -> None:
        assert mask_secrets(Decimal("12.50"), reg) == "12.50"


def test_use_registry_sets_implicit_registry(reg: SecretRegistry) -> None:
    with use_registry(reg):
        assert "48213" not in mask_text("member 48213")
    assert mask_text("member 48213") == "member 48213"


def test_safe_mask_fails_closed() -> None:
    class Exploding:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert safe_mask(Exploding()) == MASKING_ERROR
