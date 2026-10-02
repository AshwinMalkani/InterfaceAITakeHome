"""Locator strategies and dialog handling in a real (headless) browser, on legacy-style markup."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, sync_playwright
from pydantic import TypeAdapter

from cua.artifact.schema import Checkpoint, DialogExpectation, Target
from cua.surface.base import TargetNotFound
from cua.surface.web import WebSurface, xpath_literal

LEGACY_PAGE = """
<table><tr><td>
  <!-- layout table wrapping a form table: outer rows must not match -->
  <table>
    <tr><td>Member Number:</td><td><input type="text" name="f1"></td></tr>
    <tr><td>Password</td><td><input type="password" name="pw"></td></tr>
    <tr><td>Type:</td><td>
      <select name="f2"><option>Savings</option><option>Money Market</option></select>
    </td></tr>
    <tr><td>Name:</td><td><b>Avery Testwood</b></td></tr>
    <tr><td></td><td>
      <input type="submit" value="Search">
      <input type="button" value="Search" style="display:none">
    </td></tr>
  </table>
</td></tr></table>
<table class="grid">
  <tr><td>Description</td><td>Available</td><td>Balance</td></tr>
  <tr><td>Share Savings</td><td>$10.00</td><td>$4,719.56</td></tr>
  <tr><td>Checking</td><td>$1.00</td><td>$2.00</td></tr>
</table>
<a href="#x">Details</a> <a href="#y">Details</a>
<input type="button" value="Delete" onclick="confirm('Delete this record? This cannot be undone.')">
"""


_CHECKPOINT: TypeAdapter[Checkpoint] = TypeAdapter(Checkpoint)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def browser_page(browser: Browser) -> Iterator[Page]:
    page = browser.new_page()  # fresh page per test: each WebSurface adds its own dialog handler
    yield page
    page.close()


@pytest.fixture
def surface(browser_page: Page) -> WebSurface:
    browser_page.set_content(LEGACY_PAGE)
    return WebSurface(browser_page, "http://unused")


def target(*strategies: dict[str, Any], frame: str | None = None) -> Target:
    return Target.model_validate({"frame": frame, "strategies": list(strategies)})


def checkpoint(**data: Any) -> Checkpoint:
    return _CHECKPOINT.validate_python(data)


def test_xpath_literal_handles_both_quote_kinds() -> None:
    assert xpath_literal("plain") == "'plain'"
    assert xpath_literal("it's") == '"it\'s"'
    assert xpath_literal("""a'b"c""") == """concat('a', "'", 'b"c')"""


class TestStrategies:
    def test_near_text_finds_unlabelled_field_through_nested_tables(self, surface: WebSurface) -> None:
        resolved = surface.resolve(
            target({"kind": "near_text", "anchor": "Member Number", "control": "textbox"}), 1000
        )
        surface.fill(resolved, "10001", 1000)
        assert surface.page.input_value("input[name=f1]") == "10001"

    def test_near_text_password_and_select(self, surface: WebSurface) -> None:
        surface.resolve(target({"kind": "near_text", "anchor": "Password", "control": "password"}), 1000)
        select = surface.resolve(target({"kind": "near_text", "anchor": "Type", "control": "select"}), 1000)
        surface.select(select, "Money Market", 1000)
        assert surface.page.input_value("select[name=f2]") == "Money Market"

    def test_field_value(self, surface: WebSurface) -> None:
        resolved = surface.resolve(target({"kind": "field_value", "label": "Name"}), 1000)
        assert surface.read_text(resolved) == "Avery Testwood"

    def test_table_cell_uses_header_text_not_position(self, surface: WebSurface) -> None:
        # Balance is the 3rd column here; a positional locator recorded elsewhere would read Available.
        resolved = surface.resolve(
            target({"kind": "table_cell", "row": "Share Savings", "column": "Balance"}), 1000
        )
        assert surface.read_text(resolved) == "$4,719.56"

    def test_hidden_duplicates_are_ignored(self, surface: WebSurface) -> None:
        resolved = surface.resolve(target({"kind": "role", "role": "button", "name": "Search"}), 1000)
        assert resolved.strategy_index == 0

    def test_ambiguous_match_is_rejected_and_fallback_used(self, surface: WebSurface) -> None:
        resolved = surface.resolve(
            target(
                {"kind": "role", "role": "link", "name": "Details"},
                {"kind": "css", "selector": "a[href='#y']"},
            ),
            1000,
        )
        assert (resolved.strategy_index, resolved.strategy_kind) == (1, "css")

    def test_not_found_reports_match_count_per_strategy(self, surface: WebSurface) -> None:
        with pytest.raises(TargetNotFound) as exc:
            surface.resolve(
                target(
                    {"kind": "role", "role": "link", "name": "Details"},
                    {"kind": "near_text", "anchor": "Nope", "control": "textbox"},
                ),
                300,
            )
        assert exc.value.match_counts == [2, 0]


class TestFramesAndCheckpoints:
    def test_resolves_inside_named_frame(self, browser_page: Page) -> None:
        browser_page.set_content(
            '<iframe name="main" srcdoc="<table><tr><td>Member Number:</td><td><input></td></tr></table>'
            '<div>MEMBER INQUIRY</div>"></iframe>'
        )
        surface = WebSurface(browser_page, "http://unused")
        surface.resolve(
            target({"kind": "near_text", "anchor": "Member Number", "control": "textbox"}, frame="main"), 2000
        )
        assert surface.check(checkpoint(kind="text_present", text="MEMBER INQUIRY", frame="main"))
        assert not surface.check(checkpoint(kind="text_present", text="MEMBER INQUIRY"))  # not in top doc
        assert not surface.check(checkpoint(kind="text_present", text="x", frame="absent"))

    def test_composite_checkpoints(self, surface: WebSurface) -> None:
        present = {"kind": "text_present", "text": "Share Savings"}
        absent = {"kind": "text_present", "text": "No records found"}
        assert surface.check(checkpoint(kind="all_of", conditions=[present]))
        assert not surface.check(checkpoint(kind="all_of", conditions=[present, absent]))
        assert surface.check(checkpoint(kind="any_of", conditions=[absent, present]))
        assert surface.check(
            checkpoint(
                kind="element_visible", target={"strategies": [{"kind": "field_value", "label": "Name"}]}
            )
        )


class TestDialogs:
    DELETE = {"kind": "role", "role": "button", "name": "Delete"}

    def test_expected_dialog_is_answered_as_declared(self, surface: WebSurface) -> None:
        resolved = surface.resolve(target(self.DELETE), 1000)
        surface.click(resolved, DialogExpectation(accept=True, message_contains="cannot be undone"), 1000)
        [event] = surface.take_dialogs()
        assert event.expected and event.accepted

    def test_unexpected_dialog_is_dismissed_never_accepted(self, surface: WebSurface) -> None:
        surface.click(surface.resolve(target(self.DELETE), 1000), None, 1000)
        [event] = surface.take_dialogs()
        assert not event.expected and not event.accepted

    def test_dialog_with_different_message_is_not_the_expected_one(self, surface: WebSurface) -> None:
        resolved = surface.resolve(target(self.DELETE), 1000)
        surface.click(
            resolved, DialogExpectation(accept=True, message_contains="Open this sub-account"), 1000
        )
        [event] = surface.take_dialogs()
        assert not event.expected and not event.accepted


def test_observe_is_structural(surface: WebSurface) -> None:
    observation = surface.observe()
    assert "top" in observation.locations
    assert not hasattr(observation, "text")


def test_request_filter_blocks_and_records_disallowed_navigation(browser_page: Page) -> None:
    surface = WebSurface(browser_page, "http://unused", request_filter=lambda url: "evil.example" not in url)
    browser_page.set_content('<a href="http://evil.example/steal">Details</a>')
    surface.click(
        surface.resolve(target({"kind": "role", "role": "link", "name": "Details"}), 1000), None, 1000
    )
    browser_page.wait_for_timeout(300)
    assert surface.take_blocked_requests() == ["http://evil.example/steal"]
    assert "evil.example" not in browser_page.url
