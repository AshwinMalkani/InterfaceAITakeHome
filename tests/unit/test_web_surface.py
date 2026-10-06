"""Locator strategies and dialog handling in a real (headless) browser, on legacy-style markup."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError
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


class TestRedaction:
    PAGE = """
    <div>MEMBER DETAIL</div>
    <table>
      <tr><td>Name:</td><td><b>Avery Testwood</b></td></tr>
      <tr><td>Phone:</td><td>555-0117</td></tr>
      <tr><td>Notes:</td><td>Name: Avery Testwood</td></tr>
    </table>
    <input type="text" name="f1" value="10001"> <input type="submit" value="Search">
    <iframe name="main" srcdoc="<p>Tax ID:</p><p>945-77-4657</p>"></iframe>
    """
    VOCABULARY = frozenset({"member detail", "name", "phone", "search", "tax id"})
    SENSITIVE = ["Avery Testwood", "555-0117", "10001", "945-77-4657"]

    def _visible(self, page: Page) -> str:
        texts = [f.locator("body").inner_text() for f in page.frames]
        values = page.eval_on_selector_all("input", "els => els.map(e => e.value)")
        return " ".join(texts + values)

    def test_only_vocabulary_survives_redaction(self, browser_page: Page) -> None:
        browser_page.set_content(self.PAGE)
        browser_page.wait_for_timeout(200)  # iframe srcdoc
        surface = WebSurface(browser_page, "http://unused")
        frames = surface.redact_text(self.VOCABULARY)
        visible = self._visible(browser_page)
        assert not [s for s in self.SENSITIVE if s in visible]
        assert "MEMBER DETAIL" in visible and "Name:" in visible and "Tax ID:" in visible
        assert "Search" in visible  # a vocabulary button label stays readable
        surface.restore_text(frames)

    def test_mixed_label_and_value_in_one_node_is_masked_entirely(self, browser_page: Page) -> None:
        browser_page.set_content(self.PAGE)
        surface = WebSurface(browser_page, "http://unused")
        surface.redact_text(self.VOCABULARY)
        assert "Notes" not in self._visible(browser_page)

    def test_page_is_restored_exactly(self, browser_page: Page) -> None:
        browser_page.set_content(self.PAGE)
        browser_page.wait_for_timeout(200)

        def snapshot() -> tuple[list[str], str]:
            # Playwright's screenshot leaves empty style="" attributes behind (from hiding the caret);
            # that's Playwright's residue, not ours, so it's ignored here.
            html = [f.content().replace(' style=""', "") for f in browser_page.frames]
            return html, self._visible(browser_page)

        before = snapshot()
        WebSurface(browser_page, "http://unused").screenshot([], self.VOCABULARY)
        assert snapshot() == before

    def test_unredacted_screenshot_leaves_text_alone(self, browser_page: Page) -> None:
        browser_page.set_content(self.PAGE)
        assert WebSurface(browser_page, "http://unused").screenshot([], None).startswith(b"\x89PNG")


class TestSnapshot:
    PAGE = """
    <table>
      <tr><td>Member Number:</td><td>10001</td><td>Member Since:</td><td>1991</td></tr>
      <tr><td>Name:</td><td>Avery Testwood</td><td></td><td></td></tr>
    </table>
    <table>
      <tr><td>Suffix</td><td>Description</td><td>Balance</td></tr>
      <tr><td>00</td><td>Share Savings</td><td>$4,719.56</td></tr>
    </table>
    <table><tr><td>Account Type:</td><td>
      <select name="f2"><option>Savings</option><option>CD</option></select>
    </td></tr></table>
    <input type="hidden" name="secret" value="x">
    """

    def test_facts_for_legacy_markup(self, surface: WebSurface) -> None:
        surface.page.set_content(self.PAGE)
        snap = surface.snapshot()
        by_text = {e.text: e for e in snap.elements if e.text}
        balance = by_text["$4,719.56"]
        assert (balance.row_key, balance.column_header) == ("Share Savings", "Balance")  # skips numeric "00"
        name = by_text["Avery Testwood"]
        assert name.prev_cell == "Name" and name.column_header == ""  # a data first row is not a header
        select = next(e for e in snap.elements if e.role == "select")
        assert select.left_label == "Account Type" and select.options == ("Savings", "CD")
        assert not any(e.name_attr == "secret" for e in snap.elements)  # hidden inputs are not listed

    def test_refs_resolve_to_the_listed_element(self, surface: WebSurface) -> None:
        surface.page.set_content(self.PAGE)
        select = next(e for e in surface.snapshot().elements if e.role == "select")
        surface.select(surface.resolve_ref(select), "CD", 1000)
        assert surface.page.input_value("select[name=f2]") == "CD"

    def test_validation_requires_unique_match_on_the_same_element(self, surface: WebSurface) -> None:
        surface.page.set_content(self.PAGE)
        balance = next(e for e in surface.snapshot().elements if e.text == "$4,719.56")
        target = Target.model_validate(
            {
                "strategies": [
                    {
                        "kind": "table_cell",
                        "row": "Share Savings",
                        "column": "Balance",
                    },  # unique, same element
                    {
                        "kind": "table_cell",
                        "row": "Share Savings",
                        "column": "Description",
                    },  # unique, wrong element
                    {"kind": "css", "selector": "td"},  # ambiguous
                ]
            }
        )
        assert surface.matches_only(target, balance) == [True, False, False]


def test_policy_covers_every_tab_in_the_session(browser: Browser) -> None:
    """A popup or a second tab must not escape the allowlist (found in the real HITL run)."""
    context = browser.new_context()
    page = context.new_page()
    surface = WebSurface(page, "http://unused", request_filter=lambda url: "evil.example" not in url)
    other = context.new_page()  # e.g. a target=_blank popup, or a tab an operator opened
    with contextlib.suppress(PlaywrightError):  # aborted by the filter
        other.goto("http://evil.example/steal", timeout=3000)
    assert surface.take_blocked_requests() == ["http://evil.example/steal"]
    context.close()


def test_human_action_capture_is_scoped_to_the_session_page(browser: Browser) -> None:
    """Clicks in another tab (e.g. the operator console) are not actions in the handed-off session."""
    context = browser.new_context()
    page, other = context.new_page(), context.new_page()
    seen: list[str] = []
    surface = WebSurface(page, "http://unused")
    surface.capture_human_actions(lambda kind, description, frame: seen.append(description))
    # Real navigations after capture is installed (init scripts run on navigation, not set_content),
    # like the console tab opened mid-handoff in the real run.
    page.goto("data:text/html,<input type=button value='In session'>")
    other.goto("data:text/html,<input type=button value='Hand back'>")
    other.click("input")
    page.click("input")
    page.wait_for_timeout(200)
    assert seen == ['button "In session"']
    context.close()
