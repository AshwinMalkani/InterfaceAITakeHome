"""Playwright implementation of the Surface protocol.

Resolution rule (the heart of deterministic replay): poll until the timeout; on each poll try
every strategy in order and take the first that matches exactly ONE visible element. Zero
matches and ambiguous matches are both rejected, so we never act on "whichever came first".
Trying all strategies per poll means a fallback wins quickly instead of after the preferred
strategy's full timeout.

Determinism settings: fixed viewport, locale, timezone and reduced motion. Playwright's own
auto-waits are bounded by the step timeout, and the engine waits on post-conditions, never sleeps.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

from playwright.sync_api import Dialog, Frame, Locator, Page, Route, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from cua.artifact.schema import (
    AllOf,
    AnyOf,
    Checkpoint,
    ControlKind,
    CssLocator,
    DialogExpectation,
    ElementVisible,
    FieldValueLocator,
    LabelLocator,
    NearTextLocator,
    RoleLocator,
    TableCellLocator,
    Target,
    TextPresent,
    UrlMatches,
)
from cua.artifact.schema import Locator as LocatorSpec
from cua.surface.base import ActionFailed, DialogEvent, Observation, Resolved, TargetNotFound

POLL_MS = 100

# An element that is fixed/absolute, visible, and covers at least half the viewport is treated as
# a blocking overlay. Legacy modals are almost always built this way (a dimmed full-screen div).
_BLOCKING_OVERLAY_JS = """() => {
  const area = window.innerWidth * window.innerHeight;
  if (!area) return false;
  for (const el of document.querySelectorAll('body *')) {
    const s = getComputedStyle(el);
    if ((s.position !== 'fixed' && s.position !== 'absolute') || s.display === 'none'
        || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) continue;
    const r = el.getBoundingClientRect();
    if (r.width * r.height >= 0.5 * area) return true;
  }
  return false;
}"""

# Visible cell text, ignoring a trailing/embedded ':' and whitespace differences.
_NORM = "normalize-space(translate(., ':', ''))"

_CONTROL_XPATH: dict[ControlKind, str] = {
    ControlKind.TEXTBOX: "self::textarea or self::input[not(@type) or @type='text' or @type='search' "
    "or @type='email' or @type='tel' or @type='number']",
    ControlKind.PASSWORD: "self::input[@type='password']",
    ControlKind.SELECT: "self::select",
    ControlKind.BUTTON: "self::button or self::input[@type='submit' or @type='button' or @type='image']",
    ControlKind.CHECKBOX: "self::input[@type='checkbox']",
}


def xpath_literal(text: str) -> str:
    """Quote arbitrary text for XPath 1.0, which has no escape sequences."""
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    parts = text.split("'")
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in parts) + ")"


def _row_with_cell(text: str) -> str:
    """A table row with a *direct* cell reading `text`. Direct-child matching keeps outer layout
    rows (whose single cell contains a whole nested table) from matching too."""
    return f"//tr[td[{_NORM}={xpath_literal(text)}]]"


def build_locator(frame: Frame, spec: LocatorSpec) -> Locator:
    match spec:
        case RoleLocator(role=role, name=name):
            return frame.get_by_role(role, name=name, exact=True)  # type: ignore[arg-type]
        case LabelLocator(text=text):
            return frame.get_by_label(text, exact=True)
        case NearTextLocator(anchor=anchor, control=control):
            return frame.locator(f"xpath={_row_with_cell(anchor)}//*[{_CONTROL_XPATH[control]}]")
        case FieldValueLocator(label=label):
            return frame.locator(f"xpath=//td[{_NORM}={xpath_literal(label)}]/following-sibling::td[1]")
        case TableCellLocator(row=row, column=column):
            header_index = (
                f"count(ancestor::table[1]/descendant::td[{_NORM}={xpath_literal(column)}][1]"
                "/preceding-sibling::td) + 1"
            )
            return frame.locator(f"xpath={_row_with_cell(row)}/td[position() = {header_index}]")
        case CssLocator(selector=selector):
            return frame.locator(selector)
    raise AssertionError(f"unhandled locator {spec!r}")


def _path(url: str) -> str:
    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "")


class WebSurface:
    def __init__(
        self, page: Page, base_url: str, *, request_filter: Callable[[str], bool] | None = None
    ) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        self._expected_dialog: DialogExpectation | None = None
        self._dialogs: list[DialogEvent] = []
        self._blocked: list[str] = []
        self._request_filter = request_filter
        page.on("dialog", self._on_dialog)
        if request_filter is not None:
            # Network-level enforcement: every request (navigations, frames, XHR, images) passes
            # the filter, so a click that leads somewhere disallowed is stopped as well.
            page.route("**/*", self._filter_request)

    @classmethod
    @contextmanager
    def launch(
        cls,
        base_url: str,
        *,
        headless: bool = True,
        slow_mo_ms: int = 0,
        request_filter: Callable[[str], bool] | None = None,
    ) -> Iterator[WebSurface]:
        """`slow_mo_ms` delays every browser action, for watching a run; never used in production."""
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="UTC",
                reduced_motion="reduce",
            )
            try:
                yield cls(context.new_page(), base_url, request_filter=request_filter)
            finally:
                context.close()
                browser.close()

    # --- network policy -----------------------------------------------------------------

    def _filter_request(self, route: Route) -> None:
        url = route.request.url
        if self._request_filter is not None and not self._request_filter(url):
            self._blocked.append(url)
            route.abort("blockedbyclient")
        else:
            route.continue_()

    def take_blocked_requests(self) -> list[str]:
        blocked, self._blocked = self._blocked, []
        return blocked

    # --- dialogs ------------------------------------------------------------------------

    def _on_dialog(self, dialog: Dialog) -> None:
        expected = self._expected_dialog
        if expected is not None and expected.message_contains in dialog.message:
            if expected.accept:
                dialog.accept()
            else:
                dialog.dismiss()
            self._dialogs.append(DialogEvent(dialog.message, expected=True, accepted=expected.accept))
        else:
            # Never answer an unknown dialog "yes". Dismissing is the conservative default,
            # and the engine treats it as an unexpected state.
            dialog.dismiss()
            self._dialogs.append(DialogEvent(dialog.message, expected=False, accepted=False))

    def take_dialogs(self) -> list[DialogEvent]:
        events, self._dialogs = self._dialogs, []
        return events

    # --- resolution ---------------------------------------------------------------------

    def _frame(self, name: str | None) -> Frame | None:
        return self.page.main_frame if name is None else self.page.frame(name=name)

    def _find_unique(self, target: Target, counts: list[int]) -> Resolved | None:
        frame = self._frame(target.frame)
        if frame is None:
            return None
        for index, spec in enumerate(target.strategies):
            try:
                matches = build_locator(frame, spec).filter(visible=True)
                counts[index] = matches.count()
            except PlaywrightError:  # frame navigating mid-query: not ready yet
                return None
            if counts[index] == 1:
                return Resolved(matches, index, spec.kind)
        return None

    def resolve(self, target: Target, timeout_ms: int) -> Resolved:
        counts = [0] * len(target.strategies)
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            if (resolved := self._find_unique(target, counts)) is not None:
                return resolved
            if time.monotonic() >= deadline:
                raise TargetNotFound(target, counts)
            self.idle(POLL_MS)

    def idle(self, ms: int) -> None:
        # Never time.sleep() here: Playwright's sync API only processes browser events (frames
        # attaching, dialogs, navigations) while a Playwright call is running. Sleeping would
        # leave e.g. the frame list stale forever.
        self.page.wait_for_timeout(ms)

    # --- actions ------------------------------------------------------------------------

    def goto(self, route: str) -> None:
        try:
            self.page.goto(self.base_url + route, wait_until="load")
        except PlaywrightError as exc:
            if self._blocked:  # aborted by policy: the engine reports the violation, not a nav error
                return
            raise ActionFailed(_first_line(exc)) from None

    def click(self, resolved: Resolved, dialog: DialogExpectation | None, timeout_ms: int) -> None:
        self._expected_dialog = dialog
        try:
            resolved.handle.click(timeout=timeout_ms)
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None
        finally:
            self._expected_dialog = None

    def fill(self, resolved: Resolved, value: str, timeout_ms: int) -> None:
        try:
            resolved.handle.fill(value, timeout=timeout_ms)
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None

    def select(self, resolved: Resolved, option: str, timeout_ms: int) -> None:
        try:
            resolved.handle.select_option(label=option, timeout=timeout_ms)
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None

    def press(self, key: str, resolved: Resolved | None, timeout_ms: int) -> None:
        try:
            if resolved is None:
                self.page.keyboard.press(key)
            else:
                resolved.handle.press(key, timeout=timeout_ms)
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None

    def read_text(self, resolved: Resolved) -> str:
        try:
            text: str = resolved.handle.inner_text()
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None
        return text

    # --- state --------------------------------------------------------------------------

    def check(self, checkpoint: Checkpoint) -> bool:
        match checkpoint:
            case TextPresent(text=text, frame=frame_name):
                frame = self._parsed_frame(frame_name)
                return frame is not None and self._count(frame.get_by_text(text).filter(visible=True)) > 0
            case UrlMatches(pattern=pattern, frame=frame_name):
                frame = self._parsed_frame(frame_name)
                return frame is not None and re.search(pattern, _path(frame.url)) is not None
            case ElementVisible(target=target):
                return self._find_unique(target, [0] * len(target.strategies)) is not None
            case AllOf(conditions=conditions):
                return all(self.check(c) for c in conditions)
            case AnyOf(conditions=conditions):
                return any(self.check(c) for c in conditions)
        raise AssertionError(f"unhandled checkpoint {checkpoint!r}")

    def _parsed_frame(self, name: str | None) -> Frame | None:
        """The frame, only once its document is fully parsed. A half-parsed page can show the
        expected text while a modal further down the HTML doesn't exist yet."""
        frame = self._frame(name)
        if frame is None:
            return None
        try:
            return frame if frame.evaluate("document.readyState") != "loading" else None
        except PlaywrightError:  # navigating
            return None

    def blocking_overlay(self) -> str | None:
        for frame in self.page.frames:
            try:
                if frame.evaluate(_BLOCKING_OVERLAY_JS):
                    return "top" if frame is self.page.main_frame else frame.name
            except PlaywrightError:  # frame navigating; it will be checked again next poll
                continue
        return None

    @staticmethod
    def _count(locator: Locator) -> int:
        try:
            return locator.count()
        except PlaywrightError:
            return 0

    def observe(self) -> Observation:
        observation = Observation(title=self.page.title())
        for frame in self.page.frames:
            name = "top" if frame is self.page.main_frame else frame.name
            if name:
                observation.locations[name] = _path(frame.url)
        return observation

    def screenshot(self, mask: list[Target]) -> bytes:
        locators: list[Locator] = []
        for target in mask:
            frame = self._frame(target.frame)
            if frame is not None:
                locators += [build_locator(frame, spec) for spec in target.strategies]
        try:
            return self.page.screenshot(full_page=True, mask=locators, mask_color="#000000")
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0]
