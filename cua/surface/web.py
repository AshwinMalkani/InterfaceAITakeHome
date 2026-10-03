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
from cua.surface.snapshot import Element, Snapshot

POLL_MS = 100

# Allowlist redaction for evidence screenshots. `norm` must match cua.replay.redaction.normalize_label.
# Every change is recorded so the page is restored exactly afterwards (a human may take over this
# same session next).
_REDACT_JS = """(vocabulary) => {
  if (!document.body) return 0;
  const allowed = new Set(vocabulary);
  const norm = s => s.replace(/\\s+/g, ' ').trim().replace(/:$/, '').trim().toLowerCase();
  const block = s => s.replace(/\\S/g, '\u2588');
  const undo = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const tag = node.parentElement ? node.parentElement.tagName : '';
    if (tag === 'SCRIPT' || tag === 'STYLE') continue;
    const text = norm(node.nodeValue);
    if (!text || allowed.has(text)) continue;
    undo.push({node, text: node.nodeValue});
    node.nodeValue = block(node.nodeValue);
  }
  for (const el of document.querySelectorAll('input, textarea')) {
    if (!el.value || el.type === 'hidden') continue;
    const isButton = ['submit', 'button', 'reset'].includes(el.type);
    if (isButton && allowed.has(norm(el.value))) continue;
    undo.push({el, value: el.value});
    el.value = block(el.value);
  }
  window.__cuaRedaction = undo;
  return undo.length;
}"""

# Discovery snapshot. Tags each listed element with data-cua-ref (replacing old tags) so a ref can be
# acted on and so candidate locators can be checked against "the same element". Refs never end up in
# an artifact: the compiler only emits role/label/near_text/field_value/table_cell/name-attribute css.
_SNAPSHOT_JS = """([start, maxElements]) => {
  if (!document.body) return [];
  document.querySelectorAll('[data-cua-ref]').forEach(e => e.removeAttribute('data-cua-ref'));
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const labelish = s => /[A-Za-z]/.test(s);
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const CONTROLS = 'input,select,textarea,button,a[href]';
  const roleOf = el => {
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || 'text').toLowerCase();
    if (tag === 'a') return 'link';
    const buttonTypes = ['submit', 'button', 'reset', 'image'];
    if (tag === 'button' || (tag === 'input' && buttonTypes.includes(type))) return 'button';
    if (tag === 'input' && type === 'password') return 'password';
    if (tag === 'input' && type === 'checkbox') return 'checkbox';
    if (tag === 'input' || tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'select';
    if (tag === 'td' || tag === 'th') return 'cell';
    return 'text';
  };
  const labelFor = el => {
    if (el.getAttribute('aria-label')) return norm(el.getAttribute('aria-label'));
    if (el.id) {
      const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (l) return norm(l.textContent);
    }
    const wrap = el.closest('label');
    return wrap ? norm(wrap.textContent) : '';
  };
  const nameOf = (el, role) => {
    if (role === 'button' && el.tagName === 'INPUT') return norm(el.value);
    if (role === 'link' || role === 'button') return norm(el.textContent);
    return labelFor(el);
  };
  const cellText = c => norm(c.textContent).replace(/:$/, '').trim();
  const isLabelCell = c => labelish(cellText(c)) && !c.querySelector(CONTROLS);
  const context = el => {
    const cell = el.closest('td,th'), row = el.closest('tr');
    if (!cell || !row) return {};
    const cells = [...row.children].filter(c => c.matches('td,th'));
    const i = cells.indexOf(cell);
    const left = cells.slice(0, i).reverse().find(isLabelCell);
    const key = cells.find(c => c !== cell && isLabelCell(c));
    // The first row is a header only if every cell in it reads like a label. Otherwise (e.g.
    // "Member Number: | 10001") its cells are data and must never become column names.
    const table = row.closest('table'), first = table && table.rows[0];
    const isHeaderRow = first && first !== row && [...first.cells].every(isLabelCell);
    const header = isHeaderRow ? first.cells[cell.cellIndex] : null;
    return {
      left_label: left ? cellText(left) : '',
      prev_cell: i > 0 ? cellText(cells[i - 1]) : '',
      row_key: key ? cellText(key) : '',
      column_header: header ? cellText(header) : '',
    };
  };
  const out = [];
  for (const el of document.body.querySelectorAll('*')) {
    if (out.length >= maxElements) break;
    if (!visible(el)) continue;
    const role = roleOf(el);
    const isControl = el.matches(CONTROLS) && !(el.tagName === 'INPUT' && el.type === 'hidden');
    const ownText = norm([...el.childNodes].filter(n => n.nodeType === 3).map(n => n.nodeValue).join(' '));
    const isCell = role === 'cell' && norm(el.textContent) && !el.querySelector(CONTROLS);
    const isText = role === 'text' && ownText && !el.closest('td,th,a,button,label,option,select');
    if (!isControl && !isCell && !isText) continue;
    const ref = 'e' + (start + out.length);
    el.setAttribute('data-cua-ref', ref);
    out.push({
      ref, role, tag: el.tagName.toLowerCase(),
      name: isControl ? nameOf(el, role) : '',
      label: isControl ? labelFor(el) : '',
      text: isControl ? '' : (isCell ? norm(el.textContent) : ownText).slice(0, 160),
      name_attr: el.getAttribute('name') || '',
      options: role === 'select' ? [...el.options].map(o => norm(o.textContent)) : [],
      ...context(el),
    });
  }
  return out;
}"""

# Captures what an operator does in the live session: which element (role + name or nearby label),
# never the values typed. Installed in every frame; the run only records events while a human holds
# the lease, so automation's own clicks are never attributed to the operator.
_CAPTURE_JS = """(() => {
  if (window.__cuaCaptureInstalled) return;
  window.__cuaCaptureInstalled = true;
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim().slice(0, 80);
  const describe = el => {
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    const role = tag === 'a' ? 'link'
      : (tag === 'button' || ['submit', 'button', 'reset'].includes(type)) ? 'button'
      : tag === 'select' ? 'select'
      : (tag === 'input' || tag === 'textarea') ? (type === 'password' ? 'password' : 'textbox')
      : tag;
    let name = (tag === 'input' && ['submit', 'button', 'reset'].includes(type)) ? el.value
      : (role === 'link' || role === 'button') ? el.textContent : '';
    if (!name) {
      const cell = el.closest('td');
      const prev = cell && cell.previousElementSibling;
      if (prev) name = 'next to ' + norm(prev.textContent).replace(/:$/, '');
    }
    return role + (name ? ' ' + JSON.stringify(norm(name)) : '');
  };
  const report = (kind, target) => {
    const el = target.closest ? (target.closest('a,button,input,select,textarea') || target) : target;
    if (window.__cuaHuman) window.__cuaHuman(kind, describe(el));
  };
  document.addEventListener('click', e => report('click', e.target), true);
  document.addEventListener('change', e => report('change', e.target), true);
})()"""

_RESTORE_JS = """() => {
  for (const u of (window.__cuaRedaction || []).reverse()) {
    if (u.node) u.node.nodeValue = u.text; else u.el.value = u.value;
  }
  delete window.__cuaRedaction;
}"""

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
        cdp_port: int | None = None,
    ) -> Iterator[WebSurface]:
        """`slow_mo_ms` delays every browser action, for watching a run; never used in production.

        `cdp_port` exposes the live session over the Chrome DevTools Protocol, so an operator can take
        over *this* browser (chrome://inspect, or a co-browsing UI) instead of getting a fresh one.
        """
        args = [f"--remote-debugging-port={cdp_port}"] if cdp_port else []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless, slow_mo=slow_mo_ms, args=args)
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

    # --- human handoff ------------------------------------------------------------------

    def capture_human_actions(self, callback: Callable[[str, str, str | None], None]) -> None:
        """Call `callback(kind, description, frame)` for operator clicks/changes in any frame.

        Callbacks are delivered while a Playwright call is running, e.g. during `idle()`.
        """

        def on_event(source: dict[str, object], kind: str, description: str) -> None:
            frame = source.get("frame")
            name = getattr(frame, "name", None) or None
            callback(kind, description, name)

        self.page.context.expose_binding("__cuaHuman", on_event)
        self.page.context.add_init_script(script=_CAPTURE_JS)
        for frame in self.page.frames:  # documents that are already loaded
            try:
                frame.evaluate(_CAPTURE_JS)
            except PlaywrightError:
                continue

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

    # --- discovery ----------------------------------------------------------------------

    MAX_SNAPSHOT_ELEMENTS = 400

    def snapshot(self) -> Snapshot:
        """Everything visible and actionable/readable, across frames, with fresh refs."""
        snapshot = Snapshot(locations=self.observe().locations)
        for frame in self.page.frames:
            name = None if frame is self.page.main_frame else (frame.name or None)
            if frame is not self.page.main_frame and name is None:
                continue  # unnamed child frames can't be targeted by an artifact
            try:
                raw = frame.evaluate(_SNAPSHOT_JS, [len(snapshot.elements), self.MAX_SNAPSHOT_ELEMENTS])
            except PlaywrightError:
                continue
            for item in raw:
                item["options"] = tuple(item.get("options") or ())
                snapshot.elements.append(Element(frame=name, **item))
        return snapshot

    def resolve_ref(self, element: Element) -> Resolved:
        """The live control behind a snapshot ref (discovery acts on refs; replay never does)."""
        frame = self._frame(element.frame)
        if frame is None:
            raise ActionFailed(f"frame {element.frame!r} is gone")
        return Resolved(frame.locator(f'[data-cua-ref="{element.ref}"]'), 0, "ref")

    def matches_only(self, target: Target, element: Element) -> list[bool]:
        """Per strategy: does it resolve to exactly one visible element, and is that `element`?"""
        frame = self._frame(target.frame)
        results: list[bool] = []
        for spec in target.strategies:
            try:
                found = build_locator(frame, spec).filter(visible=True) if frame else None
                unique = found is not None and found.count() == 1
                ok = unique and found is not None and found.get_attribute("data-cua-ref") == element.ref
            except PlaywrightError:
                ok = False
            results.append(ok)
        return results

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

    def screenshot(self, mask: list[Target], vocabulary: frozenset[str] | None) -> bytes:
        locators: list[Locator] = []
        for target in mask:
            frame = self._frame(target.frame)
            if frame is not None:
                locators += [build_locator(frame, spec) for spec in target.strategies]
        redacted = self.redact_text(vocabulary) if vocabulary is not None else []
        try:
            return self.page.screenshot(full_page=True, mask=locators, mask_color="#000000")
        except PlaywrightError as exc:
            raise ActionFailed(_first_line(exc)) from None
        finally:
            self.restore_text(redacted)

    def redact_text(self, vocabulary: frozenset[str]) -> list[Frame]:
        """Mask all non-vocabulary text in every frame; returns the frames to restore."""
        redacted: list[Frame] = []
        for frame in self.page.frames:
            try:
                frame.evaluate(_REDACT_JS, sorted(vocabulary))
                redacted.append(frame)
            except PlaywrightError:
                # A frame we can't redact (e.g. mid-navigation) must not appear in clear: fail closed.
                self.restore_text(redacted)
                raise ActionFailed("could not redact a frame; no screenshot taken") from None
        return redacted

    @staticmethod
    def restore_text(frames: list[Frame]) -> None:
        for frame in frames:
            try:
                frame.evaluate(_RESTORE_JS)
            except PlaywrightError:
                continue


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0]
