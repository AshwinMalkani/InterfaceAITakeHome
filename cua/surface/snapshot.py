"""A surface-agnostic snapshot of what is on screen, for discovery.

Each element gets a short ref (e.g. `e12`) that the model uses to say *what* to act on. The model
never writes selectors: the compiler turns an element's facts (role, accessible name, nearby label,
row and column of a table cell, form-field name) into ordered locator strategies and validates each
one against the live page (cua/agent/recorder.py). These facts map onto desktop accessibility trees
too (UIA ControlType -> role, Name -> name), which is why they're modelled here, not in web.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Element:
    ref: str
    frame: str | None  # frame name; None = top-level document
    role: str  # link | button | textbox | password | select | checkbox | cell | text
    name: str = ""  # accessible name (button value, link text, aria-label, <label>)
    label: str = ""  # from <label for> / aria-label only
    text: str = ""  # visible text (cells and text blocks)
    left_label: str = ""  # nearest cell to the left in the same row that reads like a label
    prev_cell: str = ""  # text of the immediately preceding cell (for "Label: | value" rows)
    row_key: str = ""  # first label-like cell in the row (identifies a table row)
    column_header: str = ""  # header text of this cell's column, from the table's first row
    name_attr: str = ""  # form field name: the server's contract, so usually stable
    tag: str = ""
    options: tuple[str, ...] = ()  # visible option labels, for selects

    @property
    def is_control(self) -> bool:
        return self.role in {"button", "textbox", "password", "select", "checkbox"}


@dataclass
class Snapshot:
    elements: list[Element] = field(default_factory=list)
    locations: dict[str, str] = field(default_factory=dict)  # frame name (or "top") -> path

    def get(self, ref: str) -> Element | None:
        return next((e for e in self.elements if e.ref == ref), None)

    def render(self) -> str:
        """The text the model sees: one line per element, grouped by frame."""
        lines: list[str] = []
        for frame, path in self.locations.items():
            frame_key = None if frame == "top" else frame
            members = [e for e in self.elements if e.frame == frame_key]
            if not members:
                continue
            lines.append(f"[frame {frame}  {path}]")
            for e in members:
                lines.append(f"  {e.ref:<5} {_describe(e)}")
        return "\n".join(lines) or "(nothing visible)"


def _describe(e: Element) -> str:
    if e.role in {"cell", "text"}:
        return f'{e.role} "{e.text}"'
    parts = [e.role]
    if e.name:
        parts.append(f'"{e.name}"')
    if e.left_label and e.left_label != e.name:
        parts.append(f"(next to {e.left_label!r})")
    if e.options:
        parts.append("options=[" + ", ".join(repr(o) for o in e.options) + "]")
    return " ".join(parts)
