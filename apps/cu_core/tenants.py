"""Two tenants running the same "vendor product" (CU-Core), configured differently.

This is the stand-in for the multi-tenant reality: same flows and page structure, but
different branding, product version, and a few relabelled fields. A capability recorded on
`alpha` should replay on `beta` with a small, explicit override rather than a re-recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tenant:
    key: str
    brand: str
    product_version: str
    header_color: str
    labels: dict[str, str] = field(default_factory=dict)

    def label(self, key: str) -> str:
        return self.labels.get(key, DEFAULT_LABELS[key])


DEFAULT_LABELS: dict[str, str] = {
    "member_number": "Member Number",
    "search_button": "Search",
    "member_inquiry": "Member Inquiry",
    "share_savings": "Share Savings",
    "checking": "Share Draft Checking",
    "certificate": "Share Certificate",
    "money_market": "Money Market",
    "holiday_club": "Holiday Club",
}

TENANTS: dict[str, Tenant] = {
    "alpha": Tenant(
        key="alpha",
        brand="Summit Valley Credit Union",
        product_version="CU-Core 4.2.7",
        header_color="#1f3b73",
    ),
    "beta": Tenant(
        key="beta",
        brand="Harbor Point FCU",
        product_version="CU-Core 4.3.1",
        header_color="#5a2d0c",
        labels={
            "member_number": "Account #",
            "search_button": "Find",
            "share_savings": "Primary Share",
        },
    ),
}
