"""Synthetic, deterministic member data. No real PII.

- Tax IDs use the 9xx area range, which the SSA never issues as SSNs.
- Phone numbers use 555-01xx, reserved for fiction.
- Seeded with a fixed RNG so every run (and every regression golden file) sees the same data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import count

FIRST_NAMES = ["Avery", "Jordan", "Riley", "Casey", "Morgan", "Quinn", "Rowan", "Parker", "Emerson",
               "Hayden", "Reese", "Skyler"]
LAST_NAMES = ["Testwood", "Sampleton", "Mockridge", "Fakeley", "Demoson", "Placeholt", "Stubbs",
              "Fixturely", "Dummond", "Proxley", "Seedwell", "Nullman"]

FIRST_MEMBER_ID = 10001
RESTRICTED_MEMBER_ID = "10007"  # an employee account: tellers lack privileges to view it
UNKNOWN_MEMBER_ID = "99999"


@dataclass
class Account:
    suffix: str
    kind: str  # key into tenant labels, e.g. "share_savings"
    balance: Decimal
    available: Decimal


@dataclass
class Member:
    member_id: str
    first_name: str
    last_name: str
    tax_id: str
    phone: str
    member_since: int
    restricted: bool
    accounts: list[Account] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def account(self, suffix: str) -> Account | None:
        return next((a for a in self.accounts if a.suffix == suffix), None)


def _money(rng: random.Random, low: int, high: int) -> Decimal:
    return Decimal(rng.randint(low * 100, high * 100)) / 100


class MemberStore:
    def __init__(self, members: dict[str, Member]) -> None:
        self.members = members
        self._confirmations = count(4501)

    @classmethod
    def seeded(cls, seed: int = 42) -> MemberStore:
        rng = random.Random(seed)
        members: dict[str, Member] = {}
        for i, (first, last) in enumerate(zip(FIRST_NAMES, LAST_NAMES, strict=True)):
            member_id = str(FIRST_MEMBER_ID + i)
            savings = _money(rng, 50, 25000)
            checking = _money(rng, 10, 8000)
            members[member_id] = Member(
                member_id=member_id,
                first_name=first,
                last_name=last,
                tax_id=f"9{rng.randint(10, 99)}-{rng.randint(70, 88)}-{rng.randint(1000, 9999)}",
                phone=f"555-01{rng.randint(0, 99):02d}",
                member_since=rng.randint(1985, 2024),
                restricted=member_id == RESTRICTED_MEMBER_ID,
                accounts=[
                    Account("00", "share_savings", savings, savings),
                    Account("10", "checking", checking, checking - _money(rng, 0, 5)),
                ],
            )
        return cls(members)

    def get(self, member_id: str) -> Member | None:
        return self.members.get(member_id)

    def open_sub_account(
        self, member: Member, kind: str, funding_suffix: str, deposit: Decimal
    ) -> tuple[str, str]:
        """Irreversible: moves money and creates an account. Returns (suffix, confirmation)."""
        funding = member.account(funding_suffix)
        if funding is None or deposit > funding.available:
            raise ValueError("insufficient funds")
        funding.balance -= deposit
        funding.available -= deposit
        suffix = f"{max(int(a.suffix) for a in member.accounts) + 10:02d}"
        member.accounts.append(Account(suffix, kind, deposit, deposit))
        return suffix, f"C-{next(self._confirmations):06d}"

    def sensitive_values(self) -> list[str]:
        """Every raw PII value in the dataset, for leak scans over logs/evidence."""
        values: list[str] = []
        for m in self.members.values():
            values += [m.full_name, m.tax_id, m.phone]
            values += [f"${a.balance:,.2f}" for a in m.accounts]
        return values
