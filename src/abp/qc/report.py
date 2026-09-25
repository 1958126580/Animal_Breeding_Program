"""Structured QC findings.

Severity levels
---------------
``info``        descriptive (e.g. parents added as founders).
``review``      a signal for the breeder to inspect; data are *not* changed
                (e.g. extreme but in-range phenotypes, low-MAF markers).
``quarantine``  records excluded from the analysis by an explicit, user-approved
                QC rule; every excluded item is listed (reversible filter list).
``error``       blocking: the analysis stops and no result is published.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ABPError, _jsonable

SEVERITIES = ("info", "review", "quarantine", "error")


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    count: int
    examples: list[dict[str, Any]] = field(default_factory=list)
    error_name: str | None = None  # ABPError name for blocking findings

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"check": self.check, "severity": self.severity,
                          "message": self.message, "count": self.count,
                          "examples": self.examples})


@dataclass
class QCReport:
    section: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    excluded: list[dict[str, Any]] = field(default_factory=list)  # reversible filter list

    def add(self, check: str, severity: str, message: str, items: list[dict] | None = None,
            error_name: str | None = None, max_examples: int = 20) -> None:
        if severity not in SEVERITIES:
            raise ValueError(severity)
        items = items or []
        self.findings.append(Finding(check, severity, message, len(items) or 1,
                                     items[:max_examples], error_name))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def raise_if_blocking(self) -> None:
        """Raise one ABPError summarising *all* blocking findings."""
        errs = self.errors
        if not errs:
            return
        first = errs[0]
        summary = "; ".join(f"{f.check}: {f.message}" for f in errs)
        raise ABPError(first.error_name or "INTERNAL",
                       f"{self.section} QC found {len(errs)} blocking problem(s): {summary}",
                       findings=[f.to_dict() for f in errs])

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"section": self.section, "stats": self.stats,
                          "findings": [f.to_dict() for f in self.findings],
                          "excluded": self.excluded})
