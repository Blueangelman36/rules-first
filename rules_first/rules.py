"""Rules: plain data that sorts a message without asking anyone.

A rule is data, never a Python callable, for one reason: a model writes rules in this same
format and they are saved as JSON. A hand-written rule and a learned one are the same object,
run by the same matcher::

    {
      "name": "pager-dispatch",
      "category": "pager",
      "label": "Dispatch to {address}",
      "confidence": 0.95,
      "tags": ["dispatch"],
      "match": {"regex": "^POCSAG\\\\d*:.*Alpha:", "source": "pager-feed"},
      "extract": {"address": "Address:\\\\s*(\\\\d+)"}
    }
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

#: The longest pattern accepted from anyone. A rule's regex runs against every message that
#: reaches it, so a long one is a slow one.
MAX_PATTERN_CHARS = 300

# A group that repeats and is itself repeated, such as (a+)+ or (\w*)*: the shape behind
# catastrophic backtracking. Python's re has no timeout, so a learned rule shaped like this
# could hang the classifier on one unlucky message. A heuristic, not a proof: it catches the
# common shapes, not every one.
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*}](?:[^()\\]|\\.)*\)[+*{]")


class RuleError(ValueError):
    """A rule that cannot be used, with the reason in words."""


@dataclass
class Rule:
    name: str
    category: str
    label: str = ""
    confidence: float = 0.9
    tags: list[str] = field(default_factory=list)
    match: dict[str, Any] = field(default_factory=dict)
    extract: dict[str, str] = field(default_factory=dict)
    #: "rule" for one you wrote, "learned" for one a teacher wrote.
    origin: str = "rule"

    def __post_init__(self) -> None:
        # Compiled once, at load: a bad pattern should fail loudly at startup, not silently
        # mis-sort traffic for a week.
        if not self.name:
            raise RuleError("a rule needs a name")
        if not self.category:
            raise RuleError(f"rule {self.name!r} needs a category")
        unknown = set(self.match) - {"regex", "source", "fingerprint"}
        if unknown:
            raise RuleError(f"rule {self.name!r}: unknown match keys {sorted(unknown)}")
        self._regex = _compile(self.name, self.match.get("regex"))
        self._extract = {}
        for key, pattern in self.extract.items():
            compiled = _compile(self.name, pattern)
            if compiled is not None and compiled.groups > 1:
                raise RuleError(f"rule {self.name!r}: extract {key!r} may have at most one group")
            self._extract[key] = compiled

    def matches(self, text: str, source: str = "", fingerprint: str = "") -> bool:
        m = self.match
        if not m:
            return False  # an empty match block never fires
        if "fingerprint" in m and fingerprint != m["fingerprint"]:
            return False
        if "source" in m and source != m["source"]:
            return False
        if self._regex is not None and not self._regex.search(text):
            return False
        return True

    def fields(self, text: str) -> dict[str, str]:
        out = {}
        for key, pattern in self._extract.items():
            found = pattern.search(text) if pattern else None
            if found:
                out[key] = found.group(1) if found.groups() else found.group(0)
        return out

    def render(self, fields: dict[str, str]) -> str:
        """The label with {field} placeholders filled; a missing field stays as written."""
        if not self.label:
            return self.category
        return string.Formatter().vformat(self.label, (), _KeepMissing(fields))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "category": self.category, "label": self.label,
            "confidence": self.confidence, "tags": list(self.tags),
            "match": dict(self.match), "extract": dict(self.extract), "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rule":
        known = {"name", "category", "label", "confidence", "tags", "match", "extract", "origin"}
        extra = set(data) - known
        if extra:
            raise RuleError(f"rule {data.get('name')!r}: unknown keys {sorted(extra)}")
        return cls(**data)


class _KeepMissing(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _compile(name: str, pattern: str | None) -> re.Pattern[str] | None:
    if not pattern:
        return None
    if len(pattern) > MAX_PATTERN_CHARS:
        raise RuleError(f"rule {name!r}: pattern is {len(pattern)} characters (max {MAX_PATTERN_CHARS})")
    if _NESTED_QUANTIFIER.search(pattern):
        raise RuleError(f"rule {name!r}: nested repetition like (a+)+ can hang on one message")
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise RuleError(f"rule {name!r}: bad pattern {pattern!r}: {exc}") from exc
