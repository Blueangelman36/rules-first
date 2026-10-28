"""What a teacher is given, what it answers, and the checks every answer must pass.

A teacher is anything that, given a few structures nothing recognised, proposes rules for them:
a person at a prompt, a script, or a language model (`rules_first.claude.ClaudeTeacher`).

Its rules are never trusted as given. Telling a model to "write a pattern that matches only this
structure" does not make it so; checking does. Before a proposed rule is kept it must compile,
match every example of its own structure, and match no example of any other structure the
classifier knows about. A rule that fails stays out, and its structure stays parked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .rules import Rule, RuleError


@dataclass
class Structure:
    """One message shape nothing recognised yet."""

    fingerprint: str
    template: str
    source: str = ""
    #: A few real messages of this shape. Examples next to the template classify far more
    #: reliably than the template alone, for a negligible number of extra tokens.
    examples: list[str] = field(default_factory=list)
    count: int = 0
    attempts: int = 0
    last_rejection: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class Reply:
    #: Proposed rules as plain dicts, each naming the fingerprint it answers.
    rules: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    #: Why the teacher gave nothing back (refused, cut off), when it did.
    note: str = ""


class Teacher(Protocol):
    def teach(self, structures: Sequence[Structure], categories: Sequence[str] | None) -> Reply: ...


def accept(proposal: dict[str, Any], own: Structure, others: Sequence[Structure],
           categories: Sequence[str] | None) -> Rule:
    """Turn one proposal into a Rule, or raise RuleError saying why it cannot be kept."""
    data = dict(proposal)
    data.pop("fingerprint", None)
    data.setdefault("name", f"learned-{own.fingerprint}")
    data["origin"] = "learned"
    match = dict(data.get("match") or {})
    if own.source:
        match.setdefault("source", own.source)
    data["match"] = match
    if not match.get("regex"):
        raise RuleError("a learned rule needs a regex")
    if categories is not None and data.get("category") not in categories:
        raise RuleError(f"category {data.get('category')!r} is not one of {list(categories)}")
    confidence = data.get("confidence", 0.9)
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise RuleError(f"confidence {confidence!r} is not between 0 and 1")

    rule = Rule.from_dict(data)
    missed = [e for e in own.examples if not rule.matches(e, own.source)]
    if missed:
        raise RuleError(f"does not match its own example {missed[0]!r}")
    for other in others:
        if other.fingerprint == own.fingerprint:
            continue
        hit = next((e for e in other.examples if rule.matches(e, other.source)), None)
        if hit is not None:
            # The worst failure: one rule silently claiming another kind of message.
            raise RuleError(f"also matches another structure's example {hit!r}")
    return rule
