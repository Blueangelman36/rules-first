"""The tiers, cheapest first.

Every message falls through these in order and stops at the first that knows it:

=========  ============================================  ==================
tier       what it is                                    cost
=========  ============================================  ==================
exact      byte-identical message seen before            free
rule       one of the rules you wrote                    free
learned    a rule a teacher wrote on an earlier run      free
parked     nothing knew it: kept for the next lesson     free until taught
=========  ============================================  ==================

Teaching is a separate, explicit step (`Classifier.teach`), run when you choose and capped by a
`Budget`. It asks about structures, never messages, in batches, and keeps the rules it gets back,
so the same structure never costs twice. `stats()["resolved_without_teacher"]` is the number to
watch: in steady state it should sit near 1.0, and when it drops, something upstream is producing
new shapes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Sequence

from .budget import Budget
from .fingerprint import Fingerprinter, exact_key
from .learned import LearnedRules
from .rules import Rule, RuleError
from .teacher import Structure, Teacher, accept

TIERS = ("exact", "rule", "learned", "parked")


@dataclass(frozen=True)
class Verdict:
    category: str
    label: str
    tier: str
    fingerprint: str
    rule: str | None = None
    confidence: float = 0.0
    tags: tuple[str, ...] = ()
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class Lesson:
    """What one call to `teach` did."""

    asked: int = 0
    learned: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    gave_up: list[str] = field(default_factory=list)
    skipped: str = ""
    note: str = ""


class Classifier:
    def __init__(
        self,
        rules: Sequence[Rule | dict] = (),
        learned: LearnedRules | None = None,
        categories: Sequence[str] | None = None,
        fingerprinter: Fingerprinter | None = None,
        exact_cache_size: int = 10_000,
        max_parked: int = 10_000,
        examples_per_structure: int = 3,
        max_attempts: int = 2,
    ):
        self.rules = [r if isinstance(r, Rule) else Rule.from_dict(r) for r in rules]
        self.learned = learned if learned is not None else LearnedRules()
        self.categories = list(categories) if categories is not None else None
        self.fp = fingerprinter or Fingerprinter()
        self.exact_cache_size = exact_cache_size
        self.max_parked = max_parked
        self.examples_per_structure = examples_per_structure
        #: How many times one structure may be asked about. Two attempts and then nothing: one
        #: deployment once asked about the same failing item every minute, 243 times, and spent
        #: the day's budget by early afternoon. If answers keep failing, fix why.
        self.max_attempts = max_attempts

        self._lock = threading.Lock()
        self._exact: OrderedDict[str, Verdict] = OrderedDict()
        self._parked: dict[str, Structure] = {}
        #: Examples of structures already taught, kept so a new rule can be checked against them.
        self._taught: dict[str, Structure] = {}
        self._counts = {tier: 0 for tier in TIERS}
        self._parked_overflow = 0

    # -- classifying ---------------------------------------------------------------------------

    def classify(self, text: str, source: str = "") -> Verdict:
        key = exact_key(text, source)
        with self._lock:
            cached = self._exact.get(key)
            if cached is not None:
                self._exact.move_to_end(key)
                self._counts["exact"] += 1
                return replace(cached, tier="exact")

        fingerprint = self.fp.fingerprint(text, source)
        for tier, rules in (("rule", self.rules), ("learned", self.learned.all())):
            for rule in rules:
                if rule.matches(text, source, fingerprint):
                    verdict = self._verdict(rule, text, tier, fingerprint)
                    with self._lock:
                        self._counts[tier] += 1
                        self._remember(key, verdict)
                    return verdict

        with self._lock:
            self._counts["parked"] += 1
            self._park(fingerprint, text, source)
        return Verdict(category="unknown", label="", tier="parked", fingerprint=fingerprint)

    def _verdict(self, rule: Rule, text: str, tier: str, fingerprint: str) -> Verdict:
        values = rule.fields(text)
        return Verdict(category=rule.category, label=rule.render(values), tier=tier,
                       fingerprint=fingerprint, rule=rule.name, confidence=rule.confidence,
                       tags=tuple(rule.tags), fields=values)

    def _remember(self, key: str, verdict: Verdict) -> None:
        self._exact[key] = verdict
        while len(self._exact) > self.exact_cache_size:
            self._exact.popitem(last=False)

    def _park(self, fingerprint: str, text: str, source: str) -> None:
        structure = self._parked.get(fingerprint)
        if structure is None:
            if len(self._parked) >= self.max_parked:
                # Bounded, and counted rather than silent: stats() reports the overflow.
                self._parked_overflow += 1
                return
            structure = self._parked[fingerprint] = Structure(
                fingerprint=fingerprint, template=self.fp.template(text), source=source)
        structure.count += 1
        if len(structure.examples) < self.examples_per_structure and text not in structure.examples:
            structure.examples.append(text)

    # -- teaching ------------------------------------------------------------------------------

    def parked(self) -> list[Structure]:
        """Structures waiting for a rule, most frequent first."""
        with self._lock:
            return sorted(self._parked.values(), key=lambda s: -s.count)

    def teach(self, teacher: Teacher, budget: Budget, batch_size: int = 25) -> Lesson:
        """Ask the teacher about the most frequent parked structures, once, within the budget."""
        with self._lock:
            batch = [s for s in sorted(self._parked.values(), key=lambda s: -s.count)
                     if s.attempts < self.max_attempts][:batch_size]
        lesson = Lesson()
        if not batch:
            return lesson

        # Counted before the request is made, so a failing teacher still uses up the budget.
        request = budget.reserve(note=f"{len(batch)} structure(s)")
        if request is None:
            lesson.skipped = budget.check().reason
            return lesson
        lesson.asked = len(batch)
        for structure in batch:
            structure.attempts += 1
        try:
            reply = teacher.teach(batch, self.categories)
        except Exception as exc:  # the teacher's own failure is a reason, not a crash
            reason = f"teacher failed: {type(exc).__name__}: {exc}"
            for structure in batch:
                structure.last_rejection = reason
                lesson.rejected[structure.fingerprint] = reason
            self._note_given_up(batch, lesson)
            return lesson
        budget.record(request, reply.usage.input_tokens, reply.usage.output_tokens,
                      reply.usage.cache_read_tokens)
        lesson.note = reply.note

        proposals = {p.get("fingerprint"): p for p in reply.rules if isinstance(p, dict)}
        with self._lock:
            known = list(self._parked.values()) + list(self._taught.values())
        accepted: list[tuple[Structure, Rule]] = []
        for structure in batch:
            proposal = proposals.get(structure.fingerprint)
            if proposal is None:
                reason = reply.note or "no rule proposed"
            else:
                others = known + [s for s, _ in accepted]
                try:
                    accepted.append((structure, accept(proposal, structure, others, self.categories)))
                    continue
                except RuleError as exc:
                    reason = str(exc)
            structure.last_rejection = reason
            lesson.rejected[structure.fingerprint] = reason

        self.learned.add_many([rule for _, rule in accepted])
        with self._lock:
            for structure, rule in accepted:
                self._parked.pop(structure.fingerprint, None)
                self._taught[structure.fingerprint] = structure
                lesson.learned.append(rule.name)
        self._note_given_up(batch, lesson)
        return lesson

    def _note_given_up(self, batch: Sequence[Structure], lesson: Lesson) -> None:
        for structure in batch:
            if structure.fingerprint in self._parked and structure.attempts >= self.max_attempts:
                lesson.gave_up.append(structure.fingerprint)

    # -- reporting -----------------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            total = sum(self._counts.values())
            free = total - self._counts["parked"]
            return {
                "total": total,
                "by_tier": dict(self._counts),
                "resolved_without_teacher": (free / total) if total else 1.0,
                "parked_structures": len(self._parked),
                "parked_overflow": self._parked_overflow,
                "learned_rules": len(self.learned),
            }
