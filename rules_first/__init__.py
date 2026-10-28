"""rules-first: sort repetitive messages with rules, and let a model write the rules once.

Cheapest first: exact repeats, your rules, rules learned earlier, and only then, for a shape
nothing has seen, one batched question to a teacher whose answer becomes a rule.
"""

from .budget import Budget, Limits
from .classifier import Classifier, Lesson, Verdict
from .fingerprint import Fingerprinter
from .learned import LearnedRules
from .rules import Rule, RuleError
from .teacher import Reply, Structure, Teacher, Usage

__version__ = "0.1.0"

__all__ = [
    "Budget", "Classifier", "Fingerprinter", "LearnedRules", "Lesson", "Limits", "Reply",
    "Rule", "RuleError", "Structure", "Teacher", "Usage", "Verdict", "__version__",
]
