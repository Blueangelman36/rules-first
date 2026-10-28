"""Rules a teacher wrote, saved so the same structure never costs twice.

This is where the cost goes to zero. When a teacher classifies a new structure it does not
return a label for those messages, it returns a rule, which lands here and handles every later
message of that shape for free. Delete the file and the classifier relearns; keep it and the
teacher goes quiet.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import warnings
from pathlib import Path

from .rules import Rule, RuleError


class LearnedRules:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._rules: list[Rule] = []
        #: False once the file proved unreadable. A file that exists and cannot be read is not
        #: an empty file: saving over it would replace every rule it holds with whatever is in
        #: memory, which is nothing. So it is left alone and nothing more is written.
        self.writable = True
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                raise ValueError("expected a JSON list of rules")
        except (OSError, ValueError) as exc:
            self.writable = False
            warnings.warn(f"rules-first: {self.path} could not be read ({exc}); "
                          "learned rules will not be saved until it is fixed or moved", stacklevel=3)
            return
        for entry in entries:
            try:
                rule = Rule.from_dict({**entry, "origin": "learned"})
            except (RuleError, TypeError) as exc:
                # One bad rule must not stop the rest loading.
                warnings.warn(f"rules-first: skipping learned rule: {exc}", stacklevel=3)
                continue
            self._rules.append(rule)

    def add_many(self, rules: list[Rule]) -> None:
        if not rules:
            return
        with self._lock:
            names = {r.name for r in rules}
            self._rules = [r for r in self._rules if r.name not in names]
            for rule in rules:
                rule.origin = "learned"
                self._rules.append(rule)
            self._save_locked()

    def all(self) -> list[Rule]:
        with self._lock:
            return list(self._rules)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rules)

    def _save_locked(self) -> None:
        if self.path is None or not self.writable:
            return
        # Written to a temporary file and moved over the old one: a crash or a full disk half
        # way through leaves the previous file whole, never half a JSON file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps([r.to_dict() for r in self._rules], indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
