"""Hard ceilings on what the teacher may spend, kept in SQLite so a restart cannot reset them.

Two rules from running this for real (sdrelay, black-chamber):

* **A request is counted before it is sent.** Counting only successes looks fairer and removes
  the protection exactly when it is needed: a loop retrying against a failing service would
  never reach the cap.
* **When the budget is spent nothing stalls.** Unknown structures stay parked and are asked about
  once the window rolls over; the only cost is how long they wait for a label.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    calls_per_hour: int = 20
    calls_per_day: int = 100
    input_tokens_per_day: int = 2_000_000
    output_tokens_per_day: int = 200_000


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""


class Budget:
    """Spend ledger plus limits. ``path=":memory:"`` for one process, a file to survive restarts."""

    def __init__(self, path: str | os.PathLike[str] = ":memory:", limits: Limits = Limits(),
                 clock=time.time):
        self.limits = limits
        self._clock = clock
        self._lock = threading.Lock()
        self._tripped = False
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY, at REAL NOT NULL, "
            "input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, "
            "cache_read_tokens INTEGER NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT '')"
        )

    def trip(self) -> None:
        """The kill switch: no more requests until this process restarts."""
        self._tripped = True

    def check(self) -> Verdict:
        if self._tripped:
            return Verdict(False, "kill switch tripped")
        now = self._clock()
        hour, day = self._since(now - 3600), self._since(now - 86400)
        lim = self.limits
        if hour["calls"] >= lim.calls_per_hour:
            return Verdict(False, f"hourly request cap reached ({lim.calls_per_hour})")
        if day["calls"] >= lim.calls_per_day:
            return Verdict(False, f"daily request cap reached ({lim.calls_per_day})")
        if day["input_tokens"] >= lim.input_tokens_per_day:
            return Verdict(False, f"daily input-token cap reached ({lim.input_tokens_per_day})")
        if day["output_tokens"] >= lim.output_tokens_per_day:
            return Verdict(False, f"daily output-token cap reached ({lim.output_tokens_per_day})")
        return Verdict(True)

    def reserve(self, note: str = "") -> int | None:
        """Record a request before making it. Returns its id, or None when the budget says no."""
        with self._lock:
            if not self.check().allowed:
                return None
            cur = self._db.execute("INSERT INTO spend (at, note) VALUES (?, ?)", (self._clock(), note))
            return int(cur.lastrowid)

    def record(self, request_id: int, input_tokens: int = 0, output_tokens: int = 0,
               cache_read_tokens: int = 0) -> None:
        """Add what a request actually used, once its answer is in."""
        with self._lock:
            self._db.execute(
                "UPDATE spend SET input_tokens = ?, output_tokens = ?, cache_read_tokens = ? WHERE id = ?",
                (int(input_tokens), int(output_tokens), int(cache_read_tokens), request_id),
            )

    def snapshot(self) -> dict:
        now = self._clock()
        return {"tripped": self._tripped, "hour": self._since(now - 3600),
                "day": self._since(now - 86400), "limits": self.limits.__dict__.copy()}

    def _since(self, start: float) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(cache_read_tokens), 0) FROM spend WHERE at >= ?", (start,)
        ).fetchone()
        return {"calls": row[0], "input_tokens": row[1], "output_tokens": row[2], "cache_read_tokens": row[3]}

    def close(self) -> None:
        self._db.close()
