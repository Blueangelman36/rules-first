"""Runs with no API key: a stand-in teacher plays the model's part.

    python examples/quickstart.py

Swap in ``ClaudeTeacher()`` (and set ANTHROPIC_API_KEY) to have Claude write the rules instead.
"""

from rules_first import Budget, Classifier, Reply, Usage

FEED = [
    "POCSAG1200: Address: 1234567  Function: 3  Alpha: ENGINE 12 RESPOND TO MAIN ST",
    "POCSAG1200: Address: 9876543  Function: 1  Alpha: LADDER 4 STAGING AT CENTRAL",
    "2026-10-27T03:00:00Z disk /dev/sda1 at 91% used",
    "2026-10-27T03:05:12Z disk /dev/sdb2 at 97% used",
    "user 42 logged in",
]

# Rules you already know go in first and cost nothing.
MY_RULES = [{"name": "login", "category": "auth", "label": "Login by {user}",
             "match": {"regex": r"^user \d+ logged in$"}, "extract": {"user": r"user (\d+)"}}]


class StandInTeacher:
    """Answers the way a model would: one rule per structure it is shown."""

    def teach(self, structures, categories):
        rules = []
        for s in structures:
            if "POCSAG" in s.template:
                rules.append({"fingerprint": s.fingerprint, "category": "pager", "label": "Page {address}",
                              "match": {"regex": r"^POCSAG\d*:.*Alpha:"},
                              "extract": {"address": r"Address:\s*(\d+)"}})
            elif "disk" in s.template:
                rules.append({"fingerprint": s.fingerprint, "category": "system", "label": "Disk {pct}%",
                              "match": {"regex": r"\bdisk /dev/\S+ at \d+% used"},
                              "extract": {"pct": r"at (\d+)%"}})
        return Reply(rules=rules, usage=Usage(input_tokens=1200, output_tokens=150))


def main() -> None:
    clf = Classifier(rules=MY_RULES, categories=["pager", "system", "auth", "unknown"])

    print("First pass, before any teaching:")
    for line in FEED:
        v = clf.classify(line)
        print(f"  {v.tier:8} {v.category:8} {line[:60]}")

    print(f"\n{len(clf.parked())} structure(s) parked from {len(FEED)} messages. One batched question:")
    lesson = clf.teach(StandInTeacher(), Budget())
    print(f"  asked about {lesson.asked}, learned {len(lesson.learned)}, rejected {len(lesson.rejected)}")

    print("\nNew messages of the same shapes now cost nothing:")
    for line in ["POCSAG1200: Address: 5550001  Function: 2  Alpha: MEDIC 7 CANCEL",
                 "2026-10-28T09:00:00Z disk /dev/nvme0n1p3 at 88% used"]:
        v = clf.classify(line)
        print(f"  {v.tier:8} {v.label}")

    print(f"\nresolved without a teacher: {clf.stats()['resolved_without_teacher']:.0%}")


if __name__ == "__main__":
    main()
