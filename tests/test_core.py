import json
import tempfile
import unittest
import warnings
from pathlib import Path

from rules_first import Budget, Classifier, Fingerprinter, LearnedRules, Limits, Reply, Rule, RuleError, Usage

PAGES = [
    "POCSAG1200: Address: 1234567  Function: 3  Alpha: ENGINE 12 RESPOND TO MAIN ST",
    "POCSAG1200: Address: 9876543  Function: 1  Alpha: LADDER 4 STAGING AT CENTRAL",
    "POCSAG1200: Address: 5550001  Function: 2  Alpha: MEDIC 7 CANCEL CALL NOW",
]
DISK = [
    "2026-10-27T03:00:00Z disk /dev/sda1 at 91% used",
    "2026-10-27T03:05:12Z disk /dev/sdb2 at 97% used",
]


class ScriptedTeacher:
    """A teacher that answers from a table, keyed by an example's text, and counts its calls."""

    def __init__(self, answers, usage=Usage(100, 20, 0), fail=None):
        self.answers = answers
        self.usage = usage
        self.fail = fail
        self.calls = 0
        self.seen = []

    def teach(self, structures, categories):
        self.calls += 1
        self.seen.append([s.fingerprint for s in structures])
        if self.fail:
            raise self.fail
        rules = []
        for s in structures:
            for needle, rule in self.answers.items():
                if any(needle in e for e in s.examples):
                    rules.append({"fingerprint": s.fingerprint, **rule})
        return Reply(rules=rules, usage=self.usage)


PAGER_RULE = {"category": "pager", "label": "Page {address}", "match": {"regex": r"^POCSAG\d*:.*Alpha:"},
              "extract": {"address": r"Address:\s*(\d+)"}}
DISK_RULE = {"category": "system", "label": "Disk {pct}%", "match": {"regex": r"\bdisk /dev/\S+ at \d+% used"},
             "extract": {"pct": r"at (\d+)%"}}


class TestFingerprint(unittest.TestCase):
    def test_messages_of_one_shape_share_a_fingerprint(self):
        fp = Fingerprinter()
        self.assertEqual(len({fp.fingerprint(p) for p in PAGES}), 1)
        self.assertEqual(len({fp.fingerprint(d) for d in DISK}), 1)

    def test_different_literal_words_stay_different(self):
        # Collapsing literal words made two kinds of message share one rule in sdrelay.
        fp = Fingerprinter()
        self.assertNotEqual(fp.fingerprint("DMR RX slot 1 talkgroup 42"),
                            fp.fingerprint("TETRA-BURST slot 1 talkgroup 42"))

    def test_a_word_outside_a_field_is_part_of_the_shape(self):
        fp = Fingerprinter()
        self.assertNotEqual(fp.fingerprint("user 42 logged in"), fp.fingerprint("user 42 logged out"))

    def test_an_all_digit_address_is_a_number_not_hex(self):
        # sdrelay read the seven-digit pager address as <HEX>.
        self.assertIn("Address: <INT>", Fingerprinter().template(PAGES[0]))
        self.assertIn("<HEX>", Fingerprinter().template("crc 0x1f2e failed"))

    def test_the_source_is_part_of_the_shape(self):
        fp = Fingerprinter()
        self.assertNotEqual(fp.fingerprint(DISK[0], "host-a"), fp.fingerprint(DISK[0], "host-b"))


class TestRules(unittest.TestCase):
    def test_match_extract_and_label(self):
        rule = Rule(name="pager", **PAGER_RULE)
        self.assertTrue(rule.matches(PAGES[0]))
        self.assertEqual(rule.render(rule.fields(PAGES[0])), "Page 1234567")

    def test_a_missing_field_stays_in_the_label_instead_of_raising(self):
        rule = Rule(name="x", category="c", label="{absent} here", match={"regex": "."})
        self.assertEqual(rule.render({}), "{absent} here")

    def test_an_empty_match_never_fires(self):
        self.assertFalse(Rule(name="x", category="c").matches("anything"))

    def test_bad_rules_fail_at_load_with_a_reason(self):
        for bad in ({"match": {"regex": "("}}, {"match": {"regex": "(a+)+$"}},
                    {"match": {"regex": "x" * 301}}, {"match": {"rgx": "a"}},
                    {"match": {"regex": "a"}, "extract": {"k": "(a)(b)"}}):
            with self.subTest(bad=bad), self.assertRaises(RuleError):
                Rule(name="bad", category="c", **bad)


class TestClassifier(unittest.TestCase):
    def test_tiers_cheapest_first(self):
        clf = Classifier(rules=[{"name": "pager", **PAGER_RULE}])
        self.assertEqual(clf.classify(PAGES[0]).tier, "rule")
        self.assertEqual(clf.classify(PAGES[0]).tier, "exact")
        unknown = clf.classify(DISK[0])
        self.assertEqual((unknown.tier, unknown.category), ("parked", "unknown"))

    def test_parking_keeps_one_structure_with_examples_and_a_count(self):
        clf = Classifier()
        for line in PAGES:
            clf.classify(line)
        [structure] = clf.parked()
        self.assertEqual(structure.count, 3)
        self.assertEqual(structure.examples, PAGES)

    def test_parking_is_bounded_and_says_so(self):
        clf = Classifier(max_parked=1)
        clf.classify(PAGES[0])
        clf.classify(DISK[0])
        self.assertEqual(clf.stats()["parked_structures"], 1)
        self.assertEqual(clf.stats()["parked_overflow"], 1)

    def test_a_taught_structure_costs_once_and_is_free_afterwards(self):
        clf = Classifier()
        for line in PAGES + DISK:
            clf.classify(line)
        teacher = ScriptedTeacher({"POCSAG": PAGER_RULE, "disk /dev": DISK_RULE})
        lesson = clf.teach(teacher, Budget())
        self.assertEqual((lesson.asked, len(lesson.learned), lesson.rejected), (2, 2, {}))
        self.assertEqual(teacher.calls, 1)  # one batched request for both structures

        new_page = "POCSAG1200: Address: 4444444  Function: 3  Alpha: TANKER 2 TO STATION"
        verdict = clf.classify(new_page)
        self.assertEqual((verdict.tier, verdict.label), ("learned", "Page 4444444"))
        self.assertEqual(clf.teach(teacher, Budget()).asked, 0)
        self.assertEqual(teacher.calls, 1)

    def test_resolved_without_teacher_share(self):
        clf = Classifier(rules=[{"name": "pager", **PAGER_RULE}])
        for line in PAGES + [DISK[0]]:
            clf.classify(line)
        self.assertAlmostEqual(clf.stats()["resolved_without_teacher"], 0.75)


class TestWhatATeacherMayNotDo(unittest.TestCase):
    def lesson(self, rule, categories=None):
        clf = Classifier(categories=categories)
        for line in PAGES + DISK:
            clf.classify(line)
        return clf, clf.teach(ScriptedTeacher({"POCSAG": rule}), Budget())

    def test_a_rule_that_misses_its_own_examples_is_not_kept(self):
        _, lesson = self.lesson({**PAGER_RULE, "match": {"regex": "ENGINE"}})
        self.assertEqual(lesson.learned, [])
        self.assertIn("does not match its own example", next(iter(lesson.rejected.values())))

    def test_a_rule_that_claims_another_structure_is_not_kept(self):
        clf, lesson = self.lesson({**PAGER_RULE, "match": {"regex": r"\d"}})
        self.assertEqual(lesson.learned, [])
        self.assertTrue(any("another structure" in r for r in lesson.rejected.values()))
        self.assertEqual(clf.classify(DISK[1]).tier, "parked")

    def test_a_category_outside_the_list_is_not_kept(self):
        _, lesson = self.lesson({**PAGER_RULE, "category": "radio-stuff"}, categories=["pager", "system"])
        self.assertEqual(lesson.learned, [])

    def test_a_pattern_that_can_hang_is_not_kept(self):
        _, lesson = self.lesson({**PAGER_RULE, "match": {"regex": r"^(POCSAG\w*)+:"}})
        self.assertEqual(lesson.learned, [])

    def test_two_attempts_and_then_nothing(self):
        clf = Classifier()
        clf.classify(PAGES[0])
        teacher = ScriptedTeacher({})  # never proposes anything
        clf.teach(teacher, Budget())
        second = clf.teach(teacher, Budget())
        self.assertEqual(len(second.gave_up), 1)
        self.assertEqual(clf.teach(teacher, Budget()).asked, 0)
        self.assertEqual(teacher.calls, 2)


class TestBudget(unittest.TestCase):
    def test_a_failed_request_still_counts(self):
        clf = Classifier()
        clf.classify(PAGES[0])
        budget = Budget(limits=Limits(calls_per_hour=1))
        lesson = clf.teach(ScriptedTeacher({}, fail=TimeoutError("slow")), budget)
        self.assertIn("teacher failed", next(iter(lesson.rejected.values())))
        self.assertFalse(budget.check().allowed)

    def test_caps_and_the_kill_switch(self):
        budget = Budget(limits=Limits(calls_per_hour=2))
        self.assertIsNotNone(budget.reserve())
        budget.trip()
        self.assertEqual(budget.check().reason, "kill switch tripped")

    def test_a_spent_budget_parks_rather_than_fails(self):
        clf = Classifier()
        clf.classify(PAGES[0])
        budget = Budget(limits=Limits(calls_per_hour=0))
        lesson = clf.teach(ScriptedTeacher({"POCSAG": PAGER_RULE}), budget)
        self.assertIn("hourly request cap", lesson.skipped)
        self.assertEqual(clf.classify(PAGES[1]).tier, "parked")

    def test_tokens_count_and_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spend.sqlite"
            budget = Budget(path, limits=Limits(output_tokens_per_day=50))
            budget.record(budget.reserve(), input_tokens=10, output_tokens=60)
            budget.close()
            again = Budget(path, limits=Limits(output_tokens_per_day=50))
            self.assertIn("output-token cap", again.check().reason)
            again.close()


class TestLearnedFile(unittest.TestCase):
    def test_rules_are_saved_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.json"
            clf = Classifier(learned=LearnedRules(path))
            clf.classify(PAGES[0])
            clf.teach(ScriptedTeacher({"POCSAG": PAGER_RULE}), Budget())
            reloaded = Classifier(learned=LearnedRules(path))
            self.assertEqual(reloaded.classify(PAGES[1]).tier, "learned")

    def test_an_unreadable_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.json"
            path.write_text("{ not json", encoding="utf-8")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                store = LearnedRules(path)
            self.assertFalse(store.writable)
            store.add_many([Rule(name="x", category="c", match={"regex": "x"})])
            self.assertEqual(path.read_text(encoding="utf-8"), "{ not json")

    def test_one_bad_rule_does_not_stop_the_rest_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.json"
            path.write_text(json.dumps([
                {"name": "ok", "category": "c", "match": {"regex": "ok"}},
                {"name": "bad", "category": "c", "match": {"regex": "("}},
            ]), encoding="utf-8")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                store = LearnedRules(path)
            self.assertEqual([r.name for r in store.all()], ["ok"])


if __name__ == "__main__":
    unittest.main()
