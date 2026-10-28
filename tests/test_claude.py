"""ClaudeTeacher against a stand-in client: what it sends, and how it reads what comes back.

No network and no API key. The request shape was checked against anthropic 2.3.0's
``client.beta.messages.create`` signature; the stand-in records exactly what it is given.
"""

import json
import unittest
from types import SimpleNamespace

from rules_first import Budget, Classifier
from rules_first.claude import DEFAULT_MODEL, SYSTEM_PROMPT, ClaudeTeacher

PAGES = [
    "POCSAG1200: Address: 1234567  Function: 3  Alpha: ENGINE 12 RESPOND TO MAIN ST",
    "POCSAG1200: Address: 9876543  Function: 1  Alpha: LADDER 4 STAGING AT CENTRAL",
]


class StandInClient:
    def __init__(self, response):
        self.response = response
        self.sent = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return self.response


def response(rules=None, stop="end_turn", text=None, category=None):
    body = text if text is not None else json.dumps({"rules": rules or []})
    return SimpleNamespace(
        stop_reason=stop,
        stop_details=SimpleNamespace(category=category) if category else None,
        content=[SimpleNamespace(type="text", text=body)],
        usage=SimpleNamespace(input_tokens=900, output_tokens=120, cache_read_input_tokens=0),
    )


def parked():
    clf = Classifier(categories=["pager", "unknown"])
    for line in PAGES:
        clf.classify(line)
    return clf


class TestRequest(unittest.TestCase):
    def test_request_shape(self):
        clf = parked()
        kwargs = ClaudeTeacher(client=object()).request(clf.parked(), clf.categories)
        self.assertEqual(kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(kwargs["max_tokens"], 16_000)
        self.assertEqual(kwargs["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kwargs["output_config"]["effort"], "medium")
        self.assertEqual((kwargs["betas"], kwargs["fallbacks"]), (["server-side-fallback-2026-07-01"], "default"))
        self.assertNotIn("thinking", kwargs)  # always adaptive on this model; omitted on purpose

    def test_the_cached_instructions_never_change_between_requests(self):
        clf = parked()
        teacher = ClaudeTeacher(client=object())
        a = teacher.request(clf.parked(), ["pager"])
        b = teacher.request(clf.parked(), ["other", "list"])
        self.assertEqual(a["system"], b["system"])
        self.assertEqual(a["system"][0]["text"], SYSTEM_PROMPT)

    def test_messages_travel_inside_the_marked_block(self):
        clf = parked()
        user = ClaudeTeacher(client=object()).request(clf.parked(), ["pager"])["messages"][0]["content"]
        start, end = user.index("<structures>"), user.index("</structures>")
        self.assertTrue(all(start < user.index(json.dumps(p)[1:-1]) < end for p in PAGES))

    def test_fallbacks_can_be_switched_off(self):
        kwargs = ClaudeTeacher(fallbacks=None, client=object()).request(parked().parked(), None)
        self.assertNotIn("betas", kwargs)
        self.assertNotIn("fallbacks", kwargs)


class TestReading(unittest.TestCase):
    def run_lesson(self, resp):
        clf = parked()
        fp = clf.parked()[0].fingerprint
        if callable(resp):
            resp = resp(fp)
        client = StandInClient(resp)
        lesson = clf.teach(ClaudeTeacher(client=client), Budget())
        return clf, lesson, client

    def test_a_good_answer_becomes_a_learned_rule(self):
        clf, lesson, _ = self.run_lesson(lambda fp: response([{
            "fingerprint": fp, "category": "pager", "label": "Page {address}", "confidence": 0.9,
            "tags": ["pocsag"], "regex": r"^POCSAG\d*:.*Alpha:",
            "extract": [{"name": "address", "regex": r"Address:\s*(\d+)"}],
        }]))
        self.assertEqual(len(lesson.learned), 1)
        verdict = clf.classify("POCSAG1200: Address: 1111111  Function: 3  Alpha: SQUAD 1 AVAILABLE NOW")
        self.assertEqual((verdict.tier, verdict.label), ("learned", "Page 1111111"))

    def test_a_refusal_is_a_reason_not_a_crash(self):
        _, lesson, _ = self.run_lesson(response(stop="refusal", category="cyber"))
        self.assertEqual(lesson.learned, [])
        self.assertIn("declined (cyber)", lesson.note)

    def test_a_cut_off_answer_says_so(self):
        _, lesson, _ = self.run_lesson(response(stop="max_tokens"))
        self.assertIn("max_tokens", lesson.note)

    def test_unreadable_json_is_a_reason_not_a_crash(self):
        _, lesson, _ = self.run_lesson(response(text="not json"))
        self.assertIn("not the expected JSON", lesson.note)

    def test_usage_reaches_the_budget(self):
        clf = parked()
        budget = Budget()
        clf.teach(ClaudeTeacher(client=StandInClient(response())), budget)
        self.assertEqual(budget.snapshot()["day"]["input_tokens"], 900)
        self.assertEqual(budget.snapshot()["day"]["output_tokens"], 120)


if __name__ == "__main__":
    unittest.main()
