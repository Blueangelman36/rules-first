# rules-first

*Sort repetitive messages with rules, and let a model write each rule once.*

Log lines, pager traffic, alerts, webhook payloads: a feed of ten thousand messages an
hour is usually a few dozen *shapes* repeated forever. Sending every message to a
language model to classify it pays for the same answer ten thousand times, gives a
slightly different answer each time, and stops working when the API is down.

rules-first turns that around. Every message goes through free tiers first:

| tier | what it is | cost |
|---|---|---|
| exact | a byte-identical message seen before | free |
| rule | a rule you wrote | free |
| learned | a rule a model wrote on an earlier run | free |
| parked | nothing knew it: kept for the next lesson | free until taught |

When you choose, the parked *shapes* (not the messages) go to a teacher in one batched
request. The teacher answers with regular expressions, not labels. Each rule is
**checked in code before it is kept**: it has to compile, match every example of its
own shape and none of any other, use a category from your list, and avoid patterns
that can hang. A rule that passes is saved, and that shape never costs again.

The model is optional. With no teacher at all, rules-first is a fast rule engine
with a list of what it has not seen yet.

## Install it (easiest way)

You need Python 3.10 or newer. On Windows, get it from
[python.org](https://www.python.org/downloads/) and tick "Add python.exe to PATH".

1. Open a terminal (on Windows: press Start, type `PowerShell`, press Enter).
2. Install the library:

   ```bash
   pip install rules-first
   ```

   Add Claude as a teacher only if you want it: `pip install "rules-first[claude]"`.
3. Try it with no API key. Download
   [examples/quickstart.py](examples/quickstart.py) and run it:

   ```bash
   python quickstart.py
   ```

   It classifies a small feed, parks the shapes it does not know, asks a stand-in
   teacher about both in one batch, and then classifies new messages of those shapes
   for free.

## How it works, in short

1. **Fingerprint.** Values (times, numbers, IPs, paths, hex, UUIDs) become
   placeholders, and so does the free-text value of a labelled field (`Alpha: ...`,
   `msg=...`). Words that are not a field's value are kept, so `user logged in` and
   `user logged out` stay two shapes. These two are one:

   ```text
   POCSAG1200: Address: 1234567  Function: 3  Alpha: ENGINE 12 RESPOND TO MAIN ST
   POCSAG1200: Address: 9876543  Function: 1  Alpha: LADDER 4 STAGING AT CENTRAL
   → POCSAG1200: Address: <INT> Function: <INT> Alpha: <TEXT>
   ```

2. **Classify.** Exact cache, then your rules, then learned rules. Nothing matched:
   park the shape with up to three examples and a count.
3. **Teach**, when you call it. One request per batch of shapes, counted against a
   `Budget` before it is sent. Each shape gets at most two attempts.
4. **Keep** the rules that pass the checks, in a JSON file written atomically.

## Using it

```python
from rules_first import Budget, Classifier, LearnedRules, Limits

clf = Classifier(
    rules=[{"name": "login", "category": "auth", "label": "Login by {user}",
            "match": {"regex": r"^user \d+ logged in$"}, "extract": {"user": r"user (\d+)"}}],
    learned=LearnedRules("learned-rules.json"),
    categories=["auth", "pager", "system", "unknown"],
)

verdict = clf.classify("user 42 logged in")
print(verdict.tier, verdict.category, verdict.label)   # rule auth Login by 42
```

Teach on your schedule (a cron job, once an hour, by hand), with a budget that
persists across restarts:

```python
from rules_first.claude import ClaudeTeacher      # needs ANTHROPIC_API_KEY

budget = Budget("spend.sqlite", limits=Limits(calls_per_hour=5, calls_per_day=20))
lesson = clf.teach(ClaudeTeacher(), budget)
print(lesson.asked, lesson.learned, lesson.rejected, lesson.skipped)
```

`clf.stats()["resolved_without_teacher"]` is the number to watch. In steady state it
sits near 1.0. When it drops, something upstream started producing new shapes.

### Your own teacher

Anything with a `teach(structures, categories)` method that returns a `Reply` works:
a different model, a person at a prompt, a lookup table. The checks apply to every
teacher equally.

## What it will not do

- **Send a message to the model per message.** Only shapes, only in batches, only
  when you call `teach`.
- **Spend past its budget.** A request is counted *before* it is sent, so one that
  times out still counts. A spent budget parks messages; it does not fail them.
  `budget.trip()` stops all teaching until the ledger is reset.
- **Keep a rule it cannot check.** Wrong category, a regex that misses its own
  examples or claims another shape's, nested repetition like `(a+)+`, or a pattern
  over 300 characters: rejected, with the reason in `lesson.rejected`.
- **Overwrite a learned-rules file it could not read.** It warns and stops saving
  instead.
- **Follow instructions inside your messages.** The Claude teacher sends them inside
  a marked block that its instructions call material, not instructions. And whatever
  comes back is checked in code anyway.

## Costs

The default path costs nothing: no teacher, no network. With `ClaudeTeacher`, each
`teach` call is at most one request per batch of up to 25 shapes, and the default
`Limits` cap it at 20 requests an hour and 100 a day. The instructions carry a cache
marker, but at roughly 350 tokens (estimated from their length, not counted by the
API) they are under the 512-token minimum a prompt needs before it is cached, so today
each batch pays for them in full. That is a few hundred tokens per batch, not per
message.

## Where it came from

Extracted from a radio-monitoring relay that classified pager, trunked-radio and
weather traffic on a Raspberry Pi. Two lessons from that project are built in: a
model told "never invent a category" still invented them, so categories are checked
in code; and a fingerprint that masked plain words turned two kinds of message into
one shape, so the first rule learned for one silently claimed the other's traffic.

## What is not verified

- The Claude teacher's request shape is checked against the `anthropic` SDK 2.3.0
  signature (CI `claude-sdk` job) and against a stand-in client. It has not been run
  against the live API from this repository's CI, which holds no key.
- The default value masks were tuned on English log and pager text. Feeds with other
  formats may need their own `Fingerprinter(value_masks=...)`.

## Publishing a release

For the maintainer. One-time setup:

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a pending trusted
   publisher: project `rules-first`, owner `Blueangelman36`, repository
   `rules-first`, workflow `release.yml`, environment `pypi`.
2. In this repository on GitHub: Settings → Environments → New environment, named
   `pypi`. Add yourself under "Required reviewers" if you want to approve each
   upload.
3. Settings → Secrets and variables → Actions → **Variables** tab (the repository
   one, not the environment one) → New repository variable: `PUBLISH_TO_PYPI` =
   `true`.

Each release: set the same version in `pyproject.toml` and
`rules_first/__init__.py`, merge, then on GitHub go to Releases → Draft a new release
→ create tag `v0.1.0` (matching the version) → Publish. The release workflow checks
that the tag and both versions agree before it builds anything.

## License

MIT
