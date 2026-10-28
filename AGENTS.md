# Notes for coding agents

Before handing back a change, run:

```bash
python -m unittest discover -s tests -t .
PYTHONPATH=. python examples/quickstart.py
```

No API key is needed for either. `tests/test_claude.py` uses a stand-in client;
never add a test that calls the live API.

Things that look odd and must stay:

- `SYSTEM_PROMPT` in `rules_first/claude.py` is byte-stable. Nothing per-request
  (time, ids, categories) goes in it, or prompt caching breaks on every call.
- `Budget.reserve` writes the row *before* the request. A request that times out
  has still been paid for.
- `accept` in `rules_first/teacher.py` is the only gate between a teacher and the
  learned file. Loosen a check there only with a test showing why.
- The fingerprinter keeps words that are not a labelled field's value. Masking them
  merged two kinds of message into one shape in the project this came from.
- `LearnedRules` sets `writable = False` when its file cannot be read. Saving over
  an unreadable file would destroy every rule in it.
