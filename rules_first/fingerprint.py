"""Structural fingerprints: many messages, one shape.

The cost model rests on this file. A fingerprint collapses every message that shares a
*shape* into one key, so a feed of ten thousand messages an hour presents as a handful of
distinct structures. A teacher is only ever asked about a structure it has never seen, never
about an individual message::

    "POCSAG1200: Address: 1234567  Function: 3  Alpha: ENGINE 12 RESPOND TO MAIN ST"
    "POCSAG1200: Address: 9876543  Function: 1  Alpha: LADDER 4 STAGING AT CENTRAL"

both become ``POCSAG1200: Address: <INT> Function: <INT> Alpha: <TEXT>``: one fingerprint, one
question between them.

Two steps:

1. **Values** (timestamps, UUIDs, addresses, paths, hex, numbers) become placeholders.
2. **Free text** becomes ``<TEXT>``, but only where it is the value of a labelled field: the words
   after ``Alpha:`` or ``msg=``, up to the next label. Words that are not a field's value are kept,
   because they are what tells one kind of message from another: ``user logged in`` and ``user
   logged out`` must stay two shapes, and so must ``DMR RX`` and ``TETRA-BURST``. Two kinds of
   message sharing a fingerprint is the worst failure this file can have, because one learned
   rule then silently claims the other's traffic.

Adapted from sdrelay, whose version masked any run of three plain words. That left the first
word of a message body in place, so ``Alpha: ENGINE 12 RESPOND`` and ``Alpha: LADDER 4 STAGING``
were two shapes (and two paid questions), and it read a seven-digit address as hex.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

#: (pattern, placeholder) pairs, applied in order: whole timestamps before their parts, UUIDs and
#: addresses before the hex and number rules that would chew them into fragments.
DEFAULT_VALUE_MASKS: tuple[tuple[str, str], ...] = (
    (r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b", "<TS>"),
    (r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<TIME>"),
    (r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "<UUID>"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "<IP>"),
    (r"(?<![\w/])(?:/[\w.@-]+){2,}/?", "<PATH>"),
    # Hex needs a letter and a digit (or 0x): all digits is a number, all letters is a word
    # ("decade" is six hex digits).
    (r"\b(?:0x[0-9a-fA-F]+|(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,})\b", "<HEX>"),
    (r"-?\b\d+\.\d+\b", "<FLOAT>"),
    (r"\b\d+\b", "<INT>"),
)

#: Templates are cut to this length. A message's identity lives at its front, and an unbounded
#: template bloats both storage and every prompt it appears in.
MAX_TEMPLATE_CHARS = 300

_LETTER = re.compile(r"[A-Za-z]")


def _is_label(token: str) -> bool:
    return len(token) > 1 and token[-1] in ":="


@dataclass(frozen=True)
class Fingerprinter:
    """Turns text into a template and a stable key. Swap the masks to suit your data."""

    value_masks: tuple[tuple[str, str], ...] = DEFAULT_VALUE_MASKS
    #: Mask a labelled field's value when it is two or more words. Off: only values are masked.
    mask_free_text: bool = True
    max_chars: int = MAX_TEMPLATE_CHARS

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", tuple((re.compile(p), r) for p, r in self.value_masks))

    def template(self, text: str) -> str:
        """Reduce one message to its structural template."""
        out = text.strip()
        for pattern, placeholder in self._compiled:  # type: ignore[attr-defined]
            out = pattern.sub(placeholder, out)
        tokens = out.split()
        if not self.mask_free_text:
            return " ".join(tokens)[: self.max_chars]
        result: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            result.append(token)
            i += 1
            if not _is_label(token):
                continue
            end = i
            while end < len(tokens) and not _is_label(tokens[end]):
                end += 1
            value = tokens[i:end]
            words = [t for t in value if not t.startswith("<") and _LETTER.search(t)]
            if len(value) >= 2 and words:
                result.append("<TEXT>")
                i = end
        return " ".join(result)[: self.max_chars]

    def fingerprint(self, text: str, source: str = "") -> str:
        """Stable key for (source, message shape)."""
        return _digest(f"{source}|{self.template(text)}")


def exact_key(text: str, source: str = "") -> str:
    """Key for byte-identical messages from the same source."""
    return _digest(f"{source}|{text}")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]
