#!/usr/bin/env python3
"""Verify the six pre-registration hashes.

Each PREREG-*.sha256 was written before the corresponding data existed. Four of
the six verify directly against the whole file:

    shasum -a 256 -c PREREG-REGISTER.sha256

Two -- PREREG-13B and PREREG-RESIDUAL -- carry a dated note appended after the
hash was taken. For those the hash covers the original text only, so hashing the
whole file fails by design. This script finds the boundary and checks the
original bytes, so all six can be verified in one command:

    python3 verify_prereg.py

Exit status is 0 only if every pre-registration verifies.
"""
import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
# The separator that precedes an after-the-fact note. The hash covers everything
# above it.
NOTE_MARKER = re.compile(rb"^##\s*Appended note", re.M)


def expected_hash(path):
    """The recorded digest. Accepts bare-hash and `shasum`-style formats."""
    m = re.search(rb"[0-9a-f]{64}", path.read_bytes())
    if not m:
        raise ValueError(f"no sha256 digest found in {path.name}")
    return m.group(0).decode()


def verify(stem):
    """Return (ok, detail) for one pre-registration."""
    md = (HERE / f"{stem}.md").read_bytes()
    want = expected_hash(HERE / f"{stem}.sha256")

    if hashlib.sha256(md).hexdigest() == want:
        return True, "whole file"

    note = NOTE_MARKER.search(md)
    if not note:
        return False, "hash does not match and there is no appended note"

    # Walk back from the note marker to find the byte count the hash covers.
    # The gap is the blank line and `---` separator between text and note, so a
    # few hundred bytes of slack is ample.
    for cut in range(max(0, note.start() - 400), note.start() + 1):
        if hashlib.sha256(md[:cut]).hexdigest() == want:
            pct = 100.0 * cut / len(md)
            return True, (
                f"original text only: first {cut} of {len(md)} bytes "
                f"({pct:.0f}%), note appended after hashing"
            )
    return False, "hash matches neither the whole file nor the text above the note"


def main():
    stems = sorted(p.stem for p in HERE.glob("PREREG-*.sha256"))
    if not stems:
        print("no PREREG-*.sha256 files found", file=sys.stderr)
        return 2

    failures = 0
    width = max(len(s) for s in stems)
    for stem in stems:
        ok, detail = verify(stem)
        print(f"{stem:<{width}}  {'OK  ' if ok else 'FAIL'}  {detail}")
        failures += not ok

    print()
    if failures:
        print(f"{failures} of {len(stems)} pre-registrations FAILED to verify.")
        return 1
    print(f"All {len(stems)} pre-registrations verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
