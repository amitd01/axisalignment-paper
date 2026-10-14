#!/usr/bin/env python3
"""Mechanical numeric-fidelity check between paper-v9.md and main.tex.

Terminal Claude cannot read TeX, so paper-v9.md is the source of record and the
md -> TeX transcription has to be verified here. This compares the multiset of
every number appearing in each file, after normalising away the things that are
legitimately different between the two formats (TeX markup, citation keys,
cross-references, markdown table pipes).

A number present in one file and not the other is either a transcription error
or a drift between drafts. Both have happened.
"""
import re
import sys
from collections import Counter

MD = "/home/claude/scale/paper-v9.md"
TEX = "/home/claude/arxiv/main.tex"


def normalise(text, is_tex):
    """Strip everything that is format rather than content."""
    if is_tex:
        # Comments, which carry notes-to-self full of numbers.
        text = re.sub(r"(?<!\\)%.*", "", text)
        # Cross-references render as numbers in the PDF but are keys in source.
        text = re.sub(r"\\(?:ref|label|cite|eqref)\{[^}]*\}", " ", text)
        # Bibitem keys ("li2023", "kirk2024") are identifiers whose embedded
        # digits the extractor would otherwise read as measurements. The year
        # regex below cannot catch them: there is no word boundary in "li2023".
        text = re.sub(r"\\bibitem\{[^}]*\}", " ", text)
        text = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", " ", text)
        # Markup that wraps content without being content.
        text = re.sub(r"\\(?:emph|textbf|textit|texttt|url|href)\{", " ", text)
        text = re.sub(r"\\[a-zA-Z]+", " ", text)
        text = text.replace("$", " ").replace("{", " ").replace("}", " ")
        text = text.replace("&", " ").replace("\\\\", " ")
        # Hard-coded cross-references ("Section~6.3"). These are not \ref, so
        # the strip above misses them. NOTE: because they are hard-coded, they
        # will go stale silently if section numbering ever changes.
        text = re.sub(r"[Ss]ections?\s*~?\s*\d+(?:\.\d+)*", " ", text)
    else:
        # Headings: drop the marker AND the manual section number after it.
        # TeX numbers sections itself, so "## 4.2 Title" has no TeX counterpart.
        text = re.sub(r"^#+\s*(?:\d+(?:\.\d+)*\.?\s*)?", "", text, flags=re.M)
        text = re.sub(r"§+\s*\d+(?:\.\d+)*", " ", text)      # section refs
        text = re.sub(r"[Ss]ections?\s+\d+(?:\.\d+)*", " ", text)
        # Draft header ("Draft v0.9 -- 14 October 2026. Supersedes v0.8.") is
        # markdown-only front matter; the TeX uses \date{\today}.
        text = re.sub(r"\*\*Draft v[\d.]+.*?\*\*", " ", text, flags=re.S)
        text = re.sub(r"\bv0\.\d+\b", " ", text)
        # Ordered-list markers: "7. The 1B lambda null" has no TeX counterpart,
        # because enumerate generates its own numbers.
        text = re.sub(r"^\s*\d+\.\s+", " ", text, flags=re.M)
        text = text.replace("|", " ").replace("*", " ")
        text = re.sub(r"`([^`]*)`", r"\1", text)

    # Shared: things that are identifiers, not measurements.
    text = re.sub(r"arXiv:\d+\.\d+", " ", text)              # arXiv IDs
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)          # years
    text = re.sub(r"\bOLMo-2-\S+", " OLMONAME ", text)       # checkpoint names
    text = re.sub(r"\bLlama-3\.1-T\S*ulu-3-8B", " TULUNAME ", text)
    text = re.sub(r"\b\d{8}\b", " SEED ", text)              # seeds
    return text


def numbers(text):
    """Every numeric literal, as a canonical string."""
    out = []
    for tok in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        v = float(tok)
        # Canonical form: strip sign so +0.008 and 0.008 compare equal, since
        # TeX writes $+0.008$ and the markdown writes +0.008 inconsistently.
        out.append(f"{abs(v):.10g}")
    return Counter(out)


def main():
    md = numbers(normalise(open(MD).read(), is_tex=False))
    tex = numbers(normalise(open(TEX).read(), is_tex=True))

    only_md = md - tex
    only_tex = tex - md

    print(f"md:  {sum(md.values())} numeric tokens, {len(md)} distinct")
    print(f"tex: {sum(tex.values())} numeric tokens, {len(tex)} distinct")
    print()

    if not only_md and not only_tex:
        print("PASS -- identical numeric content.")
        return 0

    if only_md:
        print(f"IN MARKDOWN, NOT IN TEX ({sum(only_md.values())}):")
        for v, n in sorted(only_md.items(), key=lambda kv: -kv[1]):
            print(f"  {v}  x{n}")
        print()
    if only_tex:
        print(f"IN TEX, NOT IN MARKDOWN ({sum(only_tex.values())}):")
        for v, n in sorted(only_tex.items(), key=lambda kv: -kv[1]):
            print(f"  {v}  x{n}")
        print()
    print("Counts differing by small amounts are usually a number repeated in")
    print("one file (a table plus prose) and stated once in the other. Check")
    print("each by hand -- this flags, it does not judge.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
