#!/usr/bin/env python3
"""Fidelity check between paper.md and main.tex.

The markdown is the source of record; the TeX is a separate transcription of it.
Nothing keeps them in sync automatically, so divergence is the expected failure
mode and has happened repeatedly -- in both directions.

Two checks run:

  NUMBERS  every numeric literal in each file, compared as a multiset. A
           measurement present in one file and not the other is a transcription
           error. All 342 decimal measurements should match exactly.

  PROSE    the word stream of each file, compared with difflib. This exists
           because a numeric diff cannot see wording. Two overclaims survived
           earlier syncs precisely this way: "dominates the plausible set" where
           the other file said "exceeds typicality across the plausible set",
           and "first and second person down" where the other said "second
           person down". In both cases one file had been corrected and the other
           had not.

Usage:  python3 numdiff.py
Exit:   0 if numbers match, 1 otherwise. Prose findings are always advisory --
        read them, do not gate on them.
"""
import difflib
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# Default to the two files sitting beside this script in the repo, so anyone who
# clones can run it. Override with arguments for a working copy elsewhere:
#     python3 numdiff.py path/to/paper.md path/to/main.tex
MD = os.path.join(HERE, "paper.md")
TEX = os.path.join(HERE, "main.tex")
if len(sys.argv) == 3:
    MD, TEX = sys.argv[1], sys.argv[2]
elif len(sys.argv) != 1:
    sys.exit(f"usage: {sys.argv[0]} [paper.md main.tex]")


# --------------------------------------------------------------------------
# Numeric comparison
# --------------------------------------------------------------------------

def normalise(text, is_tex):
    """Strip everything that is format rather than content."""
    if is_tex:
        text = re.sub(r"(?<!\\)%.*", "", text)                    # comments
        # Layout constants are not measurements: a p{0.34\textwidth} column spec
        # or a \begin{thebibliography}{99} width hint would otherwise be counted.
        text = re.sub(r"[\d.]+\\(?:text|line|col)width", " ", text)
        text = re.sub(r"\\(?:ref|label|cite|eqref)\{[^}]*\}", " ", text)
        # Bibitem keys ("li2023") embed digits the extractor would read as
        # measurements. The year regex below cannot catch them: there is no word
        # boundary inside "li2023".
        text = re.sub(r"\\bibitem\{[^}]*\}", " ", text)
        text = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", " ", text)
        text = re.sub(r"\\(?:emph|textbf|textit|texttt|url|href)\{", " ", text)
        text = re.sub(r"\\[a-zA-Z]+", " ", text)
        text = text.replace("$", " ").replace("{", " ").replace("}", " ")
        text = text.replace("&", " ").replace("\\\\", " ")
        # Hard-coded cross-references ("Section~6.3"). NOTE: because they are
        # hard-coded rather than \ref, they go stale silently if the section
        # numbering changes.
        text = re.sub(r"[Ss]ections?\s*~?\s*\d+(?:\.\d+)*", " ", text)
    else:
        # Heading markers AND the manual section number after them: TeX numbers
        # its own sections, so "## 4.2 Title" has no counterpart.
        text = re.sub(r"^#+\s*(?:\d+(?:\.\d+)*\.?\s*)?", "", text, flags=re.M)
        text = re.sub(r"§+\s*\d+(?:\.\d+)*", " ", text)
        text = re.sub(r"[Ss]ections?\s+\d+(?:\.\d+)*", " ", text)
        text = re.sub(r"\*\*Draft v[\d.]+.*?\*\*", " ", text, flags=re.S)
        text = re.sub(r"\bv0\.\d+\b", " ", text)
        text = re.sub(r"^\s*\d+\.\s+", " ", text, flags=re.M)      # list markers
        text = text.replace("|", " ").replace("*", " ")
        text = re.sub(r"`([^`]*)`", r"\1", text)

    # Shared: identifiers, not measurements.
    text = re.sub(r"arXiv:\d+\.\d+", " ", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\bOLMo-2-\S+", " OLMONAME ", text)
    text = re.sub(r"\bLlama-3\.1-T\S*ulu-3-8B", " TULUNAME ", text)
    text = re.sub(r"\b\d{8}\b", " SEED ", text)
    return text


def numbers(text):
    """Every numeric literal, canonicalised. Sign is dropped: the two files are
    inconsistent about writing +0.008 versus 0.008."""
    return Counter(
        f"{abs(float(tok)):.10g}"
        for tok in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    )


def measurements(text):
    """Decimals only. Every stage share, CI bound, coefficient and ratio in this
    paper has a decimal point; bare integers are counts and list markers."""
    text = re.sub(r"Llama[- ]3\.1", " ", text)       # a version string
    return Counter(f"{abs(float(x)):.10g}" for x in re.findall(r"\d+\.\d+", text))


# --------------------------------------------------------------------------
# Prose comparison
# --------------------------------------------------------------------------

GREEK = {"\u03c8": " psi ", "\u03b2": " beta ", "\u03bb": " lambda ",
         "\u03b3": " gamma ", "\u03c1": " rho ", "\u00fc": "u"}


def prose_normalise(text, is_tex):
    """Reduce to a bare word stream for comparing CLAIMS, not formatting.

    Cuts the parts that legitimately exist in only one file (TeX preamble and
    bibliography, markdown front matter and reference list), collapses the
    notation the two files spell differently, then drops everything but letters.
    """
    if is_tex:
        text = re.sub(r"(?<!\\)%.*", "", text)
        text = text.split(r"\begin{document}", 1)[-1]
        text = text.split(r"\begin{thebibliography}", 1)[0]
        text = re.sub(r"\\(?:ref|label|cite|eqref|url)\{[^}]*\}", " ", text)
        # Environment delimiters and their column specs have no md counterpart.
        text = re.sub(r"\\(?:begin|end)\{[a-zA-Z*]+\}(?:\{[^}]*\})?", " ", text)
        text = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", " ", text)
        # Math macros must become the SAME word the markdown uses, before the
        # generic backslash strip eats them.
        text = text.replace(r"\rreg", " rhoreg ")
        for macro, word in ((r"\psi", " psi "), (r"\beta", " beta "),
                            (r"\lambda", " lambda "), (r"\gamma", " gamma "),
                            (r"\rho", " rho "), (r"\log", " log "),
                            (r"\times", " "), (r"\sim", " "), (r"\to", " ")):
            text = text.replace(macro, word)
        text = re.sub(r'\\"u', "u", text)
        text = re.sub(r"\\[a-zA-Z]+", " ", text)
    else:
        text = re.sub(r"^\*\*Draft v[\d.]+.*?\*\*", " ", text, flags=re.S | re.M)
        text = text.split("## Appendix B", 1)[0]
        text = text.replace("\u03c1_reg", " rhoreg ").replace("rho_reg", " rhoreg ")
        text = re.sub(r"`([^`]*)`", r"\1", text)

    for ch, repl in GREEK.items():
        text = text.replace(ch, repl)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().split()


# Fragments that differ for structural reasons, not because a claim changed.
# Anything matching is suppressed from the prose report.
PROSE_NOISE = re.compile(
    r"^(?:\s*|nothing|olmoname|tuluname|rhoreg|reg|abstract|document|enumerate"
    r"|itemize|center|tabular|toprule|midrule|bottomrule|appendix|section"
    r"|subsection|paragraph|textbf|emph|texttt|small|newblock|l+|r+|c+)$"
)


def prose_report(md_text, tex_text):
    """Print wording that appears in one file and not the other."""
    a = prose_normalise(md_text, is_tex=False)
    b = prose_normalise(tex_text, is_tex=True)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)

    hits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        in_md = " ".join(a[i1:i2])
        in_tex = " ".join(b[j1:j2])
        if PROSE_NOISE.match(in_md) and PROSE_NOISE.match(in_tex):
            continue
        # A claim divergence is a run of words, not a stray token. Short runs
        # are overwhelmingly alignment artifacts; requiring four words on one
        # side cuts the report to something a person will actually read.
        if max(i2 - i1, j2 - j1) < 4:
            continue
        lo = max(0, i1 - 8)
        hits.append((in_md or "(nothing)", in_tex or "(nothing)",
                     " ".join(a[lo:i1])))

    if not hits:
        print("PROSE: no wording divergence outside known structural noise.")
        return
    print(f"PROSE: {len(hits)} wording divergence(s) to read by hand.")
    print("Look for one file carrying a claim the other has already softened.\n")
    for in_md, in_tex, ctx in hits:
        print(f"    md : {in_md[:150]}")
        print(f"    tex: {in_tex[:150]}")
        print(f"    ctx: ...{ctx[-110:]}\n")


def main():
    md_raw, tex_raw = open(MD).read(), open(TEX).read()
    md_n = normalise(md_raw, is_tex=False)
    tex_n = normalise(tex_raw, is_tex=True)

    md_meas, tex_meas = measurements(md_n), measurements(tex_n)
    print(f"MEASUREMENTS: {sum(md_meas.values())} in md, "
          f"{sum(tex_meas.values())} in tex")
    failed = md_meas != tex_meas
    if failed:
        print(f"  md only : {dict(md_meas - tex_meas)}")
        print(f"  tex only: {dict(tex_meas - md_meas)}")
    else:
        print("  identical, both multiset and count.")

    md_all, tex_all = numbers(md_n), numbers(tex_n)
    only_md, only_tex = md_all - tex_all, tex_all - md_all
    print(f"\nALL NUMBERS: {sum(md_all.values())} in md, "
          f"{sum(tex_all.values())} in tex")
    if only_md or only_tex:
        print(f"  md only : {dict(only_md)}")
        print(f"  tex only: {dict(only_tex)}")
        print("  (bare integers here are usually list markers or table widths)")
    else:
        print("  identical.")

    print()
    prose_report(md_raw, tex_raw)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
