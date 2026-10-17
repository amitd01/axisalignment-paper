# CLAUDE.md — axisalignment-paper

## What this repo is
The public release of the alignment-axis study: the paper (`paper/main.tex`), the
scripts behind every cited number (`code/`), their logs (`logs/`), computed tables
(`results/`) and the 135M pilot. It exists so a reader can verify the numbers.

The working folder with raw model dumps is `../alignment-axis` (not a repo, ~54 GB).
Work there; release here.

## Session protocol
1. Read `../alignment-axis/START-HERE.md` for the state of the study.
2. Any change to a number in the paper must come with the script and log that
   produced it, committed in the same change.
3. Never commit `code/dumps/` or anything over a few MB; raw dumps stay in the
   working folder.
4. `Peer Review LLM/` holds LLM-generated peer reviews (Gemini, NotebookLM, OpenAI).
   **Decision 2026-10-17: local only.** It is gitignored and must never be pushed;
   the public record carries the paper and its evidence, not machine reviews of it.
   The folder lives on the owner's Mac beside this repo; treat its contents as input
   for revisions, cite nothing from it.

## Commit convention
Plain imperative summaries describing the evidence change
("release exp1d_misspec.py, the code behind the retracted psi* threshold").
