# TOOLING.md — axisalignment-paper

## Environment
| Concern | State |
|---|---|
| Language | Python 3 (`code/requirements.txt`); scripts run from `../alignment-axis/.venv` |
| Compute | Dumps need native macOS MPS with 48 GB; nothing here needs a GPU to re-check tables |
| Paper | LaTeX, `paper/main.tex`; Type 3 fonts eliminated (see commit ce6f1b1) |
| Verification | `results/verify_prereg.py`, `logs/inline_checks.py`, `numdiff.py` resolve paths relative to the script |

## Version control
- Public repo `github.com/amitd01/axisalignment-paper`, branch `main`, MIT `LICENSE`.
- `.gitignore`: `.DS_Store`, `__pycache__/`.
- Raw dumps are excluded by policy, not by `.gitignore`; they never enter this tree.

## Cost ledger
Compute is local (owner's Mac); the only spend is paper-side AI sessions.
