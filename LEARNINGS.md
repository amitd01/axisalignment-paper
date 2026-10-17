# Learnings

> Things that already cost time. Check here before debugging anything odd. Add an entry for anything that took >15 minutes.

| # | Area | Learning | Where it matters |
|---|---|---|---|
| L-01 | Scale inference | OLMo-2-0425-1B vs OLMo-2-1124-7B are different post-training releases; a within-release pair (1124 7B vs 13B) was needed before "same recipe" could be claimed. | paper §2.3, limitation 1 |
| L-02 | Compute | The desktop-bridge VM (4 CPU, 3 GB, no Metal) and the cloud container cannot run model dumps; only Terminal Claude on the Mac has MPS and 48 GB. | any new model run |
| L-03 | Reproducibility | Cloud-computed figures and MPS reruns drift numerically; consistency checks must compare within one platform. | `logs/inline_checks.py` |
| L-04 | Paths | Scripts must resolve data paths relative to the script, not the caller's cwd, or release runs break. | `numdiff.py` |
| L-05 | Thresholds | The psi* threshold was retracted after a misspecification check; keep `exp1d_misspec.py` with the paper so the retraction is auditable. | `THRESHOLD-PROBLEM.md` |
