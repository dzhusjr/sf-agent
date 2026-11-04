# CLAUDE.md

Read `HANDOFF.md` first: it has the project state, what is done, and exactly what to do next.

Rules for working in this repo:
- Build environments only through `sf/envs/__init__.py` (`make_env_fn` / `make_vec_env`). Never construct envs elsewhere: train/eval/demo mismatches silently ruin agents.
- Never commit ROMs (`*.md`, `*.zip`, `roms/`), `runs/`, or `logs/`. `states/*.state` ARE committed.
- Training happens on the KU Leuven VSC (wice), not locally. Local runs are for tests and short sanity checks only.
- Use `$VSC_DATA`, never `$VSC_HOME`. Account: `lp_deeplearning_gt_2026b`. Always `--clusters=wice` for squeue/sacct.
- Run `python -m pytest -q tests/` after changing `sf/` (7 tests, ~2 min, no ROM needed).
- Config changes go through `configs/*.yaml` or `--set key=value`, not code edits.
