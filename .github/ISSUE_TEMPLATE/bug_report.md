---
name: Bug report
about: Something misbehaves at runtime (NaN, crash, wrong metric, etc.)
title: "[BUG] "
labels: bug
assignees: ''
---

## What went wrong

<!-- One-sentence headline, then details. Include exact error message and
     traceback if any. -->

## Reproduction

```bash
# The exact commands that trigger the bug, starting from a clean clone
# (or state the required pre-state, e.g. "preprocessing has been run").
poetry run python run_pipeline.py --preprocess-only --source local
poetry run python src/<module>.py
```

Minimal Python reproducer, if applicable:

```python
# <= 20 lines. Keep it focused.
```

## Expected vs actual behaviour

- **Expected**:
- **Actual**:

## Environment

- OS:
- Python version (`python --version`):
- Poetry version (`poetry --version`):
- Git branch + commit SHA (`git log -1 --oneline`):
- torch version (`poetry run python -c "import torch; print(torch.__version__)"`):

## Plan / novelty reference (if known)

<!-- Which plan section or novelty item is affected? e.g. §6.12.2 /
     TemporalDecayBias (N3). Leave blank if you don't know. -->

## Already attempted

<!-- What have you already tried? Regenerating preprocessing? Clean reinstall? -->
