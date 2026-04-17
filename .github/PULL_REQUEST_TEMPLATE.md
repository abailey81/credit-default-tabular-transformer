<!--
Thanks for opening a PR. The checklist below codifies the lessons from
PR #8 review — follow it before requesting review to minimise churn.
Repo policy highlights:
  • No `Co-Authored-By: Claude` / AI trailers on commits.
  • One concern per PR — no bundled formatter passes.
  • Rebase onto latest `additional`; no merge commits on feature branches.
  • Update CHANGELOG.md under `[Unreleased]` as part of the PR.
-->

## What this PR does

<!-- One paragraph: the intent, not the diff. Reference the PROJECT_PLAN.md
section(s) this PR implements / closes, e.g. "Phase 6 §8 — src/train.py". -->

## Plan / novelty / ablation coverage

<!-- Tag which items this PR advances:
     - Plan phase(s): e.g. §6.7, §8
     - Novelty item(s): e.g. N4
     - Ablation(s) unlocked: e.g. A15 becomes runnable
     Leave blank if purely infrastructure / docs. -->

## Blocker checklist (all must be ✅ before merge)

- [ ] **No `Co-Authored-By: Claude` / AI trailer** in any commit on this
      branch (`git log --format=%B origin/<your-branch>` must be clean).
- [ ] **Rebased onto latest `additional`** (or `main` if targeting that).
      No merge commits on the feature branch.
- [ ] **One concern per PR**. No bundled formatter-only diffs; no
      unrelated refactors.
- [ ] **`poetry run pytest tests/ -q` passes** on the branch.
- [ ] **Relevant module smoke tests pass** if the PR touches them
      (`python src/<module>.py`).
- [ ] **`CHANGELOG.md` updated** under `[Unreleased]` with a bullet
      attributing the change.
- [ ] **`PROJECT_PLAN.md` status markers updated** if this PR changes the
      completion state of any plan section.
- [ ] **`README.md` roadmap updated** if user-facing state changes.
- [ ] **New tests added** proportionate to new code (shape + gradient +
      semantic correctness for any new `nn.Module`).

## Risk / blast-radius

<!-- What could go wrong? What's the rollback path? Is this change
     destructive (force-push, schema migration, dependency pin)? -->

## Verification commands

<!-- Paste the exact commands a reviewer should run to reproduce your
     local verification. Example:
         poetry run pytest tests/ -q
         PYTHONIOENCODING=utf-8 poetry run python src/transformer.py
         poetry run python run_pipeline.py --rf-benchmark --source local
-->

## Notes for the reviewer

<!-- Anything non-obvious: intentional design choices vs alternatives
     considered, follow-up PRs planned after this one, known limitations. -->
