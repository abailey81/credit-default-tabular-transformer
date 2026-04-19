"""Cross-cutting infrastructure.

One module at the moment: :mod:`src.infra.repro`, the reproducibility
gate. It regenerates every derivative artefact (RF predictions, the
evaluate comparison table, the split hash manifest) into a scratch
directory, diffs each against the committed copy, and exits non-zero on
any mismatch. CI wires it in as a post-merge check; the local invocation
(``python -m src.infra.repro``) is the same.

The reason this lives in ``infra`` rather than ``evaluation`` is that it
doesn't produce new results — it verifies that results haven't drifted.
Any future cross-cutting concerns (logging conventions, run-id plumbing,
environment snapshots) belong here too.
"""
