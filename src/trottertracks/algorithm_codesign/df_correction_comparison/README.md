# DF cubic-correction comparison source v2

Whole commutator A vs coin-first Pauli B, under a fixed reviewed contract.
Registered recipes are reachable only after source-bound one-shot authorization/marker.
Preparation: off-domain controls only; no CD/E result or synthesis. Shared APIs and old
evidence stay unchanged. Contract and source plan live in `docs/tracks/algorithm_codesign/`
as `df_correction_comparison_*_v2_20261011.md`.

`certification.py` defines exact ideal Gram-Schmidt frames and numerical assembly balls;
`digital.py` defines rational executed-law accounting; `angles.py` preserves strict phase;
`native.py` handles Pauli IR, partial T multiplicity and coverage; `guard.py` is external
Linux process supervision. Matrices/large norms are immutable/cached; fixed inputs are
shared across two native jobs, without worker reseeding. No GPU or cache of prior T science.

The future runner is `scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py`.
Source review plus a separate authorization-only direct child and explicit run instruction
are required. Every terminal result has mandatory STOP.
