# Local0 absolute-kappa scope entry

The normal local0 search command accepts PR_PAULI absolute min/max through an adopted
`absolute_kappa_route` service entry. It uses `run_rpe_schedule_search.py` and the existing
`SCHEDULE_WIDE_V1` physical schedule chooser. The interval is closed at both ends;
representatives are constrained without changing the physical schedule mapping.

`method_overrides.PR_PAULI.absolute_kappa_bounds` constrains saved, generated and
newly installed optimizer candidates. `search_overrides.schedule_wide.kappa_bounds`
constrains the outer chooser. The existing inner shot optimizer and fixed evaluation
contract remain in use. Absolute kappa is never interpreted as a multiplier.

The operational service derives omitted widths from adopted delta/h values and a
source-bound mapper's fixed h. It retains exact/binary64 representations, uses the
previously generated 349 worker profiles, concurrent execution and auto Q3 capacity.
Existing request-specific host eligibility remains authoritative; a new request is
not granted the PASS of a different request.

PlanOnly saves the exact incoming scope, immutable effective PROBLEM/ROUND, native
command and physical schedule state. It calls the ordinary CLI with --plan-only,
without --prepare-readiness. No Completion, Executor, worker, controller or dry-check
is instantiated in this bounded planning mode. Missing family preparation remains
explicit. The same scope/evaluation/source reuses the run; changed bounds create a
different run. PlanOnly never changes CURRENT_TARGET or grants execution.

Implementation: src/trottertracks/rpe_local0_search.py,
src/trottertracks/rpe_local0_scope_plan.py, src/trottertracks/rpe_schedule_search.py,
src/trottertracks/rpe_method_optimizer.py, scripts/run_rpe_schedule_search.py.
Validation: tests/test_local0_absolute_kappa.py, tests/test_rpe_local0_search.py,
scripts/check_rpe_schedule_search.py. Acceptance is metadata/synthetic only;
no real H4 Q3, signal generation, compilation, production queue or relay startup.
