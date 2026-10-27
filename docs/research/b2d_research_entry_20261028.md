# B2-D development entry (P3, 2026-10-28)

Use `trottertracks.b2d_evaluator.research_entry` and
`PYTHONPATH=src python scripts/run_b2d_research_entry.py --stage B1 --mode plan`.
`--mode mock --output NEW_DIRECTORY` transports artificial descriptors through
the inherited `evaluation.evaluate`, `Graph`, `Node`, `Settings` and
`b2d_adapter.actual_adjoint`. It does not run a scientific stage planner.

The new non-frozen branch is based on final-pause commit
`e87375fa9a998dce80792be0f15597c747f00d83`; that source branch remains fixed.
All seven stages have separate source/result/STOP locators in `STAGES`:
A0 reference, B0 native, B1 IID, Outer, Exact-R, Full-DF and Full-DF-q4/Yoshida4.
The existing input recipe, backend and stage modules are inherited unchanged;
they are referenced, not copied into another evaluator.

Signed append order, actual adjoint, processor, stage law and exact three-call
artificial transport budget survive into the result record. The small signed
fractions are artificial tokens, not a S2/Yoshida4 coefficient validation.
There is no operator, reference, sampling, native compile, shot or G calculation.
Unknown prices, u_num, gamma and rare-event qualifications stay null.

Scientific execution binding is DEFERRED: the real stage-specific graphs, input
qualification and existing execution contracts require separate scientific
authorization. This entry offers no real/resume mode, old-runner invocation or
callback loader. Plan/mock success is not numerical equivalence or scientific GO.
The fixed SOURCE/result/STOP, one-shot and retry rules remain in force.

Technical validation: 32 PASS, 0 FAIL, 0 SKIP in the new artificial entry tests.
This count covers routing, context refusal, DAG delegation, adjoint/order,
output collisions and clean import. It is not historical scientific evidence.
