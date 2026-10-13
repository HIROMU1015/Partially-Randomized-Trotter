# H4 B0+I parallel run02 — approved launch

SOURCE 924b779c2bc6f6e844d6283c355f91c73aed6445;12 workers,32GiB each, driver8GiB soft limit with inherited hard limit preserved for workers, observer AS256MiB/RSS64MiB.64GiB shared headroom,456.25GiB admission. Fixed240 wrappers,234 new compiles and6 fresh numeric-verified pilot reuses, one internal thread/Qiskit process per worker.

Run01 stopped before any compile/reservation because the driver8GiB hard AS limit was inherited. Worker/observer/driver exits and cleanup are recorded in the sibling run02-stop-evidence JSON.65536 bytes and0.7697398792952299s added to carry; no invocation refund/reset. Old failed output/lock/source kept. Run02 corrects only driver hard-limit propagation and carries the failure costs.25 limited stdlib/mock tests PASS, including the inherited-limits regression. No artificial compile or deep repeated scientific review.

The current user explicitly authorized parallel speed, memory-limit changes and starting this computation; normal implementation/repair proceeded without repeated user approvals. Scientific source48 unchanged; new closure59, environment/compiler/frozen6 inputs fixed. CPU roles are in authorization.json; fresh sibling load and small fixed job masks checked, broad masks are not reservations. No H8/shared environment/venv/GPU changes.

Source/plan/auth/review/binding/user approval are fixed here; fresh gate repeats before exclusive one-shot lock. Grouped heavy-first scheduling, bounded nonblocking result collection, owned pidfd cleanup,5s independent observer. Stage4h/task10min/output256MiB/observer log160MiB/inode1024, actual74805/charge17GiB retained. Completion or fail-closed STOP terminates the run, no runner auto-retry/next stage.
