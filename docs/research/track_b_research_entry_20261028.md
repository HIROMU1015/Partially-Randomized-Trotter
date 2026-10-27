# Track B development entry (P4, 2026-10-28)

The independent development branch is based on G6 draft tip
`5f750a23f023d75c8c4fbcabc6937d6953ce61f2`. The G6 draft and both M1 formal-adoption branches remain fixed.
Use `src/trottertracks/algorithm_codesign/research_entry.py` or
`PYTHONPATH=src python scripts/tracks/algorithm_codesign/research_entry.py --study G6 --stage reference`.
Select exactly one of `G6`, `M1-S-gamma`, `M1-S-pi3`; no multi-study scientific runner exists.
`--stage mock --output NEW_DIRECTORY` tests artificial descriptor persistence.

The existing G6 `Evidence.create/read` serializer and comparison `storage.dump_json/load_json`
are reused unchanged for technical records. This does not apply G6 mathematical
guarantees to M1. G6 generator, construction, native inventory and fixed handoff
remain separate reference APIs; none are executed or reimplemented.

Only the original package initializers and pure `spec.py` for each M1 study were
selected into the same checkout (four files total, original blobs unchanged).
Their dependency closure is stdlib `itertools`. No external-worktree imports,
alias loaders, copied numerical engines or execution-control modules are used.
Specs retain their candidate flags. `describe` separately names the fixed adopted
contract and its source; adoption is not scientific execution permission.

The gamma continuous-domain saved-signal conditions and pi/3 interval intersection
retain distinct input identities, forecasts, guarantees, one-shot/retry and STOP
locators. Upstream result inputs are labelled inputs, not results of this new entry.
M1 numerical/interval execution, G6 generation, optimization, native compile and
cross-algorithm mathematical equivalence are DEFERRED and untested scientifically.
The common entry supports reference/descriptor/artificial mock only.

Technical validation: 30 PASS, 0 FAIL, 0 SKIP. Tests cover isolated study bindings,
schema/import, pure descriptor serialization, existing storage round-trip and
collision refusal. They do not establish a scientific guarantee or GO.
