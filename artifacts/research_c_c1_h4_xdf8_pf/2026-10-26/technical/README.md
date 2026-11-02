# H4 XDF8 technical evidence only

No real-H4 numerical decode or scientific evaluation was performed. No scientific run01 exists in this family. Separate explicit chat permission is required for the frozen 93-unit contract.

- `SOURCE.json`: new entry/tests and reused scientific SOURCE, 21 file hashes; preparation base e3c8b3e6550b2866b6e370a13780a0ee41ebefc8.
- `validation.json`: passive CLI byte/version preflight; NumPy not imported; original input and saved records unchanged; environment and test totals.
- `artificial_final.txt/xml`: 22 new artificial tests PASS, including real artificial end-to-end routes with 26 cells/93 logical units and independent dense expm comparison.
- `regression.txt/xml`: 92 existing synthetic regression tests PASS (prefix29, H6 saved PF21, H2/H6 14, feasibility16, deterministic accuracy12).
- `artificial_initial.txt/xml.gz`: first development run, 8 FAIL/11 PASS (complex zero dtype).
- `artificial_verified.txt/xml.gz`: second development run, 5 FAIL/14 PASS (missing OF-to-Qiskit conversion). Both failure records are preserved losslessly; the final tests include additional checks.

Tests used the existing read-only `/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, task-local XDG/Matplotlib caches, and process-local OPENBLAS/OMP threads1 for the small artificial matrices. No shared environment/cache edits. Scientific runtime chooses resources with the reused resource_plan, without arbitrary CPU/RAM/worker/wall caps.

```bash
python -m pytest -q -p no:cacheprovider \
  tests/management/test_research_c_c1_h4_xdf8_pf.py
python -m pytest -q -p no:cacheprovider \
  tests/management/test_research_c_c1_h4_xdf_prefix_budget.py \
  tests/management/test_research_c_c1_h6_saved_pf.py \
  tests/management/test_research_c_c1_h2_h6.py \
  tests/management/test_research_c_c1_stage3_feasibility.py \
  tests/management/test_research_c_c1_stage3_deterministic_accuracy.py
```

[Technical report](../../../../docs/research/research_c_c1_h4_xdf8_requalification_preparation_20261026.md) / [unapproved contract](../../../../docs/research/research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json). All numerical molecular outcomes remain unqualified; mandatory STOP.
