# H4/H6 high-order PF B_ACQUIRE run01

[結果報告](../../../../docs/research/track_a_h46_higher_pf_acquire_run01_results_20261018.md)。状態 `COMPLETE_ACQUISITION_INTEGRATION_READY_NOT_RUN`。

- `summary.json` / `native_exit.json` / `completion_witness.json`：取得数・control・終了/回収。
- `shards/` / `native_cost_records.csv`：paired軸の実測RZ/CX、fingerprint、event digest。raw private eventは含めない。重み費用・SE・順位を計算していない。
- `acquisition_plan.json` / `source_input_runtime_seal.json` / `seed_audit.json` / `input_binding.json`：固定契約・SOURCE/input/runtime・seed衝突照合・rawと公開projectionのSHA。
- `grant_consumption_receipt.json`：消費済み証拠のみ。有効grantは非公開。
- `artificial_tests.json`：新旧42人工/mock PASS。
- `integrated_plan_UNAPPROVED_projection.json` / `integrated_grant_INVALID.json` / `integrated_draft_refusal.json`：統合比較は未承認、privateパスは投影。

SOURCE `d77e439a16d6953cb17e6d1ef91b0d9430144848`。最適q・母平均優位・厳密biasを認定しない。mandatory STOP。
