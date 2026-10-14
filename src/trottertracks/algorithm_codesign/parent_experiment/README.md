> [v2 owner PID修正準備](../../../../docs/tracks/algorithm_codesign/parent_allocation_v2_owner_repair_source_preparation_20261015.md)。現在launchはv2 contract/namespace、authority pending。v1結果・旧source blobsは保護。

# Parent allocation fixed experiment source

採択済み科学条件は[GPT原本](../../../../docs/research/track_b_parent_allocation_scientific_design_review_20261014.md)。
[技術準備報告](../../../../docs/tracks/algorithm_codesign/parent_allocation_execution_source_preparation_20261015.md)と
[contract](../../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/execution_contract_v1.json)を読む。

- `design.py`: 登録値を持たない有限候補、global proposal、exact sufficient ledger。
- `launch.py`: source/direct-child/remote/namespace/hash/ownershipの入口。
- `pipeline.py`: 承認後だけ旧guard→候補freeze→新価格→native/ledgerへ進む。
- `worker.py`: 予約済み一key、一backend entry、strict guard、CPU/RSS失敗保存。
- `runner.py`: outer observer、isolated ledger、complete/inconclusive・STOP。

Preparation only。zero-compute preflight以外の登録処理は別明示承認が必要。
候補を新価格で再選択しない。科学的採否はこのpackageの出力ではない。
