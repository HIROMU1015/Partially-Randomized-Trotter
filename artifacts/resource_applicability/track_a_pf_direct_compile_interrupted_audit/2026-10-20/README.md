# PF直接compile：中断後保存監査

[結果報告](../../../../docs/research/track_a_pf_direct_compile_interrupted_audit_20261020.md)。

- [中断・件数・資源・限界](interruption_audit.json)
- [788軸の状態](axis_status.csv)
- [task単位の保存受入572軸の整数metrics](accepted_gate_counts.csv) / [詳細](accepted_measurements.json)
- [元契約・grant digest・call監査・中断後byte gate](execution_contracts_and_call_audit.json)
- [worker/control/raw/ledger/exit証拠](worker_evidence.json.gz)
- [固定依存hash](bound_dependency_hashes.json.gz) / [資源観測](resource_observations.jsonl.gz)
- [配列shapeのSTOP理由](loader_failure_diagnosis.json)
- [scalar入れ子形式のSTOP理由](scalar_schema_diagnosis.json)
- [旧packet保全](prior_packet_preservation.json) / [今回元run保全](original_run_preservation.json)
- [公開範囲](publication_scope.json) / [hash](artifact_manifest.json)
- [STOP](MANDATORY_STOP.json)

572軸はtask単位の保存受入。元のrun terminal/最終gateは欠測、8 pair/16軸はnative終了未確認。中断後byte照合を元最終gateへ読み替えず、raw cosine2件も受入へ昇格しない。

新規科学実行・retry・fit・shot再計算・費用再順位0。**mandatory STOP。**
