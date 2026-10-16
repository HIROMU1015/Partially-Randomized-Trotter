# H8 Phase A：実装と人工検証、科学未実行

[実装・入力来歴・制限・実行契約](../../../../docs/research/track_a_h8_phase_a_implementation_readiness_20261016.md)。

- [固定Phase A planと1,650 probeの時間台帳](phase_A_plan_UNAPPROVED_v1.json)
- [SOURCE/input/provider/profile結合](metadata_binding_v1.json)
- [人工検証](technical_validation_v1.json)／[保全監査](preservation_receipt_v1.json)
- [公開file bytes/SHA](publication_files_v1.json)

repo rootで、科学importなしのplan表示と人工guard検査：

```bash
python3 -P -B scripts/run_h8_strengthened_phase_a.py
python3 -P -B -m unittest discover -s tests -p test_h8_strengthened_phase_a.py
```

`tests/check_h8_strengthened_synthetic_provider.py --provider <fixed-provider-checkout>`は固定科学venvで2 qubit fixtureだけを検査する。実分子入力やcompileは使わない。

private config・grant・fresh gateの起動コマンドは公開物へ含めない。SOURCEはこの実装を含む公開commitへ結合し、実行には別の明示承認が必要。旧V1/V2/H5/H7のgrant/ledger、runtime、分子配列、credentialを収録しない。mandatory STOP。
