# Phase 2D-B-PRE：限定入力binding契約の準備

2026-10-23 JST。**契約・B専用SOURCE・静的監査・公開のみ。科学実行の許可ではない。** 主状態は `B_ENVIRONMENT_UNRESOLVED`。11 logical casesの契約とSOURCEは準備したが、専用環境のwheel/package/native closureが未固定で、`READY_FOR_B_EXECUTION_APPROVAL` ではない。

目的は同じphysical HのPauli／frame辞書とfinite lawをbindingし、将来F1に渡すinput identityと件数を得ること。分類はKNOWN_METHOD_APPLICATION_CANDIDATE、独立新規性は未確立。R1反可換pair案を追加しない。

| 成果物 | 内容 |
|---|---|
| [B_EXECUTION_CONTRACT.md](B_EXECUTION_CONTRACT.md) | 2 INPUT＋4 ARMS＋5 CONTEXTS、精度、出力、STOP |
| [B_CALL_INVENTORY.json](B_CALL_INVENTORY.json) | 固定ID/DAG、logical件数と実関数呼出し |
| [B_INPUT_SPECIFICATION.md](B_INPUT_SPECIFICATION.md) | P0を変更しないexact fixture／catalog／時間 |
| [B_SOURCE_AND_ENVIRONMENT.md](B_SOURCE_AND_ENVIRONMENT.md) | active closure、専用環境、動的importの境界 |
| [B_PRICE_AND_INTERVAL_READINESS.md](B_PRICE_AND_INTERVAL_READINESS.md) | buildなしの境界、価格・interval未資格 |
| [B_OUTPUT_AND_IDENTITY_SCHEMA.md](B_OUTPUT_AND_IDENTITY_SCHEMA.md) | 層別identity、serialization、出力 |
| [B_AUTHORIZATION_AND_STOP.md](B_AUTHORIZATION_AND_STOP.md) | B専用permission gateとone-shot |
| [F1_HANDOFF_AFTER_B.md](F1_HANDOFF_AFTER_B.md) | F1/F2の件数と残る承認 |
| [READINESS_AND_BLOCKERS.md](READINESS_AND_BLOCKERS.md) | 実行承認前の解消条件 |
| [SOURCE_AND_PROVENANCE_INDEX.json](SOURCE_AND_PROVENANCE_INDEX.json) | 固定資料と確認範囲 |
| [SOURCE_CLOSURE.json](SOURCE_CLOSURE.json) | active SOURCE bytes/hash/blob、未使用経路 |
| [ENVIRONMENT_PLAN.json](ENVIRONMENT_PLAN.json) | 検証済みlockと区別した環境固定案 |
| [B_SCHEMAS.json](B_SCHEMAS.json) | schemaのみ。科学input／grantではない |
| [VERIFICATION.md](VERIFICATION.md) | 静的検査と公開照合 |
| [PUBLICATION.json](PUBLICATION.json) | 検証済み公開payloadとreceipt |
| [handoff.md](handoff.md) | ユーザー／GPT向け引継ぎ |

新規SOURCEは[B専用package](../../../src/trottertracks/pr_dictionary_binding/__init__.py)と[runner](../../../scripts/run_pr_dictionary_binding.py)。IのF1 runnerを拡張せず、既存SOURCE／P0／I資料を変更しない。B-PREでimport／起動／pytestを行っていない。

固定起点は[I最終bfaab383](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/bfaab3831042f64a7112296437c0001e34920209/docs/research/pr_algorithm_improvement_phase2d_implementation_20261023)、仕様は[P0固定b8d54708](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b8d54708f90ccef275eda3757143100b2d0c677e/docs/research/pr_algorithm_improvement_phase2d_preparation_20261023)。必須19資料を固定GitHubから取得する。

科学input、係数map、V/Gamma行列、状態、lambda/B/L、law、event、trajectory、Operator/mean/signal、build/compile、price/shot数、grant／consumed markerは全て未生成。次の担当はユーザー。専用環境の固定とB科学実行を明示承認するか判断する。公開確認後mandatory STOP。
