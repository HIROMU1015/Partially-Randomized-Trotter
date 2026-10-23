# Phase 2D-I：混在辞書prototype SOURCEの準備

2026-10-23 JST。状態 **SOURCE_IMPLEMENTATION_PREPARED_UNEXECUTED**。承認されたのは実装・静的検査・保全・公開のみ。これは実行時正しさ、F1 PASS、科学的GO、独立新規性、資源改善を意味しない。

分類は `KNOWN_METHOD_APPLICATION_CANDIDATE`、`INDEPENDENT_METHOD_DELTA_UNESTABLISHED`。元Phase 2Cのfragment別S/P辞書を実装し、R1反可換pair／unitary partitioningは追加しない。SS/SP/PS/PPを同一条件で扱う。

## SOURCEと技術記録

- [prototype package](../../../src/trottertracks/pr_dictionary_validation/__init__.py)：exact specification／serializer、exact係数代数、typed atoms／tail、既存finite-RTE接続、controlled router、共通sequence policy、独立参照、literal価格DP、interval scoreとselector。
- [F1専用runner](../../../scripts/run_pr_dictionary_f1.py)：標準libraryのauthorization検査後だけ科学moduleへ進む。P0/I・未binding・使用済みauthorizationを拒否する設計。
- [test定義](../../../tests/test_pr_dictionary_f1.py)：20関数。**TESTS_DEFINED / TESTS_NOT_EXECUTED / SCIENTIFIC_VALIDATION_NOT_PERFORMED**。

| 記録 | 内容 |
|---|---|
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | 実装責務と公開API |
| [SOURCE_CLOSURE_AND_VERSION_MAP.md](SOURCE_CLOSURE_AND_VERSION_MAP.md) | 起点・参照版差・限定port・依存関係 |
| [SEMANTICS_TRACEABILITY.md](SEMANTICS_TRACEABILITY.md) | PO-01–09、F1 gateとSOURCE/testの対応 |
| [TEST_PLAN_AND_DEFINITIONS.md](TEST_PLAN_AND_DEFINITIONS.md) | 定義済みtestと実行禁止・登録条件 |
| [INPUT_BINDING_HANDOFF.md](INPUT_BINDING_HANDOFF.md) | Bで固定すべきinput／件数／環境 |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | 新SOURCEのbytes/SHA256/blob、静的import graph |
| [REFERENCE_PROVENANCE.json](REFERENCE_PROVENANCE.json) | 固定資料・helperの確認範囲とidentity |
| [READINESS_AND_BLOCKERS.md](READINESS_AND_BLOCKERS.md) | Iの到達点と未検証項目 |
| [VERIFICATION.md](VERIFICATION.md) | 静的検査・公開照合の範囲 |
| [PUBLICATION.json](PUBLICATION.json) | 検証対象commitとfile receipt |
| [handoff.md](handoff.md) | ユーザー／GPT向け引継ぎ |

P0仕様の正本は[固定commit b8d54708](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b8d54708f90ccef275eda3757143100b2d0c677e/docs/research/pr_algorithm_improvement_phase2d_preparation_20261023)。このI branchは同commitから新規追加だけで作り、既存の科学SOURCE／入力／raw／契約／レビュー／grant／ledger／marker／STOP／原稿を変更しない。既存索引も保全し、新しい入口はこのREADMEへ集約する。

今回、新package／runner／testをimport・起動していない。pytest、fixture materialize、係数map／数値行列／状態／lambda/B/L／input hash生成、event／trajectory生成、sampling、Operator／mean／signal評価、build／transpile／compile、科学grant発行は全て0件。実装関数の存在と、その関数を呼んだ証拠は別である。

次の担当はユーザー。SOURCEと未検証項目を確認し、段階Bの限定入力bindingを別途明示承認するか判断する。B、F1、F2は未承認。公開確認後 **mandatory STOP**。
