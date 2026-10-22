# 非DF基盤の堅牢化とH4準備（2026-10-23）

[技術報告](../../../docs/non_df_h4_readiness_validation.md)。H4科学実行・H2追加sampling/compile・科学再ランキング・詳細科学レビューは0。

- `authorization.md`：今回の技術作業・準備の承認全文。H4科学実行の承認ではない。
- `H4_execution_contract_DRAFT.json`：物理系・prefix決定規則・4候補・T/q/r/K・精度・MC・数値gate・SOURCE/一括承認・資源の未採用案。
- `rare_event_diagnosis.json/md`：H2保存法則/count/128価格を照合。Pauli event-to-cost joinは欠測、DFはorder0だけと集計から判定。
- `rare_event_acquisition_DRAFT.json`：未取得情報の条件付き診断案。新推定器・母平均保証は未採用、実行禁止。
- `technical/`：人工異常系・Qiskit同時preflight・8 qubit人工系・既存安全回帰、全ログ/JUnit/STOP。
- `technical/development/`：開発時の初期失敗・途中PASS・初期診断/人工証拠を保全し、最終検証と区別する。

SOURCEは技術改修のcommitを示す。旧H2のSOURCE・結果・図・STOPやgrantは変更しない。
最終ローカルcommit後mandatory STOP。push・H4科学計算には別の明示承認が必要。
