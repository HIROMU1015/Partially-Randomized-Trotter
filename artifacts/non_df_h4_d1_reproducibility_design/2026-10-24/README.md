# H4 D1 SOURCE再現性の限定調査・設計

2026-10-24 JST。[調査報告](../../../docs/non_df_h4_d1_reproducibility_design_20261024.md)。
旧hashは3一致／Pauli-PR不一致を維持。実際の不一致の排他的原因はUNKNOWN。
元SOURCEと保存入力・高水準命令の対応から、SOURCE基準を使う研究用GO候補を整理した。
基準採用・実行承認・dispatchはfalse、新grantなし。完了後mandatory STOP。

| 資料 | 意味 |
|---|---|
| [REQUEST.txt](REQUEST.txt) | 今回の許可・禁止・完了条件の原文 |
| [inspection.json](inspection.json) | SOURCE/runtime/event検査、4方式の独立A/B status、今回0 call |
| [correspondence.json](correspondence.json) | 12辺のSOURCE対応・根拠・限界（author assessment） |
| [source_locations.json](source_locations.json) | 参照関数の行番号とfile SHA、固定runtime SOURCE SHA |
| [saved_QPY_structure.json](saved_QPY_structure.json) | 保存命令token一致、制御/位相/型、再合成禁止下の読取結果 |
| [bounded_recovery.json](bounded_recovery.json) | 旧71fileに限定した回収範囲、payload未回収 |
| [preservation_before.json](preservation_before.json) | 旧611＋D1 code/input/固定subtree、725file SHA |
| [contract_DRAFT.json](contract_DRAFT.json) | 未実行・未採用・無grantの新契約案、将来最大9compile |
| [contract_verification.json](contract_verification.json) | 9対象・固定条件・phase/式継承と無効flagの構造検査 |
| [artificial/result.json](artificial/result.json) | 既存8＋新9＝17人工検査PASS、H4同等性への外挿なし |
| [github_verification.json](github_verification.json) | 基点GitHub取得と固定文書blob照合 |
| [completion.json](completion.json) | 今回調査だけの保全/終了、旧native証拠の代用ではない |
| [publication_status.json](publication_status.json) | commit/push/remote照合の結果 |

再利用：旧`h4_d1_adapter`、fingerprint payload、RTE/Pauli append/wrapper、
既存人工fixtures/tests、元9構成とphase所有/非加法性式。旧SOURCEは変更していない。
新moduleはsource link欠測・意味差のSTOPと表示名候補を整理する設計補助で、実行を許可しない。

実行済みの人工検査コマンド（実H4は読まない）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B tests/test_h4_d1_reproducibility.py artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/artificial/result.json
```

`inspect_saved_evidence.py`は固定基点上の未commit設計から実行した読取調査script。
既存H4の数値・回路構築・費用処理を呼ばず、QPYでlazy定義へ触れない。
今回の現在hashを再計算して旧hashと比較する処理は実行していない。

次はユーザーがSOURCE基準Bと有限D1を承認するか判断する。
Codexはその後だけ新controller接続・SOURCE固定・新grant・Stage A/B/Cへ進む。
