# H2未完了2caseの別契約取得（2026-10-23 JST）

[技術報告](../../../docs/non_df_h2_process_recovery_validation.md)を参照。
旧 `artifacts/non_df_h2_pilot/2026-10-22/` は読み取り専用。新SCF/FCI/DF/state solveは行わない。

- `technical/`：人工thread再現、同負荷4process、4case逐次対並行preflight、安全な回帰、input/hash/resource照合。
- `execution_contract.json` / `authorization.md`：実行前に固定した新契約・今回の利用者承認全文。
- `SOURCE.json` / `grant.json`：clean commitと新grantの固定。旧消費済みgrantは再利用しない。
- `run02/`：承認されたPauli PR/DF PF各one-shotの独立process。marker・ledger・ログ・結果・STOP。
- `comparison.json` / `comparison.md`：成功済みPauli PF/DF PRを再計算せずhash結合する4方式・5精度表。

経験的数値allowance下のHoeffding十分N。PR C/Gは64 paired MC平均、名目95%は厳密母平均保証ではない。
同じL_Dが同じ分割を意味しない。H2固定条件のみ。詳細科学レビュー・H4以降・pushなし。完了後mandatory STOP。
