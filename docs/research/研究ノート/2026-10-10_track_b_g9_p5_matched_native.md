# 2026-10-10 Track B G9

採用GPT G8 review §14に従い別worktreeで数学/source準備。
P5独立first-adjacent oracleで31 parents/63 events/10 groups。任意Lの群数とDPを独立導出。
指定providerのexact Pauli取得、direct phase/adjointとhelperをoff-angleで確認。23 focused tests PASS。
19 keys/既存identity再利用18/CTS新規1、6 primary directと5 helper診断、22軸の新failure配分を結果前固定。
初期arity/import/inventory estimateはsource/marker/取得前修正。原科学結果/STOPは不変。
source固定後一束で完了またはtechnical STOPし、GPTへ戻す。

## One-shot終了・STOP

source `d6ecc6bc82c11158a66d175d78007a121d0fac46` をpush・clean SHA照合後一回呼出し。
`G9_TECHNICAL_INCONCLUSIVE`：`TypeError: cannot create mpf from Fraction(1, 1000000)`。
new helper attempt1、backend呼出し前の引数評価で停止、新規sequence0、native row0。
marker `80d6f42056f8aadb8ce15dafd3eee21c37ca0857419c28e41015d119339b2dd9` は消費済み、retry0。source修正・再実行なし。
23 saved-output checksで過去941path・critical69不変。
[結果/GPT handoff](../../tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)へ戻しmandatory STOP。
