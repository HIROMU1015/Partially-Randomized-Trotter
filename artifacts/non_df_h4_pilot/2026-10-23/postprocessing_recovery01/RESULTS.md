# H4非DF/DF固定pilot：保存結果と技術検証

**保存証拠の検証は未受理。部分結果と失敗を保全。 mandatory STOP。**

ユーザーの一括実行承認と「検証結果をまとめてコミット」指示に基づく専用branchの記録。H4非DFは並行研究側であり、本チャット本来の研究・発表資料へ統合しない。

科学SOURCE：`0c329b30a81accb35c212be8defa66fee0dd0f12`。branch：`codex/non-df-h4-execution-foundation-20261023`。

条件：linear H4・隣接1.00 Å・STO-3G、共通未閾値MO積分、Pauli-JW/DF-native、DF tol-only1e-8/rank10、Pauli184項のPR prefix27、DF PR prefix1/10＋one-body。非DF Nalpha=Nbeta=2 sector36最低状態を共有、各表現自身のfull-H reference。S2/T=.8/q8/δ=.1、PR r4/K4/各256 paired trajectories。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし、ordinary-controlled full wrapper RZ、P=0。

| case | 保存状態 | paired本数 | cosine RZ平均 ± MC SE | sine RZ平均 ± MC SE |
|---|---|---:|---:|---:|
| pauli_jw_PF | COMPLETE_FIXED_CASE | 1 | 255500 ± 0 | 255500 ± 0 |
| pauli_jw_PR | COMPLETE_FIXED_CASE | 256 | 9321.1016 ± 13.8475 | 9321.1016 ± 13.8475 |
| df_native_PF | COMPLETE_FIXED_CASE | 1 | 135355 ± 0 | 135355 ± 0 |
| df_native_PR | COMPLETE_FIXED_CASE | 256 | 23678.441 ± 46.718 | 23678.441 ± 46.718 |

保存精度行：20、N/G null行：20。
ε=.05/.01/.005/.001/.0005の保存会計をそのまま転記する。精度不適格・数値証拠不足はN/G nullで、allowance・閾値・候補・標本数を変更していない。総費用Gが未定義の行について方式の総費用順位を主張しない。

PRのCは観測256本のfull-wrapper価格平均であり、MC SE/nominal Student-tは有限標本の診断。希少eventの未観測費用・母平均・一般的優位性は保証しない。event記録はproductionの実戻り値を観測し、追加抽選・seed replayを行わず、trajectory IDとcos/sin価格をhash joinした。

実行基盤の事前検査は既存371＋新規65＝436 PASS、FAIL/skip0、分子計算3件除外。これは事前の人工/回帰結果であり、今回の実H4保存証拠監査と区別する。保存データ監査はSOURCE/runtime・入力・数値gate・event/価格hash・固定予算・旧証拠保全を確認する。元のcase監視によるexit/reapと正常STOPは未確認。

自動後処理のGit範囲・STOP待ち・他作業保全guardは、人工temp repositoryの8検査でPASS。この後処理検査による科学callは0。詳細は`finalizer_checks.json`とlogに保存した。

保存物：`run01/`の原結果・SOURCE/承認/grant/CONSUMED・入力snapshot・実event/価格join・監視ledger/marker/STOP、本directoryの`verification.json`、`result_summary.json`、`precision_table.csv`、`result_inventory.json`。raw・旧SOURCE・旧grant・旧STOPは後から書き換えない。

科学retry・追加sampling・再compile・詳細科学レビュー・pushは行っていない。ローカルcommitの実SHAと成否はGit logおよび`/tmp/h4-recovery-859ce9f6.receipt.json`で確認する。

次の作業なし。この保存・ローカルcommitの範囲を終了しmandatory STOP。

検証未受理の理由：`Original supervisor completion/exit/reap evidence is missing; no native PASS.`。科学caseは再実行しない。

## 監視消失後の保存処理

元の監視・後処理はDF PRの161件前後で記録が途絶え、確認時に消失していた。原因は未特定。元のDF PR workerのみを継続観測し、新規科学計算は起動していない。元runのsummary/STOP/ledger/marker/claimは新造・上書きしていない。このdirectoryの保存データ検査は、元の入力・SOURCE・runtime・event/価格・予算・数値gate検査を維持するが、元監視のexit codeとreapを確認できないため、通常実行の完全受理PASSとは区別する。上記の保存物は元runに実在するもののみを指し、欠けたSTOPを存在すると扱わない。検査JSON・表・inventoryは本directory内。commit receiptは`/tmp/h4-recovery-859ce9f6.receipt.json`。
