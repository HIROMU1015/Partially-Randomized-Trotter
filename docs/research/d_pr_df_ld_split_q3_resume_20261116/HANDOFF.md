# 統合担当／C向けHANDOFF：PR-DF LD1/LD2の固定Q3 4件完了

**状態: FINITE_SIX_ENTRY_COMPARISON_COMPLETE。** 新規4 callを各1回完了し全てモデル適合。保存V0 2件を再利用し、各LDでV0/A_SHAPE/B_PRICEの3件を比較した。[REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[MENU.csv](MENU.csv)。

| LD | 確認した3配分内の最小 | 実モデル費用 RZ | 選択X/Y列 m0..12 | m7..m12占比 |
|---|---|---:|---|---:|
| 1 | B_PRICE | 189,232,849,610.910739 | X/Y各 `[64,60,56,52,47,43,38,34,30,25,21,17,7]` | 99.8948493697037429356922% |
| 2 | B_PRICE | 46,341,978,336.212733 | X/Y各 `[64,60,56,52,47,43,38,34,30,25,21,17,7]` | 99.8570983645607075586689% |

各選択配分のMSE exact上下界、6行の判定・費用、固定SOURCE/locatorはSUMMARY。全て採択済み閾値（約2.539561285063813282041692×10⁻⁶ Ha²）以内。モデル費用は予算額を代用せず4ef70274の実額を無変更再利用した。

## 固定locator

- 新規raw: `75f416de6623a21a3b705e3f770a7c148325cb20` / `docs/research/d_pr_df_ld_split_q3_resume_20261116/run01/<LD1_A_SHAPE|LD1_B_PRICE|LD2_A_SHAPE|LD2_B_PRICE>/RESULT.json`。
- 原API: 同commit/同directoryの`API_RETURN.json`、selector `[0]`。RESULTのselector `record`と4件とも一致。closure/lemma/DP・資源・PERSISTEDも同directory。
- 実行code: `005dc8c019061c07ba2ac8da389af0117f7aa27b` / `scripts/run_pr_df_ld_split_q3_resume.py`。公開commitと区別する。
- 再開INPUTS: `edf78cf5d43f707c59d7931d271a8d78163cac12` / `docs/research/d_pr_df_ld_split_q3_resume_20261116/INPUTS.json`。元固定入力4d3aa8a6と価格/配分4ef70274を直接保存し、全4 caseの一致を検査。
- 集計code: `0c0395609a48b268b37c2f02412ff7188177e15a` / `scripts/summarize_pr_df_ld_split_q3_resume.py`。価格/配分の再生成・fitは行わず、保存MSEと費用を結び付けた。
- 前回STOP: `e0b3cad2a024070389f757160f3a20d7da02c5ff` / `docs/research/d_pr_df_ld_split_results_20261115/q3_calls/COMPLETION.json`。旧記録は書き換えていない。

## 実行環境と検証

B2成功Q3 package738df226の検証記録でPython3.11.0rc1と既存venv `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`を特定。今回は同executableのversion/platformを実INPUTS/EXECUTIONへ保存し、固定AST hash3d043e71…を元guardのまま通過した。kernel bytes/accelerator/期待hash/照合規則の変更0、共有環境変更0。[ENVIRONMENT](ENVIRONMENT.json)参照。過去のscientific RESULTにversion別欄はなく、過去環境の根拠は同成功package内TESTS.txtである。

本番と同supervisor・workerから、実constructorと人工rawのT.evaluateを通す子process保存検証がPASS。metadata intake以外は実経路。初回の人工fixture参照欄不足は補い、初回logを保全して2件PASS。科学call前にSOURCEとINPUTSをcommitした。前回以降の同候補/配分の同等receipt・jobは0で、新規は未開始4件のみ。

4件ともclosure VERIFIED、MSE上下界あり、API/RESULT/資源保存済み、outer worker exit0、errors/monitor_errors0。全体wall232.919794秒。個別値とprocess-tree RSSの観測範囲はREPORT/SUMMARYに保存。新sampling・回路build/compile・再fit・価格再予測・配分再生成・追加Q3・科学retryは0。

## 比較の資格とロードマップ

同じ精度条件の固定3配分内では、各LDが選んだB_PRICEの探索モデル費用はLD2の方が小さい。LD1はE_I[S]＋残差モデル、LD2は自身のfull-wrapper 3点モデルで根拠が違い、両者ともほぼ全額をm7..12が占める。Q3の適合は保存ideal finite-RTE raw/IID・solver proxy・total decoder条件のモデルMSE判定であり、長roundの価格検証ではない。全law E[C]/CI・実PCG64移送・正式G・方式winner・全配分/全L_D最適性は未認定。

統合ロードマップの「4 call未開始／AST待ち」を完了へ更新できる。次の研究優先は前回同様LD2長round価格の検査だが、今回その新規取得は行わない。LD3や別配分を増やす判断も別途。旧m4独立holdout・原価格・凍結モデル・raw・旧STOPを保全し、ロードマップ本体は統合担当へ渡す。

commit/push・remote照合後mandatory STOP。
