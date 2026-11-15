# C・D2・統合担当へ：PR-DF LD7のPF予算4列のQ3

**4列ともQ3モデル不適合（下界>τ）。新規Q3は4件、判定保留は0件。LD7の確認済み最安はA_SHAPEの1,235,340,731.912698 RZのまま。** 正本は[REPORT](REPORT.md)。

| 順 | 予算・規則 | N_X = N_Y（m0〜m12） | 条件付き費用 RZ | m7〜m12割合 | 下界/τ | 判定 |
|---:|---|---|---:|---:|---:|---|
| 1 | p7 395,962,678・A_SHAPE | [15,14,13,12,11,10,9,8,7,6,5,4,2] | 352,435,885.500000 | 95.5767% | 337.1 | 不適合 |
| 2 | p7・B_PRICE | [64,60,56,52,47,43,38,34,26,10,4,1,1] | 390,742,769.682540 | 82.9922% | 13.62 | 不適合 |
| 3 | Full PF 559,457,386・A_SHAPE | [22,21,19,18,16,15,13,11,10,8,7,5,4] | 537,571,380.119048 | 95.7475% | 6.128 | 不適合 |
| 4 | Full PF・B_PRICE | [64,60,56,52,47,43,38,34,30,19,7,3,1] | 548,149,803.126984 | 87.8761% | 3.439 | 不適合 |

- 生成集合内に適合点は無い。不適合の結論はこの4列に限る（別の列・規則・振替は未評価）。
- LD7の確認済み最安1,235,340,731.912698 RZ（D2 `2b3b3bcc9`）は、p7の約3.12倍、Full PFの約2.21倍。
- 価格はLD7自身の探索用モデル（m7〜m12はq64より先の外挿、費用の83〜96%）。E[C]・CIではない。

## Cが取り込む機械可読レコード

固定commit **`7ccb3e63b687bd13426f3e53d96b93d918707641`**（branch `codex/b2-pr-df-ld7-pf-budget-20261116`）

| パス | 内容 |
|---|---|
| `docs/research/pr_df_ld7_pf_budget_q3_results_20261116/C_RESULT_JOIN.json` | 下記の欄を1つのレコードにまとめたもの |
| `docs/research/pr_df_ld7_pf_budget_q3_results_20261116/SUMMARY.json`／`SUMMARY.csv` | 4列の表（exact値と表示値）、停止理由、保留・未評価の一覧 |
| `docs/research/pr_df_ld7_pf_budget_q3_results_20261116/run01/NN/<label>/API_RETURN.json`（NN=01〜04） | 新規Q3の原API return（MSEの正本） |
| `docs/research/pr_df_ld7_pf_budget_q3_results_20261116/INPUTS.json` | 実行前に固定した4列・費用・順序・評価条件（科学実行SOURCE `cf19fc36ff76167fc2fe86d2a671c40918ea2c5f`） |

`C_RESULT_JOIN.json` に入っている欄：
- 共通：configuration `ce9c7f08…`、total configuration、binding、信号packet、target、価格参照 `4326422a4`・索引 `fd010b111`、τ（exact）、PF両基準
- 4列ごと：allocation、26軸価格行、exact費用・残額、m0〜m6／m7〜m12の内訳と割合、status、MSE上下界（exact）、結果参照、T−T_PF
- 集計：`best_confirmed`=null、`held`=[]、`LD7_confirmed_cheapest`（label `LD7_A_SHAPE`、cost 155652932221/126、`updated`=false、元Q3参照）

## 条件と資格

- 信号・評価契約はD2 `0defa286c` のLD7 V0と同一（Z1＋T1 total decoder、保存DF10 solver proxy、全事象平方loss、ideal finite-K raw IID、exact τ）。経路はD2のLD7 Q3と同じ（`run_pr_df_ld_split_q3_resume.py`、view `27e70efe5`、元AST guard、Python 3.11.0rc1）。
- 4件とも、閉包VERIFIED、遷移組が上限内、残余会計あり、既存の結果検査をPASS。原API・partials・資源をworkerが保存し、すべて終了code 0。
- モデル判定であり、十分N・物理的global RMSE・G_qualified・winnerではない。

## ロードマップへの差分（本体は統合担当が更新）

- PR-DF LD7：PF達成費用（p7 395,962,678／Full PF 559,457,386 RZ）を予算とするV0上限のA_SHAPE・B_PRICE 4列は、すべてモデル不適合。LD7の確認済み最安は1,235,340,731.912698 RZのまま（p7の約3.1倍、Full PFの約2.2倍）。
- 次の判断（ユーザー）：中間予算や別の配分族を評価するか、LD9など別のsplitの価格を待つか。

新規信号・sampling・compile・価格再fit・retryは0件。固定SOURCE・原結果・旧STOPを保全。公開・remote照合後 **mandatory STOP**。
