# C・統合担当へ：Full PF/S2・16h/7のm12=1固定Q3

**1shotはモデル不適合（下界>τ、新規Q3は1件だけ）。Full PF/S2・16h/7の確認済み達成費用は559,457,386 RZ（m12各2shot）を維持する。** 他roundを固定し、m12をX/Y同数・各軸1shot以上とする範囲では、2shotが最安の適合点。正本は[REPORT](REPORT.md)。

| m12のX/Y各軸shot | 条件付き費用 RZ | MSE区間 Ha²（表示） | 上界/τ | 判定 |
|---:|---:|---|---:|---|
| 4 | 615,304,502 | [1.495783385949356167254620E−6, 1.495783385949356167254624E−6] | 0.589 | 保存適合・`b4296b110` |
| 2 | 559,457,386 | [2.402471387794261100742341E−6, 2.402471387794261100742346E−6] | 0.946 | 保存適合・`a006b8e15`（**達成費用**） |
| **1** | **531,533,828** | **[4.706720749798782654111633E−6, 4.706720749798782654111638E−6]** | **1.853** | **新規・モデル不適合** |

- exact τ=(0.00159360010199040 Ha)²と原APIのexact有理数で判定した。1shotの下界はτの約1.853倍で、判定保留ではない。
- 1shotは2shotより27,923,558 RZ低いが適合しない。不適合はこの1列についての結論で、m11以前の変更・X/Y非同数・振替は未評価。
- p7（`cbc471c91`）も同じくm12=1で不適合（上界4.726×10⁻⁶）だった。別候補なので直接の比較ではない。

## Cが取り込む機械可読レコード

固定commit **`7349f6f9b1dad3e67a77ffe948bfbf347b5c23a7`**（branch `codex/b2-full-pf-s2-m12-two-20261116`）

| パス | 内容 |
|---|---|
| `docs/research/full_pf_s2_m12_one_results_20261116/C_RESULT_JOIN.json` | configuration（元 `11cf8ca6…`・total view `af18d37d…`・binding `69cc82b4…`）、信号packet参照、target、1shot allocation（`FULL_PF_S2_16H_OVER7_M12_ONE_01`）と親（2shot `FULL_PF_S2_16H_OVER7_M12_TWO_01`）、26軸の価格行、費用・差額、Q3のstatus・MSE上下界・τ（exact）、結果参照、4shot・2shotの参照結果、判定の扱い |
| `docs/research/full_pf_s2_m12_one_results_20261116/EXACT_COMPARISON.json` | 4・2・1shotの3行（exact値、各原API参照） |
| `docs/research/full_pf_s2_m12_one_results_20261116/run01/FULL_PF_S2_M12_ONE_FIXED/API_RETURN.json` | 新規Q3の原API return（MSEの正本） |
| `docs/research/full_pf_s2_m12_one_results_20261116/INPUTS.json` | 実行前に固定した入力（科学実行SOURCE `d6366b03d278836139107e311d45aa82b380140e`） |

`C_RESULT_JOIN.json` の主な欄：
- `one_shot`=`MODEL_INFEASIBLE`
- `achieved_conditional_cost_RZ`=559457386
- `achieved_allocation_id`=`FULL_PF_S2_16H_OVER7_M12_TWO_01`
- `achieved_cost_updated`=false
- `cheapest_in_scope`：範囲（他round固定・m12のN_X=N_Y≥1）、`cheapest_model_feasible_m12`=2、費用559457386、根拠

小数表示はSUMMARYだけで、取り込みにはexact有理数を使う。2shot点の機械可読レコードは前回の `a006b8e15` の `docs/research/full_pf_s2_m12_two_results_20261116/C_RESULT_JOIN.json` のまま有効である。

## 条件と資格

- 価格はFull PF自身の26軸（16直接軸＋10条件付き閉包軸、m12は閉包軸）。閉包の前提と価格誤差上界／CI=nullを引き継ぐ。
- 評価条件は4shot・2shot点と同一（Z1＋T1 total decoder、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW）。
- 数値gate（閉包2,555,118組・遷移86,427,423組・残余会計）はPASS。原API・partials・資源をworkerが保存し、caller・子process 97件とも終了code 0。
- 最小N・十分N・物理的global RMSE・G_qualified・winnerではない。

## ロードマップへの差分（本体は統合担当が更新）

- Full PF/S2・16h/7の達成費用は559,457,386 RZ（m12各2shot）のまま。m12=1（531,533,828 RZ）はモデル不適合として記録できる。
- 他round固定・m12のX/Y同数かつ各軸1shot以上の範囲では、2shotが最安の適合点。それより広い配分空間の最安は未確認。
- 次の判断（ユーザー）：m11など他roundの削減、X/Y非同数、振替を評価するか。

新規信号・compile・fit・他配分Q3・retryは0件。固定SOURCE・原結果・旧STOPを保全。公開・remote照合後 **mandatory STOP**。
