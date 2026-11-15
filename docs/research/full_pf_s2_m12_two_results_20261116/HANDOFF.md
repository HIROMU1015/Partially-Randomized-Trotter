# C・統合担当へ：Full PF/S2・16h/7のm12=2固定Q3

**2shotもモデル適合（新規Q3は1件だけ）。Full PF/S2・16h/7の確認済み達成費用を615,304,502 RZから559,457,386 RZへ更新する。** 正本は[REPORT](REPORT.md)。

| m12のX/Y各軸shot | 条件付き費用 RZ | MSE区間 Ha²（表示） | 上界/τ | 判定 |
|---:|---:|---|---:|---|
| 4 | 615,304,502 | [1.495783385949356167254620E−6, 1.495783385949356167254624E−6] | 0.589 | 保存適合・`b4296b110` |
| **2** | **559,457,386** | **[2.402471387794261100742341E−6, 2.402471387794261100742346E−6]** | **0.946** | **新規適合** |

- 差額：**−55,847,116 RZ**（m12のX/Y各2shot×13,961,779 RZ）。m0〜m11は各軸 `[59,55,51,47,43,39,35,31,27,23,19,13]` で、費用503,610,270 RZは同じ。
- exact τ=(0.00159360010199040 Ha)²と原APIのexact有理数で判定した。余裕は小さい（上界はτの約94.6%）。
- p7の同じ変更（`93ab9dbbe`、395,962,678 RZで適合）と同じ傾向だが、別候補なので直接の比較ではない。Full PFのm12=1・他配分は未評価。

## Cが取り込む機械可読レコード

固定commit **`a006b8e15f86ee42ed66f89fbd1def721d027e87`**（branch `codex/b2-full-pf-s2-m12-two-20261116`）

| パス | 内容 |
|---|---|
| `docs/research/full_pf_s2_m12_two_results_20261116/C_RESULT_JOIN.json` | configuration（元 `11cf8ca6…`・total view `af18d37d…`・binding `69cc82b4…`）、信号packet参照、target、allocation（`FULL_PF_S2_16H_OVER7_M12_TWO_01`）と親allocation（`FULL_PF_S2_16h_over_7_B_PRICE_COST_BUDGET_3O4_01`）、26軸の価格行（N・C・N×C・physical ID・元/total候補ID・価格区分・`1002f1dfea` の価格参照）、条件付き費用・差額、Q3のstatus・MSE上下界・τ（exact）、結果参照、達成費用と更新flag |
| `docs/research/full_pf_s2_m12_two_results_20261116/EXACT_COMPARISON.json` | 4shot・2shotの2行（exact値、各原API参照） |
| `docs/research/full_pf_s2_m12_two_results_20261116/run01/FULL_PF_S2_M12_TWO_FIXED/API_RETURN.json` | 新規Q3の原API return（MSEの正本） |
| `docs/research/full_pf_s2_m12_two_results_20261116/INPUTS.json` | 実行前に固定した入力（科学実行SOURCE `b7f147ad81c88568ff2793546689da959f0751c4`） |

`C_RESULT_JOIN.json` の `achieved_conditional_cost_RZ`=559457386、`achieved_cost_updated`=true、`achieved_allocation_id`=`FULL_PF_S2_16H_OVER7_M12_TWO_01`。`result_source` は科学実行SOURCEと結果パスを持つ。小数表示はSUMMARYだけで、取り込みにはexact有理数を使う。

## 条件と資格

- 価格はFull PF自身の26軸（16直接軸＋10条件付き閉包軸、m12は閉包軸）。閉包の固定compiler等の前提、価格誤差上界／CI=nullをそのまま引き継ぐ。元receipt・physical ID・価格生成IDを付け替えていない。
- 評価条件はb4296b110と同一（Z1＋T1 total decoder、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW）。数値gate（閉包2,555,169組・遷移86,468,383組・残余会計）はPASS。原API・partials・資源をworkerが保存し、caller・子process 99件とも終了code 0。
- 2点比較での条件付き更新であり、最小N・十分N・物理的global RMSE・G_qualified・winnerではない。

## ロードマップへの差分（本体は統合担当が更新）

- PF基準のFull PF/S2・16h/7の達成費用を **559,457,386 RZ**（m12各2shot）へ更新できる。4shot点615,304,502 RZは保存参照として残す。
- これまで615,304,502 RZを基準にしていた比較（例：PR-Pauli LD2 FPN-B の予算・差、`738df2262`）は、旧基準で行ったものとして読む。今回は再計算していない。
- 次の判断（ユーザー）：Full PFのm12=1や他配分を評価するか。

新規信号・compile・価格fit・他配分Q3・retryは0件。固定SOURCE・原結果・旧STOPを保全。公開・remote照合後 **mandatory STOP**。
