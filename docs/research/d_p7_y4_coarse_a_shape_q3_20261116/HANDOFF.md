# C・Bへ：粗幅Yoshida4 A_SHAPEはQ3適合・価格待ち

**`Q3_ACCEPTED_PRICE_PENDING`。粗幅32h/5の指定A_SHAPEは元exact τ以下でQ3モデル適合。科学API1件、粗幅総費用null。** 新信号/reference/build/compile/価格/fit/他配分0。旧B_PRICE FAIL・旧UNKNOWN・旧STOPは保全。

Cが自動取込みする正本：

```text
8855f420a9fe2577efc77f099c5a39ab0cb9b7e6:
  docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json
```

同commitの `SUMMARY.json` が4点比較のexact両端と元結果locatorを持つ。C_RESULT_JOINは元configuration・total view・binding・26軸のphysical ID/m/axis/N・新配分ID・元整数列source・exact MSE/τ/assessment・固定API/RESULT sourceを持つ。cost/sufficient_N/G_qualified/winnerはnullで、価格や十分Nの資格へ昇格しない。

## configuration・配分・結果の固定結合

- 元configuration：`dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。
- total view：`97ffe0a66f2dc14fa4b3264e885428216a2716c7a8a288085ddfe3b8a0e622ec`、binding：`ef7912d2ede9f55de01c4bcf91c5c3adb935874370aaed84a9c6d48b7e9445ba`。
- 新配分ID：`B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01`、X/Y各軸 `[37,34,32,29,27,24,22,19,17,14,11,9,6]`。
- 配分整数の出所：`b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json#cases[8].allocation`（label `B0I_P7_Y4_16h_over_3__A_SHAPE`）。現幅の候補ID・信号・価格は粗幅へ流用していない。
- 保存粗幅信号：`f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53:docs/research/d_p7_y4_coarse_signal_q3_20261116/BOUND_SIGNAL_INPUT.json`。
- 新固定入力と実行SOURCE：`1cf462e3ae1f0bb16cf2f488d830cc8595909c34:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/INPUTS.json` の `inputs.cases[0]` と元contract/options/adopted_evaluation。
- 新raw：**`442a386bc6985652fc6b65548f714829bb19d375:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/run01/B0I_P7_Y4_32h_over_5__A_SHAPE/RESULT.json`**。同dirのAPI_RETURN/partials/RESOURCEと親dirのCOMPLETION/EXECUTION/child exitも保存。

## 同じ2配分を幅ごとに比較

| 幅 | 配分 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 判定 |
|---|---|---|---|---|
| 16h/3 | B_PRICE | 2.418371981954278179229700e−6 | 2.418371981954278179229706e−6 | 保存適合 |
| 16h/3 | A_SHAPE | 1.174900865076655516197599e−6 | 1.174900865076752235939216e−6 | 保存適合 |
| 32h/5 | B_PRICE | 3.409058131550481301103526e−6 | 3.409058131550481301103539e−6 | 保存・この配分で不適合 |
| **32h/5** | **A_SHAPE** | **2.074377165792602390524467e−6** | **2.074377165792685953356238e−6** | **今回・適合／価格待ち** |

exact τは `15136964828156312001/5960464477539062500000000 Ha²`。丸め前の有理数で判定。新規は最終行だけ、残り3行はSUMMARYの各result_sourceから引用した。B_PRICEのX/Y各軸は `[59,55,51,47,43,39,35,31,27,23,19,7,2]`。配分が異なる結果を同一観測として扱わない。

## 技術gate・実測・制限

保存H4/1Å/STO-3G/tol-only DF10・B0I p7/標準Y4/PF要求幅32h/5/h_RPE=.1/P0除外、元Z1＋T1/DF10 solver proxy/IID deterministic raw/全事象平方loss/exact τを維持。新numerical mean・scalar評価は0。

専用Python `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`（実体3.11.0rc1）と元PF worker/8-worker DP/AST guardを再利用。元kernel/accelerator・期待hash3d043e71…は無変更。新case selectorと共有監督helperは人工・metadata8 test PASS。実input/閉包/保存も正常、`VERIFIED_FINITE_BINARY64_SUPPORT` 296,579 pairs、DP27,206,693遷移、102記録child終了entry全exit0、API_RETURNとRESULT.record一致。supervisor wall89.807685 s、worker peak409,684 KiB、観測tree最大2,773,708 KiB。残余loss約8.3563e−20は上下界に残し、UNKNOWNとモデル判定を区別した。

§10.5の関連D/B/A後続metadata・同ユーザーjob差分はDELTA_CHECK。確認範囲で同候補・同配分の同等結果/進行jobは0。他候補の稼働processは変更していない。詳しいSOURCE・人工テスト・作用範囲・資源はREPORTを参照。

## ロードマップ差分・STOP

統合担当へ：**粗幅32h/5に固定A_SHAPEのQ3適合点1件を追加、価格待ち。** B_PRICE不適合は配分限定のまま保持。旧UNKNOWN/STOPの履歴も変更しない。ロードマップ本体は本branchで編集していない。

粗幅総費用は **null**。粗幅固有価格・費用比較・最適配分は未評価で、現幅モデルや細幅S2の価格規則を流用しない。適合は供給raw信号/target/IID lawに条件付けたQ3モデル判定であり、物理的global RMSE・十分N・G_qualified・winnerの保証ではない。

commit/push・remote SHA/tree/成果物照合後 **mandatory STOP**。追加信号・compile・価格fit・別配分Q3・N/G/winnerは、今回の完了を実行許可として開始しない。
