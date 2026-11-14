# Cへ：PR-DF LD1 m4独立holdoutの記述的結果

**`DESCRIPTIVE_COMPLETE`、fresh4 path・X/Y計8実compile、8有効receipt、2比較を公開した。** 予定8／再利用0／新規8（試行8）／有効8／保留・欠測0、retry0。Cによる受入/解釈は未実施。

- 科学実行SOURCE：`7bdfc6f629aa74a5c1347abf84ad6486c669c9d8`（取得前INPUTSとcallerを固定）。
- 元raw公開SHA：`3042e6e66ad693ed514274334adecd3d1b42dc02`。元receipt bytesは同commit `run01/receipts/p0..p3/`、全locatorは[C_HANDOFF.json](C_HANDOFF.json) `rows[*].axis_receipt/original_receipt/structural_receipt`。
- 最終公開SHA：このHANDOFFを収録したcommit。`git log -1 --format=%H -- docs/research/d_pr_df_ld1_m4_holdout_results_20261115/HANDOFF_C.md` で解決し、remote一致の完全SHAは最終回答へ記載する。
- 固定予測：`3cac53da1a4a8b2228245593c74c1ff9ec1f5212` の `docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json#models[axis].m4_prediction.predicted_RZ` を軸別に直接読取。C branch merge、再fit/再予測はない。

physical `3df0cfd3356c358aeda8`、m4/T1.6/q16/R32、cohort `D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1`、index0〜3、各16固定seed。元正本は `2f423befbba5075bad4ede3057f89edd788d2661` のm3 INPUTS `future_holdout.row/cell`。較正cohortと分離し、保存seed表を元normal samplerに使用。H4/DF10/LD1/S2/Δ.1/K2・元κ/H/state/snapshot/順序・scalar総T一回・新ZZ・symmetric-directional control・測定付きfull wrapper/P0・Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を維持した。

## 8行の個別入力

| path | axis | S | C | 固定seed selector | 元axis receipt |
|---|---|---|---|---|---|
| 0 | X | 124411 | 129955 | `cells[0].seeds[0]` | [receipt](run01/receipts/p0/cosine.json) |
| 0 | Y | 124411 | 129955 | `cells[0].seeds[0]` | [receipt](run01/receipts/p0/sine.json) |
| 1 | X | 125935 | 131685 | `cells[0].seeds[1]` | [receipt](run01/receipts/p1/cosine.json) |
| 1 | Y | 125935 | 131685 | `cells[0].seeds[1]` | [receipt](run01/receipts/p1/sine.json) |
| 2 | X | 128133 | 133389 | `cells[0].seeds[2]` | [receipt](run01/receipts/p2/cosine.json) |
| 2 | Y | 128133 | 133389 | `cells[0].seeds[2]` | [receipt](run01/receipts/p2/sine.json) |
| 3 | X | 118424 | 123741 | `cells[0].seeds[3]` | [receipt](run01/receipts/p3/cosine.json) |
| 3 | Y | 118424 | 123741 | `cells[0].seeds[3]` | [receipt](run01/receipts/p3/sine.json) |

全行でevent digest・fingerprint・source・seed表・元receiptへ戻れる。[ADMITTED_RECEIPTS](ADMITTED_RECEIPTS.json) と[公開後binding照合](RECEIPT_BINDING_CHECK.json)：8/8一致、失敗0。X/Yは同一pathの別軸。数値action gate=`NOT_RUN_AS_IN_ORIGINAL_P2_FRESH`、full-wrapper statevector=false。新しい量子作用保証は付けない。8行のvalidは取得/identity検査の状態で、モデル評価と別である。

## 軸別2件のexact比較

| axis | P (exact RZ) | O (exact RZ) | P−O | 絶対誤差 | (P−O)/O | 絶対相対誤差 |
|---|---|---|---|---|---|---|
| X | 217164876899444503/1683627180032 | 259385/2 | -1188941146855657/1683627180032 | 1188941146855657/1683627180032 | -1188941146855657/218353818046300160 | 1188941146855657/218353818046300160 |
| Y | 217164876899444503/1683627180032 | 259385/2 | -1188941146855657/1683627180032 | 1188941146855657/1683627180032 | -1188941146855657/218353818046300160 | 1188941146855657/218353818046300160 |

両軸のOは129692.5 RZ。P−O≈−706.178399206563 RZ、相対≈−0.544502110150211%で過小予測。[COMPARISON](COMPARISON.json)／[CSV](COMPARISON.csv)。Pへ構造項/残差を再加算せず、軸別4 Cのexact平均を用いた。O=0/欠測時のnull規則も事前固定のまま（今回は該当0）。

**`model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE`、tolerance=null。モデルPASS/FAILは付けない。** C2/Pauliや決定論PFの許容差を流用していない。今回のD2によるbinding照合と、C自身の消費者受入は別である。

資源：GNU wall61.27秒、CPU397%、user235.07/system8.17秒、max RSS593212 KiB、swap0、exit0。4 path並列、compiler内1、caller内部60.382917秒。個別stage/RSSは[REPORT](REPORT.md)、[time](EXECUTION_RESOURCES.txt)、run01のprogress/timingを参照。2048 eventすべてorder0、order2 product/非自明位相は未観測UNKNOWNで追加sampling0。

ロードマップ差分：指定有限cohortのm4取得/比較待ちは解消。保存ideal I構造期待値＋Q有限cohort残差の予測と、Qの4 path平均の差を得た範囲に留まる。I/Q差・rare・小標本/係数不確かさ・全law E[C]/CI・長round移送・N/G/winner・モデル資格は未認定。ロードマップ本体は統合担当が更新する。

m3とPauli LD2のraw/旧STOP、Dのm9 worktree/jobを保全。追加取得、再fit、事後閾値、他roundへ進まず、commit/push・remote一致確認と引渡しで **mandatory STOP**。
