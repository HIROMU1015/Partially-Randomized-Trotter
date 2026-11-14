# A → C：candidate identityの引継ぎ

[REPORT](REPORT.md)／[SIGNAL_COVERAGE.csv](SIGNAL_COVERAGE.csv)（`price_side`・`signal_price_identity`の列）。

## 1. 照合結果（新しい価格の評価はしていない）

| 系列 | 価格側 | 精度側との一致 |
|---|---|---|
| Full PF shortlist | CELL_PLAN_V2（`855573064`）のm4/m5 CALIBRATION・m6 HOLDOUT。M5 m4は保存receiptを受入済み | **configuration IDが一致**：`11cf8ca6`、`a799697f`、`70f7b317`。**physical IDが一致**：9 round×X/Y |
| Truncated B0I p7 shortlist | 同上（TRUNCATED_PF） | configuration ID（`a6e55777`、`defc5e88`、`424eecbf`）とphysical IDが、T2と一致 |
| Truncated B0I p7・S2・0.1 | **価格なし** | 信号が揃っているのはこの候補で、価格は別候補にある |
| Full PF S2・0.1（C manifest `f2d58f32`） | m3/m4はD `e7dca5f7`のreceipt metadata（C `c5466946`の記述）。新しい価格計画はない | 信号もS2/S2L/S3が混在している |
| PR-DF LD1/LD2 | PR_DF_COVERAGEの52軸 | 26 round×X/Yの**physical IDが一致** |
| PR-Pauli LD1 m0 | D B0（単一path） | **identity・law_bindingが完全一致** |
| Full-H κ2 m3 | P-1（`a04470c0a`、単一path） | **identity・law_bindingが完全一致** |
| Full-H R448 | なし | κ2の価格を適用しない |

不一致は0件である（builderの`--check`で確認）。

## 2. Cが判断・実施すること

- **決定論PFの候補を揃える**：次のどちらかを、ユーザーの判断で選ぶ。
  - (a) shortlistの全13round信号をAが計算する（22（＋2）件×2方式、承認待ち）。
  - (b) S2・0.1の価格計画を新たに立てる。
  - それまで、Truncatedの信号（S2・0.1）と価格（shortlist）を同じGに結合しない。
- **全13roundのE[C]**：CELL_PLAN_V2はm4〜m6のmodel cellだけを持つ。m0〜m3とm7〜m12はmodelの外挿か、別の価格が必要である。信号側が揃っても、価格側の資格は別に判断する。
- **Pauli辞書とDF tail**：E[S]・C1・P2はDF tailの価格である。PR-Pauliへ流用しない（route `PR_PAULI_TAIL`とlowering D2で区別する）。
- **PR-DFのscalar SOURCE bridge**：D `e8273676`の受入を、価格側で継続する。信号側のphysical IDは一致している。

## 3. 参照

- 精度側の元ID：BのREADINESS（`99930491`）、A follow-upの`C_HANDOFF.json`（`714247619`）、T2の`CONFIGURATIONS.csv`（`5fb2c32e`）。
- 価格側：CのCELL_PLAN_V2・PR_DF_COVERAGE（`855573064`）、D B0（`8502dd33`）、P-1（`a04470c0a`）。
