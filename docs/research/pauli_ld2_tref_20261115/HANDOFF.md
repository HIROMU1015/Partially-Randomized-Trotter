# 統合担当・A/C2/Bへ：PR-Pauli LD2のモデル参照総額 T_ref

[REPORT](REPORT.md)／[RESULT.json](RESULT.json)／[AXIS_TABLE.json](AXIS_TABLE.json)／[AXIS_TABLE.csv](AXIS_TABLE.csv)／[INPUTS.json](INPUTS.json)。入力とscriptの固定commitは`310c08d50d0408febfe576f24c5e76b27879ac4c`。

## 結果

- **T_ref = 305225682539751998/90343469 ≈ 3,378,503,016.52 RZ**（PR-Pauli LD2、Q3適合配分 `[48,45,42,39,36,33,29,26,23,19,16,13,9]`、X/Y各軸）。
- C2の凍結モデルを26軸に当てはめた**モデル参照値**であり、E[C]ではない。保存予測6軸（m5・m6・m9のX/Y）を再利用し、20軸を新たに評価した。負の予測・入力不一致は無い。
- T_refの約94.65%はm10〜m12に由来する（直接比較のある最長roundはm9）。

| 基準 | T_PF | D_PF = T_PF − T_ref |
|---|---:|---:|
| B0I p7 S2 | 593,555,352 | −2,784,947,664.524105…（exact −251601832996555910/90343469） |
| Full PF S2 | 838,693,580 | −2,539,809,436.524105…（exact −229455195094522978/90343469） |

**読み方**：Δ = T_LD2 − T_ref が D_PF より小さければLD2が低費用、等しければ同額。どちらの D_PF も負なので、LD2が低費用になるには、対象lawの期待総費用が T_ref より約82.43%（B0I p7 S2基準）、約75.18%（Full PF S2基準）以上小さい必要がある。

## 守った条件

- 凍結モデルの`application_scope`（較正とm5/m6の宣言済み予測だけ）・`qualified_for_candidate=false`は変更していない。26軸への適用は本指示で承認された診断目的に限る。
- m9の観測平均と+0.274418%は独立比較の記録として保持し、補正・許容差・CIには使っていない。
- 予測値は整数RZへ丸めずexactに合算した。
- 正式E[C]、CI、正式G、winnerは未認定。新規path・build・compile・再fit・Q3は0件。

## 使用した入力

| 用途 | 固定SHA | path |
|---|---|---|
| 26軸対応表・固定shot重み | `df5e5c452a3814f7e04ea9cacf05c8f4bdcc047c` | `docs/research/a_pauli_ld2_expected_price_plan_20261115/AXIS_MAP.json` |
| 凍結モデル・m5/m6予測 | `b74e4e536392c2bbe94cd371436615fc56c0564f` | `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json` |
| m9予測 | `3f6e6604a4877dc882fd1ee7538a854464160a5b` | `docs/research/d_pauli_ld2_m9_longpoint_results_20261115/PREDICTIONS.json` |
| m9比較・HANDOFF | `cb034e547c444ea67a4f4f90a8534c13f4db4451`（比較本体 `adeae8f83`） | `docs/research/d_pauli_ld2_m9_longpoint_results_20261115/` |
| Q3固定配分 | `50be68b803a90fd893a1499ebce74e033225144d` | `docs/research/pauli_q3_three_quarter_20261115/run01/RESULT_LD2.json` |
| PF固定総額 | `1002f1dfea34de439ca1b704a0192bca473c5387` | `docs/research/pf6_allround_cost_20261115/RESULT.json` |
| 比較範囲 | `aae9f21799b72a03cff4d75769526c30d966aee1` | `docs/research/h4_fixed_shot_selection_scope_20261115/REPORT.md` |

## ロードマップへの影響（本体は統合担当が更新）

- PR-Pauli LD2の行：「Q3適合配分でのモデル参照総額 T_ref ≈ 3.3785×10⁹ RZ（凍結モデルの診断的適用、E[C]ではない）を公開。PF 2基準との差 D_PF はどちらも負（B0I p7 S2：−2.785×10⁹、Full PF S2：−2.540×10⁹）。」
- 判断材料：LD2がPF基準を下回るには、対象lawの期待総費用がモデル参照値より75〜82%以上小さい必要がある。T_refの大半はモデルの較正・比較点の外側（m10〜m12）にあるので、ここでの価格の根拠（Aの計画したS0・S1など）が判断を左右する。
- 正式G・winnerは未確定のまま。

**mandatory STOP**。
