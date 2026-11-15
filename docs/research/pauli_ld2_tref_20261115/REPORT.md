# PR-Pauli LD2：固定shotモデル参照総額 T_ref と、決定論PF 2基準に対する境界

2026-11-15 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

**出力はモデル参照値である。** C2の凍結モデル（X/Y別）を、診断目的でm0〜m12の26軸に当てはめた値であり、対象lawの期待価格 E[C] ではない。凍結モデルの元の`application_scope`（`LD2_FINITE_COHORT_CALIBRATION_AND_DECLARED_M5_M6_PREDICTIONS_ONLY`）と資格（`qualified_for_candidate=false`、`CALIBRATED_UNVALIDATED`）は変更していない。正式E[C]、CI、正式G、winnerは未認定のままである。

## 主結果

対象：PR-Pauli LD2（configuration `d7dd4f52…`、S2・幅0.1・K2）、Q3適合配分 `PAULI_LD2_USER_SCALE_3O4_01`。X/Y各軸で `N = [48,45,42,39,36,33,29,26,23,19,16,13,9]`（m0〜m12）。

$$
T_{\mathrm{ref}}=\sum_{m=0}^{12}\sum_{a\in\{X,Y\}} N_{m,a}\,C_{\mathrm{ref}}[m,a],\qquad
C_{\mathrm{ref}}[m,a]=\beta_0[a]+\beta_1[a]\,q_m+\beta_2[a]\,q_mR_m
$$

| 量 | 値（RZ） |
|---|---|
| **T_ref**（exact） | **305225682539751998 / 90343469** |
| T_ref（表示） | **3,378,503,016.524105…** |

予測値は整数RZへ丸めず、有理数のまま合算した（表示値だけを小数第6位で丸めた）。

### 決定論PF 2基準との差 D_PF = T_PF − T_ref

| 基準（B2の固定総額） | T_PF | D_PF（exact） | D_PF（表示） | T_ref / T_PF |
|---|---:|---|---:|---:|
| B0I p7 S2（16h/7） | 593,555,352 | −251601832996555910 / 90343469 | **−2,784,947,664.524105…** | 約5.69 |
| Full PF S2（16h/7） | 838,693,580 | −229455195094522978 / 90343469 | **−2,539,809,436.524105…** | 約4.03 |

どちらの基準でも D_PF は負である。つまり、モデル参照値の上では、LD2の固定shot総額はPF基準より大きい。

**境界条件**（対象lawでの未知の期待総費用を T_LD2、Δ = T_LD2 − T_ref とする）
- **B0I p7 S2 基準**：Δ < −2,784,947,664.524105… RZ ならLD2が低費用、Δ がこの値に等しければ同額である（LD2が低費用になるには、T_LD2 が T_ref より約82.43%以上小さい必要がある）。
- **Full PF S2 基準**：Δ < −2,539,809,436.524105… RZ ならLD2が低費用、Δ がこの値に等しければ同額である（同じく約75.18%以上小さい必要がある）。

これは T_LD2 と T_PF の比較を、T_ref を基準に言い換えたものである。Δ の大きさや符号についての上限・区間は、今回求めていない。

## 26軸の内訳

機械可読表：[AXIS_TABLE.json](AXIS_TABLE.json)／[AXIS_TABLE.csv](AXIS_TABLE.csv)（各軸の物理ID・q・R・N・C_ref・N×C_ref、値の出所）、[RESULT.json](RESULT.json)、[INPUTS.json](INPUTS.json)。

| m | q | R | N（各軸） | C_ref（各軸、表示） | 値の出所 | X+Yの加重項（表示） | T_refに占める割合 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 1 | 2 | 48 | 2,226.04 | 新規適用 | 213,699.91 | 0.01% |
| 1 | 2 | 3 | 45 | 4,167.23 | 新規適用 | 375,051.08 | 0.01% |
| 2 | 4 | 5 | 42 | 8,087.63 | 新規適用 | 679,360.94 | 0.02% |
| 3 | 8 | 9 | 39 | 16,080.46 | 新規適用 | 1,254,275.79 | 0.04% |
| 4 | 16 | 17 | 36 | 32,674.26 | 新規適用 | 2,352,546.88 | 0.07% |
| 5 | 32 | 33 | 33 | 68,294.45 | 保存予測（b74e） | 4,507,433.95 | 0.13% |
| 6 | 64 | 66 | 29 | 149,670.61 | 保存予測（b74e） | 8,680,895.26 | 0.26% |
| 7 | 128 | 132 | 26 | 352,560.57 | 新規適用 | 18,333,149.51 | 0.54% |
| 8 | 256 | 263 | 23 | 917,269.37 | 新規適用 | 42,194,391.02 | 1.25% |
| 9 | 512 | 526 | 19 | 2,687,267.68 | 保存予測（3f6e） | 102,116,171.75 | 3.02% |
| 10 | 1024 | 1052 | 16 | 8,786,343.66 | 新規適用 | 281,162,997.01 | 8.32% |
| 11 | 2048 | 2103 | 13 | 31,207,839.28 | 新規適用 | 811,403,821.39 | 24.02% |
| 12 | 4096 | 4205 | 9 | 116,957,179.00 | 新規適用 | 2,105,229,222.04 | 62.31% |

- 26軸のうち6軸（m5・m6・m9のX/Y）は保存予測をそのまま使い、残り20軸だけを新たに評価した。
- 凍結係数はX/Yで同一なので、C_ref も各roundでX/Yが同じ値になる。それでもX/Yは別の行として、それぞれのshot数を掛けた。
- 負の予測は無く、入力不一致も無い。補正値は一切使っていない。
- **T_refの約94.65%がm10〜m12（モデルの較正・比較点の外側）に由来する。** 較正cellはq≤8・1 pathあたりevent数≤4205（S66・S526・S4205・Q8R66・P2R66）で、直接比較のある最長roundはm9（q=512、event数269,312）である。

## m9の独立比較（記録としてのみ保持）

C2の`cb034e547`のHANDOFF（比較本体は`adeae8f83`の`COMPARISON.json`）によると、m9のX/Yで、予測 P = 242777084125865/90343469 に対し、有限2 pathの観測平均 O = 5359827/2、符号付き相対差（(P−O)/O）は **+0.274418%** だった。

この記録は`INPUTS.json`と`RESULT.json`にそのまま転記した。**モデルの補正、他roundの許容差、CIには使っていない。** O は有限2 pathの平均であり E[C] ではない。

## 入力（固定SHA）と照合

| 入力 | 固定SHA・path |
|---|---|
| Aの26軸対応表・固定shot重み | `df5e5c452a3814f7e04ea9cacf05c8f4bdcc047c`、`docs/research/a_pauli_ld2_expected_price_plan_20261115/AXIS_MAP.json` |
| C2のX/Y別凍結モデル、m5/m6予測 | `b74e4e536392c2bbe94cd371436615fc56c0564f`、`docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json` |
| m9の観測前予測 | `3f6e6604a4877dc882fd1ee7538a854464160a5b`、`docs/research/d_pauli_ld2_m9_longpoint_results_20261115/PREDICTIONS.json` |
| m9の独立比較・最終HANDOFF | `cb034e547c444ea67a4f4f90a8534c13f4db4451`、`docs/research/d_pauli_ld2_m9_longpoint_results_20261115/HANDOFF.md`（比較本体は`adeae8f831b0cf7ffbf3448ad39b3c0ea259a655`の`COMPARISON.json`） |
| PauliのQ3固定配分 | `50be68b803a90fd893a1499ebce74e033225144d`、`docs/research/pauli_q3_three_quarter_20261115/run01/RESULT_LD2.json`（status `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`） |
| 決定論PFの固定総額 | `1002f1dfea34de439ca1b704a0192bca473c5387`、`docs/research/pf6_allround_cost_20261115/RESULT.json` |
| 候補・配分の比較範囲 | `aae9f21799b72a03cff4d75769526c30d966aee1`、`docs/research/h4_fixed_shot_selection_scope_20261115/REPORT.md` |

**照合した内容**
- AXIS_MAPの26行（13 round×X/Y、重複なし）が、LD2のconfiguration・配分と一致した。各行のq・Rは正の整数で、特徴量は `one=1`、`pf_step_count=q`、`event_count=q·R`（C2の`PR_STEP_V1`）。
- Q3の配分（X/Y各軸）が上のshot数と一致した。
- 保存予測6件：物理IDがAXIS_MAPと一致し、保存された特徴量が q・R から作る値と一致した。m9予測の`model_source`はb74eの同じ軸のモデルを指す。さらに人工testの中で、保存6値が凍結係数の式と一致することを確認した（値の差し替えには使っていない）。
- PFの2基準値がB2の`RESULT.json`と一致した。
- **検算**：係数がX/Yで同一なので、Aの固定shot重み（**両軸合算**：shots 756、steps 205,844、events 471,030,556）と係数の内積 `756·β0 + 205844·β1 + 471030556·β2` を求め、T_ref とexactに一致した。

**重複照合（共通運用指示§10.5）**：同じ候補・凍結係数・配分による26軸の評価・加重総額は、確認した範囲（全remote branch、2026-11-15 18時以降のcommit）に無かった。Aの計画（df5e5c452）は評価前（`price_values_computed=0`）。作業中に公開された`85dab58d6`・`1fb47bf92`はPR-DF LD1の別候補であり、重複ではない。保存予測6件は再利用し、評価したのは不足の20軸だけである。

## 実行・検証・保全

- 手順：`tref.py prepare`（入力の結合、issue 0件）→ scriptと`INPUTS.json`をcommit `310c08d50d0408febfe576f24c5e76b27879ac4c`で固定・push → `tref.py evaluate`を一度だけ実行（exit 0、wall 0.08秒）。[EXECUTION.txt](EXECUTION.txt)。
- 計算は`fractions.Fraction`だけで行った。26軸の有理数演算なので、並列化は不要と判断した。
- 人工test 5件PASS（[TESTS.txt](TESTS.txt)）：exactな総和とX/Yの個別加重、境界の符号、保存値をそのまま使うこと、負の予測・未解決行で総額を出さないこと、表示用の丸めを計算に使わないこと、保存6予測と凍結式の一致。
- 新規path・build・compile・再fit・Q3はいずれも0件。
- 決定論PFの既存結果（`pf6_allround_cost_20261115/`）は変更していない。凍結モデル、保存予測、m9の比較、A・C2・Dのbranchも変更していない。
- **mandatory STOP**：この参照値を根拠に、価格の追加取得、モデルの適用範囲の拡張、補正、G比較へは進まない。
