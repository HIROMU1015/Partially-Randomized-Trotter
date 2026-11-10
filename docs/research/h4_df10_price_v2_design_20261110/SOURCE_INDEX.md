# V2調査の固定出典と再利用境界

[REPORT](REPORT.md)。2026-11-10 JST。現branchの基点は`eec5bdf9cc391fe49b55b5673f3e385deeb20d42`。以下の旧資料・コード・JSONはこの固定Git objectで読む。carrierを元科学実行SOURCEへ付け替えない。今回はSOURCE本文・保存欄の静的確認だけで、旧検証関数を実行/importしていない。

## DF10とV1

| 固定版 | 参照と用途 |
|---|---|
| `eec5bdf9cc391fe49b55b5673f3e385deeb20d42` | [V1 REPORT](../h4_df10_price_v1_20261110/REPORT.md)、PLAN、TECHNICAL_NOTE、DEVELOPMENT/HOLDOUT、PATH_RESIDUALS。引用した残差・cohort役割・モデル不一致判定・保存schema。モデルSOURCEは`43f748655c64756c2c2a21d754870c2f7b321a77`、完了順adapterのexecution SOURCEは`5a2e6edf0cc2f7c00be491baffb86e4f139222c3` |
| `10132af4a67ce112f79c539a9552f5ca0e83e516` | [前設計](../h4_df10_longtime_price_design_20261110/REPORT.md)と[SOURCE_INDEX](../h4_df10_longtime_price_design_20261110/SOURCE_INDEX.md)。元builder/provider、候補tuple、価格表と欠測を継承。V2では係数fitの方が安く試せるという未実行案を追加するが、V1の次案を遡って書き換えない |
| `155529e20a8fac39e00786c40b57830b71203c4f` | [元価格REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md)、`scripts/h46_long_time_native.py`のcompose/sample_requests/compile_cached、保存価格表。科学SOURCEは元scientific_provenanceが正本 |
| `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | 元provider `src/trottertracks/resource_applicability/ax2b_native_df_v5.py`。head/control/scalar/full wrapper。`src/trotterlib/df_rte_qiskit.py`のbuild_sequence。今回の実行SOURCEではない |

主対象candidate：PF q8 `49709717a6c3d405c60d`、PF q16 `8e50c5560bb8f47f026f`、PR q8 `14bd81a375400c9b0305`。PR探索9/確認33は分離する。元compiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

V1保存schemaの静的確認：各rowにmethod/cohort/candidate_id/q/T/path_index/axis/compiled_rz/residual_rz/predictionがある。`prediction.structural_rz`は両方式、`prediction.features.tail_basis_groups`はPRに存在する。S1に必要な保存値があることだけを確認し、これらで新しい予測・相関・集計を計算していない。

## 旧研究の方法とAPI

以下のpathはrepository相対。固定読取版eec5bdf9は旧実行時のSOURCEではない。各JSON内のsource/environment/preparation欄と旧validation文書を元の来歴として保持する。

| コード／資料 | 読んだ機能と再利用判断 |
|---|---|
| `src/trotterlib/rte_boundary_cost_validation.py`、`docs/rte_boundary_cost_validation.md` | C1/C2/C3とpair/triple予測、controlled既定false、独立推定の誤差伝播。式と検証設計のみ再利用 |
| `src/trotterlib/rte_boundary_pair_validation.py`、`docs/rte_boundary_pair_validation.md` | same/different-fragment条件付き境界差。異basis側の補正も無視できないという旧domainの証拠。DF10値へ転用しない |
| `src/trotterlib/rte_connected_cluster_cost_validation.py`、`docs/rte_connected_cluster_cost_validation.md` | κ局所窓、order/basisの条件付き校正、`_sequence_form`/`_iid_sequence_form`の窓重み、`_evaluate_form`のSE。係数間の独立性仮定とcalibration SEの範囲に注意 |
| 同 `calibrate_connected_cluster_cost_model` | sampling/build/compileを含む。現在のDF10校正artifactはなく、今回呼ばない |
| 同 `predict_connected_cluster_cost` | 保存calibrationからの純粋な線形予測として再利用可能。ただしmodern calibration_v2のschema/条件bindingが必要で、旧validation_v1 JSONを直接渡せない。DF10のfull価格を分解して局所校正へ代用することもできない |
| 同 `validate_connected_cluster_calibration_holdout` | 直接full-sequence MCとの照合。新sampling/compileを含む後続科学作業であり今回対象外 |
| `src/trotterlib/hierarchical_cost_validation.py`、`docs/hierarchical_cost_validation.md` | `validate_controlled_repetition_extension`とcontrolled-q1/q2 affine。固定δ・K0・controlled subcircuit、Hadamard/readoutなし。qPFとの対応・独立q検証を再利用 |
| `src/trottertracks/track_d_c7_4_models.py` | `fit(x,y,q)`のequal-q重み・rank不足検出・multiresponse最小二乗は小adapterで使える。design/residual_summaryはD固有、response labelsも異なるのでそのまま呼ばない。旧科学SOURCE等は前設計索引を継承 |
| `src/trotterlib/research_direction_full_scope_extension.py` | `fit_affine_holdouts`はq1/q2と旧axes/metricsへ固定。一般の実qアンカーへ置換する場合はgeneric fit等を使う。旧パイプラインは呼ばない |
| `src/trottertracks/h4_df10_price_v1.py` | 解析的native分解、head/tail・literal basis対消去・wire局所Z融合、tail特徴。コードを改変/再実行せず保存出力をS1入力にする |

旧`_prediction_result`にはprediction−actualの符号がある。V1/S1はcompiled−predictionへ統一し、出典の誤差値を符号付き残差へ無断変換しない。独立アンカーのSE合成を、共有校正・共有trajectoryの共分散を含むCIとは呼ばない。

## 旧研究の保存結果

以下は既存JSONのsummary/config欄からの転記で、新しい評価ではない。すべてH4/1Å/STO-3G/**DF12/LD3**、δ=.1/r4/短時間.025、Qiskit1.3.0/opt1/seed17。非制御tailとcontrolled subcircuitを区別する。旧点推定・95%診断・5%判定は旧domainの記録で、DF10の許容値ではない。

| repository相対artifact | 条件・保存値 |
|---|---|
| `artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_n1000_seed20260823_v1.json` | K0、非制御event列、preparation `400b18b5…`。pair max relative error .08074690645865598、triple .023278818253758167 |
| `artifacts/hierarchical_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_l8_ncal2000_nhold2000_v1.json` | K0、非制御L8、別preparation `fbed0c65…`。triple RZ max relative error .017442609235885986、95% diagnostic .08306705787248214。点推定と不確かさを区別 |
| `artifacts/hierarchical_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_controlled_q4_ncal300_nhold300_v1.json` | K0、preparation `fbed0c65…`、controlled-q1/q2校正→q4照合。RZ relative error .002932679996584643、6metric max .0030683179419640147、RZ95% diagnostic .009166088499936532 |
| `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.json` | K2、非制御tail、preparation `50c77f54…`。max relative error .02935658114687385、prediction95% halfwidth .015372040020020272、95% diagnostic .05723993981740553、allocation cap到達 |

独立holdoutでの改善があってもcompiler一般性を保証しない。旧数値を使った新しい統計解析・旧検証の重複実行は行っていない。

## Qiskit一次SOURCEと検査範囲

- [公式Qiskit 1.3.0 Optimize1qGatesDecomposition](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py)：`_resynthesize_run`、`_substitution_checks`、`_error`、`run`。同版のローカルインストールを静的に読み、公式GitHubの155–232行もconnectorで取得して照合。公式file blobは`55232ac1be20955df6a99cabaa9b9b151227c797`。実H4へのpass適用や原因別RZ寄与の測定はしていない。
- projects/AGENTS、worktree AGENTS/PROJECT_MAP、現研究概要・正式共通指示の方針を適用。主担当のみで調査し、他AIへの委任・独立科学レビューは行わない。
- 今回は新検証パイプライン・コード・科学結果を追加しないため、新規テスト、validation manifest、旧結果の更新を省く。本directoryの2文書とPROJECT_MAP入口に限定する。公開前に出典path/リンク・引用値・差分形式・V1無変更を検査し、公開後にremote SHAと新文書を照合する。
- 継続branch：`codex/resource-eval-h4-df10-longtime-price-design`。新しい科学実行SOURCEではない。科学計算・詳細科学レビューは次の明示許可まで開始せずmandatory STOP。
