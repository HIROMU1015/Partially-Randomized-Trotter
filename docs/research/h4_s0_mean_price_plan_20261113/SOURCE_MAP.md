# 固定SOURCE・条件と再利用

本表はSOURCE/metadataの読取結果。新preparation/support数値生成、価格再解析、回路生成はしていない。前段の[完全な方法索引](../bv2_mean_price_validation_design_20261113/SOURCE_MAP.md)と[構造・閉包の論点](../bv2_mean_price_validation_design_20261113/METHODS.md)を再利用する。

| 役割 | 固定commit / file | 今回の利用と制限 |
|---|---|---|
| A S0 | `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`: `docs/research/pr_rpe_rmse_s0_results_20261113/REPORT.md`, `run01/CONFIGURATION.json` | 固定inputs/conditionsと全PR scheduleのmetadata。数値pilotの価格・十分N/RMSE保証へ昇格しない |
| A実行SOURCE | `7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`: `scripts/run_pr_rpe_rmse_s0.py` | 元RMSE policy/sourceの宣言をidentityへそのまま保持。今回実行しない |
| 元H4接続入力 | `cefc01622a3cd75dd16d30de7b9fb68274f3eddd`: `docs/research/h4_df10_rpe_d1_20261110/result.json` | saved DF10/H/state/snapshot/分割/参照profileの出典。Aのdesign/contractをcopyし、horizon/Nを再計算しない |
| B V2 | `6a9d7fd36b687e58aecbed1278c6d7927d6827d6`: `src/trottertracks/pr_rpe_rmse_controller.py`, `pr_rpe_rmse_integration.py` | generate/a_spec/identity/_validate/既存MeanPriceLookup。固定Aからmetadataだけ再構成。B cost/selectは未実行 |
| B最新 | `af284d853890155bacd19e61850c196bd1d5c660`: 上記2moduleと `docs/research/pr_rpe_evidence_preparation_20261113/{REPORT,SOURCE_CONTRACTS}.md` | core2moduleは6a9とbyte一致。最新のplanner/資産索引は正式固定S0 publicationではない。未公開B成果は参照しない |
| C前段 | `1db4d6e7ebce872c069e337b76632046f60d8b56`: `src/trottertracks/bv2_mean_price.py`, `docs/research/bv2_mean_price_validation_design_20261113/` | C facadeとMeanPriceLookup、SOURCE/METHODS、118候補/236軸価格metadataを再利用。価格数値を含めない |
| B価格索引の元 | `6a9d7fd36b687e58aecbed1278c6d7927d6827d6`: `docs/research/pr_rpe_method_coverage_20261113/COVERAGE.json` | 上記C metadataの出典。元物理recipe/cohort/domainと新V2 requestの科学資格を分離 |
| C1 | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`: `docs/research/r_price_phase1_results_20261111/` | H4 q8/16×R4/8の保存C/S、元3校正/1検証、別rare unit。係数/統計/資格不変 |
| C1科学SOURCE | `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`: `src/trottertracks/r_price_phase1_science.py` | structural/Backend.context/price_requests/acquireのSOURCEを読む。ループ/scalar/wrapper・保持量の調査だけ。元runner/消費済みscopeは再起動しない |
| 元H4 Accounting | `8457c443554af5fadaa3ba36f97f6316ca3c5050`: `src/trottertracks/h4_df10_price_v1.py` | literal inverse basis、pending Z/phase、foldとterminalを再利用。predictの旧q制限を変更せず、S専用consumerは既存拡張を参照 |
| 元prep/native | `155529e20a8fac39e00786c40b57830b71203c4f`: `scripts/h46_new2_native.py`, `h46_long_time_native.py`, `src/trotterlib/h46_m5_plan.py` | prepare/gaussian_operations/sample_requests/compose、allocation/scheduleを再利用候補とする。元binary64・fragment/basis/phase/scalar順の維持 |
| 元law/provider | `c31ef424175c02379eafebe8fc6c7548c1d9aabe`: `src/trotterlib/{rte,df_partial_s2,df_rte_tail}.py` | identity extraction/finite-K/PCG64 float64 CDF sampler。元科学SOURCEを変更しない |
| A law carrier | `71ba49bb28b295aa28e51101bc3b1bf07d500767`: `src/trotterlib/rte.py` | 元C1 providerとの5関数AST一致を静的確認。実samplerとideal lawの距離や統計的独立性の証明ではない |
| S0a | `a11efd130e9dfe7086b88386570594ebe54d46af`: `docs/research/s0a_saved_structure_results_20261113/run02/` | 実行SOURCE `9516351e639278349acf55334e838520a7b3833e`。保存4path/XY8のparityだけ。原run01失敗/run02/STOP保全、今回新S0件 |
| mean/stream | 現C前段の `src/trottertracks/mean_compiled_price.py` | MeanKernel/full_wrapper_mean、stream_structure、local_residual_mean/combine_mean。新物理kernelを仮定して起動しない |
| H4 Mq | `1cf87201ebc6c9e0296fe647354981b6f7f8b0c1`: `docs/research/h4_df10_price_v2_independent_q_20261110/completion01/REPORT.md` | 固定R4 PRq16、別PFp7q32の直接検証。q64/R128価格は次段未実行案であり、保存済みではない |
| C2準備 | `f78bbc0d6ab350a2b2534ea6d169565895f0f1dc`: `docs/research/r_price_longq_largeR_preparation_20261111/{REPORT,HANDOFF}.md`, `EXECUTION_PLAN.json` | 旧HP条件のA P1候補/40・16 wrapper案。未実行。固定S0のm6に近い案の存在と実価格を混同しない |
| 旧cluster | `docs/rte_compiled_cost_validation_summary.md`、`src/trotterlib/rte_connected_cluster_cost_validation.py`、前段 `mean_compiled_price_preparation_20261112/SOURCE_INDEX.json` | _sequence_form/_iid_sequence_form、校正/holdout手順を再利用。dirty実行SOURCE/hashは元索引を正本とし、e07a HEADだけで科学SOURCEを再構成したとは言わない |
| 他保存系 | `7a8bc7f132e166390c943e6b32c92cc6abe1baf4`: `artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json`、B最新SOURCE_CONTRACTSのfinite-time/native/Pauli索引 | Full-H T2.0・one-body randomized、Pauli、異DF/別controlのTrack D価格は今回PRへ数値移植不可。旧成果と現domainを区別 |

## exact physical binding

| 項目 | 正本 |
|---|---|
| Hamiltonian | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| saved state | `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b` |
| snapshot | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| compiler | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |
| control/wrapper | symmetric_directional / full_Hadamard_with_measurement / X=cosine,Y=sine / P0 |
| compiler settings | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17 |
| decomposition | saved_weighted_Hermitian_completion / saved_snapshot_order / tol_only1e−8保存DF10 |

λはA CONFIGURATIONが保存索引からcopyした値とexact表現を保持する。今回normを再評価しない。A/Bのlawは`ideal_finite_RTE_law`宣言、C1実samplerは`PCG64_numpy_float64_CDF_saved_cohort`。同じsupport/実装SOURCEを参照しても両期待値が一致すると仮定しない。

静的AST照合の5関数は`_make_event`, `make_rte_config`, `finite_rte_distribution`, `iter_sample_rte_events`, `sample_rte_events`。コード一致以外のlaw資格を付けない。元basis operations、component/product順、signed time/phase、scalar一回、tail単位とouter間Accounting継続、finish一回は維持すべき再利用条件。H/state/snapshotのhash一致だけで回路law/phase/compilerまで同じとはしない。

公開価格索引での欠測はSOURCEを限定した結論である。private raw/cacheの回収・同一fingerprint照合が必要な場合も、今回新しい回路を作って同一性を推測しない。元normal/forced rare、EXPLORATION/CONFIRMATION、CALIBRATION/VALIDATIONの役割は変更しない。
