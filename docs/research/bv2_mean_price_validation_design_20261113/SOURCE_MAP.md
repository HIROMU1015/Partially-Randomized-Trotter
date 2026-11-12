# 固定SOURCEと再利用の範囲

以下はSOURCE/保存metadataの調査。科学値を再解析した記録ではない。実行コードはB/Aの作業中worktreeから読まずGit固定objectを参照した。

| 対象 | 固定ref/入口 | 再利用と境界 |
|---|---|---|
| B V2 | `6a9d7fd36b687e58aecbed1278c6d7927d6827d6`、pr_rpe_rmse_controller.py / pr_rpe_rmse_integration.py / method_coverage REPORT,HANDOFF,COVERAGE | configuration/physical identity、Full/truncated、既存MeanPriceLookupとassess。cost/selectをCで再実装・実行しない |
| B共通accounting | 同refのh4_df10_rpe_connection.py / pf_pr_rpe_pipeline.py | Bのexact-scalar版を固定test processへ読取。C worktreeの旧helperをB版として扱わず、どちらの実ファイルも変更しない |
| A最新準備 | `ee15bbd29ec571dd2526feecdb81ed757f783557`、docs/research/pr_rpe_rmse_pilot_preparation_20261113/REPORT.md | 1系/2時点のRMSE pilot、λ/source/同κ条件、未知M/κ/n。旧B f09 pinへの依存はA/Bへ引渡す。CでN/契約を実装しない |
| 現C平均kernel | `2a882fc9d17ed5e32f5581d0f220514daad0235a`、mean_compiled_price.MeanKernel/MeanPiece/full_wrapper_mean/local_residual_mean/combine_mean/MeanPriceProvider | 閉状態代数と受動lookup、局所D期待formの再利用。physical kernel/mean Cは未取得 |
| C1状態/stream | `fcf5b64fac24f3897ab3408b84fd8256f46e8513`、docs/research/c1_physical_state_stream_preparation_20261112/STATE_AUDIT.json、mean_compiled_price.stream_structure | 元wordのone-pass consumer、literal zero/phase/flushの未閉包条件。人工反例と物理到達性を分離 |
| S0a | `a11efd130e9dfe7086b88386570594ebe54d46af`、docs/research/s0a_saved_structure_results_20261113/run02/ | 保存4path/XY8 S parityの完了証拠。実行SOURCE9516351e639278349acf55334e838520a7b3833e。旧run01失敗、run02 PLAN/RESULTS/STOPを不変保持。新S/mean評価はしない |
| C1 compiled/残差 | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`、r_price_phase1 PLAN/CALIBRATION/VALIDATION/RARE_DIAGNOSTICS | 元q/R8セル・3校正/1検証/system、有限cohort/paired統計とC−S補正。SOURCE1a87e5331、凍結7eeabf0a。fit/統計を再実行しない、forced rareは別unit |
| C1回路word | SOURCE`1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`:r_price_phase1_science.structural/Backend.price_requests | head→tail→reverse、q/J loop、scalar once、axis wrapper、finish。SCI入口は今回未実行 |
| C1 native S | H4`8457c443554af5fadaa3ba36f97f6316ca3c5050`:h4_df10_price_v1.Accounting、H6`ae44283d169b61deace7e1c0d18c4538d4a3fc0c`:h6_df19_price_v1.Accounting | pending basis/Z/phase、literal expansion。C1.calibrateが流用するTrack D C7-4 fitはアルゴリズムだけで、Strong/Referenceや別DFの数値係数は移植しない |
| 元prep/law | carrier`155529e20a8fac39e00786c40b57830b71203c4f`:h46_new2_native.prepare/gaussian_operations、h46_m5_plan.weights/schedule、h46_long_time_native.sample_requests。provider`c31ef424175c02379eafebe8fc6c7548c1d9aabe`:df_partial_s2/df_rte_tail/rte | identity extraction、元signed allocation、sampler/make_event。元law/sourceを保持する方法として再利用、今回prep/samplerを起動しない |
| 局所cluster | docs/rte_compiled_cost_validation_summary.md、rte_connected_cluster_cost_validation._sequence_form/_iid_sequence_form/calibrate_connected_cluster_cost_model/predict_connected_cluster_cost/validate_connected_cluster_calibration_holdout | 局所1..3-event form/calibration→freeze→holdout。元H4 DF12/H5 DF9/uncontrolled等、現full-wrapperへ数値を移植しない |
| 旧cluster実行provenance | C固定`2a882fc9d...`:docs/research/mean_compiled_price_preparation_20261112/SOURCE_INDEX.json | 記録HEAD e07a5e60...だけでなくdirty SOURCE SHA/snapshot hashが正本。今回旧校正SOURCE一式を再構成したと主張しない |
| H4 Mq | `1cf87201ebc6c9e0296fe647354981b6f7f8b0c1`:h4_df10_price_v2_independent_q_20261110/completion01/REPORT | PR p1/R4/δ.1のq16、B0/p7 q32の固定係数独立検証。q増大/別R/全law/CI未資格 |
| H6 Affine | `73dcbe23678bfc50d5654f09e600d4cfbe75e697`:h6_df19_price_v2_20261110/REPORT、Q4_FROZEN_MODELS/Q2_DIRECT/Q4_DIRECT/Q_VALIDATION | q1/2校正・q4独立。p7/R16/δ1.6、Full19 PFのdomain。初期p5/別δ/C1 R32へ移さない |
| Full/truncated PF | B基点`f09af82482aad11460c4bb73d3e13d7ada40be6f`:h46_long_time_signal.candidates、h46_long_time_native.cost_trial、pf_common_run_native.cost_one/overlay | FULL rank全prefixとB0/B0I保持Hの価格経路、元native full wrapper/compiler。元固定T/公式/幅を維持し、新RMSE用に無条件再使用しない |
| 保存PF direct | pf_pr_rpe_pipeline.saved_deterministic_points、rpe_provider_integration.FixedSavedPrices | 保存決定論価格をcopyする既存API。Full/retained-H、δ/時間/制御/compiler/SOURCEがexactに一致して初めて参照候補 |
| DF Full-H RTE | `9b83b96b83f62142c9d87858dc24a94db7ac5d61`:h46_full_h_pure_rte.prepare_full_h/cost_job | one-bodyの非scalar Z成分をDF tailへ追加、deterministic blocks空、scalar一回。prepare(ham,0)だけでone-bodyを残すB3とは別 |
| Full-H保存価格 | `7a8bc7f132e166390c943e6b32c92cc6abe1baf4`:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json | evaluation_T=2.0、価格group n9・別確認n33のmetadataあり。kernel/source hashes・元candidate inventoryが正本。current RMSE law/event_count/mean C bridgeは未接続 |
| Pauli資産 | 固定結果`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、科学SOURCE`e755ecf581600192fd2745d7066e410cebd933d3`。BのSOURCE_MAPにpauli_h2_h6_study/h4_pauli_b0_execution/h4_pauli_pr_search等 | Pauli Full/truncated PF、PR、L_D0全非identity RTEとfull-wrapper。別表現/state/固定signal policyをDF RMSEへaliasしない |

既存SOURCE、S/C、calibration/validation、grant/ledger/marker/KEEP/STOPは保全する。今回private raw/cache/snapshotや実行中A/B branchは読まず、Gitのpublic metadataとSOURCEだけを用いた。
