# 研究別SOURCE

[RPE候補探索の価格設計](../../docs/research/d_rpe_price_search_design_20261113/REPORT.md)：既存V2 assess/evidence.plan/exact会計を再利用し、部分下界・差分・bound-only接続を仕様化。新探索器・N solver・価格providerは追加せず、人工12＋既存42項目を検証。

[outer_window_residual_design](../../docs/research/d_compiled_price_methods_20261113/SOURCES.md)：connected_coefficients / truncation_coefficients / paired_mean_variance / conditional_geometric_tail。I/O・compiler・fitを呼ばない設計代数。未知tailを0とせず、常に価格未資格。[C向け特徴量](../../docs/research/d_compiled_price_methods_20261113/C_HANDOFF.md)。

[p2_direct_pilot](../../docs/research/d_p2_direct_pilot_20261113/REPORT.md)：基点5a67cea7の高速化本体を維持し、明示承認のreplay2＋fresh8 pathをSOURCE ea0706ab0で実行。20 compile、fresh統計のみ、価格資格・N/Gは供給しない。

[compiled_price_fast](../../docs/research/d_compiled_price_pipeline_20261113/TECHNICAL_HANDOFF.md)：元AccountingのRZ投影とS共通本体、既存fast hashによる明示C1拡張。人工同値性・性能のみ、実H4/P2未実行。

[h4_s0_v2_intake](../../docs/research/h4_s0_v2_integration_20261113/HANDOFF.md)：固定S0のcanonical metadata・V1/V2照合・診断view・将来A V2 packetの検査。bv2_mean_priceはC1dbからbyte一致取込み。科学leaf/価格predictor・Gは呼ばない。

[pr_rpe_rmse_evidence](../../docs/research/pr_rpe_evidence_preparation_20261113/REPORT.md)：供給V2 finite menuから全schedule/axisの不足証拠・保存再利用条件・A/C next_workを出す薄いadapter。既存controller/providerは不変、実G/最適化なし。

[PF/PR/Full-H RTEのRMSE coverage](../../docs/research/pr_rpe_method_coverage_20261113/HANDOFF.md)：pr_rpe_rmse_controllerのcomparison_method/assess coverage、明示formula menu、truncated-Hの元H誤差契約。pr_rpe_rmse_integrationのA scopeを明示。新14＋回帰85技術test、実N/C/Gなし。

[R可変C価格受入](../../docs/research/rpe_r_price_intake_20261111/REPORT.md)：rpe_r_price_intake（C1PriceIntake/RScopedFixedPrices/records_for/coverage）を追加、Cのr_variable_compiled_priceをbyte一致で取込み。既存B moduleは不変。

[R screeningと価格Phase1整合](../../docs/research/r_screening_price_alignment_20261111/REPORT.md)：元p/r/b provider実装を変更せず、metadata CLIで候補/SOURCE/law差と実験依存関係を照合。実科学計算0。

[R可変N/C provider統合](../../docs/research/rpe_provider_integration_20261111/REPORT.md)：pr_r_dependent_n/h6_df19_price_provider/rpe_nc_coverageとrpe_provider_integrationを追加。既存p/r APIを再利用し、新R欠測・N比較policy・価格資格を保持。83技術test、新実N/C/G0。

[R可変PF/PR費用](../../docs/research/rpe_r_selection_20261111/REPORT.md)：rpe_r_selection.pyとcheck_rpe_r_selection.py。R-only menuとA/C law/domain形式、exact積和・条件付き選択、39技術test PASS。実科学計算/実R最適化0。

[H4 price providerのB接続](../../docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：h4_df10_price_providerと固定V1/Mq依存を取込み。B factory/run_missing/coverageへ接続し、34技術test PASS、実N/C/G0。

共通 `trotterlib` APIと区別した実験別SOURCEの名前空間。
[mainへの収録範囲・原SOURCE・元の制約](../../docs/research/main_collection_20261103.md)を参照する。

- [G6 return aggregation prototype](algorithm_codesign/return_aggregation.py)：元snapshotの内容を保持。native資源評価は未取得、mandatory STOP。

## 他系列のSOURCEと版差

[研究別SOURCEの公開版と配置](../../docs/research/main_collection_versions_20261103.md)を参照する。P1 pipelineとprice-statisticsの旧版は保存原版から元snapshotへ戻る。選定したmain版は下記のSOURCEの組を参照する。
同名SOURCEの全版を保存し、元の入力・契約・STOPを区別している。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。
