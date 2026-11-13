# trotterlib モジュール索引

[H4 Full PF H_eff実行結果](../../docs/research/h4_full_pf_heff_results_20261113/REPORT.md)：既存評価器/direct-expm/保存loaderを再利用したS2/Δ.1の限定実行。branch依存shift +1.33712e−4 Ha、二時点の全spectral再構成は数値gate通過。個別RPE bias/十分N/global RMSEはnull、公開後STOP。

[pf_effective_hamiltonianの元H/state診断追加](../../docs/research/h4_full_pf_heff_preparation_20261113/REPORT.md)：同じeigh結果から元state weights・縮退射影trace・固有系gateを追加。Full/Truncated PFの方法採用を反映し、個別RPE資格/nullは維持。新4＋旧15人工PASS、実科学0。

[pf_effective_hamiltonian](../trottertracks/pf_effective_hamiltonian.py)：明示PRINCIPAL/ENERGY_WINDOW branchとunitary/Schur/再構成gate、state overlap/縮退group・reference比較を返す。PF builder/ground同定/RPE bias保証ではない。[REPORT](../../docs/research/pf_effective_hamiltonian_20261113/REPORT.md)／[人工15件](../../tests/check_pf_effective_hamiltonian.py)。旧8回帰含め23PASS、実科学0。

[PR-LD1/LD2実行結果](../../docs/research/pr_rpe_first_signal_results_20261113/REPORT.md)：既存[signal provider](../trottertracks/pr_rpe_signal_provider.py)を無変更で使い、全26候補が数値gate通過。新reference/旧mean再解析/N/C/G0。[caller](../../scripts/run_pr_rpe_first_signal.py)／[5mock test](../../tests/check_pr_rpe_first_signal_caller.py)／[B/D HANDOFF](../../docs/research/pr_rpe_first_signal_results_20261113/HANDOFF.md)。局所結果でglobal RMSEを認定しない。

[pr_rpe_signal_provider](../trottertracks/pr_rpe_signal_provider.py)：B V2の全roundを直接expm/既存paired Taylor・normalizer・powerへ接続。uniform PR S2/M5・K2/4/6、独立raw/corrected、SOURCE付き部分FAIL保存。H4 saved factoryあり、H6初期例外/PR-Y4未対応。[REPORT](../../docs/research/pr_rpe_signal_pipeline_20261113/REPORT.md)／[10人工test](../../tests/check_pr_rpe_signal_provider.py)。実H4/N/C/G0。

[pr_rpe_signal_plan](../trottertracks/pr_rpe_signal_plan.py)：Bのvalidator/exact_scalarと既存physical recipeを受け、全M+1roundのsignal job・欠測・SOURCEを整理するmetadata-only helper。candidate生成/管理・normalizer/N/C/G・optimizerなし。[REPORT](../../docs/research/pr_rpe_search_design_20261113/REPORT.md)／[7人工test](../../tests/check_pr_rpe_signal_plan.py)。新numerical PR providerは設計段階。

[H4 PF長時間の二経路診断](../../docs/research/pf_power_stability_h4_20261113/REPORT.md)：既存`pf_power_stability`の同順expm・失敗値保持、screenのdeterministic_step、保存H/state loaderを再利用。helper dd24dd329、recipe256caad34を保全し、[専用runner](../../scripts/run_pf_power_stability_h4.py)で1候補2meanのみ実行。directの局所gate通過と全RMSEを区別し、production screen backendを勝手に変更しない。

[pf_power_stability](../trottertracks/pf_power_stability.py)：供給matrix/specのGram・再構成診断、明示driver・同順expm、失敗時も残差を返すpower診断。既存`h46_m5_plan.weights`・screenのsigned順・`algorithm_codesign.numerics.Spectrum` observerを再利用。旧screenへは未接続。[人工REPORT](../../docs/research/pf_power_stability_artificial_20261113/REPORT.md)／[test](../../tests/check_pf_power_stability_artificial.py)。実H4未実行。

[H4・4方式signal screen](../trottertracks/h4_four_method_screen.py)／[数値leaf](../trottertracks/h4_four_method_screen_mean.py)：既存`h46_m5_plan`のsigned weights、`rte.finite_rte_distribution`、旧spectral polynomialとNumPy powerを再利用。metadata-only入口と明示的な将来science leafを分離。[REPORT](../../docs/research/h4_four_method_signal_screen_preparation_20261113/REPORT.md)／[人工test](../../tests/check_h4_four_method_screen.py)。17技術test PASS、実科学0。

[RPE decoder R3高速化・B S3接続](../../docs/research/rpe_decoder_r3_20261113/REPORT.md)：独立R3 decoderの4候補branch選択を実装し、元scalar/vector各tie規則とBのcount/RNG/loss/CIを保全。新12＋旧20の32技術test PASS。人工M12/512trialで旧vector比96.3倍、B batch接続664.6倍、実H4/S3/保存再解析/C/G0。旧vector exact-tieのみ互換fallback、公開照合後STOP。

[H4 PR全13roundの物理誤差契約](../../docs/research/h4_pr_physical_error_contract_20261113/REPORT.md)：保存S0/m1/S1＋S2の全signal・bias・allowanceをSOURCE付きで転記。raw↔X/Y・PF/finite-K/state/lawとMinkowskiの適用条件をBへ整理。新8＋既存13の21技術テストPASS、新科学・保存再解析0、十分N/global RMSEは未資格/null。次は別承認のB ideal-law decoder統計、公開照合後STOP。

[H4 PR S1＋S2・全13round nominal結果](../../docs/research/h4_pr_s1_s2_rmse_results_20261113/REPORT.md)：固定SOURCE c76ffe5dbでS1＋S2を一回完了。全13roundの経験的nominal各軸64/60/56/52/47/43/38/34/30/25/21/17/12、旧3round不変。保存2mean＋新reference4 action＋高速mean8/対照＋m2直接照合。13技術test PASS、十分N/global RMSEはnull、S3/C/G0、公開照合後STOP。

[H4 PR残10roundの証拠補完準備](../../docs/research/h4_pr_remaining_rmse_preparation_20261113/REPORT.md)：m3/m4は観測norm欠測を保持したSOURCE条件付き再利用、欠測8meanは元spectral powerへ接続。S1 reference4 action・S2高速mean8＋m2直接照合1の有限案を固定。新13＋旧35＝48技術test PASS、元m0/m1/m6不変、新実科学/保存再解析/decoder/C/G0、公開照合後STOP。

[H4 PR m1保存RMSE診断結果](../../docs/research/h4_pr_m1_rmse_results_20261113/REPORT.md)：固定科学SOURCE ac873e7fcでm1/T.2/qPF2/R4の保存scalar診断を一回実行し数値gate通過、経験的nominal Nは各軸60。m0/m6の64/38をcopy、残10未評価、十分N/全RMSEはnull。35人工・metadata回帰PASS、新mean/reference/normalizer/array/C/G0、公開照合後STOP。

[H4 m1保存RMSE準備・Full PF条件](../../docs/research/h4_m1_rmse_full_pf_preparation_20261113/REPORT.md)：H4 m1/T.2/qPF2/R4の保存scalar adapterを整備し、正式B030のV2正本・S0 packetを照合。旧Hoeffding非起動、実m1数値再解析0。DのFull PF3公式と136保存carrierをfieldで整理し、PF policyは未採択。45技術テストPASS、公開照合後STOP。

[H4 S0のB V2接続・全round検証計画](../../docs/research/pr_rpe_s0_v2_connection_20261113/REPORT.md)：S0をB af284d853のcanonical V2へ接続し、旧ID/physical SOURCEとm0/m6 nominal64/38を保持。追加一致meanはm1/m3/m4、欠測meanは8候補、m3/m4の独立pairとallowance bridgeが不足。47技術テストPASS、十分N/全RMSE/Gはnull、実数値再解析0、公開照合後STOP。

[H4 RMSE保存二時点S0結果](../../docs/research/pr_rpe_rmse_s0_results_20261113/REPORT.md)：科学SOURCE7c3e1e5dbで提案Aを一回実行、M12/κ≈24.30527458、m0 R2/m6 R128のnominal Nは各軸64/38。元数値gate通過、実RMSE/十分Nは未資格、他roundは未評価。新mean/reference/normalizer/array/C/G0、公開照合後STOP。

既存`pr_rpe_rmse`/`pr_rpe_rmse_pilot`、P1 reference checkerとscalar geometryをSOURCE不変で再利用。旧Hoeffding wrapper・physical actionはS0から呼ばない。

[PR二時点RMSE pilot準備](../../docs/research/pr_rpe_rmse_pilot_preparation_20261113/REPORT.md)：B f09のA固定経路でH4 m0/m6またはH6 m4/m8のPR scope・保存law SOURCE・部分packetを接続。M/κ/予算は未選択、旧Nと保証へ昇格せずnull保持。13新規＋31既存人工回帰PASS、新実科学計算0、公開照合後STOP。

[実RMSE評価のA技術準備](../../docs/research/pr_rpe_rmse_science_preparation_20261112/REPORT.md)：固定B82bcのmetadata検証を再利用し、全round/SOURCE・保存証拠・追加call案を科学計算なしで照合。H4 B0/full-H差とH6初期例外をBLOCKED、N/boundはnull。14新規＋29既存人工回帰PASS、実signal/N/RMSE/C/G0、公開照合後STOP。

A専用module：`src/trottertracks/pr_rpe_rmse_preflight.py`。旧RMSE/Hoeffding実装は不変。

[RMSE型Nとκ schedule](../../docs/research/pr_rpe_rmse_kappa_20261112/REPORT.md)：`trottertracks.pr_rpe_rmse`が既存finite action/normalizer/decoder・純metadataを再利用。旧Hoeffding/SOURCE不変、新17人工/mock PASS、実科学0。

R可変PRの[P1結果](../../docs/research/pr_r_detailed_n_p1_results_20261111/REPORT.md)：`trottertracks.pr_r_detailed_n_p1`が既存B grouping/A action/scalarを不変で結線。6人工/mock PASS、固定12実候補の条件付きNを公開。C/G・R採択なし。

R可変PRの[詳細N実行計画](../../docs/research/pr_r_detailed_n_preparation_20261111/REPORT.md)：既存finite action/normalizerは不変。metadata生成器と人工/保存fixtureテストを追加、分子数値計算は未実行。

[固定24 Phase0結果](../../docs/research/pr_r_normalization_phase0_results_20261111/REPORT.md)：既存K2 finite normalizer/allocation不変でlogB/attenuation/係数を記録。丸め感度を併記し、N/C/GやRの科学的採否は未評価。

[固定24 Phase0準備](../../docs/research/pr_r_normalization_phase0_preparation_20261111/REPORT.md)：`trottertracks.pr_r_normalization_phase0`は保存値/入力/SOURCE照合と既存screenの出力保存だけを追加。normalizer/allocation不変、新10人工/mock PASS、実logB計算0。

[R候補screen](../../docs/research/pr_r_screening_20261111/REPORT.md)：`trottertracks.pr_r_screening`が旧A、固定B metadata、K2 finite normalizerと整数allocationを再利用する。新solver/価格/最適化なし、人工12件。

[PR R依存N provider](../../docs/research/pr_r_dependent_n_model_20261111/REPORT.md)：`trottertracks.pr_r_dependent_n`が既存finite_rte_distribution/finite_taylor_operator・m5 allocation・scalar geometryを再利用。旧moduleは不変、人工/mock12件、実新R/保存再解析0。

[H4 DF10 N側の完了結果](../../docs/research/h4_df10_rpe_batch_results_20261110/REPORT.md)：旧scalar/action不変で全13roundの条件付きNへ接続。巨大PRの整数精度を別補助処理で照合しN_HANDOFFへ保存、元helper/原結果は保全。actual/全RPE/G未資格。

[H4 N batch adapter](../../docs/research/h4_df10_rpe_batch_preparation_20261110/REPORT.md)：`trottertracks.h4_df10_rpe_batch_provider`が既存H4 scalar・汎用行列作用・保存H4 loaderを再利用。B0/p7のscalarとPR/p1/R4/K2を保持、共通N/C APIは不変、人工13件・実科学0。

[H4価格provider×B統合](../../docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：trottertracks.pf_pr_rpe_pipeline.h4_saved_price_providerを追加。A固定部品と元会計を再利用、B環境34技術test PASS、新実科学計算0。

[H6残5roundの一括N結果](../../docs/research/h6_df19_rpe_m8_12_results_20261110/REPORT.md)：既存kernel不変で固定10候補全ideal条件付きPASS。保存値と合わせ全13roundのNを共通pipelineへ静的接続。actual/全RPE/価格Gは未資格、人工試験は既存7件の記録を再利用。

H6残5roundの[専用batch入口](../../docs/research/h6_df19_rpe_m8_12_preparation_20261110/REPORT.md)：`pf_pr_rpe_pipeline`・`h6_rpe_batch_provider`・旧D2数値kernelは不変。新runner/PLANだけで10候補を固定し、保存/FAIL coverageを接続。人工差分7件、実科学計算0。

PF/PR全RPEの[総費用接続](../../docs/research/pf_pr_rpe_cost_integration_20261110/REPORT.md)：`trottertracks.pf_pr_rpe_pipeline`が既存supplied-axis会計・比較を再利用する。`h6_rpe_batch_provider`は旧action kernelへ固定batchを接続。全体12人工/mock確認、実N/C予測/G0。

[H6 m7結果](../../docs/research/h6_df19_rpe_m7_results_20261110/REPORT.md)：既存D2 action/geometryを不変で再利用し、SOURCE dcd6a7c87でT12.8・qPF8のFull19対PR7/R16/K2を一回評価。ideal条件付きN各軸PF705/PR846、actual/全G未資格。

H6 RPE m7の[専用接続](../../docs/research/h6_df19_rpe_m7_preparation_20261110/REPORT.md)：library・数値kernel変更なし。`run_h6_df19_rpe_m7.py`が既存`h6_df19_rpe_d2`のaction/geometryを再利用する。人工差分3件PASS、実H6は別途明示承認待ち。

[H6 m6結果](../../docs/research/h6_df19_rpe_m6_results_20261110/REPORT.md)：既存D2 action/geometryを不変で再利用し、SOURCE c9aa89548でT6.4のFull19/PR7・qPF4を一回評価。ideal条件付きN各軸PF705/PR772、actual/全Gは未資格。

H6 RPE m6の[専用接続](../../docs/research/h6_df19_rpe_m6_preparation_20261110/REPORT.md)：library追加・数値kernel変更は行わず、`run_h6_df19_rpe_m6.py`が既存`h6_df19_rpe_d2`のaction/geometryを再利用する。人工差分3件、実H6は別途科学許可待ち。旧D1/D2 SOURCEとguardを保持する。

`trottertracks.h6_df19_rpe_d2`の[H6完了結果](../../docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md)：固定8候補の新signal/Nを一回評価しideal条件付きround PASS。D1と合わせm0..5、actual/全Gは未資格。SOURCE・run01・旧STOPを保全する。

`trottertracks.h6_df19_rpe_d2`：固定[H6 D2](../../docs/research/h6_df19_rpe_d2_preparation_20261110/REPORT.md)へのaction adapter。既存occupation/m5 schedule/finite Taylor/normalizerとD1算術を再利用し、matrix exponential backend・数値allowance・H6入力接続だけを追加。人工7件PASS、import時に保存入力を読まない。回路・sampler・価格・独立RPE solverなし。

`trottertracks.h6_df19_rpe_d1`：[H6係数prior・D1準備](../../docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md)。`h4_df10_rpe_d1.coefficient_prior/analyze_bound_point`を再利用し、H6 schema・二つのh候補・固定4点・部分価格表を接続。人工12件、実H6は別許可待ち。

`trottertracks.h6_df19_rpe_connection`：[H6 D1](../../docs/research/h6_df19_rpe_d1_20261110/REPORT.md)。固定binding・外部prior・uniform α・保存cohortの薄い接続。`h4_df10_rpe_connection`の算術、既存geometry/Hoeffding/horizonを再利用。人工25件PASS、実H6数値評価は別許可待ち。

[Track D×元PR/RPE共通interface](../../docs/research/track_d_resource_interface_20261110/REPORT.md)（2026-11-10）：Strong／whole-path Referenceの科学inputs・control価格を分離し、既存総集計へ薄く接続。固定H4/CDF8/LD2/S2/s1/r32/K0/h=.1・元H ground/準備費除外・13round/P0/C1 OFFは旧N/価格/総値の読取専用。人工＋read-only 24 test method PASS、実H4の新N・再集計・回路・fit0。他algorithm統合なし。次候補は同条件r16（未実行）、公開確認後STOP。 実装：`trottertracks.track_d_resource_interface`、読取CLI：`scripts/inspect_track_d_resource_interface.py`、test：`tests/check_track_d_resource_interface.py`。

[Track D run内部再走査削減](../../docs/research/track_d_bridge_run_reuse_20261110/REPORT.md)（2026-11-10）：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。run内部再走査削減で旧新の各反復word／位相／全56資源一致、70人工回帰＋保存54case PASS。前回高速版に比べnative段階q4 1.15〜1.18倍・q8 1.32〜1.33倍、q8 RSSは4〜8%増。全unit epochとCX frontier再構築／最悪O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。 実装：`src/trottertracks/track_d_bridge_run_reuse.py`、`scripts/run_track_d_bridge_run_reuse.py`、`tests/check_track_d_bridge_run_reuse.py`。

`trottertracks.track_d_stream_bridge_fast`／`track_d_stream_resources`：[同一rule高速化](../../docs/research/track_d_stream_bridge_fast_20261110/REPORT.md)。旧1Q関数／CX feedを再利用し、exact bit keyの二pass cacheとstream直結、late phase ledgerの接続を追加。旧counter／compiler／phase実装は不変。63人工回帰・保存54case・固定H4 q4/q8 exact差分PASS、一般P0／長qは未資格。

`trottertracks.track_d_native_phase160`：[Phase＋q8結果](../../docs/research/track_d_phase_q8_transfer_20261110/REPORT.md)。旧emitter出力のbinary64 phase項をexact比からMPFR160へ集約するside ledger。局所合成／1Q／CX ruleは無変更、旧binary64値も保存。q4命令列不変・q8全28価格一致、一般P0／長qは未資格。

[Track D H4 native価格移送](../../docs/research/track_d_h4_native_bridge_20261110/REPORT.md)：既存 `track_d_native_emitter`／T0／1Q／CX bridgeをbyte不変で再利用。固定raw4の全28資源が保存P0と一致。新最適化ruleなし、一般P0／長qは未資格。新runnerはI/O・観測・比較、原本非公開・公開照合後STOP。

`trottertracks.track_d_native_emitter`／`track_d_p0_1q_bridge`：[接続と技術結果](../../docs/research/track_d_native_emitter_development_20261109/REPORT.md)。既存emitterのconstructor局所binding、bounded chunk、標準definition lowering、公式ZSXXのstrict-shorter 1Q＋旧CX bridge。47人工回帰・保存54case全指標一致、一般P0_UNQUALIFIED。

`trottertracks.track_d_p0_cx_bridge`：[T1〜H0](../../docs/research/track_d_t1_h0_p0_bridge_20261109/REPORT.md)。既存T0を再利用し、両wireで隣接する同方向CXだけを相殺する。可変frontierを保持、1Q再合成や全P0互換は未実装／未資格。新8＋既存29回帰PASS、保存H4は指定2pathの資源照合だけ。

`trottertracks.track_d_native_resources`：[T0実装](../../docs/research/track_d_t0_native_resources_20261109/REPORT.md)。stdlibのnative stream counter、三指標max-plus、template連結／embedding。既存科学builder/counterは無変更、`NATIVE_EXACT`／`P0_UNQUALIFIED`を区別する。人工回帰29件PASS、新科学実行0。

[O5 C1実H4/q4結果](../../docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/Number-CDF8/元順LD2/S2/s1/r32/K0/δ=.1、保存q4_sample00/cut2・元H ground仮定/準備費除外。全256 raw/新4 post PASS、両方式CX−30、depthはStrong−178/Reference−81、相対depth差縮小97・CX差不変。旧4適格再利用、新P0合計4。χ要約の配列形状エラーSTOPを保全し保存結果集約のみ完了、科学retry0。全law/全RPE未資格、原本非公開・公開照合後STOP。

`trottertracks.track_d_head_tail_artificial`：既存C3/StrongBuilderを用いるopt-inの人工head→tail橋。local_checks/raw_checksは既存checkerを再利用し、分子入力を扱わない。

O5 metadata cohort入口は`scripts/run_track_d_o5_q4_metadata.py`。固定`track_d_cut_selection`/`track_d_cut_h4_saved`とC7-4 `event_features`を再利用し、回路や価格を評価しない。

`trottertracks.track_d_c8_1_completion`：旧raw/P0/修正版postの継承bindingと、既存unfused Aerの完了batch観測。

`trottertracks.track_d_c8_1_nofusion`：公式Aer doubleを再利用しsimulation fusionを無効化。P0/精度/全列を維持。

`trottertracks.track_d_c8_1_diagnostics`：旧checkerの例外前診断保存。

[C8-1の科学結果](../../docs/research/track_d_c8_1_q64_20261108/REPORT.md)：CPU予約修正と19人工/回帰PASSは維持したが、q64 sample00のcompiled CAR-observable検査は両方式X/YでFAIL。失敗値保存の不足も記録し、verifier/閾値の変更・H4再診断は未実施。既存判定・SOURCE・STOPを保持する。

[C8-1 handoff helper](../trottertracks/track_d_c8_1_handoff.py)：完了済みprior hash、drain後の継承・不整合拒否、stage別CPU予約を扱う薄い補助。[PLAN](../../docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)／[人工tests](../../tests/check_track_d_c8_1_handoff.py)。sampler/builder/compiler/verifierは既存を再利用し、科学条件・閾値を変えない。

[B/C全13-round scalar adapter](../trottertracks/track_d_bc_total.py)：既存B6b guard/aggregate_totalsを再利用し、事前固定A/F/Sの52価格展開と保存共分散の線形伝播だけ。[結果](../../docs/research/track_d_bc_total_20261108/REPORT.md)／[人工テスト](../../tests/check_track_d_bc_total.py)。sampling/circuit/fit/shot solverなし。通常libraryの既定動作を変更しない。

[B6b B/C adapter](../trottertracks/track_d_b6b_bc.py)／[人工検証](../trottertracks/track_d_b6b_toy.py)：B4-2固定handoff/13-round coverageとideal共通N条件、既存aggregate_totalsへのguard付き接続。[報告](../../docs/research/track_d_b6b_bc_integration_20261108/REPORT.md)。会計検証は人工値だけ、H4 totalなし。既存gadget/phase compilerと[tests/check_track_d_b6b.py](../../tests/check_track_d_b6b.py)を使用。通常libraryの既定動作を変えない。

`trottertracks/track_d_state_action_acceleration.py` は既存Aer/CAR検査用のopt-in高速化。cost回路・P0・precisionを変更しない。[C6b限定引継ぎPLAN](../../docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)の新workerだけで選択する。

The opt-in `../trottertracks/track_d_c6b_holdout.py`, `../../scripts/run_track_d_c6b.py` and `../../tests/check_track_d_c6b.py` share the existing canonical/CAR/whole-path/P0 primitives. The [C6b PLAN](../../docs/research/track_d_c6b_q8_20261107/PLAN.md) limits new execution to q8/n32 exploratory paired pricing; import does not execute a scientific run.

Track D C6a：`trottertracks/track_d_c6a_pilot.py` はfresh canonical event接続・paired統計、既存C3/C5 builder/価格検査を共用する。[PLAN](../../docs/research/track_d_c6a_paired_price_20261106/PLAN.md)。旧ライブラリ入口からは起動しない。

[C5結果](../../docs/research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[C5共通request/有限fixture価格](../trottertracks/track_d_whole_path_price.py)は、C3の共通化した演算を再利用するopt-in入口。method/axis、SOURCE/law/cut、full-wrapper scopeと資格を記録し、未検証q/母平均を誤受理しない。[固定PLAN](../../docs/research/track_d_c5_q2_20261106/PLAN.md)、人工testは`tests/check_track_d_c5.py`。

[C3 q4/q8補完結果](../../docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。

Track D C3のmode混在head境界の最小拡張は[`track_d_c3_whole_path.py`](../trottertracks/track_d_c3_whole_path.py)。StrongBuilder・既存emitterを再利用し、[PLAN](../../docs/research/track_d_c3_q4_q8_20261106/PLAN.md)のq4/q8限定fixtureを検証する。人工testは`tests/check_track_d_c3.py`。
[Track D B4-2結果](../../docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md)：既存元Hadapter・Hoeffding・horizonを変更せず再利用し、保存13-roundの条件付き十分N=14738。library追加変更なし、actual／価格未資格、mandatory STOP。

[元H用RPE統計adapter](../trottertracks/track_d_original_h_accuracy.py)は[Track D B4-1](../../docs/research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md)用。既存accuracyの共通幾何・energy本体とHoeffding／horizon／decoderを再利用。実H4のN算定は未実施。

[C2固定q2結果](../../docs/research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](../../docs/research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](../../docs/research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](../../docs/research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Stage A結果](../../docs/research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

[保存32 path Reference拡張](../trottertracks/track_d_h4_paired_reference.py) は[Track D Stage A](../../docs/research/track_d_h4_paired32_reference_20261106/PLAN.md)に限定し、既存H4 adapter／check／P0 pipelineを再利用する。通常libraryの既定動作を変更しない。

[実H4 Reference adapter](../trottertracks/track_d_h4_reference.py) は[固定word02検証](../../docs/research/track_d_h4_reference_validation_20261106/PLAN.md)専用。元CのStrongBuilder・event/phase compiler、既存directional diagonalと人工Track D shellを再利用する。共通libraryの既定動作は変更しない。

[track_d_minimal_reference.py](track_d_minimal_reference.py) は[Track D固定人工検証](../../docs/research/track_d_minimal_reference_validation_20261106/PLAN.md)専用adapter。既存DF primitive/event/native controlとphase保持compileを再利用し、分子・sampling・全RPEの実行入口を持たない。

このディレクトリが実装本体である。`scripts/` は主にここにある関数を呼び出す実行入口、
`tests/` は回帰検査である。研究全体の入口は[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

## 現行研究の中心実装

### Hamiltonian・DF分割・基底状態

- `chemistry_hamiltonian.py`、`df_hamiltonian.py`：分子HamiltonianとDF表現
- `df_partial_randomized_pf.py`：DF分割、fragment変換、state-action PF係数計算の共通基盤
- `df_trotter/`：決定論DF回路とstate-actionの低レベル実装
- `df_gpu_statevector.py`：大きめの系を想定したstate-action支援
- `io_cache.py`：入力・計算結果の保存と読込み

### partial-$S_2$・finite RTE

- `df_partial_s2.py`：決定論half sweepとRTE中央部を持つpartial-$S_2$
- `df_partial_s2_repeated.py`：partial-$S_2$の反復
- `df_rte_tail.py`：ランダム側HamiltonianのI/Z/ZZ表現
- `df_rte_circuit.py`、`df_rte_qiskit.py`：RTEイベント回路構築
- `rte.py`：有限RTEの確率、打切り、誤差計算の基礎

### コンパイル後回路コスト

- `rte_compiled_cost.py`：イベント列の構築、コンパイル、metric集計
- `df_partial_s2_cost.py`、`df_partial_s2_repeated_cost.py`：partial-$S_2$のcost集計
- `rte_connected_cluster_cost_validation.py`：局所境界補正の較正・予測・独立検証
- `rte_boundary_cost_validation.py`、`rte_boundary_pair_validation.py`：境界補正の基礎検証
- `rte_order_stratified_cost_validation.py`：Taylor次数の条件付き抽出
- `rte_system_size_cost_validation.py`：系サイズ方向の検証
- `rte_cost_angle_invariance_validation.py`：回転角とmetric cacheの診断

### RPE

- `rpe_hadamard_interrogation.py`：X/Y Hadamard interrogation回路
- `df_rpe_resource.py`、`rpe_resource_accounting.py`：誤差・shot・1 shot costの資源集計
- `rpe_short_round_optimization.py`：短いRPE段の候補選択
- `df_rpe_hadamard_compiled_cost.py`：Hadamard回路のコンパイル後cost
- `rpe_hadamard_compiled_cost_proxy.py`：反復数方向のcost proxy
- `rpe_hadamard_validated_proxy_provider.py`：検証範囲を守ってproxyを供給

## 共有サーバー向け実行基盤

- `parallel_validation_executor.py`：決定論的task ID、資源guard、bounded subprocess、
  atomic checkpoint、resume、GPU job割当、決定論的集約

科学ロジックや科学的判定は含めない。運用方法と安全上の既定値は
[`../../docs/server_parallel_validation_execution.md`](../../docs/server_parallel_validation_execution.md)を参照する。

## 検証モジュール

次のモジュールは、研究用APIそのものではなく、条件固定、比較、判定、成果物生成を担当する。

- `finite_rte_signal_validation.py`
- `pf_delta_validation.py`
- `pf_c_system_size_validation.py`
- `pauli_partial_cgs_validation.py`
- `random_circuit_cost_validation.py`
- `hierarchical_cost_validation.py`
- `rpe_round_cost_connection_validation.py`
- `rpe_hadamard_failure_validation.py`
- `rpe_hadamard_proxy_resource_validation.py`
- `rpe_allocation_sensitivity_validation.py`
- `rpe_four_round_accounting_validation.py`
- `rpe_four_round_phase_validation.py`
- `rpe_target_round_horizon_validation.py`
- `rpe_delta_round_schedule_validation.py`
- `rpe_delta_compiled_cost_validation.py`
- `rpe_hadamard_compiled_cost_benchmark.py`

同名のrunner、test、文書、artifactと合わせて読む。モジュールが実装済みでも、対象範囲が
科学的に検証済みとは限らない。

## 解析モデルと旧経路が同居するモジュール

- `partial_randomized_pf.py`：誤差配分・解析cost modelの共通関数と旧Pauli screening経路
- `df_screening_cost.py`、`df_cost_plotting.py`：DF screening・可視化

これらは現行コードから共通関数を利用する場合がある一方、保存済みの旧screening結果は
失効している。モジュール全体を旧式とみなすのでも、既存出力を現行結果とみなすのでもなく、
利用する関数とartifactのstatusを個別に確認する。

## 旧高次PF・比較用実装

次はプロジェクトの出発点である高次積公式、UWC、旧screening経路を支える。現行の
DF部分ランダム化研究と混同しない。

- `optimal_trotter.py`、`product_formula.py`、`pf_decomposition.py`
- `Almost_optimal_grouping.py`
- `qiskit_time_evolution_grouping.py`、`qiskit_time_evolution_ungrouped.py`
- `qiskit_time_evolution_pyscf.py`、`qiskit_time_evolution_utils.py`
- `plots_timeevo_error.py`、`cost_extrapolation.py`、`rz_layers.py`
- `uwc.py`、`grouped_uwc_comparison.py`、`grouped_uwc_theta_sweep.py`

これらの結果を引用するときは、先に
[`../../VALIDATION_STATUS.md`](../../VALIDATION_STATUS.md)で失効・欠落状態を確認する。

## 共通支援

- `config.py`：共通設定
- `analysis_utils.py`、`plot_utils.py`：解析・可視化支援
- `matrix_multiply.py`、`matrix_pf_build.py`：行列ベースの参照計算
- `eig_error.py`、`qpe_beta.py`：固有値誤差・位相推定の旧来支援

新しい機能は、runnerへ直接大きな処理を書くのではなく、ここへテスト可能な関数として置く。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](../../docs/research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

[Track D C7-1](../trottertracks/track_d_c7_1_q16.py)：[PLAN](../../docs/research/track_d_c7_1_q16_20261107/PLAN.md)のq16/cut8/n16専用request・価格統計/凍結モデル診断。旧q2/4・q8 guardを保全し、C3/C5/C6a/C6bの科学部品を再利用。明示runnerのみ、全law/長q未資格、mandatory STOP。

[C7-2人工adapter](../trottertracks/track_d_c7_2_toy.py)：実StrongBuilder/C3 emittersの2mode人工束縛、小規模独立exterior reference、max-plus/DAG feature。旧分子guardを変更せず、[固定PLAN](../../docs/research/track_d_c7_2_cx_structure_20261107/PLAN.md)の54人工回路のみ。

[C7-3 q32 adapter](../trottertracks/track_d_c7_3_q32.py)：新Q32Request/seed、旧q16 helperのisolated domain束縛、C7-2 max-plus feature再利用。[固定PLAN](../../docs/research/track_d_c7_3_q32_20261107/PLAN.md)専用で旧guardを変更しない。

[C7-3検証高速化adapter](../trottertracks/track_d_c7_3_acceleration.py)：exact標準gate cache、公式Aer double/unitaryは9qubit以下だけ、Reference17qubitは既存全column statevector。P0/cost QPY/precisionを変更せず、新SOURCE worker内でだけ選択。

## Track D O5 cut接続

`trottertracks.track_d_cut_selection` はC7-4 descriptorで中央±1を選び、C3 `prepare_steps` を呼ぶ人工cohort専用adapter。`track_d_cut_artificial` はC7-2の2-mode fixture/checkerと固定2ケースを接続する。新gadget/solver/compilerはない。[結果と制約](../../docs/research/track_d_cut_artificial_20261108/REPORT.md)。

[O5 q4移送の入力状態](../../docs/research/track_d_cut_h4_q4_20261108/INPUT_AND_REUSE.md)：既存cut選択器/C3/Strong/gadget/compilerのSOURCEを保全。原本不足で科学接続未実行、新adapter/gadget/テスト重複実装0。

## O5 q4保存payload接続

`trottertracks.track_d_cut_h4_saved` は既存archive/recordのlossless hash binding、`track_d_cut_h4_reuse` は中央cutの同request/P0/旧PASS資格を確認する薄いguard。既存selector/C3/Strong/gadget/compilerは変更なし。人工guardは`tests/check_track_d_cut_h4_saved.py`と`check_track_d_cut_h4_reuse.py`。[資料](../../docs/research/track_d_cut_h4_q4_completion_20261108/IMPLEMENTATION_AND_REUSE.md)。
C7-4の[保存モデル再解析](../trottertracks/track_d_c7_4_models.py)と[人工native比較](../trottertracks/track_d_c7_4_stability.py)は明示runnerだけから利用する。q≥64分子計算を含まず、[技術報告](../../docs/research/track_d_c7_4_model_stability_20261107/REPORT.md)を参照する。
