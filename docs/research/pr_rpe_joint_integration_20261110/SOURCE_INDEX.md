# 固定出典・再利用API・観測範囲

[REPORT](REPORT.md)の科学的記述は以下の保存文書とSOURCE本文に基づく。閲覧／公開commitと科学実行SOURCEを区別する。今回は研究moduleをimportせず、Git object・本文・ASTとJSON keyを読んだ。全branch／全raw／他SERVER非公開資産の網羅調査ではない。未取得の後続作業はUNKNOWNである。

## 1. 固定した研究成果

URLのprefixは `https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/`。

| ID | 閲覧／公開commit | 必須資料・実行SOURCEへの入口 |
|---|---|---|
| D-TOTAL | 8828b298975c88a2c0d1fd57ff6ed905233840f5 | docs/research/track_d_bc_total_20261108/REPORT.md、SCENARIOS.json。artifacts/track_d_bc_total/2026-11-08/run01/SOURCE.json、input_bindings.json、A/F/Sのprices52.json。科学実行b09b7e7abaa3b817c26657974a62c3bd3047af89 |
| D-TECH | c002153a98a2f2d209a4a60f035c6bd7e36293c9 | docs/research/track_d_bridge_run_reuse_20261110/REPORT.md、性能／正確性／STOP。有限検査器SOURCE a01dc6620037d9feccd00560e0a6b8d203c06dc0 |
| D-B1 | d3913866e6ade4ab97ac02645fed91b2d5fcfa75 | docs/research/track_d_b1_c1_validation_20261106/RESULT.md、artifacts/track_d_b1_c1_validation/2026-11-06/run01/B1/result.json。科学実行8c6eeb02e89f6cf14870789719493e75482f62f8 |
| D-B4 | 07d18bfe1d5145d6f8de026c0d994d5cb1c783c8 | docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md、artifacts/track_d_b4_2_rpe_shots/2026-11-06/Stage_C_handoff.json、run01/result.json、axis_shots.csv、SOURCE.json。元実行／公開SOURCE対応はhandoffの固定fieldを参照 |
| D-B6b | 7ffe8b281b76cef3a6fd06f43413904d6766d0ec | docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md、REPORT.md。科学実行13114b2b（原SOURCE.jsonに完全SHA） |
| C7-4 | 64072ba81bb70db1df1b34e94adb08e78926f24b | docs/research/track_d_c7_4_model_stability_20261107/MODEL_REPORT.md。科学SOURCE dd31e1e2878479eabe2e7cc27c21b26c8c7dc794。A/Sはq2〜32使用、FはC6a凍結 |
| D-q64 | fe361b206a6572142775ac01972545d312e5fb14 | docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md。旧FAIL/STOPと後続完了の両方を保全 |
| CROSS | 2e79d80dc9cce2885580e50fe6f1705da8fa41af | docs/research/pr_rpe_cross_track_20261109/{REPORT,EXISTING_RESULTS_AND_OVERLAP,MINIMUM_VALIDATION_PLANS}.md。重複調査の基礎。HP/Q3/Bridge後続で更新して読む |
| A-SURVEY | 162f3d3f0f42973048fd14a7a6624e147a1e961a | docs/research/track_a_algo_improvement_survey_20261109/{README,TRACK_A_RESEARCH_SURVEY,K3_A3_TECHNICAL_SURVEY}.md。既存別担当の静的調査として利用し、独立科学保証にしない |
| A-HP | e162ef708d7f721cd9c95ae07a833f7301d2b449 | docs/research/track_a_b2_p1_hp_branch_shots_20261107/REPORT.md。新5 arm科学SOURCE509f829f5d58f67887f2ade5301965051f89fa04、HP信号結果3e316e071fe671aa61b31bc89f83bf4380d0e92f |
| C-OPT | c5a3c102199fe97f4d5c21837c059ae2684f5ef2 | docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md、extended/FINAL.json等。原科学head e83968cca8a69cc1d6d4c9c9861bed452f74800fと段階履歴はSOURCE.json/bundleへ保存。公開SHAへ付け替えない |
| C-ALL | b3c81c2677988c7b9491eeee31fbcc0f9b29b078 | docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md。Truncated da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13、相殺92495a04d52c65142ed16b3abe977345065f9745を含む |
| O2 | d28164b5d51e0ca5a0ade74ded8ff176b443dab8 | docs/research/o2_research_results_integration_20261108/{O2_RESEARCH_SUMMARY,O2_EVIDENCE_INDEX}.md。G6/Parent/IS/CV/LCU・負結果の索引 |
| RC | 91417fb3ef75e7eec0849ad39f0649d170754664 | docs/research/research_c_h4_strong_d_relocation_results_20261107.md。own-DFの条件を確認。実行SOURCE e6652ab841aae5ae700286bf1ab249ad4fb85461 |
| Q3 | 08f9cbe77456ec29d50a992d7165c4d726df7bf7 | docs/research/pr_reference_feedback_q3cd_20261110/{REPORT,NATIVE_GUARANTEE,RPE_AND_COST_RESULT}.md、artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/SOURCE.json。元科学canonical f2638a7845685a119619382b850441f5d8d33713、Q3-B1795d72b704cfd89aeac7c0075ebb576406ba624 |

新A/C-OPT/CROSSの固定公開REPORTはGitHub接続でも取得した。主要refをgit ls-remoteで観測し、new-A=08f9cbe7、C統合=b3c81c26、O2=d28164b5、A調査=162f3d3f、D評価器=c002153aを確認した。C-OPTの旧branch名は今回のremote照会に現れないが、固定c5a3の文書は取得できる。旧branch名の不存在を成果消失とは扱わない。

## 2. 元PR/RPEとDの具体的API

以下はc002153aの保存treeで確認した。今回のbranchはこれらを変更しない。

| SOURCE path | 主なsymbol | 再利用と制約 |
|---|---|---|
| src/trotterlib/df_partial_s2.py | DFPartialS2Preparation、prepare_df_partial_s2 | H_D block／tail／identity／exact λ_Rのbinding。新入力のprepareは科学処理を含むため今回呼ばない |
| src/trotterlib/rte.py | normalize_involutory_tail、finite_rte_distribution、finite_rte_attenuation、select_rpe_round_taylor_orders | signed law、finite cutoffとnormalization。attenuation B_K^(−r×occurrences)はshot側へ接続 |
| src/trotterlib/rpe_resource_accounting.py | RPERoundSpecification、RPEErrorAllocation、RPEPFErrorModel、evaluate_rpe_round_candidate、_hoeffding_shots、build_rpe_resource_summary | 一般短roundの誤差／shot／単価。state/radius/empirical PF資格とcost scopeを維持 |
| src/trotterlib/rpe_short_round_optimization.py | optimize_rpe_short_round、optimize_rpe_short_rounds | 既存r/K有限grid、cost_metricを明示。直接q≤4。全horizon optimizerではない |
| src/trotterlib/df_rpe_resource.py | DFLevel5RCompiledCostProvider | ordinary diag(I,U)・境界最適化・直接compile。古典sample_countは量子shot倍率ではない |
| src/trotterlib/rpe_allocation_sensitivity_validation.py | validate_rpe_allocation_sensitivity、_evaluate_scenario | cost感度α式を確認。全validatorはprovider/数値評価を呼ぶのでそのまま静的用途に使わない |
| src/trotterlib/rpe_target_round_horizon_validation.py／rpe_four_round_phase_validation.py | required_rpe_round_horizon、reconstruct_rpe_phase | 必要horizon／dyadic branch decoder。元H adapterのprincipal prior等の制約を保持 |
| src/trottertracks/track_d_original_h_accuracy.py | OriginalHTarget、SignalBinding、OriginalHRoundEvidence、assess_round、assess_energy、decode_energy | 元H operator誤差をsignalへ一回。unweighted X/Y・uniform alpha_axis・連続dyadic horizonに限定 |
| src/trottertracks/c1_stage3_accuracy.py | _assess_raw_signal、_assess_energy_interval、aggregate_totals | 幾何・shot・equal-X/Y費用の共通部。own-DF assess_energyを元Hへ誤用しない。共通helperだけではstate/aliasを資格付けない |
| src/trottertracks/track_d_b6b_bc.py | read_b4、binding、common_N、price_catalog、aggregate_registered | 固定B4/law/target/52価格のguardと既存集計接続。common_Nの宣言booleansは証明ではない |
| src/trottertracks/track_d_bc_total.py | load_inputs、model_value、price_records、validate_records | 保存mean／A/F/S／価格scopeを既存会計へ接続。frozen scenario/q外を拒否する |
| scripts/run_track_d_b4_2.py／run_track_d_bc_total.py | prepare/evaluate、campaign入口 | 既存科学実行runner。今回は起動しない。元SOURCE/STOPを再開しない |

B4 result.jsonのschemaはroot target/binding/rows/alias/horizon/energy/ideal_lane/actual_lane等。13行にはm/q/time、attenuation、original_H_corrected_operator_error、radius_lower、phase_slack、epsilon_coordinate_X/Y、N_X/Y、alpha_X/Yがある。今回はkeyと行数だけを確認し、新しいsum・N・価格を計算していない。

## 3. 別系列から再利用できるSOURCE

| 固定tree | SOURCE／確認symbol | 接続上の注意 |
|---|---|---|
| C-OPT c5a3 | src/trottertracks/track_c_closed_p3_joint_opt.py：menu、point、optimize、freeze、evaluate／track_c_bottleneck.py：allocate、account | Cのh/rate/α・training/holdoutの実装。allocateは18×2とfailure1/20に固定、CDF8へ直呼び不可 |
| C-ALL b3c81c26 | src/trottertracks/track_c_df_truncation_pr.py：partition、generator、screen、qualify_point、build_isometries、event_circuit、price_one、independent_one | partitionはscalarとsigned symbolsを再構成する。新LD評価はPF/mean/native/価格を要する。後発fileなのでc5a3 treeには存在せず、正しいb3c81c26版を使う |
| Q3 08f9cbe7 | src/trottertracks/pr_reference_feedback_q1.py：prefix_from_counts、feedback_from_prefix、feedback_readout、decode_counts | prefix・unknown radius・feedback/decoderの既存資産。CDF8のatan2 decoderの同義語ではない |
| Q3 08f9cbe7 | src/trottertracks/pr_reference_feedback_q3cd.py：AdaptiveReadout、physical_shot、decode | canonical FixedInputsとarmを厳密bind。physical_shotはfresh event生成・full m17 build/compileを呼ぶため今回実行0 |
| Q3 08f9cbe7 | pr_reference_v6q_producer／consumer／schedule、Q3-A/Bの区間・branch資料 | B1 vacuum／actual F.inverse／native context移送を再利用する候補。Closed/joint構成への適格性は別確認 |
| D-TECH c002153a | track_d_native_resources、track_d_stream_bridge_fast、track_d_bridge_run_reuse | 固定nativeの有限校正・整合性検査に保全。今回再計数・高速化・新P0なし |

既存の人工PASS／条件付きRPE保証／価格MC／構造的価格境界／P0保存一致は、それぞれ別の証拠である。異なる SOURCE の改善率を掛け合わせたり、すべてが同時に達成されたpolicyとして表示したりしない。

## 4. branch・保全

今回の親はc002153a、branchはcodex/pr-rpe-joint-integration-plan-20261110。既存resource-eval branchは有限評価器として完了状態を保持し、新AとCの作業branchも変更しない。既存横断調査branchの文章を上書きせず、今回の差分を本directoryへ記録する。公開後STOP。新たなSOURCE実行や研究採択は行わない。
