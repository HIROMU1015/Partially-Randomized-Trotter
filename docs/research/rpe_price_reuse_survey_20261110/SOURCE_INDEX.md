# 固定SOURCE・保存成果物の再利用索引

[技術調査報告](REPORT.md)。以下のcarrierは資料の取得版で、科学実行SOURCEへ付け替えない。pathはrepository相対。既存runnerや数値関数は今回実行せず、コードは本文として確認した。GitHub blob URLのbaseは `https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/<carrier>/`。

## Dの価格・shot・総費用

| 成果 / carrier | 科学SOURCE・再利用コード | 保存データ・原本への入口 |
|---|---|---|
| C6a / `e9d4e2f602d5f9e014507ca7e391719bab017b88` | `6b6667026ec8b670f610e41141d195883a9eb513`。`src/trottertracks/track_d_c6a_pilot.py`のdescribe/paired_statistics/exploratory_affine、`track_d_whole_path_price.py`、`scripts/run_track_d_c6a.py`、`tests/check_track_d_c6a.py` | `docs/research/track_d_c6a_paired_price_20261106/RESULT.md`。`artifacts/track_d_c6a_paired_price/2026-11-06/run01/`のprices.json、paired_statistics.json、exploratory_affine.json、SOURCE/result/STOP。RESULTのPRICES/STATS/SOURCEリンクも同じ原本を指す |
| C6b / `fed5f458be2f625de4e1b8c98b8cc5da6e551b14` | 限定引継ぎ`44a58b9fba5d3aba7a31d0a6b12e3aa25be9def4`。`track_d_c6b_holdout.py`、run/collect_track_d_c6b系 | `docs/research/track_d_c6b_q8_20261107/RESULT.md`、同RESULTのprices/paired_statistics/model診断/SOURCE/STOP |
| C7-1 / `86c5c9d48dbddc085f388b6521be8cd92ee49d0d` | `462f6bff66a1e3fd84a56e32aba83e9aed076257` | `docs/research/track_d_c7_1_q16_20261107/RESULT.md`、同リンクのn16価格/統計/診断 |
| C7-3 / `acea43987edae0d5eb0c186a89aa8cd7139fbd7b` | sampling/raw `35bfd0423d08114b0a24107501ce545cdfb28f32`、compile/post `94e94e8c951a16487c5b1bc1e30369d595626d31`。`track_d_c7_3_q32.py`、`track_d_c7_3_acceleration.py` | `docs/research/track_d_c7_3_q32_20261107/RESULT.md`、同リンクのn8価格/統計/引継ぎ/STOP |
| C7-4 / `64072ba81bb70db1df1b34e94adb08e78926f24b` | 実行`dd31e1e2878479eabe2e7cc27c21b26c8c7dc794`。`src/trottertracks/track_d_c7_4_models.py`、`track_d_c7_4_stability.py`、run/summarize_track_d_c7_4.py、check_track_d_c7_4.py | `artifacts/track_d_c7_4_models/2026-11-07/run02/`：`integrated_statistics.json`、`saved_prices_scalar.json`、`input_bindings.json`、`model_affine.json`、`model_frozen_C6a.json`、`model_sqrt_lane.json`、`model_basis_runs.json`、`model_lane_runs.json`、`frozen_C6a_future.json`、`model_scores.json`、`event_features.json`、`saved_numerical_recheck.json`、SOURCE/STOP |
| C8-1 / `fe361b206a6572142775ac01972545d312e5fb14` | sampling/sample00 P0 `6b1d613ded963b7851afa1c07ba8c646124dd44f`、数値修正`f1e89590c7daea12319352b85b959928ad166213`、限定完了`b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7`。`track_d_c8_1_{q64,nofusion,completion}.py`、run/check対応 | `docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md`。`artifacts/track_d_c8_1_q64/2026-11-08/run04_limited_completion/`：`paired_statistics.json`、`model_diagnostics.json`、SOURCE/result/STOP、path価格metadata、payload_index、aggregation_20261109。旧run01/run02_diagnostics/run03_nofusion_repairも別保存 |
| B4-2 / `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` | 実行SOURCEと公開SOURCEの対応はrun01/SOURCE.json・Stage_C_handoffの固定欄を使う。`track_d_original_h_accuracy.py`、`scripts/run_track_d_b4_2.py` | `artifacts/track_d_b4_2_rpe_shots/2026-11-06/Stage_C_handoff.json`、`run01/result.json`、`run01/axis_shots.csv`、SOURCE/STOP。m/q/time/attenuation/error/radius/phase/epsilon-coordinate/N/αを保存 |
| B6b / `7ffe8b281b76cef3a6fd06f43413904d6766d0ec` | `src/trottertracks/track_d_b6b_bc.py`のread_b4/common_N/price_catalog/aggregate_registered、`c1_stage3_accuracy.aggregate_totals` | `docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md`。同REPORTから科学SOURCE13114b2bの完全SHAと入力/STOPへ辿る |
| B/C / `8828b298975c88a2c0d1fd57ff6ed905233840f5` | 実行`b09b7e7abaa3b817c26657974a62c3bd3047af89`。`src/trottertracks/track_d_bc_total.py`、run_track_d_bc_total.py、check_track_d_bc_total.py | `artifacts/track_d_bc_total/2026-11-08/run01/{A,F,S}/`：`prices52.json`、`contributions.csv`、`paired_rounds.csv`、`result.json`。run rootのSOURCE/input_bindings/result/totals.csv/STOP。共分散は各result.uncertainty、説明はUNCERTAINTY.md |

Dの固定科学入力：元C SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。元H packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、r32 law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。これらは科学的identityで、approvalの証明ではない。

## 実装済み接続・関連する後続

| 取得版 | コード・成果物 | 再利用境界 |
|---|---|---|
| `1f5ea971f8ceac54f2dc00e71824ab8dabf95837` | `src/trottertracks/track_d_resource_interface.py`：ScientificInputs/ControlProfile/MethodPrices、SavedCDF8Study.common_shots/select、aggregate_supplied。`scripts/inspect_track_d_resource_interface.py`、`tests/check_track_d_resource_interface.py`、`docs/research/track_d_resource_interface_20261110/{REPORT,INTERFACE}.md` | 固定読取部分は既存値を返すだけ。aggregate_suppliedや旧bc_total.evaluateを呼べば新数値処理となる。今回はどちらも起動しない |
| `c002153a98a2f2d209a4a60f035c6bd7e36293c9` | `track_d_native_resources.py`、`track_d_stream_bridge_fast.py`、`track_d_bridge_run_reuse.py`。`docs/research/track_d_longq_structural_resource_design_20261109/`と`track_d_bridge_run_reuse_20261110/` | 固定native列の計数とq4/q8の有限P0一致。現段階で全lawの価格予測器ではない。固定比較SOURCE a01dc6620037d9feccd00560e0a6b8d203c06dc0 |
| `d68cf3c19685da586ceadbbac8193f55b9ba0fa0` | `docs/research/track_d_r16_limited_validation_20261110/{REPORT,R16_RPE_ACCURACY}.md`、`track_d_r16_price_calibration_20261110/REPORT.md`。`artifacts/track_d_r16_price_calibration/2026-11-10/run01/{prices,paired_statistics,SOURCE,result,STOP}.json` | r16 q2/4/8の48保存価格・共分散。価格実行SOURCE890bd13337066c2f263a6a14dd8107df63ca768a。旧S1 Nを再計算せず利用。r32係数と混合不可 |

## 有限時間PF/PRとlegacy18round

| 固定版 | SOURCE・成果物 | 再利用用途 |
|---|---|---|
| `1f001cfc4e8e0531279a26613be29a52c7c0640c` | `docs/research/track_a_pr_resource_evaluation_protocol_v2.md`。`artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/`：`system_conditions.json`、`candidate_registrations.json`、`H468_independent_confirmations.csv`、`stage_status.csv` | DF10/19のq/r/K/PF/prefix/input/control/compilerと直接/modelの区別。草案namespaceの保存資料は正式採用文書から参照されているが、元statusを改変しない |
| H4 `821cc0f6ced57168293ea166644024867b064ffb` | `artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/result.json`のcomparisons/0,1 | 固定2条件の独立P結果。プロトコル表の引用元。D190/A50/FPとは別cohort |
| H6 `3cccfcb2051dacf29e91473e2e1152acef1574bd` | `artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json`のscopes/0,1 | 固定2条件の独立P結果。PF136全域の直接価格ではない |
| M06-F `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | `artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json`：coherent_reoptimization、direct_compiled_rz_measurements、保存PF phase proxy/RTE bound/attenuation/N/α/価格。A0=`m06f_a0_proxy_lineage_reconciliation_v1.json`、同日fresh32_audit。`src/trotterlib/research_direction_{full_opt2,full_opt2_extension_analysis,prevalidation,ablation,full_scope}.py` | 元実行initial `26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c`／fresh32 `7753eb71844e429facdb6f32c2def6e37379541f`のdirty provenanceは元run_provenanceへ残す。carrierをclean実行SOURCEとしない |
| V1 `e520e6e396f7e0def9e522ea1cff58eb10ddfa8f` | `docs/research/pr_rpe_v1_e0_state_radius_20261103/{REPORT.md,V1_ROUND_RESULTS.csv,VALIDATION.json}`、reanalyze_saved_signals.py | PF/RTE信号と共通E0・radius・stateの対応。再実行しない |
| V2 `78c49d0b5f1d71e63b5959e78813237923bbc648` | `docs/research/pr_rpe_v2_branch_success_20261103/{REPORT.md,V2_ROUND_RESULTS.csv,BRANCH_RESULTS.json,SUCCESS_PROBABILITY_RESULTS.json}`、run_saved_v2.py | 保存Nでの条件付き分枝・成功率。新しいshot数の根拠へ付替えない |
| V3 `cc60beb51930d4311d2ded84440682e583ccafc7` | `docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/{REPORT.md,PRICE_EVIDENCE_MAP.csv,UNCERTAINTY_SCOPE.md,EVIDENCE_INDEX.json}`、validate_metadata.py | 72軸のq/r/K/N/Ĉ/SE・直接coverage・matched holdout・SOURCE selectorをそのまま利用。round費用のBOTH_AXES列を両軸で二重加算しない |
| R1 carrier `141fa7624d74aa5deacbd752d09df516cad9206f` | `docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/{REPORT.md,R1_FIXED_PLAN_RESULTS.csv,DEPENDENCE_AND_SENSITIVITY.json,MINIMAL_NEXT_ACTION.md,VALIDATION.json}`、analyze_fixed_plan.py | 共有28anchor/14group、未知共分散・systematic帯・break-even・低寄与域の情報価値。既に実施済みの再解析結果を読む。carrierに含まれる本文は2026-11-03の原版 |

B/normalizationは同M06-F carrierの`artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json`の`points[*].finite_distribution_normalization`。同pointのq_m/r_m/K_m/attenuation/state_resultsと、最終opt2のround/axesを条件対応させる。biasはPF proxy／RTE bound／実signal残差を別項目として扱う。有限時間H4 Pの保存Nは`comparisons[*].saved_shots`。保存compiled RZやNを使い、未保存CX全格子やBを新計算で埋めない。

一般API `rpe_resource_accounting.py`、`rpe_short_round_optimization.py`、`df_rpe_resource.py`には既存実装があるが、signal・N再計算や実build/compileを含む入口を静的調査で呼ばない。再利用時は対象domainとSOURCE版を固定する。

## Track B/C・新A

- C最終共同選択：`c5a3c102199fe97f4d5c21837c059ae2684f5ef2`の`docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md`、`src/trottertracks/track_c_closed_p3_joint_opt*.py`、`track_c_closed_p3_resources.py`、`track_c_bottleneck.py`。**最終データは**`artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended/FINAL.json`等（原報告のリンクを正本とする）。training/holdout再分類・最終独立holdout・shared cell covarianceを再利用する。
- C統合：`b3c81c2677988c7b9491eeee31fbcc0f9b29b078`の`docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md`、`track_c_df_truncation(_pr).py`。Full/Truncated/PRの公平な対照とround別LDの既存成果。別basis/rank/sectorの係数移植はしない。
- 新A：`224440f740051ed5794e8db87a57908fc02dd02f`のPROJECT_MAPとV3/V5/CT4資料。max-plus frontier、回転分類、有限strict-T価格を参照。全support期待T／全18round総Tは別資格。Q3後続は`08f9cbe77456ec29d50a992d7165c4d726df7bf7`の`docs/research/pr_reference_feedback_q3cd_20261110/`を同日統合調査が参照している。canonical NをCDF8へ移さない。
- Track B/O2：`d28164b5d51e0ca5a0ade74ded8ff176b443dab8`の`docs/research/o2_research_results_integration_20261108/{O2_RESEARCH_SUMMARY,O2_EVIDENCE_INDEX}.md`。finite-bit moment/range/weightと価格MC・量子trialの区別に方法を再利用。CDF8 K0との科学条件一致は認定しない。
- 再利用した横断調査：`2e79d80dc9cce2885580e50fe6f1705da8fa41af`、同日接続調査`1f5ea971f8ceac54f2dc00e71824ab8dabf95837`、元Track A調査`162f3d3f0f42973048fd14a7a6624e147a1e961a`。再棚卸し・候補再実装を避け、今回必要な価格域の差分だけを追加した。

## 調査の限界と保全

Git object／保存JSON/CSV欄／SOURCE本文・関数定義を確認した。数値payloadの再評価、QPY decode、module import、科学test、全raw再監査は行わない。原結果に記された検証・hash照合は**既存結果の引用**で、今回の再実行とはしない。価格がないことは今回固定資料内のcoverageを指し、全ての私有保存先に存在しないとの断定ではない。

本索引は原runのgrant/ledger/marker/STOPやSOURCEを変更しない。非公開原本は元保存先のままにし、GitHubへ追加公開しない。公開照合後mandatory STOP。
