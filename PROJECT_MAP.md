# プロジェクト案内

[PF・truncated PF・PR・Full-H RTEのRMSE coverage](docs/research/pr_rpe_method_coverage_20261113/REPORT.md)（2026-11-13）：既存controllerを拡張しH4 B0/p7とFull PFを分離。供給N/Cの資格と236軸の保存metadataを整理、新14＋回帰85＝99技術test PASS。実N/C/G未実行、公開後STOP。

[PR–RPE RMSE A/B/C最終統合](docs/research/pr_rpe_rmse_abc_integration_20261112/REPORT.md)（2026-11-12）：A510fbacdの固定grid/早期override・nominal/null NとC09489eedの受動lookupをB最終controllerへ接続。新12＋回帰123＝135技術test PASS。実RMSE/価格は未資格、実N/C/G/最適化0、公開後STOP。

[PR–RPE RMSE型controller](docs/research/pr_rpe_rmse_controller_20261112/REPORT.md)（2026-11-12）：A/C共通仕様をd562bc38dで先行公開後、L_D/要求PF幅/M/共通κのwhole-schedule生成・資格判定・exact N×C・有限menu選択を実装。新17＋回帰54＝71技術test PASS。RMSE/κ数学定義は未採択、旧高確率N/C1 falseを保持、新実N/C/G0、公開後STOP。

[C1実compiled価格32 recordの取込み](docs/research/rpe_c1_price_results_intake_20261111/REPORT.md)（2026-11-11）：C公開2e6a0cdf5を既存adapter 724d313bで受入。凍結7eeabf0→検証の順序・SOURCE・8セルidentity・law・compiler・値を照合し、保存校正/新校正/独立検証/model予測を別scenarioで保持。Aの全R menu＋C1でcoverage更新（新R4セルはN欠測、長q・大Rは価格未取得、G未評価）。130技術test、新科学計算0、公開後STOP。

[R可変C価格のB受入interface](docs/research/rpe_r_price_intake_20261111/REPORT.md)（2026-11-11）：Cの`describe_request`をbyte一致で取込み、固定C1 PLAN 8セルを最新Bのidentity/law/recordへ照合。校正/独立検証/model/rare/欠測を分けて公開commitから受入れ、固定R価格の新R移植を停止。130技術test（129 PASS＋任意skip1）、実C1未取込み・科学計算0、公開後STOP。

[A0実結果24件の受入/C1接続](docs/research/rpe_normalization_results_intake_20261111/REPORT.md)（2026-11-11）：公開4745bb676を既存adapterで無変更取込み。SOURCE9/入力hash7一致、診断24有効・欠測/STOP/数値未資格0、C1共有2/他6未補完、全A86候補を保持。既存47技術test＋保存fixture PASS、新数値計算0、公開後STOP。

[A0 normalization受入準備/C1接続](docs/research/rpe_normalization_intake_20261111/REPORT.md)（2026-11-11）：固定24件の診断専用adapter、欠測/STOP/数値未資格の保全、同一SOURCE/requestでH6重複2点だけ共有。新8＋既存39＝47技術test PASS、実A0未消費・科学計算0、公開後STOP。

[A R候補×C Phase1の計画整合](docs/research/r_screening_price_alignment_20261111/REPORT.md)（2026-11-11）：C8セルのうち6はA自動menu内、H4新R8/q8・q16の2点は明示校正/検証laneを保全。A24 scalarとC64+16=80 wrapperを別scopeとして整理。m12 N例とC小qは重複0、長qへの価格移送を資格しない。7 metadata技術test PASS、新科学計算0、公開後STOP。

[A R依存N・C H6価格のB統合](docs/research/rpe_provider_integration_20261111/REPORT.md)（2026-11-11）：固定A RBatch/partition・C provider/coverageを再利用し、複数R adapter、既存loader、保存入力→batch→exact費用/R選択を接続。83技術test PASS。固定52行/R候補54行の静的coverage、補正整数N保持・新R欠測・baseline allowance比較資格を分離。実N/C/再解析/fit/compile/G/R最適化0、公開照合後STOP。

[有限R候補の総費用選択pipeline](docs/research/rpe_r_selection_20261111/REPORT.md)（2026-11-11）：R-only menu・N/C full recipe/law/domain照合・exact有理数積和・round別点選択を追加。固定Rモデルの流用、欠測を除いた全menu最適化の表示、round PASSから全RPE成功への昇格を防ぐ。人工12＋既存27＝39件PASS、実N/C/再解析/compile/実G/実R最適化0。A/C共通仕様と未承認の有限実行案を公開しSTOP。

[H4 DF10全13round N結果](docs/research/h4_df10_rpe_batch_results_20261110/REPORT.md)：SOURCE90dcc3ed2で保存m4 scalar2＋新11時点22候補を一回評価、全ideal条件付きPASS。PF各軸705〜764、PR m12約3.78e75。原本保全・長時間PRの整数丸めだけ上方向検証したN_HANDOFFをCへ渡す。actual/全RPE/C/G未資格、公開照合後STOP。

[H4 DF10 N batch整備](docs/research/h4_df10_rpe_batch_preparation_20261110/REPORT.md)：保存m3 N保持、m4保存signal scalarとfresh11時点22候補を共通N callbackへ接続。人工/mock/静的fixture13件PASS、実H4再解析・新N/C/G0、C interface/H6不変、科学許可待ち・公開照合後STOP。

[H4価格providerのB接続完了](docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：Aの凍結Mq・固定依存をB統合branchへ取込み。B factory→run_missing、null取得扱いの修正、paired cohort/資格不足の再呼出し抑止を接続。A8/B12/結合7/H6 N専用7の計34技術test PASS、新実科学計算・価格予測・実G0。全law/長q/CI/全RPE採用は未資格、公開照合後STOP。

[H6残5round一括結果](docs/research/h6_df19_rpe_m8_12_results_20261110/REPORT.md)：SOURCE fca50d36fの固定10候補を一回評価し全てideal条件付きPASS。各軸Nはm8..12 PF705/706/707/709/714、PR1015/1462/3033/13064/242418。保存m0..7を再計算せず全13roundのNへ接続、actual/全RPE/G未資格、C0、公開照合後STOP。

[H6残5round一括準備](docs/research/h6_df19_rpe_m8_12_preparation_20261110/REPORT.md)：既存統合N providerを不変で再利用し、m8..12/T25.6..409.6・固定10候補のN専用runnerと保存/FAIL coverageを追加。人工7件PASS、実H6科学計算0。SOURCE20 fileを固定し、明示実行許可待ち・公開照合後STOP。

[PF/PR総費用pipeline統合](docs/research/pf_pr_rpe_cost_integration_20261110/REPORT.md)：保存schedule/N/C・候補/科学/回路binding・不足batch・既存RZ会計・共有不確かさ・比較を接続。人工/mock12件PASS、実H4/H6新signal/N/C予測/G0。m8逐次追加を優先せず、H6残5時点10候補の一括案とH4保存signal再利用を準備。価格資格とH4 fresh provider接続は未完了、公開照合後STOP。

[H6 RPE m7結果](docs/research/h6_df19_rpe_m7_results_20261110/REPORT.md)：SOURCE dcd6a7c87、T12.8/qPF8/δ1.6の固定Full19対PR7/R16/K2を一回評価。両方ideal条件付きPASS、X/Y各軸PF705/PR846。旧m0..6不変、8個の異なるroundへ接続。actual/全G未資格、価格0、公開照合後STOP。

[H6 RPE m7準備](docs/research/h6_df19_rpe_m7_preparation_20261110/REPORT.md)：T12.8/qPF8/δ1.6のFull19対PR7/R16/K2を既存actionへ接続。人工3件PASS、実H6新計算0。m0..6と旧SOURCE保持、指示§11の明示実行許可待ち、公開照合後STOP。

[H6 m6 run02](docs/research/h6_df19_rpe_m6_results_20261110/run02/REPORT.md)：利用者の再度の実行指示を受け、SOURCE49523b11d・数値SOURCE不変で同条件2候補を別runとして評価。両方ideal条件付きPASS、各軸PF705/PR772、全candidate数値欄はrun01と一致。run01/coverage保持、7個の異なるroundのまま、価格0、公開照合後STOP。

[H6 RPE m6結果](docs/research/h6_df19_rpe_m6_results_20261110/REPORT.md)：SOURCE c9aa89548、T6.4/qPF4/δ1.6のFull19対PR7/R16/K2を一回評価。両方ideal条件付きPASS、各軸NはPF705/PR772。保存値と合わせm0..6、残りm7..12。価格作業・G0、actual/全RPE未資格、公開照合後STOP。

[H6 RPE m6準備](docs/research/h6_df19_rpe_m6_preparation_20261110/REPORT.md)：T6.4/qPF4/δ1.6のFull19対PR7/R16/K2だけの専用入口。旧D2数値kernel・8 ID guardは不変、人工差分3件PASS。実H6 signal/phase/Nは未実行、価格照会・予測なし。別途科学実行許可待ち、公開照合後STOP。

[H6 DF19 RPE D2結果](docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md)（2026-11-10）：固定m0/1/2/5の8候補をSOURCE e0503b769で一回評価しideal条件付きPASS。新各軸NはPF全705、PR705/708/719/737。保存D1と合わせm0..5、残りm6..12。3新reference・保存T=.4再利用、人工7件、対角化/compile/G0、actual/全RPE未資格、公開照合後STOP。

[H6 DF19 RPE D2準備](docs/research/h6_df19_rpe_d2_preparation_20261110/REPORT.md)（2026-11-10）：m0/1/2/5のPF/PR計8候補を提案、m5はq_PF2/δ1.6。T=.4の保存referenceとp5/7 tail定数を確認し、新referenceは3時点だけの計画。固有値問題を避けるaction接続仕様、C未取得を明示。新signal/N/G0、別途科学実行許可待ち、公開照合後STOP。

[H6 DF19 RPE D1結果](docs/research/h6_df19_rpe_d1_results_20261110/REPORT.md)（2026-11-10）：係数priorからh=.1、M12・13round・α=.05/26を導出。保存m3/4の各軸NはPF705/705、PR765/721でideal条件付きPASS、3 cohort別部分費用を評価。actual/全RPE成功・全Gは未資格、新signal/build/compile0。SOURCE4c928b90、公開照合後mandatory STOP。

[H6係数prior・D1実行準備](docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md)：H4完了prior/scalar helperを再利用、H6 loader・h二候補STOP・固定4点と3 cohort別部分費用CLIを用意。人工7＋5＝12件PASS。添付指示§9により実H6 prior/phase/radius/N/部分Gは別途明示許可待ち、公開照合後STOP。

[H6 RPE D1実装・人工検証](docs/research/h6_df19_rpe_d1_20261110/REPORT.md)（2026-11-10）：固定H4 adapterの入力層だけをH6 bindingへ拡張、既存15＋H6差分10の人工25件PASS。実H6は保存4点のread-only inspectionのみ。prior不足を保持し、新phase/radius/N/部分G・実target horizonは未承認・未実行。実行計画と公開照合後mandatory STOP。

[H6 DF19のRPE接続準備](docs/research/h6_df19_rpe_preparation_20261110/REPORT.md)（2026-11-10）：保存DF-H ground／1 kcal/mol／αtotal=.05／P=0／compiled RZを今回条件として、H6入力・signal・直接価格4点を読取専用で接続整理。H4 adapterの共通化差分、h/prior/β候補、未評価roundと最小scalar計画を報告。新signal/N/G・sampling/build/compile・詳細レビュー0、公開照合後mandatory STOP。

[RPE shot・誤差配分の接続準備](docs/research/rpe_shot_error_connection_20261110/REPORT.md)（2026-11-10）：legacy DF12の18round、Track D CDF8の13roundと全費用会計、現H4 DF10/H6 DF19の有限時間signal/shotを静的照合。再利用関数、q_RPEとq_PF、αaxis=.025と全round配分、候補IDとN/Cの対応、未確定条件と最小評価計画を整理。新N/signal/総費用再解析・sampling・build/compile・詳細科学レビュー0、公開照合後mandatory STOP。

[Track D×元PR/RPE共通interface](docs/research/track_d_resource_interface_20261110/REPORT.md)（2026-11-10）：Strong／whole-path Referenceの科学inputs・control価格を分離し、既存総集計へ薄く接続。固定H4/CDF8/LD2/S2/s1/r32/K0/h=.1・元H ground/準備費除外・13round/P0/C1 OFFは旧N/価格/総値の読取専用。人工＋read-only 24 test method PASS、実H4の新N・再集計・回路・fit0。他algorithm統合なし。次候補は同条件r16（未実行）、公開確認後STOP。

[PR＋RPE共同最適化への統合調査](docs/research/pr_rpe_joint_integration_20261110/REPORT.md)（2026-11-10）：利用者指示により評価器追加開発から総資源共同最適化の接続へ重点移行。元PR/RPE API・D-CDF8の既存13-round接続・Closed P3共同選択・新A Q3後続・強PF再資格を静的照合。推奨初手は固定CDF8のα配分×control方式の限定scalar検証案。新N／価格再解析／fit／回路／全round再集計0、実行計画のみ。公開確認後STOP。

[Track D run内部再走査削減](docs/research/track_d_bridge_run_reuse_20261110/REPORT.md)（2026-11-10）：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。run内部再走査削減で旧新の各反復word／位相／全56資源一致、70人工回帰＋保存54case PASS。前回高速版に比べnative段階q4 1.15〜1.18倍・q8 1.32〜1.33倍、q8 RSSは4〜8%増。全unit epochとCX frontier再構築／最悪O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。

[Track D 同一rule stream bridge高速化](docs/research/track_d_stream_bridge_fast_20261110/REPORT.md)：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。旧新8wrapperのword／phase／round／56資源一致。同時16-process比較でq8約2.09〜2.16倍高速・RSS約53〜54%減（native列以降）。63人工回帰＋保存54case PASS、O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。

[Track D Phase精度＋保存q8移送](docs/research/track_d_phase_q8_transfer_20261110/REPORT.md)：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／LD2／S2/s1/r32/K0／h=.1、q8_sample00／cut4／C1 OFF、元H ground仮定・準備費除外。raw4予測固定後、保存P0全28資源一致。53人工回帰PASS、q4 native列不変でphase集約差を約1e−11へ縮小。q8は20反復・約491〜494 MiB/worker、一般P0／長q未資格、新P0／sampling／fit／N／全RPE更新0、公開照合後STOP。

[Track D 保存H4 native価格移送](docs/research/track_d_h4_native_bridge_20261110/REPORT.md)：元H／Linear H4・1Å・STO-3G・Nα=Nβ2、number-CDF8／LD2／S2/s1/r32/K0／h=.1、q4_sample00／cut2／C1 OFF・元H ground仮定／準備費除外。raw4予測を先にcommit固定し、保存P0全28資源と一致。4並列24.96秒・peak276〜281 MiB/worker、49人工回帰PASS。宣言phase差を記録し新作用保証なし、一般P0／長q未資格、新P0/build/simulation/fit/N0、公開照合後STOP。

[Track D native emitter・1Q bridge](docs/research/track_d_native_emitter_development_20261109/REPORT.md)：既存head／tail chunk／Referenceをnative streamへ接続。新10＋旧37＝47人工回帰PASS、保存54人工case全てで位相付き作用と保存P0全7資源が一致。一般P0／H4／長qは未資格。新P0・H4実行／再解析・fit/N/13round更新0、旧SOURCE・結果・STOPを保全、公開照合後mandatory STOP。

[Track D T1〜H0](docs/research/track_d_t1_h0_p0_bridge_20261109/REPORT.md)：両wire隣接CX bridge、54人工case中70 native列の計数一致、固定3人工P0、保存H4 q4/q64_sample00の8絶対価格照合を完了。1Q/Gaussian/CSWAPの全P0 bridgeは未資格。新H4 build/P0/simulation・sampling・fit/N/13round更新0、公開照合後STOP。

[Track D T0 native資源計数](docs/research/track_d_t0_native_resources_20261109/REPORT.md)：`trottertracks.track_d_native_resources`は確定rz/sx/x/cx/measure streamのcount・三depth・size・幅、max-plus transfer／embeddingを実装。人工回帰29件PASS、P0対応未資格。旧SOURCE・結果・STOPは不変、新H4/P0/再解析/fit/N/13round更新0。

[Track D branch整理](docs/research/track_d_branch_consolidation_20261109/FINAL_REPORT.md)：43 Track D履歴を統合、O2は独立context保持、既存64人工技術回帰PASS。認証復旧後に固定38 remote枝を削除・全履歴保全、元対象7＋統合1を保持（q64関連5保護、q32未追跡1 HOLD）。新科学計算0、公開照合後mandatory STOP。

[O5 C1実H4/q4結果](docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/Number-CDF8/元順LD2/S2/s1/r32/K0/δ=.1、保存q4_sample00/cut2・元H ground仮定/準備費除外。全256 raw/新4 post PASS、両方式CX−30、depthはStrong−178/Reference−81、相対depth差縮小97・CX差不変。旧4適格再利用、新P0合計4。χ要約の配列形状エラーSTOPを保全し保存結果集約のみ完了、科学retry0。全law/全RPE未資格、原本非公開・公開照合後STOP。

[O5 C1人工結果](docs/research/track_d_o5_c1_head_tail_artificial_20261108/REPORT.md)：固定2-mode/3-block q2_mixed/cut1/h=.1/τ=.2/K0、scalar .17/identity .04（分子geometry/basis/rank/LD非該当）。head→tailの既存relative橋でlocal7 positive/7 negative・raw41比較・post8 PASS。Strong/Reference双方でRZ−5/CX−2/depth−7/size−11、幅不変、相対R−S差は不変。新P0 4・旧4適格再利用。保存型エラーの旧STOPを保全し、SOURCEaf6fc07fで同scope未完了処理を継続。H4/fit/N/13round/q64操作0、公開照合後mandatory STOP。

[O5 C1固定人工PLAN](docs/research/track_d_o5_c1_head_tail_artificial_20261108/PLAN.md)：q2_mixed/cut1だけ、既存relative橋をhead→tailへ接続。局所/全入力PASS後最大8 P0、Strong/Reference公平比較、q64非介入・公開後STOP。

[O5候補C1：head-tail静的調査](docs/research/track_d_o5_c1_head_tail_static_20261108/REPORT.md)：閉じたhead出口→tail入口の異frame Gaussianを既存relative変換へ接続する候補を同定。head/tail内部共有・step融合は実装済み、P0局所重複と全接口UNKNOWNを区別。C1継続案は一種類の最小人工判別までで未採択。新コード/人工fixture/correctness/P0/保存価格再解析/fit/N/13round0、q64非照会・非介入、公開照合後mandatory STOP。

[O5保存q4・16 path cut構造調査PLAN](docs/research/track_d_o5_q4_cohort_metadata_20261108/PLAN.md)：C6a保存16本metadataだけを固定descriptor選択器で一回評価。sample00 decision再利用・新選択15、H4回路/compile/simulation/価格再解析0。公開照合後mandatory STOP、B後続/C実装へ自動進行しない。

[O5構造調査結果](docs/research/track_d_o5_q4_cohort_metadata_20261108/REPORT.md)：全16本中央cut2、非中央0。有限calibration metadata結果、追加H4 compile保留・C接口融合は次候補案、実装未開始。

[C8-1 q64残作業の限定実行PLAN](docs/research/track_d_c8_1_q64_20261108/COMPLETION_PLAN.md)：利用者承認により旧raw1/P04/post4を継承し、最終pairedχ/sample00 GO後のみ残7raw/最大28P0・post。追加sampling0、固定q64/cut32/精度/P0/モデル不変、23人工回帰PASS。新SOURCE固定後に明示実行、公開照合後mandatory STOP。

[C8-1保存4回路の数値修正結果](docs/research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_RESULT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。同じsample00/4 P0に対し、SOURCE46ce57e1で旧FAILの値を保存し、SOURCEf1e89590のsimulation fusion無効化で4 postcheck全PASS。固定精度不変、旧原本46項目/STOP保持、新sampling/raw/compile0。最終独立paired/χ総合gate・残7・n8価格統計・長q/全RPEは未実施、公開照合後mandatory STOP。

[C8-1保存4wrapperの数値修正版](docs/research/track_d_c8_1_q64_20261108/NOFUSION_REPAIR_PLAN.md)：既存公式Aerのsimulation fusionだけを無効化し同じ4 P0を再検証。閾値/P0/入力不変、9人工/回帰PASS。旧診断完了を待ち、残7 pathへ自動進行しない。

[C8-1保存q64の数値診断](docs/research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_PLAN.md)：利用者承認による数値修正・再検証。旧SOURCE/失敗/STOPを保持し、保存sample00の4 P0回路を既存checker/同閾値で再postcheck、例外前のscalar/observableを保全する。sampling/回路生成/compile0、残7は未解放。

[Track D C8-1数値preflight不成立](docs/research/track_d_c8_1_q64_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。全8 event/512 occurrence/16384 eventsを固定し、sample00 raw1 PASS・旧P0 compile4完了をSOURCE8a6bbdacへ承認済み引継ぎ。両方式X/Y全4 postcheckがCAR-observable Frobenius2e−10でFAILしSTOP。残7/追加28 compile未開始、価格統計/モデル診断未評価。精度緩和/再実行/fit/N/全RPE更新0、旧SOURCE/STOP・非公開原本保持。

[C8-1 CPU予約修正・残task引継ぎ](docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)：利用者承認のSOURCE切替。旧6b1d613dの完了済みraw1・compile4を保全し、compile予約1 CPUとraw/postへの残CPU配分へ修正。sampling追加0、残7raw/最大28新P0、sample00全post gate維持、固定精度/モデル/32総上限は不変。

[Track D B/C全13-round条件付き評価](docs/research/track_d_bc_total_20261108/REPORT.md)：同Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、元H厳密ground・準備費除外。保存十分N14738と固定A/F/Sで全13-round条件付き資源会計を一回実施。主AはReferenceでCX11.17%・total depth54.65%・RZ1.12%減、幅9→17。約99.5%が長q proxy、全体不確かさ/actual未資格。新sampling/compile/fit/N0、SOURCEb09b7e7a、公開照合後mandatory STOP。

[Track D B6b＋B/C統合準備](docs/research/track_d_b6b_bc_integration_20261108/REPORT.md)：Referenceのideal coherenceを任意入力へ導出し、whole-V-ray条件とDF半path/最終真空条件を区別。SOURCE13114b2b、人工/回帰24件と人工P0 16 wrapper postcheck PASS。B4-2保存Nの条件付き共通利用・13-round価格coverage・薄い会計adapterを整理。actual N/長q期待価格は未資格、実H4の新計算/価格再解析/N算定/全RPE数値会計0。公開照合後mandatory STOP。
[Track D C7-4](docs/research/track_d_c7_4_model_stability_20261107/REPORT.md)：同H4/CDF8/LD2/S2/s1/r32/K0/h=.1の保存q2〜32・88 path再解析。SOURCEdd31e1e2、4探索fit family/5fold/共有bootstrap、最大残差の保存1 path再照合PASS、人工6長系列/短P0 2本・distinct回帰18 PASS。CX整数残差0、paired depthのbasis-runモデル改善は探索的、native版の精度改善は未確認。q≥64分子処理/N/全RPE費用0、未採択の次段階案、公開照合後mandatory STOP。

[Track D C7-3結果](docs/research/track_d_c7_3_q32_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/8 path。sampling/raw SOURCE35bfd、承認済み高速化compile/post SOURCE94e94。旧raw8再利用・新raw/sampling0、全32 P0/post・pairedχ8 PASS。CX差は全8本で−6960、RZ/size/各depthも有限全8本で減、qubits9→17。一次12量は探索的幅内、別population-line6量幅外。統計/全law/q≥64/N/全RPE未資格、旧STOP/raw非公開を維持、公開照合後mandatory STOP。

[C7-3承認済み高速化引継ぎ](docs/research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)：旧SOURCE35bfdのraw8 PASSを再利用し、旧controller STOP/集約未完を保全。未着手32 P0/各postだけを分離並列化し、exact局所cache・公式Aer double小演算子検証を選択。sampling0、旧新合算32compile上限、公開後mandatory STOP。

[Track D C7-3実行前PLAN](docs/research/track_d_c7_3_q32_20261107/PLAN.md)：同H4/CDF8/LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/n8。凍結C6aモデル・新seedをsampling前固定し、raw PASS後最大32 P0 wrapper。構造/DAG特徴とpaired価格の探索的検証、旧SOURCE/STOP保全、公開照合後mandatory STOP。

[Track D C7-2](docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md)／[q32未採択案](docs/research/track_d_c7_2_cx_structure_20261107/C7_3_EXECUTION_DRAFT.md)：実H4の新計算/価格再解析0、固定54人工P0・raw98項目/post54/observable等36項目・回帰12件PASS。局所非加法/単純lane式へ反例、208−224qの全path bridgeは未取得。旧SOURCE/STOP保全、q32未実行、公開照合後mandatory STOP。

[Track D C7-1](docs/research/track_d_c7_1_q16_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q16/cut8/16 path。SOURCE462f6bff、raw16・post64・pairedχ16 PASS。CX差は全観測pathで−3376、RZ/size/各depthも有限全16本で減、幅9→17。一次12量は探索的幅内、別population-line4量幅外。統計/全law/q≥32/N/全RPE未資格、raw非公開、mandatory STOP。

[Track D C7-1固定PLAN](docs/research/track_d_c7_1_q16_20261107/PLAN.md)：同H4/CDF8/LD2/S2/s1/r32/K0/h=.1・独立q16/cut8/16 path。凍結C6a予測・新seedをsampling前公開、raw PASS後だけ最大64 P0 wrapper。探索的paired価格/CX整数残差、raw非公開、完了公開後mandatory STOP。

[Track D C7-0静的構造解析](docs/research/track_d_c7_0_structural_preparation_20261107/REPORT.md)／[次段階条件案](docs/research/track_d_c7_0_structural_preparation_20261107/C7_1_EXECUTION_DRAFT.md)：C6a/C6b保存価格とSOURCEからCX・RZ/size・lane/depth、校正不確かさと長q移送条件を整理。CX一次式は全path未資格。新計算/計数/fit/テスト/compileなし、q16案は未採択・未実行、公開照合後mandatory STOP。

[Track D C6b](docs/research/track_d_c6b_q8_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1/q8/cut4、独立domainの32 path。元H厳密ground仮定・準備費除外。SOURCE44a58b9fへ承認済み限定引継ぎ、raw32・postcheck128・pairedχ32 PASS。一次frozen条件付き12量は探索的幅内、別population-line診断4量は幅外。統計未資格、全law/長q/N/全RPE未評価、raw非公開、mandatory STOP。

[C6b高速化・限定引継ぎ](docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)：旧SOURCE/raw16・compile/postcheck48を保全し、新SOURCEで未着手raw16/未compile80だけ。sampling0、旧新合算128上限、同じq8/cohort/精度/モデル、完了公開後mandatory STOP。

[Track D C6b独立q8](docs/research/track_d_c6b_q8_20261107/PLAN.md)：fresh32 whole-path/cut4、凍結C6aモデル・paired12量の探索的診断。各path raw PASS後最大128 P0 wrapper。旧SOURCE/STOP保全、raw非公開、長q/全RPE未資格、公開後mandatory STOP。

[Track D C6b-0静的準備](docs/research/track_d_c6b0_preparation_20261107/REPORT.md)：paired12量と絶対24量、旧20 floorの所在、CX構造の証明義務、追加校正/独立q8の最小案と未採択契約。新計算・再解析・実装0、公開照合後mandatory STOP。

[C6a結果](docs/research/track_d_c6a_paired_price_20261106/RESULT.md)：fresh q2/q4各16、raw32/P0 postcheck128/pairedχ PASS。有限pilot統計/共分散・C6b提案、rawローカル保全、全RPE未評価、mandatory STOP。

[Track D C6a](docs/research/track_d_c6a_paired_price_20261106/PLAN.md)：C5公開基点。canonical fresh q2/q4各16 path、fixed half-step cut、paired価格pilot。raw PASS後最大128 P0 wrapper、q8 holdout/全RPE対象外、公開後mandatory STOP。

[C5結果](docs/research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[Track D C5固定q2](docs/research/track_d_c5_q2_20261106/PLAN.md)：C3の演算を共通request入口へ最小抽出し、保存00→01/cut1のstrong境界融合とReferenceを比較。人工検証・raw PASS後だけ最大4 P0 wrapper、公開後mandatory STOP。C6 sampling/fit/holdoutは対象外。

[Track D C4価格モデル準備](docs/research/track_d_c4_price_model_preparation_20261106/REPORT.md)：保存q1/2/4/8の条件差、whole-pathとfenced kernel会計の境界、全law期待価格・共分散・長q holdoutの不足と最小実行案。静的調査のみ、科学実行・実装変更なし、公開後mandatory STOP。

[C3 q4/q8補完結果](docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。

[C3保存回路の中断補完PLAN](docs/research/track_d_c3_q4_q8_20261106/COMPLETION_PLAN.md)：run02の8 compile・7 postcheckを保持し、最後のq8 Reference Y postcheckと最終照合のみを別出力で実施。回路再生成・再compileなし。

[C3 run02再実行PLAN](docs/research/track_d_c3_q4_q8_20261106/RUN02_PLAN.md)：利用者承認によりcontract参照先を修正。同じq4/q8 fixture/cut/精度で再実行し、run01のSOURCE・失敗・STOPを保持する。

[Track D C3結果](docs/research/track_d_c3_q4_q8_20261106/RESULT.md)：contract参照先の実装誤りでq4/q8ともraw構築前STOP。科学的correctness・価格未評価、compile/retry/resume0。SOURCE・失敗・STOPを保全し、修正／再実行は新たな判断へ戻す。

[Track D C3固定q4/q8](docs/research/track_d_c3_q4_q8_20261106/PLAN.md)：ascending保存fixture・complete半path cutを固定し、合法なstep境界融合を含むstrong対照でraw correctness、PASS時だけ最大8 P0 wrapper。公開後mandatory STOP。
[Track D B5：ideal→actual保証接続の静的調査](docs/research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md)：finite-bit law／shot joint-law／Reference・compile誤差の不足、共通Nの条件、最小追加検証案。科学実行・実装変更なし、公開後mandatory STOP。

[Track D B4-2結果](docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md)：元H向けideal IID/exact nativeの13-round十分NはX/Y合計14738、energy幅.0012783173232380342 Ha・failure和.01。条件付き数値pilot、actual・費用未資格、mandatory STOP。

[Track D B4-1：元H統計adapter](docs/research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md)：own-DF guardを保持した共通化、元H用N／energy／decoder接続。人工52件＋既存21件PASS、実H4評価なし、mandatory STOP。

[Track D B3：元H向けRPE統計の接続準備](docs/research/track_d_b3_rpe_statistics_preparation_20261106/REPORT.md)：B1保存誤差から統計・decoder・全RPE会計へ接続する既存部品と不足、次の最小実装・実行許可範囲。静的調査のみ、N算定・科学実装なし、公開後mandatory STOP。

[C2固定q2結果](docs/research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](docs/research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](docs/research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](docs/research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Track D Stage B・Cの技術準備](docs/research/track_d_bc_static_preparation_20261106/REPORT.md)：元H向け信号・統計とwhole-round Reference価格の再利用条件・不足・最小検証案。静的調査のみ、科学計算・実装0、公開後mandatory STOP。

[Stage A結果](docs/research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

Track D Stage Aの[保存32 trajectory paired検証](docs/research/track_d_h4_paired32_reference_20261106/PLAN.md)は同CDF8 endpoint・event16後cut・2方式×X/Yに限定する。全32 precompile PASS時のみ最大128 wrapper compile、公開後mandatory STOP。B/C段階は対象外。

Track Dの実H4移送検証は[固定word02/cutと実行範囲](docs/research/track_d_h4_reference_validation_20261106/PLAN.md)／[結果](docs/research/track_d_h4_reference_validation_20261106/RESULT.md)を参照する。旧SOURCEを別treeから再利用し、通常のtestや旧研究runnerから実行しない。

Track Dの限定人工検証は[固定scopeとSOURCE](docs/research/track_d_minimal_reference_validation_20261106/PLAN.md)を参照する。分子・全RPE検証とは別で、明示runnerだけが実行入口である。

**研究結果への入口（2026-11-02）：[統一索引](docs/research/RESEARCH_RESULTS_INDEX.md)。**
複数ブランチの総括・後続結果・SOURCE・raw・検証状態を固定commitで辿れる。
以下の研究記述は、このcheckoutに保存された時点の履歴として読む。

最終更新：2026-09-20

このファイルは、人またはGPTがリポジトリ全体を読むときの入口である。研究内容の正本、
実装、検証コード、結果データ、発表資料を区別し、古い研究経路を現在の結論として読まない
ための案内をまとめる。

## 最初に読む順序

1. [`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)
   研究目的、採用中の前提、現在の実装・検証段階、主要結果、未解決事項の最新要約。
2. [`docs/research/研究目的・研究課題.md`](docs/research/研究目的・研究課題.md) と
   [`docs/research/研究方法・解析手順.md`](docs/research/研究方法・解析手順.md)
   研究設計と解析手順の正本。
3. [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md)
   各結果を再利用できるか、欠落データや再計算が必要かを示す状態表。
4. [`artifacts/validation_manifest.json`](artifacts/validation_manifest.json)
   検証結果、生成コード、テスト、成果物を結ぶ機械可読な証拠台帳。
5. 数値を使う場合だけ、対応する `docs/*_validation.md` と `artifacts/` のJSONを確認する。

`docs/research/研究ノート/` は意思決定の時系列記録であり、現在の仕様ではない。過去の暫定値が
後日のノートや研究概要で変更されている場合は、最新の研究概要と規範文書を優先する。

## 旧snapshotに記録された研究段階（2026-09-20）

現在の中心課題は、DF Hamiltonianを決定論部分とランダム部分へ分けたpartial-$S_2$について、
有限RTE、RPEの信号半径・測定回数、1 shot当たりのコンパイル後回路コストを接続することである。
PF係数、有限RTE、ランダム回路コスト、短いRPE段の接続は限定条件で検証を進めている。
既存設定`CA/10`を暫定目標にすると$q_{\max}=32768$が必要で、従来の固定
$\delta=0.1,r=4,K=2$は長roundへ単純外挿できないことを確認した。その後、H4の
実行済み$\delta$窓でround別$(r_m,K_m)$を再探索し、$\delta=0.01,0.0125,0.02$に
行列検査を通るscheduleを構成した。さらに、短時間幅0.02--0.000390625で局所回路指標が
変わらないことと、イベント列長8、16、32への移送を検査し、$r\leq16$では1--3イベント、
$r=32$では4イベント補正を用いる中央RTEブロックcost proxyを接続した。この限定proxyでは
$\delta=0.02$が全6指標で最小となった。現在は$\delta=0.02$と比較対照0.01について、
制御付きpartial-$S_2$反復とHadamard 1 shot costへ接続する段階である。

最終的な全RPE段の総コスト最適化と、決定論PFに対する最終的な優位性評価はまだ行っていない。
最新の到達点と次の検証は、必ず
[`研究概要・現状.md`](docs/research/研究概要・現状.md)で確認する。

## ディレクトリの役割

| 場所 | 役割 | 読み方 |
|---|---|---|
| `src/trotterlib/` | ライブラリ本体 | 現行実装と検証ロジック。分類は[`src/trotterlib/README.md`](src/trotterlib/README.md) |
| `scripts/` | 実行入口 | 検証runner、batch、診断、旧経路。分類は[`scripts/README.md`](scripts/README.md) |
| `tests/` | 自動テスト | API・数値恒等式・成果物schemaの回帰検査。科学的結論そのものではない |
| `docs/research/` | 研究方針の正本 | 概要、目的、方法、評価計画、研究ノート |
| `docs/` | 実装・検証の説明 | 各検証の条件、結果、限界、実装規約。索引は[`docs/README.md`](docs/README.md) |
| `artifacts/` | 計算結果と入力snapshot | JSON等の証拠、キャッシュ、途中状態。利用規則は[`artifacts/README.md`](artifacts/README.md) |
| `BentoSlide構成案*.md` | 発表資料生成用の指示 | 研究の正本ではない。位置づけは[`docs/presentations/README.md`](docs/presentations/README.md) |
| ルートのPDF | 参考論文または発表資料 | 種別は[`docs/references/README.md`](docs/references/README.md)で確認 |

## 実装と検証の関係

```text
研究方針・条件
  docs/research/
        ↓
ライブラリ実装
  src/trotterlib/
        ↓
実行入口                 回帰テスト
  scripts/run_*.py   ↔    tests/test_*.py
        ↓
結果・入力snapshot
  artifacts/
        ↓
結果の説明と状態
  docs/*_validation.md
  VALIDATION_STATUS.md
  artifacts/validation_manifest.json
```

`scripts/run_*.py` はメインロジックの置き場所ではなく、引数を受け取り
`src/trotterlib/` の処理を呼び出して成果物を保存する入口である。検証名と同名の
ライブラリ、runner、test、文書、artifactを一組として読む。

## 共有サーバー向けexecution infrastructure

長時間検証を独立taskへ分割して安全に実行・再開する経路は、
`src/trotterlib/parallel_validation_executor.py`、`scripts/run_parallel_validation_batch.py`、
`tests/test_parallel_validation_executor.py`、
[`docs/server_parallel_validation_execution.md`](docs/server_parallel_validation_execution.md)を一組として読む。
これは実装・運用基盤であり、新しい科学的検証結果や最終総cost評価ではない。

## ファイルの状態区分

### 現行の正本

- `docs/research/研究概要・現状.md`
- `docs/research/研究目的・研究課題.md`
- `docs/research/研究方法・解析手順.md`
- `docs/research/数値実験・評価計画.md`
- `VALIDATION_STATUS.md`
- `artifacts/validation_manifest.json`

### 現行実装・検証

- `src/trotterlib/` のDF、RTE、RPE、compiled-cost関連モジュール
- 対応する `scripts/run_*.py`、`tests/test_*.py`、`docs/*_validation.md`
- manifestに登録され、statusと限界が明示されたartifact

### 歴史的・補助的資料

- `abe_trotter_project.ipynb`：旧来の高次PF解析ノートブック
- `Partial Randomized Study Protocol.md`：初期計画と意思決定の履歴
- `codex-inst.md`：過去の実装依頼メモ
- `README_partial_randomized_pf.md`：旧screeningを含む実装経路の説明
- `docs/main_audit_20260801.md`：特定時点の監査記録

これらは削除していないが、現在の研究方針や最新結果を確定する根拠には使わない。

## GPTが回答・資料作成するときの確認事項

- 「理論上の関係」「ローカル検証済み」「実装のみ」「未検証」「最終結論」を分ける。
- 数値には、分子、距離、basis、DF rank、$L_D$、時間幅、Taylor cutoff、検証範囲を添える。
- `C_use`を厳密上界と呼ばない。実行したdelta窓上の経験的包絡である。
- H4/H6の係数をH12へ外挿しない。
- PFの決定論的biasとQPE/RPEの統計誤差を混同しない。
- dirty worktreeの結果やローカルテストを、公開済み・CI固定済みの証拠と表現しない。
- スライド構成案や研究ノートの単独記述より、研究概要、検証文書、manifestを優先する。

## 保守規則

研究上の決定や検証結果が変わった場合は、次を同じ変更で更新する。

1. `docs/research/研究概要・現状.md`
2. 対応する研究方針文書または検証文書
3. 当日の研究ノート
4. 証拠構成が変わる場合は `artifacts/validation_manifest.json`

新しい検証を追加するときは、可能な限り同じ語幹で
`src/trotterlib/`、`scripts/`、`tests/`、`docs/`、`artifacts/`を対応させる。

## 既存研究成果のmain収録（2026-11-03）

[収録済み成果と原SOURCE](docs/research/main_collection_20261103.md)から、main内のG6報告・保存結果・STOP・再現用SOURCEを辿れる。
これは過去の証拠とSOURCEの収録であり、新しい科学結果・実行許可・共通ライブラリ採択ではない。

第2回は[公開snapshot間で衝突のない追加ファイルと原データ所在](docs/research/main_collection_20261103.md#第2回競合のない追加ファイルの収録)を収録。既存pathの別版と枝全体のmergeは保留し、元の科学SOURCE・結果・STOPを区別して保存する。

2026-11-03のmain集約で追加したルートの旧計画・`REPORT.md`・`references/` 等は、
当時の研究別記録として[出典対応](docs/research/main_collection_20261103_additional_files.tsv)から辿る。
現行の統合仕様・実行許可は、この入口で示す優先順位と各研究の既存契約に従う。

## 索引・SOURCEの版差（2026-11-03）

[同名37 pathの整理結果](docs/research/main_collection_versions_20261103.md)から、統合した7索引、原内容で配置したSOURCE、全88保存版を辿る。
旧snapshotの研究記述を、全研究系列の現在の状態へ読み替えない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルに残っていた文書・SOURCEの収録（2026-11-03）

[原稿・受領レビュー・研究ノート・SOURCEと保存別版の対応](docs/research/main_local_results_collection_20261103.md)から辿る。元の科学SOURCE・status・STOPは維持する。

2026-11-08：[Track D O5 改善候補の静的調査](docs/research/track_d_o5_algorithm_design_20261108/REPORT.md)。strong/Reference/既知SE/O2の構造を対応付け、A:compile前方式選択、B:近傍cut、C1:target保持head-tail接口を比較。結論案B（追加効果/新規性未確定）、新実装/価格再解析/科学計算0。q64別担当とO2未採択H4契約へ介入せず、公開後STOP。

## Track D O5：近傍L/R cutの人工検証（2026-11-08）

[実装・結果](docs/research/track_d_cut_artificial_20261108/REPORT.md)。C3 cut入口とC7-4 descriptorを再利用したcompile前selector。固定2-mode/q6/16-event人工2例、8 unit tests・110比較・10 P0 unique compile PASS。偏り例は中央cut3→2でfull depth934→646、count/size/幅は同じ。均衡例は中央維持・追加効果0。実H4/価格再解析/fit/N/13round0、q64非介入、次は未採択の最小H4移送案のみ。

## O5実H4 q4単一path：入力preflight STOP（2026-11-08）

[結果・不足原本](docs/research/track_d_cut_h4_q4_20261108/REPORT.md)。指定C6a q4_sample00の公開binding/SOURCE互換を確認したが、非公開event/QPY原本がSERVER1確認範囲で取得できず、cut選択前にSTOP。新科学call0、改善/価格差UNKNOWN、別path/再samplingなし。

## O5実H4 q4移送完了：中央cut選択（2026-11-08）

[最終報告](docs/research/track_d_cut_h4_q4_completion_20261108/REPORT.md)。提供されたC6a q4_sample00 archive全20 memberのhash一致後、固定ruleを一回適用しcut2を選択。旧raw1/compiled4 PASS・全7価格を適格再利用、新build/compile/post0、選択−中央は全7差0。前回入力STOPは履歴保全、他cut/path・q64は対象外、公開後mandatory STOP。
