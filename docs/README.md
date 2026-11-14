# 文書索引

[Full-H κ2 P-2 D7較正の実行前STOP](research/d_full_h_p2_calibration_20261115/REPORT.md)／[PREFLIGHT](research/d_full_h_p2_calibration_20261115/PREFLIGHT.json)／[C引渡し](research/d_full_h_p2_calibration_20261115/HANDOFF_C.md)：予定18 compile・実行0。rank3とSOURCE/inputは確認済み。paired再samplingと較正実装証拠欠落のため固定SOURCE変更・再開指示待ち。

[Full-H κ2 P-2 caller・人工回帰・次の有限実行案](research/d_full_h_p2_caller_preparation_20261114/REPORT.md)。22/24/26 compileの未採択案、rank検査、explicit cell/paired seed、C受渡しを技術対応。新H4計算・保存科学値再解析0。m5 HOLDOUTとA資源順序の整合は採択前に要解決。 mandatory STOP。

[Full-H κ2 P-1 run01・run02 STOP](research/d_full_h_kappa2_p1_results_20261114/REPORT.md)／[実行記録](research/d_full_h_kappa2_p1_results_20261114/EXECUTION.txt)：run01はtracked file欠落で科学計算前に停止。run02は作用検査PASS・2 compile済みだが、`scalar_once`判定の偽陰性で停止（[研究ノート](research/研究ノート/2026-11-14_d_full_h_kappa2_p1_run02_stop.md)）。run02の[再検証](research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/REPORT.md)：修正判定でPASS、価格record・sidecarをC guardで受理（[研究ノート](research/研究ノート/2026-11-14_d_full_h_kappa2_p1_run02_revalidation.md)）。

[Full-H κ2 S-1結果](research/d_full_h_kappa2_s1_results_20261114/REPORT.md)／[実行記録](research/d_full_h_kappa2_s1_results_20261114/EXECUTION.txt)：13round raw/corrected/logB、R448との比較条件。

[D κ2 S-1/P-1と価格接続REPORT](research/d_kappa_s1_price_builders_20261114/REPORT.md)／[次の実行](research/d_kappa_s1_price_builders_20261114/NEXT_EXECUTION.md)／[HANDOFF](research/d_kappa_s1_price_builders_20261114/HANDOFF.md)：固定caller `04bead11`、30人工・metadata PASS、科学実行・compileなし、mandatory STOP。

[Truncated PF T2結果](research/truncated_pf_t2_results_20261114/REPORT.md)／[CELLS.csv](research/truncated_pf_t2_results_20261114/CELLS.csv)／[T3候補](research/truncated_pf_t2_results_20261114/T3_CANDIDATES.json)／[run01](research/truncated_pf_t2_results_20261114/run01/RESULT.json)：切断・PF・合計シフト、分枝・重なり、Full PF比較。T3/T2bは未承認・STOP。

[Truncated PF T1結果](research/truncated_pf_t1_results_20261114/REPORT.md)／[SHIFTS.csv](research/truncated_pf_t1_results_20261114/SHIFTS.csv)／[run01](research/truncated_pf_t1_results_20261114/run01/RESULT.json)：p0..9の切断シフト・重なり・λ_R・c_tail。T2は未承認・STOP。

[Truncated PF計画REPORT](research/truncated_pf_optimization_plan_20261114/REPORT.md)／[段階計画](research/truncated_pf_optimization_plan_20261114/SCIENCE_PLAN.md)／[schedule metadata](research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json)／[toy検証](research/truncated_pf_optimization_plan_20261114/TOY_CHECKS.json)：未承認の有限計画、新科学計算なし・STOP。

[最新Full PF価格接続・S1比較guard](research/full_pf_latest_price_guard_20261114/REPORT.md)（2026-11-14）。

[Full PF3候補のcompiled価格証拠接続](research/full_pf_candidate_price_evidence_20261114/REPORT.md)（2026-11-14）：A3ae055faのFull PF S2/.2・Y4/.4・M5/.8を39round/78軸でD e7 receiptへ照合。direct5round/10軸、M5 m4欠測、他33roundはcatalog外。元native/scalar/wrapper/P0のD証拠を再利用し、consumer lowering採用は未実施。追加4セル/最大8wrapperの未承認q検証案、early幅/長q未資格・必要精度UNKNOWN。価格再集計/予測/fit/compile/N/G0、production変更0、公開後mandatory STOP。

[B0I・Full-Hの全13round Raw signal取得](research/truncated_fullh_signal_results_20261114/REPORT.md)（2026-11-14）：固定H4 DF10のB0I p7/S2/Δ.1とFull-H finite-K2/q=2^m/R448を各13round、一回評価。SOURCE5a172629d、26件すべて元数値gate PASS、Raw/corrected/reference/normalizationとB packetを保存。人工caller3件PASS、数値module不変、context復元1、旧signal受入/retry0。数値誤差・law距離・state誤差は未資格、MC/N/価格/G0。旧c546/6306計画・STOPを保全し公開後mandatory STOP。

[B0I・Full-Hの26round Raw signal有限scope](research/truncated_fullh_signal_scope_20261114/REPORT.md)（2026-11-14）：c546原本からB0I p7/S2/.1とFull-H K2/q=2^m/R448の26roundを同identity/SOURCE/gateで分離。Full PF・m6受入依存・旧signal再資格0。新reference/抽出0、共有saved context/cacheを再利用。metadata26件照合PASS、実input/数値計算/再解析/旧test再実施0。数値実行は明示許可待ちSTOP。

[平均compiled価格の証拠coverage・有限検証案](research/rpe_price_evidence_validation_20261113/REPORT.md)（2026-11-13）：PR-LD1/LD2の52軸＋旧Full PF6 recipeの156軸を静的coverage化。PF direct receipt16軸、LD1 E[S]26軸/P2有限cohort4軸の所在と未受入bridgeを区別。D corrected scalarの人工1＋旧13＝14 PASS、production変更0。条件付きPR新paired6セル等を提案し、BのN/精度とA選択待ち。保存S/C再集計・fit/compile/G0、公開後STOP。

[全方式の平均compiled価格モデル技術準備](research/rpe_mean_price_model_preparation_20261113/REPORT.md)（2026-11-13）：外部係数の加法式・境界covers・既存S＋単一C−S・4方式の受動B接続を追加。人工新11＋既存27＝38 PASS。全モデル未資格、実H4/H6の新S/C・保存再解析・sampling/compile/fit/G0。C1係数/26 E[S]/D requestと既存STOPを保全、公開照合後STOP。

[D最新版scalar bindingの技術接続](research/d_scalar_binding_connection_20261113/REPORT.md)（2026-11-13）：C価格bindingをD748へ接続し、expected_scalarの生成/再構成・schedule anchor・resolved pathのbinary64完全一致を追加。新5＋既存20＝25人工/回帰PASS、省略時の旧request互換を保持。実H4/preparation/sampling/compile/保存科学再解析/E[C]・G0、D T2のscalar原因はUNKNOWNのまま。ベースライン科学計画の未承認STOPを保全、公開照合後STOP。

[ベースライン3系列・全RPE signal有限実行計画](research/baseline_finite_signal_plan_20261113/REPORT.md)（2026-11-13）：Full PF S2/.1/Full10、B0I p7/S2/.1、Full-H K2/q=2^m/R448の3 canonical schedule・39roundを固定。提案は保存Full PF m6の受入再資格1件→PASS時だけ新38 mean。無条件再利用0、X/Yは倍計数しない。既存13 reference pair/保存scalarを利用し、新reference/抽出0。metadataのみ、実input読込/再資格/新signal/旧test再実施/N・価格・G0、明示科学許可待ちSTOP。

[D価格adapterとRPE pipelineの統合](research/d_price_pipeline_integration_20261113/REPORT.md)（2026-11-13）：公開D candidate_price_adapterをC price_requestsへ接続。A固定LD1/LD2の13round metadataとD path→pending/外部mean→B会計を照合し、新8＋既存36＝44 PASS。path/S/finite cohortを適格E[C]へ昇格せず、軸・SOURCE/law/preparation/compiler/loweringを保持。実H4/preparation/sampling/compile/保存再解析/実N・G0、D物理資格と平均/N証拠は後続。

[RPE総コストpipelineの人工統合](research/rpe_optimization_integration_20261113/REPORT.md)（2026-11-13）：A PR・C baselineの全round signalをB shot配分/cost/selectへ接続。D向け未評価requestとphysical/lowering bindingを整備。人工新10＋B metadata2＋C/M5回帰24＝36 PASS、4方式の人工全13round・shot最小/G最小・点/区間・未資格/欠測を区別。実H4/saved再解析/実N・価格・G0、科学N資格producerとD価格は未取得。

[ベースラインsignal pipelineの技術準備](research/baseline_signal_pipeline_preparation_20261113/REPORT.md)（2026-11-13）：ベースライン全round signal経路を既存B/D/数値部品から接続。H4 DF10のFull PF・B0/B0I・Full-H pure RTE、S2/Y4/M5・prefix・明示q/R・有限Kを扱う。人工新12＋既存回帰12＝24 PASS、数値失敗値/欠測/未資格を保持。実H4・保存再解析・残screen・sampling/compile/N/RMSE/G0。

[全RPE比較ベースラインの最適化設計](research/rpe_baseline_optimization_design_20261113/REPORT.md)（2026-11-13）：Full PF(S2/Y4/M5)、B0/B0I切断PF、one-body込みFull-H pure RTEを全RPEのΣN E[C]で探索する計画。旧11案はseed、局所screen/構造proxyと根拠付きrisk/cost除外を区別。既存APIを再利用し新framework0、人工回帰20 PASS。実H4計算/保存再解析/候補最適化/compile/N/G0、正式選定なし、公開照合後STOP。

[残差構造特徴量の期待値評価方法](research/structural_feature_expectation_design_20261113/REPORT.md)（2026-11-13）：D P2のfeature定義を照合し、tail Z/ZZ/order/product数とbasis group/shareの解析式・小kernelを整備。新9＋既存19＝28人工/技術test PASS。native/compiled相殺や非線形joint量とは区別し、Dの直接残差法を置換しない。実H4確率解析/F0/K0/K1/E[S]/sampling/compile/fit/N/G0、旧26値とSTOP保全。

[finite-K sampler law整合性](research/finite_k_sampler_law_alignment_20261113/REPORT.md)（2026-11-13）：ideal finite-K/source lawとPCG64/CDFの接続調査。生成構造・rte.py blob対応を確認し、既存53-bit helperを再利用するTV分解/条件付き上界を追加。人工12＋既存toy5 PASS。実H4のlaw距離・PRNG joint-law・価格systematicはUNKNOWN/null、既存26 E[S]不変、新科学計算0。公開照合後mandatory STOP。

[H4 PR全13roundの平均構造RZ結果](research/h4_structural_schedule_results_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2・同preparationで残11lawのK0全PASS、新22 E[S]を取得。m0/m6四値を再実行せずcopyし全13round/26軸が同canonicalへ接続。SOURCE de2427d4a、99 wire-law検査、最大mass defect1.714e−12（tol1e−10）。元Sのideal有限law平均でE[C]/Gはnull。新sampling/full-path S/build/compile/fit/N/G0、計算再起動0、旧STOP保全。公開照合後mandatory STOP。

[H4構造価格・全13round経路の技術準備](research/h4_structural_schedule_preparation_20261113/REPORT.md)（2026-11-13）：固定B3216/同canonicalの13round metadataを接続し、m0/m6四値は83d219cc6からcopy専用。残11lawのK0→PASS時K1 batchを既存core/MeanKernelで準備、new8＋回帰18＝26技術test PASS。R255等を保持し、実T/terminal入力・同γ16外向き二乗法で巨大qR展開を回避。実新K0/E[S]/sampling/compile/fit/N/G0。最大22新Sの一括計画を公開、残差はD担当、公開照合後STOP。

[H4 kernel run02：K0 PASS・K1平均構造RZ四値](research/h4_structural_kernel_20261113/run02/REPORT.md)（2026-11-13）：SOURCE19b95d8b3で同H4 DF10/S2/p1/δ.1/K2・1prep/2lawを一回実行。K0全9wire PASS、kernel/状態辞書を保存後、m0/m6×XYの実E[S]四値を取得（各mでX=Y、1743.6682557187337/1764659.275564426）。数値区間・元Accounting parityを保存。sampling/full-path S/build/compile/fit/N/G0、E[C]はnull・C1未資格のまま。旧run01/STOP保全、公開照合後mandatory STOP。

[H4 K0入力STOPと技術修正版](research/h4_structural_kernel_20261113/REPORT.md)（2026-11-13）：K0→PASS時K1 runnerと数値契約を実装、新8（人工7/固定metadata1）＋旧10技術test PASS。SOURCE0f7d65d5fはB configuration locator誤りで物理K0前にUNKNOWN STOP。実descriptor/kernel/parity/平均S四値は未実施・null。参照chainを修正版19b95d8b3でmetadata照合し、科学再実行なし。指示§4 STOP保全、run02案のみ公開。新sampling/full-path S/compile/fit/N/G0。

[H4 PRの物理閉状態診断](research/h4_pr_physical_closure_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2の1prep・R2/R128を診断。324非identity/9basis、reset/phase/head境界からS observableの有限wire-marginal状態の十分条件を確認（tail enclosure80〜120）。物理kernel/parity/実平均S/Cは未取得、MODEL false。最終10技術test PASS。初回seed設定STOPを保全し同scope修正、成功SOURCE eeccdf18e・縮約解析2b3b0ee7f。新sampling/full-path S/build/compile/fit/N/G0、旧S0a等を保全、公開照合後STOP。

[固定H4 S0の平均compiled価格検証計画](research/h4_s0_mean_price_plan_20261113/REPORT.md)（2026-11-13）：SOURCE bdd2b9356。固定A S0のexactκ/M12・H4 m0(q1/R2)/m6(q64/R128)を公開B V2へmetadata接続。新7技術test PASS、13round価格null。両候補の一致direct/modelは公開索引に未確認。閉状態診断1prep/2law・compile0を第一案、別案は有限4path/最大8wrapper。新S/価格再解析/sampling/compile/fit/N/G0、C1未資格・S0a履歴を保全し、公開照合後STOP。

[B V2価格接続と長q平均価格の次段階設計](research/bv2_mean_price_validation_design_20261113/REPORT.md)（2026-11-13）：SOURCE58382c54c。固定B6a9 V2 MeanPriceLookupを再利用するC受動入口を追加、新12人工/metadata結合test PASS。値ありrecordに平均compiled quantity明示を要求し、S0a/旧ID付替え/資格昇格を拒否。旧118候補/236軸と4方式の資産/欠測を静的整理。実平均S/C・物理閉包は未取得、選択B domainのclosure診断等4候補を未承認計画化。新S/C再解析/sampling/compile/fit/N/G0、旧結果/STOP保全、公開照合後STOP。

[S0a run02：実保存8件の構造S照合](research/s0a_saved_structure_results_20261113/run02/REPORT.md)（2026-11-13）：修正版SOURCE9516351e6で保存normal index0の4path×XY8 Sを一回評価し、元C1保存Sと8/8整数完全一致。各X/Y SはH4 q8/R8=21664、q16/R8=46559、H6 q1/R32=158669、q2/R32=321151。input/preparation/law/phase/allocation/digest検査を通過、元roleとrun01失敗STOPを保全。平均S/C・物理閉包・モデル資格は未取得、qualified=false。新sampling/compile/fit/N/G0、公開照合後STOP。

[S0a run01の設定エラーと技術修正](research/s0a_saved_structure_results_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65でrun01を1回起動、context exec scopeのast渡し漏れでS開始前に失敗STOP。新S0/8、保存S照合0、全4path未評価。修正版SOURCE9516351e6と人工14件PASSを公開、run02案はmetadata準備だけで未実行。旧STOP/role/モデル資格を保全、修正版再試行はユーザー判断、公開照合後mandatory STOP。

[S0a保存構造価格の実行準備](research/s0a_saved_structure_preparation_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65。固定C1の保存4path/XY8計数に向けreader・law/identity検査・S専用runnerを準備。新13＋旧9＝22技術test PASS。raw/inputは所在のみ確認、実S・保存S比較は未実行。元係数/資格/平均provider不変、内容hashと実contextは実行時検査。初回人工importのlowering副作用と修正をREPORTへ記録。明示科学許可待ち、公開照合後STOP。

[C1状態閉包とStreaming計数の技術検証](research/c1_physical_state_stream_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE c1135fdb9。C1のpending basis/Z・tail phase・terminalをSOURCEで調査し、元計数wordへforward-only tailを渡す最小consumerを追加。新9＋旧3＋sampler toy1＝13技術test PASS。単純縮約の人工反例は確認、実H4/H6の閉包・event生成なしの平均S/compiled価格は未確認。consumerは全event処理時間が残り、producer全接続は未実装。S0a保存4path/XY8件は未実行案、実科学計算・再解析・compile・fit・N/G0、公開照合後mandatory STOP。

[長q・大Rの平均compiled価格の技術準備](research/mean_compiled_price_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE09489eedb。C1 Accounting/旧局所期待formを再利用し、closed-state期待報酬の二乗合成と受動c_providerを実装。新15＋旧pure3＝18技術test PASS、人工全列挙と一致。固定状態ならqR展開を避けられるが、実C1の累積Z/phase閉包・平均S・compiler残差は未資格で実価格missing。C−S残差のみを扱い二重計上を拒否。κ scheduleはB外部contextのcopy、M/N最適化なし。旧40/16案は非実行、将来S0保存再解析等は別許可。実科学計算/compile/価格fit予測/N/G0、公開照合後STOP。

[A P1対応・長q/大R価格検証の準備](research/r_price_longq_largeR_preparation_20261111/REPORT.md)（2026-11-11）：公開A71ba49bbの12候補・同policy条件付きNを無変更copy。C1価格との重複はH6 m4/R16,R32の校正2点。第一案はH4 m6/H6 m8のbaseline＋R128、4セル・最大40 wrapper、原因分離R-onlyと短時間各16は別の任意案。metadata SOURCE f0daeb96b、8技術test PASS。構造Sのstream化/局所境界法を比較し、m12やR256を自動compile/外挿しない。新科学計算・価格予測・再解析・N/G0、科学実行未承認、公開照合後STOP。

[R価格Phase 1実行結果](research/r_price_phase1_results_20261111/REPORT.md)（2026-11-11）：Linear H4/H6・1Å・STO-3G・保存DF10/19、H4 S2/δ.1/p1/K2とH6 m5_best/δ1.6/p7/K2。SOURCE1a87e5331で通常64＋別rare16 compile、保存C/S59 path再利用。係数を7eeabf0aで検証取得前に凍結し再fitなし。独立n8の相対RMSEはH4 q16/R8で1.039841%、H6 q2/R32で0.229959%。有限cohort観測、採用閾値未設定・model未採用、全law/長q-R/総CI/全Gは未資格。新signal/N/G0、公開照合後mandatory STOP。以下の準備記録・過去STOPは履歴として保持。

[R価格Phase 1の実行準備](research/r_price_phase1_preparation_20261111/REPORT.md)（2026-11-11）：保存C/S接続・paired3係数fit・Git凍結後validation・別unit rare診断の専用runnerを準備。公開A/B metadata整合、新14＋旧R回帰9＝23人工/mock PASS。固定SOURCE/PLAN、最大64＋16 wrapperの科学scopeは未実行。実分子sampling/S再計数/compile/fit/価格予測/再解析/N/G0、公開照合後STOP。

[R可変compiled価格の調査・設計](research/r_variable_compiled_price_design_20261111/REPORT.md)（2026-11-11）：H4 S2/δ.1/p1/K2・H6 m5_best/δ1.6/p7/K2の固定Rモデルを保全し、構造価格＋少数compiled残差補正を提案。R binding/保存lookup adapter、人工/mock9件PASS。新R Cはnull、実科学計算・価格取得/予測/fit/G0。最小新q×R/希少event計画は未承認・未実行、公開照合後STOP。

[H6 provider統合・H4/H6全13round N/C coverage](research/h6_provider_integration_nc_coverage_20261111/REPORT.md)（2026-11-11）：固定B cb2 APIへ保存lookup factoryを追加。C provider/テストをbyte一致再利用、24技術test PASS。H4公開89c91c69の整数検証済みNとH6公開f4c0bdaのNを静的転記し52行条件付きPASS。直接CはH4 5候補/H6 8候補、PR全law/長q/共有CI/全RPE採用は未資格。新科学計算・実N×C/G0、B実行領域へ非介入、公開照合後STOP。

[H4価格providerのB統合完了](research/h4_price_pipeline_integration_20261110/REPORT.md)：B API接続、保存direct/Mq・有限cohort・null/外挿資格の保持。34人工・保存fixture回帰、新実科学計算/G0。

[H6 DF19全13roundのN coverage](research/h6_df19_rpe_m8_12_results_20261110/REPORT.md)：残5時点10候補は全ideal条件付きPASS。[正本](research/h6_df19_rpe_m8_12_results_20261110/run01/result.json)／[coverage・handoff](research/h6_df19_rpe_m8_12_results_20261110/run01/COVERAGE.json)／[全新signal・N CSV](research/h6_df19_rpe_m8_12_results_20261110/ROUND_SUMMARY.csv)。actual/C/G未資格、公開照合後STOP。

[H6 DF19残5round一括実行準備](research/h6_df19_rpe_m8_12_preparation_20261110/REPORT.md)：[固定PLAN](research/h6_df19_rpe_m8_12_preparation_20261110/PLAN.json)／[SOURCE](research/h6_df19_rpe_m8_12_preparation_20261110/SOURCE_INDEX.json)／[人工7件](research/h6_df19_rpe_m8_12_preparation_20261110/TESTS.txt)。実H6未実行、明示許可待ち。

[PF/PRエネルギー推定・総費用pipeline](research/pf_pr_rpe_cost_integration_20261110/REPORT.md)：[一括科学計画](research/pf_pr_rpe_cost_integration_20261110/EXECUTION_PLAN.json)／[静的H4/H6計画](research/pf_pr_rpe_cost_integration_20261110/static_plan/)／[試験](research/pf_pr_rpe_cost_integration_20261110/TESTS.txt)。共通interfaceと会計を接続、実科学/G0、公開照合後STOP。

[H6 DF19 RPE m7結果](research/h6_df19_rpe_m7_results_20261110/REPORT.md)：固定2候補のreference/signal/phase/radius/十分N。[正本](research/h6_df19_rpe_m7_results_20261110/run01/result.json)／[8round coverage・価格handoff](research/h6_df19_rpe_m7_results_20261110/COVERAGE.json)。各軸PF705/PR846、残りm8..12、actual/全G未資格、公開照合後STOP。

[H6 DF19 RPE m7実行準備](research/h6_df19_rpe_m7_preparation_20261110/REPORT.md)：固定2候補・旧kernel再利用・人工3件PASS。[PLAN](research/h6_df19_rpe_m7_preparation_20261110/PLAN.json)／[人工結果](research/h6_df19_rpe_m7_preparation_20261110/synthetic_tests.txt)。実m7のreference/signal/Nは別途明示承認待ち。

[H6 DF19 m6 run02結果](research/h6_df19_rpe_m6_results_20261110/run02/REPORT.md)：明示実行指示後の同条件再実行。[正本](research/h6_df19_rpe_m6_results_20261110/run02/result.json)／[照合](research/h6_df19_rpe_m6_results_20261110/run02/CHECKS.txt)。各軸PF705/PR772、run01は保全、N・標本数を合算しない。公開照合後STOP。

[m6既存成果再利用・価格設計への接続](research/h6_df19_rpe_m6_results_20261110/REUSE_AND_PRICE_CONNECTION.md)：再送指示と完了済み7903e3633の条件・候補が一致。指定価格設計4c965ff6のmetadataと接続を確認し、新科学計算・test・価格処理は0。既存各軸NはPF705/PR772、公開照合後STOP。

[H6 DF19 RPE m6結果](research/h6_df19_rpe_m6_results_20261110/REPORT.md)：reference/finite mean/phase/radius/十分Nの限定2候補。[正本](research/h6_df19_rpe_m6_results_20261110/run01/result.json)／[7round対応・価格担当handoff](research/h6_df19_rpe_m6_results_20261110/COVERAGE.json)。各軸PF705/PR772、actual/全G未資格、公開照合後STOP。

[H6 DF19 RPE m6実行準備](research/h6_df19_rpe_m6_preparation_20261110/REPORT.md)：固定2候補・call範囲・D2 backend再利用・人工差分3件。[PLAN](research/h6_df19_rpe_m6_preparation_20261110/PLAN.json)／[人工結果](research/h6_df19_rpe_m6_preparation_20261110/synthetic_tests.txt)。実m6のreference/signal/Nは別途許可待ち。

[H6 DF19 RPE D2結果](research/h6_df19_rpe_d2_results_20261110/REPORT.md)：固定8候補のsignal・bias・normalization・phase・radius・十分N。[正本](research/h6_df19_rpe_d2_results_20261110/run01/result.json)／[D1と合わせた6roundの転記](research/h6_df19_rpe_d2_results_20261110/COVERAGE.json)。actual/全Gは未資格、残りm6..12、公開照合後STOP。

[H6 DF19 RPE D2計画](research/h6_df19_rpe_d2_preparation_20261110/REPORT.md)：4round・8候補、保存T=.4 reference再利用、新reference3時点、対角化なしのaction接続仕様とSTOP条件。[有限menu](research/h6_df19_rpe_d2_preparation_20261110/PLAN.json)／[静的確認](research/h6_df19_rpe_d2_preparation_20261110/CHECKS.txt)。科学実行は別途許可待ち。

[H6 DF19 RPE D1結果](research/h6_df19_rpe_d1_results_20261110/REPORT.md)：係数prior・h=.1・13round条件、保存4候補の条件付き十分N、cohort別2round部分費用。actual/全Gは未資格、公開照合後STOP。

[H6係数prior・D1実行準備](research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md)：H4 prior/scalarを再利用し、人工12件PASS。実H6科学評価は添付指示§9の別許可待ち。公開照合後mandatory STOP。

[H6 RPE D1実装・人工検証](research/h6_df19_rpe_d1_20261110/REPORT.md)：H4算術を保持したH6入力層、人工25件PASS、固定4点の科学評価は別許可待ち。[実行計画](research/h6_df19_rpe_d1_20261110/D1_PLAN.md)。公開照合後mandatory STOP。

[Track D×元PR/RPE共通interface](research/track_d_resource_interface_20261110/REPORT.md)（2026-11-10）：Strong／whole-path Referenceの科学inputs・control価格を分離し、既存総集計へ薄く接続。固定H4/CDF8/LD2/S2/s1/r32/K0/h=.1・元H ground/準備費除外・13round/P0/C1 OFFは旧N/価格/総値の読取専用。人工＋read-only 24 test method PASS、実H4の新N・再集計・回路・fit0。他algorithm統合なし。次候補は同条件r16（未実行）、公開確認後STOP。

[PR＋RPE共同最適化への統合調査](research/pr_rpe_joint_integration_20261110/REPORT.md)（2026-11-10）：利用者指示により評価器追加開発から総資源共同最適化の接続へ重点移行。元PR/RPE API・D-CDF8の既存13-round接続・Closed P3共同選択・新A Q3後続・強PF再資格を静的照合。推奨初手は固定CDF8のα配分×control方式の限定scalar検証案。新N／価格再解析／fit／回路／全round再集計0、実行計画のみ。公開確認後STOP。

[Track D run内部再走査削減](research/track_d_bridge_run_reuse_20261110/REPORT.md)（2026-11-10）：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。run内部再走査削減で旧新の各反復word／位相／全56資源一致、70人工回帰＋保存54case PASS。前回高速版に比べnative段階q4 1.15〜1.18倍・q8 1.32〜1.33倍、q8 RSSは4〜8%増。全unit epochとCX frontier再構築／最悪O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。

[Track D 同一rule高速化](research/track_d_stream_bridge_fast_20261110/REPORT.md)：1Q→CX直結・exact run cache・冗長再計数削減。q4/q8の最終word／phase／round／全資源を保持し、q8 native-stage約2.1倍高速・RSS約53%減。raw loweringとend-to-endは別評価、一般P0／長q保証なし、公開照合後mandatory STOP。

[Track D Phase精度と保存q8移送](research/track_d_phase_q8_transfer_20261110/REPORT.md)：MPFR160 ledgerによる全体phase会計、q4新旧word同一、保存q8_sample00の全28資源一致。53人工回帰PASS、20反復と履歴増加を計測。一般P0／長qへの保証なし、新P0／sampling／simulation0、公開照合後STOP。

[Track D 保存H4 raw→P0価格移送](research/track_d_h4_native_bridge_20261110/REPORT.md)：固定q4_sample00／cut2／C1 OFFのStrong/Reference X/Y、予測commit固定後の全28資源一致。4並列約25秒、49人工回帰、宣言phase差と長qの限界を記録、新P0／H4 build／simulation0、公開照合後STOP。

[Track D native emitter・1Q界面結果](research/track_d_native_emitter_development_20261109/REPORT.md)：47人工回帰、保存54人工caseの作用／全7資源一致。全wrapper QCを省くsinkと公式2×2合成の部分bridgeを追加。一般P0・H4長qへの資格なし、新P0／H4／保存H4再解析0、公開照合後STOP。

[Track D T1〜H0結果](research/track_d_t1_h0_p0_bridge_20261109/REPORT.md)：固定54人工case・70適格native列、37人工回帰、指定3人工P0、C1 OFFの保存H4 q4/q64_sample00・8資源照合を完了。CX隣接bridgeの有限対応は取得、全H4 pre-P0価格はnull。主A/F/Sと旧証拠を維持し、公開照合後mandatory STOP。

[Track D T0実装](research/track_d_t0_native_resources_20261109/REPORT.md)：確定native streamのcount・RZ/CX/total depth・size・幅、wire transfer／連結／embeddingを実装。人工回帰29件PASS、全word保持・新P0・simulation不要。P0 bridgeは未資格、A/F/S・旧SOURCE/結果/STOPを保全、新H4/再解析/fit/N/13round更新0。T1は提案のみ、公開照合後mandatory STOP。

[Track D 長q構造価格設計](research/track_d_longq_structural_resource_design_20261109/REPORT.md)：固定native列のcount/max-plusは厳密、未compile入力からのP0価格bridgeは未完成。208−224qの条件付き傾き説明と208/界面残差の未解決を整理し、既存emitter/counter/特徴のstream再利用を提案。整数mock7件PASS、新科学計算・P0・保存価格再解析・fit/N/13round更新0。A/F/Sと既存STOPを維持し、公開照合後mandatory STOP。

[O5 C1実H4/q4結果](../docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/Number-CDF8/元順LD2/S2/s1/r32/K0/δ=.1、保存q4_sample00/cut2・元H ground仮定/準備費除外。全256 raw/新4 post PASS、両方式CX−30、depthはStrong−178/Reference−81、相対depth差縮小97・CX差不変。旧4適格再利用、新P0合計4。χ要約の配列形状エラーSTOPを保全し保存結果集約のみ完了、科学retry0。全law/全RPE未資格、原本非公開・公開照合後STOP。

[O5 C1人工結果](research/track_d_o5_c1_head_tail_artificial_20261108/REPORT.md)：固定2-mode/3-block q2_mixed/cut1/h=.1/τ=.2/K0、scalar .17/identity .04（分子geometry/basis/rank/LD非該当）。head→tailの既存relative橋でlocal7 positive/7 negative・raw41比較・post8 PASS。Strong/Reference双方でRZ−5/CX−2/depth−7/size−11、幅不変、相対R−S差は不変。新P0 4・旧4適格再利用。保存型エラーの旧STOPを保全し、SOURCEaf6fc07fで同scope未完了処理を継続。H4/fit/N/13round/q64操作0、公開照合後mandatory STOP。

[O5 C1人工PLAN](research/track_d_o5_c1_head_tail_artificial_20261108/PLAN.md)：固定q2_mixed、局所negative/全入力correctness、最大8 wrapper。

[O5候補C1：head-tail静的調査](research/track_d_o5_c1_head_tail_static_20261108/REPORT.md)：閉じたhead出口→tail入口の異frame Gaussianを既存relative変換へ接続する候補を同定。head/tail内部共有・step融合は実装済み、P0局所重複と全接口UNKNOWNを区別。C1継続案は一種類の最小人工判別までで未採択。新コード/人工fixture/correctness/P0/保存価格再解析/fit/N/13round0、q64非照会・非介入、公開照合後mandatory STOP。

[O5 q4 cohort metadata PLAN](research/track_d_o5_q4_cohort_metadata_20261108/PLAN.md)／[結果](research/track_d_o5_q4_cohort_metadata_20261108/REPORT.md)：保存16本すべて中央cut2。新選択15・旧decision再利用1、価格再解析/回路/compileなし、B追加H4保留・Cは提案のみ。

[C8-1 q64限定再開PLAN](research/track_d_c8_1_q64_20261108/COMPLETION_PLAN.md)：追加sampling0、保存sample00最終gate→残7、旧STOP保持。

[C8-1保存4回路の数値修正結果](research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_RESULT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。同じsample00/4 P0に対し、SOURCE46ce57e1で旧FAILの値を保存し、SOURCEf1e89590のsimulation fusion無効化で4 postcheck全PASS。固定精度不変、旧原本46項目/STOP保持、新sampling/raw/compile0。最終独立paired/χ総合gate・残7・n8価格統計・長q/全RPEは未実施、公開照合後mandatory STOP。

[C8-1保存4wrapperの数値修正PLAN](research/track_d_c8_1_q64_20261108/NOFUSION_REPAIR_PLAN.md)。残7 pathは対象外。

[C8-1数値修正PLAN](research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_PLAN.md)：保存回路だけの承認済み再検証。

[Track D C8-1最終結果](research/track_d_c8_1_q64_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。全8 event/512 occurrence/16384 eventsを固定し、sample00 raw1 PASS・旧P0 compile4完了をSOURCE8a6bbdacへ承認済み引継ぎ。両方式X/Y全4 postcheckがCAR-observable Frobenius2e−10でFAILしSTOP。残7/追加28 compile未開始、価格統計/モデル診断未評価。精度緩和/再実行/fit/N/全RPE更新0、旧SOURCE/STOP・非公開原本保持。

[C8-1高速化SOURCE引継ぎ](research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)：compileの過大CPU予約を修正し、旧task完了待ち後の未着手処理を継続。19人工/回帰PASS、sampling追加0、old raw1/compile4再利用。固定q64/cut32・精度・モデル・全32 compile上限維持。最終C8-1結果公開照合後mandatory STOP。

[Track D B/C全13-round条件付き評価](research/track_d_bc_total_20261108/REPORT.md)：同Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、元H厳密ground・準備費除外。保存十分N14738と固定A/F/Sで全13-round条件付き資源会計を一回実施。主AはReferenceでCX11.17%・total depth54.65%・RZ1.12%減、幅9→17。約99.5%が長q proxy、全体不確かさ/actual未資格。新sampling/compile/fit/N0、SOURCEb09b7e7a、公開照合後mandatory STOP。

[Track D B6b＋B/C統合準備](research/track_d_b6b_bc_integration_20261108/REPORT.md)：Referenceのideal coherenceを任意入力へ導出し、whole-V-ray条件とDF半path/最終真空条件を区別。SOURCE13114b2b、人工/回帰24件と人工P0 16 wrapper postcheck PASS。B4-2保存Nの条件付き共通利用・13-round価格coverage・薄い会計adapterを整理。actual N/長q期待価格は未資格、実H4の新計算/価格再解析/N算定/全RPE数値会計0。公開照合後mandatory STOP。
[Track D C7-4](research/track_d_c7_4_model_stability_20261107/REPORT.md)：同H4/CDF8/LD2/S2/s1/r32/K0/h=.1の保存q2〜32・88 path再解析。SOURCEdd31e1e2、4探索fit family/5fold/共有bootstrap、最大残差の保存1 path再照合PASS、人工6長系列/短P0 2本・distinct回帰18 PASS。CX整数残差0、paired depthのbasis-runモデル改善は探索的、native版の精度改善は未確認。q≥64分子処理/N/全RPE費用0、未採択の次段階案、公開照合後mandatory STOP。

[Track D C7-3結果](research/track_d_c7_3_q32_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/8 path。sampling/raw SOURCE35bfd、承認済み高速化compile/post SOURCE94e94。旧raw8再利用・新raw/sampling0、全32 P0/post・pairedχ8 PASS。CX差は全8本で−6960、RZ/size/各depthも有限全8本で減、qubits9→17。一次12量は探索的幅内、別population-line6量幅外。統計/全law/q≥64/N/全RPE未資格、旧STOP/raw非公開を維持、公開照合後mandatory STOP。

[C7-3高速化・切替PLAN](research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)：利用者承認のSOURCE変更。旧raw8 PASSと旧STOPを保全し、未着手compile/postへ切替。case/model/精度/P0は維持、追加sampling/raw再実行0、人工/回帰61 PASS。

[Track D C7-3 PLAN](research/track_d_c7_3_q32_20261107/PLAN.md)：独立q32/cut16/8 path、凍結予測と新domainを事前固定。raw gate後最大32 full P0、C7-2構造feature、探索的価格のみ。q≥64/新fit/N/全RPE対象外、公開後mandatory STOP。

[Track D C7-2](research/track_d_c7_2_cx_structure_20261107/REPORT.md)／[q32案](research/track_d_c7_2_cx_structure_20261107/C7_3_EXECUTION_DRAFT.md)：固定54人工P0と局所CX/phase/DAG検証、全54 postcheck・回帰12 PASS。非加法/単純lane式へ反例。実H4の新計算・再解析0、全path/長q未資格、mandatory STOP。

[Track D C7-1](research/track_d_c7_1_q16_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q16/cut8/16 path。SOURCE462f6bff、raw16・post64・pairedχ16 PASS。CX差は全観測pathで−3376、RZ/size/各depthも有限全16本で減、幅9→17。一次12量は探索的幅内、別population-line4量幅外。統計/全law/q≥32/N/全RPE未資格、raw非公開、mandatory STOP。

[Track D C7-1実行前PLAN](research/track_d_c7_1_q16_20261107/PLAN.md)：独立q16/cut8/16 path、凍結モデル・新seed、raw gateと最大64 P0 wrapperの探索的paired検証。旧SOURCE/STOP保全、原本ローカル、公開照合後mandatory STOP。

[Track D C7-0構造解析](research/track_d_c7_0_structural_preparation_20261107/REPORT.md)／[C7-1以降の条件案](research/track_d_c7_0_structural_preparation_20261107/C7_1_EXECUTION_DRAFT.md)：保存q2/4/8とSOURCEの静的調査。CXの条件付き構造、RZ/size、DAG/lane、estimand/校正/biasと最小q16案。新科学計算・再解析・実装・テストなし、全path/長q/全RPE未資格、公開照合後mandatory STOP。

[Track D C6b](research/track_d_c6b_q8_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1/q8/cut4、独立domainの32 path。元H厳密ground仮定・準備費除外。SOURCE44a58b9fへ承認済み限定引継ぎ、raw32・postcheck128・pairedχ32 PASS。一次frozen条件付き12量は探索的幅内、別population-line診断4量は幅外。統計未資格、全law/長q/N/全RPE未評価、raw非公開、mandatory STOP。

[C6b高速化による残task限定引継ぎ](research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)：旧科学条件・eventを保持し、完了済みraw/compile/postcheckを再実行しない。追加sampling0、全32本/旧新合算128 compile、公開後mandatory STOP。

[Track D C6b独立q8](research/track_d_c6b_q8_20261107/PLAN.md)：32 fresh whole-path、凍結C6aモデル、paired12量の探索的価格診断。確認的PASS・長q・全RPE費用は対象外、rawローカル保全、公開後mandatory STOP。

[Track D C6b-0](research/track_d_c6b0_preparation_20261107/REPORT.md)／[C6b実行前契約案](research/track_d_c6b0_preparation_20261107/C6B_EXECUTION_DRAFT.md)：C6a固定保存値とSOURCEの静的調査だけ。paired優先・追加校正要否・q8事前条件を整理、未採択・未実行、mandatory STOP。

[C6a結果](research/track_d_c6a_paired_price_20261106/RESULT.md)：fresh q2/q4各16、raw32/P0 postcheck128/pairedχ PASS。有限pilot統計・共有校正共分散・C6b提案、rawローカル保全、mandatory STOP。

[Track D C6a固定PLAN](research/track_d_c6a_paired_price_20261106/PLAN.md)：q2/q4各16 fresh paired価格pilot、最大128 P0、C6b未実施。

[C5結果](research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[Track D C5：共通whole-path入口とq2公平比較](research/track_d_c5_q2_20261106/PLAN.md)：保存sample00→01/cut1、C3と同じ強い境界融合・位相集約、raw PASS後だけ最大4 P0 wrapper。C6や新価格標本は未許可。

[Track D C4：期待価格・長qモデルの技術準備](research/track_d_c4_price_model_preparation_20261106/REPORT.md)：強いwhole-path比較の費用境界、保存価格の条件差、校正・独立holdout・不確かさ・B側Nとのinterface。新集計/fit/sampling/回路生成/compile/実装は行わない静的報告。

[C3 q4/q8補完結果](research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。

[Track D C3 q4/q8固定PLAN](research/track_d_c3_q4_q8_20261106/PLAN.md)：whole-path correctnessとstrong境界融合を維持した最大8 wrapper P0価格検証。q1/q2再実行・全law/RPE費用は対象外。
[Track D B5：ideal→actual保証接続](research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md)：sampling law・IID／joint-law・whole-path測定保存・raw誤差予算・方式間共通Nの資格と最小追加検証案。静的調査のみ、公開後mandatory STOP。

[Track D B4-2：元H十分shot数](research/track_d_b4_2_rpe_shots_20261106/RESULT.md)：固定13-round、ideal IID/exact nativeで14738 shots。元H energy/failure条件付きPASS、actual保証・総回路費用なし、mandatory STOP。

[Track D B4-1実装・人工検証](research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md)：元H統計・energy／decoder接続、互換性、interface、B4-2許可境界。73人工／回帰PASS、実H4科学計算なし。

[Track D B3：元H向けRPE統計の接続準備](research/track_d_b3_rpe_statistics_preparation_20261106/REPORT.md)：元H専用接続、N・alias・decoder資格、ReferenceとのN共通化、資源会計とC3受取条件。静的調査・計画のみ、公開後mandatory STOP。

[C2固定q2結果](research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Track D Stage B・Cの技術準備](research/track_d_bc_static_preparation_20261106/REPORT.md)：B（元H向け信号・統計）、C（whole-round構成・価格）、共通の最小検証順序とSTOP。新規科学計算・実装を伴わない静的調査。

[Stage A結果](research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

[Track D Stage A：保存32 trajectoryのpaired Reference検証](research/track_d_h4_paired32_reference_20261106/PLAN.md)：同endpoint・固定cut、全precompile PASS後だけ2方式×X/Y×32の最大128 wrapper比較。元H厳密ground仮定・準備費除外、B/C段階は対象外。

[Track D実H4 Reference移送](research/track_d_h4_reference_validation_20261106/PLAN.md)／[結果](research/track_d_h4_reference_validation_20261106/RESULT.md)：保存word02・固定cut、strong symmetric対照、correctnessと6 full-wrapper compileの限定scope。

[Track D人工Reference検証](research/track_d_minimal_reference_validation_20261106/PLAN.md)：固定1 wordのcorrectness、3方式×X/Yの6 wrapper比較、SOURCE・STOP範囲。

研究別の成果・SOURCE・raw・検証状態への入口は[研究結果への統一索引](research/RESEARCH_RESULTS_INDEX.md)。
このcheckoutにない後続資料も、固定commitから参照できる。

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
- [`research/README.md`](research/README.md)：研究文書内の索引
- [`research/研究目的・研究課題.md`](research/研究目的・研究課題.md)：目的と研究課題
- [`research/研究方法・解析手順.md`](research/研究方法・解析手順.md)：採用する解析手順
- [`research/数値実験・評価計画.md`](research/数値実験・評価計画.md)：検証と評価の計画
- `research/研究ノート/`：時系列の判断記録。現在の仕様ではない

## 現在の主な検証文書

### PF係数

- [`pf_delta_validation.md`](pf_delta_validation.md)
- [`pf_c_system_size_validation.md`](pf_c_system_size_validation.md)

### finite RTE

- [`rte_conventions.md`](rte_conventions.md)
- [`rte_truncation_budget.md`](rte_truncation_budget.md)
- [`finite_rte_signal_validation.md`](finite_rte_signal_validation.md)
- [`df_rte_tail_extraction.md`](df_rte_tail_extraction.md)
- [`df_rte_event_circuit_api.md`](df_rte_event_circuit_api.md)

### コンパイル後回路コスト

- [`random_circuit_cost_validation.md`](random_circuit_cost_validation.md)
- [`rte_boundary_cost_validation.md`](rte_boundary_cost_validation.md)
- [`rte_boundary_pair_validation.md`](rte_boundary_pair_validation.md)
- [`hierarchical_cost_validation.md`](hierarchical_cost_validation.md)
- [`rte_connected_cluster_cost_validation.md`](rte_connected_cluster_cost_validation.md)
- [`rte_compiled_cost_validation_summary.md`](rte_compiled_cost_validation_summary.md)
- [`rte_compiled_event_cost.md`](rte_compiled_event_cost.md)
- [`df_partial_s2_compiled_cost.md`](df_partial_s2_compiled_cost.md)
- [`df_partial_s2_repeated_compiled_cost.md`](df_partial_s2_repeated_compiled_cost.md)

### RPEへの接続

- [`rpe_resource_accounting.md`](rpe_resource_accounting.md)
- [`rpe_hadamard_interrogation.md`](rpe_hadamard_interrogation.md)
- [`rpe_round_cost_connection_validation.md`](rpe_round_cost_connection_validation.md)
- [`rpe_hadamard_failure_validation.md`](rpe_hadamard_failure_validation.md)
- [`rpe_hadamard_proxy_resource_validation.md`](rpe_hadamard_proxy_resource_validation.md)
- [`rpe_allocation_sensitivity_validation.md`](rpe_allocation_sensitivity_validation.md)
- [`rpe_four_round_accounting_validation.md`](rpe_four_round_accounting_validation.md)
- [`rpe_four_round_phase_validation.md`](rpe_four_round_phase_validation.md)
- [`rpe_target_round_horizon_validation.md`](rpe_target_round_horizon_validation.md)
- [`rpe_delta_round_schedule_validation.md`](rpe_delta_round_schedule_validation.md)
- [`rpe_delta_compiled_cost_validation.md`](rpe_delta_compiled_cost_validation.md)

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。

## mainに収録した過去の成果

[収録済み成果と原SOURCE](research/main_collection_20261103.md)：第1回はTrack B G6の既存結果・文書・再現用SOURCE。元の科学的区分とSTOPを保持する。

[枝ごとの収録範囲](research/main_collection_20261103_branches.tsv)と[大きな原データの既存所在](research/main_collection_20261103_raw_locations.tsv)から、main内の成果と原snapshotを辿れる。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルの未コミット資料からの追加収録（2026-11-03）

[原稿・レビュー・研究ノートと保存版の対応](research/main_local_results_collection_20261103.md)。原文の科学的採用や新規実行の認定ではない。

2026-11-08：[Track D（O5）アルゴリズム改善設計](research/track_d_o5_algorithm_design_20261108/REPORT.md)。構造map・固定証拠・A/B/C候補・先行法境界・次の最小作業を6文書へ整理。資料読解/設計のみで、新builder/test/科学計算/価格再解析なし。

## Track D O5 cut人工検証

[REPORT](research/track_d_cut_artificial_20261108/REPORT.md)：実装・全入力correctness・全7資源・均衡負結果・未採択H4移送案。2-mode/q6/16eventの有限人工結果で、分子優位・全law価格・全RPE保証ではない。

[O5保存H4 q4入力STOP](research/track_d_cut_h4_q4_20261108/REPORT.md)：q4_sample00の非公開event/QPY未取得。SOURCE互換確認のみ、新科学call0・資源差UNKNOWN。

[O5実H4 q4最終結果](research/track_d_cut_h4_q4_completion_20261108/REPORT.md)：原本回収後に中央cut2を選択し追加差0。旧4 P0適格再利用、重複compile0。
