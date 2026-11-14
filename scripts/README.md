# scripts 索引

[Full PF／p7保存4件の接続](../docs/research/d_pf_price_saved4_revision_20261114/ADMISSION_UPDATE.md)：既存stage/callerをC新版CELL_PLAN・p7正式受入へ束縛。新規20＋12、元SOURCE/cohortの保存4、別IDの物理schedule重複照合。18 metadata/人工test PASS、実compileなし・STOP。

[最新Full PF metadata接続](../docs/research/full_pf_latest_price_guard_20261114/BUILD_COVERAGE.py)：旧coverage builderを再利用。科学runnerの新設なし。

[B0I・Full-Hの全13round Raw signal取得](../docs/research/truncated_fullh_signal_results_20261114/REPORT.md)（2026-11-14）：固定H4 DF10のB0I p7/S2/Δ.1とFull-H finite-K2/q=2^m/R448を各13round、一回評価。SOURCE5a172629d、26件すべて元数値gate PASS、Raw/corrected/reference/normalizationとB packetを保存。人工caller3件PASS、数値module不変、context復元1、旧signal受入/retry0。数値誤差・law距離・state誤差は未資格、MC/N/価格/G0。旧c546/6306計画・STOPを保全し公開後mandatory STOP。

[全方式の平均compiled価格モデル技術準備](../docs/research/rpe_mean_price_model_preparation_20261113/REPORT.md)（2026-11-13）：外部係数の加法式・境界covers・既存S＋単一C−S・4方式の受動B接続を追加。人工新11＋既存27＝38 PASS。全モデル未資格、実H4/H6の新S/C・保存再解析・sampling/compile/fit/G0。C1係数/26 E[S]/D requestと既存STOPを保全、公開照合後STOP。

[D価格adapterとRPE pipelineの統合](../docs/research/d_price_pipeline_integration_20261113/REPORT.md)（2026-11-13）：公開D candidate_price_adapterをC price_requestsへ接続。A固定LD1/LD2の13round metadataとD path→pending/外部mean→B会計を照合し、新8＋既存36＝44 PASS。path/S/finite cohortを適格E[C]へ昇格せず、軸・SOURCE/law/preparation/compiler/loweringを保持。実H4/preparation/sampling/compile/保存再解析/実N・G0、D物理資格と平均/N証拠は後続。

[RPE総コストpipelineの人工統合](../docs/research/rpe_optimization_integration_20261113/REPORT.md)（2026-11-13）：A PR・C baselineの全round signalをB shot配分/cost/selectへ接続。D向け未評価requestとphysical/lowering bindingを整備。人工新10＋B metadata2＋C/M5回帰24＝36 PASS、4方式の人工全13round・shot最小/G最小・点/区間・未資格/欠測を区別。実H4/saved再解析/実N・価格・G0、科学N資格producerとD価格は未取得。

[ベースラインsignal pipelineの技術準備](../docs/research/baseline_signal_pipeline_preparation_20261113/REPORT.md)（2026-11-13）：ベースライン全round signal経路を既存B/D/数値部品から接続。H4 DF10のFull PF・B0/B0I・Full-H pure RTE、S2/Y4/M5・prefix・明示q/R・有限Kを扱う。人工新12＋既存回帰12＝24 PASS、数値失敗値/欠測/未資格を保持。実H4・保存再解析・残screen・sampling/compile/N/RMSE/G0。

[H4 PR全13roundの平均構造RZ結果](../docs/research/h4_structural_schedule_results_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2・同preparationで残11lawのK0全PASS、新22 E[S]を取得。m0/m6四値を再実行せずcopyし全13round/26軸が同canonicalへ接続。SOURCE de2427d4a、99 wire-law検査、最大mass defect1.714e−12（tol1e−10）。元Sのideal有限law平均でE[C]/Gはnull。新sampling/full-path S/build/compile/fit/N/G0、計算再起動0、旧STOP保全。公開照合後mandatory STOP。

[H4構造価格・全13round経路の技術準備](../docs/research/h4_structural_schedule_preparation_20261113/REPORT.md)（2026-11-13）：固定B3216/同canonicalの13round metadataを接続し、m0/m6四値は83d219cc6からcopy専用。残11lawのK0→PASS時K1 batchを既存core/MeanKernelで準備、new8＋回帰18＝26技術test PASS。R255等を保持し、実T/terminal入力・同γ16外向き二乗法で巨大qR展開を回避。実新K0/E[S]/sampling/compile/fit/N/G0。最大22新Sの一括計画を公開、残差はD担当、公開照合後STOP。

[H4 kernel run02：K0 PASS・K1平均構造RZ四値](../docs/research/h4_structural_kernel_20261113/run02/REPORT.md)（2026-11-13）：SOURCE19b95d8b3で同H4 DF10/S2/p1/δ.1/K2・1prep/2lawを一回実行。K0全9wire PASS、kernel/状態辞書を保存後、m0/m6×XYの実E[S]四値を取得（各mでX=Y、1743.6682557187337/1764659.275564426）。数値区間・元Accounting parityを保存。sampling/full-path S/build/compile/fit/N/G0、E[C]はnull・C1未資格のまま。旧run01/STOP保全、公開照合後mandatory STOP。

[H4 K0入力STOPと技術修正版](../docs/research/h4_structural_kernel_20261113/REPORT.md)（2026-11-13）：K0→PASS時K1 runnerと数値契約を実装、新8（人工7/固定metadata1）＋旧10技術test PASS。SOURCE0f7d65d5fはB configuration locator誤りで物理K0前にUNKNOWN STOP。実descriptor/kernel/parity/平均S四値は未実施・null。参照chainを修正版19b95d8b3でmetadata照合し、科学再実行なし。指示§4 STOP保全、run02案のみ公開。新sampling/full-path S/compile/fit/N/G0。

[H4 PRの物理閉状態診断](../docs/research/h4_pr_physical_closure_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2の1prep・R2/R128を診断。324非identity/9basis、reset/phase/head境界からS observableの有限wire-marginal状態の十分条件を確認（tail enclosure80〜120）。物理kernel/parity/実平均S/Cは未取得、MODEL false。最終10技術test PASS。初回seed設定STOPを保全し同scope修正、成功SOURCE eeccdf18e・縮約解析2b3b0ee7f。新sampling/full-path S/build/compile/fit/N/G0、旧S0a等を保全、公開照合後STOP。

[固定H4 S0の平均compiled価格検証計画](../docs/research/h4_s0_mean_price_plan_20261113/REPORT.md)（2026-11-13）：SOURCE bdd2b9356。固定A S0のexactκ/M12・H4 m0(q1/R2)/m6(q64/R128)を公開B V2へmetadata接続。新7技術test PASS、13round価格null。両候補の一致direct/modelは公開索引に未確認。閉状態診断1prep/2law・compile0を第一案、別案は有限4path/最大8wrapper。新S/価格再解析/sampling/compile/fit/N/G0、C1未資格・S0a履歴を保全し、公開照合後STOP。

[B V2価格接続と長q平均価格の次段階設計](../docs/research/bv2_mean_price_validation_design_20261113/REPORT.md)（2026-11-13）：SOURCE58382c54c。固定B6a9 V2 MeanPriceLookupを再利用するC受動入口を追加、新12人工/metadata結合test PASS。値ありrecordに平均compiled quantity明示を要求し、S0a/旧ID付替え/資格昇格を拒否。旧118候補/236軸と4方式の資産/欠測を静的整理。実平均S/C・物理閉包は未取得、選択B domainのclosure診断等4候補を未承認計画化。新S/C再解析/sampling/compile/fit/N/G0、旧結果/STOP保全、公開照合後STOP。

[S0a run02：実保存8件の構造S照合](../docs/research/s0a_saved_structure_results_20261113/run02/REPORT.md)（2026-11-13）：修正版SOURCE9516351e6で保存normal index0の4path×XY8 Sを一回評価し、元C1保存Sと8/8整数完全一致。各X/Y SはH4 q8/R8=21664、q16/R8=46559、H6 q1/R32=158669、q2/R32=321151。input/preparation/law/phase/allocation/digest検査を通過、元roleとrun01失敗STOPを保全。平均S/C・物理閉包・モデル資格は未取得、qualified=false。新sampling/compile/fit/N/G0、公開照合後STOP。

[S0a run01の設定エラーと技術修正](../docs/research/s0a_saved_structure_results_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65でrun01を1回起動、context exec scopeのast渡し漏れでS開始前に失敗STOP。新S0/8、保存S照合0、全4path未評価。修正版SOURCE9516351e6と人工14件PASSを公開、run02案はmetadata準備だけで未実行。旧STOP/role/モデル資格を保全、修正版再試行はユーザー判断、公開照合後mandatory STOP。

[S0a保存構造価格の実行準備](../docs/research/s0a_saved_structure_preparation_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65。固定C1の保存4path/XY8計数に向けreader・law/identity検査・S専用runnerを準備。新13＋旧9＝22技術test PASS。raw/inputは所在のみ確認、実S・保存S比較は未実行。元係数/資格/平均provider不変、内容hashと実contextは実行時検査。初回人工importのlowering副作用と修正をREPORTへ記録。明示科学許可待ち、公開照合後STOP。

[C1状態閉包とStreaming計数の技術検証](../docs/research/c1_physical_state_stream_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE c1135fdb9。C1のpending basis/Z・tail phase・terminalをSOURCEで調査し、元計数wordへforward-only tailを渡す最小consumerを追加。新9＋旧3＋sampler toy1＝13技術test PASS。単純縮約の人工反例は確認、実H4/H6の閉包・event生成なしの平均S/compiled価格は未確認。consumerは全event処理時間が残り、producer全接続は未実装。S0a保存4path/XY8件は未実行案、実科学計算・再解析・compile・fit・N/G0、公開照合後mandatory STOP。

[長q・大Rの平均compiled価格の技術準備](../docs/research/mean_compiled_price_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE09489eedb。C1 Accounting/旧局所期待formを再利用し、closed-state期待報酬の二乗合成と受動c_providerを実装。新15＋旧pure3＝18技術test PASS、人工全列挙と一致。固定状態ならqR展開を避けられるが、実C1の累積Z/phase閉包・平均S・compiler残差は未資格で実価格missing。C−S残差のみを扱い二重計上を拒否。κ scheduleはB外部contextのcopy、M/N最適化なし。旧40/16案は非実行、将来S0保存再解析等は別許可。実科学計算/compile/価格fit予測/N/G0、公開照合後STOP。

[A P1対応・長q/大R価格検証の準備](../docs/research/r_price_longq_largeR_preparation_20261111/REPORT.md)（2026-11-11）：公開A71ba49bbの12候補・同policy条件付きNを無変更copy。C1価格との重複はH6 m4/R16,R32の校正2点。第一案はH4 m6/H6 m8のbaseline＋R128、4セル・最大40 wrapper、原因分離R-onlyと短時間各16は別の任意案。metadata SOURCE f0daeb96b、8技術test PASS。構造Sのstream化/局所境界法を比較し、m12やR256を自動compile/外挿しない。新科学計算・価格予測・再解析・N/G0、科学実行未承認、公開照合後STOP。

[R価格Phase 1実行結果](../docs/research/r_price_phase1_results_20261111/REPORT.md)（2026-11-11）：Linear H4/H6・1Å・STO-3G・保存DF10/19、H4 S2/δ.1/p1/K2とH6 m5_best/δ1.6/p7/K2。SOURCE1a87e5331で通常64＋別rare16 compile、保存C/S59 path再利用。係数を7eeabf0aで検証取得前に凍結し再fitなし。独立n8の相対RMSEはH4 q16/R8で1.039841%、H6 q2/R32で0.229959%。有限cohort観測、採用閾値未設定・model未採用、全law/長q-R/総CI/全Gは未資格。新signal/N/G0、公開照合後mandatory STOP。以下の準備記録・過去STOPは履歴として保持。

[R価格Phase 1の実行準備](../docs/research/r_price_phase1_preparation_20261111/REPORT.md)（2026-11-11）：保存C/S接続・paired3係数fit・Git凍結後validation・別unit rare診断の専用runnerを準備。公開A/B metadata整合、新14＋旧R回帰9＝23人工/mock PASS。固定SOURCE/PLAN、最大64＋16 wrapperの科学scopeは未実行。実分子sampling/S再計数/compile/fit/価格予測/再解析/N/G0、公開照合後STOP。

[R可変compiled価格の調査・設計](../docs/research/r_variable_compiled_price_design_20261111/REPORT.md)（2026-11-11）：H4 S2/δ.1/p1/K2・H6 m5_best/δ1.6/p7/K2の固定Rモデルを保全し、構造価格＋少数compiled残差補正を提案。R binding/保存lookup adapter、人工/mock9件PASS。新R Cはnull、実科学計算・価格取得/予測/fit/G0。最小新q×R/希少event計画は未承認・未実行、公開照合後STOP。

[H6 provider統合・H4/H6全13round N/C coverage](../docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md)（2026-11-11）：固定B cb2 APIへ保存lookup factoryを追加。C provider/テストをbyte一致再利用、24技術test PASS。H4公開89c91c69の整数検証済みNとH6公開f4c0bdaのNを静的転記し52行条件付きPASS。直接CはH4 5候補/H6 8候補、PR全law/長q/共有CI/全RPE採用は未資格。新科学計算・実N×C/G0、B実行領域へ非介入、公開照合後STOP。

[H4価格providerのB接続](../docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：科学runnerは追加せず、Bのh4_saved_price_providerをrun_missingのC callbackへ渡す。lookupのみ、null/資格不足の自動再呼出しなし。

`run_h6_df19_rpe_m8_12.py`の[完了結果](../docs/research/h6_df19_rpe_m8_12_results_20261110/REPORT.md)：SOURCE fca50d36fの一括10候補全ideal条件付きPASS。保存m0..7を再計算せず26候補/13roundのNを取得。価格/G/最適化へ自動進行せず、再実行しない。

`run_h6_df19_rpe_m8_12.py`：[残5roundのN専用一括準備](../docs/research/h6_df19_rpe_m8_12_preparation_20261110/REPORT.md)。既存`run_missing/H6ActionBatch`を1回呼び、保存m0..7を再計算せず13round coverageへ接続する。追加人工7件PASS。実行は明示許可後だけ、C/G・retry/resumeなし。

`plan_pf_pr_rpe_cost.py`：[統合REPORT](../docs/research/pf_pr_rpe_cost_integration_20261110/REPORT.md)。固定Git JSONからH4/H6 schedule・candidate・保存N/C・不足batchを静的に出力する。実signal/N/価格予測/G実行modeなし。数値入口は`pf_pr_rpe_pipeline.run_fixed_evaluation`で、実データ使用には別途科学scopeが必要。

`run_h6_df19_rpe_m7.py`の[完了結果](../docs/research/h6_df19_rpe_m7_results_20261110/REPORT.md)：SOURCE dcd6a7c87の固定2候補を一回評価、ideal条件付きNは各軸PF705/PR846。旧roundは再計算せず保存値へ接続。m8以降・価格へ自動進行しない。

`run_h6_df19_rpe_m7.py`：[m7準備](../docs/research/h6_df19_rpe_m7_preparation_20261110/REPORT.md)。T12.8/qPF8の固定PF/PR各1候補だけを旧D2 actionへ接続する。`check_h6_df19_rpe_m7.py`の人工3件PASS、旧m6/D2 guard不変。実H6は明示許可後にfresh outputで実行し、m8以降・価格へ進まない。

`run_h6_df19_rpe_m6.py`の[完了結果](../docs/research/h6_df19_rpe_m6_results_20261110/REPORT.md)：SOURCE c9aa89548で固定2候補を一回評価し両方ideal条件付きPASS、PF/PR各軸N705/772。旧結果とrun01を保全、m7以降・価格へ自動進行しない。

`run_h6_df19_rpe_m6.py`：[m6準備](../docs/research/h6_df19_rpe_m6_preparation_20261110/REPORT.md)。既存D2のaction/geometryを変更せず、T6.4・PF/PR各1候補だけへ接続する。`check_h6_df19_rpe_m6.py`の人工3件PASS。実H6は別途明示許可後にfresh outputで一回実行し、価格・m7以降へ進まない。

`run_h6_df19_rpe_d2.py`の[完了結果](../docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md)：SOURCE e0503b769、固定8候補を一回評価し全てideal条件付きPASS。run01と旧D1/計画を保全、m6以降・価格・全Gへ自動進行しない。

`run_h6_df19_rpe_d2.py`：[固定D2計画](../docs/research/h6_df19_rpe_d2_preparation_20261110/REPORT.md)の4round・8候補だけを実行する入口。保存H/stateとtail定数、T=.4 reference、D1政策を再利用し、対角化なしの行列指数/Taylor作用と既存D1 geometryでNを評価する。科学実行は利用者の明示許可範囲に限り、fresh outputを指定する。sampling/build/compile/fit/Gなし。人工7件の記録は[こちら](../docs/research/h6_df19_rpe_d2_results_20261110/synthetic_tests.txt)。

`run_h6_df19_rpe_d1.py`：[係数prior・D1一括計画](../docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md)。保存係数prior→h=.8/.1→固定4点N→条件付き部分費用の明示入口。今回実行したのは`--help`と人工testのみ。実H6入力を渡す実行は別途科学許可が必要。

[H6 RPE D1](../docs/research/h6_df19_rpe_d1_20261110/REPORT.md)：今回追加したのはAPIと人工test。`python -B tests/check_h4_df10_rpe_connection.py -v`／`python -B tests/check_h6_df19_rpe_connection.py -v`で人工25件。実H6の数値driverは起動していない。[別許可後の計画](../docs/research/h6_df19_rpe_d1_20261110/D1_PLAN.md)。

[Track D×元PR/RPE共通interface](../docs/research/track_d_resource_interface_20261110/REPORT.md)（2026-11-10）：Strong／whole-path Referenceの科学inputs・control価格を分離し、既存総集計へ薄く接続。固定H4/CDF8/LD2/S2/s1/r32/K0/h=.1・元H ground/準備費除外・13round/P0/C1 OFFは旧N/価格/総値の読取専用。人工＋read-only 24 test method PASS、実H4の新N・再集計・回路・fit0。他algorithm統合なし。次候補は同条件r16（未実行）、公開確認後STOP。 実装：`trottertracks.track_d_resource_interface`、読取CLI：`scripts/inspect_track_d_resource_interface.py`、test：`tests/check_track_d_resource_interface.py`。

[Track D run内部再走査削減](../docs/research/track_d_bridge_run_reuse_20261110/REPORT.md)（2026-11-10）：Linear H4／1Å／STO-3G／Nα=Nβ2・元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、保存q4/cut2・q8/cut4各sample00／C1 OFF、元H ground仮定・準備費除外。run内部再走査削減で旧新の各反復word／位相／全56資源一致、70人工回帰＋保存54case PASS。前回高速版に比べnative段階q4 1.15〜1.18倍・q8 1.32〜1.33倍、q8 RSSは4〜8%増。全unit epochとCX frontier再構築／最悪O(RL)/O(L)は残る。A/F/S不変、新P0／sampling／q16以上0、公開照合後STOP。 実装：`src/trottertracks/track_d_bridge_run_reuse.py`、`scripts/run_track_d_bridge_run_reuse.py`、`tests/check_track_d_bridge_run_reuse.py`。

[Track D stream bridge性能結果](../docs/research/track_d_stream_bridge_fast_20261110/REPORT.md)：`run_track_d_stream_bridge_fast.py`の固定入力準備・人工54比較、旧keyword不備を保全した`complete_track_d_stream_bridge_benchmark.py`、同時主比較`measure_track_d_stream_bridge_paired.py`。最初のfrozen runnerのbenchmark keyword不備は履歴として保持し、完成済みCLIを再開しない。新しい実行ではmodule APIと互換adapterをscopeに合わせて再利用する。原本／run01／run02／STOP保持、新P0=0。

`run_track_d_phase_q8_transfer.py`：[Phase＋q8結果](../docs/research/track_d_phase_q8_transfer_20261110/REPORT.md)。selectで価格非依存にq8先頭IDを固定、executeでq4 phase4＋q8 raw予測4、予測commit後のみcompare。旧q4観測／counterを再利用し、新P0は呼ばない。run01の初回preflight失敗／修正／結果／STOPを保持し、追加pathやq16以上へ自動進行しない。

`run_track_d_h4_native_bridge.py`：[保存H4 q4結果](../docs/research/track_d_h4_native_bridge_20261110/REPORT.md)。predictはraw4のみ、比較前にcommitで固定し、compareが既存compiled4と照合。`check_track_d_h4_native_phase.py`は固定予測word／compiledのread-only順序・角・phase診断だけ。run01／旧STOPを保全し、再開／追加pathへ自動進行しない。

`run_track_d_native_emitter_development.py`：[技術結果](../docs/research/track_d_native_emitter_development_20261109/REPORT.md)。固定54保存人工raw QPYをhash照合し、最大5qubitのnative lowering／部分bridge／位相付き作用／保存価格を確認した完了済みrunner。新P0・H4・sampling・fit/N0。run01／STOPを保全し、追加実行へ自動進行しない。

`run_track_d_t1_h0.py`：[固定PLAN](../docs/research/track_d_t1_h0_p0_bridge_20261109/PLAN.md)の保存54人工case、指定3人工P0、保存H4 q4_sample00/q64_sample00だけを扱う完了済み一括runner。新sampling/H4 compile/H4 simulationは行わず、原本はread-only。run01/STOPを保持し、再開・追加実行しない。[結果](../docs/research/track_d_t1_h0_p0_bridge_20261109/REPORT.md)。

[Track D T0](../docs/research/track_d_t0_native_resources_20261109/REPORT.md)：新しい科学runnerは追加せず、`python3 -B tests/check_track_d_native_resources.py NativeTests`と既存環境の`python -B tests/check_track_d_native_resources.py QiskitNativeTests`で人工計数／readerを検査する。新P0・H4・simulationなし。既存設計mockも回帰実行し合計29件PASS。

[O5 C1実H4/q4結果](../docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/Number-CDF8/元順LD2/S2/s1/r32/K0/δ=.1、保存q4_sample00/cut2・元H ground仮定/準備費除外。全256 raw/新4 post PASS、両方式CX−30、depthはStrong−178/Reference−81、相対depth差縮小97・CX差不変。旧4適格再利用、新P0合計4。χ要約の配列形状エラーSTOPを保全し保存結果集約のみ完了、科学retry0。全law/全RPE未資格、原本非公開・公開照合後STOP。

`run_track_d_head_tail_artificial.py`：固定q2_mixedの一回限定C1比較。局所/全入力PASS後だけP0、旧C7-2 baseline適格再利用、公開後STOP。

`run_track_d_o5_q4_metadata.py`：保存C6a q4全16本のevent-only読取、固定selector/featureへの薄い接続と集計。

`run_track_d_c8_1_completion.py`：保存sample00の最終pairedχを入口とし、PASS後だけ保存残7raw/最大28P0・post、全8適格時だけ価格統計。

`run_track_d_c8_1_nofusion.py`：旧診断完了後、保存q64の4 postcheckだけをsimulation fusion無効で再検証しSTOP。

`run_track_d_c8_1_diagnostics.py`：承認済み保存q64 postcheckの失敗診断、数値条件不変。

[C8-1停止済み結果](../docs/research/track_d_c8_1_q64_20261108/REPORT.md)：旧SOURCE6b1d613d sampling/raw/P0と新SOURCE8a6bbdac postの来歴を保全。sample00 post全4が固定observable許容値で不成立、残7未開始。両runnerは再実行・resumeしない。次の数値診断はユーザー/GPTの新しい明示許可後のみ。

[C8-1高速化引継ぎrunner](run_track_d_c8_1_accelerated.py)：[承認済みPLAN](../docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)。旧controllerのactive task drain/STOPを待って未着手taskだけ継続する。compile予約1、raw/post native並列、sampling0/完了済み再実行0。旧SOURCE/STOPを保持、sample00 gate・固定精度・総32 compile上限は不変。

[B/C固定scalar runner](run_track_d_bc_total.py)：[PLAN](../docs/research/track_d_bc_total_20261108/PLAN.md)の保存N/直接平均/固定A・F・Sによる一campaign。`--preflight`はmetadata照合、`--execute`は全13-round条件付き会計。新回路/fit/Nなし、失敗後retry/resume禁止、公開照合後mandatory STOP。

[B6b明示runner](run_track_d_b6b.py)：[固定PLAN](../docs/research/track_d_b6b_bc_integration_20261108/PLAN.md)の固定Git metadata/coverageと4人工case・最大16 P0 wrapperだけ。実H4のsampling/build/compile・価格再解析・N算定・全RPE集計を実行しない。人工検証後は結果公開とmandatory STOP。

`run_track_d_c6b_accelerated.py` は[利用者承認済みSOURCE引継ぎ](../docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)の残task専用入口。固定済みeventを読み、完了済みraw/compile/postcheckを再実行しない。新sampling0、旧新128 compile上限。

`run_track_d_c6b.py` is the explicit q8/n32 exploratory runner, separate from the frozen C6a q2/q4 entry. See [fixed scope](../docs/research/track_d_c6b_q8_20261107/PLAN.md). Canonical sampling, C3/C5 builders and the existing P0/checker are reused. Raw PASS gates each path's maximum four compiles; no automatic additional stage after STOP.

C6a結果のC6b提案整理は `summarize_track_d_c6a.py`（完全pilotの既存統計・モデルだけ、q8実行なし）。[結果](../docs/research/track_d_c6a_paired_price_20261106/RESULT.md)。

C6a専用入口：`run_track_d_c6a.py`。[固定PLAN](../docs/research/track_d_c6a_paired_price_20261106/PLAN.md)。既存canonical sampler/CAR/C3/C5/P0を再利用し、明示実行だけ。

[C5結果](../docs/research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[run_track_d_c5_q2.py](run_track_d_c5_q2.py)は[C5固定q2 PLAN](../docs/research/track_d_c5_q2_20261106/PLAN.md)専用。共通ordered requestでrawを検査し、PASS時だけ最大4 wrapper。完全byte一致のC2 compiled保存回路は再利用し、新samplingやq4/q8再実行はしない。

[C3 q4/q8補完結果](../docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。 補完入口は[`complete_track_d_c3_run02.py`](complete_track_d_c3_run02.py)、保全回帰testは`tests/check_track_d_c3_completion.py`。

Track D C3は[`run_track_d_c3_q4_q8.py`](run_track_d_c3_q4_q8.py)。[固定PLAN](../docs/research/track_d_c3_q4_q8_20261106/PLAN.md)でq4/q8だけを扱い、既存compile/postcheckを再利用する。通常testから科学実行しない。
[run_track_d_b4_2.py](run_track_d_b4_2.py)は[B4-2固定PLAN](../docs/research/track_d_b4_2_rpe_shots_20261106/PLAN.md)の保存B1→元H ideal統計専用。明示`--execute`、fresh run01、retry/resume0、mandatory STOP。一般testや旧runnerから自動実行しない。

[Track D B4-1](../docs/research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md)は科学runnerを追加しない。人工検証入口は `PYTHONPATH=src python -B -m pytest -p no:cacheprovider -q tests/check_track_d_original_h_accuracy.py`。実H4評価は別許可。

[C2固定q2結果](../docs/research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](../docs/research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](../docs/research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](../docs/research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Stage A結果](../docs/research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

[run_track_d_h4_paired_reference.py](run_track_d_h4_paired_reference.py) は[Stage A保存32 path](../docs/research/track_d_h4_paired32_reference_20261106/PLAN.md)専用。元SOURCE・保存events/reference/rawを必要とし、全precompile PASS後だけ最大128 wrapper compile。新sampling・RPE統計／全費用へ進まない。

[run_track_d_h4_reference.py](run_track_d_h4_reference.py) は[実H4固定word02](../docs/research/track_d_h4_reference_validation_20261106/PLAN.md)のcorrectnessとPASS時6 full-wrapper compileに限定する。元SOURCEと保存結果の隔離入力が必要で、終了後mandatory STOP。

[run_track_d_minimal_reference.py](run_track_d_minimal_reference.py) は[固定人工path](../docs/research/track_d_minimal_reference_validation_20261106/PLAN.md)のcorrectnessとPASS時の6 wrapper compileだけを実行し、mandatory STOPする。通常の回帰testではない。

`scripts/` はコマンドラインから実行する入口を置く。研究ロジックの本体は原則として
`src/trotterlib/` にあり、runnerは条件の読込み、呼出し、成果物保存を担当する。

最新の研究段階は[`../docs/research/研究概要・現状.md`](../docs/research/研究概要・現状.md)、
各runnerと成果物の対応は[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を
参照する。下記の「現行」は、最終科学結論を意味せず、現在の近似検証で使う経路を表す。

## 現行の検証runner

### PF誤差係数

| runner | 用途 |
|---|---|
| `run_pf_delta_validation.py` | 小規模系でPF係数の定義・delta依存・参照値を比較 |
| `run_pf_c_system_size_validation.py` | H-chainの系サイズ方向とstate-action経路を検証 |
| `run_df_ground_state.py` | DF Hamiltonianの基底状態snapshotを作成 |
| `benchmark_df_ground_state.py` | 基底状態計算経路の計測 |
| `validate_pauli_partial_cgs_all_prefixes.py` | Pauli partial回路係数のprefix検査 |

### finite RTE

| runner | 用途 |
|---|---|
| `run_finite_rte_signal_validation.py` | 演算子誤差、状態上の複素期待値、半径、位相上界を検証 |

### コンパイル後回路コスト

| runner | 用途 |
|---|---|
| `run_random_circuit_cost_validation.py` | イベント単純加算と一体コンパイルを比較 |
| `run_rte_boundary_cost_validation.py` | 2・3イベントの境界補正を検証 |
| `run_rte_boundary_cost_replication.py` | 境界分類を別乱数seedで再検証 |
| `run_hierarchical_cost_validation.py` | 局所補正、長さ、制御付き反復をまとめて検証 |
| `run_rte_order_stratified_cost_validation.py` | 低確率Taylor次数を条件付き抽出して検証 |
| `run_rte_connected_cluster_cost_validation.py` | 1--3イベント境界補正の運用推定を検証 |
| `run_rte_connected_cluster_transfer_validation.py` | $L_D$・短時間幅が異なる条件へ移した場合を検証 |
| `run_rte_connected_cluster_holdout_supplement.py` | 独立検証の標本を補充 |
| `run_rte_connected_cluster_k4_calibration.py` | 不通過条件の4イベント係数を較正 |
| `run_rte_connected_cluster_k4_extrapolation_diagnostic.py` | 4イベント補正の外挿診断 |
| `run_rte_paired_k4_l8_validation.py` | 同一イベント列で4イベント補正の構造残差を検証 |
| `run_controlled_repetition_holdout.py` | 制御付き反復回路の反復数依存を検証 |
| `run_rte_cost_angle_invariance_validation.py` | 回転角をまたぐmetric cache再利用可否を検証 |
| `run_h5_system_size_cost_validation.py` | H5の対応あり構造検証 |
| `run_h5_independent_cost_validation_batch.py` | H5の独立係数較正・未使用回路検証 |

### RPE信号・測定回数・1 shotコストへの接続

| runner | 用途 |
|---|---|
| `run_rpe_round_cost_connection_validation.py` | 短いRPE段で信号半径、shot数、回路costを接続 |
| `run_rpe_hadamard_failure_validation.py` | 仮想Hadamard測定と失敗確率を検証 |
| `run_rpe_hadamard_proxy_resource_validation.py` | $q=8$の1 shot cost proxyと資源集計を検証 |
| `run_rpe_allocation_sensitivity_validation.py` | 位相誤差・失敗確率配分の感度を比較 |
| `run_rpe_four_round_accounting_validation.py` | $q=1,2,4,8$の限定4段を集計 |
| `run_rpe_four_round_phase_validation.py` | $q=8$物理信号と4段の分枝付き位相復元を検証 |
| `run_rpe_target_round_horizon_validation.py` | 目標精度から必要round数を決め、固定設定の長$q$可否を行列診断 |
| `run_rpe_delta_round_schedule_validation.py` | 実行済み$\delta$をPFでscreeningし、round別$(r_m,K_m)$ scheduleを行列検証 |
| `run_rpe_delta_compiled_cost_validation.py` | 検証済み局所境界係数をround scheduleへ接続し、中央RTEブロックのcompiled-cost proxyを比較 |

## 長時間・複数条件のbatch runner

次のrunnerは、上記の検証を複数jobで実行・再開するためのもの。単独の数値だけで結論を
判断せず、対応する検証文書とmanifestを読む。

- `run_parallel_validation_batch.py`：共有サーバー向けtask manifest生成、dry-run、bounded実行、resume、status
- `run_rte_cost_data_batch.py`
- `run_rte_cost_followup_batch.py`
- `run_df_anchor_refinement_pipeline.py`
- `run_df_ld_window_cgs.py`
- `run_df_screening_anchor_cgs_h3_h14.py`
- `run_h5_df_cgs_gpu_slopes.py`
- `run_h9_ld5_df_cgs_ground_cache_slope_check.py`

## 診断・変換・可視化

| script | 用途 |
|---|---|
| `retire_archived_remote_branches.py prepare / tags / delete` | 固定済みの承認済みbranch整理計画を段階実行する。先に完全バックアップと保存tagを検証し、公開台帳とSHAが一致する旧branchだけを削除する。[引き継ぎ](../docs/research/branch_archive_20261102/HANDOFF.md)を読む |
| `check_validation_manifest.py` | manifestのschemaと参照パスを検査 |
| `check_partial_randomized_diagnostics.py` | partial-randomized計算の診断値を確認 |
| `analyze_df_ld_direction_trend.py` | $L_D$方向の傾向を解析 |
| `compare_df_error_budget_rules.py` | 誤差予算規則を比較 |
| `compare_df_error_budget_rules_from_summaries.py` | 保存済み要約から誤差予算規則を比較 |
| `cache_grouping_step_costs.py` | grouping step costをキャッシュ |
| `export_df_cgs_cost_table.py` | CGS cost表を書き出す |
| `plot_df_optimized_costs.py` | DF最適化costを可視化 |
| `plot_df_partial_grouping_cost_comparison.py` | partial grouping間のcostを比較 |
| `profile_h5_8th_cgs.py` | 旧高次PF経路のH5 profiling |

## 旧経路・現在の結論に直接使わないrunner

次は履歴・比較・再現のため残している。出力を現行結果として使う前に
[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)を確認する。

- `run_partial_randomized_pf.py`
- `run_df_screening_cost_minimization.py`
- `run_grouped_uwc_pf_qpe.py`
- `run_grouped_uwc_theta_sweep.py`
- `run_uwc_grouped_alpha_h2_h6.py`

特に旧DF screeningの入力表は基底状態不一致のため失効しており、UWC数値は参照raw artifactが
未登録である。runnerが存在することと、研究結果として利用可能であることを混同しない。

## runnerを追加するとき

新しい検証runnerは、可能なら次の同名セットを用意する。

```text
src/trotterlib/<validation_name>.py
scripts/run_<validation_name>.py
tests/test_<validation_name>.py
docs/<validation_name>.md
artifacts/<validation_name>/
```

結果を研究上の証拠へ加える場合は、`VALIDATION_STATUS.md`、研究概要、研究ノート、
`artifacts/validation_manifest.json`も更新する。

## 過去の研究SOURCEの保存

[G6 technical runner SOURCE](tracks/algorithm_codesign/g6_return_generator_audit.py)を元の内容で収録した。
[収録範囲と元snapshot](../docs/research/main_collection_20261103.md)を参照。元の一回実行・SOURCE/runtime binding・marker・STOPを継承し、今回の収録では実行していない。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](../docs/research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルSOURCE入口の収録（2026-11-03）

[AX-2A metadata-only writer](resource_applicability/prepare_track_a_ax2a.py)と[AX-2B metadata-only writer](resource_applicability/prepare_track_a_ax2b_preflight.py)の原版を追加した。[原本対応と未検証範囲](../docs/research/main_local_results_collection_20261103.md)を参照する。既存moduleの存在と構文だけを確認し、runner・研究test・科学計算は実行していない。


## 保存済み公開ファイルの固定commit照合

[`verify_archived_publication_bytes.py`](verify_archived_publication_bytes.py)は、E-R3準備版とB3 API preflightの
既存公開リストを使い、固定commitの文書・SOURCE・小さなmetadataのbytesを読み取り専用で確認する。
使い方・対象外ファイル・原検証との違いは[固定commit参照の案内](../docs/research/branch_reference_navigation_20261103.md#削除したbranchへ問い合わせない公開bytes確認)を参照する。
科学runnerや原結果の再解析は実行しない。

[Track D C7-1 q16](run_track_d_c7_1_q16.py)：[固定PLAN](../docs/research/track_d_c7_1_q16_20261107/PLAN.md)の16 fresh whole-path・raw PASS後最大64 P0 wrapperだけ。凍結C6aモデル/CX残差、既存高速化再利用、公開後mandatory STOP。通常testから分子実行しない。

[C7-2人工runner](run_track_d_c7_2_toy.py)／[結果projection](summarize_track_d_c7_2_toy.py)：[固定54case](../docs/research/track_d_c7_2_cx_structure_20261107/PLAN.md)だけを実行。実H4 input/イベントは読み込まず、旧研究runnerのmainを呼ばない。post/source/phase照合後STOP、q32実行なし。

[C7-3 q32明示runner](run_track_d_c7_3_q32.py)：[PLAN](../docs/research/track_d_c7_3_q32_20261107/PLAN.md)の8 fresh whole-path/cut16、raw PASS後最大32 P0。旧scheduler/compile/state-action再利用、構造feature/凍結モデル診断、通常testから分子実行しない。

[C7-3高速化completion](run_track_d_c7_3_accelerated.py)：旧raw8再利用、未着手P0/postを別taskで実行。旧SOURCE/STOPは不変、[承認済み切替PLAN](../docs/research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)と引継ぎhashを固定して明示起動する。

[C7-3停止済み結果の収録](collect_track_d_c7_3_q32.py)／[scalar報告書](summarize_track_d_c7_3.py)：SOURCE94e94のSTOP後だけ、旧sampling/raw35bfdと新compile/post94e94の来歴・160 payload member hashを整理。保存check・統計の読取りのみ、回路/統計の再計算なし。

## Track D O5 cut：人工専用入口

`run_track_d_cut_artificial.py --output <新directory>`：事前固定2ケースのdescriptor選択→既存builder/checker→P0 full wrapper→STOP。旧H4 loaderやsamplingは呼ばない。実行済み結果は[報告](../docs/research/track_d_cut_artificial_20261108/REPORT.md)。

[O5 q4移送preflight STOP](../docs/research/track_d_cut_h4_q4_20261108/REPORT.md)：C6a保存event/QPY原本が未取得のため、既存runner再実行・新runner追加は行っていない。原本回収が先。

## O5 q4保存原本の限定接続

`run_track_d_cut_h4_select.py`：hash一致したq4_sample00だけから不変ruleで一回decisionを価格参照前に保存。`run_track_d_cut_h4_reuse.py`：中央cut選択時だけ、旧QPY/PASS/全7価格を照合して新compileなしでSTOP。[完了済み結果](../docs/research/track_d_cut_h4_q4_completion_20261108/REPORT.md)。再実行しない。
[C7-4保存再解析runner](run_track_d_c7_4.py)／[結果projection](summarize_track_d_c7_4.py)：保存88 pathのみの探索fit/CV/bootstrapと固定人工6系列/短P0 2本。新H4 sampling/build/compileなし。既存runのmain/guardを変更しない。
