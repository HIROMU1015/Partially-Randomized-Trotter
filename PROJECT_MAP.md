# プロジェクト案内

[Full-H κ2 P-2較正設計の修正・PR-Pauli B1価格検証計画（設計のみ）](docs/research/a_full_h_p2_pauli_b1_price_design_20261114/REPORT.md)：現行P-2（q=1のみ）はcが識別不能（rank2）。取得前の識別・領域・分離検査`finite_price_design`（Cのfitはc628e9e8 blobを無変更で使用）。改訂案D7（q8独立＋paired q2、24 compile）／D6（22）。Pauli B1の2 pathはmodel不可（rank1）、B1-W（LD2配線）とB1-CAL（辞書あたり24）を提案。科学実行0、未採択。

[Full-H κ2 P-1 run01・run02（STOP）とrun02再検証（PASS）](docs/research/d_full_h_kappa2_p1_results_20261114/REPORT.md)：run01はreferences/欠落で科学計算0。run02（caller 72d4f4ff2）は`scalar_once`の偽陰性でSTOP（記録は保全）。[再検証](docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/REPORT.md)（SOURCE 8258b2200、証拠 6d188defd）は修正したscalar/event位相判定で23条件PASS。X/Y RZ 92,448の単一path direct価格をC c628e9e8 guardで受理し、Cへ引渡し（E[C]・P-2・Gには使わない）。

[Full-H κ2 S-1結果（承認済み13 mean）](docs/research/d_full_h_kappa2_s1_results_20261114/REPORT.md)／[SIGNALS.csv](docs/research/d_full_h_kappa2_s1_results_20261114/run01/SIGNALS.csv)／[B packet](docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/B_SIGNAL_INPUT.json)：固定caller 04bead115、全gate・logB包絡PASS、N/MSE/G未評価、mandatory STOP。

[Full-H κ2 S-1/P-1準備・Truncated/Pauli価格接続（D）](docs/research/d_kappa_s1_price_builders_20261114/REPORT.md)／[次の実行とSTOP](docs/research/d_kappa_s1_price_builders_20261114/NEXT_EXECUTION.md)／[HANDOFF](docs/research/d_kappa_s1_price_builders_20261114/HANDOFF.md)：固定SOURCE `04bead11`。S-1 caller（κ2 13 mean、1 ULP付きlogB包絡gate、R448信号の流用禁止）、P-1新ZZ配線（provider-root子process、parity・scalar・one-body・X/Y）、T3 B0I p7の決定論価格（既存Full PF native・18 wrapper候補）、A P0辞書付きPauli tail接続を整備。人工・metadata 30 PASS、実H4 mean/sampling/build/compile/価格0、mandatory STOP。

[Truncated PF T2：H_eff 42 cell](docs/research/truncated_pf_t2_results_20261114/REPORT.md)／[CELLS](docs/research/truncated_pf_t2_results_20261114/CELLS.csv)／[構成比較](docs/research/truncated_pf_t2_results_20261114/CONFIGURATIONS.csv)：承認済みT2を一回実行。p7/8/9×S2/Y4/M5の42 cell全PASS・UNKNOWN0、保存B0I 13round照合 最大2.7e−12。PF|retはFull PFと≤1.1e−7 Haの差、予算内最粗幅はFull PFと同じ（S2 16h/7・Y4 16h/3・M5 16h）、相殺依存0。T3候補p7×3 template（任意p8）。Track Cとは数値記録上整合・SOURCE同一性UNKNOWN。価格/N/G0、mandatory STOP。

[Truncated PF T1：切断固有値シフト](docs/research/truncated_pf_t1_results_20261114/REPORT.md)／[一覧](docs/research/truncated_pf_t1_results_20261114/SHIFTS.csv)：承認済みT1を一回実行。保存H4 DF10・B0I p0..9の保持H固有系10件、全PASS/UNKNOWN0。p7 +8.336e−6 Ha（ε_trotの1.05%）、p8 −2.75e−6、p9 −2.9e−8、p≤6は1.7e−2 Ha以上。保存6構成のc_tail/λ_Rはextractorとbit一致。T2優先はp7→p8→p9（42 cell案）。PF/H_eff/価格/N/G0、公開後mandatory STOP。

[Truncated PF誤差評価と有限最適化計画](docs/research/truncated_pf_optimization_plan_20261114/REPORT.md)／[段階計画](docs/research/truncated_pf_optimization_plan_20261114/SCIENCE_PLAN.md)：切断・保持H上PF・合計シフトの分解、符号相殺とoracle水準L1〜L3、p0..9×3公式×A同格子の99構成/154 cell metadata（既存ID再現）、T1〜T6の件数・SOURCE・STOP、PR公平比較の課題。toy 9/9 PASS、実H4科学計算・保存値再解析・compile0、公開後mandatory STOP。

[最新Full PF価格接続・S1比較guard](docs/research/full_pf_latest_price_guard_20261114/REPORT.md)：A71f04の78軸metadata、SOURCE付きguard、B直接consumerの取込み手順。

[Full PF3候補のcompiled価格証拠接続](docs/research/full_pf_candidate_price_evidence_20261114/REPORT.md)（2026-11-14）：A3ae055faのFull PF S2/.2・Y4/.4・M5/.8を39round/78軸でD e7 receiptへ照合。direct5round/10軸、M5 m4欠測、他33roundはcatalog外。元native/scalar/wrapper/P0のD証拠を再利用し、consumer lowering採用は未実施。追加4セル/最大8wrapperの未承認q検証案、early幅/長q未資格・必要精度UNKNOWN。価格再集計/予測/fit/compile/N/G0、production変更0、公開後mandatory STOP。

[B0I・Full-Hの全13round Raw signal取得](docs/research/truncated_fullh_signal_results_20261114/REPORT.md)（2026-11-14）：固定H4 DF10のB0I p7/S2/Δ.1とFull-H finite-K2/q=2^m/R448を各13round、一回評価。SOURCE5a172629d、26件すべて元数値gate PASS、Raw/corrected/reference/normalizationとB packetを保存。人工caller3件PASS、数値module不変、context復元1、旧signal受入/retry0。数値誤差・law距離・state誤差は未資格、MC/N/価格/G0。旧c546/6306計画・STOPを保全し公開後mandatory STOP。

[B0I・Full-Hの26round Raw signal有限scope](docs/research/truncated_fullh_signal_scope_20261114/REPORT.md)（2026-11-14）：c546原本からB0I p7/S2/.1とFull-H K2/q=2^m/R448の26roundを同identity/SOURCE/gateで分離。Full PF・m6受入依存・旧signal再資格0。新reference/抽出0、共有saved context/cacheを再利用。metadata26件照合PASS、実input/数値計算/再解析/旧test再実施0。数値実行は明示許可待ちSTOP。

[平均compiled価格の証拠coverage・有限検証案](docs/research/rpe_price_evidence_validation_20261113/REPORT.md)（2026-11-13）：PR-LD1/LD2の52軸＋旧Full PF6 recipeの156軸を静的coverage化。PF direct receipt16軸、LD1 E[S]26軸/P2有限cohort4軸の所在と未受入bridgeを区別。D corrected scalarの人工1＋旧13＝14 PASS、production変更0。条件付きPR新paired6セル等を提案し、BのN/精度とA選択待ち。保存S/C再集計・fit/compile/G0、公開後STOP。

[全方式の平均compiled価格モデル技術準備](docs/research/rpe_mean_price_model_preparation_20261113/REPORT.md)（2026-11-13）：外部係数の加法式・境界covers・既存S＋単一C−S・4方式の受動B接続を追加。人工新11＋既存27＝38 PASS。全モデル未資格、実H4/H6の新S/C・保存再解析・sampling/compile/fit/G0。C1係数/26 E[S]/D requestと既存STOPを保全、公開照合後STOP。

[D最新版scalar bindingの技術接続](docs/research/d_scalar_binding_connection_20261113/REPORT.md)（2026-11-13）：C価格bindingをD748へ接続し、expected_scalarの生成/再構成・schedule anchor・resolved pathのbinary64完全一致を追加。新5＋既存20＝25人工/回帰PASS、省略時の旧request互換を保持。実H4/preparation/sampling/compile/保存科学再解析/E[C]・G0、D T2のscalar原因はUNKNOWNのまま。ベースライン科学計画の未承認STOPを保全、公開照合後STOP。

[ベースライン3系列・全RPE signal有限実行計画](docs/research/baseline_finite_signal_plan_20261113/REPORT.md)（2026-11-13）：Full PF S2/.1/Full10、B0I p7/S2/.1、Full-H K2/q=2^m/R448の3 canonical schedule・39roundを固定。提案は保存Full PF m6の受入再資格1件→PASS時だけ新38 mean。無条件再利用0、X/Yは倍計数しない。既存13 reference pair/保存scalarを利用し、新reference/抽出0。metadataのみ、実input読込/再資格/新signal/旧test再実施/N・価格・G0、明示科学許可待ちSTOP。

[D価格adapterとRPE pipelineの統合](docs/research/d_price_pipeline_integration_20261113/REPORT.md)（2026-11-13）：公開D candidate_price_adapterをC price_requestsへ接続。A固定LD1/LD2の13round metadataとD path→pending/外部mean→B会計を照合し、新8＋既存36＝44 PASS。path/S/finite cohortを適格E[C]へ昇格せず、軸・SOURCE/law/preparation/compiler/loweringを保持。実H4/preparation/sampling/compile/保存再解析/実N・G0、D物理資格と平均/N証拠は後続。

[RPE総コストpipelineの人工統合](docs/research/rpe_optimization_integration_20261113/REPORT.md)（2026-11-13）：A PR・C baselineの全round signalをB shot配分/cost/selectへ接続。D向け未評価requestとphysical/lowering bindingを整備。人工新10＋B metadata2＋C/M5回帰24＝36 PASS、4方式の人工全13round・shot最小/G最小・点/区間・未資格/欠測を区別。実H4/saved再解析/実N・価格・G0、科学N資格producerとD価格は未取得。

[ベースラインsignal pipelineの技術準備](docs/research/baseline_signal_pipeline_preparation_20261113/REPORT.md)（2026-11-13）：ベースライン全round signal経路を既存B/D/数値部品から接続。H4 DF10のFull PF・B0/B0I・Full-H pure RTE、S2/Y4/M5・prefix・明示q/R・有限Kを扱う。人工新12＋既存回帰12＝24 PASS、数値失敗値/欠測/未資格を保持。実H4・保存再解析・残screen・sampling/compile/N/RMSE/G0。

[全RPE比較ベースラインの最適化設計](docs/research/rpe_baseline_optimization_design_20261113/REPORT.md)（2026-11-13）：Full PF(S2/Y4/M5)、B0/B0I切断PF、one-body込みFull-H pure RTEを全RPEのΣN E[C]で探索する計画。旧11案はseed、局所screen/構造proxyと根拠付きrisk/cost除外を区別。既存APIを再利用し新framework0、人工回帰20 PASS。実H4計算/保存再解析/候補最適化/compile/N/G0、正式選定なし、公開照合後STOP。

[残差構造特徴量の期待値評価方法](docs/research/structural_feature_expectation_design_20261113/REPORT.md)（2026-11-13）：D P2のfeature定義を照合し、tail Z/ZZ/order/product数とbasis group/shareの解析式・小kernelを整備。新9＋既存19＝28人工/技術test PASS。native/compiled相殺や非線形joint量とは区別し、Dの直接残差法を置換しない。実H4確率解析/F0/K0/K1/E[S]/sampling/compile/fit/N/G0、旧26値とSTOP保全。

[finite-K sampler law整合性](docs/research/finite_k_sampler_law_alignment_20261113/REPORT.md)（2026-11-13）：ideal finite-K/source lawとPCG64/CDFの接続調査。生成構造・rte.py blob対応を確認し、既存53-bit helperを再利用するTV分解/条件付き上界を追加。人工12＋既存toy5 PASS。実H4のlaw距離・PRNG joint-law・価格systematicはUNKNOWN/null、既存26 E[S]不変、新科学計算0。公開照合後mandatory STOP。

[H4 PR全13roundの平均構造RZ結果](docs/research/h4_structural_schedule_results_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2・同preparationで残11lawのK0全PASS、新22 E[S]を取得。m0/m6四値を再実行せずcopyし全13round/26軸が同canonicalへ接続。SOURCE de2427d4a、99 wire-law検査、最大mass defect1.714e−12（tol1e−10）。元Sのideal有限law平均でE[C]/Gはnull。新sampling/full-path S/build/compile/fit/N/G0、計算再起動0、旧STOP保全。公開照合後mandatory STOP。

[H4構造価格・全13round経路の技術準備](docs/research/h4_structural_schedule_preparation_20261113/REPORT.md)（2026-11-13）：固定B3216/同canonicalの13round metadataを接続し、m0/m6四値は83d219cc6からcopy専用。残11lawのK0→PASS時K1 batchを既存core/MeanKernelで準備、new8＋回帰18＝26技術test PASS。R255等を保持し、実T/terminal入力・同γ16外向き二乗法で巨大qR展開を回避。実新K0/E[S]/sampling/compile/fit/N/G0。最大22新Sの一括計画を公開、残差はD担当、公開照合後STOP。

[H4 kernel run02：K0 PASS・K1平均構造RZ四値](docs/research/h4_structural_kernel_20261113/run02/REPORT.md)（2026-11-13）：SOURCE19b95d8b3で同H4 DF10/S2/p1/δ.1/K2・1prep/2lawを一回実行。K0全9wire PASS、kernel/状態辞書を保存後、m0/m6×XYの実E[S]四値を取得（各mでX=Y、1743.6682557187337/1764659.275564426）。数値区間・元Accounting parityを保存。sampling/full-path S/build/compile/fit/N/G0、E[C]はnull・C1未資格のまま。旧run01/STOP保全、公開照合後mandatory STOP。

[H4 K0入力STOPと技術修正版](docs/research/h4_structural_kernel_20261113/REPORT.md)（2026-11-13）：K0→PASS時K1 runnerと数値契約を実装、新8（人工7/固定metadata1）＋旧10技術test PASS。SOURCE0f7d65d5fはB configuration locator誤りで物理K0前にUNKNOWN STOP。実descriptor/kernel/parity/平均S四値は未実施・null。参照chainを修正版19b95d8b3でmetadata照合し、科学再実行なし。指示§4 STOP保全、run02案のみ公開。新sampling/full-path S/compile/fit/N/G0。

[H4 PRの物理閉状態診断](docs/research/h4_pr_physical_closure_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2の1prep・R2/R128を診断。324非identity/9basis、reset/phase/head境界からS observableの有限wire-marginal状態の十分条件を確認（tail enclosure80〜120）。物理kernel/parity/実平均S/Cは未取得、MODEL false。最終10技術test PASS。初回seed設定STOPを保全し同scope修正、成功SOURCE eeccdf18e・縮約解析2b3b0ee7f。新sampling/full-path S/build/compile/fit/N/G0、旧S0a等を保全、公開照合後STOP。

[固定H4 S0の平均compiled価格検証計画](docs/research/h4_s0_mean_price_plan_20261113/REPORT.md)（2026-11-13）：SOURCE bdd2b9356。固定A S0のexactκ/M12・H4 m0(q1/R2)/m6(q64/R128)を公開B V2へmetadata接続。新7技術test PASS、13round価格null。両候補の一致direct/modelは公開索引に未確認。閉状態診断1prep/2law・compile0を第一案、別案は有限4path/最大8wrapper。新S/価格再解析/sampling/compile/fit/N/G0、C1未資格・S0a履歴を保全し、公開照合後STOP。

[B V2価格接続と長q平均価格の次段階設計](docs/research/bv2_mean_price_validation_design_20261113/REPORT.md)（2026-11-13）：SOURCE58382c54c。固定B6a9 V2 MeanPriceLookupを再利用するC受動入口を追加、新12人工/metadata結合test PASS。値ありrecordに平均compiled quantity明示を要求し、S0a/旧ID付替え/資格昇格を拒否。旧118候補/236軸と4方式の資産/欠測を静的整理。実平均S/C・物理閉包は未取得、選択B domainのclosure診断等4候補を未承認計画化。新S/C再解析/sampling/compile/fit/N/G0、旧結果/STOP保全、公開照合後STOP。

[S0a run02：実保存8件の構造S照合](docs/research/s0a_saved_structure_results_20261113/run02/REPORT.md)（2026-11-13）：修正版SOURCE9516351e6で保存normal index0の4path×XY8 Sを一回評価し、元C1保存Sと8/8整数完全一致。各X/Y SはH4 q8/R8=21664、q16/R8=46559、H6 q1/R32=158669、q2/R32=321151。input/preparation/law/phase/allocation/digest検査を通過、元roleとrun01失敗STOPを保全。平均S/C・物理閉包・モデル資格は未取得、qualified=false。新sampling/compile/fit/N/G0、公開照合後STOP。

[S0a run01の設定エラーと技術修正](docs/research/s0a_saved_structure_results_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65でrun01を1回起動、context exec scopeのast渡し漏れでS開始前に失敗STOP。新S0/8、保存S照合0、全4path未評価。修正版SOURCE9516351e6と人工14件PASSを公開、run02案はmetadata準備だけで未実行。旧STOP/role/モデル資格を保全、修正版再試行はユーザー判断、公開照合後mandatory STOP。

[S0a保存構造価格の実行準備](docs/research/s0a_saved_structure_preparation_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65。固定C1の保存4path/XY8計数に向けreader・law/identity検査・S専用runnerを準備。新13＋旧9＝22技術test PASS。raw/inputは所在のみ確認、実S・保存S比較は未実行。元係数/資格/平均provider不変、内容hashと実contextは実行時検査。初回人工importのlowering副作用と修正をREPORTへ記録。明示科学許可待ち、公開照合後STOP。

[C1状態閉包とStreaming計数の技術検証](docs/research/c1_physical_state_stream_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE c1135fdb9。C1のpending basis/Z・tail phase・terminalをSOURCEで調査し、元計数wordへforward-only tailを渡す最小consumerを追加。新9＋旧3＋sampler toy1＝13技術test PASS。単純縮約の人工反例は確認、実H4/H6の閉包・event生成なしの平均S/compiled価格は未確認。consumerは全event処理時間が残り、producer全接続は未実装。S0a保存4path/XY8件は未実行案、実科学計算・再解析・compile・fit・N/G0、公開照合後mandatory STOP。

[長q・大Rの平均compiled価格の技術準備](docs/research/mean_compiled_price_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE09489eedb。C1 Accounting/旧局所期待formを再利用し、closed-state期待報酬の二乗合成と受動c_providerを実装。新15＋旧pure3＝18技術test PASS、人工全列挙と一致。固定状態ならqR展開を避けられるが、実C1の累積Z/phase閉包・平均S・compiler残差は未資格で実価格missing。C−S残差のみを扱い二重計上を拒否。κ scheduleはB外部contextのcopy、M/N最適化なし。旧40/16案は非実行、将来S0保存再解析等は別許可。実科学計算/compile/価格fit予測/N/G0、公開照合後STOP。

[A P1対応・長q/大R価格検証の準備](docs/research/r_price_longq_largeR_preparation_20261111/REPORT.md)（2026-11-11）：公開A71ba49bbの12候補・同policy条件付きNを無変更copy。C1価格との重複はH6 m4/R16,R32の校正2点。第一案はH4 m6/H6 m8のbaseline＋R128、4セル・最大40 wrapper、原因分離R-onlyと短時間各16は別の任意案。metadata SOURCE f0daeb96b、8技術test PASS。構造Sのstream化/局所境界法を比較し、m12やR256を自動compile/外挿しない。新科学計算・価格予測・再解析・N/G0、科学実行未承認、公開照合後STOP。

[R価格Phase 1実行結果](docs/research/r_price_phase1_results_20261111/REPORT.md)（2026-11-11）：Linear H4/H6・1Å・STO-3G・保存DF10/19、H4 S2/δ.1/p1/K2とH6 m5_best/δ1.6/p7/K2。SOURCE1a87e5331で通常64＋別rare16 compile、保存C/S59 path再利用。係数を7eeabf0aで検証取得前に凍結し再fitなし。独立n8の相対RMSEはH4 q16/R8で1.039841%、H6 q2/R32で0.229959%。有限cohort観測、採用閾値未設定・model未採用、全law/長q-R/総CI/全Gは未資格。新signal/N/G0、公開照合後mandatory STOP。以下の準備記録・過去STOPは履歴として保持。

[R価格Phase 1の実行準備](docs/research/r_price_phase1_preparation_20261111/REPORT.md)（2026-11-11）：保存C/S接続・paired3係数fit・Git凍結後validation・別unit rare診断の専用runnerを準備。公開A/B metadata整合、新14＋旧R回帰9＝23人工/mock PASS。固定SOURCE/PLAN、最大64＋16 wrapperの科学scopeは未実行。実分子sampling/S再計数/compile/fit/価格予測/再解析/N/G0、公開照合後STOP。

[R可変compiled価格の調査・設計](docs/research/r_variable_compiled_price_design_20261111/REPORT.md)（2026-11-11）：H4 S2/δ.1/p1/K2・H6 m5_best/δ1.6/p7/K2の固定Rモデルを保全し、構造価格＋少数compiled残差補正を提案。R binding/保存lookup adapter、人工/mock9件PASS。新R Cはnull、実科学計算・価格取得/予測/fit/G0。最小新q×R/希少event計画は未承認・未実行、公開照合後STOP。

[H6 provider統合・H4/H6全13round N/C coverage](docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md)（2026-11-11）：固定B cb2 APIへ保存lookup factoryを追加。C provider/テストをbyte一致再利用、24技術test PASS。H4公開89c91c69の整数検証済みNとH6公開f4c0bdaのNを静的転記し52行条件付きPASS。直接CはH4 5候補/H6 8候補、PR全law/長q/共有CI/全RPE採用は未資格。新科学計算・実N×C/G0、B実行領域へ非介入、公開照合後STOP。

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
