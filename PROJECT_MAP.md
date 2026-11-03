# プロジェクト案内

研究C・B2：[reference数値不確かさの方法・誤差伝播・統合interface](docs/research/research_c_reference_numerical_uncertainty_readiness_20261103.md)。既存Arb/CARを再利用し人工検証。実H4契約NOT_READY、Stage C未実行、旧SOURCE/STOP保全。

研究C：[B1/B2統合schema・精度scope比較（B2公開未観測）](docs/research/research_c_b1_b2_accuracy_scope_integration_20261103.md)。開始時のB2照会は一致0、依存判定は保留。誤差partition・有限/確率的案・既存accuracy gateを人工284 PASS（旧2件除外）で接続した。新H4科学/Stage C/重要レビュー0、未承認設計は非実行、mandatory STOP。

研究C・B1：[Whole-P0一様synthesis法の技術具体化](docs/research/research_c_p0_uniform_method_readiness_20261103.md)。既存C2-A/ZSXX composerを現r32へ接続、人工/回帰252 PASS（旧2件は理由付き除外）。構造上限は具体化したが全domain再合成・phase等は未資格。実H4契約は非実行/calls未確定、B2未統合、新科学0、mandatory STOP。

研究C：[IID-grid Stage A・38-call実行結果](docs/research/research_c_iid_grid_stage_a_execution_20261103.md)。固定H4/1Å・新PR/CDF8/LD2/S2/s1/r32/K0/h=.1/m0..12のreference bridge完了。38 COMPLETE/FAIL・retry・resume0、categorical TV約3.43e-15。Whole-P0/数値reference不確かさ/実PCG64/actual CAは未資格、Stage B/C未実行、mandatory STOP。

研究C：[IID finite-bit条件付き全law/RPE評価の具体化](docs/research/research_c_iid_grid_all_law_readiness_20261103.md)。現PR/r32/s1のStage A reference bridgeは未承認38-callで技術GO、人工182 PASS、passive PASS。Whole-P0未資格でStage B/CはNOT_READY。実PCG64/actual CAは未認定、新実H4計算0、mandatory STOP。

研究C：[全law synthesis・implemented-law signal・CA資格の技術準備](docs/research/research_c_all_law_signal_ca_readiness_20261103.md)。最小adapter・人工/回帰152 PASS・passive確認PASS。Whole-P0一様coverageと実stream/reference対応が未成立のため契約はUNAPPROVED_DESIGN_NOT_EXECUTABLE、実H4新科学call0、mandatory STOP。

## 研究C：新PR対保存XDF8/CDF8の直接統計比較（2026-11-03）

固定H4・直鎖1.00 Å/STO-3G、sector36/rank8、PR元順LD2・partial S2/s1/r32/K0・δ=h=.1/m0..12、XDF/CDF全frame S2/s1、own-DF ground oracle・状態準備除外、P0/fenced serial/shell込み。保存PR native32の事後的探索的一標本t比較を固定7/7callで実行。PR/XDF点比.940365699780、近似95%区間[.912991218406,.967740181154]、一側p=5.26154e-5（α=.025）、精度/診断PASS、POSTHOC_EXPLORATORY_NOMINAL_MODEL_PR_LOWER。CDFは同一区間から導く副比較で追加検定0。Actual CA・全law・ideal-law価格移送・confirmatory資格は未認定。

[7-call実行結果・固定SOURCE・限界](docs/research/research_c_pr_xdf_cdf_statistical_comparison_execution_20261103.md)。新sampling/build/compile0、旧SOURCE/raw/STOP保全、mandatory STOP。

研究Cの保存PR native32対XDF8/CDF8：[直接統計比較の技術準備](docs/research/research_c_pr_xdf_cdf_statistical_comparison_preparation_20261103.md)。事後的・探索的一標本t、未承認7-call、人工66 PASS、passive preflight PASS。実H4の新CI/p値0、mandatory STOP。

## 研究C：PR対weighted L1・共通r32の条件付き資源比較（2026-11-03）

[1,231-call結果・全26round・各native32・同J/U強D・資格の限界](docs/research/research_c_near_practical_r32_downstream_execution_20261103.md)。固定H4/1Å/STO-3G・number CDF8/sector36、元順LD2・S2/s1/r32/K0・δ=h=.1/m0..12、own-DF ground/状態準備除外、P0/fenced serial。PR/ENTRY総RZ点比.49418、名目Welch95%モデルでPR低費用。全law/actual CAは未資格、旧SOURCE/raw/STOP保全、公開照合後mandatory STOP。

## 研究C：比較II・共通r32下流評価（2026-11-03）

[最小接続・人工80 PASS・1,231-call固定条件](docs/research/research_c_near_practical_r32_downstream_preparation_20261103.md)。`c_near_practical_r32.py` / `run_research_c_near_practical_r32.py`。PR/ENTRYの保存endpoint、各native32、各同J/U強Dを接続。既定runnerはpassive preflight。

## 研究C：近接目的比較IIの実行結果（2026-11-03）

[645-call結果・PR/L1終端・制約](docs/research/research_c_near_objective_practical_comparison_execution_20261103.md)。両arm16更新/DF予算PASS、J MET/U NOT_MET。Exact λはPR .500044243591Ha/ENTRY .941312847393Ha。下流資源は未評価、mandatory STOP。

研究Cの近接weighted L1比較II：[技術準備・未承認1,629-call Stage O](docs/research/research_c_near_objective_practical_comparison_preparation_20261103.md)。`c_near_stable.py`/`c_near_practical.py`、人工tests、SOURCE、passive runnerを接続。新実H4計算0。

## 研究C：11月3日成果統合・nominal価格比較（2026-11-03）

[新版Evidence Map・旧24 Claimとの対応・8追加項目](docs/research/research_c_evidence_map_updated_20261103.md)。固定H4・直鎖1.00 Å/STO-3G、number CDF8・sector36、update064/r32（元順LD2、partial S2/s1、δ=h=.1、K0、m0..12、own-DF ground oracle・状態準備除外、P0/fenced serial/shell込み）の条件付き総RZは、独立256件の価格校正で同一J/U強決定論S2より約4.48%低い点比を得た。Nominal95%近似t比区間[.946594634483,.963772600308]、登録1%精度PASS、`NOMINAL_MODEL_PR_LOWER`。全law誤差・実装lawのsignal・actual CA・一般性は未資格。

update005/r32の独立640件は0.5%精度目標未達・`UNRESOLVED_PRECISION_TARGET`。追加64 U更新はDF予算を維持したがU一次未達。A〜Eの科学SOURCE・結果commitと2候補を分離し、旧資料・旧判定を保全した技術的統合。新科学計算・重要科学レビュー・新規性/論文化判断は0。以下の11月2日以前の記述は各時点の条件付き履歴として読む。

## 研究C：Claimと科学的貢献候補の統合（2026-11-02）

[最新Evidence Map](docs/research/research_c_evidence_map_updated_20261102.md) ／ [科学的貢献候補・論文構成案](docs/research/research_c_contribution_positioning_20261102.md) ／ [24 Claimと貢献候補の対応・固定証拠照合](docs/research/research_c_claim_contribution_alignment_20261102.md)。A/B原本を保持した文書統合であり、新科学計算・証拠資格の変更・新規性や論文化の正式判断は行っていない。

## 研究C：native価格改善機構の統合技術判断（2026-11-02）

[統合報告・Q1〜Q10・再検討条件](docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md)。**TECHNICAL_NO_GO_CURRENT_EVIDENCE**。同一新J/Uの強D比較（38b8d0df）と別branchのPR寄与分析（2639221a）を接続し、旧surrogate・第三DF転移・P4 support・native融合の負結果を整理した。λ以外のbasis確率・U構造情報は存在するが、新selectedのcomplete P0価格を下げる操作は未特定。新目的・runner・科学契約は追加せず、保存scalarの損益分岐照合以外の新規科学計算0。これは技術的な実験準備判断で、研究C全体の終了・一般的不可能性を採択しない。旧SOURCE/raw/STOP保全、公開照合後mandatory STOP。

## 研究C：同じ新J/Uでも主RZは強決定論が低費用（2026-11-02）

[156-call結果・全13round・同一J/U比較と限界](docs/research/research_c_same_ju_strong_d_execution_20261102.md)。固定SOURCE5f27539ae/156-call契約を無変更でfresh run01へ一度実行し156/156 COMPLETE・correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、number CDF8/sector36、統合update005の同じJ/U、元g補正κ/scalar/frame順、own-DF ground oracle/準備除外。全frame ordered S2/s1/δ=h=.1/m0..12、保存DF予算PASS再利用・新DF norm/fit/optimizer/PR再計算0。9因子CAR/sector・ground/alias・full512 raw/compiled/actual-adjoint/X-Yと13roundがPASS。P0 kernel RZ11026、shots13728、条件付き総RZ105532782860。保存新PR idealは207919150188、同じ新J/UのPR/D比1.97018542062（CX depth比.724994660275で副指標tradeoffあり）。旧/新のD価格6指標は一致、D総RZは3.64%増、新PR自身は既報11.19%減。Numerical D/両ideal反実laneは同じ整数費用だが、厳密CA・PR全law誤差/support/母平均優位・目的だけの因果効果は未資格。旧SOURCE/raw/STOP保全、公開照合後mandatory STOP。

## 研究C：新selected J/Uの強決定論公平比較・技術準備（2026-11-02）

[再利用・134人工/回帰PASS・未承認156-call契約](docs/research/research_c_same_ju_strong_d_preparation_20261102.md)。`c_same_ju_strong_d.py`／`run_research_c_same_ju_strong_d.py`。同じH4/CDF8の全frame S2/s1、保存sector/ground/DF誤差を再利用し、1 kernel/P0と13round、数値合成lane＋共通ideal反実条件を準備。実H4追加decode/計算0、旧SOURCE/raw/STOP保全。別の科学許可までmandatory STOP。

## 研究C：物理適格5 U更新・新候補の条件付きPR資源評価（2026-11-02）

[統合結果・Q1〜Q10・SOURCE/calls/未資格事項](docs/research/research_c_pr_integrated_execution_20261102.md)。固定SOURCE980b514c8/1759-call統合契約を無変更で一度実行し543/543 COMPLETE・correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、number CDF8/元順LD2・γ=.1、sector36。旧19-call受理点の物理DF予算を初めて評価しPASS、新trajectoryは5受理U更新/全受理点DF予算PASS。F .01079246794→.006617509795、exact λ .7258193873→.5683498547、J∞9.59327e-8 MET/U₂ .6589358143 NOT_METで同時一次未達。次の登録3qは全数値適格だが降下せずFINITE_STEP_GRID_NO_ACCEPT。選定update005を新s1/r128/K0のPF/RTE13round・新native32/128QPYへ接続し各correctness検査PASS。Ideal反実laneの条件付き総RZ 207919150188、旧PR比0.88809634、保存強D benchmark比2.0419077。実CA/全law合成・実装law/support・母平均優位はnull、強Dは旧J/U/異なる誤差資格、RC2は別有限endpointで公平な収束比較ではない。旧SOURCE/raw2579件/STOPを保全、新raw500件、公開照合後mandatory STOP。

## 研究C：物理DF・複数U更新・条件付きPR統合実験（2026-11-02）

[固定条件・312人工/回帰PASS・統合1759-call契約](docs/research/research_c_pr_integrated_preparation_20261102.md)。新 `c_pr_integrated*`／`run_research_c_pr_integrated.py`／人工tests。H4/CDF8/LD2・γ.1、保存19-call受理点の物理評価を先行し、最大64 recentered更新、各3q、毎受理点DF予算。適格選定点だけを新PF/RTE/RPE・条件付きnative32へ接続。旧SOURCE/raw/STOP保全、統合完了/停止条件でmandatory STOP。

## 研究C：小stepで1回の資格付きU更新を受理（2026-11-02）

[19-call結果・独立終端・未収束の区別](docs/research/research_c_pr_small_step_execution_20261102.md)。固定SOURCEa426d62f2/19-call契約を無変更で一度実行し19/19 COMPLETE・correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、number CDF8/元順LD2・γ=.1、元qualified J/U開始。α=F0/|s|/8=.000175615780642の単一trialを、全admission・値尺度Armijo・独立終端確認後にU更新として1回受理。F .01226764443→.01079246794、独立J∞1.22271e-7はMET、partial U₂2.95498→2.66969はNOT_METで同時一次未達。Inner6/refinement4、raw precision loss1、新ffsim fit/外部outer optimizer0。真の勾配精度・厳密降下は未資格、DF物理予算/PR性能/下流は未評価。旧raw2554/SOURCE/STOP保全、新raw25、公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：小step・最大1回U更新の技術準備（2026-11-02）

[最小接続・297人工/回帰PASS・未承認19-call契約](docs/research/research_c_pr_small_step_preparation_20261102.md)。`c_pr_small_step.py`／専用pilot・runner・人工testsを追加。元H4/1Å/STO-3G、CDF8/LD2・γ=.1の初期点からα=F0/|s|/8の1trial、値尺度を含むArmijo＋独立終端state後のみ受理。旧inner/資格/閾値を維持、実H4追加計算0。旧SOURCE/raw/49-call STOP保全、別実行許可までmandatory STOP。

## 研究C：PR局所対称診断・3尺度でF低下を観測（2026-11-02）

[49-call結果・全7点・FDと未資格事項](docs/research/research_c_pr_local_direction_execution_20261102.md)。固定SOURCEe06913a23/49-call契約を無変更で一度実行し49/49 COMPLETE・correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、number CDF8/元順LD2・γ=.1、元qualified J/U開始。初期＋6側点すべて数値資格PASS、h=1.75616e-4/4.39039e-5/1.09760e-5で正方向Fがすべて低下。最小hの中央FD−候補s=-8.10735e-6、工程的FD尺度1.01876e-5。真のgradient/inner誤差/truncation/降下保証は未資格null。Inner21/refinement14、raw precision loss3、外側最適化/U受理0、J/U同時一次未達。DF物理予算/PR性能/下流は未評価。旧raw2495/SOURCE/STOP保全、新raw59、公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：局所step幅・reduced勾配精度の技術切り分け（2026-11-02）

[保存scalar・人工反例・局所FD・269人工/回帰PASS](docs/research/research_c_pr_local_step_accuracy_preparation_20261102.md)／[未承認49-call診断](docs/research/research_c_pr_local_direction_contract_unapproved_20261102.json)。旧最小αでも一次予測減少/F0=1.90066。Step不足とinner由来のenvelope微分誤差、F比較尺度を人工で分離。新案は元初期候補方向の3段階±probe、各点でJを解き直し、U受理/optimizer0。FD一致は真の勾配資格や誤差enclosureにしない。実H4追加query/NPZ数値再解析0、旧raw2495/SOURCE/60-call STOP保全、別許可までmandatory STOP。以下は各時点の履歴。

## 研究C：Guarded Armijo・8 trialの範囲で受理なし（2026-11-02）

[60/60-call結果・全8 trial・pair診断](docs/research/research_c_pr_guarded_step_execution_20261102.md)。固定SOURCEd03351ea0/60-call契約を無変更で一度実行し60 COMPLETE・correctness FAIL/retry/resume0。保存H4/直鎖1Å/STO-3G、number CDF8/元順LD2・γ=.1、元qualified J/U開始。8 trials中stability未資格2、数値資格PASSだがArmijo不成立6、受理U更新0。最小α=.002670284759でもF .0205435180>初期 .0122676444。27 inner/18 refinement/9点、最終J/rank-nullspaceは全点PASS。Raw precision loss11/maxiter1を保全。範囲外に降下stepがない証明ではなく、DF予算/PR性能は未評価。旧raw2423/SOURCE/STOPを保全、新raw72。公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：PR stability切り分け・qualified-only stepの技術準備（2026-11-02）

[保存JSON調査・人工機構・政策A/B・Q1〜Q10](docs/research/research_c_pr_reduced_stability_investigation_20261102.md)／[未承認60-call案](docs/research/research_c_pr_guarded_step_contract_unapproved_20261102.json)。実H4の7.3109e-6の原因は未確定。新25＋既存226＝251人工/回帰PASS。技術推奨Bはinner/1e-6資格を維持し、未資格gradientを使わない1回のqualified-only Armijo step（最大8 trials/9点）。Wolfe/BFGSの継続・厳密Φ保証ではなく正式採択は未判断。実H4 decode/新科学call0、旧raw2423/SOURCE/18-call STOPを保全、別許可までmandatory STOP。以下は各時点の履歴。

## 研究C：PR-only reduced・初回trialのstability未資格STOP（2026-11-02）

[18/54-call結果・資格とcorrectnessの区別](docs/research/research_c_pr_reduced_limited_execution_20261102.md)。固定SOURCEb2e5ab4d5/54-call契約を無変更で一度実行し18 COMPLETE・correctness FAIL/retry/resume0。保存H4/直鎖1Å/STO-3G、number CDF8/元順LD2・γ=.1、元qualified J/U開始。初期点はadmission PASS、最初の未受理trialはJ/rank-nullspace PASSだがlatest-pair U勾配差7.31089e-6>1e-6でREDUCED_GRADIENT_UNQUALIFIED。受理U更新0、distinct U点2/inner6/refinement4/outer1。Raw precision loss2件を保持。旧trial2未到達、DF予算/PR性能/qualified比較null、旧raw2401/SOURCE/STOP保全、新raw22。公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：PR-only reduced限定検証・54-call技術準備（2026-11-02）

[接続・226人工/回帰PASS・未資格の区別](docs/research/research_c_pr_reduced_limited_preparation_20261102.md)／[未承認54-call契約](docs/research/research_c_pr_reduced_limited_contract_unapproved_20261102.json)。既存opt-in PR inner＋2 refinement/独立state/rank-nullspaceを再利用し、raw flag・J・stability・admissionを別記する限定entryを追加。次案は保存H4/直鎖1Å/STO-3G、number CDF8/元順LD2・γ=.1、original qualified J/U開始、受理U更新≤2/distinct U点≤8。実H4 decode/新科学call0、旧raw2401/SOURCE/STOP不変。別明示許可・fresh runまでmandatory STOP。以下は各時点の履歴。

## 研究C：PR固定U inner1回・J基準PASS、raw precision loss（2026-11-02）

[8-call結果・独立J・raw状態の区別](docs/research/research_c_pr_inner_point_execution_20261102.md)。固定SOURCEab5d80bd6/8-call契約を無変更で実行し8/8 COMPLETE・correctness FAIL/retry/resume0。保存H4/直鎖1Å/STO-3G、number CDF8/元順LD2・γ=.1、保存trial2 U固定・元J初期のNewton-CGを1回実行。終端chain差1.05214e-17、独立J∞9.83543e-6＋元marginでMET。一方raw success=false/status2 precision loss/nit708、partial U₂1204.251で同時一次未達。INNER_CORRECTNESS_PASS_J_METは工程的数値基準で、正常optimizer収束/DF予算/PR性能/reduced資格ではない。Refinement/U更新/outer/RC2継続0、旧raw2388/SOURCE/STOP保全、新raw13を保存し公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：PR inner同一配列評価器のopt-in接続・8-call準備（2026-11-02）

[接続・人工256 unique PASS・制約](docs/research/research_c_pr_inner_point_preparation_20261102.md)／[未承認8-call契約](docs/research/research_c_pr_inner_point_contract_unapproved_20261102.json)。新 `c_pr_inner_point_solver.py` は既存SciPy Newton-CGのvg/hess・raw pre-step検査を同一decoded J評価器へ接続し、明示PRReducedOracleの初回/2refinement呼出しも接続。旧default/sourceへのpatchなし。次案は保存trial2 U固定・元J初期のinner1回だけ、outer/refinement/RC2/DF normは対象外。Raw flagとJ一次資格を分離し、人工precision-loss例も保全。SOURCE149/input6のpassiveのみで実分子decode/新科学call0。旧raw2388/SOURCE/STOP・132科学recordsを保全し、別許可までmandatory STOP。以下は各時点の履歴。

## 研究C：保存PR trial2・旧FAIL再現／新数値整合PASS（2026-11-02）

[10-call結果・点identity・数値的限界](docs/research/research_c_pr_coefficient_point_execution_20261102.md)。固定SOURCE565de95fc/10-call契約を無変更で実行し10/10 COMPLETE・correctness FAIL/retry/resume0。保存H4/直鎖1Å/STO-3G、number CDF8/元順LD2・γ=.1・同元qualified x0から保存PR trial2の点をdry replayで復元し、保存η/U/W/x identity PASS。旧経路の差2.37552e-7で不一致を再現、新同一配列・拡張精度経路は差9.18462e-18で元allowance内、objective差7.10543e-15もPASS。単一保存点の数値整合のみで厳密gradient上界/optimizer収束/DF予算/PR性能は未認定。新J solve/fit/最適化trajectory0、RC2継続・paired再実行なし。旧raw2378/SOURCE/旧480-call STOPを保全、新raw10を保存し公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：PR inner勾配整合の人工切分け・10-call準備（2026-11-02）

[人工再現・最小修正候補・239 unique PASS・RC2ログ診断](docs/research/research_c_pr_gradient_consistency_investigation_20261102.md)／[未承認10-call契約](docs/research/research_c_pr_coefficient_point_contract_unapproved_20261102.json)。`c_pr_coefficient_point.py`と専用diagnostic/runner/testsが、同じdecoded J配列の残差/係数/chain評価へ揃える。既存inner/paired経路は無変更、実H4での原因確定・修正成功は未検証。RC2は保存JSONのみ、反復延長の成功は未判定。失敗trialUは直接未保存のため保存2点のdry BFGS replayと厳密な点identity確認を用意。実分子decode/新科学call0、旧raw2378/SOURCE/480-call STOPと131科学recordsを保全。別許可・fresh runまでmandatory STOP。以下は各時点の履歴。

## 研究C：改善reduced pilot・PR inner整合FAILで部分STOP（2026-11-02）

[480-call部分結果・新admission・未実施範囲](docs/research/research_c_j_u_stable_execution_20261102.md)。SOURCE12e0079a2/1563-call契約を無変更で実行し、480 attempts・478 COMPLETE・2 FAIL（PR inner例外と包むouter driver）、retry/resume0。保存H4/直鎖1Å/STO-3G、number CDF8/sector36/元順LD2、同元qualified x0・α=γ=.1・元scales。PR初期と未受理trial1の新stabilityはPASSしたが、trial2のchart対physical-chain gradient差2.39695e-7でcorrectness STOP、受理U更新0。開始済みRC2は75点admission PASS/64受理更新、F .01742887277/J∞3.67648e-8/U₂ .0047620784でU未資格。親の新DF norm/一般K-CAR/比較はNOT_RUN/null、旧/初期DF PASSを終端へ転用しない。旧raw1879/SOURCE/STOP・旧閾値を保全、新raw499を保存し公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：Reduced勾配の数値安定性・次pilot準備（2026-11-02）

[原因切分け・人工比較・220 items PASS・限界](docs/research/research_c_j_u_gradient_stability_investigation_20261102.md)／[未承認1563-call案](docs/research/research_c_j_u_stable_contract_unapproved_20261102.json)。新entry `scripts/run_research_c_j_u_stable.py`、core `c_j_u_precision.py`/`c_j_u_stable.py`、専用pilot/tests、証拠 `artifacts/research_c_j_u_stability/2026-11-02/technical/`。既存solver/目的/PR gaugeを維持し、拡張精度残差と2回のrefinementを接続。最後のpairを元1e-6で判定する新しい数値pilotで、厳密gradient上界ではない。人工改善・未解決例/joint対照を保全、実H4 decode/新評価/最適化0、旧raw1879/SOURCE/旧STOP不変。別明示許可・fresh runまでmandatory STOP。以下は各時点の履歴。

## 研究C：実H4 reduced pilot・両armのgradient stability未資格（2026-11-02）

[38/1051-call結果・全point・資格の限界](docs/research/research_c_j_u_reduced_execution_20261102.md)。SOURCEe1582a562を無変更で実行し38 COMPLETE・correctness FAIL/retry/resume0。保存H4/直鎖1Å/STO-3G、number CDF8/sector36/元順LD2・同元qualified x0・α=γ=.1。PRの共通head gauge数値条件と全inner J一次はPASSしたが、refinement U差はPR初期5.60584e-6、RC2未受理trial1で8.56926e-6となり固定1e-6を満たさず両REDUCED_GRADIENT_UNQUALIFIED。受理U更新0、外側性能・rebound回避は未判定。保持診断終点のDF予算は両PASS、qualified tail比較/下流/actual CAはnull。旧raw1825とSOURCE/STOPを保全、公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：J/U結合を扱うreduced objectiveの技術準備（2026-11-02）

[数学・再利用・人工186 items PASS・限界](docs/research/research_c_j_u_coupled_reduced_objective_preparation_20261102.md)／[未承認1051-call](docs/research/research_c_j_u_reduced_contract_unapproved_20261102.json)。新entry `scripts/run_research_c_j_u_reduced.py`、core/pilot `src/trottertracks/c_j_u_reduced*.py`、人工tests `tests/management/test_research_c_j_u_reduced*.py`、証拠 `artifacts/research_c_j_u_reduced/2026-11-02/technical/`。RC2一意な内側解とPRの共通head gauge条件を使い、全trialでJをsolve・再精密化する。追加null/inner未達は未資格STOP、refinement一致は厳密gradient上界ではない。実H4 decode/再解析/最適化/科学call0、旧raw1825保全、別明示許可までmandatory STOP。以下は各時点の履歴。

## 研究C：評価点修正・再実行完了、DF両PASS・一次未収束（2026-11-02公開）

[原因証拠・修正・新結果](docs/research/research_c_j_u_consistency_repair_execution_20261101.md)。保存H4/1Å/STO-3G、number CDF8/sector36/元順LD2・同元qualified x0/α=γ=.1。保存診断15＋修正版paired328＝343 units COMPLETE/FAIL0、修正版再実行1回・各新run内retry/resume0。SciPy Newton-CGのpre-step jacと終点を取り違えた検証、およびcodec前後の異点gradient比較を修正。数値allowance/一次閾値/DF予算は不変。新DF normはPR1.10629e-8/RC2 5.01800e-9Haで両PASS、終端同一点微分・sector表現PASS。PRは35cyclesでJ/U未資格、RC2は64cyclesでU未資格。Qualified比較null/下流0、旧FAILと旧raw1471保全、公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：曲率利用J/U・2整合gate FAILで部分STOP（2026-11-01）

[実行・partial・未実施範囲](docs/research/research_c_separated_J_U_curvature_execution_20261101.md)。固定SOURCEecb2e7dc0無変更、保存H4/1Å/STO-3G、number CDF8/sector36/元順LD2・同元qualified x0/α=γ=.1の両arm各1 trajectory。210/最大415 attempts、208 COMPLETE/2 FAIL、retry-resume0。PRはJ0後のchart/gradient整合FAIL、開始済みRC2は64 cycles後U未資格を記録した後、独立ffsim目的/J-gradient整合FAIL。両final export・新DF norm/表現/比較はNOT_RUN/null。複合gate内訳の保存不足で原因は未確定、追加再評価や閾値変更をせずraw226/旧原本を保全。公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：J/U停滞の人工切り分け・曲率利用版を準備（2026-11-01）

[原因候補・実装・257人工/回帰PASS](docs/research/research_c_separated_J_U_stagnation_diagnostic_preparation_20261101.md)。PR Jはaffine residual＋解析Hessian/Newton-CG、Uは固定J residual/Jacobian＋TRF/exactへ変更し、RC2 augmented gelsdは維持。元の目的/rank/一次閾値/64 cyclesを保持。人工の固定J U残差は改善例があるが、J refit後のU勾配reboundと強いconditioningの同時未収束は残る。新415-call案は保存H4/1Å/STO-3G、number CDF8/sector36/元順LD2・同元qualified x0の両arm各1 trajectory。SOURCE119/input17のpassiveのみ、実H4 decode/trace数値再解析/最適化/科学call0。旧258-call原本は保全し、残り159 unitsを使わない。公開照合後mandatory STOP。以下は旧結果・準備の履歴。

## 研究C：J/U分離pilot・DF両PASS、一次収束は両未資格（2026-11-01）

[全結果・一次診断・限界](docs/research/research_c_separated_J_U_execution_20261101.md)。科学SOURCEcbbb11e7e無変更で新規交互最適化2 trajectory、258/最大417 COMPLETE・correctness FAIL/retry-resume0。保存直鎖H4/1Å/STO-3G、number CDF8/sector36/元順LD2、同元qualified x0・α=γ=.1。PRは11 cycles後J勾配未達、RC2はJ基準METだがU未達で64 cycles上限。両DF予算PASS、PR λ .954543284658→.773820763063Ha（約18.93%減）は未収束終点の記述値。RC2自己目的は旧jointより高く、両U残差も旧より大きい。QualifiedなPR/RC2比較はnull、PF/RTE/native/shots/CA/総資源0、旧SOURCE/raw/STOP保全。公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：J凸問題・直交U更新を分離する技術準備（2026-11-01）

[数学・実装・227人工/回帰PASS](docs/research/research_c_separated_J_U_optimization_preparation_20261101.md)／[未承認417-call契約](docs/research/research_c_separated_J_U_contract_unapproved_20261101.json)。Entryはscripts/run_research_c_separated_J_U.py、core/pilotはsrc/trottertracks/c_separated_j_u*.py、testはtests/management/test_research_c_separated_J_U*.py、証拠はartifacts/research_c_separated_J_U/2026-11-01/technical/。全signed対称JをPR凸BFGS／RC2 augmented SVDで解き、局所直交U更新後にJを再solveする。旧tail tensor保存等式は継承しない。保存H4/1Å/STO-3G、number CDF8/sector36/元順LD2の同qualified x0から両arm各1 trajectoryを提案。SOURCE114/input13のpassiveのみ、実H4 decode/最適化/再解析/科学call0。実収束・DF予算・λ改善は未検証、公開照合後mandatory STOP。以下は過去の結果・準備の履歴。

## 研究C：固定U LP・係数保持profileで50/50 PASS（2026-11-01）

[数値資格・保持gate・全結果](docs/research/research_c_fixed_u_tail_lp_numerics_execution_20261101.md)。科学SOURCEc40f81281無変更、LP1回・50 COMPLETE・FAIL/retry/resume0。保存H4/直鎖1Å/STO-3G、PR通常CDF8/sector36/元順LD2、固定U/head0/1/tail2..7、h.1/s1/r128/K0。元全行列のdual/gap/guardを通り、λ .954543284658→.954432028127Ha（約.0117%減）。独立tensor・sector/full CAR・phase・DF予算・1h-step PF/K0保持PASS。旧optimizer未資格・厳密certificate/CA/shots/native/総資源nullは維持。Raw31/旧原本保全、公開照合後mandatory STOP。以下は準備・旧実行の履歴。

## 研究C：固定U LP・微小係数保持profileの技術準備（2026-11-01）

[人工切り分け・104 test methods PASS](docs/research/research_c_fixed_u_tail_lp_numerics_preparation_20261101.md)／[新しい未承認50-call](docs/research/research_c_fixed_u_tail_lp_numerics_contract_unapproved_20261101.json)。Entryはscripts/run_research_c_fixed_u_tail_lp_numerics.py、adapterはsrc/trottertracks/c_fixed_u_tail_lp_numerics.py、人工testはtests/management/test_research_c_fixed_u_tail_lp_numerics.py。small_matrix_value=1e-12と固定SciPy警告資格を追加し、元全行列・RHS・dual/gap/guard・旧SOURCEを維持。実H4 decode/LP/科学call0、SOURCE109/input12のpassiveのみ。旧UNRESOLVEDは未解決のまま、公開照合後mandatory STOP。

## 研究C：固定U LP・数値最適性未資格でSTOP（2026-11-01）

[LP1回・独立診断・9/50結果](docs/research/research_c_fixed_u_tail_preserving_lp_execution_20261101.md)。SOURCE56a6acb8f無変更、9 COMPLETE/例外FAIL0、retry/resume0。保存H4/直鎖1Å/STO-3G、PR通常CDF8/sector36/元順LD2、head0/1・tail2..7、h.1/s1/r128/K0。HiGHS Optimalだがdual等式残差7.8707>1e-9、gap/相殺guardも不適格でUNRESOLVED_LP。新candidate・保持/CAR/PF-RTE/DF/AはNOT_RUN。改善不能とは認定せず、raw11/旧原本を保全して公開後mandatory STOP。以下は準備時の履歴。

## 研究C：固定U・tail保存係数LPの技術準備（2026-11-01）

[数学仕様・Q1〜Q10・93人工PASS＋29 subtests](docs/research/research_c_fixed_u_tail_preserving_lp_preparation_20261101.md)／[未承認50-call](docs/research/research_c_fixed_u_tail_preserving_lp_contract_unapproved_20261101.json)。研究優先方向はユーザー指定の候補A。Entryはscripts/run_research_c_fixed_u_tail_lp.py、線形作用/LPはsrc/trottertracks/c_fixed_u_tail_lp.py、接続は同c_fixed_u_tail_lp_runner.py、testはtests/management/test_research_c_fixed_u_tail_lp.py。固定U/head/full tail tensorとexact Pauli L1、full-row primal/dual/conditioning、既存CAR/PF-K0へ接続。SOURCE105/input9のpassiveのみ、実H4 decode/LP/科学call0、実改善は未評価。公開照合後mandatory STOP、候補Bや総資源へ自動進行しない。

## 研究C：保存RC2／PR native・655-call完了（2026-11-01）

[全gate・6価格指標・資格](docs/research/research_c_rc2_pr_native_execution_20261101.md)。固定SOURCE14843a28f無変更、655/655 COMPLETE・FAIL/retry/resume0、64 kernel/67 compile/262 QPYを保存。保存H4/1Å/STO-3G、未収束CDF8両arm/sector36/元順LD2、partial S2 δ=h=.1/s1/r128/K0、own-oracle/準備除外、各32非paired trajectory・P0の完全h-step平均RZはRC2 20487／PR13246.375。全law/CA/support/総資源は未資格・null。旧SOURCE/raw/STOP不変、公開照合後mandatory STOP。以下の準備表示は履歴。

## 研究C：保存RC2／PR-tailのnative校正準備（2026-11-01）

[技術報告・105人工PASS](docs/research/research_c_rc2_pr_native_preparation_20261101.md)／[未承認655-call](docs/research/research_c_rc2_pr_native_contract_unapproved_20261101.json)。新entryはscripts/run_research_c_rc2_pr_native.py、moduleはsrc/trottertracks/c_rc2_pr_native.py、testはtests/management/test_research_c_rc2_pr_native.py。保存H4/1Å/STO-3G、未収束CDF8両arm、LD2/partial S2 δ.1/s1/r128/K0、各32 trajectoryのfull512数値gate・P0価格を提案。SOURCE130/入力26のpassiveのみ、実H4評価0。全law/CA/support/総資源は未資格、科学実行には別許可が必要。公開照合後mandatory STOP。

## 研究C：条件付きshots／W・105-call完了（2026-11-01）

[全3 lane・156 axis・結果と限界](docs/research/research_c_rc2_pr_conditional_shots_execution_20261101.md)。SOURCEe88fa599a無変更、105/105 COMPLETE・FAIL/retry/resume0。保存H4/1Å/STO-3G、CDF8/sector36/元順LD2、partial S2 δ.1/s1/r128/K0・13round、own-DF oracle/準備除外。Ideal NはRC2 18200774092602／PR14504、Wは74550091248614488／11822240。AssumedでもPRが小さく、evidenceはUNQUALIFIED。未収束候補の条件付き値で、μ/S/actual CA/総資源はnull。旧SOURCE/raw/STOP保全、公開照合後mandatory STOP。以下の準備表示は履歴。

## 研究C：保存RC2／PR-tailの条件付きshots／W準備（2026-11-01）

[Three lanes・数値域・125人工PASS・科学契約](docs/research/research_c_rc2_pr_conditional_shots_preparation_20261101.md)／[未承認105-call](docs/research/research_c_rc2_pr_conditional_shots_contract_unapproved_20261101.json)。保存H4/1Å/STO-3G、CDF8/sector36/元順LD2、partial S2 δ.1/s1/r128/K0・13roundのStage B26セルを、既存C1 accuracy/energy/Hoeffdingと大整数ceilへ接続。Evidenceはnull、ideal/assumedは明示仮定、NとWは十進文字列、rをWへ重ねない。新entryはscripts/run_research_c_rc2_pr_conditional_shots.py、moduleはsrc/trottertracks/c_rc2_pr_conditional_shots.py、testはtests/management/test_research_c_rc2_pr_conditional_shots.py。SOURCE101/input14のpassiveのみ、実H4 shots/W/energy・科学call0。Native価格／総資源は未知、旧SOURCE/raw/STOP保全、公開照合後mandatory STOP。

## 研究C：保存RC2／PR-tail Stage B・143-call完了（2026-11-01）

[結果・26セル・gate・未資格](docs/research/research_c_rc2_pr_stage_b_execution_20261101.md)。固定SOURCEd6b17b794無変更、143/143 COMPLETE・correctness FAIL/retry-resume0。保存H4/1Å/STO-3G、CDF8/sector36/元順LD2、partial S2 s1/r128/K0・δ.1・13round、own-ground/準備除外。両候補13/13 POTENTIALLY_FEASIBLE、m12のAはRC2 8.42470113e-6／PR .864344018。PRはPF誤差も大きく、RTE改善はm6..12。固定未収束終点の観測で、最適RC-DF／CA／shots／総資源の資格へ昇格しない。旧SOURCE/raw/failed STOP保全、公開照合後mandatory STOP。以下の準備表示は履歴。

## 研究C：保存RC2／PR-tailのStage B準備（2026-11-01）

[目的・SOURCE・人工88 PASS・Q1〜Q10](docs/research/research_c_rc2_pr_stage_b_preparation_20261101.md)／[未承認143-call契約](docs/research/research_c_rc2_pr_stage_b_contract_unapproved_20261101.json)。直近の共通収束2-fit保存終点を新identityで接続し、DF予算PASSと両optimizer未資格を分離。保存H4/1Å/STO-3G、CDF8/sector36/元順LD2、partial S2 s1/r128/K0・δ.1・13roundのPF/RTE/Aを将来独立評価する準備。SOURCE74/input13のpassive確認のみで実H4 decode/科学call0、future run未作成。旧SOURCE/raw/partial/failed STOP保全、公開照合後mandatory STOP。実装はsrc/trottertracks/c_rc2_pr_stage_b.py、CLIはscripts/run_research_c_rc2_pr_stage_b.py、人工testはtests/management/test_research_c_rc2_pr_stage_b.py。

## 研究C：JSON出力を修正、保存scalarだけで集計復旧（2026-11-01）

[修正・40人工PASS・復旧結果](docs/research/research_c_common_convergence_export_repair_20261101.md)／[有効result.json](artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/result.json)。旧SOURCE44/partial/raw37/failed STOPを保全し、独立adapterでNumPy scalar変換とatomic/exclusive出力・例外経路を修正。追加fit/optimizer/NPZ decode/新数値kernel0、保存scalar集計1 unitのみ。DF予算は両PASS、共通収束資格は両未達でONE_OR_BOTH_UNQUALIFIEDのまま。旧25＋修復1 unit、累計fit2。新runのSTOP成功と旧科学runの技術STOPを区別し、公開照合後mandatory STOP。

## 研究C：共通収束2-fit・25計算完了後、JSON出力で技術STOP（2026-11-01）

[終了値・未資格・serialization不具合・原本保全](docs/research/research_c_common_convergence_execution_20261101.md)。固定SOURCEe2f7dfd44で新fit2／25 COMPLETE、数値gate FAIL0、最終JSON出力FAIL、親STOP.failed=true、retry/resume0。保存直鎖H4/1Å/STO-3G、CDF8/sector36/元順LD2、同x0・α=γ=.1の両armは評価回数上限で終了し二つの一次基準未達。λ RC2=8.54568712823／PR=.954543284658 Haは未収束の記述値、共通収束後の比較は未成立。Partial result.jsonとraw37件を保全し、追加fit／集計call再実行／Stage B-Cへ進まずmandatory STOP。以下は準備時の履歴。

## 研究C：RC2／PR共通収束政策・最大2-fitの技術準備（2026-11-01）

[数理・options・資格・Q1〜Q10](docs/research/research_c_common_convergence_policy_design_20261101.md)／[未承認25-call契約](docs/research/research_c_common_convergence_contract_unapproved_20261101.json)。同じ元x0・α=γ=.1で両armを独立fitし、native gradient∞／正規化identity-null投影とDF予算を分離する準備。旧loss/codec/fitter無変更、新25＋回帰123＝148人工PASS、SOURCE44/input14の両環境passive preflight PASS。実H4 decode/fit/科学call0、future run01未作成。Stage Bは再利用経路のみ、公開照合後mandatory STOP。

## 研究C：保存3点・一次診断14-call実行完了（2026-11-01）

[結果・全gate・9 value-gradient・次の判断](docs/research/research_c_regularization_first_order_execution_20261101.md)。固定SOURCE0996a5abb、14 COMPLETE/FAIL0、fit/optimizer/retry-resume0。保存H4/1Å/STO-3G、CDF8/sector36/元順LD2、α=γ=.1の初期寄与一致と勾配影響の差を確認。両自己終点はgtol未達、RC2の既知tensor-null方向にL2下降成分が残る。最適ρ/旧停止原因は未確定。旧SOURCE/raw/STOP不変、追加fit/B-Cへ進まずmandatory STOP。以下の準備表示は履歴。

## 研究C：RC2強度・停止条件の診断と0-fit準備（2026-11-01）

[技術診断・Q1〜Q10](docs/research/research_c_regularization_strength_stopping_diagnostic_20261101.md)／[未承認14-call契約](docs/research/research_c_regularization_first_order_contract_unapproved_20261101.json)。固定pilotは有効、強い最適化済み対照の資格は未確立。B4共通物理予算を優先する提案と、3保存点の既存loss/penalty一次診断を準備。人工123 PASS、SOURCE39/input9のpassive preflight PASS、実H4科学call/decode/fit0。旧SOURCE・run01・STOP不変、別許可までmandatory STOP。

## 研究C：RC2対照の1-fit/17-call完了（2026-11-01）

[結果・optimizer・物理予算・旧候補比較](docs/research/research_c_regularization_comparator_execution_20261101.md)。科学SOURCE150f904f6無変更、新fit1・17/17 COMPLETE、correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、通常CDF8/sector36/元順LD2の同x0、共通optimizer/full ordered Z=J L2・初期寄与.1。終了sector norm5.8389057e-9Ha＋guard3.2787812e-9Haで予算PASS、optimizer success/nit97。Tail λ2.20976187195Haは保存PR-tail γ.1の1.44242988558Haより大きい。固定pilotのproxy差で、最適RC-DF/資源/CA/新規性の優位ではない。PF/RTE/native/shots0、B/C未承認、公開照合後mandatory STOP。

## 研究C：既存正則化との目的比較・Stage A準備（2026-11-01）

[数理対応・強度規則・人工171 PASS＋29 subtests](docs/research/research_c_regularization_comparator_design_20261101.md)／[未承認17-call契約](docs/research/research_c_regularization_comparator_contract_unapproved_20261101.json)。Z_RCDF=空間J、spin K=J/2。一般signed J・現在のsmooth tail目的は固定entry L1/L2と一般に非同値、非平滑/符号制約下のL1二乗帰着は別記。共通optimizer下の全frame uniform L2を初期penalty寄与一致で比較するCOMPARABLE_DISTINCT_OBJECTIVE。新penalty/専用entryのみ、既存SOURCE/runner不変。新H4 fit/再解析/科学call0、将来Stage Aは1 fit/17 units、B/C未承認、公開照合後mandatory STOP。

## 研究C：既存成果のEvidence Mapと研究判断資料（2026-10-31）

[中心資料・23項目の証拠対応・Q1〜Q10](docs/research/research_c_evidence_synthesis_and_decision_readiness_20261031.md)。PR-aware DFのtail mass/DF予算、特定条件のA/shots、PR内部条件付き資源改善をClaim A〜Cとして分離。同じJ/Uの強決定論への主RZ優位は未支持。P4・CDF7・surrogate・round割当の限定負結果、shared-basisの数学的帰着、lawの方法差未特定を区別する。既存値の引用とSOURCE/原本照合のみで新科学call・保存数値再解析0。論文化GO/独立新規性/正式終了は未判断、選択肢A/B/Cはユーザー／研究Cの判断材料。公開照合後mandatory STOP。

## 研究C：PR sampling lawの同値性調査・追加実験なしの提案（2026-10-31）

[数学仕様・6分類・Q1〜Q9](docs/research/research_c_sampling_law_freedom_survey_20261030.md)。SOURCE fb0c803f1から独立調査。NO_DISTINCT_FREEDOMは確認範囲で独立した実装可能機構を特定できなかったという技術判断で、全law不可能性や正式な研究終了ではない。K0 common-parentのL1最小性、affine辞書依存の例外、全path IS、±1測定とcontrolled位相を区別。Track B D＋E維持、新9＋既存4＝13人工PASS、実H4再解析・sampling・build/compile・科学call0。実行可能科学契約なし、公開照合後mandatory STOP。

## 研究C：Round別D/PR・189-call完了、主RZは全Dを選択（2026-10-30）

[全13round・3lane・6指標・証拠](docs/research/research_c_roundwise_det_pr_execution_20261030.md)。SOURCE79fa01076無変更、189/189 COMPLETE、correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、同じCDF8γ.1/sector36、D全frame S2 s1／PR LD2 partial S2 s1/r128/K0、δ=h=.1/m0..12、own-ground/準備除外。主RZの両仮定laneで全13roundがDを選びMIXED_ROUND_DOMINATED、evidenceはPR誤差未資格でUNQUALIFIED。Ideal総RZは全D＝選択101825929455、全PR234117788135。固定点モデル内のNO-GOであり、全指標支配/母平均/実CA/全law保証ではない。原本保全・公開後mandatory STOP。

## 研究C：Round別D/PRの共通energy接続・保存189-call準備（2026-10-30）

[数学的十分条件・SOURCE・人工117 PASS](docs/research/research_c_roundwise_det_pr_allocation_design_20261030.md)／[未承認契約](docs/research/research_c_roundwise_det_pr_contract_unapproved_20261030.json)。同じ保存CDF8γ.1/LD2のM2全frame S2 s1とM4 PR s1/r128/K0、δ.1/m0..12・own-oracle/準備除外。既存nearest-branch/accuracy/serial会計を再利用し、固定β/α下でO(13×2)選択。実H4の価格順位・混合結果は未判定、科学call0。別実行許可までmandatory STOP。

## 研究C：PR native構造の技術調査・追加実験はNO-GO提案（2026-10-30）

[構造・A/B/C・人工68 PASS・判断](docs/research/research_c_pr_native_structure_optimization_design_20261030.md)。Head相対融合／tail exact-key共有は既済。人工complete kernelでAはP0後RZ増加、Cは不変（局所削減例あり）。実H4効果はUNKNOWN。候補採択0、非実行判断call0、実H4 decode/build/compile/再解析0。Round別決定論/PR割当の共通energy接続を次の技術課題として提案し、公開照合後mandatory STOP。

## 研究C：CDF7 native32完了・主指標RZの点改善なし（2026-10-30）

[結果・full512・費用6指標・条件付き比較](docs/research/research_c_cdf7_native32_execution_20261030.md)。SOURCE173ba2fd8無変更、332/332 COMPLETE、correctness FAIL/retry/resume0、QPY128。保存直鎖H4/1Å/STO-3G、通常CDF7/sector36/LD2、partial S2 δ=h=.1/s1/r128/K0、own-ground/状態準備除外。Raw/compiled/actual adjoint/X-Y各32 PASS。Kernel RZ mean±SE17754.9375±169.1912、ideal仮定総RZ638880821593.5（通常CDF8γ0比1.0188545／M4比2.7288863）。主RZの記述的点改善なし、母平均の優劣・全law・実CA・保証付き優位は未資格。旧PF/RTE/shots再計算・新fit0、SOURCE107/input33/旧保護2292不変、raw369件。公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：CDF7 native32・332-call独立entry（2026-10-30）

[技術接続・SOURCE・科学契約](docs/research/research_c_cdf7_native32_preparation_20261030.md)。旧native本体323＋保存入力接続9＝332 units。保存H4/1Å/STO-3G、CDF7/sector36/LD2、partial S2 δ.1/s1/r128/K0、32独立trajectory／full512／P0／QPY128。保存PF/RTE・ground・shotsを再利用し再計算0、fit0。ユーザーの一括指示により人工検証・SOURCE固定後に一度実行し、有限cohortの条件付き比較まで保存してmandatory STOPする。

## 研究C：保存CDF7 PR119-call完了・native価格は未測定（2026-10-30）

[結果・全13 round・RPE・損益分岐](docs/research/research_c_cdf7_PR_execution_20261030.md)。SOURCE3e75b1690無変更、119/119 COMPLETE、correctness FAIL/retry/resume0。保存直鎖H4/1Å/STO-3G、通常CDF7/sector36/元順LD2、partial S2 δ=h=.1/s1/r128/K0、own-ground/状態準備除外。8因子・ground/alias・13/13 PF-RTE十分条件PASS。Ideal仮定shots21768、W35983272、RZ kernel損益分岐は通常CDF8γ0に17426.372／M4に6506.293。Evidence laneはUNQUALIFIED、実CA／新native価格／総資源はnull。Nativeの別提案対象となったが今回build/compile/sampling/fit0。SOURCE102/input20/旧保護2214不変、公開照合後mandatory STOP。以下は各時点の履歴。

## 研究C：保存CDF7のPR119-call経路・技術準備完了（2026-10-30）

[技術報告・SOURCE・新ready契約](docs/research/research_c_cdf7_PR_preparation_20261030.md)。Rank7専用adapterでbase＋7 frame、sector36の独立CAR／PF-RTE／RPE／2比較先の損益分岐を接続。新11＋回帰28＝39人工PASS、passive preflight PASS。最大119 units、登録十分条件不適格なら63で終了。今回の実H4科学call0、native／sampling／fit0、未知合成・law誤差と価格はnull。旧非実行draft・1-fit結果は保全。科学実行は別途明示許可後、公開照合後mandatory STOP。

## 研究C：通常CDF7・限定1-fitでDF予算PASS（2026-10-30）

[結果・初期/終了誤差・CDF8比較・後続案](docs/research/research_c_cdf7_onefit_execution_20261030.md)。保存直鎖H4/1Å/STO-3G、Nα=Nβ2/sector36、通常number CDF7/元順LD2、γ0 normalized tensor loss、ffsim0.0.83/修正codec/L-BFGS-B options=None。SOURCEc13d6f82a無変更で25/25 COMPLETE、correctness FAIL0、rank7 Cholesky初期化1/fit1/retry-resume0。初期sector+guard9.0647035e-4Haは予算未達の診断としてfitを許容し、終了3.02264175e-7HaでCA/8=.00019875Ha PASS。Optimizerは215iterの相対目的減少停止、正規化gradient∞=.05556で停留/大域保証なし。人工45 PASS＋29 subtests、旧保護2153件不変。PR/PF-RTE/ground/native/shots/総資源0。後続119-call案はrank7 adapter/SOURCE未固定でexecutable=false、native32は別案。公開照合後mandatory STOP。


## 研究C：通常CDF7の限定1-fit準備（2026-10-30）

[実装・人工45 PASS・固定25-call](docs/research/research_c_cdf7_onefit_preparation_20261030.md)。新専用module/CLIは既存ffsim Cholesky initializer・γ0 loss・修正codec・SectorCARを再利用。初期予算は診断のみ、表現correctness後に1-fit。保存H4/1Å/STO-3G、sector36、rank7/LD2、options=None、最終CA/8、fresh run01/retry-resume0。SOURCE35/input8/旧保護2153件固定、後続PRは技術案のみ。Shared-basisのexecutable=false契約は保全。


## 研究C：Shared-basisは通常低rank CDFと同値（2026-10-29）

[定義・証明・native対応・61人工PASS](docs/research/research_c_shared_basis_df_design_20261029.md)。一般対称Jの共有G groupは通常CDF≤Gと同じ表現集合で、CDF8のtail F2/F3共有はCDF≤7に対応。冗長Jと通常rank削減を新DF方式と呼ばず、専用fitter/重複fitはNO-GO提案。固定H_D/H_R・h/r/Kで現partial PF/補正RTEは不変だが、atom相殺でλ/law/減衰は変化し得る。Signed-zeroのtail keyと決定論tuple判定、head相対融合とtail共有を区別した。新21＋回帰40＝61 PASS＋29 subtests、旧SOURCE/原本2140件照合、実H4 decode/fit/新再解析/build/compile/科学call0。非実行の未承認案はcall0/executable=false。通常CDF7の物理予算は未評価、次方向は未採択。公開照合後mandatory STOP。


## 研究C：第三γ0 DF・修正後493-call完了、switch転移判定不成立（2026-10-29）

[結果・32 full512・holdout・条件付き費用](docs/research/research_c_df_basis_third_repair_execution_20261029.md)。保存直鎖H4/1Å/STO-3G、通常γ0 CDF8/元順LD2、partial S2 δ=h=.1/s1/r128/K0・13round、own-ground/状態準備除外。Components metadataだけを修正し人工46 PASS、SOURCE30436b142を固定してfresh run02で493/493 COMPLETE、correctness FAIL0。Raw/compiled/actual adjoint/X/Yは各32 PASS、QPY128。RZ holdout MAEはA369.78698、B303.32856、mean2892.23438のためSWITCH_TRANSFER_NOT_SUPPORTED。Kernel平均RZ17454.1875、ideal仮定shots21754/総RZ627057959562.5（M4比2.678387、P4比2.276714）。有限数値pilotで全law/actual CA/cost support/因果・汎化/η採択は未資格。旧SOURCE112/run01/保護1725件不変、新fit0、技術relaunch1/run内retry-resume0。公開照合後mandatory STOP。以下の未完了/旧STOP表記は各時点の履歴。


## 研究C：第三DF metadata修正v2（2026-10-29）

[差分・実B2 gate回帰・fresh run02](docs/research/research_c_df_basis_third_repair_preparation_20261029.md)。`c_df_basis_third_v2.py`／`run_research_c_df_basis_third_v2.py`／人工testを追加し、旧SOURCE112不変で既存component-recordを接続。新4＋回帰42＝46 PASS。科学条件・493-call・sampling/P0/gate不変。旧141-call STOPを保全し、チャット許可に基づく新SOURCE115/fresh run02の準備を完了。以下の未修正表記は履歴。

## 研究C：第三γ0 DF・141-callでnative metadata技術STOP（2026-10-29）

[部分結果・全13round・STOP原因](docs/research/research_c_df_basis_third_execution_20261029.md)。保存H4/1Å/STO-3G、γ0 CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、own-ground/状態準備除外。SOURCE14270239a無変更で141 attempts=140 COMPLETE+1 technical FAIL。表現/ground/alias・13/13 PF-RTE十分条件PASS、ideal仮定shots21754。Native full_law_gateへ渡すmetadataのcomponents欠損でSTOPし、sampling/build/compile/DF holdout0。第三DF価格/予測誤差/総費用はnull、旧C不適格不変。SOURCE112/input93/旧保護1667件照合、新fit/retry/resume0、旧原本保全。公開照合後mandatory STOP。以下の未実行表記は当時の履歴。

## 研究C：basis-switch-aware設計・保存γ0第三DFの準備（2026-10-29）

[数式・再利用・人工検証・未承認493-call案](docs/research/research_c_df_basis_switch_optimization_design_20261029.md)。固定exact-key partitionのswitch項はJのmass比に依存し、自由Uの幾何を直接最適化しない。Exact/smooth零tail・ε bias・非凸性・gaugeを分離し、η/新目的は未採択。保存H4/1Å/STO-3G、γ0 CDF8/LD2を第三候補としてown h=.1/s1/r128/K0の13round、条件付きRPE、GO時native32、DF holdoutへ接続。旧2cohortの共通interceptを第三価格前に固定し、Cは復活させない。新21＋回帰110＝131 PASS、科学call0、SOURCE112/input93/旧原本1667件照合、旧103 SOURCE/STOP不変。別科学許可までmandatory STOP。

## 研究C：basis-key修正後148-call完了・複合surrogate未資格（2026-10-29）

[結果・モデル比較・旧STOP保全](docs/research/research_c_df_basis_surrogate_repair_execution_20261029.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、M4 Proposed/tail-P4各32件。既存spin_liftによる厳密key修正、新7＋回帰43＝50 PASS、SOURCE030128649…を固定しfresh run02で148/148 COMPLETE、correctness FAIL0。RZ MAEはM4でA221.41<C227.80、P4でB261.54<C273.58のため両cohortともSURROGATE_NOT_QUALIFIED。全192 fold full rank・負予測0。静的DFは2個、因果/新DF汎化/η採択は未資格。新fit/sampling/build/compile0、旧SOURCE99/run01/保護1570件不変。技術relaunch1、run内retry/resume0、公開照合後mandatory STOP。以下の未実施/旧STOP表記は履歴。

## 研究C：basis-aware saved64・3-callで技術STOP（2026-10-29）

[部分証拠・原因候補・残る修正](docs/research/research_c_df_basis_surrogate_execution_20261029.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、PR s1/r128/K0の2候補各32件を対象に固定SOURCEb1985ff01で開始。3 attempts=2 COMPLETE+1 FAIL、M4_Proposed/static_featuresのEXACT_SAVED_BASIS_KEYでSTOP。Trajectory/QPY/model未実施、性能未判定。既存spin_liftと新np.kron経路のsigned-zero repr差を人工2×2だけで再現。SOURCE/契約変更・retry/resume0、旧1458件保全、公開照合後mandatory STOP。

## 研究C：basis-aware native surrogateの技術設計（2026-10-29）

[3案・実装対応・2cohort schema・未承認148-call案](docs/research/research_c_df_basis_native_cost_surrogate_design_20261029.md)。切替/overlap/複合run-atomを比較し、複合特徴ベクトルを推奨。新12＋回帰31＝43人工PASS。保存H4の新特徴量・相関・NPZ/QPY数値decode・fit/sampling/build/compileは0。η/scale/新目的は未採択。2静的DFを64標本扱いせず、次は保存各32件の独立cohort内記述検証だけを別許可で実行。旧原本保全、公開照合後mandatory STOP。

## 研究C：tail-P4 PRの449-call完了・総資源改善なし（2026-10-28）

[PF/RTE・shots・損益分岐・native・総資源](docs/research/research_c_df_structural_pr_performance_20261028.md)。保存直鎖H4・1Å/STO-3G、tail-P4 CDF8/LD2、partial S2 δ.1/s1/r128/K0/m0..12、own-DF oracle/状態準備除外。SOURCE4ac0e5f96無変更、449/449 COMPLETE、correctness FAIL/retry/resume0。全13round適格、主仮定shots16,568、fixed32/full512/adjoint/X/Y PASS。Kernel平均RZ15,047.46875、M4-r128比1.068983。条件付き総RZ275,422,425,481.5/旧234,117,788,135=1.176427。疎化による価格/総資源改善は今回の点比較では観測なし。全law/actual CA/母平均優位は未資格。旧1054件保全、公開照合後mandatory STOP。

## 研究C：tail-P4のPR・条件付きnative32経路（2026-10-28）

[再利用・40人工PASS・449-call段階付き契約](docs/research/research_c_df_structural_pr_preparation_20261028.md)。候補固有role、保存J/U固定、new13round/ground/law、登録GO時だけnative32。科学開始前SOURCE固定、旧gate/source不変、未知全law誤差はnull。

## 研究C：tail J固定support CDFの1 fit・DF予算PASS（2026-10-28）

[結果・SOURCE・call・STOP](docs/research/research_c_df_structural_support_execution_20261028.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、head-full/tail-P4、γ.1。SOURCE59e186ad無変更、21/21 COMPLETE、1 fit、retry/resume0、correctness FAIL0。終了sector誤差7.45156e-9Ha＋guard3.26388e-9HaでCA/8 PASS。Tail係数位置144だがλ1.51617Haは旧PR-aware1.44243Haより大きく、性能GOではない。PF/RTE/shots/native/総費用0、旧1006件保全、公開照合後mandatory STOP。

## 研究C：DF構造改良の技術提案・科学実行未承認（2026-10-28）

[横断調査・3案・再利用地図・最小契約](docs/research/research_c_df_structural_design_20261028.md)。保存H4・1Å/STO-3G、CDF8/LD2を維持し、既存γ/r探索を拡張せずtail Jの固定support制約を最初の検証案とした。既知support制約CDFの限定拡張で独立新規性/資源優位は未認定。旧loss/AD/codec/SciPyを再利用するcompact adapter、1-fit限定runner、人工60 tests＋29 subtests PASS。未承認21-callはDF評価だけで、実H4decode/fit/PF-RTE/shots/native0。後続は新J/U固有の別契約が必要。旧SOURCE/raw/STOP保全、remote照合後mandatory STOP。

## 研究C M4-Native128：固定32校正・332-call完了（2026-10-28）

[full512・全36判定・条件付き資源](docs/research/research_c_pr_aware_df_M4_native128_execution_20261028.md)。保存直鎖H4・1Å/STO-3G、Proposed CDF8/LD2、partial S2 δ.1/s1/r128/K0/m0..12、own-DF oracle/状態準備除外。旧87SOURCE無変更、説明128化の独立revision＋metadata test、人工回帰66 PASS。332/332 COMPLETE、32 full512/actual adjoint/X/Y PASS、retry/resume0。P0平均RZ 14076.437500、ideal仮定総RZ 234117788135.000、M3r64比0.793138/同JU決定論比2.299196。M4保存shots16056/W/S/36閾値を再利用、新PF/RTE/shot/CA/fit0。有限cohort条件付き点比較で、母平均/actual CA/全law/support未資格。旧642件保全、公開照合後mandatory STOP。前段のnative未実行表記は当時の履歴。

## 研究C M4：r128・保存PF/shots/損益分岐98-call完了（2026-10-28）

[結果・全13round・6指標閾値・次段階案](docs/research/research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md)。新`c_pr_aware_df_m4.py`/専用runnerはM3PF13を再利用して98/98 COMPLETE、FAIL/retry/resume0。保存Proposed CDF8/LD2/δ.1/s1/r128/K0・own-DF oracle/状態準備除外。全13適格、主仮定shots16056。未知kernel価格のRZ損益分岐はM3比17747.772397、同J/U決定論比6122.327980。実r128価格/総費用はnull、実sampling/build/compile0。別native adapter/CLIは未承認332-call/32sample案、人工5のみ。SOURCE/旧原本保全、公開照合後mandatory STOP。

## 研究C M3：JSON修正後のs1/r64・471-call完了（2026-10-28）

[結果・全13round・32 native校正・条件付き資源](docs/research/research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md)。修正SOURCE 5354edecc…を固定しfresh run02で471/471 COMPLETE、correctness FAIL0。全13適格・32 full512/adjoint/X/Y PASS。保存Proposed CDF8/LD2/δ.1/s1/r64/K0・own-DF oracle/状態準備除外、ideal仮定のshots 20734、総RZ 295179018953、同J/U決定論比 2.89885907。実装CA/全law/母平均優位は未資格。旧run01の19-call schemaSTOPを保持、技術relaunch1/run内retry-resume0。公開照合後mandatory STOP。

## 研究C M3：s1/r64・A途中の技術STOP（2026-10-28）

[結果・原因・19-call部分証拠](docs/research/research_c_pr_aware_df_M3_s1_r64_execution_20261028.md)。SOURCE9752f28cf…を変更せず一度実行し19 attempts/18 COMPLETE/1 schema FAIL。Atom catalogのtupleと保存JSONのlistを直接比較した不備でSTOP。PF/RTE round0、B/C未実行、科学的不適格は未判定。新`c_pr_aware_df_m3.py`／`run_research_c_pr_aware_df_M3.py`／人工test、92 PASSとJSON境界の限界を記録。Retry/resume0、旧原本保全、mandatory STOP。

## 研究C M2：同J/U決定論s1の253-call完了（2026-10-28）

[結果・再利用・68人工/回帰・253-call契約](docs/research/research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md)。新`c_pr_aware_df_m2.py`／`run_research_c_pr_aware_df_M2.py`／同名人工test。両full512/CA数値条件PASS、旧PR仮定に対し今回決定論s1が低費用。SOURCE無変更、retry/resume0、消費済みrun01を再起動せずmandatory STOP。

## 研究C M1・20-call保存値機構分解完了（2026-10-28）

[全結果・20-call・未資格事項](docs/research/research_c_pr_aware_df_mechanism_execution_20261028.md)。20 COMPLETE、correctness FAIL/retry/resume0、科学SOURCE変更0。旧2仮定でshots側項が価格側より大きい記述的分解。消費済みM1-run01を再起動せず、因果/CA/強baseline順位を認定しない。公開照合後mandatory STOP。

## 研究C：性能・機構・一般化／M1準備（2026-10-28）

[技術報告・SOURCE・未承認契約](docs/research/research_c_pr_aware_df_performance_mechanism_readiness_20261028.md)。`c_pr_aware_df_mechanism.py`／`run_research_c_pr_aware_df_mechanism.py`、passive既定・科学許可後だけ`--execute`。次の一実験には保存shots×保存whole-kernel価格の4corner・記述的寄与分解M1を推奨。新adapter/CLI/testの3ファイル、人工新30＋回帰23＝53 PASS、SOURCE7/input9・未承認20-call契約を固定した。因果効果/actual CA/保証shots/母平均優位への昇格なし。新科学call0、M1-run01未作成。公開照合後mandatory STOP、別明示許可まで再解析・新fit・PF/RTE・build/compile・sampling・移送・詳細レビューを開始しない。

## 2026-10-28：F3-K1 Proposed固定trajectory・7-call実行完了

[結果・証拠・資格範囲](docs/research/research_c_pr_aware_df_F3_K1_execution_20261028.md)。保存直鎖H4・1Å/STO-3G、通常number CDF8 Proposed proposed_00、Nα=Nβ2を含むfull256/controlled512、元順LD2/partial S2 δ.1/s4/h1/40/r16/K0。SOURCE 9b0f4d2b40159b77f28caf3f215ad22d0d075879を無変更で7/7 COMPLETE、correctness FAIL/retry/resume0。Arb192全512基底・位相込みでraw/B0≤3.33805E-14、κ bridge≤2.07978E-17、U bridge≤8.18918E-16、raw/K1≤3.42202E-14、compiled/K1≤3.24349E-12（上向き表示）。両方allowance1e-8内。Proposed F1の同一raw certificateを再利用し、Baseline値は転用していない。 保証はexact κH＋Proposed polar Qへの固定一trajectoryのみ。両armは非pairedで、全law/IID/物理Hamiltonian新保証/CA/保証shots/cost support/母平均・総資源優位は未資格。C1約83%は元の仮定付き点推定のまま。旧SOURCE/raw/STOPを保全し、追加計算せず公開照合後mandatory STOP。以下は各段階の履歴。

## 2026-10-28：Proposed F3-K1の独立7-call技術準備

[Proposed固有binding・F1 reuse・193 tests＋29 subtests](docs/research/research_c_pr_aware_df_F3_K1_readiness_20261027.md)／[未承認7-call契約](docs/research/research_c_pr_aware_df_F3_K1_contract_draft_20261027.json)。新F3 adapter/CLI/testの3ファイルのみ追加、旧K1数学/全512列verifier不変。Proposed F1の同一raw certificateだけを使用し、Baselineの値は転用しない。SOURCE145・入力・seed固定、実H4科学call0・別科学許可までmandatory STOP。

## 2026-10-27：F2-K1 Baseline固定trajectoryのCAR接続がvalidated PASS

[7-call結果・exact上界・資格の限界](docs/research/research_c_pr_aware_df_F2_K1_execution_20261027.md)。保存直鎖H4・1Å/STO-3G、通常number CDF8 Baseline baseline_00、元順LD2/partial S2 δ.1/s4/h1/40/r16/K0、full512・位相込み。SOURCE3742ec10dd…無変更、7 COMPLETE/correctness FAIL0/retry-resume0。Raw/B0≤2.57295e-14、κ bridge≤2.07978e-17、U bridge≤9.53720e-16、raw/K1≤2.67040e-14、compiled/K1≤3.49618e-12（上向き表示）。Exact κH＋polar Qの新target一trajectoryのみ。旧F2 STOP・元κ/旧SOURCE/raw不変、F3/全law/CA/shots/総資源は未資格、公開照合後mandatory STOP。

## 2026-10-27：F2-K1 exact κH＋polar U・技術準備

[target定義・κ/U bridge証明・153 tests＋29 subtests・未承認7-call契約](docs/research/research_c_pr_aware_df_F2_K1_target_readiness_20261027.md)。新opt-in adapter/runnerは元κ/UのB0を全25 CAR blocks・full512で囲い、exact κHとright-polar Qへの2 bridgeを各1回計上する。既存SOURCE・旧κ gate不変、新実H4科学call0、実入力J/eta等は将来の登録gate、公開照合後mandatory STOP。

## 2026-10-27：F2 κ限定診断が6/6完了・exact条件は不成立

[結果・原因分解・未資格事項](docs/research/research_c_pr_aware_df_F2_kappa_diagnostic_execution_20261027.md)。保存直鎖H4・1Å/STO-3G、Baseline通常number CDF8文脈、元h/g/scalarのみ。SOURCE802b2b0a16…無変更、6 COMPLETE/correctness FAIL0/retry-resume0。κ residual F=8.66047e-17はStage 1 allowance内だがexact Hermiticityなし。保存hとgの厳密縮約に継承非対称性、縮約・減算にも丸め寄与。元生成原因は未解決。CAR/target変更/F2再実行/F3なし、旧STOP不変、公開照合後mandatory STOP。

## 2026-10-27：F2 κ Hermiticity限定診断の技術準備

[SOURCE監査・Stage 1比較・K0/K1・未承認6-call契約](docs/research/research_c_pr_aware_df_F2_kappa_hermiticity_diagnostic_readiness_20261027.md)。`c_pr_aware_df_kappa_diagnostic.py`／専用runner／人工testが元補正式と独立exact dyadic縮約を接続。新規実H4科学call0、元snapshotは歴史blobのbyte同一回収のみ。旧F2の4-call STOPとSOURCE/raw不変、κ原因・targetは未確定、別明示科学許可までmandatory STOP。

## 2026-10-27：固定F2 polar・κの厳密Hermiticity条件でSTOP

[結果・partial evidence](docs/research/research_c_pr_aware_df_fixed_F2_polar_execution_20261027.md)。保存直鎖H4・1Å/STO-3G、通常number CDF8 Baseline baseline_00、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、sector36/full512、own-DF oracle/状態準備除外。Right-polar CAR targetを今回限定採択し、SOURCE45ae98943492e921c22a0f527d5587cd1e06462f／新契約を固定して実行。4/最大7 attempts＝3 COMPLETE＋1 FAIL、残3 NOT_RUN。CAR_reference入口の元g補正κがEXACT_HERMITIAN_INPUT_REQUIREDでSTOP。Raw/CAR・bridge・triangleは未評価/nullで、raw回路の測定correctness FAILではない。追加診断・κ変更・retry/resume・F3なし。旧B1/F1/C1の資格とraw/STOP不変、公開照合後mandatory STOP。

## 2026-10-27：固定Proposed F1・全基底validated評価

[結果・資格](docs/research/research_c_pr_aware_df_fixed_proposed_f1_execution_20261027.md)。保存H4・直鎖1Å/STO-3G、通常number CDF8 proposed_00、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、system8+last ancilla/full512、own-DF oracle/状態準備除外。SOURCE `e7d22a404442234662ef6d117c55dd9868217c9e` を無変更で一度実行し7/7 calls、PASS3/correctness FAIL0/UNRESOLVED0/NOT_RUN0、retry/resume0。Arb192全512基底・位相込みの有限固定回路保証。共通shell6 certificateはidentity一致で再利用。Raw/CAR・全law・CA/保証shots/総資源は未資格、F2 executable=falseを維持。

## 2026-10-27：固定CAR reference・Proposed検証の技術準備

[CAR接続・両arm準備／人工210 PASS](docs/research/research_c_pr_aware_df_fixed_car_reference_readiness_20261027.md)。新`c_pr_aware_df_fixed_car`／`fixed_car_execution`と専用runner・test。保存Proposed固定回路F1は未承認7 calls、独立CARのF2はliteral dyadic mapとpolar理想targetの選択が未採択のためexecutable=false。共通shell6保証再利用、新実H4科学call0、旧B1証拠不変、公開照合後mandatory STOP。

## 2026-10-27：C2-A B1・保存固定回路の事後validated評価

[実行報告](docs/research/research_c_pr_aware_df_c2a_fixed_b1_execution_20261027.md)。固定SOURCE `518a730510a57d0cafa1575a91f5f1979c203be3`を無変更で一度実行。保存H4・1Å/STO-3G、通常CDF8 baseline_00、元順LD2、partial S2 δ.1/s4/r16/K0、full512、own-oracle/状態準備除外。19/19 units、9判定はPASS9/FAIL0/UNRESOLVED0/NOT_RUN0、retry/resume0。Arb192全基底・位相保持の固定gate列証拠であり、raw-Hamiltonian/reference・全law・CA/保証shots/総資源は未資格/null。新fit/build-compile/sampling/C1再解析0、旧SOURCE/raw/STOPとC1資格は不変。公開照合後mandatory STOP。

## 2026-10-27：C2-A保証方式の判定・固定回路B1案

[技術報告・A/B比較](docs/research/research_c_pr_aware_df_c2a_guarantee_strategy_readiness_20261027.md)／[未承認19-call B1契約](docs/research/research_c_pr_aware_df_c2a_guarantee_contract_draft_20261027.json)。Stock P0全lawは現証拠では困難（A-3、数学的不可能性ではない）。Arb192全基底・recentered local actionで固定raw/compiledの事後検証を人工実装。保存baseline_00＋actual adjoints＋3shellだけの新有限scopeを準備。新37＋旧193 PASS、旧不安定1件deselect。実H4科学call0、旧A blocked契約・C1資格は不変。公開照合後mandatory STOP。

## 2026-10-27：ZSXX全到達domain・条件付きwhole-P0

[技術報告・Q1〜Q5](docs/research/research_c_pr_aware_df_zsxx_uniform_domain_readiness_20261027.md)／[新しい実行不可契約](docs/research/research_c_pr_aware_df_zsxx_uniform_contract_draft_20261027.json)。`c_pr_aware_df_zsxx_uniform.py`・固定人工fixture・passive/artifical-only entry・management test。RZ/SX/Xのdet4 coset、固定kernel grammarの条件付きN≤67731、run長／phase／whole-P0 proof-input式を接続。実Rust uniform imageは未資格。新規32＋旧回帰161 PASS、旧不安定1件deselect、実H4科学call0・mandatory STOP。証拠は`artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/`。

## 2026-10-27：C2-A・ZSXX captured-pair validated enclosure

[技術報告](docs/research/research_c_pr_aware_df_c2a_zsxx_enclosure_readiness_20261027.md)／[実行不可の新契約](docs/research/research_c_pr_aware_df_c2a_zsxx_contract_draft_20261027.json)。`c_pr_aware_df_zsxx_enclosure.py`・人工fixture・passive契約helper、`check_research_c_pr_aware_df_zsxx.py`、同名management test。Arb192で固定Rust出力との位相込み距離を囲う。新規52＋回帰109 PASS、旧順序依存1件deselect。全law/translation未資格、科学call上限null、実H4科学call0。証拠は`artifacts/research_c_pr_aware_df_zsxx/2026-10-27/technical/`。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF C2-A・whole-P0構造と未資格上界の技術準備

[固定ZSXX graph・条件付き回数制限・人工/回帰110 PASS](docs/research/research_c_pr_aware_df_c2a_synthesis_readiness_20261027.md)／[実行不可の未承認草案](docs/research/research_c_pr_aware_df_c2a_contract_draft_20261027.json)。新`c_pr_aware_df_c2a.py`／passive-only `check_research_c_pr_aware_df_c2a.py`／同名management test。Basis閉包等の前提下で採用rewrite≤N・採用matrix入力≤6Nを構成。Uniform ZSXX/initial translation/reference enclosureは未資格、科学call上限null・executable=false。SOURCE93/input23と固定Qiskit14SOURCEをpassive確認、新実H4科学call0、旧証拠保全、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C2-Bridge・47-call数値対応評価完了

[全gate・26round・未資格事項](docs/research/research_c_pr_aware_df_stage_c2_bridge_execution_20261027.md)。固定SOURCE3f64842d69d54a70a5b736486b5faefa398903b8を無変更で一度実行し47/47 COMPLETE、correctness FAIL/retry/resume0。保存直鎖H4・1Å/STO-3G、通常number CDF8 Baseline/Proposed、Nα=Nβ2/sector36、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0/m0..12、own-DF oracle/状態準備除外。両216 atom/K0 mean/保存CAR kernel gateと26round replayを確認。観測operator2差の最大はBaseline6.53148e-11（m11）/Proposed9.46816e-11（m12）、signal差の最大は5.98984e-11/8.10860e-11。数値対応pilotであり、厳密Bridge/reference enclosure・全law P0・PCG64 IID・CA/保証shots/totalは未資格/null。C1の約83%点推定不変、新fit/sampling/build-compile/DF-PF再計算/shots/総費用/C1再解析0。SOURCE82/input20・旧1799件不変、raw44と全26round CSVを保存、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C2・Reference Bridge統合準備

[対応関係・whole-P0の限界・人工/回帰85 PASS](docs/research/research_c_pr_aware_df_stage_c2_bridge_synthesis_readiness_20261027.md)／[未承認47-call Bridge契約](docs/research/research_c_pr_aware_df_stage_c2_bridge_contract_draft_20261027.json)。新`c_pr_aware_df_c2_bridge.py`／専用runner／management testは、既存general-K・CAR・K0 event・保存referenceへの接続とraw-signal誤差schemaを用意する。SOURCE82/input20のpassive確認のみで実H4科学call0。全law synthesis・reference enclosure・実装lawは未資格、保証shots/totalはnull、C1再解析なし。Bridge科学実行は別途明示許可後のみ。旧SOURCE/raw/STOP保全、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF C2-B・35-call確率表解析完了

[全gate・2表／26round・資格の限界](docs/research/research_c_pr_aware_df_stage_c2b_execution_20261027.md)。固定SOURCE3c1bb2d9c…を無変更で一度実行し35/35 COMPLETE、correctness FAIL/retry/resume0。保存H4・1Å/STO-3G、通常CDF8二arm、sector36、元順LD2、partial S2 δ.1/s4/r16/K0/m0..12、own-DF oracle／状態準備除外、共通Gaussian/P0。各216 atomの既存make_tail再生PASS。IID一様53-bit格子モデルのTVはBaseline3.6470735282e-15／Proposed3.2261762373e-15、m12条件付きraw信号上界1.9121168860e-9／1.6914454871e-9。PCG64 IID、CAR/atom/angle/normalization bridge、全law P0 synthesis、support、actual CA／保証shots／totalは未資格/null。新fit/sampling/build-compile/operator/shots・総費用・C1再解析0、SOURCE76/input12・旧1768ファイル保全。公開照合後mandatory STOP。以下は当時のscopeの履歴。


## 2026-10-27：PR-aware DF C2・全law資格の技術調査／未承認C2-B準備

[NumPy/P0 SOURCE・資格の限界・人工99 PASS](docs/research/research_c_pr_aware_df_stage_c2_qualification_readiness_20261027.md)／[未承認35-call表解析契約](docs/research/research_c_pr_aware_df_stage_c2_contract_draft_20261027.json)。`c_pr_aware_df_stage_c2.py`／静的`c_pr_aware_df_stage_c2_audit.py`／`run_research_c_pr_aware_df_stage_c2.py`／同名management test。Exact finite-grid TVと条件付きraw-law伝播、scoped synthesis／support入力、null C1-previewを人工検証。現whole-P0合成・PCG64 IID／CAR-atom-reference bridge・費用supportは未資格。SOURCE76/input12 passive PASS、新規実分子科学call0、C1再解析0。保存H4・CDF8二arm・元順LD2・δ.1/s4/r16/K0/m0..12・own-oracle/状態準備除外を維持。別科学許可後のみC2-B実行、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C1・仮定付きshots／serial総資源評価完了

[164-call結果・全78round・資格の限界](docs/research/research_c_pr_aware_df_stage_c1_execution_20261027.md)。固定SOURCE b1c735da3…を無変更で一度実行し164/164 COMPLETE、correctness FAIL/retry/resume0。保存H4・1Å/STO-3G、通常CDF8 Baseline/Proposed、sector36、元順LD2、partial S2 δ.1/s4/r16/K0/m0..12、own-DF oracle／状態準備除外、共通precision/P0／CA.00159／serial model。証拠経路26roundはUNRESOLVED・shots/total null。2つの明示仮定scenarioでは両armが条件付き十分条件内。Idealのshots53136/19574、総RZ点推定3,194,333,274,946/540,290,160,109（Proposed約83.086%低い）。.001Aをsynthesis/law各項へ仮定するscenarioも約83.089%低い。全law誤差・費用support・actual CA・保証shots/優位は未資格/null。SOURCE60/input408・既存1644ファイル不変、新fit/sampling/build-compile/量子shots0。C2は未実行・次指示待ち、公開照合後mandatory STOP。以下は以前のscopeの履歴。


## 2026-10-27：PR-aware DF Stage C・保存証拠からshots／serial資源への技術準備

[2経路・資格不足・人工73 PASS・C1/C2設計](docs/research/research_c_pr_aware_df_stage_c_readiness_20261027.md)／[未承認C1・164-call案](docs/research/research_c_pr_aware_df_stage_c1_contract_draft_20261027.json)。新`c_pr_aware_df_stage_c.py`／専用runner／人工test。Evidence laneは全law合成・実装law不足でnull、2明示scenarioで既存accuracy/Hoeffding/serial-costへ接続する案。SOURCE60/saved408のpassive照合PASS、旧1361ファイル不変。実H4新shots/総費用/再解析・新科学call0。C2はC1後の選択を要する非実行草案。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF B2・64 native kernel校正完了

[全gate・費用6指標・不確かさ・技術再実行](docs/research/research_c_pr_aware_df_stage_b2_execution_20261027.md)／[修正契約](docs/research/research_c_pr_aware_df_stage_b2_compat_contract_20261027.json)。保存H4・1Å/STO-3G、CDF8 Baseline/Proposed、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、own-DF oracle/状態準備除外、共通precision/P0。Run01は429 unitsでQiskit Already borrowed技術STOP（数値FAIL0）を保全し、共有build/compile/adjoint mutex・元64seedのfresh run02が652/652 COMPLETE／FAIL0、累積1081、技術再起動1・run02 retry/resume0。64 full512/adjoint/X-Yと67 compileがPASS。RZ標本mean±SEは5198.3125±66.7146対4765.875±80.3845（今回約8.3188%低い）。母平均保証・support・CA/shots/総資源は未資格。SOURCE50/input22/旧628不変、公開照合後mandatory STOP。

修正entryは`run_research_c_pr_aware_df_stage_b2_compat.py`、moduleは`c_pr_aware_df_stage_b2_compat.py`、testは同名management test。旧B2 runner/sourceは不変。

## 2026-10-27：PR-aware DF B2・native費用校正の技術準備

[共通native・full512・人工72 PASS・費用資格と制限](docs/research/research_c_pr_aware_df_stage_b2_preparation_20261027.md)／[未承認652-call契約](docs/research/research_c_pr_aware_df_stage_b2_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b2.py`／専用runner／人工test。32独立trajectory/arm、64 kernel＋3shell、67 P0 compile、全operator/adjoint/X-Y後にwhole-kernel metricsを記録する案。SOURCE47/input22のbyte/header preflight PASS、旧175ファイル不変。実H4 decode/sampling/build-compile/cost・新科学call0、CI/support/ideal費用/CA/総資源は未資格。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF B1・両armの13/13 roundが潜在適格

[全gate・26セル・信号・call・資格の限界](docs/research/research_c_pr_aware_df_stage_b1_execution_20261027.md)。固定SOURCE7da0fdb3f…無変更、141/141 COMPLETE、correctness FAIL/中断/retry/resume0。保存H4・1Å/STO-3G、sector36、Stage A最終CDF8 Baseline/Proposed、元順LD2、partial S2 δ.1/s4/r16/K0/m0..12、own-DF ground oracle・状態準備除外。両POTENTIALLY_FEASIBLE。m12のAは.209598/.513867、補正RTE誤差.164305/.125612、PF.00447695/.00447625。newfit/DF norm/native/build-compile/sampling/shots/cost0、全law合成・実装law・CA・総資源未資格。旧証拠保全・公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF Stage Bの接続・人工検証準備

[2 CDF arm／law／sector reference／共通native接続・人工56 PASS](docs/research/research_c_pr_aware_df_stage_b_readiness_20261027.md)／[未承認B1-only141-call案](docs/research/research_c_pr_aware_df_stage_b_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b.py`／`stage_b_runner.py`／同名runner・management test。保存Stage A J/Uの新評価は0、hash/header/JSON preflightのみPASS、旧97ファイル不変。B1は2arm×13roundのPF/RTE/A/signalをsector36で検証し、build/compile/sampling/shots/costは0の案。B2は人工native部品まで、実行範囲は別固定。科学call0、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF Stage A V2・物理予算とexact tail mass基準がPASS

[全gate・paired fit・gradient・科学call・限界](docs/research/research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)。固定SOURCEe1085c95f…無変更、30/30 COMPLETE、correctness FAIL0、各1 fit計2、retry/resume0（旧18含む累積48）。保存H4・1Å/STO-3G、sector36、通常CDF8、元順prefix LD2、共通初期値、γ0/.1。両終了DF norm＋guard1.02976e−8/1.22956e−8 Haで予算内、exact tail mass2.20976184/1.44242989 HaでProposed34.7246%削減、STAGE_A_PROXY_AND_DF_PASS。相対目的減少停止でgradient収束・PR費用・最終CAは未資格、Stage B/Cなし。旧証拠保全・公開照合後mandatory STOP。

## 2026-10-27：PR-aware DFのx0接続を技術修正

[単一qualified x0引渡し・人工61 tests＋29 subtests](docs/research/research_c_pr_aware_df_x0_repair_20261027.md)／[新V2・未承認30-call契約](docs/research/research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json)。既存corrected-codecへopt-in prepared初期値を接続し、fit内の再logを除去。旧default・目的／gradient／optimizer条件は維持。実分子decode・新科学call0、旧18-call STOPを保全、新2026-10-27/run01未作成。別明示許可までmandatory STOP。

## 2026-10-27：PR-aware DF Stage A・optimizer前の技術的STOP

[初期予算・18-call部分証拠・x0接続](docs/research/research_c_pr_aware_df_stage_a_execution_20261027.md)。固定SOURCE94d1c7b4b…無変更、18/30 units（16 COMPLETE／2 FAIL）、ffsim entry各1・実optimizer0・完了fit0・retry/resume0。保存H4・1Å/STO-3G、sector36、CDF8元順prefix LD2、γ0/.1。初期general-K/raw-CARとDF予算はPASS、再pack後の`QUALIFIED_X0_CHANGED`で両arm STOP。終了DF・exact mass削減・PR資源効果は未資格。旧SOURCE/raw/STOPを保全し、公開照合後mandatory STOP。以下の技術準備状態は履歴。

## 研究C：PR-aware DFの初期設計・技術準備（依頼識別子20261026）

[設計・再利用・人工49 tests＋29 subtests](<docs/research/research_c_pr_aware_df_optimization_design_20261026.md>)／[未承認Stage A・30-call案](<docs/research/research_c_pr_aware_df_pilot_contract_draft_20261026.json>)。`c_pr_aware_df.py`／`c_pr_aware_df_pilot.py`、`run_research_c_pr_aware_df.py`、同名management tests。既存ffsim loss/gradient・修正codec・SectorCARへ固定tail mass代理目的を最小接続。実optimizerは人工mock nit0、両env byte preflight PASS。実分子decode・新科学call0、科学的有効性／新規性・Stage B/Cは未資格・未承認、mandatory STOP。

## C1：H4 XDF9費用screen・両sでstrict改善を除外（2026-10-26）

[全gate・metrics・shots・比較・制約](docs/research/research_c_c1_h4_xdf9_cost_screen_execution_20261026.md)／[raw48・QPY4・83-call journal・STOP・CSV](artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/)。固定SOURCE7328ecc9a…無変更、83/83 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å/STO-3G、Nα=Nβ2、元順XDF9、全frame S2 s1/s4、δ.1/m0..12、own-ground/preparation excluded、同precision/P0/shared/fenced serial。kernel RZ15720/各s。楽観的総RZ s1=144805110560/s4=530107748679はいずれも保存XDF8/CDF8より大きく、固定十分処方・実装・数値資格内でstrict改善を除外。full operator/actual合成誤差/CA/保証付きshots/最終費用は未資格・未評価。旧結果保全、公開照合後mandatory STOP。

## C1：H4 XDF9 native／楽観的shots screenの技術準備（2026-10-26）

[最小接続・単調性／ceil・限界・full512後続案](docs/research/research_c_c1_h4_xdf9_cost_screen_preparation_20261026.md)／[未承認83-call契約](docs/research/research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)／[SOURCE・人工／回帰97 PASS・passive証拠](artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/)。新entry `run_research_c_c1_h4_xdf9_cost_screen.py`／module `c1_h4_xdf9_cost_screen.py`／同名management test。既存PrecisionBuilder/P0/solver/serial/rational楽観集計を再利用、XDF9 S2 s1/s4の2 kernelとPF26から固定処方の楽観的RZ下限を接続。実H4のdecode/build-compile/shots/cost・科学call0。下限は固定処方／serial実装内の数値screen、full operator／strict enclosure／actual CA・shots・資源優位ではない。SOURCE56/input24 byte preflight PASS、旧証拠不変、別明示許可までmandatory STOP。

## C1：H4 XDF9・表現／ground-alias／PF26がPASS（2026-10-26）

[全gate・全26 PF値・科学call・資格の限界](docs/research/research_c_c1_h4_xdf9_pf_execution_20261026.md)／[raw45・SOURCE・journal・STOP・CSV](artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/)。固定SOURCE9a4cda44a…無変更、95/95 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、Nα=Nβ2/sector36、保存OpenFermion元順prefix9、元g補正κ・核scalar・frame順、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-XDF9 oracle/状態準備除外。保存DF予算PASSを再利用、表現10因子・sum・own-ground/alias PASS、両sとも13/13 PASSのPOTENTIALLY_FEASIBLE。最大PF誤差s1 .06843041535127227/s4 .004277093283650178。新fit/DF norm/native/合成/CA/shots/総費用0、27 protected filesと原packet不変。旧約27%／約12%比較を保全し、公開照合後mandatory STOP。以下は過去の準備・結果の履歴。

## C1：H2／旧H4／新H4／H6統合と次期XDF9準備（2026-10-26）

[保存16候補・baseline差分・SOURCE来歴](docs/research/research_c_c1_h2_h4_h6_integration_update_20261026.md)／[A:H4 XDF9・B:H6 CDF13の比較と未採択推奨](docs/research/research_c_c1_next_minimal_experiment_selection_20261026.md)／[人工78 tests＋29 subtests・技術準備](docs/research/research_c_c1_h4_xdf9_preparation_20261026.md)。H4約27%→約12%をrank単独の因果効果とせず、3系をサイズscaling実験と扱わない。新entry `run_research_c_c1_h4_xdf9_pf.py`／module `c1_h4_xdf9_pf.py`／同名management testは、既存CAR／一般K／PFを再利用する[未承認95-call契約](docs/research/research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)の準備。original input／SOURCE24／保存JSON3のbyte preflight PASS、実分子decode／新科学call0。旧証拠保全、推奨は未採択、公開照合後mandatory STOP。後続native／CA／shots／費用は別範囲。 [CSV・来歴](artifacts/research_c_c1_integration_update/2026-10-26/)／[SOURCE・人工log・passive証拠](artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/)。

## C1：H4 CDF8/XDF8・共通CA/条件付き総資源pilot完了（2026-10-26）

[結果・全gate・shots/energy/総費用](docs/research/research_c_c1_h4_ca_resources_execution_20261026.md)／[raw19・52round/104軸CSV・SOURCE/calls/STOP](artifacts/research_c_c1_h4_ca_resources/2026-10-26/)。固定SOURCE18045333…無変更、119/119 COMPLETE、FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。4候補の共通CA=.00159 Ha十分条件が数値PASS。CDFの条件付き総RZ削減は同sでs1 11.853846%／s4 11.990819%。保存level B・ideal IID Born・固定serial native modelのpilotで、厳密CA・無条件優位ではない。新fit/DF-PF-合成誤差再計算/build-compile/量子shots抽選0、旧証拠保全・mandatory STOP。以下は実行前準備などの履歴。

## C1：H4 CDF8/XDF8・CA/RPE shots/serial資源の技術準備（2026-10-26）

[既存再利用・人工/回帰88 PASS・資格の限界](docs/research/research_c_c1_h4_ca_resources_preparation_20261026.md)／[未承認119-call契約](docs/research/research_c_c1_h4_ca_resources_contract_unapproved_20261026.json)。新`c1_h4_ca_resources.py`／同名runner・testが、保存52 full512 profile・各DF norm＋guard・共通precision/P0 metricsから既存RPE/Hoeffding/energy/serial費用を接続。直鎖H4・1Å・STO-3G、CDF8/保存順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。科学call0、SOURCE24/input12のbyte preflight PASS。新shots/CA/総費用は未評価、厳密enclosure・無条件優位は未資格。旧証拠保全・別明示科学許可までmandatory STOP。

## C1：H4 CDF8/XDF8共通precision・full512限定実行完了（2026-10-26）

[結果・全gate・7 metrics・52 profile](docs/research/research_c_c1_h4_native_fair_execution_20261026.md)／[SOURCE・QPY40・call journal・STOP](artifacts/research_c_c1_h4_native_fair/2026-10-26/run01/)。SOURCE862a97d…を無変更実行し305/305 COMPLETE、correctness FAIL/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。同H6 Gaussian/P0・既存shared routeで局所18/kernel4/shell3/adjoint8/X-YがPASS、full512数値Frobeniusによる52/52位相条件内。単一h-step RZはCDF12,272対XDF13,944（約11.99%減）。丸めenclosure・共通CA・shots・総資源は未資格/未評価。旧SOURCE/raw/STOPを保全しmandatory STOP。以下の技術準備欄は実行前時点の記録。

## C1：H4 CDF8/XDF8共通precision・full512の技術準備（2026-10-26）

[再利用・人工156 PASS・数値資格の限界](docs/research/research_c_c1_h4_native_fair_preparation_20261026.md)／[未承認305-call契約](docs/research/research_c_c1_h4_native_fair_contract_unapproved_20261026.json)。新SOURCE `c1_h4_native_fair.py`／`c1_h4_full_synthesis.py`、同名runner/test、[technical証拠](artifacts/research_c_c1_h4_native_fair/2026-10-26/technical/)。両arm共通H6 Gaussian/P0、4 kernel+3 shell、full512 Frobeniusと52roundへの接続を準備。実H4 decode/build/compile/新科学0、旧CDF8/XDF10結果不変・mandatory STOP。

## C1：H4 XDF8・表現/ground/alias/PF26セルがPASS（2026-10-26）

[結果・全26値・資格の限界](docs/research/research_c_c1_h4_xdf8_pf_execution_20261026.md)／[原本45・93 calls・STOP](artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/)。固定SOURCE328f31007…無変更、93/93 COMPLETE・correctness FAIL/retry0。元順XDF8、S2 s1/s4とも13/13 PASS・POTENTIALLY_FEASIBLE。新fit・DF norm再計算・CDF8/XDF10新評価・native/合成/CA/shots/費用0。旧結果保全・mandatory STOP。以下は準備時点の履歴。

## C1：H4 XDF8を追加採用・第1段階の技術準備（2026-10-26）

[表現・own-ground/alias・PF接続と公平な次段階設計](docs/research/research_c_c1_h4_xdf8_requalification_preparation_20261026.md)／[未承認93-call契約](docs/research/research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json)。SOURCE `c1_h4_xdf8_pf.py`／同名runner・人工test、[技術証拠](artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/technical/)。新22＋回帰92 PASS、実H4 decode・新科学0。保存順prefix8を追加候補に採用し、旧CDF8/XDF10結果は保持。CDF8/XDF8への共通precision/P0は推奨案、第2・3段階は未採択・未実行。mandatory STOP。

## C1：保存H4 XDFのDF予算first-passingはrank8（2026-10-26）

[全9prefix・gate・SOURCE・制約](docs/research/research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)／[原本・CSV・356 calls・STOP](artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/)。SOURCE f7bc730d1…無変更、rank1..7 NOT_MET・8/9 PASS、correctness FAIL/retry0。rank10は保存値のみ、PF／native／CA／shots／総費用の新評価なし。旧費用結果を保持し、新XDF8 baseline採用は次判断。mandatory STOP。

## C1：H4保存XDF prefixのDF予算評価準備（2026-10-26）

[接続・人工検証・制約](docs/research/research_c_c1_h4_xdf_prefix_budget_preparation_20261026.md)／[未承認356-call契約](docs/research/research_c_c1_h4_xdf_prefix_budget_contract_unapproved_20261026.json)。SOURCE `c1_h4_xdf_prefix_budget.py`、同名runner／管理test、証拠 `artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/technical/`。案A採択・新科学0、57関連test PASS。旧full256回帰1件の環境異常を報告し、当該旧経路は新sector36評価に使用しない。別科学許可までmandatory STOP。

## C1：H6共通CA・shots・条件付き資源pilot完了（2026-10-26）

[4候補CA数値PASS・shots・総RZ](docs/research/research_c_c1_h6_ca_resources_execution_20261026.md)／[原本・104軸・119 calls・STOP](artifacts/research_c_c1_h6_ca_resources/2026-10-26/)。SOURCE55b3d868…無変更、全52round PASS、FAIL/retry/resume0。同sのshotsはCDF/XDF同じで、CDF総RZはXDFよりs1+1.978607%／s4+1.962935%。資格B／ideal IID Born／oracle・状態準備除外／serialモデルの数値pilot、厳密・無条件保証ではない。旧証拠872件不変・mandatory STOP。

## C1：H6共通CA・RPE shots・総資源の技術準備（2026-10-26）

[再利用・誤差接続・102人工/回帰PASS](docs/research/research_c_c1_h6_ca_resources_preparation_20261026.md)／[未承認119-call契約](docs/research/research_c_c1_h6_ca_resources_contract_unapproved_20261026.json)。新 `c1_h6_ca_resources.py`／専用runner／人工test／[technical証拠](artifacts/research_c_c1_h6_ca_resources/2026-10-26/technical/)。既存H2/H4 RPE・Hoeffding・energy・serial費用へ保存H6資格Bを接続。実H6のjoin/shots/CA/cost0、旧結果不変・mandatory STOP。

## C1：H6全基底合成上界・限定実行完了（2026-10-26）

[結果・8経路上界・104位相条件PASS](docs/research/research_c_c1_h6_synthesis_bound_execution_20261026.md)／[3200列・104行・call・STOP](artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/)。SOURCE e36e056…無変更、exact IR共用G4、3335 calls・correctness FAIL0・retry/resume0。入力800／出力8192の資格Bを取得、厳密C・full全入力・CA/shots/cost未資格。旧証拠849件不変、mandatory STOP。以下は準備・過去時点の履歴。

## C1：H6合成誤差上界の技術準備（2026-10-25）

[方式比較・導出・人工134 PASS](docs/research/research_c_c1_h6_synthesis_bound_preparation_20261025.md)／[未承認6535-call契約](docs/research/research_c_c1_h6_synthesis_bound_contract_unapproved_20261025.json)。新SOURCE `c1_subspace_synthesis_bound.py`／`c1_h6_synthesis_bound.py`、専用runner・人工tests・[技術証拠](artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/technical/)。全800入力×full8192出力のstreaming数値上界を準備。厳密enclosure・CA・shots・総費用は未資格、実H6追加科学0、mandatory STOP。

## C1：H6 native精度修正後・298 calls全PASS（2026-10-25）

[結果・8経路の費用表](docs/research/research_c_c1_h6_native_stable_phase_execution_20261025.md)／[raw・QPY52・104 profile・call・STOP](artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/)。SOURCE3d7b56b448…無変更、correctness FAIL0。局所30／kernel8／shell3／adjoint8／X-Y8 PASS。H6 CDF14はXDF14よりkernel RZが約1.98%/1.96%多く、literal/shared差0。Gaussian小角・P0 phase集計を修正しallowance維持。旧2 FAIL＋診断を含む累積627 units、run内retry0、global合成誤差/CA/shots/総費用未資格・mandatory STOP。以下は準備・旧結果の履歴。

## C1：H6 P0位相集計の安定化（2026-10-25）

[修正・関連97 PASS](docs/research/research_c_c1_h6_native_stable_phase_repair_20261025.md)／[同一8経路298-call契約](docs/research/research_c_c1_h6_native_stable_phase_contract_20261025.json)。既存P0のgate本体を変えずdeclared phaseだけ160-bitで集計。旧2回の162-call FAILと独立診断5-callを保持し、累積最大627 units。

## C1：H6 nativeのGaussian精度修正・fresh再実行（2026-10-25）

[修正・人工185 PASS](docs/research/research_c_c1_h6_native_precision_repair_20261025.md)／[新298-call契約](docs/research/research_c_c1_h6_native_precision_contract_20261025.json)。H6専用の既存Nature JW数値修正（exact-zero/LAPACK/atan2）とP0/inverse生成mutex。旧run01・162-call STOPは保持。同一8経路・allowance不変、別fresh outputで最大298、累積最大460 units。

## C1：H6 nativeのcorrectness FAIL・部分証拠（2026-10-25）

[実行結果](docs/research/research_c_c1_h6_native_execution_20261025.md)／[raw・QPY52・call・STOP](artifacts/research_c_c1_h6_native/2026-10-25/)。SOURCE bf431765…無変更、162/298 units（154 COMPLETE／8 FAIL）。局所30・ancilla shell3 PASS、最初のCDF_s1_shared raw/reference差1.0400731857e-10が固定1e-10を超過。8 kernel bundleはFAIL、adjoint/X-Y/kernel費用/104 profileは未到達。原因未確定・合成誤差/CA/資源未資格、retry/resume0、旧証拠保全・mandatory STOP。以下の準備・未実行表示は履歴。

## C1：H6 native正しさ・kernel費用の技術準備（2026-10-25）

[技術報告](docs/research/research_c_c1_h6_native_preparation_20261025.md)／[未承認298-call契約](docs/research/research_c_c1_h6_native_contract_unapproved_20261025.json)／[SOURCE・人工証拠](artifacts/research_c_c1_h6_native/2026-10-25/technical/)。既存NativeBuilder/一般K/P0へ保存CDF14/XDF14を接続。full8192 state-actionの有限probeと局所CAR、literal/sharedの8 kernels、11 compile。117 tests＋29 subtests PASS、実H6新科学0、global合成誤差/CA/shots/総費用未資格、mandatory STOP。

## C1：H6 rank14・4候補52 PF roundがPASS（2026-10-25）

[結果](docs/research/research_c_c1_h6_saved_pf_execution_20261025.md)／[全52 round・原本・call・STOP](artifacts/research_c_c1_h6_saved_pf/2026-10-25/)。SOURCE87ea924ae…無変更、72/72 units・correctness FAIL0。一般K/ground/alias全PASS、CDF14/XDF14のS2 s1/s4すべてPOTENTIALLY_FEASIBLE。native/CA/shots/costは未資格・未評価、fit/DF誤差再計算/retry0、旧証拠保全・mandatory STOP。以下は準備・旧結果の履歴。

## C1：保存H6 rank14の一般K・ground/alias・S2 PF準備（2026-10-25）

[技術報告](docs/research/research_c_c1_h6_saved_pf_preparation_20261025.md)／[未承認72-call契約](docs/research/research_c_c1_h6_saved_pf_contract_unapproved_20261025.json)／[SOURCE・人工証拠](artifacts/research_c_c1_h6_saved_pf/2026-10-25/technical/)。`c1_h6_saved_pf.py`＋専用CLI・人工test。保存CDF14/XDF14、一般K2・ground/alias3・S2 s1/s4の52セルへ接続。新21＋回帰64 tests・29 subtests PASS。実H6 decode/新科学0、fit/DFnorm再計算/native/shots/cost除外、mandatory STOP。

## C1：H6 rank14修正codecの1 fit・DF予算PASS（2026-10-25）

[結果](docs/research/research_c_c1_h6_corrected_fit_execution_20261025.md)／[原本・生x/jac・4状態・call・STOP](artifacts/research_c_c1_h6_corrected_fit/2026-10-25/)。SOURCE33432e786…無変更、20/20 calls、fit/optimizer1、retry/resume0。optimizer成功・最終DF予算PASS、sector1.0689213237e-4＋guard1.1384527737e-6 Ha。旧CDF14比93.9132%減、初期XDF比sector4.3116%増。PF/native/shots/cost未実施、厳密CA未資格、旧結果保全・mandatory STOP。以下は準備・旧結果の履歴。

## C1：H6修正codecのffsim接続・rank14 1 fit準備（2026-10-25）

[技術報告](docs/research/research_c_c1_h6_corrected_fit_preparation_20261025.md)／[未承認20-call契約](docs/research/research_c_c1_h6_corrected_fit_contract_unapproved_20261025.json)／[人工証拠・SOURCE](artifacts/research_c_c1_h6_corrected_fit/2026-10-25/technical/)。新 `c1_ffsim_corrected_codec.py`＋`c1_h6_corrected_fit.py`、専用CLI・人工tests。保存XDF14→修正pack→実ffsim x0を接続、loss/gradient/L-BFGS-Bは維持。新27＋回帰134 tests・29 subtests PASS。実H6 decode／fit／再解析0、今後最大1 fit・retry0、mandatory STOP。

## C1：保存H6の修正codec・DF予算保持PASS（2026-10-25）

[結果](docs/research/research_c_c1_h6_codec_roundtrip_execution_20261025.md)／[4状態・gate・原本](artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/)。SOURCE43941ed8…無変更、rank14/18のJ/projector/tensor/sector同値性とCA/8保持がPASS。20/20 calls、fit/optimizer/retry/resume0、人工54 PASS。frame9も数値allowance内で復元。修正fit・PF・資源比較は未実施、旧証拠保全・mandatory STOP。以下の準備表示は履歴。

## C1：H6修正codec・20-call評価準備（2026-10-25）

[方式・人工検証](docs/research/research_c_c1_h6_codec_roundtrip_preparation_20261025.md)／[未承認契約](docs/research/research_c_c1_h6_codec_roundtrip_contract_unapproved_20261025.json)／[SOURCE・人工証拠](artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/technical/)。新 `c1_real_orthogonal_codec.py`＋`c1_h6_codec_roundtrip.py`、専用CLIと人工tests。列符号＋real Schurのπ平面でffsim形式へ接続し、J/projector/tensorをgate化。新54＋回帰80 tests・29 subtests PASS。保存H6新評価0、将来はrank14/18・before/afterの20 calls、fit0、mandatory STOP。

## C1：保存H6の初期codec劣化を再現（2026-10-25）

[技術・限定再解析報告](docs/research/research_c_c1_h6_initialization_execution_20261025.md)／[6状態・frame・gradient・STOP](artifacts/research_c_c1_h6_initialization/2026-10-25/)。module `c1_h6_initialization.py`／専用runner `run_research_c_c1_h6_initialization.py`／人工test同名。SOURCE14cf3558…、rank14/18の初期J/Uは保存XDFと完全一致、旧real codec直後に両方の誤差が悪化。35/35 calls、新fit・optimizer・retry/resume0、人工40 PASS。修正は人工のみ、最終未達の唯一原因・修正fit性能は未確定、旧結果保全・mandatory STOP。

## C1：H6はCDFのDF予算未達でSTOP（2026-10-25）

[結果](docs/research/research_c_c1_h6_execution_20261025.md)／[要約](artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_execution_summary.json)／[原本](artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/)。SOURCE33c380d8…無変更、XDF R14は予算PASS、CDF requested1..21は未達（最小sector norm4.6795e-4 Ha＋guard、actual18）。61/129 calls、fit21・retry/resume0、一般K／ground／PF未実行。回路・shots・費用なし、旧H2/H4証拠保全・mandatory STOP。以下の未実行・準備表示は履歴。

## C1：H2の限定CA数値pilot完了（2026-10-25）

[結果](docs/research/research_c_c1_h2_execution_20261025.md)／[証拠](artifacts/research_c_c1_h2_h6_science/2026-10-25/)。SOURCE33c380d8…無変更、CDF R2／XDF R3、S2 s1/s4の4候補・52round数値CA PASS。193/196 calls、fit2・retry/resume0、条件付き総RZ削減36.4122%／36.1527%。厳密CAは未資格、H6未実行、mandatory STOP。以下の準備表示は履歴。

## C1：H2・H6の決定論S2検証準備（2026-10-25）

[技術報告](docs/research/research_c_c1_h2_h6_preparation_20261025.md)／[未承認実行案](docs/research/research_c_c1_h2_h6_contract_unapproved_20261025.json)。H2は新h/g→rank選定→native/P0→数値CA・shots・総費用、H6はsector400のDF／PFだけ。新module `c1_molecule_extension.py`／CLI `run_research_c_c1_h2_h6.py`／人工test `test_research_c_c1_h2_h6.py`。CDF rank ladder1..3/21とCA/8・engineering guard・explicit-XDF方式は未承認案。上限H2 196／H6 129 calls、実科学0。人工14＋回帰166 tests・29 subtests PASS、旧H4証拠保全・mandatory STOP。

## C1 Stage3：決定論S2の固定N・CA数値pilotが全PASS（2026-10-25）

[結果・4候補・資格の限界](docs/research/research_c_c1_stage3_deterministic_accuracy_execution_20261025.md)／[52round・call・STOP](artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/)。SOURCE002d0dc2…で110/110 calls、4候補／52roundが全PASS、固定保存shotsで共通CAの数値条件を満たし、s1/s4ともCDFの条件付き総RZがXDFより約27%小さい。rigorous CAはreference enclosure不足で未資格。fit／retry／resume／新build・compile0、旧証拠保全・mandatory STOP。以下は過去時点の記録。

## C1 Stage3：決定論S2の固定shots精度資格付け準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage3_deterministic_accuracy_preparation_20261025.md)／[未承認110-call契約](docs/research/research_c_c1_stage3_deterministic_accuracy_contract_unapproved_20261025.json)。CDF R8／XDF R10のD S2 s1主対象・s4対照、保存PF・DF・shots・費用を再利用し13roundの数値pilot CA判定へ接続。s1保存Frobenius・s4保存compiler残差を再利用、将来の新規local reference2・QPY8のみ。module `c1_stage3_deterministic_accuracy.py`／同名runner・人工test。新12＋独立process回帰140 tests・29 subtests PASS、SOURCE・保存input byte preflight PASS、実H4科学call0。厳密CAはreference enclosure不足で未資格、mandatory STOP。

## C1 Stage3：修正後8候補cost-onlyが完了（2026-10-25）

[結果・条件付き費用・制約](docs/research/research_c_c1_stage3_cost_compat_execution_20261025.md)／[8候補・208行・12比較・call・STOP](artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/)。SOURCE9ce6703c…でcompile競合再発0、936/936 calls、compile／位相・unitarity各133 PASS。人工100 PASS、旧STOP106 calls保全・fresh再実行1・累積1042。PR4候補の費用support／合成・law／CAは未資格、保証付き優位・最終順位なし、mandatory STOP。以下は過去時点の記録。

## C1 Stage3：Qiskit compile競合修正・限定再実行（2026-10-25）

[修正・人工再現](docs/research/research_c_c1_stage3_cost_compat_repair_20261025.md)／[新契約](docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json)。旧SOURCEを保持し、`c1_stage3_cost_compat.py`がP0 compileだけをtask内で排他化。人工100 tests PASS、大きい人工8回路で旧3例外→修正後0。ユーザーが同8候補・936-call条件で修正再実行を許可。entry `scripts/run_research_c_c1_stage3_cost_compat.py`、test `tests/management/test_research_c_c1_stage3_cost_compat.py`。結果取得後mandatory STOP。

## C1 Stage3：cost-onlyの技術的STOP（2026-10-25）

[例外・到達範囲・証拠](docs/research/research_c_c1_stage3_cost_execution_20261025.md)／[run01原本・partial CSV・STOP](artifacts/research_c_c1_stage3_cost_science/2026-10-25/)。固定SOURCE edb9f6ef…の8候補実行はQiskit1.3並列compile `Already mutably borrowed`でSTOP。106/936 calls、9 compile attempt／7完了／2例外、readout3チェック、部分費用34行・QPY16。総費用・比較未取得、科学的NO-GOではない。retry/resume0、旧SOURCE保全、mandatory STOP。以下は準備履歴。

## C1 Stage3：8候補限定cost-only準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage3_cost_preparation_20261025.md)／[未承認936-call契約](docs/research/research_c_c1_stage3_cost_contract_unapproved_20261025.json)。専用module `c1_stage3_cost_only.py`／CLI `scripts/run_research_c_c1_stage3_cost_only.py`／人工test `tests/management/test_research_c_c1_stage3_cost_only.py`。CDF/XDF各D s1/s4とPR s4/r16/K0/K2の8候補、保存楽観shots・native校正・serial期待費用を接続。新13＋既存81 tests PASS、実H4 decode／回路／保存値総費用再解析0、CA・保証付き資源優位なし・mandatory STOP。

## C1 Stage3：楽観的shots限定再解析（2026-10-25）

[結果・20候補表](docs/research/research_c_c1_stage3_shots_execution_20261025.md)／[260round・520軸・call・STOP](artifacts/research_c_c1_stage3_shots_science/2026-10-25/)。公開SOURCE f8fd2d373…無変更、282/282 calls、14合計／前段不適格6はnull。PR s4/r16/K0はCDF58,356／XDF21,408 shotsの仮定付き十分式算定値。未知実装誤差・CA・総費用・資源順位は未資格、fit／retry／resume0、mandatory STOP。以下は準備時点の履歴。

## C1 Stage3：楽観的shots下界スクリーニング準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage3_shots_preparation_20261025.md)／[未承認282-call契約](docs/research/research_c_c1_stage3_shots_contract_unapproved_20261025.json)。新module `c1_stage3_shots_screen.py`、CLI `scripts/run_research_c_c1_stage3_shots_screen.py`、人工test `tests/management/test_research_c_c1_stage3_shots_screen.py`。保存PF/RTE/Aを保った仮定付き十分shots式で、全20候補・13round・X/Yを保持し、潜在適格14を主対象とする。巨大整数は文字列、未知実装誤差・CAは未資格。人工21＋既存60 PASS、保存値再解析0・科学実行未承認・mandatory STOP。

## C1 Stage3：feasibility-only限定科学実行（2026-10-25）

[結果・候補別値・制約](docs/research/research_c_c1_stage3_feasibility_execution_20261025.md)／[260セルCSV・20候補・call・STOP](artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/)。固定SOURCE6ea5c6a9…無変更で639/639 calls、潜在適格14／十分条件不適格6、correctness FAIL0、fit／retry／resume0。回路・shots・総費用・順位付けは未実施、CA達成を意味しない。mandatory STOP。以下の未承認・準備表示は履歴。

## C1 Stage3：feasibility-only限定経路の準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage3_feasibility_preparation_20261025.md)／[未承認639-call契約](docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json)。専用entry `scripts/run_research_c_c1_stage3_feasibility.py --mode feasibility-only`、処理 `c1_stage3_feasibility.py`、人工test `tests/management/test_research_c_c1_stage3_feasibility.py`。旧SOURCE無変更で20候補×13roundのPF／補正RTE誤差・attenuationと分類だけを接続。回路／compile／shots／総費用／順位付けは除外。人工16＋既存44 tests PASS。実H4科学call0・mandatory STOP。


## C1 Stage3：統合実行経路の完成（2026-10-25）

[統合技術報告](docs/research/research_c_c1_stage3_integrated_preparation_20261025.md)／[新未承認契約](docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json)。entry `scripts/run_research_c_c1_stage3_integrated.py` は保存H4からground／alias、20×13round、native費用、evidence評価、RPE診断、条件付き比較まで接続する。`c1_stage3_integrated/molecular/qualification/native_bridge.py`、人工19＋既存65 tests／29 subtests PASS。core pilot採用、実H4科学call0、全lawの有用な合成誤差上界はUNRESOLVED。旧SOURCE保全、remote照合後mandatory STOP。


## C1 Stage3：共通CA総資源比較の準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage3_preparation_20261025.md)／[未承認科学契約](docs/research/research_c_c1_stage3_contract_unapproved_20261025.json)。既存native一般K・RTE・RPE・compilerを再利用し、`c1_stage3_connection/accuracy/reference.py`、`scripts/run_research_c_c1_stage3.py`、人工testを追加。新規25＋既存40 tests・29 subtests PASS。CA初回pilot、状態準備除外・oracle／law／合成誤差／費用supportの未決定条件を明記。科学call0、旧証拠保全、mandatory STOP。


## C1 Stage2限定科学実行：native接続PASS（2026-10-25）

[結果・全gate・native費用・STOP](docs/research/research_c_c1_stage2_native_execution_20261025.md)。公開SOURCE `a977a4390668bc107360c8ff05f43d851dc81dc9` を無変更で使用し、局所40／raw full4／actual adjoint4／compiled full4／X/Y8組の60主検査とunitarity52件が全PASS。P-H4・CDF rank8／XDF rank10、T1/10 ordered S2・P0 profile。科学call243/243、fit／optimizer／retry／resume0、transpile12・QPY68。CDF fullのRZ12118／CX7352／depth12724、XDF16659／9520／16215。物理DF精度が異なり同精度総資源優位とはしない。証拠 `artifacts/research_c_c1_stage2_science/2026-10-25/`、Stage3未承認・mandatory STOP。下記の準備時点の未承認表示は履歴。

## C1 Stage2 native接続の技術準備（2026-10-25）

[技術報告・再利用・人工検証](docs/research/research_c_c1_stage2_native_preparation_20261025.md)、[未承認科学契約](docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json)。ユーザーがCDF rank8を限定対象に採用。新SOURCE `c1_stage2_native.py`／`c1_stage2_validation.py`／`c1_stage2_execution.py`、entry `scripts/run_research_c_c1_stage2_native.py`、test `tests/management/test_research_c_c1_stage2_native.py`。既存Gaussian・中央RZ/RZZ・phase・controlled／X/Y／compilerを接続し、人工80件＋29 subtests PASS。SOURCE・52 pins・証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`。時間／S2／compiler／allowance／最大243 unitsは未採択の推奨案。実H4 decode・実回路build/compile・科学call0、別科学承認待ち・mandatory STOP。同精度総資源優位の主張なし。

## C1 保存fit評価専用科学実行：全gate PASS（2026-10-25）

[実行結果・全gate・誤差・STOP](docs/research/research_c_c1_stage1_eval_only_execution_20261025.md)。公開SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0` の既存評価専用entryを無変更で実行した。P-H4・sector36/full256、保存C-DF rank8／XDF rank10で37 gate PASS、以前不一致のframe0も今回PASS。物理sector誤差はCDF `7.666027761392501e-5`／XDF `3.3751558354542566e-15` Hartreeの別測定。科学call141/141、新fit／optimizer／retry／resume0。原本 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`、人工19件PASS。旧FAIL保全、Stage2適格性は別判断、mandatory STOP。下記の準備時点の未承認表示は履歴。

## C1 保存fit評価専用経路の準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md)、[未承認科学契約](docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)。新entry `scripts/run_research_c_c1_stage1_eval_only.py`／`src/trottertracks/c1_stage1_eval_only.py` は旧fitのSOURCE/input/exportと新評価SOURCEを独立に保持し、新fit0で既存CAR/K/XDF検証を再利用する。新test `tests/management/test_research_c_c1_stage1_eval_only.py`、人工82件＋29 subtests PASS。証拠・SOURCE `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`、call上限141、retry/resume0。実H4 decode・評価・fit0、旧科学的STOP保全、別明示承認待ち・mandatory STOP。

## C1 raw CAR frame修正・人工検証（2026-10-25）

[修正報告](docs/research/research_c_c1_frame_repair_20261025.md)、[未承認引継ぎ](docs/research/research_c_c1_frame_repair_handoff_unapproved_20261025.json)。`src/trottertracks/c1_stage1.py` のraw縮約をexplicit batched matmulへ変更し、一般K／物理モデル／solver／allowanceは維持。新test `tests/management/test_research_c_c1_frame_repair.py` と既存回帰63件＋29 subtests PASS、別プロセス64人工frame比較もPASS。証拠・SOURCE `artifacts/research_c_c1_frame_repair/2026-10-25/`。分子exportの数値再解析・新fitは0。旧科学的FAILとの因果帰属・H4資格は未確認、旧STOP保全・mandatory STOP。現行one-fit runnerは評価専用入口ではない。

## C1 Stage 1互換性修正版の科学的STOP（2026-10-25）

[実行結果・gate・部分証拠](docs/research/research_c_c1_stage1_compat_execution_20261025.md)。公開SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`、新approval／fresh outputで1 fit収束・実R8。κ/J/Uとfull baseはPASS、full256 frame0のraw CAR／K同値性がFAILしmandatory STOP。物理CDF/XDF sector norm・sector adapterは未到達。科学call107/147、retry/resume0。原本は `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`、既存runner `scripts/run_research_c_c1_stage1.py` とSOURCEは無変更。追加計算・Stage2/3なし。

## C1 Stage 1互換性修正・人工検証（2026-10-25）

[修正報告](docs/research/research_c_c1_stage1_compatibility_20261025.md)、[新未承認契約](docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)。専用bridge `src/trottertracks/c1_ffsim_compat.py` を既存Stage1 runnerへ接続し、venv Python symlinkを保持。新test `tests/management/test_research_c_c1_ffsim_compat.py` と既存回帰59件＋29 subtests PASS。証拠・SOURCE・同版依存lockは `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。実ffsim人工初期化はoptimizerを遮断して確認、分子科学call0。旧run01/run02/STOP保全、新しい1 fitは別明示承認が必要。mandatory STOP。

## 研究C・C1 Stage 1科学実行の技術的STOP（2026-10-25）

[結果・固定SOURCE・部分証拠・STOP](docs/research/research_c_c1_stage1_execution_20261025.md)。固定 `c1_stage1.py` / `c1_stage1_execution.py` と `scripts/run_research_c_c1_stage1.py` を使用。artifactは `artifacts/research_c_c1_stage1_science/2026-10-25/`。1 fitはffsim/SciPy batch logm非互換でoptimizer前に停止、科学call2/147、CDF/XDF誤差とadapter同値性は未測定。追加fit・Stage2/3なし、mandatory STOP。

## 研究C・C1 Stage 1技術準備（2026-10-25）

[実装・人工検証・未承認契約](docs/research/research_c_c1_stage1_technical_preparation_20261025.md)。libraryは `src/trottertracks/c1_stage1.py` と `c1_stage1_execution.py`、入口は `scripts/run_research_c_c1_stage1.py`、人工testsは `tests/management/test_research_c_c1_stage1*.py`、証拠と独立環境lockは `artifacts/research_c_c1_stage1_technical/2026-10-24/`。既存ffsim/一般K/CAR/sector演算を接続、人工45件＋29 subtests PASS。科学runは未承認、allowance採択と新明示承認が必要。

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

## 現在の研究段階

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


## PR混在辞書 F1の保存B入力adapter（準備のみ）

[Phase 2D-F1-PRE](docs/research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)に、専用環境、既存prototypeとの接続、runner／管理test／将来artifact path、実行承認境界をまとめた。科学的検証は未実行。既存SOURCE・入力・結果を保全した独立branchの技術準備である。


## 2026-10-24：PR混在辞書F1のone-shot部分結果

[docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md](docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)に、固定SOURCE・runner・管理testとartifactの対応、metadata TypeErrorによるSTOP、実行済みPASSと未完了を記録した。科学SOURCEはF1-PREのまま、retry／resume・F2なし。要約とhash台帳は `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/`。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

## 2026-10-24：保存F2回路資源の寄与分析

[結果と分類定義](docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)、`src/trottertracks/pr_dictionary_validation/f2_attribution.py`、`scripts/run_pr_dictionary_f2_attribution_saved.py`、`tests/management/test_pr_dictionary_f2_attribution.py`、`artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`を一組として読む。既存exact集計を再利用する保存データの再集計だけで、回路build／compile・sampling・law生成・N計算はしない。主7指標＋measureの公開期待値へexact照合し、因果効果の認定や次の実験には進まない。

## Rectangular Givens候補Aの限定実装・PS48準備（2026-10-24）

[条件・実行境界](docs/research/pr_rectangular_givens_preparation_20261024.md)、`src/trottertracks/pr_dictionary_validation/rectangular.py`・`rectangular_preflight.py`・`rectangular_execution.py`、`scripts/run_pr_rectangular_givens_saved.py`、`tests/management/test_pr_rectangular_givens_preparation.py`、`artifacts/pr_rectangular_givens_preparation/2026-10-24/`を一組として読む。実装・metadata/mock確認のみ。保存PS48の新科学実行は別承認・新grantが必要で、今回build／compileは0件。

## Rectangular Givens PS48科学結果（2026-10-24）

[実行結果・資源差・STOP](docs/research/pr_rectangular_givens_ps48_execution_20261024.md)。固定SOURCE `770e53a` の既存runner／builder／reference target／compilerを再利用し、raw／compiled各48がPASS。結果・CSV・provenance・保存結果集計utilityは `artifacts/pr_rectangular_givens/2026-10-24/`。費用差はcase依存で、全F2への一律展開や追加実験は未実施。

## Gaussian singleton構造選択の静的設計調査（2026-10-24）

[設計メモ・限界・独立検証案・STOP](docs/research/pr_rectangular_selection_design_20261024.md)。PS48結果と既存SOURCEを読むだけで、数値再集計・recipe生成・選択器実装・build／compileはしない。一般的な正の選択条件は未確立で、一旦終了の技術提案はユーザー／GPTの採否待ち。既存の科学的PASS・結果・承認状態は保持する。

## Gaussian singleton独立研究の一時終了・次研究への引継ぎ（2026-10-24、現在の採択状態）

[採択記録と引継ぎ](docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md)。ユーザーが一時終了を正式採択した。上記の採否待ちは当時の履歴で、科学的PASS・負結果・未確立事項は変更しない。既存Track BのC4／Parent allocation／G9／C1と先行研究整理を再利用し、辞書・law・実回路費用・統計負担の共同検討へ論点を渡す。具体的機構の採択・新実装・科学実行は含めずmandatory STOP。


## 研究C・C4：Q13限定科学検証（2026-10-24）

[実行契約・履歴・結果](docs/research/research_c_rustiq_execution_20261024.md)。SOURCEは `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` と `rustiq_execution.py`、入口は `scripts/run_research_c_rustiq_limited.py`、管理testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_limited/2026-10-24/`。保存F1の4case・B0/B1/R1だけを対象とし、旧SOURCE・grant・STOPを保持する。


## 研究C・C4：B2角度結合baseline限定科学比較（2026-10-24）

[結果・固定SOURCE・契約・証拠](docs/research/research_c_angle_fusion_b2_execution_20261024.md)。既存 `rustiq_adapter.py`／`rustiq_execution.py` と `scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1` を使用。testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`。全登録gateがPASSし、全8wrapper・全8compiled指標でB2=保存R1。単純融合がこのfixtureの改善を再現した。旧route再実行・新reference・Rustiq/R2・retryは0。独立研究を区切るかはユーザー／研究C／統括の次判断であり未採択、mandatory STOP。

## 研究C：C4一時終了・C1接口調査（2026-10-24、現在の判断）

[C4終了・成果引継ぎ](docs/research/research_c_c4_closeout_handoff_20261024.md)でユーザー採択済みの一時終了を追記した。上記の未採択状態は当時の履歴。B2等のSOURCE・検証資産・旧科学結果・STOPは維持する。
[C1圧縮DFの静的接口調査](docs/research/research_c_compressed_df_interface_survey_20261024.md)は、X-DF／通常C-DF／RC-DF／SCDF、外部実装と既存DF下流の再利用、general diagonal・scalar/sector/phase、強いbaselineと別承認の最小案を整理する。C1の方式未採択、新実装・科学計算なし、公開照合後mandatory STOP。

## 2026-10-26：H4強baselineの技術準備

[技術報告](docs/research/research_c_c1_h4_strong_baseline_preparation_20261026.md)／[未承認227-call案](docs/research/research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json)。`src/trottertracks/c1_h4_strong_baseline.py`、同名runner・人工testを対応させる。XDFのexact spin対応basisと、両armの位相保持relative Gaussianをopt-in接続。新25＋回帰76＝101 tests PASS、実H4新評価0。B0保存値と新3routeのs1比較を準備、B3／s4／CA・shots・総費用は契約外。別明示許可までmandatory STOP。

## 2026-10-26：H4強baseline s1科学実行完了

[227-call結果・full512検証・5比較](docs/research/research_c_c1_h4_strong_baseline_execution_20261026.md)。9 local／3 kernel／6 adjoint／9 complex X/Y gate PASS。B1 XDF RZ11821、B2 CDF11062/XDF11307、保存B0は再buildなし。単一kernelの数値pilotで、B3・s4・CA/shots/総費用は未評価。旧結果・SOURCE・STOPを保全し、公開照合後mandatory STOP。

## 2026-10-26：H4 B3の技術準備

[技術報告](docs/research/research_c_c1_h4_b3_preparation_20261026.md)／[未承認65-call案](docs/research/research_c_c1_h4_b3_science_contract_draft_20261026.json)。`src/trottertracks/c1_h4_b3.py`、同名runner・人工testがB1 spin基底とB2 relative Gaussianを併用するB3一件専用の入口。新21＋旧101＝122 unique PASS、旧SOURCE/default/227-call契約は無変更。保存reference・shell・ground・metricsを再利用し、実H4新科学call0。別明示許可までmandatory STOP。

## 2026-10-26：H4 B3 s1・65-call限定実行完了

[全結果・full512 gate・6指標・3比較](docs/research/research_c_c1_h4_b3_execution_20261026.md)。固定SOURCE b40fce5c…無変更、65/65 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、元順XDF8/保存CDF8、Nα=Nβ2、全frame S2 s1/h.1、PRなし、own-ground/状態準備除外、共通precision/P0。新B3 spin＋relative一件のみbuild/compileし、raw/P0/actual adjoint/X-Yは全512入力・出力で数値PASS。RZ10885/CX6892/totaldepth11973は保存B2 CDF11062/7008/12127より小さく、他3指標も小さい。単一kernel比較で、同CA総順位・13round・shots・費用・厳密enclosureは未評価／未資格。旧raw・SOURCE・STOP保全、mandatory STOP。上の準備・未実行表記は履歴。

## 2026-10-26：H4 B2 CDF／B3 XDFのCA・RPE・総資源の技術準備

[再利用・接続・人工76 PASS・資格の限界](docs/research/research_c_c1_h4_strong_baseline_ca_resources_preparation_20261026.md)／[未承認86-call契約](docs/research/research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json)。新module `c1_h4_strong_ca_resources.py`／同名runner・testsが保存full512 segment・norm guard・shell・元PF13を既存propagate/RPE/energy/serial集計へ接続する。H4直鎖1Å/STO-3G、CDF8/XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/状態準備除外、共通precision/P0/CA.00159。回路／Operator再生成不要、実科学call0。旧SOURCE/raw/STOP・旧約12%総資源とB3単一kernel結果を保全し、新総順位・厳密enclosureは未評価／未資格。公開照合後mandatory STOP。

## 2026-10-26：H4強baseline CA・実行前binding修正

[原因・人工78 PASS・修正契約v2・実科学call0](docs/research/research_c_c1_h4_strong_ca_binding_repair_20261026.md)。旧86-call契約の保存contract参照が、同一bytesのdocs pathとartifact pathの文字列差でSTOPするため、canonical参照だけを修正した。科学SOURCE31件・数値条件・call上限86・旧契約/raw/STOP不変。run01未作成・未消費、科学実行は未開始。修正契約での別明示許可までmandatory STOP。

## 2026-10-26：H4強baselineの共通CA・serial総資源比較完了

[全結果・26round・shots・総費用・資格の限界](docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)。固定SOURCE1aeedd961…／binding-v2無変更、86/86 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、B2 CDF8/B3 XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/ideal IID Born/状態準備除外、共通precision/P0/CA.00159。両候補のCA数値十分条件PASS、shots13606/13600、条件付き総RZ102056444569/100267421030でXDFが1.75297%少ない。level B数値pilotの限定比較で、厳密enclosure・無条件／一般の優位は未資格。旧結果保全、公開照合後mandatory STOP。上の準備・未実行表示は履歴。
