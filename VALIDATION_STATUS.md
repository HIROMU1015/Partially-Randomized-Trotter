## 2026-10-19 H4/H6 m5：tol-only pilot終了・29/30 PASS、H6 full m5でSTOP

[結果・数値不一致・終了監査](docs/research/track_a_h46_m5_tol_pilot_run01_results_20261019.md)。SOURCE c7d53f04…、13:30:50 JST終了、約27分25秒。H4全lane/H6 PR完了、H6 full m5/q2 cosineのcompiled vector差1.504e-9が固定1e-9未満基準を超過。58受理軸・59予約・sine1未投入、全24 worker exit0/reap・launcher exit1/reap。元SOURCE/input/grant/STOPを保全しretry/追加compile/会計0、AD03/後続Core保留、mandatory STOP。次はCodexの静的原因調査案と、ユーザーの追加検証判断。以下は各時点の履歴。

## 2026-10-19 H4/H6 m5：tol-only technical pilot一回起動・完了未確認

[起動・fresh gate・承認範囲・終了監査](docs/research/track_a_h46_m5_tol_pilot_run01_start_20261019.md)。SOURCE c7d53f04…、最大60軸/PR12本/reference2、最大30 worker。13:07 JST snapshotでreference両PASS・21/30 task完了、STOPなし。新4grantを一回消費し旧DRAFT/INVALIDとSOURCEを保全。m5採用AD03/後続Core/会計/fitは未承認。バックグラウンド継続、終了後自動mandatory STOP。Codexの終了監査・結果報告は未実施。以下は各時点の履歴。

## 2026-10-19 H4/H6 m5：tol-only adapter・4契約の技術準備完了（実行未承認）

[新SOURCE・scope・検査・承認境界](docs/research/track_a_h46_m5_tol_native_readiness_20261019.md)。SOURCE c7d53f04…。旧science関数/保存loaderを再利用し、H4 F/FP tol-only10とH6 tol-only19を独立seriesで接続。最大60軸/PR12本/reference2、4無効grant、実分子配列復元/科学計算/数値再解析0。人工16項目（数値code不変のnative2は既検査再利用）、封印metadata改変16種・実CLI拒否PASS。H4各309/H6各310 fileをseal、暫定120/既知除外4483衝突0、未知外部seed網羅は未主張。旧SOURCE/契約・別チャット保全。主比較AD03/後続Coreは保留、local commit後mandatory STOP、push0。次担当ユーザーのtechnical pilot一回実行判断。以下は各時点の履歴。

## 2026-10-19 H4/H6 m5_best：共通v1に沿う計画差分・実行未承認

[計画・入力差分・疎な候補・並列DAG](docs/research/track_a_h46_m5_protocol_alignment_20261019.md)。H4は保存tol-only10へ新adapterで結合する計画、legacy12/旧SOURCE/4契約は保全。H6はtol-only19を継続。技術pilot4lane/最大60軸と、後続Coreの124 signal候補案・単価・5ε会計・個別nの独立P・費用分解を分離。18要件/23固定fileの静的照合と人工構造12件PASS。AD03主比較採用は保留。新科学計算/数値再解析/有効grant/push0、local commit後mandatory STOP。次はユーザーの範囲確認とCodexの新adapter/契約seal準備（未実施）。以下は各時点の履歴。

## 2026-10-19 H4/H6：m5_best native pilot準備完了・実計算未承認

[実装・4契約・並列化・STOP](docs/research/track_a_h46_m5_best_native_readiness_20261019.md)。SOURCE7a4d126a…、3公式/full・B0・B0+I・PRの4laneを接続。24条件/PR12本/60軸/共有full-H reference2件。人工回帰120件PASS、metadata改変20件拒否、暫定seed120/既知4243衝突0。SOURCE/input/runtime closure各303/305 file封印、4無効grant。旧main profileは参照のみ、新native費用系列として識別。実分子signal/sampling/compile/会計/fit0、mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H4/H6：m5_bestの並列検証計画・科学未実行

[計画・再利用・未完了範囲](docs/research/track_a_h46_m5_best_parallel_plan_20261019.md)。既存4th(m5_best)・11段S2を再利用するH4/H6計画。主controlはsymmetric_directional。4lane独立並列pilot案24条件・PR12本・60軸、後続は依存DAG/各Phase STOP。新12人工PASS、計画SOURCE834b738a…。実分子計算0、native adapter/launcher未seal・無効grant。係数非ZEROのみで棄却せず、主Track Aへ接続する差分を準備。以下は各時点の履歴。

## 2026-10-19 H6：高次PF独立確認の固定2点順位を再現・正常終了

[結果・終了監査・限界](docs/research/track_a_h6_higher_pf_confirmation_run01_results_20261019.md)。SOURCE e3ac6fd8…、最小3候補13/45/45本・新103 trajectory/206軸・control536件PASS。2026-10-18 23:28:31 JST、native exit0/全56 worker回収、約42分12秒。固定2点順位は両方REPRODUCED、近似区間は両方0を含まず。点費用はY4/q2がS2/q8比3.748%低下、Y4/q4がq2比1.751%増。s4/s2_multi未観測、母平均保証なし、探索n8混合/追加sampling/compile/retry0。mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H6：最小3候補の独立確認native準備完了・実計算未承認

[実行契約・検証・限界](docs/research/track_a_h6_higher_pf_confirmation_native_readiness_20261018.md)。H6高次PFの最小3候補S2/q8・Yoshida4/q2/q4について専用native adapter・one-shot launcher・paired集計を実装。SOURCE e3ac6fd8…、13/45/45本＝103本206軸は名目90%の未採択案。306 closure・暫定seed914/旧seed3329との衝突0・人工回帰92件PASS。無効grantで実CLIの事前拒否を確認。目的Aの点順位再現に限定し、目的Bの母平均保証・A_COSTを分離。新分子sampling/compile/signal/費用再解析0、有効grant0、mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：独立確認の3/7候補計画、実計算未承認

[標本数・希少層・資源・契約案](docs/research/track_a_h46_higher_pf_confirmation_readiness_20261018.md)。固定1fc58aa3…のn8を使用。最小3候補の名目80%案10/35/35（80本160軸）、90%案13/45/45（103本206軸）、分散4倍感度案39/137/137。拡張7候補・family7も別案。14設計・47人工PASS、metadata71file。全案未採用、native C adapter/SOURCE/n_C未確定、新sampling/compile/signal0、mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：高次PFの統合比較完了・独立確認は未実行

[結果・選択・確認案](docs/research/track_a_h46_higher_pf_integrated_run01_results_20261018.md)。SOURCE a65e4f6e…、18適格費用群n8/24条件・旧shot24行/費用6群再現、全6層の点選択完了。両系S2 q8/Yoshida4 q2/new_2元8桁 q1。H6 Y4 q2/q4は約0.974%差、S2対Y4は約3.294%差。native exit0・wall0.580秒、42人工PASS。MCSE案は未採用n=null、C sampling/新compile/signal0。母平均・厳密biasを認定せずmandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：高次PFの不足native取得・統合比較は未実行

[結果・制約・SOURCE](docs/research/track_a_h46_higher_pf_acquire_run01_results_20261018.md)。SOURCE d77e439a…、固定12群n8、取得96 trajectory/compile entry192・検証軸192。control636、native exit0・回収。旧6群を保持し、統合metadata状態METADATA_BOUND_NO_ACCOUNTING。重み費用/SE/順位・最良qの新計算0、新signal/DF/state/係数変更/C sampling0。結果保存後mandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：高次PF B保存値解析完了・最適q未確定

[結果・限定・来歴](docs/research/track_a_h46_higher_pf_B_saved_run01_results_20261018.md)。SOURCE ed14d951…/承認one-shot/native exit0・wait回収・wall0.7575秒。保存24条件は経験的18適格/6不適格、保存費用6群n8を再現。適格費用欠測12群のため全6層のfull-grid winnerはnull。新signal/sampling/compile/係数最適化0、元n8の標本追加なし。B取得/C確認を自動起動せずmandatory STOP。以下は各時点の履歴。

## 2026-10-18 H4/H6：高次PF・q比較・独立確認の準備のみ

[技術報告](docs/research/track_a_h46_higher_pf_readiness_20261018.md)。PF repository固定20d64c2…の係数・次数証拠を先に再利用し、projector import先の2関数欠測と元高精度値の未取得を明示。SOURCE ed14d951…、新32人工/mock PASS。保存signal24/費用6/欠測18をmetadataのみ照合、frozen shot・paired RZ/CX・q選択・独立確認設計と無効4grantを準備。実H4/H6数値再解析・順位・sampling・compile・係数最適化0。新公式未採用、n未固定、A_COST/理論費用作業と分離しmandatory STOP。以下は過去各時点の履歴。

## 2026-10-18 H4/H6：new_2 PRのnative・同精度代表比較完了

[結果・SOURCE・限界](docs/research/track_a_h46_new2_native_validation_20261018.md)。保存1Å/STO-3G/H4 rank12 p3・H6 rank19 p5/T=.8/K4/ε=.001、S2 q4・Yoshida q2・new_2 q1の同じPR分割を比較。各n8・採用96 compiled軸、人工18件/実数値222件PASS、最大差1.57e-10。new_2のshot重みRZ標本平均はS2比H4 73.45%/H6 91.59%、Yoshida比41.61%/56.03%低下。ただし固定代表点/empirical allowance/P=0で、最適q・母平均・対決定論・全RPE費用の保証ではない。8桁係数の立方和7.35e-9により厳密4次は未成立。初回compile不一致をexplicit XY分解で修正し、旧STOPを保存。H6を24軌道並列へ引継ぎ、重複13軸一致・全回収。sourceはH4 687679a4…/H6 f0dd2a9a…、旧成果/既存環境・他job不変、mandatory STOP。以下は過去各時点の履歴。

## 2026-10-18 H8：追加6精度の保存値検証完了

[結果・実測/model区分・来歴](docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md)。SOURCE3156b036…、利用者の準備後実行承認に基づくC専用one-shot。保存anchor1,520行を再現後、新2,280行/12比較、native exit0・最終gate PASS、wall0.832秒。通常ε=.00075で+12.77%、.000625で−0.50%（有限標本の点比較）。人工164件PASS、新compile/sampling/signal/fit0。他のA/B/D/E/Fは未実行、条件付き実行承認は保持し、各単位のSTOPを維持。以下は各時点の履歴。

## 2026-10-18 Track A targeted parallel plan: metadata only

[報告](docs/research/track_a_targeted_parallel_validation_plan_20261018.md)。SOURCE daf701d8…、6系列/15承認単位、未承認/無効grant、科学dispatchなし。人工156件PASS、18固定source fileのGit blob/SHA/AST確認。native adapter・実入力・control予算は未seal、実行準備完了とはしない。旧科学結果・V2 NO_GOを保全。

## 2026-10-18 H8：希少層境界・選択効果の保存値解析完了

[結果・来歴・限界](docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md)。RARE 40条件（32境界/8不適格）とSELECTION元4精度の探索的bootstrap2000を別grantで各一回完了、両native exit0・mandatory STOP。人工135件PASS。未知費用はnull、母平均/厳密bias/selection-adjusted coverageなし。PR sampling/compile/fit/shot再計算0、旧結果/ledger/STOPは保持。 以下は各時点の履歴。

## 2026-10-18 H8：s4・複数2次・選択後効果の技術準備（未承認）

[統合報告](docs/research/track_a_h8_rare_tail_selection_readiness_20261018.md)。完了s2 run02（b21e90b8…）と既存正本を36固定file/closure9でmetadata照合。count条件付きsampler・統計schema/人工bootstrap・mock wrapperを実装、人工73件PASS。未承認4stage/INVALID grant、production dispatch未接続。実H8 sampling/build/compile/統計/感度/shot・費用再会計/新順位は0。旧STOP/control376/376・既存成果を保全し、追加検証なしの選択肢を残してmandatory STOP。以下は各時点の履歴。

## 2026-10-18 Track A H4〜H8：横断証拠監査（対象結果2026-10-17）

[証拠整理・metadata検査](docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md)。19系列・222保存値/状態・32固定ファイル、12比較・13不足を別表に記録。実測/model/有限標本を分離し、新科学計算・再会計・ランキング・統計解析は0。H8希少次数pilotは別チャットの未完了作業として収録せず、既存SOURCE/grant/ledger/他worktreeを変更しない。新しい科学的採否・新規性判断なし。以下の記録は各時点の履歴。

## 2026-10-17 H8：主要6軸実測反映の限定再会計完了

[結果・監査](docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)。SOURCE aa984d98…、保存3800行/40比較、科学dispatch1、native exit0・全回収・最終gate PASS。旧32/PR/SE/shot不変、winner変更0、model推定は残存。追加compile/signal/sampling/fit/retry0、mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8：現runを保持した準備工程の並列化

[実装・検証・契約](docs/research/track_a_h8_key_compile_parallel_preparation_20261017.md)。準備を3構成並行・最大6 workerへ拡張し、wrapper生成も軸workerへ移した。人工32件PASS、SOURCE 7f8d74660b2914a1223d0ac7990dde8d4645032e。compileは固定順直列・各軸2%以上でSTOP。新方式の科学起動0・無効grant・未seal。旧runの25 immutable filesはSHA一致し、稼働継続。速度/peak RSSは未測定。以下は各時点の履歴。

## 2026-10-17 H8：主要3構成・6軸の直接compile準備（未承認）

[準備・人工検証](docs/research/track_a_h8_key_compile_readiness_20261017.md)。SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e`。保存入力11件のbyte照合、人工22件PASS。準備2並列・compile直列/各軸2%未満、own-process cleanupを実装。実H8 build/control/compile/費用再会計0、無効grant・CPU未指定・未seal。実compileと後続再会計は別承認。以下は各時点の履歴。

## 2026-10-17 H8 Phase C：B2完了結果の統合・実行準備（未承認）

[資料入口](docs/research/track_a_h8_phase_c_integration_readiness_20261017.md)。既存Phase C snapshotを統合し、B2最終4軸契約（PASS_REGISTERED_FOUR_AXES、保存3＋新規1、軸別SOURCE）へ再結合。B0 p15/q1 S4は実測168,353 RZ優先、未測定S4はmodel推定と明示、非相殺行は拒否、旧32/184は別判定。資源上限とPSI-STOP observerを廃止しwait4 launcherへ。人工37件PASS（実launcher→driver人工e2eを含む）。SOURCE 29ed76544441d534ad99e5c708a2ae17eec78c18、plan 9e42fc76…、unsealed・未承認・無効grant。実H8費用再会計・ランキング・削減率・新signal/compile/sampling/fit0、B2/M1-S/既存worktree不変。以下は各時点の履歴。

## 2026-10-17 H8 B2：登録S4全4軸の実測PASS・正常終了

[結果・結合来歴・終了監査](docs/research/track_a_h8_b2_completed_results_20261017.md)。H8/隣接1Å/STO-3G/DF28/T=.8、B0 p15q1・B0+I p21q8 S4の両軸を実測。B0各168,353 RZ/誤差0.2153638062%、B0+I各1,852,640 RZ/誤差0%、登録4軸PASS。2026-10-17T17:46:48.422662+09:00に最後のsineが終了、SOURCE f58da959…/native driver・launcher exit0/全回収。保存3軸を別受入契約で結合し、元runと過去のSTOP・未settle予約は保持。最新run wall1632.031秒/peak26.252GiB、33source/265provider/34273runtime/31input・702保護/135frozen/286過去immutableを照合。新fit/費用再会計/Phase C/非相殺stage0、全格子・厳密bias・PR母平均の保証は未認定。mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：PSI契約修正・sine再開snapshot

[実測・技術STOP・再開と保全](docs/research/track_a_h8_b2_sine_psi_running_20261017.md)。本人session full PSI0.36%による17:01のsine中断を保全。現行資源方針と技術修正/再開承認に基づき、新SOURCE f58da959…ではPSIを診断記録へ変更し、OOM・実効headroom・観測異常STOPを維持。26人工tests PASS。2026-10-17T17:22:24.738031+09:00時点、2回fresh gate PASS後、保存sine回路の読込/照合を157.1秒継続確認。保存cosine実測1,852,640 RZ/誤差0%を再利用し、新候補/control再計算/fit/Phase Cは0。33source/265provider/34,273runtime/31input、過去286 immutable・702保護/135 frozen worktreeと明示1並行観測。全4軸PASSは未確定、B2終了時mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：cosine実測PASS・sine限定再開snapshot

[実測・技術STOP・再開と保全](docs/research/track_a_h8_b2_sine_recovery_running_20261017.md)。保存cosine実測1,852,640 RZ/誤差0%/native exit0を確認。元attempt02はsine読込中に上位cgroup full PSI0.18%でSTOP、旧記録/未settle保持。利用者の技術回復承認で、検証済みQPYからsine1軸だけを新SOURCE e3ecc03a…で再開。2026-10-17T16:48:38.918552+09:00時点、2回fresh gate PASS・136.0秒の同一worker CPU増加確認、最終RZ/判定は未保存。21人工tests PASS、33source/265provider/34,273runtime/31inputを固定。699 immutable/133 frozen worktreeを厳密照合し、明示1並行worktreeは読取専用の更新観測。Phase C未承認、B2終了時mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：CPU起動修正・新試行の稼働snapshot

[修正・承認・STOP保全・稼働確認](docs/research/track_a_h8_b2_cpu_handshake_running_20261017.md)。起動時CPU固定・witness blocking wait・同窓process CPU証拠を修正。SOURCE2e604e4a…、40人工tests PASS。利用者の技術修正/再開承認に基づくattempt01は科学前CPU再確認でSTOPし全回収・原記録保持、別grant/未使用域のattempt02は2回fresh gate PASS。2026-10-17T14:40:26.878362+09:00時点、S4回路構築を194.8秒の複数観測で継続確認、新compile結果なし。31source/265provider/34,273runtime/16input、旧692保護/127worktree・117過去immutable不変。準備2 worker/compile1の固定範囲。Phase C未承認、完了時mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2 recovery run01：初回CPU gateで科学起動前STOP

[失敗counter・native exit・保全](docs/research/track_a_h8_b2_recovery_run01_cpu_stop_20261017.md)。H8 B2残り2軸の新scopeを利用者承認で一回dispatchしたが、初回fresh CPU gateでcore6のraw busy82%>5%を検出し、科学driver前STOP。driver/worker/control/QPY/compile/fit/再会計0、output/control/lock/fresh receiptなし。SOURCE77ce12d8…、31source/265provider/34,273runtime/16input照合。独立witnessが実native launcher exit1をwait完了・保存、dispatcher exit0と区別。旧688保護/125worktree・元B2原45files不変。CPU再割当/retry0、旧STOP/未settle1/C未承認を保持し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 B2：停止記録修正・2 worker並列準備、科学再実行は未承認

[修正・並列化・受入契約・資源](docs/research/track_a_h8_phase_b2_recovery_parallel_readiness_20261017.md)。B2 run01 MEMORY_PRESSURE/STOPを保持し、残るB0I p21q8 S4の2軸用に修正・並列準備を実装。left/rightとcosine/sine検証・fingerprint/QPY保存を各2 worker、compileは各軸2%判定を挟む直列1。検証済み回路をQPYへ保存し、親の大回路を解放して1軸ずつ読込。PSI raw/対象groupをcleanup前保存、独立witnessでCLI終了後もnative waitを記録、marker集計をaxis_*へ修正。31 source/265 provider/34,273 runtime/16 inputsをseal、34人工testsと2 qubitの4人工compileで指標・fingerprint一致。5物理core/112.3125 GiB案、無効grantのみ。旧687保護/124worktree・B2原45files保持、H8新compile/fit/有効grant0。旧B0/旧B2 STOP・V2 NO_GO、C未承認を維持、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 Phase B2：契約STOP・自動再実行なし

[実測・判定・終了監査](docs/research/track_a_h8_phase_b2_run01_results_20261017.md)。H8/1Å/STO-3G/DF28/T=.8、S4の固定4軸を直列・一回実行。実compile entry3、保存判定2軸（PASS2）。SOURCE244c45e7…、model8420b9f2…を固定。STOP理由は結果報告と原terminalに保存（MEMORY_PRESSURE）。B0I cosineは中断、sine未起動、未settle予約1を保持し、後続compile・retry/refit0。driver native exit−9、launcher報告exit1（外側native wait欠測）、全owned process回収。旧684保護files/123worktree・B0原53/B1原34 rawを保持。新fit/signal/再会計0、元B0 STOP/V2 NO_GOを維持。C未承認、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 Phase B2：S4の固定4軸・直列検証を実装、実行未承認

[実装・固定条件・資源・承認範囲](docs/research/track_a_h8_phase_b2_execution_readiness_20261017.md)。B1正本83a3c4a6…/model8420b9f2…を固定し、B0 p15q1・B0I p21q8 S4の両軸を順番に検証するadapter/launcherを準備。各軸2%以上で次軸を起動せずSTOP。SOURCE244c45e7…、21 source/265 provider/34,273 runtime/15 inputをseal、新19＋既存100＝119人工testsと2 qubit S4接続11項目PASS（人工4compile、実H8 0）。3物理core・80.25 GiB、無効grant/CPU[]/command null、fresh gate未発行。旧683保護filesと別チャットの許可済み1worktree更新を区別、更新後122worktree不変。B1・元B0 STOP・V2 NO_GOを保持、新fit/再会計/有効grant0、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 Phase B1：全4層PASS・費用model固定・mandatory STOP

[結果・model・検証・保全](docs/research/track_a_h8_phase_b1_run01_results_20261017.md)。H8/1Å/STO-3G/DF28/T=.8。新B0保存8軸と既存B0I16軸で4独立S2 fit。B0のp21/q8独立holdoutは両軸0%、B0Iは各8 training anchorの残差0%。係数は各層(a,b,d)=(1592,1314,25271/7)。SOURCE6fb8a0a7…、model8420b9f2…、計算本体wall.092秒/native driver・launcher exit0/全回収、SOURCE19/runtime34,273/input82照合。35人工tests PASS、旧681保護/119worktree不変。元B0 run02 STOPは保持、新compile/signal/再会計0、S4は未実測仮説、B2/C別承認。以下は各時点の履歴。

## 2026-10-17 H8 B0：終了監査修正・保存8軸の読取専用audit完了

[技術判定・修正・証拠](docs/research/track_a_h8_b0_saved_anchor_audit_20261017.md)。H8/1Å/STO-3G/DF28/T=.8、p14/21・q1/8のB0 S2保存8軸を照合し、欠測・hash/lineage不一致なし。再compileを要求する技術的欠損はなく、B1入力候補として構造的に再利用可。元run02のPROCESS_THREADS/契約STOP・過去の違反PID未記録を保持。管理Gitはthread1/optional lock無効、監視は違反identity/exe/thread数をcleanup前保存。新14＋既存83の97人工testsを固定PythonでもPASS。旧681保護/118worktree不変、科学compile/fit/新grant0、B1別承認・公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 B0 run02：8軸を保存、終了時のthread監視でSTOP

[保存測定・停止箇所・保全](docs/research/track_a_h8_b0_anchor_run02_stop_20261016.md)。H8/1Å/STO-3G/DF28/T=.8、B0 S2のp14q1/p21q1/p14q8/p21q8各cosine/sineを4 worker・2 waveで一回取得。8 worker全exit0/reap、TRAIN6/HOLDOUT2保存。約80.76分後、終了時のworktree確認中にPROCESS_THREADSでSTOPし、driver exit−9・launcher exit1、全回収。8軸の保存と取得契約の正常完了を区別し、fit/残差/再会計/自動retry0、旧B1 STOP・V2 NO_GOを維持。以下は各時点の履歴。

## 2026-10-16 H8 B0：CPU選択・起動前STOP記録を修正、再起動なし

[修正・診断・条件維持・人工検証](docs/research/track_a_h8_b0_cpu_gate_fix_20261016.md)。H8 B0 run01のCPU gateを修正。新SOURCE db559a51…、承認前の3固定window候補probe、全選択coreのcounter/負荷率/自身のCPU使用、科学起動前STOP receipt、試行域の再利用拒否を追加。raw5%・6物理core/176.25 GiB・4構成8軸/4 worker/2 waveは不変。新11＋既存27 tests PASS、source15/provider265/runtime34,273/input8照合。run01原因は遡及断定せず、旧677保護/116worktreeを維持。新compile/fit/有効grant/取得再起動0、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 B0 run01：fresh CPU gateで科学起動前STOP

[停止点・証拠・未実測・保全](docs/research/track_a_h8_b0_anchor_run01_stop_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。SOURCE572a337d…の4構成8軸取得を利用者承認で一回起動したが、fresh CPU idle checkがPHYSICAL_CORE_BUSYで拒否。driver/worker・fresh receipt・output/control/lock・科学予約・compileすべて0。自動retry/CPU再割当0。失敗coreの詳細sampleとlauncher exit codeは未捕捉のまま保持。旧676保護/115worktree不変、SOURCE14/provider265/runtime34,273/input8照合PASS。旧B1 STOP/V2 NO_GO・B1/B2/C別承認を維持し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 B0：8軸取得の実装・seal完了、実行未承認

[実装・gate・人工検証・承認範囲](docs/research/track_a_h8_b0_anchor_execution_readiness_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。固定4構成8軸・4 worker/2 waveのadapter/launcher、SOURCE/input/profile・資源監視/回収を実装。SOURCE 572a337d…、14 own/265 provider/34,273 runtime/8 inputのbytes照合PASS。17新unittest＋旧計画10、2 qubit fixture8項目PASS（人工4 compile・実H8 0）。6物理core/176.25 GiBとfresh gate必須、無効grant・CPU[]。旧B1 STOP/V2 NO_GO・旧673保護/114worktreeを維持。科学起動/fit/残差/再会計0、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 B0：新anchor最小取得計画・未承認

[候補・来歴契約・並列資源・STOP](docs/research/track_a_h8_b0_anchor_acquisition_plan_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。新B0 S2はTRAIN p14q1/p21q1/p14q8とHOLDOUT p21q8の各cosine/sine、最小4構成8軸を提案。設計座標のみのdet−49、実fit/compile/再会計0。4 worker/thread1・2 wave、後続B1で各軸holdout誤差2%以上STOP。265 provider/7入力照合・10人工tests PASS。旧B1 STOP・B0I16軸再利用・V2 NO_GOを維持。実adapter/launcher未seal、無効grant案のみ。旧673保護/113worktreeを保全し公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 Phase B1：anchor照合完了・来歴不足でSTOP

[照合・契約・資源・STOP判定](docs/research/track_a_h8_phase_b1_anchor_readiness_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。保存32実測軸を照合し、B0I16軸はprofile/preparationへ結合、旧B0の16軸はruntime全体・cost用basis未結合で全B1 STOP。H6係数移送・実rank/fit/compile/再会計0。82既存証拠、8人工tests PASS、無効契約/grant案とB2事前選択4軸（予測null）を公開。旧673保護/112worktreeを保全、V2 NO_GO、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 Phase A run01：152候補完了・mandatory STOP

[結果・適格性表・資源/来歴/保全監査](docs/research/track_a_h8_phase_a_run01_results_20261016.md)。SOURCE fc463d9e…、H8/1Å/STO-3G/DF28/T=.8/q1/2/4/8。追加S2 96＋S4 56候補、B0/B0Iを評価。wall72.293秒・35 worker全exit0/reap・peak RSS6.846 GiB、132候補が少なくとも1条件で経験的適格。中断/科学retry0、費用fit/compile/再会計0、PR費用優位の新判定なし。原worker SHA欄1欠測は公開時の別SHAと区別。旧671保護/111他worktree不変、V2 NO_GO・B1/B2/C別承認、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 Phase A：実装・人工検証完了、科学実行は未承認

[H8 adapter・保存操作列復元・並列制御・契約](docs/research/track_a_h8_phase_a_implementation_readiness_20261016.md)。追加152候補は不変。22 primitive tasks/1,650 probesと13 prefix tasks、最大16 worker。28人工tests＋2 qubit fixture8項目PASS、provider265/runtime34,273/input11を結合。実H8復元・signal・数値再解析・compile・新有効grant0、別途明示承認とfresh gateが必要。V2 NO_GO・旧grant/ledger・V1/H5/H7を保持し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8対照強化の準備・V2現行残差設計のNO_GO終了

[H8準備状況・152追加候補案・Phase別資源/STOP](docs/research/track_a_h8_strengthened_deterministic_readiness_20261016.md)。旧32候補・適格B0I16実測軸・保存full preparation exportを再利用。S2 p8〜20（14再利用）とS4 p15〜21、B0/B0I・q1/2/4/8。metadata checkerのみ実装、H8 science adapter/decoder検証・SOURCE sealは未完了。新科学/数値再解析/compile/有効grant0、Phaseごとの別承認が必要。[V2現行fitは利用者指示でCLOSED_NO_GO](docs/research/track_a_v2_residual_design_closure_20261016.md)。832理論行の代替利用は技術的可否だけ、旧条件の緩和なし。公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V2：H6探索PFの最終照合・現行残差設計NO_GO候補

[最終証拠・不足理由・固定条件での判定](docs/research/track_a_v2_h6_pf_final_reconciliation_20261016.md)。旧PF20候補（測定16・未測定4）を最後に限定照合。比較可能PF0・fit適格0。15測定候補は固定profile未結合、profile確認済みfull S4/q8の1候補もbasis lineage未完了かつtrain側のみ。固定分割・最小件数を維持し、広域回収を終了した。141入力・旧667保護ファイルを照合、PR/確認/H8/B0I・V1/H5/H7不変、新fit/残差/compile/config/grant0。現契約をNO_GO候補として利用者判断へ、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V2：来歴25件を条件付き結合・固定fit条件は不成立

[回収証拠・件数・不足理由・次の判断](docs/research/track_a_v2_provenance_reconciliation_20261016.md)。PR158 entriesのseed来歴を回収し、H6探索17＋確認8の25件を条件付きmetadata結合済みへ更新、237件は未解決。結合済み層にPFはなくfit適格0。来歴無視の上限でもH8全PRとH6確認PRは固定holdout側0で、条件を変更していない。原receipt7,043件と旧保護667件を照合し、SOURCE・結果・grant/ledger・V1/H5/H7を保全。実fit/残差/compile/新grant0、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V2：残差runner人工検証完了・実データ来歴は未解決

[実装・43人工検証・契約・不足証拠](docs/research/track_a_v2_residual_implementation_20261016.md)。保存RZ runner・metadata準備・未承認契約を実装し、人工36＋observer/cleanup7＝43件PASS。SOURCE5703dbda…、source9/runtime34,273/input5をseal。現台帳の262 scope entriesは全件UNRESOLVED_PROVENANCEでfit対象0、保存receiptへの独立結合が必要。実fit/残差/compile/primitive/law/DP・有効grant0、CPU[]/command=null。V1・旧V2・H5/H7と旧保護665件を保全し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V2：保存結果を使う残差解析の再開準備・未実行

[構成対応・未解決事項・最小実行計画](docs/research/track_a_v2_residual_analysis_readiness_20261016.md)。既存7e416705…のA/B832理論行・保存接続10,480行を再利用。uniform A/BとnativeのA-prefix/B-tailを区別し、精度反復のdedup・来歴層別・RZ単価の共通較正/heldout残差・各層.05即STOPを未承認案として固定。新数値再解析/compile/primitive/law/DP/新grant0、V1/H5/H7・既存V2不変。数値runner/人工検証/最終seal/別実行承認が必要。準備資料だけを公開しmandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：実測確認・モデル条件付き結果の成果整理

[現在の主張・5精度証拠表・追加実測の判断材料](docs/research/track_a_v1_results_synthesis_20261016.md)。H6/1Å/STO-3G/DF19/T=.8/P=0。ε=.05はp7両軸実測単価×保存shotによるPR削減率−13.2901%の限定点比較。残る4精度の30.89/34.97/31.17/10.03%は固定model条件付き。厳密bias/PR母平均/全grid実測最良は未認定。追加実測は利用者判断、厳しい側p11/q4 S4を条件付き第一候補に整理。新計算・新grant0、V2保留、科学STOPを維持。以下は各時点の履歴。

## 2026-10-16 V1：p7実測費用の限定比較完了・STOP

[結果・算術・保全監査](docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md)。H6/1Å/STO-3G/DF19/T=.8/B0+I p7 q1 S2/ε=.05/P=0/RZ。保存shot×実測2軸で決定論356,097,424 RZ、保存確認PR403,423,200 RZ、PR削減率−13.29012029%（倍率1/10）。通常bias・経験的allowanceの保存点比較で逆転を維持。全候補再選択・母平均/厳密bias保証は未実施/未認定。旧grant等663/compile raw22不変、新compile/signal/sampling/shot/fit/V2なし、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：p7/q1 S2の2軸実測PASS・完了STOP

[実測・終了監査](docs/research/track_a_v1_p7_compile_results_20261016.md)。SOURCE84b9ea22…、H6/1Å/STO-3G/DF19/T=.8/B0+I p7 q1 S2。cosine/sine各14,492 RZ、予測14,524.2から各0.222191554%でPASS。wall73.241秒/native exit0/全回収、SOURCE21/provider230/runtime34273/input15/旧grant等660不変。登録compile2のみ、再fit/retry/新signal/費用再会計/V2なし。総費用比較は別承認待ち、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：p7/q1 S2の2軸実測を準備・未承認

[実装・90人工tests・入力監査・実行契約](docs/research/track_a_v1_p7_compile_readiness_20261016.md)。指定結果25bbdd33…から専用branchを分離。B0+I p7/q1 S2 cosine/sine最大2軸、予測14524.2 RZ/軸を固定、各軸2%以上で即STOP。旧SOURCE17/provider230/runtime34273・15入力・H5/H7含む660ファイル不変。新compile/build/probe/signal・有効grant発行0。未承認draft/CPU[]/command null、実測後の費用比較も別承認。準備公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：5精度費用再評価・完了STOP

[結果・全4320会計行・60比較・監査](docs/research/track_a_v1_reaccount_results_20261016.md)。SOURCE2261a3ad…、H6/1Å/STO-3G/DF19/T=.8/P=0/RZ。通常biasと固定費用modelの条件下で、細格子＋S4後のPR削減率はε=.05/.01/.005/.001/.0005で−13.04/30.89/34.97/31.17/10.03%。新最良対照はmodel推定、p7は外挿。緩精度の優位反転・中間10pp以上低下を報告。非相殺診断は旧B0I12候補の分解欠測で全domain最良未解決。旧2170照合一致、wall6.024秒/native exit0/全回収、旧653ファイル不変、新signal/compile/fit/V2なし、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1：5精度の保存費用再評価を準備

[比較・実行契約・実装・人工検証](docs/research/track_a_v1_reaccount_readiness_20261016.md)。利用者の継続指示を受け、Phase Bの固定modelと実測4軸を入力とするone-shot再会計を準備。旧32→S2追加96→S4追加136候補を比較し、確認PR union8/n32と探索72/n8を区別。実測優先・新compile/fit/V2なし。pinned71人工tests PASS、厳密bias保証と全gridモデル誤差保証は未解決のまま、fresh gate後実行・終了時mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1 Phase B run01：登録全4軸PASS・完了STOP

[結果・全表・終了監査](docs/research/track_a_v1_phase_b_run01_results_20261016.md)。SOURCEfcf37e67…、H6/1Å/STO-3G/DF19/T=.8のS4登録4軸を検証。B0 p11q1の両軸誤差0.0983742%、B0+I p15q8の両軸0%、全軸2%未満。wall3037.384秒/native exit0/全回収。固定model・650旧grant/ledger等・runtime34273不変。再fit/追加compile/5精度再会計/V2実行0、mandatory STOP。全grid保証・保守的bias保証・旧B0 binary同値は未認定。以下は各時点の履歴。

## 2026-10-16 V1 Phase B run01：検証compile起動

[時刻付き起動記録](docs/research/track_a_v1_phase_b_run01_started_20261016.md)。SOURCEfcf37e67…をpush/remote照合後fresh gate PASS、2026-10-16 14:10 JSTに最初のB0 p11 q1 S4 cosine compileの開始を確認。登録最大4軸・各軸2%以上で即STOP・再fit/追加compileなし。650旧grant/ledger等不変。全軸PASS/完了はまだ未確認、Phase終了時mandatory STOP、5精度再会計/V2は自動実行しない。以下は各時点の履歴。

## 2026-10-16 V1 Phase B：費用モデル固定・検証起動準備

[契約・モデル・実装・人工検証](docs/research/track_a_v1_phase_b_readiness_20261016.md)。Phase A追加104候補と保存28費用軸を照合、4独立S2 fitの残差0。S4予測を固定し最大4軸を各軸2%以上で即STOP、再fit/追加compileなし。pinned57人工tests PASS。旧B0 binary profile未確認・経験的u・bias保証未解決を保持。後続の利用者開始指示を受けfresh gate後one-shot起動へ、V2/5精度再会計は保留。以下は各時点の履歴。

## 2026-10-16 V1 Phase A run02：追加signal完了STOP

[結果・全表・終了監査](docs/research/track_a_v1_phase_a_run02_results_20261016.md)。SOURCE26b9fde6…、新104候補＋PR診断8件を取得。wall95.102秒/native exit0/全27worker回収、factor1/10の保存適格性は一致。経験的u・保守的bias保証未解決を保持。旧grant/ledger等646不変、compile/費用model/V2/次Phase0、mandatory STOP。以下は各時点の履歴。

## 2026-10-16 V1 Phase A run01：科学予約前STOP

[実行・回収・保全・診断](docs/research/track_a_v1_phase_a_run01_stop_20261016.md)。利用者の起動承認後、fresh gate通過。最初の監視でOBSERVER_RSS_CEILING、driver exit -15/全回収、科学lock/output/予約0。旧grant/ledger等644ファイル不変。実測RSSは未記録で原因未確定。監視修正を47人工＋pinned影響17件で検証、再実行は未承認、自動retryなし・V2保留・mandatory STOP。以下は各時点の履歴。

## 2026-10-16 Track A V1 Phase A並列準備・V2保存結果照合

[Phase A資料](docs/research/track_a_v1_phase_a_parallel_readiness_20261016.md)。V1は最大16worker・44限定人工tests PASS、SOURCE/input/profile静的照合一致。新科学計算・新grant0、未承認・mandatory STOP。V2は指定commit7e416705…のH6/H8計832理論行を読み取り専用照合し、重複生成・計算を保留。以下は各時点の履歴。

## 2026-10-16 Native DFモデル：Qiskit 1.3.0接続検証

[42 local人工tests](artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/connection_validation_v1.json)／[適用範囲](docs/research/track_a_native_df_qiskit_connection_validation_20261016.md)。実Qiskit 1.3.0の小回路接続を確認。旧1ea counterのzero RZバグを訂正、旧receipt不変。実H6/H8・全環境互換性・compiled母平均は未認定。
実分子評価・保存H6/H8再解析・sampling・transpile・server接続・新grant・原稿改訂0。公開後mandatory STOP。

## 2026-10-16 Native DF理論ゲート数モデル（人工検証のみ）

[技術資料](docs/research/track_a_native_df_theoretical_gate_model_20261016.md)／[27件のlocal stdlib人工検証](artifacts/resource_applicability/track_a_native_df_gate_model/2026-10-16/synthetic_validation_v1.json)。解析unitary・相対位相・有限分布DPを人工値だけで確認。実Qiskit接続・compiled母平均・実分子適用は未検証。
新しい実H6/H8数値評価・sampling・compile・server操作・grant発行は0。既存結果・科学SOURCEは保持し、公開後mandatory STOP。

## 2026-10-15 H6固定δ T依存性：結果

[全結果・図・終了監査](docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md)。H6固定δ=.2/固定5構成、T=.4/.8/1.6。新144/144 compile・PR64完了、T=.8確認32再利用。ε=.05〜.005ではPR点費用約50〜55%削減、厳しい精度ではS4優位またはPR不適格。outer-PF biasが主因、希少次数・母平均未保証。wall3117.158秒、native exit0/全74 reap・残存0。保存314条件/7算術tests/900公開paired metrics PASS、新科学計算・共有変更0、mandatory STOP。
以下は各時点の履歴。

# Validation status

## 2026-10-15 H6 native B0+I全12構成：結果

[結果・全表・終了監査](docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)。H6 native B0+I全12構成の必要取得が完了。新12軸＋旧6軸再利用、全条件不適格6軸は未測定。SOURCEac6d28e3…、P=0/RZのPR確認union8/n32点費用は登録B0/B0I/full S2/S4最良より5精度で29.35〜56.14%低い。経験的u・点選択・matched B0未取得の限界保持。wall2326.546秒、native wait0/12worker全exit0・reap、保存表2607条件PASS。新科学計算/新grant/原稿改訂/共有環境変更0、科学採否はGPTへ、mandatory STOP。 以下は各時点の履歴。

## 2026-10-15 H6 B0+I blueprint結合修正

[原因・修正・検査・契約案](docs/research/track_a_h6_b0i_blueprint_fix_20261015.md)。run01停止原因の旧preparation hashと現在profileの不一致を3caseで再現。旧hash保持＋現profile/input/providerに固定した3blueprintの厳密照合、実/期待field差の保存を実装。SOURCE6b4d3026…、新12＋影響70の82人工tests PASS。数学/compiler/最大6並列・合計6compileは維持。サブエージェント0、限定snapshot診断2/preparation6以外のsignal/native build/probe/compile・新grant・本計算再起動0。未認可draftへ再結合し、旧run/carryを返却せず保全。 以下は各時点の履歴。

## 2026-10-15 H6 native B0+I pilot6 run01：Phase A STOP

[起動・停止・回収・carry監査](docs/research/track_a_h6_native_b0i_pilot6_run01_20261015.md)。利用者の明示承認を固定し、SOURCE7300b39d…/最大6並列・合計6compileのpilotを一度起動。Phase AでB0_BLUEPRINT_FINGERPRINT_MISMATCH、native exit2/約8.60秒、snapshot1・overlay1予約、native build/control/compile0。driver/launcher退出、SOURCE/input不変、失敗分をrefund/resetせず保持。どのblueprint fieldが違うかは現メッセージで未特定、guard迂回・自動retryなし、mandatory STOP。 以下は各時点の履歴。

## 2026-10-15 H6 native B0+I 6軸compile並列化準備

[並列化実装・契約・検証入口](docs/research/track_a_h6_native_b0i_parallel_readiness_20261015.md)。既存検査済み6wrapperをfork継承し、A/B全3case合格後のcompileだけ最大6processへ並列化。SOURCE7300b39d…、新24＋影響46の70人工tests PASS、軽い独立差分レビューPASS。再load/build/QPY転送を追加せず総compile6、各thread1、owned cleanup/PDEATHSIG/総CPU会計を実装。実fork/科学/compile/高速化実測0。各AS32/RSS9GiB、stage総CPU・wall4hを保持し、全role込みadmission240.5GiBを未承認案へ。CPU[]/false/未seal/commandnull、明示実行承認とfresh binding待ちSTOP。 以下は各時点の履歴。

## 2026-10-15 Track A H6 native B0+I pilot準備・承認待ちSTOP

[実装・契約・人工検証・独立監査入口](docs/research/track_a_h6_native_b0i_pilot_readiness_20261015.md)。独立pilot runner/adapter/owned launcherを追加。SOURCE b4c12d7a…、author23＋独立23＋旧bridge17の63限定人工tests PASS。旧75結果・科学provider/bridge/shot規則は保持。5入力bytes/SHA・provider230/profile34,273 files一致、実H6数値読込・新scalar/signal/native build/probe/compile・新grant0。実行は未承認、CPU[]/未seal/commandnull。明示承認とfresh source/input/profile/carry/CPU・資源/quota/output binding後に3case6軸だけone-shot、matched B0/全面mapへの自動進行なし。 以下は各時点の履歴。

## 2026-10-15 Track A H8 D0 one-shot完了STOP

[結果・全table・監査入口](docs/research/track_a_h8_d0_saved_value_result_20261015.md)。H8 D0 one-shot完了。SOURCE20f89bc3…、旧precision944/cost3576 EXACT PASS後に5精度/factor1・10/6metric/Pを評価。確認32のε=.0005/P=0/RZ PR/取得済B0-B1最良.761667659、探索Pのunion外coverageを分離。native exit0/wall2.97095秒/監視RSS52.3359MiB、source/input/profile不変、新science/compile/H4-H6再実行0。H6は別承認待ち、mandatory STOP。以下は各時点の履歴。

## 2026-10-15 Track A 独立技術監査・最小修正

[独立監査入口](docs/research/track_a_independent_technical_audit_20261015.md)。著者文脈を引き継がない別担当が監査し、6制御不具合を再現・最小修正。SOURCE20f89bc3…、新22＋既存41＋変更adapter旧12の75限定人工tests PASS。旧D0/shot/math/native科学SOURCE・73旧worktree/grant/venv不変、実科学/再会計/build/compile/新grant0。empirical u・実一致/native・監視/運用bindingの限界を残し、未認可・未seal、GPT科学判断へ戻してmandatory STOP。以下は各時点の履歴。

## 2026-10-14 Track A共通研究情報 サーバー暫定初版

H4 B0+I/result98cdb35c・v0.4 47dc17a8、H6確認/pilot、H8最新resume02を固定SHAと読取metadataで索引化。
67 worktreeのremote/dirty/commit到達性、Git管理外レビュー4件の公開copy、private原本の所在を区別。ローカル側の状態は未確認。
H8 snapshotは92 signal・203/592探索group・8 active、最終確認/result未取得。実行を変更せず既存監視下で継続。
[共通入口](docs/research/track_a_shared_research_status.md)。未来の1Å/H4〜H8/H5-H7/T/精度候補は新計算未認可。科学/compile/sampling/NPZ読込/共有venv・H8変更0、資料整備後STOP。以下は各時点の履歴。

## 2026-10-14 H4 B0+I実測原稿v0.4・科学review資料完成

固定結果98cdb35cを専用worktreeで統合。v0.3/旧4図/SOURCE59/元結果を保全し、新通しdraft・4主図＋S1（PNG/SVG/PDF15 assets）、GPT handoff・claim対応・条件表・旧版変更一覧を作成。
元集合1812点B2と、実測B0+I追加後397/1415を分離。費用因子1812・SE差1812・method12/template19区間・matched120・固定display216行を保存。
374点がB2点推定−2SEも下回ることを再確認。same32選抜/推定、NOT_SAVED/172丸め、母平均/energy/RPE等の限界は保持する。
8限定stdlib検査と保存算術/bytes/links照合PASS。主張の科学的採否・新規性・論文化・追加検証はGPT判断待ち。
[統合入口・GPT handoff](docs/research/track_a_h4_b0i_manuscript_integration_20261014.md)。新科学計算/compile/sampling/NPZ読込/共有venv・H8変更0、旧grant再利用/reset0、整備後mandatory STOP。以下は各時点の履歴。

## 2026-10-14 H4 B0+I全240回路 実測完了・費用地図STOP

12 workerで234新規compile＋6 numeric照合再利用、全240 COMPLETE、stage193.990秒。SOURCE924b779c2bc6f6e844d6283c355f91c73aed6445・closure59とimmutable launch artifactは不変。
実測B0+Iを加えた1812点の費用点最小はB0+I397/B2 1415、ties0。全240軸で旧B0との差はRZ/CX数・各depth・sizeとも0。
ε=.05では.70/.80/1.40 ÅでB0+IがB2点推定比25.97/10.75/3.91%低く、他3距離ではB2が低い。ε=.005では6距離すべてB2が低い。
保存shot×実compiled RZ、状態準備P=0のlocal事後解析。32 paired trajectoryの選抜・推定、希少event、母平均/同時保証、energy/RPE・実backend未評価を保持する。
worker native exit12件全0、observer意図的−15、owned14残存0、monitor最大interval1.01093秒、PSI0/OOM増分0。累積actual73263/74805、旧carry返却なし。
[結果・全地図・監査入口](docs/research/track_a_h4_b0i_map_results_20261014.md)。公開projection/全候補再最小化/Fraction照合PASS。追加compile/NPZ decode/共有環境・venv・H8変更0、次stageなし。主要claim確定は結果reviewへ戻す。以下は過去の各時点の記録。

## 2026-10-14 H4 B0+I12並列run02 起動

利用者の並列高速化・memory上限見直し・起動承認を反映し、12worker/各32GiBで全240 wrapperを開始した。旧6件numeric照合再利用、234 new compile、科学条件とcompiler不変。SOURCE 924b779c2bc6f6e844d6283c355f91c73aed6445、25限定mock検査PASS。run01のhard AS継承bootstrap STOPは0 actualで修正し、bytes/wall carryを保持した。
[起動記録・監視入口](docs/research/track_a_h4_b0i_parallel_run02_20261014.md)。起動確認は最終完了ではなく、計算は監視下で継続。共有環境/既存venv/H8変更0。以下は過去の各時点の記録。

## 2026-10-14 H4 B0+I全面240回路 準備完成・新grant待ち

代表6実測の公開commit c377b074から独立し、240 wrapper／234 new compile／6 verified reuseの閉じたregistryを実装した。
SOURCE 0f84119302de869c904bfd0135af3880a6c16bf2、旧48 source不変・closure55。新19 stdlib/mock検査PASS、compile/scientific import/array/spawn/affinity0。
6入力計82321596 bytesとfreezeのstreaming SHA・H/DF/state/input metadata binding・既存profile一致を確認した。
既存6件の再利用は新buildのexact numeric FP一致が条件、不一致でSTOPし自動compile fallbackなし。
累積actual案73029＋234＝73263／74805、carryと17GiB charge capは保持。
1worker・CPU0/1/2候補・role8GiB維持、stage4h／observer log160MiB／1024 inodeへ拡張する具体的な未承認案を固定した。
[全scope・予算・承認条件の入口](docs/research/track_a_h4_b0i_map_preparation_20261014.md)。approved=false、runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。
旧科学レビュー・44人工testsは繰り返さず、限定差分とbindingを確認した。共有環境/既存venv/H8変更0、計算未起動。
以下は過去の各時点の記録。

## 2026-10-14 H4 B0+I代表6回路 実測完了・mandatory STOP

科学レビューで追加対照に採択後、利用者が固定6回路とCPU/資源契約を承認した。
SOURCE 8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b不変、grant commit449614525d5de74daccf9f193e3199eb931fb388。
.70/p5、.80/p5、1.40/p3のq1両軸を実compileし6/6 COMPLETE、全RZ増分0、stage11.73秒。
ε=.05/P=0の保存shot会計でB0+Iは保存B2最小点推定比25.97%／10.75%／3.91%低い。
固定3構成×302精度の906行で178／103／87点がB2点推定を下回る。全B0+I family/1812点のwinnerや母平均優劣は未確定。
worker sampled RSS最大110.66MiB、最大監視interval1.0075秒、native driver/worker exit0・observer意図的−15、cleanup・残存0。
累積73029/74805 actual、旧carry保持。科学arrayを再生成せず、H8/共有環境/既存venv変更0、追加compile0。
[実測・結果監査の入口](docs/research/track_a_h4_b0i_pilot6_results_20261014.md)。既存科学レビュー・44人工testsは再実施せず、公開binding/Git blob/Fraction会計照合PASS。
next_stage_authorized=false、mandatory_stop=true。全面240候補／条件付き追加234は別scope・別grant・別利用者承認が必要。
以下は過去の各時点の記録。

## 2026-10-13 H4 B0+I検証用実装・限定compile準備

新しい独立method/adapterを追加し、元B0の38 Python source・結果・原稿v0.3を保持した。
44限定人工/mock tests PASS（3qubit以下、process1/thread1、実compile/transpile/NPZ decode/spawn0）。
対象は.70 p5/q1、.80 p5/q1、1.40 p3/q1のcos/sin計6 task。SOURCE 8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b、48 source closure・既存profile・3入力streaming SHA・freeze array byte metadataを結合。
carry73023件/13615088338 bytes/268318.34715263546秒を保持し、追加上限6、CPU/RAM/storage/STOPを未承認のproposalとして固定。
approved=false、runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。新baseline採用/主要claim変更/実H4起動/H8・共有環境変更0。
[検証用実装・準備の入口](docs/research/track_a_h4_b0i_compile_preparation_20261013.md)。実numeric FPと実compiled RZは未測定、GPT採否・新実行承認・final review・fresh gateが残る。報告後STOP。
以下は過去の各時点の記録。

## 2026-10-13 H4 geometry：追加技術解析・原稿統合完成

利用者の新指示により、前回保留したA〜Eを専用branchで完成した。元登録B0/B1/B2/B3内では全1812点でB2/prefix3の点最小を保持。
B0+Iは正式baselineに採用せず、compiled費用未測定の追加感度として397点と候補/axis別Δを分離した。
全1308×302の保存算術、15構成領域、shot×effective-RZ分解、4803近接候補のpaired SE、1164候補のrare-law診断、4図、原稿v0.3を公開bundleへ固定する。
H4のevent次数/component観測はNOT_SAVEDであり0回観測と扱わない。H6補正の流用・新科学計算・compiler・H8/共有環境変更は0。
14限定人工算術と公開Fraction照合PASS、local技術検証であり独立科学review/CIではない。
[技術統合の入口](docs/research/track_a_h4_geometry_technical_integration_20261013.md)。B0+I採否・追加6/10/240 compile案・母平均/主要claimはGPTへ判断を戻し、成果報告後STOP。
以下の保留・STOPは各時点の履歴である。

## 2026-10-13 H4 B0+I保存値診断・科学判断待ちSTOP

優先Bでtail scalarのsource符号/UPLO=L規約と6入力SHA、120保存complex signalを照合し、13人工算術PASS。
追加compiled RZ/shotをゼロと置いた参考モデルで397/1812点が登録B2点最小を下回り得る。ε=.05では0.70/0.80/1.40 Å。
B0+Iの実compile費用は未測定。元登録集合の結果は不変で、主要claim変更の可能性という利用者停止条件に従いGPT判断へ戻す。
原稿本体・A/C/Dの全面統合は未実施、新eigh/SCF/DF/state/量子signal/trajectory/build/compile0、H8/共有環境/他job変更0。
[科学判断待ち資料入口](docs/research/track_a_h4_geometry_scalar_diagnostic_20261013.md)。B0+Iは未測定対照の感度であり、実測勝敗・比較採択を変更しない。

## 2026-10-13 H4 geometry map 全件完了・軽量結果公開

H4 linear/STO-3G/DF12、T=0.8、二次DF-prefix PF/finite RTE、6距離の1,308候補・74,784 wrappersがMAP_COMPLETE_STOPで完了。
全record/signalの保存identity・paired metrics・coverageを照合。公開projectionだけから302精度点×6距離の点最小と18比較行を追跡できる。
ε=0.05/P=0ではB2 L_D=3が全距離の点推定最小、最小deterministic候補比27.9〜42.7%低い。全302点でもB2点最小。
近接B2の±2SEは重なり厳密winner未認定、energy/RPE/H6/noise/state prepへ一般化しない。旧dense/compiler系列と混合しない。
累積wall74.53h、compile消費/予約73023/74805、charge12.68/17GiB。own6役退出、追加science0・次stage未認可。
[完了結果・比較表・公開データ入口](docs/research/track_a_h4_geometry_results_20261013.md)。local事後保存値照合でありCI/独立最終scientific reviewではない。

## 2026-10-07 H4 run05 monitor STOP・新hostへの移行準備のみ

[引継ぎ資料](docs/research/track_a_h4_new_server_handoff_20261007.md)にrun05停止後の状態を追加した。
run05完成compile records0、signal0/1308、全owned processes終了。75 local人工testsはproduction成功を示さない。
H4 linear/STO-3G/DF12、追加6距離、T0.8、L_D0/3/4/5/6/9/12、q1/2/4/8、delta0.8/0.4/0.2/0.1のmapは未完了。
消費/予約actual invocations20、charge165214360 bytes、保守的wall carry5466.188392877579 sを引き継ぐ。
既存source19とplan/auth/reviewは不変。今回の追加は軽量引継ぎ資料/索引のみで新科学処理・追加transpile0。
NPZ/runtime/controlは別転送対象。新hostのCPU利用・最終review・明示launchは未承認、本計算前STOP。
以下の旧状態表・manifest entryは各時点の履歴として保持する。

## 2026-10-05 Track A PM-2保存値解析完了 mandatory STOP

[結果照合](docs/pr2_pm2_precision_resource_result_validation.md)はPOSTHOC保存値解析。
source `324435d`の6 blobsと保存4 JSONが不変、ε=0.05の223候補の整数shots/eligibility・6費用が一致。
固定302点・67,346行、適格境界223行、P envelope4,693行、代表2,334行を検査した。
pre/post62 local synthetic tests passed、fail/skip0。CPU1、BLAS1、wall約2.92秒、peak RSS210,872 KiB。
新signal/sampling/build/compile、分子データ/runtime/registry、量子shots、GPU access0。
`PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`、next-stage=false、research decision=null。
利用者指示でresult commitへ収録するlocal resultでimmutable CIではない。point±2SEをformal CIや厳密winnerへ読み替えない。
runner manifestは不変。研究方針reviewへ戻し、追加科学計算をしない。以下は各milestone時点の履歴である。

## 2026-10-05 Track A PM-2保存値解析source固定 STOP

[解析実装](docs/research/pr2_pm2_precision_analysis_implementation.md)をsource commit `324435d77b6642dbd44e8d1f178420daf62e77ed`で固定。
62 local synthetic tests passed、fail/skip0、testsのreal evidence/保護データaccess試行0。
実データpositive numerical gate・precision mapはまだ未検証。準備契約・旧証拠・sourceは不変。
`PM2_ANALYSIS_SOURCE_FROZEN_AWAITING_USER_LAUNCH`、本解析は別の明示指示待ち、new science/次段認可false。
固定後hash/blobとpost-source testは[監査](artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/source_freeze_v1.json)に保存。
以下のlocal-uncommitted/未実装記述は準備時点の履歴である。

## 2026-10-05 Track A PM-2保存値解析の契約準備 STOP

[PM-2契約](docs/research/pr2_pm2_precision_resource_contract_v1.md)とschema/input identityを固定した。
evidence commit `194cc604b90c56a0e7e949b91b064a4bcfc846da`の保存JSON4件だけを読み、全218 development候補と
M2元5構成の別集合、保存axis field・paired cost samplesを照合する。基準適格214件を記録するが候補除外はしない。
POSTHOC、ε=0.005〜0.1、α_axis=0.025、軸別compile平均、common P>=0、strict適格境界、元ε再現を契約化した。
local未commitの準備であり科学結果ではない。precision sweep、shot/work再評価、順位、P envelope、
新signal/trajectory/build/compile、分子データ/runtime/cache/GPU accessは0。
`PM2_PRECISION_CONTRACT_FROZEN_ANALYSIS_NOT_AUTHORIZED`、mandatory STOP、next-stage=false。
旧PM-1/M2 result/status/source/manifestは不変。解析実装・source固定と別の利用者指示が次のbarrierである。

## 2026-10-05 Track A PM-1一回実行・結果照合完了、mandatory STOP

利用者の明示指示「PM-1を実行して」を受け、固定source/plan/true authorizationで一回だけ実行した。
[PM-1結果照合](docs/pr2_pm1_discard_result_validation.md)は
H4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、T=0.8、
B0 rank4/5 × q=1/2/4/8、delta=0.8/0.4/0.2/0.1、r=K=0の8件が全件accuracy適格であることを記録する。
最小の新B0 rank5・q1はprimary G_RZ=229,718,060、保存B2 rank3・q1・r4・K2点推定の1.75659倍、
旧B0 rank6・q1より9.34%低い。比較は固定development集合内、pure discard/PF分解はnullのまま。

134 source blobs、8 fingerprints、16 wrapper keys、signal/cost式、40 point ratios、
manifest2件のbytes/SHAとcompletion markerを照合した。runner manifest/resultは不変。
8 signal＋16 wrapper、CPU1/BLAS1、wall141.614秒、peak RSS2,145,856 KiB。
development hash/load各1、random/held-out/GPU/quantum shots/retry0。
pre/post限定201 local tests passed、fail/skip0、guard attempts0。result SHA-256：
`9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b`。

statusは`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、研究判断null、next-stage=false、
mandatory STOP。次は研究方針reviewであり追加科学計算を認可しない。
軽量result/audit/reportは利用者指示によるresult commitへ収録するlocal execution evidence。immutable CI・外部再現や最終総costではない。
以前のdraft/finalization/準備entryはそれぞれの監査時点の状態として保持する。

## 2026-10-05 Track A PM-1最終承認・一項目確定、明示launch待ち

利用者提示review `APPROVE_PM1_EXECUTION`を受領し、
[確定記録](docs/research/pr2_pm1_execution_finalization_20261005.md)を追加した。
単独commit `bf9eaeec868361df0c8e05d06e9570a2bfc5a7a4`はauthorization JSONの`final_review_approved`だけをtrueにした。
SHA-256は`2113978b360ca763071b860c7ea2d14b83eccc68a471510a98da5ea5c2282530`。
source134、sealed plan、8候補/16 wrapper、他field・環境・資源上限・root/outputは不変。
committed authorizationのscience-free実gate PASS、限定201 local tests passed、fail/skip0。
H4 science、NPZ stat/hash/load、runtime/cache access、GPU操作は今回0。PM-1 output/registry未作成。

運用statusは`PM1_FINAL_REVIEW_APPROVED_AUTHORIZATION_FINALIZED_AWAITING_USER_LAUNCH`。
本計算は未実行。明示launch後もH4 linear 1.00 Å、STO-3G、DF rank12、8 qubits、T=0.8、
B0 rank4/5 × q=1/2/4/8、delta=0.8/0.4/0.2/0.1、r=K=0、8 signals/16 wrappers、CPU1に固定。
成功/failureでmandatory STOP、研究判断null、PM-2以降未認可。
旧draft manifestはreview bundleのfalse blobへ照合する履歴として不変。
新しいreceipt/gate/test/manifestは`artifacts/resource_applicability/pr2_pm1_finalization/2026-10-05/`。
過去4件stat/hash違反を保持し、現在のaccess0やlocal testsをimmutable CIと混同しない。
以下のdraft/準備statusは当時の履歴である。

## 2026-10-04 Track A PM-1 authorization draft固定 最終review待ち

準備reviewの `APPROVE_PM1_RESULT_PRIOR_AUTHORIZATION_DRAFT` を受け、
[authorization draft](docs/research/pr2_pm1_discard_execution_authorization_v1.md)を別commit
`b6175a0f4fdeb1d2f0cd61c09dce37675624b73a`へ固定した。SHA-256は
`3916f1e050bf4d4174f0b2b5d4d1d82cdb12396f9eefe3a34a75e496e716377d`。
運用statusは `PM1_AUTHORIZATION_DRAFT_FROZEN_AWAITING_FINAL_REVIEW`。
final_review_approved=falseで現runnerは拒否し、最終承認済みとは記録しない。
[最終review依頼](docs/research/pr2_pm1_execution_authorization_external_review_request_b6175a0.md)へ渡す。承認後もboolean一項目の別commit照合・利用者launch指示までSTOP。

source134、sealed plan、固定8候補/16 wrapper、環境は一致し科学sourceは不変。
H4 linear 1.00 Å、STO-3G、DF rank12、8 qubits、T=0.8、B0 rank4/5 × q=1/2/4/8、
delta=0.8/0.4/0.2/0.1、r=K=0。将来の8 signal＋16 wrapper、CPU1、BLAS1上限を変更しない。
限定201 local testsが再びpassed、fail/skip0。actual draft拒否、conditional mock positive、9改変拒否と
uncommitted flag模擬変更拒否を別監査へ記録した。mockをproduction PASSと呼ばない。
NPZ/runtime/cacheアクセス、H4 signal/build/compile、science runner、GPUは0。output/registryは未作成。
過去4件stat/hash違反の監査を保持し、immutable CIや外部再現とはしない。
成功/failureどちらもmandatory STOP、研究判断null、PM-2以降未認可。

## 2026-10-04 Track A PM-1：契約・実装準備完了、本実行未認可

[PM-1契約](docs/research/pr2_pm1_nearby_discard_contract_v1.md)と
`artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/`にplan/schema/source/test auditを保存した。
H4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、T=0.8、B0 rank4/5 × q=1/2/4/8、
delta=0.8/0.4/0.2/0.1、r=K=0。将来8 signal＋16 full wrappers、CPU1、random追加0。
保存済み5 development comparatorとのpoint比較だけで、general method最適性やheld-out再探索ではない。

sourceはlocal commit `fd7552edc0334ccf57ecf501a128c85c8d22822a`、134 source hash/blobをplanに結合した。
plan SHA-256は`cae692bee2be748ddbf17bace2a5652613537a244fe94238cdf669f5d2ca5624`、
fingerprintは`144824b70264dd3d7d1d22898afbb1f9e1c248f105ecac8096f31a2c8f78ec9e`。
statusは`PM1_DISCARD_PREPARED_EXECUTION_NOT_AUTHORIZED`。
限定201 local tests（専用49、PM-0 18、helper134）がpassed、fail/skip0。
synthetic2-qubit compileを含む実装検査で、H4 science、科学結果、immutable CIではない。
前attemptのNPZ4件stat/hash、load0は違反として別記録。再開後のNPZ/runtimeアクセス・H4計算・GPUは0。
M1/M2 evidence/正式statusは不変。利用者の別指示でprep artifactと関連索引をレビューbundleへ収録する。
[GPTレビュー依頼](docs/research/pr2_pm1_preparation_external_review_request_fd7552e.md)から固定source/plan/監査へ辿れる。
別authorization→最終review→明示launchまでmandatory STOP。PM-2以降も未認可。

## 2026-10-04 Track A PM-0：POSTHOC保存値再解析、追加科学計算0

[PM-0報告](docs/research/pr2_post_m2_evidence_attribution.md)と
`artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/`へ事後成果物を分離した。
H4 linear 1.00/1.30 Å、STO-3G、DF rank12、8 system qubits、T=0.8、ε_complex=0.05。
M1 L_D=0/3/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1。M2はdevelopment固定5構成だけ。
登録集合のq8でもprimary最小はB2 rank3。旧16件selectorのRZ regret0、secondary regret最大2.2554%、
actual six-metric Pareto2件中1件を落とすがproxy64件は両方保持する。
共通5構成のP感度ではM1/M2とも大きいPでB1へ移り、候補domain差をgeometry効果と混同しない。
B0のpure discard/PF分解はexact truncated-H signal欠測として保存した。

入力JSON5件/source4件を基準commit `b6e65c6123475add5e620ec1064f361378bead95`と前後byte照合し、
専用18 testsがlocal passed、fail/skip0。今回のレビューbundleへ収録するPOSTHOC bookkeepingで、新しいscience evidence、
immutable CI、外部再現、formal CIではない。NPZ/runtime/cache access、signal、sampling、build/compile、GPUは0。
M1/M2結果・正式status・科学source・manifestは不変、中央台帳にはPM-0 entryのみ追加する。
PM-1近接discard試験案、PM-2 ε/P感度、PM-3構造/energy接続は未認可。mandatory STOP。

## 2026-10-04 PR-2 M2一回実行・結果照合完了：`TRANSFER_SUPPORTED`、mandatory STOP

最終外部review承認と利用者の実行指示を受け、source/plan/authorizationを変更せず固定5構成のM2を一度実行した。
H4 linear 1.30 Å、STO-3G、DF rank 12、8 system qubits、`T=0.8`。B2 `L_D=3,q=1,r=4/8,K=2`と
B0 `L_D=6,q=1`、B1 `L_D=12,q=1`は`delta=0.8`、B3 `L_D=0,q=8,r=32,K=4`は`delta=0.1`。
候補、threshold、32 trajectoryを再探索していない。

[M2結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)は196 unique wrappers/checkpoints、
96 trajectories、128 source blobs、signal/cost identity、paired covariance、判定、manifest bytes/SHAを照合した。
computed196、reused0、未解決予約0。5構成が全件accuracy適格、usable B2二件が6指標point Paretoに残った。
primary最小B2／最小endpoint比は0.586090、engineering upper 2SEは0.593584で基準1.10以下。
formal CIではない。result SHA-256は`f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931`、
fingerprintは`d9003ac6e32b2888d69aa1fed226dbef48cf13e1a6f10829e137c136824bb320`。

5 spawned CPU workers、各BLAS1でwall971.824秒、snapshot hash/load各1、GPU query/allocation/kernel0。
pre/postともlocal focused84、helper134 passed、fail/skip0。結果照合は分子NPZを再度開かず、科学計算を追加していない。
結果と監査をresult commitへ収録するlocal execution evidenceであり、immutable CIや外部再現、
methodの一般的最適性、最終総cost評価ではない。
`TRANSFER_SUPPORTED`は固定5構成のtransferだけを支持する。`next_stage_authorized=false`、mandatory STOPで
研究方針の全面reviewへ戻る。追加96、retuning、別geometry/分子、S3、長RPEを自動実行しない。
以下の未実行・未認可・access0の記述は各準備milestone当時の履歴であり、現在の実行状態ではない。

## 2026-10-04 PR-2 M2実行authorization固定 最終review待ち

actual science source `2978e2fea672b7a1ff20cac74269ec9a610159dc`とsource-bound execution planに結合した
[authorization](docs/research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)を別commit
`90a9f24707ec439cd3618cc4ec2616a8caaf1148`へ固定した。JSON SHA-256は
`dbc8b66fad5316004ef404fe8253d3f6cb0f29bf7065502ea995240e5fbd7ff1`。
固定5構成、196 wrappers、最大5 workers、32 paired trajectories、一回限り、全status後STOPを維持する。

運用statusは`M2_AUTHORIZATION_FROZEN_AWAITING_FINAL_REVIEW`。
[最終review依頼](docs/research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)へ渡し、
独立review承認と利用者の実行指示までheld-outを開かない。machine JSONの`M2_EXECUTION_AUTHORIZED_ONCE`は
固定条件の定義であり、このturnのlaunch許可ではない。review待ちは運用barrierで、runnerがreview artifactを
機械検査するとは主張しない。128 source blob一致、plan/authorization/環境identityのzero-science gateを検査する。

local focused84件とhelper回帰134件がpassed、fail/skip 0。分子NPZを読まないsynthetic/保存JSON/mock検査であり、
immutable CI、外部再現、科学transfer結果ではない。held-out resolve/stat/hash/load、H4 science、GPU操作は0。
固定result outputとregistryは未作成である。旧v1/draft/planおよびM1-B1科学artifactを上書きしない。

## 2026-10-04 PR-2 M2契約v2正式freezeと科学実行sourceの実装

usable B2修正をcommit `a529e9434d2e62fe752fdab5bd4c9a63fb15e830`へ固定し、正式contract planを`40888b8`へ保存した。
plan SHA-256は`d2bb5c5e57002fac5e8045f89a048913f4dadd5177d6f6d1465cc40a8755af7c`、fingerprintは
`7880c8fed57a02f30a07ff7463eb65e423098ad9420e38410f365fad2e24cc4f`。
旧v1と明示的draftは保持するが、現行planはCOMMIT_BOUND、科学実行未認可である。

[実装資料](docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)に対応するscience module/runner/testを
追加した。source/plan/authorization/環境gateはheld-out読み込みより前、source commitは全library Pythonへ結合する。
signal/cost candidate identity、32 paired trajectories、196-wrapper上限、usable B2、相関を保持するSE、全status後STOPを
synthetic検証した。one-shot registryで同じauthorizationの再実行を拒否し、snapshot load一回のためresumeは設けない。

execution専用35 passed、関連focused全84 passed、fail/skip 0。小型synthetic baselineのQiskit compile二軸を含む
local implementation evidenceであり、immutable CI・外部再現・M2科学結果ではない。
held-out resolve/stat/hash/load、H4 signal/trajectory/compile、transfer、GPUは0。execution authorizationは未作成で、
source commit・source-bound zero-compute plan固定の後、別authorizationと最終reviewを経るまで停止する。

actual sourceは`2978e2fea672b7a1ff20cac74269ec9a610159dc`へ固定済み。execution plan SHA-256は
`2aa09a927e5ac58ebe417397802ace0e70c0097e8d0c53c05457075a41e85527`、fingerprintは
`ff7ed3d74bf4a0316adecdbac6b633978d71ca87bc1856d03db6fff9a2153276`。
source/plan freezeは完了したが、別authorizationと最終review前で停止中である。

## 2026-10-04 PR-2 M2外部review修正：usable B2統一（未commit draft）

外部reviewの`REVISE_M2_CONTRACT_BEFORE_IMPLEMENTATION`を受け、
[amendment v2](docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)を追加した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
usable集合が空ならNOT_SUPPORTED、usable集合がありendpointが空ならINCONCLUSIVEとする。
20%過小評価でもactual ratioが良い反例、最安unusable B2の除外、schema/plan改変拒否を検査した。

M2専用28 passed、関連focused全49 passed、fail/skip 0。これはdirty-worktreeのlocal implementation evidenceで、
immutable CIまたはtransfer科学結果ではない。v1契約・schema・planのbyte identity、固定5構成・seed・196-wrapper
上限を保存する。v2 zero-compute planは`M2_TRANSFER_CONTRACT_DRAFT_EXECUTION_NOT_AUTHORIZED` /
`WORKTREE_DRAFT`。正式planはsource commitとのbyte照合後にだけfreezeする。
held-out resolve/stat/hash/load、signal、trajectory、compile、transfer、GPUは0のままであり、science source、
別result-prior authorization、最終reviewを経るまで科学実行しない。

## 2026-10-04 PR-2 M2 held-out transfer契約・zero-compute plan固定完了

M1-B1の`CONTINUE_RESOURCE_STUDY`を受け、[M2 transfer契約](docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)を
追加した。development actual ParetoのB2二件とB0/B1/B3代表の計5構成、primary RZ、6指標point Pareto、
10% materiality、重大cost underestimate、4 terminal status、random各32 trajectory、計196 wrapper上限を固定する。

これはsource/schema/test段階のimplementation evidenceであり科学結果ではない。held-out H4 1.30 Åはpath literalを
記録するだけで、resolve/stat/hash/load、signal、trajectory、circuit、compile、transfer、GPUは0に固定する。
現行statusは`M2_TRANSFER_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`。独立review、science source commit、
result-prior authorization前にM2を実行しない。contract source commitは`06b2c32528713a5270ee4915432bcd3898e0e5e1`、
zero-compute plan fingerprintは`e6b6ae0bcf9ff7b97eb61b2703305d793f7d531460456baf2337960fb36fb198`、
focused testsは30 passedである。

## 2026-10-03 PR-2 M1-B1 actual compile map検証完了・`CONTINUE_RESOURCE_STUDY`

H4 linear 1.00 Å、STO-3G、DF rank 12、8 qubits、`T=0.8`、`q={1,2,4,8}`の
development-only M1-B1を、Qiskit 1.3.0、optimization level 1、最大6 workersで完了した。
science runnerは予定どおり`M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW`で停止し、研究判断、
追加96、held-out、transfer、S3を実行していない。

[結果検証](docs/pr2_matched_accuracy_m1_b1_result_validation.md)は、210 compile-map cell、210 checkpoint、
210 candidate-scoped SQLite cache、12,448 wrapper recordを再検査した。6,208 seedの衝突0、
12,128 unique actual circuit transpile、同一candidate内のsemantic cache reuse 320、cross-cell reuse 0を
確認し、各axisの平均・分散・標準誤差を再集計した。validation statusは
`M1_B1_RESULT_VALIDATED_RESEARCH_REVIEW_COMPLETE`、fingerprintは
`c3cf1c084ebfe343d576236de2803c9c69855e0247ca6a2d628c496ee0546214`である。

accuracy適格206 cellのactual six-metric Paretoは`B2-rank3-q1-r4-K2`と
`B2-rank3-q1-r8-K2`の2件。primary RZ point minimumは前者の130,774,896.656で、RZ最小10%内は
全てB2 rank 3、q=1だった。旧16-cell selectorはactual Paretoを1/2件しか保持しなかった。q=8固定の
B2 rank 3最小からmatched-accuracy point minimumへの低下は約80.88%で、状態準備cost `P>=0`の
lower envelopeも全てB2だったため、外部研究判断を`CONTINUE_RESOURCE_STUDY`とした。

ただし1位と2位のRZ差0.741%は32 trajectoryで解像しておらず、厳密なr/K winnerを確定しない。
本結果はlocal development evidenceで、immutable CIまたは外部再現ではない。次は別result-prior
held-out transfer reviewのdraftであり、H4 1.30 Åをまだload、hash、stat、評価しない。

主要identityはM1-B1 result SHA-256
`71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4`、result fingerprint
`504d9c9089726800a291a8259e87b2d37c1fdea046263db6c9582bb659c77975`、validation artifact SHA-256
`c9a05babed99cd1e80eaec5b58e47f25d74513c7ba2e5a00775cfd5959c37a0f`である。

## 2026-09-30 PR-2 M1-B1 execution source・authorization固定、本計算未実行

実行前外部reviewの`REVISE_CONTRACT_BEFORE_AUTHORIZATION`を
[execution contract amendment v2](docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)へ反映した。
12,448 wrapperを実際に生成・compileするmodule/runner/testとresult schema v2を、execution authorizationより
先に実装する。result terminal statusは`M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW`または
`IMPLEMENTATION_GATE_FAILED`に限定し、resource study継続・technical note化・重複停止・inconclusiveの
研究判断はcompile map完成後の外部reviewへ戻す。

source-bound planは実benchmarkと同じtrajectory seed列を固定し、source/compiler/candidate/axis/trajectoryを
cache keyへ含める。cacheはcandidate別、checkpointはtask fingerprint完全一致時だけ再利用する。専用synthetic
testは8 passed。M1-B1本計算は未実行で、追加96、held-out、transfer、S3は未実行・未承認である。

actual execution source commitは`33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62`。source-bound execution plan v2の
SHA-256は`5afc94fac0571b38c74b0b00cfcf68e34e491a5fc65ef579d3ec884d079e5aa5`、fingerprintは
`17c91d41e77d7c085629b60ea470abc87e9590f448ba9bcdfa4342410cd89607`である。
[execution authorization v1](docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)とmachine JSONは
一回のbounded runだけを固定した。実runnerのzero-compute authorization gateはPASSしたが、bundleの外部review前で
trajectory sampling、circuit build、compileは0のままである。

## 2026-09-30 PR-2 M1-B1 bounded compile実行前契約・zero-compute plan完了

[M1-B1契約](docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)は、M1-Aの
`SELECTION_LIMITED`を旧proxy selectorの監査結果として保存したまま、`PROCEED_BOUNDED_COMPILE_EXPANSION`
を固定した。M1-Aでaccuracy適格だったrandom B2/B3 194 fingerprintを追加・除外せず、各32 trajectoryを
cosine/sine二軸で共有する12,416 wrapperと、B0/B1全16 cellの32 wrapper、総上限12,448 wrapperを定めた。
B0/B1のaccuracy不適格4 cellはbaseline completeness用にcompileするが、matched-accuracy frontierへ入れない。

standard-library-only実装はcandidate、axis、trajectory seed、compiler、source commitを含むcache/checkpoint
identityを生成し、cross-cell reuseを禁止する。source commitは
`12281687fe13c13ac19688d328f9da26a1d63f34`、zero-compute plan fingerprintは
`94592dbddce9b21cfe9fd31c61c578943264002655379e255aa36161072b5814`、file SHA-256は
`d0c2234cb787b57af85457a6232504be39bbb2140d60f9b67bbda3d252622e6c`である。専用testは9 passed。
snapshot load、signal再評価、trajectory/occurrence sampling、circuit、compile、quantum shot、GPU、held-out
accessは全て0である。

現行statusは`M1_B1_BOUNDED_COMPILE_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`。M1-B1科学実行には別の
result-prior authorizationと実行前reviewが必要である。B1は32 trajectoryのactual compiled resource mapで
停止し、追加96、held-out候補確定、transfer、winner精密化、S3へ自動進行しない。

## 2026-09-30 PR-2 M1-A `SELECTION_LIMITED`完了

[M1-A結果](docs/pr2_matched_accuracy_m1_a_validation.md)は、H4 linear 1.00 Å、STO-3G、DF rank 12、
sector 8 qubitのdevelopment snapshotで208 base＋2 r64 boundary候補、計210候補を評価した。206候補が
accuracy適格で、random B2/B3は194/194適格だった。16-cell selectorに対してproxy frontierは64件、
未選択frontierは52件残り、理由`unselected_proxy_nondominated_candidates`で正式status
`SELECTION_LIMITED`となった。

hard barrierはcompile job 0、circuit/compile/trajectory/full wrapper/quantum shot 0で停止した。held-out
path/stat/hash/load/signal/cost/rankingも0、S3は未承認である。result fingerprintは
`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`、file SHA-256は
`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`。pre/post focused testsは各26 passed、
専用validatorと凍結JSON schemaも通過した。winnerは未確定で、M1-Bへ進まない。

## 2026-09-30 PR-2 M1-A v1停止・v1.1 result-prior再認可

[M1-A execution authorization v1](docs/research/pr2_matched_accuracy_m1_execution_authorization_v1.md)で、保存済み
development H4 linear 1.00 Å、STO-3G、DF rank 12、sector 8 qubitだけを対象とするcompile-free M1-Aを
結果前固定した。base 208候補、r64最大4候補、signal最大212、development NPZ load 1、単一process、
BLAS thread 1を上限とする。M1-A sourceはQiskit circuitを作らず、dense small-system actionでsignal、bias、
normalization、analytic shot、action proxy、selector、hard barrierまでを評価する。

v1初回実行は固定`K={2,4}`に暫定`truncation_tolerance=1.0`を渡した実装不整合により、result作成前に
停止した。selector結果、compile、circuit、trajectory、量子shotは0である。
[v1.1再認可](docs/research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md)は候補・K・signal・selector・
thresholdを変えず、固定Kの一step残差を受理するself-consistent toleranceだけを既存validation規則で
構成する。現行statusは`M1_A_RETRY_AUTHORIZED_AFTER_IMPLEMENTATION_GATE_FAILURE`で、M1-A結果は未作成である。
`SELECTION_LIMITED`なら全compile counter 0で停止する。clearでもM1-A artifactを
commitしてbyte固定し、別のresult-prior M1-B source/authorizationを固定するまでtrajectory/circuit/compileを
開始しない。held-out path access/NPZ load/signal/cost/ranking、S3、量子shotは0・未承認である。focused
v1.1 authorization gateは26 passedで、immutable CIまたは外部再現ではない。

## 2026-09-29 PR-2 M1前最終amendment・precompile hard barrier完了

[M1前最終amendment](docs/research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)で、Cugini--Atif--
Subasiのrandomized protocolに対するcost–variance共同最適化と、Kanasugi et al.のsingle-ancilla
Trotter QPE・部分ランダム化・化学end-to-end resource estimateを既知貢献として追加した。一般的な
importance samplingまたは化学QPE resource optimalityを新規claimから除外し、固定DF-prefix候補の
matched-accuracy・discard baseline・full-wrapper比較という限定scopeで`PROCEED_RESOURCE_STUDY`を維持した。

M1をM1-A signal/selectorとM1-B direct compileへ分けるstandard-library-only hard barrierを追加した。
M1-Aで`selection_limited=true`なら正式statusを`SELECTION_LIMITED`とし、deterministic/discard/randomの
compile jobを一件も生成せず、circuit build、compile、full-wrapper counter 0のままmandatory STOPする。
synthetic dry-runではlimited fixtureがこの停止分岐に入り、clear controlだけが16+16 cellのplan identityを
生成した。科学counterは全て0、専用testは10 passedである。

現行statusは`M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED`。M1 signal、trajectory、compile、
held-out、S3は未実行・未承認である。次はM1-Aと条件付きM1-Bを固定する別execution authorizationであり、
本amendmentから自動実行しない。

## 2026-09-29 PR-2 M1実装契約・zero-compute dry-run完了

[M1実装契約](docs/research/pr2_matched_accuracy_m1_implementation_contract_v1.md)として、standard-library-only
module、non-overwrite runner、専用test、dry-run/result schema、machine authorizationを固定した。候補は
B0 12、B1 4、B2 144、B3 48のbase 208件、r64境界は最大4件、signal候補上限212件、random direct
compile上限16 cellである。candidate fingerprintはsnapshot/Hamiltonian/state、method/rank、`T,q,delta,r,K`、
compiler、wrapper、seed policyを含む。

zero-compute dry-runはbase 208件、boundary request 4件、synthetic selector選抜16件を生成し、
`SELECTION_LIMITED`経路を含む停止規則を機械検査した。専用testは7 passed。development/held-out NPZ load、
分子計算、signal、trajectory、circuit build、compile、量子shotは全て0である。synthetic proxyと選抜結果は
科学値でもM1候補選択でもない。

現行statusは`M1_IMPLEMENTATION_CONTRACT_FROZEN_SCIENCE_NOT_AUTHORIZED`。次は独立reviewであり、M1を
実行する場合はsource identity、process/resource上限、output、test gateを別authorizationへ固定する。
held-out、S3、追加geometry、H12、長RPE、最終総costは未承認のままである。

## 2026-09-29 PR-2 matched-accuracy再設計契約固定・M1未承認

S2後のmandatory reviewを完了し、PR-2を新しいprefix法またはrank 6の優位性主張から、固定DF snapshot上の
discard／deterministic／partial／random-dominantを同じ複素signal精度で比較する限定resource studyへ
狭めた。[M1前先行研究gate](docs/research/pr2_matched_accuracy_prior_art_gate_v1.md)は、最接近研究の
既知貢献を除外したうえで`PROCEED_RESOURCE_STUDY`と判定した。これは新algorithm、投稿可能性、
一般的優位性の判定ではない。

[M1前契約](docs/research/pr2_matched_accuracy_resource_contract_v1.md)は`q={1,2,4,8}`と`delta=T/q`、
可変q correctness、signal/cost fingerprint、random direct-compile最大16 cellの結果前選抜、
`SELECTION_LIMITED`、held-out前の最大5構成・数値判定の別freezeを固定した。

現行statusは`PR2_MATCHED_ACCURACY_CONTRACT_FIXED_M1_NOT_AUTHORIZED`。新しい分子計算、signal、trajectory、
compile、量子shot、held-out loadは0件である。旧S2の`S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`、
B2/B3 frontier、rank 3/9 control、10% materiality、旧S0 STOPを変更しない。次はM1 source/runner/test/
schemaと機械可読dry-runを別commit・別authorizationへ固定するかのレビューであり、M1科学計算、S3、
H12、長RPE、最終総costへ自動進行しない。

## 2026-09-29 PR-2 V4/S2 development比較完了・mandatory STOP

結果前authorization v5とsource commit `e098c54`に従い、H4 linear 1.00 Å、STO-3G、DF rank 12、
`T=0.8`、`delta=0.1`、`q=8`のdevelopment-only比較を完了した。6-worker並列層はcell、seed、
32/96 trajectory拡張、段階barrier、canonical結果順を変えず、source commit `16331cc`から実行した。

正式statusは`S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`。rank 6 B2 `r=1,K=2`のno-prep RZ workは
`1.236973821e9`で、B0/B1に対するratio intervalはそれぞれ`[0.370062,0.370218]`、
`[0.552534,0.552768]`だった。B3 `r=32,K=4`は`1.179054305e9`で、B2/B3 intervalは
`[1.035433,1.063175]`。従って10%基準のprimary frontierはB2/B3の2候補で、materially dominating
endpointはない。rank 3 controlは`6.882866351e8`だったが、事前規則どおりprimary winnerへ混ぜない。

result fingerprintは`51fb92fdbcedb67299964eddd25e81c1faeaa55d7f7966765600a05e71d41a49`、
artifact SHA-256は`bbe665724af438d242569b020ab6148dacce84716a75681949bea22c632dd27c`。
development NPZ load 1、held-out NPZ load 0、full wrapper compile 3,204、random trajectory compile
1,600、signal evaluation 28、量子shot 0、分子計算0。専用testは15 passedである。

`mandatory_stop_reached=true`、`S3_authorized=false`、`automatic_next_stage=null`。held-out 1.30 Åの
signal/cost/rankingは未開封で、transfer、backend/noise、状態準備実回路、H12、長RPE、最終総costは
未検証である。これはlocal validationでありimmutable CIまたは外部再現ではない。次は追加計算ではなく、
rank 6 transfer契約を維持するかsplit/resource-map研究へ再設計するかの方針reviewである。詳細は
[S2検証報告](docs/pr2_v4_s2_development_validation.md)を参照する。


## 2026-09-28 PR-2別系列 V1–V3通過・V4 review待ち

旧S0の`STOP_INPUT_REPRODUCTION_MISMATCH`と`S1_authorized=false`を維持したまま、外部レビューの
`AMEND_AND_RESTART_FROM_NEW_S0`に従い、別系列`pr2-rebaseline-de7a5492-v1`を開始した。V0の
read-only一巡監査では旧pilot完全入力を回収できず、性能未評価の最初の保存済みdevelopment snapshotを
新系列入力として固定した。specification commitは`30ea857`、source commitは`ef86868`である。

保存済みH4 linear 1.0 Å、STO-3G、DF rank 12、N=4、N_alpha=N_beta=2、S_z=0入力に対し、V1の
raw/layer hash、二回load、shape/dtype、Hermiticity、state/sector、Rayleigh residualを通過した。V2の
rank 3/6/9では、B2-G/B2-Wのexact cover、sampling sign、確率和、identity coefficient、repeat
preparation、`H_D+H_R=H`再構成を全て通過した。G/Wは全rankでordered prefixまで同一だったため、
`collapse_B2_G_and_B2_W=true`である。

統合statusは`S0_PRIME_PASS_V4_REVIEW_REQUIRED`。result fingerprintは
`b210b394e9cd5a8eded947b0fd12cefe19ce9f27eb6b8140b3e863df73ea7961`、artifact SHA-256は
`0eb22c813eb838169eb455334146140467ebbc5636db78bd923b1e6bdaed46d8`。専用testは7 passed。
分子計算、signal、trajectory、compile、quantum shot、held-out NPZ loadは0件である。

`V4_authorized=false`、`S1_prime_authorized=false`、`automatic_next_stage=null`として停止した。
これはresource winner、PR-2優位性、held-out transfer、S2/S3、最終総costの証拠ではない。詳細は
[V0–V3結果packet](docs/research/pr2_v1_v3_result_packet_ef86868_20260928.md)を参照する。

## 2026-09-28 PR-2 S0 input reproduction不一致・mandatory STOP

[amendment v3](docs/research/pr2_s0_s1_execution_amendment_v3.md)とauthorization manifestをcommit
`e9bffb8`、S0/S1実装と専用testをsource commit `c644925`で結果前固定した。専用・関連test packetは
`123 passed`である。

S0はH4 linear 1.00 Å、STO-3G、8 qubits、4-electron singlet、DF rank 12 development inputと、H4
1.30 Å held-out geometry inputを生成・freezeした。固定環境versionは完全一致したが、developmentのcanonical
Hamiltonian hashはpilot expected `d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`に対し
observed `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`となった。

従ってterminal statusは`STOP_INPUT_REPRODUCTION_MISMATCH`、`S1_authorized=false`である。分子build/
ground-state solveは2件、signal evaluation、circuit compile、trajectory sampling、quantum shotは0件。
prefix identityは実行せず、held-out signal/cost/rankingも開封していない。S1/S2/S3へ進まない。

限定診断ではpilot source hashとpackage版が一致し、ground energy差は約`1.38e-14` Ha、rank 3/6/9 residual
$\lambda_R$差は`1.2e-15`以下だったが、byte-level Hamiltonian/tail hashは不一致だった。近似一致を同一snapshot
扱いせず、旧pilot結果と新snapshotを混ぜない。S0 result fingerprintは
`6d44888a1b806bc3b6418b49a18fdbcee09b621dd345838dc0d426abb1005182`、file SHA-256は
`cf082a81ed70dcee774906ff2391683811cbdf544c1217127e100a977a11fbd7`。詳細は
[S0停止報告](docs/research/pr2_s0_reproduction_stop_c644925.md)を参照する。これはlocal dirty-worktree evidenceで、
immutable CIまたは外部再現ではない。

## 2026-09-27 FR研究完成フェーズ契約（文書のみ）

FR-R1bの`MECHANISM_ONLY_NO_PRACTICAL_GO`と強制停止を変更せず、
[研究主張・証明義務・完成原稿契約](docs/research/fr_research_claim_and_manuscript.md)を固定した。
中核はC1の正scalar・方向依存補正の分離と、C2の同情報strict improvement／改善不能条件である。
resource designのC3は、先行研究監査と証明義務T1--T4を通過した場合だけ行う条件付き応用とした。

これは数値検証ではない。新しいHamiltonian評価、対角化、RTE sampling、回路compile、artifact、testは
いずれも0件で、FR-R1bのfingerprint、gate、結果を変更していない。先行研究を定理単位で照合し、
`PROCEED_THEORY`、`TECHNICAL_NOTE`、`STOP_NEW_METHOD`、`ONE_OPEN_ITEM`の一つへ収束するまで、
FR-R2、H4/H12、長RPE、新規gridを開始しない。

## 2026-09-27 FR-R1b 非一様4×4検証

結果前に凍結した[FR-R1b事前登録](docs/research/fr_revision_nonuniform_preregistration.md)どおり、
20 matrix conditions、61 state rows、2 semantic controlsを実行した。610 method record中227が適用可能で、
soundness違反は0。非一様$\nu=0,0.5$ではHermitian radial widthが正、involution controlの$\nu=1$では
0となり、$q=4,8$の可換・非可換条件で同情報FR境界がnorm境界より厳しいwitnessを8件確認した。

ただし位相予算$10^{-2},10^{-3},10^{-4}$ radでFRだけが認証する条件は0件だった。R0--R4/R6/R7は
通過、R5 decision relevanceは不通過で、判定は`MECHANISM_ONLY_NO_PRACTICAL_GO`である。
事前登録の強制停止に従い、FR-R2、H4/H12、compile、Monte Carlo、長RPE、最終総costへ進まない。

result fingerprintは`affac0ae8132450ccb2de3512b6a463a3f9d7b1ac8a6f12cc38303ac75e891d4`、
file SHA-256は`e2a6f9326951fe67e979022dc733704e0c036d3342ab3cb317c5f77b885aeebc`。
専用testは`5 passed`、FR系列関連testは`13 passed`、全suiteは
`628 passed, 2 skipped, 4 warnings`で失敗0。詳細は[FR-R1b結果](docs/fr_revision_nonuniform.md)。

## 2026-09-26 FR-R1a 正scalar事後再解析

凍結済み[FR-R1a計画](docs/research/fr_revision_fr1a_posthoc_plan.md)に従い、完了済みFR-1の
33条件・99状態を決定論的に再構成した。元result、凍結事前登録、計画hashは全て一致し、
state fingerprintは99/99一致、最大数値差0だった。正scalar処理を含む9手法を全状態へ適用した
891 method recordで位相上界・物理半径下界の違反は0である。

主5行では`SCALAR-NORM-COMMON`が`SCALAR-FR-COMMON`より全て厳しく、共通scalar比較と
各法最適化比較のどちらにも、FRだけが固定位相予算を通るone-sided certificationはなかった。
I2 `DENSE-ORACLE`だけの片側認証も0で、分類は`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`である。
これは事後説明監査であり、旧G2不通過と`GO_FR2_MECHANISM_ONLY`を変更せず、新しい研究GOを
認めない。このFR-R1a実行時点ではFR-R1bは未実行だったが、翌日の独立結果は上節に記録した。
FR-R1a自体のH4/H12、compile、Monte Carlo、RPE総costは0件である。

result fingerprintは`5cc5c29656b9ceb69a00674cee5987b93c7cc8d9403892c57b341a51c878a18a`、
file SHA-256は`3fab6acdde3798dc7005101713ecaa708f5cf0b5fc24d8c77f80939dcaa6a610`。
専用testは`4 passed`、元FR-1との関連testは`8 passed`、全suiteは
`623 passed, 2 skipped, 4 warnings`で失敗0。詳細は[FR-R1a結果](docs/fr_revision_fr1a_posthoc.md)。

## 2026-09-26 FR-R1a/R1b 計画固定（実行前履歴）

[FR-R1a事後計画](docs/research/fr_revision_fr1a_posthoc_plan.md)は既存FR-1 artifactのfingerprintと
33条件・99状態を固定し、正scalar、共通γ、最適scalar norm、I0/I1/I2を再集計する。ただし
posthocであり、旧G2不通過と`GO_FR2_MECHANISM_ONLY`を変更せず、新しい研究GOにも使わない。

[FR-R1b事前登録](docs/research/fr_revision_nonuniform_preregistration.md)は非一様4×4について、
18 primary＋負時間/K4の20 matrix conditions、61 state rows、2 semantic controlsを固定した。
同一状態を全qで再利用し、位相予算は`1e-2,1e-3,1e-4 rad`、物理半径下限は0.2、
入力certificateは0.8である。R0--R7と終了分類を結果前に固定し、FR-R1b後は必ず停止する。

この節の時点のstatusは`PREREGISTERED_NOT_RUN`だった。後続でFR-R1aだけを完了し、FR-R1b契約は
その結果で変更していない。
FR-R1a計画SHA-256は`a0aa2e75d2e304e008646f138211ed1e4da4d98def697f8503b07c3e7b56c95f`、
FR-R1b事前登録SHA-256は`1bc2a72fa8dec98e2bdbe3504e65366d8daa93f7a7797837622ba7545607515e`。

## 2026-09-26 FR-R0 正scalar分離・構造比較契約

完了済みFR-1の`GO_FR2_MECHANISM_ONLY`を変更せず、正の共通scalarを位相誤差から除く代数、
$\Gamma_c/\mathcal B$を物理半径へ戻す規約、I0/I1/I2情報層、共通$\gamma$と各法最適化の
二層比較を[正式契約](docs/research/fr_revision_scalar_structure_contract.md)へ固定した。FR-R1bのGOには
soundness、非一様系、同情報利益、事前位相予算またはK/r選択差、oracle非依存、機構整合を全て要求する。

FR-R0作成時statusは`FR_R0_COMPLETE_FR_R1_NOT_PREREGISTERED_NOT_STARTED`だった。後続の
FR-R1a/R1b計画固定により現行statusは上節へ移ったが、FR-R0自体が文書契約だけで、新しいHamiltonian、
対角化、RTE sampling、compile、test、数値artifactが0件だった事実は変わらない。旧FR-1のG2不通過を
再分類せず、H4/H12、長RPE、最終総costへ進まない。

## 2026-09-26 finite-RTE phase/radius FR-1

事前登録した2×2 fixed gridを変更せず実行した。33条件、99状態評価、495個の適用可能method recordで
位相上界・信号半径下界の違反は0だった。paired-eventのordinary平均最大residualは
`4.4431e-16`、controlled relative-phase最大residualは`3.3379e-16`である。負時間、K=4、
非対称配置を含め、G0/G1/G3/G4は通過した。

主判定の`analytic_mixture_state`へ利用可能な$\underline\rho=0.8$だけを与えると、
`PROPOSED-AVAILABLE`対`STRONG-NORM`の位相上界比は最小0.631998で、固定50%基準を通らなかった。
提案法だけが`1e-3 rad`を認証する点もなくG2は不通過。一方、$\rho=1$またはdense真値を使う
`PROPOSED-REF`では改善したため、事前規則どおり`GO_FR2_MECHANISM_ONLY`とした。
これは利用可能情報による実用的GOでなく、FR-2は開始しない。

artifact fingerprintは`d96b200163f3a432652656ae97c65c837e01fe81323cd69528373dff16ce6152`、
file SHA-256は`6a81a0ba6e39ba0f5d79ba026a2a45c3c65709e071e9c6fa5606b3e99a89b0f7`。
完全な実行契約は凍結FR-1事前登録SHA-256
`bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3`である。凍結親契約は後の監査で
末尾切断が判明したため不変の監査履歴として残し、現行親契約で修復したが、FR-1のgrid・gate・判定には影響しない。
専用testは`4 passed`、全suiteは`619 passed, 2 skipped, 4 warnings`で失敗0。artifactはdirty worktreeで生成したlocal evidenceであり、immutable CIではない。
H4/H12、circuit compilation、Monte Carlo sampling、RPE総cost、最終costは0件。詳細は
[FR-1検証](docs/finite_rte_phase_amplitude_validation.md)。

事前検証カタログの実施ID・work packageと、以下のstatus、専用文書、artifact、runner、testの対応は
[事前検証カタログ実施証拠索引](docs/research/prevalidation_catalog_evidence_map.md)を参照する。

## 2026-09-26 P-D S1 固定artifact事後再解析

S1 v2のfingerprintとfile SHA-256を固定し、保存済み308候補だけを再集計した。一次の
`Case B + undetermined_boundary`と停止statusは変更していない。主baselineをB1b/B2/B4、B0/B1aを
診断用ablationとし、各scopeの選択、regret、false acceptance、B4最良から5%以内、B1aの`m_D`列、
nested/native work内訳を別schemaへ保存した。

B1b/B2/B4はnested/native/combinedの各scopeで同一候補を選び、false acceptanceなし、B4 regret 0。
5%近傍はnested 5候補、native/combined各1候補で、B2 objective相対誤差は最大0.03132%だった。
従ってこの固定候補集合とB4参照に限り、主baselineをCase A相当と事後解釈する。ただし全候補では
B2 proxy受理/B4不適格が45件あり、全域のfinite feasibility判定能力は支持されない。

B1a選択の固定tailはunit signal radiusでもfinite位相上界`1.8178593e-6 rad`が予算`8e-7 rad`を超え、
`m_D`だけの追加では採用中のB4上界を満たせない。nested/native B4 objective比13.2297は主に
deterministic action差だが、解析的component-action proxyなのでcompiled circuit優位性とはしない。

artifact fingerprintは`976212ee45a472bf0091064e8baf3eb7a861f2c240cdcfda445e64b4c72e0245`、
file SHA-256は`d8657c52e609e524c4e43a3004948ffb3fd54e6f60219442638b847ade3e699b`。専用testは
`4 passed`、全suiteは`615 passed, 2 skipped, 4 warnings`で失敗0。新しいHamiltonian、対角化、RTE sampling、compile、H12、長RPE、最終総costは0件。
P-D S2には進まず、R3は先行研究差分と別契約を固定する前の候補段階である。詳細は
[P-D S1事後再解析](docs/research_direction_pd_s1_posthoc.md)。

## 2026-09-26 P-D S1 公平PF再最適化

S0で主RQ、既知baseline、共通比較契約、停止規則を固定した。S1はH4 linear chain、1.0 Å、
STO-3G、8 qubit、4-electron sector、DF rank 12、`L_D=3`、固定5公式を対象とし、共通物理時間
`T=0.8`、総位相誤差予算`8e-7 rad`でnested/native構成を比較した。primary gridは
`delta={0.1,0.2,0.4}`、nested `m_D={8,16,32,64}`、`R={16,32,64,128}`、K2で、
事前登録規則による一段境界延長と限定K4感度だけを追加した。

one-shot workを含むB1b、leading absolute-tail-time modelのB2、finite modelのB4は全scopeで同じ
new fourthを選んだ。nestedは`delta=0.2,m_D=16,R=16`、native/combinedは
`delta=0.2,R=16`で、B2のB4 regretは0だった。限定K4でもformula・delta・m_D・Rは不変で、
decision-relevantなfinite補正は確認されなかった。

outer-stageだけを見るB1aはnested/combinedでfinite-infeasibleな`delta=0.4,m_D=128,R=16`を
選び、一段延長後も`m_D`上限依存が残った。従って一次分類はCase Bだが、正式statusは
`stop_s1_undetermined_boundary_no_go_decision`である。Case Bを生じさせたのはB1aだけで、B1bは
B2/B4と一致する。Case C/Dの証拠はなく、S2へ進まず計算を停止した。

v1本実行は高段PFでtail occurrence数が`R=16`を超える配分不能点をinfeasibleとして保存せず、
result生成前に停止した。grid・閾値・分類規則を変えず修正し、v1 expectedを保持したままv2へ
非上書きで再固定した。v2 expected fingerprintは
`e3eacbb9d8928f781df7709208f048c59aaab7052304e4eae4adc346bbf6d0d5`、result fingerprintは
`0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2`、result file SHA-256は
`6b8de6e255eb0796d93398c767017c2899837a63c7230e9956fb2d9beedbeaec`。専用testは`5 passed`、
全suiteは`611 passed, 2 skipped, 4 warnings`で失敗0、artifact validatorも通過した。これはdirty worktreeのlocal evidenceで、immutable CIまたは外部
独立再現ではない。H12、長RPE、compiled total cost、sampled H4 finite operatorは未評価である。
詳細は[P-D S1公平再最適化](docs/research_direction_pd_fair_comparison.md)。


## 2026-09-25 P-D現実化 Go/No-Go gate

H4 linear chain、1.0 Å、STO-3G、8 qubit、4-electron sector、DF rank 12、固定5公式、
診断delta 0.2、主判断delta 0.4を用いた。D1負時間14 task、D2 `L_D=3,4`、D3 fresh
`L_D=5`を結果前に固定し、energy tolerance `1e-6 Ha`、minimum weight 0.9995、tail burden
20%削減、内部`H_D` 32 substep等の閾値を結果後に変更していない。

D1は14/14通過した。ordinary oracle residual最大`3.3314e-16`、signed adjoint 0、controlled
residual最大`6.6613e-16`、identity relative-phase residual最大`3.4694e-18 rad`、sampled mean
最大絶対誤差`5.1531e-4`、最大standardized residual 2.4874だった。

fragment内部`H_D`誤差を戻した主判断deltaでは、`L_D=3,4`のenergy-only選択はMorales 8次、
tail-aware選択は新4次のまま残った。fresh `L_D=5`ではenergy-onlyがMorales 8次、tail-awareが
二次となり、`log B_K`を96.580%、stage proxyを6400から768へ減らした。全splitのfragment再構成
residualは0、最大unitary defectは`1.4382e-11`、最小target weightは`0.9999959245`である。
D1--D3は全て通過し、statusを
`advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign`とした。

P-Dを正式主研究候補としてRQ・新規性・最小着地点・必要な本検証の再設計へ進めるが、計算は
ここで停止する。D1はdense small-matrix oracleでcompiled Qiskit controlではない。D2/D3の
各`H_R` occurrenceはexactで、sampled H4 operator、compiled depth、long RPE、最終総cost、H12、
backend/noise、global PF optimality、科学的優位性は未評価である。

v1本実行は旧P-D負担表にないdelta 0.2を参照する技術的`KeyError`でresult生成前に停止した。
task、seed、thresholdを変えず修正し、v1 expectedを保持したままv2へ非上書きで再固定した。
v2 expected fingerprintは`8924d637e52b03900f32e2f167e63593cf9729b4b78d4dee3af87fca661183f4`、
result fingerprintは`805a17f95497a4d61286748a126c01b1235fbe0d987528be86ea3938700b9ede`、
file SHA-256は`59e8019805e429280a5f6338e38a1cbfd803ae8b51c30e87a965a2ef3dcb694d`。
専用testは`4 passed`、全suiteは`606 passed, 2 skipped, 4 warnings`で失敗0。これはclean
worktreeから生成したlocal evidenceだが、immutable CIまたは外部独立再現ではない。詳細は[P-D現実化Go/No-Go](docs/research_direction_pd_realization.md)。

## 2026-09-25 P-D energy係数・random-tail負担Pareto監査

H4 linear chain、1.0 Å、STO-3G、8 qubit、4-electron sector、DF rank 12の固定snapshotで、
二次、標準/新四次、Yoshida/Morales八次の5公式を比較した。`L_D=3`は探索を開示したdevelopment、
expected-task固定前に未確認の`L_D=4`をblind holdoutとした。非可換3次元toyの局所/global
operator次数と、`H_D`内部だけの高次化が外側Strangの二次を変えないことも確認した。

`delta=0.4`、energy tolerance `1e-6 Ha`では、両splitともenergy-only選択は
`8th(Morales)`、tail-aware選択は`4th(new_2)`となった。新4次はMorales 8次より
`Gamma_R`が56.702%小さく、full exponential stage数も35から11へ減る。係数・次数、
nested/global区別、finite-RTE配分、development/blind逆転、unitarity、target weightの7 gateは
全て通過し、statusを
`advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation`とした。

これはexact dense `H_D/H_R`二blockと解析的finite-RTE normalizationのlocal dirty-worktree
evidenceである。負のtail係数を含むsampled operator、identity/control位相、実際のfragment列での
`H_D`内部誤差、compiled回路、RPE総cost、H12、全PF family最適性、科学的優位性は未評価である。
従ってP-Dは正式主題ではなく条件付き候補で、次はsigned-time RTE oracleと内部`H_D`誤差だけを
検証する。

expected fingerprintは`f302dafce37fb90f3acfe32aa83edf563dd1609015a1b3d50972880e13407c7d`、
result fingerprintは`846362ab808e9b26e5648f7f9d12d541dd01a6f9c2954e45d9194b4dfef8d835`。
専用testは`4 passed`、P-D追加後のlocal全suiteは`602 passed, 2 skipped, 4 warnings`、失敗0。
これはimmutable CIまたは外部独立再現ではない。詳細は
[P-D energy・tail Pareto監査](docs/research_direction_energy_tail_pareto.md)。

## 2026-09-25 P-C geometry tracking・breakdown validation

H4 linear chain、STO-3G、8 qubit、4-electron sector、DF rank 12、`L_D=3`、
二次partial-`S_2` exact-tail参照を固定し、0.70--1.60 Åの8 geometryでindependent/tracked
prefixを比較した。trainingは0.80/1.00/1.20 Å、blindは0.70/0.90/1.10/1.40/1.60 Å、
fit deltaは0.025/0.05/0.10、holdout deltaは0.20である。compile-before expected taskは16件、
本計算は16/16完了した。

tracked prefixは全点で独立先頭3 fragmentと同じだった。blind coefficient予測の15%基準は3/5点、
pair予測の15%基準は1/4 pairだけが通過した。0.90/1.10 Åの係数誤差は6.135%/3.461%だったが、
1.40/1.60 Åでは33.653%/123.245%へ増えた。分類可能blind 4点のcontinuity診断正解率は50%で、
stretch側2 breakdownをflagできなかった。

固定7 gate中、representation integrity、delta holdout、nontrivial cancellationの3 gateだけが通過した。
thresholdを変更せずstatusを`stop_pc_current_h4_family_as_primary`とする。局所P-C pilotの
0.80--1.20 Å結果は保持するが、同じH4 pathへの点追加で主張を復活させない。P-A interval、
P-B current grid、P-C current H4 familyはいずれも停止点に達し、現時点でA/B/Cに確認済み主題はない。

expected fingerprintは`cbe260750d081316070d3684a2a24194d91d029ad700a54cf4f61152f2ed4a4e`、
final fingerprintは`26845effe8efda56390aabdf9e40d61fa3a033e3ac7e6ff6911e2e124156f07a`、
final file SHA-256は`58e31d3877d45e91e9c6c9f4238d2c1f876f02bb875643813a126f283f2dd2b7`。
専用testは`4 passed`、訂正済み先行P-Cと合わせて`7 passed`、関連testは`14 passed`。
全suiteは`598 passed, 2 skipped, 4 warnings`で失敗0だった。
結果はlocal dirty-worktree evidenceであり、immutable CIまたは外部再現ではない。
詳細は[P-C geometry tracking・breakdown validation](docs/research_direction_geometry_tracking_breakdown.md)。


## 2026-09-25 P-A nondegenerate mechanism validation

形式化監査後に事前登録した明示的`one_segment_per_source_run` baselineとの比較を完了した。
固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`、$\delta=0.02$、
Qiskit 1.3.0 opt1で、training fragment 3/5/7とblind fragment 4/6/8を分離し、5種類の
forced-support profileを全てTaylor order 2で評価した。

training 15、blind 15の全30 taskで`interval_union_dp`は一区間baselineと同じplanを選び、
run内分割、plan変更、RZ改善rowはいずれも0件だった。blind pooled RZは4,378対4,378で、
RZ depth、CX、depth、circuit sizeも完全に一致した。5 operator probeの最大差は
$8.327\times10^{-16}$でrelative ancilla phaseも一致した。

事前固定した7 gate中、order-2 coverage、最大個別RZ悪化、operator同値性の3 gateだけが通過し、
分割のbasis/profile移送、25% plan変更、2% pooled RZ改善の4 gateは不通過だった。thresholdを変更せず、
statusを`stop_pa_interval_dp_as_primary_and_return_to_pc`とする。P-Aの既存run-level
full/support-union改善は保持するが、interval subdivisionを独立研究寄与として主張しない。
この時点ではP-Cへ戻った。後続tracking・breakdown validationによりcurrent H4 familyのP-Cも
固定停止条件に達した。

expected-task fingerprintは`e8b064e9821fae5c2e7a44d0c98d3f9a0ed973a4cd87945fb051151e446a96fc`、
final artifact fingerprintは`fdc89974e89a4a6809cecd2c5608a36d684d40d76d9b3055fbbe6ec9276abbaf`。
詳細は[P-A非退化mechanism validation](docs/research_direction_joint_synthesis_mechanism_validation.md)を参照する。

## 2026-09-25 P-A v1 formalization / mechanism audit

凍結済みP-A v1について、有限候補問題、4成分の辞書式目的、DP漸化式、計算量、operator同値性条件を
形式化し、blind artifactの48 holdoutと6 operator probeを再解析した。DP遷移数は全recordで
$\sum_r n_r(n_r+1)$と一致し、保存された選択目的もsegment metadataから完全に再構成できた。

ただし、全54 recordでselected segment数はsource-basis run数と一致し、run内部を複数区間へ分けた
recordは0だった。H4/H5 holdoutの256 eventは全てTaylor order 0で、6 probeもapplication数から
非零orderを含まない。従ってblind検証が直接支持するのはrunごとのfull/support-union選択であり、
明示的なone-segment-per-source-run baselineに対するinterval分割の増分利益は未識別である。

本監査完了時点のstatusは
`pa_v1_formalized_but_interval_mechanism_not_empirically_distinguished`、
P-Aは`conditional_candidate_pending_nondegenerate_mechanism_validation`とした。既存blind gate通過と
H5/H4のRZ改善値は有効だが、interval分割の効果または非零Taylor-order移送を主張しない。次は
one-segment baseline、support変化を持つforced run、Taylor order 2を含む小さいmechanism判別を
事前登録する、とした。後続P-A検証は完了してinterval DPを停止し、その後のP-C tracking検証でも
current H4 familyが停止条件に達した。現行statusは本書先頭のP-C節を優先する。
H12、長RPE総cost、full wrapper、backend/noiseは不要である。

artifact fingerprintは`aaa5fdba8ddc6ec25fe1f286d886aba14a7a440c435f1ca5dd78ce33404b3676`。
詳細は[P-A v1 DP形式化・mechanism監査](docs/research/pa_joint_synthesis_v1_formalization.md)を参照する。


## 2026-09-25 P-A v1 blind transfer validation

事前登録した固定v1、4 policy、6 gateを変更せず、未使用H5 physical snapshotと元H4 event streamの
Qiskit optimization level 2へのpaired compiler transferを実行した。holdoutは48/48、operator probeは
6/6完了し、両stratumで全6 gateが通過した。H5では現行policy比pooled RZが-17.076%、最大個別悪化
0%、4-policy oracle regret/full RZが0%、basis列変更率95.83%、operator最大残差が
$2.998\times10^{-15}$だった。H4 opt2ではそれぞれ-6.598%、+0.265%、0.0116%、87.5%、
$3.126\times10^{-15}$だった。

blind gateだけに基づくこの時点のstatusを`advance_pa_v1_to_formal_primary_theme_candidate`とした。
これは文献上の新規性証明、全Gaussian circuitに対するglobal optimum、
coupling/noise/backend、full partial-$S_2$ wrapper、RPE総cost、H12または科学的優位性の検証ではない。
後続形式化でmechanism範囲を狭めたため、現行statusは直前節を優先する。

final artifact fingerprintは`78af3474898dbf989780ea5f2881cb5b61595609dd2698164b9846c1ce1c5919`、
file SHA-256は`ff6a8f846795b3f56e3688c62eab3ad26c3ace6632063ba72ff97f4a12ac4ea3`である。
詳細は[P-A v1 blind transfer validation](docs/research_direction_joint_synthesis_blind_validation.md)を参照する。
結果はlocal dirty-worktree evidenceであり、immutable CIまたは外部再現ではない。
## 2026-09-25 P-A scoped prior-art audit / blind preregistration


P-A v1について、DF/low-rank回路、partial basis rotation、fermionic Gaussian/Givens合成、
隣接network融合、completion自由度、DP/block synthesisを対象にscoped prior-art auditを行った。
各構成要素は既知だが、明示した検索範囲では、同一source-basis runを区間分割し、full basisまたは
support-union completionを選ぶ現行v1と同じ組合せは確認できなかった。これは網羅的な新規性証明、
特許調査または査読上の新規性判定ではない。

statusを`provisional_pending_blind_validation_after_scoped_prior_art_audit`へ更新した。次の計算は
事前登録済みの2 stratum、すなわち未使用H5 physical snapshotへの移送と、元H4 event streamの
Qiskit optimization level 2へのpaired compiler移送に限定する。両stratumは同じ4 policyと6 gateで
別々に判定し、どちらかが不通過ならP-Aを正式主題化せずP-Cへ戻る。P-A v2、H12、長RPE総costは
このblind検証へ含めない。この事前登録の固定時点では両stratumとも未実行だったが、後続のblind
transfer validationで両方が完了・通過した。

## 2026-09-25 P-B/P-C/P-A theme selection

3件のfingerprint済みpilotを4問で比較した。P-Bは現H4 gridで実用的なenergy/signal選択差がなく停止、
P-Cは未使用geometry/deltaの差分bias予測gateを通過、P-Aは強いproject baselineに対して未使用列長で
compiled-cost改善とoperator同値性を示した。

選定はP-Aを暫定主題、P-Cを副候補、P-Bを現範囲で停止とする。その後のscoped prior-art auditでは、
各構成要素は既知だが現行v1と同じ組合せを検索範囲内で確認できなかった。これは新規性の証明ではなく、
この時点のstatusを`provisional_pending_blind_validation_after_scoped_prior_art_audit`とした。その後、
事前登録済みH5 physical transferとH4 optimization-level-2 compiler transferは両方とも全gateを通過した。
H12、長RPE総cost、追加$q>32$は次の必須作業ではない。selection artifact fingerprintは
`7daa49c3c30b453d69d52830d0889ce04104cc5a0f02a2c3083517486d575fbb`。

## 2026-09-25 P-A interval-aware joint synthesis pilot

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`、$\delta=0.02$、K=2、
topology-free Qiskit 1.3.0 opt1で、full共有、event-support、現行`support_run_le_1`、interval-union DPを
比較した。training列長2/4/6とは独立なholdout列長3/5/8を各8 trajectory評価した。

DPのpooled RZは現行比-7.194%、各長で改善、最大個別悪化+0.231%、4候補内oracle regret/full RZは
0.00482%、basis列変更率87.5%、operator最大残差$3.126\times10^{-15}$だった。事前固定した6 gateは
全て通過した。full wrapper、coupling/backend/noise、RPE総cost、科学的優位性または外部文献上の新規性は
評価していない。artifact fingerprintは
`1a9840a4ee46daa6e3749272593acf3e8ae29be9fcecc1a05b0bd9ea817fbc37`。専用testは`3 passed`、全suiteは`582 passed, 2 skipped, 4 warnings`で失敗0。

## 2026-09-25 P-C geometry signed-error pilot

H4 linear chainの0.80、0.85、1.00、1.15、1.20 Åについて、STO-3G、8 qubit、DF rank 12、
`L_D=3`、二次partial-$S_2$を固定し、geometry間エネルギー差の符号付きPF biasを評価した。
0.80/1.00/1.20 Åをcoefficient training、0.85/1.15 Åをgeometry holdout、$\delta=0.4$をdelta holdoutとした。

geometry holdout coefficient誤差は最大4.409%、delta holdout誤差は最大0.840%。0.85→1.15 Å、
$\delta=0.4$の二重holdoutではactual/predicted difference biasが-0.00169849/-0.00178045 Haで、
endpoint-bias正規化誤差は2.539%だった。coefficient spanは63.303%、$\delta=0.1$隣接pairの
最小cancellation ratioは9.786%で、固定した5 gateを全て通過した。

従って案Cを次段へ残した。後続tracking・breakdown validationではcurrent H4 familyを固定条件で停止した。
これは0.80--1.20 Åの単一H4/DF/PF条件に限るlocal dirty-worktree結果で、potential-energy surface、
別系移送、RPE/RTE cost、回路compile、最終総costまたは科学的優位性の評価ではない。

初版統合artifactのexact-energy欄はsurrogate値を誤って転記していた。訂正版ではfull
DF-rank-12 Hamiltonian値へ直したが、PF bias、係数、holdout予測、5 gateは変わらない。
訂正版fingerprintは`3c199dbc20d0892cf1bcfad4b27646ea8c90d1c1d32fc5e2c75be610bb1fe611`。
専用testは`3 passed`、当時のPF/P-B/P-C関連は`10 passed`、全suiteは
`576 passed, 2 skipped, 4 warnings`で失敗0だった。
詳細は[P-C geometry signed-error pilot](docs/research_direction_geometry_energy_difference_pilot.md)。

## 2026-09-25 P-B energy-bias / target-weight pilot

研究テーマ選定の最初のpilotとして、固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12の
`pf_delta_validation_v5`を再解析した。二次partial-$S_2$、$L_D=0,...,11$、6 delta、$q=1,2,4$の
12 artifact・72候補を使用し、新しいHamiltonian/PF計算または回路compileは行っていない。

screeningの最小target weightは0.9999803239、最小q別signal半径は0.9999744904で、全候補が通過した。
各deltaのenergy-only最良とsignal-screened後のenergy最良は全て`L_D=0`で一致し、選択不一致は0/6。
ordering inversionは118組あるが、最大leakage比1.00349で、意味のあるinversionは0だった。

従って、このH4・二次partial-$S_2$・prefix gridではenergy-only基準の実質的なsignal failureを
確認できず、案Bを主題へ進めない。これは他PF family、小gap、別系に対する一般的な棄却ではなく、
weight問題を区別するstate-action診断も未検証である。後続のP-CとP-Aを含む比較では現範囲の停止を維持した。

artifact fingerprintは`e7b6e98d1a4049f5e061c4ec47de891edf40689ee9ccfc03f62c9022ea350197`、専用
testは`3 passed`、全suiteは`573 passed, 2 skipped, 4 warnings`で失敗0だった。詳細は
[P-B signal-weight pilot](docs/research_direction_signal_weight_pilot.md)。
新しい結果はlocal dirty-worktree解析であり、immutable CIまたは外部再現ではない。

## 2026-09-25 A0 fresh-proxy / legacy-holdout再照合

新規compileなしで、固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、
`L_D=3, delta=0.02, r=32, K=2`、Qiskit 1.3.0 optimization level 2の同一条件を再照合した。
最新fresh-32の`q=1,2`だけでaffine compiled-RZ proxyを固定し、旧M08/opt2の`q=16,32`
各8 trajectoryをfitへ混ぜずholdoutとして適用した。

selected `support_run_le_1`の絶対相対誤差は`q=16`で3.345%、`q=32`で4.911%となり、
既存5%基準を両方通過した。旧holdoutのdirect RZ relative SE最大は、full basisの`q=16`に
おける1.3865%で2%基準内だった。ただしselected `q=32`の標準化残差は2.012で1.96を僅かに
超える。非gatingのfull-basis診断は`q=32`で5.598%となり5%を超えた。

従って、最新fresh較正の適用domainを`q=32`まで接続できるのは、この単一cellのselected-policy
compiled RZに限る。全metric、他の`delta/r`、`q>32`へは拡張しない。schedule・cost再最適化、
最終総cost、科学的優位性の評価も行っておらず、外部instance pilotを次のdiscriminator候補とする
既存方針は変更しない。artifact fingerprintは
`b786b1d48995d320c2301718cc958018462cf1cbb05fcfa19ca6cc9460dd2295`、専用testは`3 passed`。
M06-F関連全体は`13 passed, 2 skipped`、全suiteは`570 passed, 2 skipped, 4 warnings`で
失敗0、manifest検査もpassした。
詳細は[M06-F all-r coherent opt2](docs/research_direction_full_opt2.md)を参照する。

## 2026-09-25 M06-F fresh-32 and coherent opt2 result

事前登録済みfresh-32拡張は固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、
`L_D=3`、`delta=0.01,0.02`、`r=1,2,4,8,16,32`、`q=1,2,8`のうち初期精度gateで
指定された5 groupだけをQiskit 1.3.0、optimization level 2、basis `rz,sx,x,cx`、seed 17で
実行した。15/15 task、1,920 direct transpile、failed 0、wrapper exit code 0、13,833秒で完了した。
初期36 taskと統合して51/51/0 expected/completed/failedで、missing、partial、duplicate、seed重複、
破損JSON、aggregate不一致は0。CPU-only、各thread=1、compute source hash不変も確認した。

12個のrandomized group全てが事前gateを通過した。direct RZ relative SE最大1.9844%、
selected-policy RZ holdout最大4.4898%、selected全metric holdout最大4.7488%である。
`q=1,2`は較正、`q=8`は固定holdoutとして分離した。同一opt2 contextでbeta、alpha、integer
shot、scheduleを再最適化すると、両候補とも`delta=0.02`を選び、状態準備なしのcompiled-RZ
点推定は`L_D=3/12`で`1.263314e12/1.327822e12`、shotは13,588/11,162だった。
点推定は`L_D=3`が4.858%低いが、local 5%、per-r実測幅、25%移送の全区間は重なる。

旧mixed compiler-context focused推定に対し、coherentな`L_D=3`点推定は3.8918%増え、
点利得は8.422%から4.858%へ縮小した。共通状態準備costの点break-evenは26,590,335
compiled-RZ相当/shotだが、P=0ですでに区間が重なるため頑健な非負P範囲はない。
all-rのcompiler-context欠落は現H4測定範囲で解消した一方、q>32、状態準備、外部instance、
coupling/backend、H12、immutable再現は未解決であり、最終総costまたは科学的優位性は主張しない。

監査artifact fingerprintは
`b39960a630746e2c05009f8d7e13bd982ff565b3a3c70a7abfc2dc65dc7009ca`、
coherent解析fingerprintは
`5ce368a94daa39680b4edc0cfb59b30168d8bc159ad2538b29cb67b928e3cdba`。
T4/T7主軸、T1範囲変更、T2/T5/T6限定、T3保留は維持し、局所compiler精密化を止めて
外部instance pilotを次のdiscriminator候補として再開可能にする。
commit済みartifactを使うM06-F専用testは`10 passed`、Git管理外raw checkpointを再集計する
optional integration testは`2 skipped`。manifest検査はpass。clean checkout相当の全suiteは
`567 passed, 2 skipped, 4 warnings`で、失敗は0だった。raw再集計はserver evidenceが利用できる
環境でのみ実行し、clean checkoutの通常testを失敗させない。
詳細は[M06-F all-r coherent opt2](docs/research_direction_full_opt2.md)。

## 2026-09-24 M06-F all-r coherent opt2 initial result

終了後の完全性監査で、初期36 taskのtask spec、worker result、checkpoint、aggregateを再照合した。
failed、missing、partial、duplicate、破損JSON、fingerprint不一致はいずれも0で、tmuxと関連processも
終了していた。runner契約から終了code 0と判定したが、元shellの`$?`自体は独立保存されていない。
初期batchは36/36完了、所要17,475.324秒、事前登録workflow全体はfresh-32 15 task未実行のため
36/51である。監査artifact fingerprintは
`7d184c4ebde0665fbc85452c69b9de14997f101fb51f9e2969fd13bf1d5ebf35`。
この監査は下記の研究判断を変更せず、coherent再最適化を完了扱いにしない。
監査専用testは`3 passed`、関連testは`33 passed`。全suiteは`562 passed, 2 failed, 5 warnings`で、
失敗2件は今回の変更外にある保存referenceのPython版不一致と既存four-round artifactの
DF preparation hash不一致である。この監査では無関係な保存artifactを変更しない。


WP11が選択した`all_r_coherent_opt2_reoptimization`について、固定H4 linear chain、1.0 Å、
STO-3G、8 qubit、DF rank 12、`L_D=3,12`、`delta=0.01,0.02`、Qiskit 1.3.0、
optimization level 2、basis `rz,sx,x,cx`、seed 17の同一compiler contextで初期計算を実行した。
36 cell task、1,062 direct transpileは全て完了し、失敗・中断は0だった。

12個のrandomized `(delta,r)` groupのうち7個は全基準を通過した。selected-policy RZおよび
全compiled metricのholdout誤差は全groupで5%以内だった。一方、direct RZ relative SEの2%基準は
`(0.01,16)`, `(0.01,32)`, `(0.02,8)`, `(0.02,16)`, `(0.02,32)`の5 groupで不通過となった。
最大値は順に3.911%、2.416%、2.735%、2.543%、2.141%である。これは初期8 trajectoryの
精度不足として扱い、事前規則どおり各groupの`q=1,2,8`をfresh 32 trajectoryで再計算する
15 task、1,920 direct transpileのextension manifestを生成した。追加計算は未実行である。

初期aggregate fingerprintは`ae0e0d9b616da5c31cbdd09d27d2b6e103cc07de05f0b400ab04f51da8f0c63a`、
解析fingerprintは`b256b47a83fb54657716d4d7f910a8772aa0f583c54ec9a63259f37784d0bcaf`、
extension manifest fingerprintは`c150a17c92ade7bd5257a8b99a92bdfdae688fb467000500a9e92754ac23e997`である。
statusは`requires_fresh_32_trajectory_extension`であり、coherent再最適化、`L_D=3/12`比較、
状態準備、q>32、backend/noise、最終総cost、科学的優位性は未評価である。専用testは`4 passed`、
関連testは`32 passed`。これはlocal dirty-worktree evidenceであり、immutable CIではない。
詳細は[M06-F all-r coherent opt2](docs/research_direction_full_opt2.md)。

## 2026-09-23 WP11 scoped direction synthesis note

Gate S1、WP06-a/b、WP05-a/b/R、WP01-D/C07、G08、M08、M06/L08、N07/P03の11個の
fingerprint済みartifactを新規compileなしで統合した。固定範囲はH4 linear chain、1.0 Å、
STO-3G、8 qubit、DF rank 12、`CA/10`、`L_D=3,12`、compiled RZである。

T4（長いランダム回路cost予測）とT7（信頼できる資源評価・否定的結果）を主軸として継続し、
T1は一般的優位性からH4条件付き成立限界へ範囲変更、T2/T5/T6は限定継続、T3は保留とした。
状態準備なしの点推定候補は`L_D=3`だが、頑健判定は
`undetermined_under_compiler_transfer_and_preparation_sensitivity`のままである。

次の一件は`all_r_coherent_opt2_reoptimization`とした。既存opt2証拠は`L_D=3,r=32`だけなので、
未測定`r=1,2,4,8,16`をoptimization level 2で較正・holdoutし、同一compiler contextで
beta、alpha、shot、scheduleを再最適化する。外部instance pilotは棄却せず、この比較を完了または
停止した後へ延期する。選択次段はCPU transpile中心でGPUを必要としない。

WP11自体は新規物理計算、full opt2実行、q>32直接検証、状態準備計測、backend/noise、最終総cost、
科学的優位性を含まない。artifact fingerprintは
`45def7f696eddba574878cc7530837dfdfc5c6e9c2767ee3e115cf4a5f1ac092`。専用testは`3 passed`、
変更後のlocal全suiteは`557 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
詳細は[WP11限定判断統合](docs/research_direction_wp11_synthesis.md)。

## 2026-09-23 N07/P03 uncertainty ledger and preparation break-even note

WP01-D/C07、M08再集計、M06/L08 compiler-transferのfingerprint済みartifactを使い、新しい
回路compileなしで不確かさ台帳と状態準備costのbreak-evenを再集計した。固定H4 linear chain、
1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3,12$、$\delta=0.02$、compiled RZ比較である。
shot数は13,538と11,162で、$L_D=3$が2,376多い。

共通の1 shot当たり状態準備costをPとすると、点推定break-evenはopt1で99,045,126、opt2
focusedで47,067,344 compiled-RZ相当/shotだった。opt1 local 5%区間で$L_D=3$が確実に低い
P範囲は0--640,843に限られる。opt2 focused selected実測幅はP=0ですでに区間が重なるため、
compiler-robustに$L_D=3$区間が低い非負P範囲はない。候補別準備では共通Pを使わず、
`13538*P3-11162*P12`の二次元境界を使用する。

sampling、model discrepancy、compiler、q>32移送、opt2の$r<32$、状態準備、外部snapshot/backend/noiseを
別classとして記録した。頑健判定は
`undetermined_under_compiler_transfer_and_preparation_sensitivity`である。後続WP11限定判断統合へ入力済みである。
状態準備costは測定しておらず、q>32、full opt2、backend/noise、最終総cost、科学的優位性を含まない。
artifact fingerprintは`ff70308a64798c6ba8c9d20533c9e9e8c614e58c0d433dd861b7de45ac70c32d`。
専用testは`2 passed`。変更後のlocal全suiteは`540 passed, 4 warnings`で、warningは既存
grouped-UWC test由来である。詳細は
[N07/P03不確かさ・break-even](docs/research_direction_uncertainty_break_even.md)。

## 2026-09-23 M06/L08 compiler-transfer analysis and reaggregation note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、$\delta=0.02$で、
同一trajectoryをQiskit optimization level 1から2へ変更して再compileした。$L_D=3,r=32$の
$q=1,2,16,32$とtail-free $L_D=12$を評価し、その他のcompiler条件は固定した。
optimization level 2の$q=1,2$ affine proxyは$q=16,32$でselected RZ最大2.340%、selected
全metric最大2.470%、full-basis RZ最大3.420%、direct RZ relative SE最大1.387%となり、
5%点誤差・2%精度基準を通過した。

WP01-D/C07のschedule、shot、$\alpha$、$\beta$を固定し、$L_D=3$は直接証拠のある$r=32$の
最後3 roundだけ、$L_D=12$は決定論provider全体をoptimization level 2へ置換した。点推定は
$1.2159903\times10^{12}$と$1.3278223\times10^{12}$で$L_D=3$が8.42%低いが、対称discrepancyの
分離上限1.881%に対しselected実測値は2.340%で、区間は重なった。一様比率を未測定$r<32$へ
移す反実仮想もdiscrepancyの選択で分離判定が変わる。従ってcompiler-invariantなlocal分離は
未確立で、頑健判定は`undetermined_under_compiler_and_transfer_sensitivity`とする。

raw artifact fingerprintは`87f66c8944dedfaa9fe5f0edf864d2a2a07cb82e23245af4f5944f90bda83ad6`、
解析・再集計artifact fingerprintは`7ebe8815a633a182b4afa4608b969df5c53d1b6d1efe6eeaf2ce4e653fcbbd95`。
専用testは`3 passed`、変更後のlocal全suiteは`538 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。これはlocal dirty-worktree evidenceであり、$L_D=3,r<32$のopt2直接検証、
$q>32$、full opt2再最適化、最終総cost、科学的優位性を含まない。詳細は
[M06/L08 compiler-transfer解析](docs/research_direction_compiler_transfer.md)。

## 2026-09-22 M08 late-round holdout and WP01-D/C07 reaggregation note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3$、$\delta=0.02$、
$r=32,K=2$について、G08で固定した未使用$q=16,32$を各8 fresh trajectory、cosine/sine両軸、
selected `support_run_le_1`／full basisのcomplete controlled Hadamard wrapperで直接transpileした。
selected policyのRZ誤差は最大2.466%、全metric最大2.569%、full-basis RZ誤差最大3.286%、
direct RZ relative SE最大1.353%で、5%基準、5.0484%分離限界、2%精度診断を全て通過した。
事前規則による$q=64$ follow-upは発火しなかった。

WP01-D/C07の点推定・schedule・shot数・較正half-widthを保持し、M08 selected RZ 2.466%と
観測RZ最大3.286%を共通の対称model-discrepancy scenarioとして再集計した。両scenarioとも
$L_D=3$と$L_D=12$の区間は分離し、従来5%区間も僅かに分離したままだった。一方、25%移送区間は
重なったため、頑健な方向判断は`undetermined_under_transfer_sensitivity`を維持する。

M08測定は$q\leq32$だけの直接証拠で、実scheduleの$q_{\max}=131072$、$L_D=12$、別snapshot、
別compilerへの直接証拠ではない。再集計は同じ許容幅を両候補へ置く反実仮想であり、点推定の
再最適化でも最終総cost評価でもない。M08 artifact fingerprintは
`e010a63bfa5aecd7de01ba074f56300615460de9e6f51a40dca352954992aaf8`、再集計artifactは
`6aa69bc756a97aa025e26994f596f9ee3df999be2e64be870ce938dca5180b3e`。詳細は
[G08/M08後半round proxy精度](docs/research_direction_late_round_proxy.md)。変更後のlocal全suiteは
`535 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 G08 round-dominance note

WP01-D/C07のfingerprint済みschema-v2 computeと判断統合artifactを、新しい回路compileなしで
round別に再集計した。最後3 roundは$L_D=3$のcompiled RZ cost 90.94%、較正不確かさ87.70%、
$L_D=12$のcost 83.03%を占めた。両候補の最大cost/PF-riskはround 17だが、$L_D=3$の最大
finite-RTE riskはround 7であり、最大cost roundとは一致しない。

この結果からM08を$\delta=0.02,r=32,K=2,q=16,32$の各8 fresh trajectoryへ限定した。
判定はselected RZ・全metricとfull-basis RZの5%、local区間分離限界5.0484%、direct RZ
relative SE 2%である。G08 artifact fingerprintは
`e696ced27b06e871368f3afa164f507c240d4aa6693223f9f6abe7990b30d064`。これはlocal
dirty-worktreeの検証資源配分判断で、$q>8$精度または最終総costそのものの検証ではない。詳細は
[G08/M08後半round proxy精度](docs/research_direction_late_round_proxy.md)。

## 2026-09-22 WP01-D/C07 candidate-specific optimization and interval synthesis note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、CA/10について、$L_D=3$の
`support_run_le_1`とtail-free $L_D=12$、$\delta=0.01,0.02$を比較した。WP05-bRまでの
complete controlled partial-$S_2$／Hadamard wrapperの$q=1,2$ affine proxyと$q=4,8$ holdoutを
使い、compiled RZ、$\beta_{\mathrm{RPE}}=0.40$ rad、$\alpha_{\mathrm{tot}}=0.05$、
cost-weighted $\alpha$の下でPF/RTE/statistical budgetと整数shot数を候補ごとに再最適化した。
coarse/fine gridの後、上位20解のPF/RTE制約境界を反復的に締めている。

両候補とも$\delta=0.02$を選んだ。$L_D=3$は13,538 shot、点推定
$1.4557921\times10^{12}$、$L_D=12$は11,162 shot、$1.6911234\times10^{12}$で、前者が
13.916%低い。5% local model区間はそれぞれ$[1.30654,1.60504]\times10^{12}$と
$[1.60657,1.77568]\times10^{12}$で僅かに分離した。ただし分離幅は$L_D=12$点推定の約0.090%、
対称model discrepancy 5.0484%が分離限界で、採用した5%との差は0.0484 percentage pointに過ぎない。
25%移送区間$[1.01538,1.89620]\times10^{12}$と$[1.26834,2.11390]\times10^{12}$は重なる。

したがってlocal model条件付きの候補は$L_D=3$だが、頑健な方向判断は
`undetermined_under_transfer_sensitivity`であり、部分ランダム化の科学的優位性または最終総costを
示さない。次はM08/G08として、総costの83--91%を占める後半3 round付近の$q>8$ proxyを直接
較正・holdoutする。状態準備、実backend、noise、別snapshot/系サイズ、immutable CI、外部再現は
含まない。詳細は[WP01-D/C07再最適化](docs/research_direction_decision_cost.md)。
compute artifact fingerprintは
`709142f76a78232804cae9971a24bc5757b3ef2e5258adb7314fc42e91937474`、判断統合artifactは
`7d85b472851a6c4046b847a7e9898a718e96b2bad7a50fd491d939b34244250f`である。schema-v1 computeは
grid下端依存を検出した予備診断で、現行判断には使わない。変更後のlocal全suiteは
`530 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 WP05-b/R full-scope extension and focused replication note

WP05-aと同じ固定H4 snapshot・compilerで、$L_D=3$の`support_run_le_1`とfull basisを
$q=8$および比較対照$\delta=0.01$へ拡張した。$\delta=0.01$は選択policyの$q=8$ RZ誤差
最大2.650%、全metric最大2.804%、full basis RZ誤差最大4.321%、全metric最大4.500%で
事前5%基準を通過した。初回$\delta=0.02,r=32,q=8$は選択policy RZ 5.084%、
全metric 5.392%、full basis RZ 8.995%で不通過だった。

この一点を新しいseedの独立32 trajectoryで再検証したWP05-bRでは、選択policyの$q=8$ RZ誤差
0.516%、全metric最大0.537%、full basis RZ誤差0.829%、選択policy$q=4$全metric最大1.094%で、
全checkが5%基準を通過した。初回超過は高統計再検証で再現しなかった。これは1 snapshot・
1 compiler・最大$q=8$のlocal dirty-worktree evidenceであり、$q>8$や別条件への精度移送、
最終総costを保証しない。詳細は
[WP05-b/R拡張・再検証](docs/research_direction_full_scope_extension.md)。初回artifact fingerprintは
`363ac90ace643556ff068ff7b22ed8e43c51c6dbd0085a2256dac29a0303bfda`、再検証artifactは
`7bfeddaccfe10f28b67bdf857ebd76cd43b0eb74b5af5d209a435ee9d8abb472`である。

## 2026-09-22 WP05-a full controlled-interrogation connection note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3,12$、$\delta=0.02$、
Qiskit 1.3.0の`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしで実施した。
WP06-bの`support_run_le_1`を明示basis planとしてcomplete controlled partial-$S_2$、反復回路、
cosine/sine Hadamard wrapper、ancilla Z測定まで伝播した。production既定値はfull basisのままである。

$L_D=3$では$r=1,2,4,8,16,32$、$q=1,2,4$、各8 trajectoryについてfull/policyを対応付け、
576本のmeasurement-bearing wrapperを直接transpileした。$q=1,2$から固定したaffine式は、独立seedの
$q=4$で選択policyのRZを最大2.288%、全6 metricを最大2.431%で予測した。full basisのRZ最大誤差は
4.234%、tail-free $L_D=12$は0%だった。WP06-b中央RTE additive bridgeと今回の直接wrapper差の
RZ残差はfull-wrapper平均比で最大2.625%となり、事前5%基準を通過した。H4の固定ランダム状態作用
比較はcontrolled evolutionと両wrapperで最大$1.08\times10^{-16}$、relative ancilla phaseも一致した。
専用小系testでは完全operatorを比較している。

WP04のround、shot、$\alpha$を固定した非decision-grade長$q$感度では、選択policyの$L_D=3$が
$1.5933\times10^{12}$、tail-free $L_D=12$が$1.6963\times10^{12}$で、点推定は$L_D=3$が6.07%
低い。ただしlocal区間$[1.3489,1.8377]\times10^{12}$と
$[1.6115,1.7811]\times10^{12}$は重なる。$q>4$を直接transpileせず、$\alpha$・shot・scheduleも
再最適化していないため、最終総costまたは科学的優位性ではない。

後続のWP05-b/Rで$q=8$と$\delta=0.01$の5%基準を通過し、WP01-D/C07の再最適化まで完了した。
状態準備、backend実行、noise、量子shot、immutable CI、外部再現は含まない。詳細は
[WP05-a full-scope接続](docs/research_direction_full_scope.md)。専用testは`4 passed`、成果物fingerprintは
`d8196ef1d8a576b7b7ab443c8613d2bd70b7a4fa57f43ac495be62bf2748f512`である。変更後のlocal全suiteは
`524 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 WP06-b sequence-policy and proxy-bridge note

WP06-aと同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot、$L_D=3$、
$\delta=0.02,K=2$、Qiskit 1.3.0 compilerでsequence-aware basis policyを比較した。元DF basisの
run長1,2,3以下または全runをsupport限定へ置換する候補を列長1,2,4の独立trainingで比較し、各trajectory
のRZ悪化5% guardを通る`support_run_le_1`を固定した。production builderの既定値は変更せず、明示
basis planとして渡す。

未使用列長3,6の各12本では、full basis共有比でRZ -10.67%、CX -8.13%、total depth -2.25%、
circuit size -10.63%だった。trajectory別oracleに対するpooled RZ regretはfull RZ基準0.390%。sampled
列長1,3と強制$K=2$非零phase eventのcontrolled operator残差は最大$1.34\times10^{-15}$で、強制
eventのrelative ancilla phaseも一致した。

$r=1,2,4,8,16,32$の独立各8本から中央RTE差を既存Hadamard proxyの$q$ slopeへ加えると、RZ slopeは
最大8.00%変化した。WP04のround、shot、$\alpha$、wrapper interceptを固定したbridgeでは$L_D=3$の
点推定が$1.7848\times10^{12}$から$1.6503\times10^{12}$へ下がり、$L_D=3/12$の点順位が反転した。
ただし両local区間は重なるため未決定を維持する。

後続WP05-aで選択policyをcomplete controlled partial-$S_2$／Hadamard wrapperへ直接接続し、
未使用$q=4$とadditive bridgeの5%基準を通過した。さらにWP05-b/Rの$q=8$・$\delta=0.01$、
WP01-D/C07の候補別再最適化まで完了した。
WP06-b自体は中央RTEだけのadditive bridgeであり、full wrapper、$\alpha$・shot再最適化、
状態準備、実backend、noise、最終総cost、immutable CIまたは外部再現ではない。詳細は
[WP06-b sequence policy](docs/research_direction_sequence_policy.md)。専用testは`5 passed`だった。
変更後のlocal全suiteは`520 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 WP06-a circuit-structure pilot note

Gate S1と同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot、$L_D=3$、
$\delta=0.02$で、係数最大のZ/ZZ event、同一fragmentの異なるsupportからなる長さ1--3列、
full Gaussian basis、support限定completion、basis融合、control、scalar/relative phaseを比較した。
compilerはQiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしである。

support限定completionは必要なunitary列を保持し、単一controlled ZのRZを204から78へ61.76%、
ZZを311から189へ39.23%減らした。全比較の最大operator残差は$4.73\times10^{-15}$で、
$10^{-10}$基準を通過した。Gate S1で事前固定した5% triggerは発火した。一方、異なるZZ supportの
列では長さ2がRZ 19.58%減・depth 45.26%増、長さ3がRZ 15.01%増・depth 86.64%増となり、
full basis共有とsupport限定の優劣が列長で反転した。support限定への一律置換は採用しない。

whole-event controlと現行diagonal-only controlはrelative phaseを含め一致し、現行方針はRZを
2,567から311へ87.88%、CXを1,834から98へ94.66%減らした。controlled scalar補償を省くと
operator差0.0477098が生じ、現行補償を入れると$4.73\times10^{-15}$以内で一致した。controlと
phase方針は維持する。

次はfocused WP06-bとしてsequence-awareなfull/support basis policyをproduction builderへ統合し、
物理event分布と未使用短列holdoutでproxyを再較正する。その後にWP05へ進む。これはlocal
dirty-worktreeの構造pilotであり、$L_D$候補順位、RPE $q$傾き、full controlled interrogation、
最終総cost、immutable CIまたは外部再現ではない。詳細は
[WP06-a回路構造pilot](docs/research_direction_structure_pilot.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`515 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 Gate-S1 research-direction synthesis note

WP00、WP02、WP01-S、WP04、WP03のfingerprint済み成果物を統合した。新しい物理計算または
回路compileは行っていない。WP04の$L_D=12$点推定は$L_D=3$より4.958%低いが、local 5%と
transfer 25%の両scenarioが重なるため、`undetermined_not_tied`を維持する。主な共通利得は
$\beta$、次いで$\alpha$再配分で、PF係数policyは全て$L_D=12,\delta=0.02$を選択した。
partial randomization固有の優位性は示されていない。

最大の残存不確かさをfull controlled interrogationの回路scope・構造と判定し、T4/T7を主軸、
T1/T2/T5/T6を限定継続、T3を保留とした。次はWP06-aだけを行い、その後WP05へ進む。
WP06-a専用のresearch-routing triggerはRZ相対変化5%、候補順位反転、$q$傾き・適用domainの変更、
またはcontrolled relative phase補償の欠落である。5%は現点推定差4.958%に合わせたtask固有値で、
普遍的な回路精度保証ではない。

これはlocal dirty-worktreeの研究方向判断で、科学的優位性のdecision-grade評価、最終総cost、
immutable CIまたは外部再現ではない。詳細は
[Gate S1判断](docs/research_direction_gate_s1.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`511 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 research-direction WP03 PF-coefficient sensitivity note

WP00/WP04と同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot
`56e4df83...31e5`、CA/10 taskで、costed候補$L_D=3,12$、
$\delta=0.01,0.0125,0.02$について、$C_D$、論文D6、支配固有位相係数だけを差し替えた。
compiled-cost provider、round別compiled-RZ schedule、cost感度重み$\alpha$、WP04の候補別
$\beta$配分は固定した。$L_D=0$はWP01-Sでcompiled-cost評価前にscreen out済みなので、
係数反例の監査だけに含めた。

$C_D$もD6・支配固有位相と同じconditioned窓$\delta=0.05,0.1,0.2,0.4$で再fitした。
$L_D=3$の係数は$C_D=0.0117236$、D6 0.0133991、支配固有位相0.0133569で、$C_D$はD6比
12.50%低かった。$L_D=12$では0.0134411、0.0134257、0.0133833だった。costed候補のD6と
支配固有位相係数差は最大0.317%である。$L_D=0$では$C_D=0$でもD6 0.0115338、支配固有位相
0.0114931が非零なので、$C_D$は引き続き広いscreening専用とし、shortlist後は候補ごとのD6を使う。

3係数×2候補×3$\delta$の18条件は全てPF予算内で、全係数が$L_D=12,\delta=0.02$を
RZ点推定最良とした。D6で$L_D=3,\delta=0.02$を選ぶ点regretは5.216%だが、相対差の
local 5%＋較正scenarioは$[-15.91\%,28.57\%]$、25%移送scenarioは
$[-46.20\%,90.91\%]$で0を跨ぐ。従って係数選択はWP04の`undetermined`判定を解消しない。
$\delta=0.02$がPF予算外になるまでのD6係数増加余裕は$L_D=3$で6.76%、$L_D=12$で6.55%で、
現行D6対支配固有位相差より大きいが、別instanceへ無条件に移送できる余裕ではない。

保存されたD6のsigned biasは支配固有位相と逆符号で、定義の符号規約が直接揃っていない。
また個々のmixed/tail交換子項は分解していないため、符号差を物理的相殺と解釈せず、$C_D$から
full partial係数への差も省略効果の合計として扱う。

これは$q=1,2$直接較正から長$q$へaffine外挿したlocal dirty-worktree screeningである。
係数familyは統計的信頼分布でなく、状態準備、実backend、noise、full-scope holdout、最終総cost、
優位性、immutable CIまたは外部再現を含まない。Gate S1に必要なWP01-S、WP02、WP04、WP03が
揃ったため、次は研究方向判断を統合する。詳細は
[WP03係数感度](docs/research_direction_pf_sensitivity.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`507 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-21 research-direction WP04 ablation note

WP00/WP01-Sと同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot
`56e4df83...31e5`、CA/10、$\delta=0.02$、$M=17$、$q_{\max}=131072$で、$L_D=3,12$の
round schedule、$\beta$、$\alpha$、schedule-selection cost providerを分離した。状態準備なし
Hadamard scopeの$q=1,2$直接較正を再利用し、$q>2$は軸別affine外挿とした。

3 schedule policy、2 alpha policy、候補別beta profileから42 factorial cellを評価し、固定順の
逐次差分、完全設定からのleave-one-out、beta--alpha interactionを保存した。完全設定から
beta再配分を戻すとRZ点推定は$L_D=3,12$で143.2%、149.6%、alpha再配分を戻すと35.4%、
26.3%増えた。$L_D=3$のcompiled-RZに整合したround scheduleは同じ完全設定の固定schedule比
1.93%減に留まる一方、成分作用数を目的に選ぶscheduleはRZを16.87%増やした。したがって、
WP04での主要な共通利得はbeta、次いでalpha再配分であり、cost provider変更をfinite-RTE固有の
schedule利得と同一視しない。

完全設定のRZ点推定は$L_D=3$で$1.7848\times10^{12}$、$L_D=12$で
$1.6963\times10^{12}$となり、決定論endpointは4.96%低い。ただしlocal 5%＋較正区間と
25%移送＋較正区間はともに重なるため、方向判定は引き続き`undetermined`である。
$L_D=3$の選択schedule全18点を含むsector行列gridは既存validatorを通過し、最小観測半径
0.5727013937は最小保守下界0.5727013920以上だった。$L_D=12$のtailなし信号の最小半径は
0.9999999998だった。最後の3 roundは両候補のRZの91.9%、83.0%を占める。

これはlocal dirty-worktreeの`model_conditional_screening`である。長$q$ costは未使用holdoutの
ない$q=1,2$ affine外挿、beta gridは連続最適化でなく、厳密二項値は既知の小系信号に対する
counterfactualである。状態準備、実backend、noise、full-scope holdout、最終総cost、部分
ランダム化の優位性、immutable CIまたは外部再現を主張しない。次はWP03でPF係数だけを
差し替え、候補順位とregretの変化を評価する。詳細は
[WP04寄与分解](docs/research_direction_ablation.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`503 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-21 research-direction WP00/WP02/WP01-S note

H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12の固定Hamiltonian snapshot
`56e4df83...31e5`について、研究方向screeningの最初の3段階を実行した。WP00では
$L_D=0,3,12$のPF入力を同じsnapshotから再生成し、CA/10、$\beta=(0.02,0.02,0.36)$、
$\alpha_{\rm total}=0.05$一様配分、状態準備なしHadamard scope、同一compilerの比較契約を固定した。
既存PF artifactはcost snapshotとHamiltonian hashが一致しなかったため、この比較には使用しない。

WP02ではCA、CA/10、CA/100と$\delta=0.01,0.0125,0.02$の9条件を監査した。
CA/10の$q_{\max}$は131,072--262,144で、既存3 schedule・56点sector行列検査を再利用した。
CA/100の$q_{\max}$は2,097,152--4,194,304で、3条件すべてが経験的
$qC\delta^3\leq0.02$を満たさない。従ってCA/100は長回路costを先に外挿せず、
小さい$\delta$と新scheduleを作る必要がある。

WP01-Sでは$L_D=0$だけ$r=1,\ldots,65536$、$K=0,2,\ldots,16$へ探索域を拡張した。
成分作用数proxyの最良条件は$\delta=0.02$、半径下界0.5473、shot合計25,400、
proxy値$4.4839\times10^{12}$で、探索境界には当たらなかった。同じ目的関数の$L_D=3$最良値の
560.6倍だったため、compiled RZ cost評価前にscreen outした。$L_D=3$の7種類の$(r,K)$は
$\delta=0.02,q=1,2$のfull Hadamard wrapperを各8 classical trajectoryで直接compileし、
$L_D=12$はtailなしの$q=1,2$を厳密評価した。CA/10の全roundへaffine外挿すると、
両候補とも$\delta=0.02$が最小で、no-prep総RZ点推定は$L_D=3$が
$3.0814\times10^{12}$、$L_D=12$が$2.4329\times10^{12}$だった。決定論endpointは
点推定で21.0%低く、5% scenarioでは区間が分離するが、25%移送scenarioでは重なる。

従って現状は、$L_D=0$をこの解析的screen内で強く不利とし、$L_D=3$対12は
`undetermined`とする。$L_D=0$対3の560.6倍はcomponent-application proxyの比較であり、
compiled RZ cost比または一般的な理論上の棄却ではない。
PF係数は経験値、schedule別$q=1,2$ fitには未使用holdoutがなく、$q>2$は直接compileしていない。
scenario幅も統計的信頼区間ではない。この結果を最終総cost、部分ランダム化の優位性、
immutable CIまたは外部再現とは扱わない。次はWP04、WP03で寄与と係数感度を分離し、
WP05後のWP01-Dでdecision-gradeに再評価する。詳細は
[研究方向screening検証](docs/research_direction_prevalidation.md)。関連testの部分実行は`8 passed`で、
変更後のlocal全suiteは`499 passed, 4 warnings`だった。warningは既存grouped-UWC test由来である。

## 2026-09-20 delta-schedule central-RTE compiled-cost note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしで、
前項の$\delta=0.01,0.0125,0.02$ scheduleへ境界補正型compiled-cost modelを接続した。

scheduleで現れる短時間幅0.02--0.000390625の12点について、同じ1--4イベント列を
角度だけ変えて個別にコンパイルした。1--3イベントは280 trajectory・3,080 metric比較、
4イベントは160 trajectory・1,760比較で、RZ/CX数・深さ、全体深さ、回路サイズの差は
すべて0だった。したがって、この固定compiler範囲では短時間幅0.02の係数を再利用した。

固定K1--K3を未使用イベント列へ適用すると、$L=8$は全6指標最大2.319%、$L=16$は
3.556%以下で点基準を通過したが、$L=32$は全次数0 RZ誤差7.966%、$z=3.151$で不通過だった。
各4イベントTaylorパターン100標本でK4を較正すると、$L=32$の最大点誤差は4.175%へ下がった。
ただしK4の点ごとの95%診断は5%を超え、厳密な5%保証ではない。次数2が2か所以上のK4窓は
今回のround分布では最大期待数$7.64\times10^{-13}$以下だった。

$r\leq16$にK1--K3、$r=32$にK1--K4を使い、各roundの解析Taylor確率、$q_m$、暫定shot数で
中央RTEブロックを集計した。RZ代理値は$\delta=0.02$で$7.9877\times10^{11}$、0.01で
$9.2021\times10^{11}$、0.0125で$1.2626\times10^{12}$となり、全6指標で0.02が最小だった。
0.01はRZで15.203%増、0.0125は58.073%増である。

これは中央$\widetilde U_{\rm RTE}$ブロックだけのlocal dirty-worktree proxyである。
決定論DF half sweep、決定論/RTE外側境界、制御化、Hadamard wrapper、状態準備、$q>8$一体回路、
最終総cost、immutable CIは未評価。従って$\delta=0.02$は次の優先候補であって最終採用ではない。
次は0.02と比較対照0.01の制御付きpartial-$S_2$反復proxyを検証する。詳細は
[delta schedule中央RTE cost検証](docs/rpe_delta_compiled_cost_validation.md)。
変更後のlocal全テストは`494 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 delta and round-specific finite-RTE schedule note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$で、
既存PF検証で実行済みの10個の$\delta$候補を、暫定`CA/10`、
$\beta_{\rm RPE}=0.4$、経験的$C=0.01342567$でscreeningした。PF位相予算0.02 radを
通過したのは$\delta=0.01,0.0125,0.02$の3候補だった。$\delta=0.0125$は較正と
独立なPF検証gridにおける唯一の通過点である。

各候補の18--19 roundで$r_m\in\{1,2,4,8,16,32,64,128\}$、
$K_m\in\{0,2,4,6,8\}$を走査し、shot数で重み付けしたランダム成分作用数を
暫定proxyとしてround別scheduleを構成した。選択した全56 round点のH4 sector行列検査で、
演算子・信号・適用可能な位相上界、PF/RTE予算、半径下界がすべて通過した。
最小観測半径は0.572701、最大実PF位相誤差は0.0139884 rad、最大有限RTE位相上界は
0.0176405 radだった。

暫定proxyでは$\delta=0.02$が最小だが、$\delta=0.01$との差は約0.87%である。
このproxyはcompiled costではないため、両者をshortlistとし、回路cost modelの
再較正またはholdout後に比較する。$q>8$一体compile、fresh-IID実験、最終総cost、
immutable CIは未評価。詳細は
[delta/round schedule検証](docs/rpe_delta_round_schedule_validation.md)。
変更後のlocal全テストは`490 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 target-precision round-horizon note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1$、$\beta_{\rm RPE}=0.4$について、
$\beta_{\rm RPE}/(2^M\delta)\leq\epsilon_E$を満たす最小round範囲を計算した。
正本文書では$\epsilon_E$は外部入力なので、既存設定`TARGET_ERROR=CA/10`を暫定主条件、
化学精度$CA$を感度比較とした。

補足資料の架空例$\epsilon_E=0.50$は$M=3,q_{\max}=8$、化学精度は
$M=12,q_{\max}=4096$、暫定`CA/10`は$M=15,q_{\max}=32768$となった。
従って、既存4段検証は実目標候補のround範囲を覆わない。

固定$r=4,K=2$をsector行列上で$q=8,4096,32768$へ延ばすと、$q=32768$の
PF位相誤差は0.437285 radで0.02 rad予算を超え、attenuationは
$7.90584\times10^{-13}$だった。finite-RTE演算子・信号上界は通過したが、この固定設定を
長roundへ単純外挿する候補は棄却する。次は$\delta$候補とround別$(r_m,K_m)$を再探索する。

これはlocal dirty-worktreeの小規模行列診断である。$q>8$回路コンパイル、cost proxy、
fresh-IID shot、最終総cost、immutable CIは未評価。詳細は
[round範囲診断](docs/rpe_target_round_horizon_validation.md)。
変更後のlocal全テストは`487 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 physical q=8 and four-round branch-reconstruction note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1,r=4,K=2$、$q=1,2,4,8$で、限定4段集計と同じ
$\beta=(0.02,0.02,0.36)$および重み付き$\alpha$を使い、$q=8$物理信号と4段の
逐次分枝復元を検証した。

$q=8$のfinite-RTE信号半径は0.993219824261、exact信号からの系統位相差は
$1.06819\times10^{-4}$ rad、厳密二項の統計位相失敗率は$2.07129\times10^{-6}$だった。
4段合成の座標失敗率は$1.33625\times10^{-3}$、統計位相失敗率は
$2.14873\times10^{-6}$で、割当予算0.05以内だった。

4段8軸の全1,572 shotへ異なるRTE trajectory seedを割り当てた明示的監査を行い、seed重複なし、
trajectory平均の解析信号からの差は最大1.925標準誤差だった。解析的周辺分布から生成した
10万回の4段測定では分枝失敗・最終位相失敗とも0件で、最終失敗率の片側95%上限は
$2.99569\times10^{-5}$だった。

固定H4の4段end-to-end接続はlocalに通過したが、目標エネルギー精度からのround数決定、$q>8$、
別条件、実backend、noise、状態準備、最終総cost、immutable CIは未評価である。PF係数が経験値の
ため保証statusは引き続き`empirical_screening`である。詳細は
[4段分枝復元検証](docs/rpe_four_round_phase_validation.md)。
変更後のlocal全テストは`484 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-18 limited four-round RPE accounting note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1,r=4,K=2$、$q=1,2,4,8$で、前回選んだ暫定配分
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$と重み付き
$\alpha$を既存の厳格な資源集計APIへ接続した。$q=1,2,4$の固定Hadamard直接costと、
未使用$q=8$ holdout通過済みproxyを出典付き複合providerとして使用した。

4段・8軸の合計は1,572 shot、RZ数32,673,960.607143で、shot×1 shot costの直接再計算と
前回診断値に一致した。保守的$\alpha$ union boundは0.05の予算内だった。
前回の$q=1,2,4$物理信号を固定し、新しいshot数と$\beta_{\rm stat}=0.36$で厳密二項失敗率を
再計算すると、合成座標失敗率$2.2246\times10^{-4}$、統計位相失敗率
$7.7444\times10^{-8}$で、各軸・各段の割当額を満たした。

これはlocal dirty-worktreeの限定4段診断で、PF入力が経験値のため`empirical_screening`である。
$q=8$物理信号・厳密失敗率、4段branch復元、最終全round総コスト、実backend、noise、
immutable CIは未評価。詳細は[限定4段検証](docs/rpe_four_round_accounting_validation.md)。
変更後のlocal全テストは`481 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-01 beta/alpha allocation-sensitivity note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4,8$で、RPE位相誤差・失敗確率配分の感度を評価した。
$q=1,2,4$の状態準備なしHadamard直接compiled costと、未使用holdoutを通過した$q=8$ proxyを
固定入力としたため、sweep中の回路再compileはない。

5種類の$\beta$配分と、一様／cost感度重み$\alpha$配分の10 scenarioは、$\beta$和、$\alpha$和、
PF・RTE実寄与、正半径、shot式、round cost恒等式を全て通過した。単一条件への過適合を避ける
暫定100倍headroom規則では$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$と
cost感度重み$\alpha$を選んだ。各軸shotは$q=1,2,4,8$で229、207、186、164、RZ comparison costは
$3.2674\times10^7$である。現行$(0.08,0.08,0.24)$・一様$\alpha$比では56.35%小さいが、同じ
$\beta$での$\alpha$変更単独は4.35%だった。

これは1 snapshot・1 compiler・固定$(L_D,\delta,r,K)$・RZ指標のlocal dirty-worktree sensitivity
diagnosticである。100倍guardは理論値でなく、比較costは最終総costでもその削減率でもない。
選択後の非一様$\alpha$に対する厳密二項失敗率・仮想測定も再実行していない。今回の失敗確率条件は
Hoeffding shot式とunion boundである。$q=8$物理信号・位相、branch reconstruction、$q>8$、noise、
実backend、immutable CIは未評価である。
変更後のlocal全test suiteは`479 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-01 q=8 Hadamard cost-proxy/resource connection note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしで、状態準備を除くHadamard interrogation全体のcompiled costを
評価した。各$q$ 8 trajectoryの$q=1,2,4$だけでaxis・metric別affine proxyを較正し、
係数固定後に未使用$q=8$の一体compileを予測した。

両軸・全6指標の最大相対点誤差はcircuit sizeの0.726%、RZ countは0.675%で、
事前の5%基準を通過した。較正点とholdoutを含むRZ平均の最大相対標準誤差は
0.732%で、事前2%条件を通過した。holdoutはfitに使用していない。

通過したvalidation fingerprint、Hamiltonian・DF split、$L_D,\delta,r,K$、compilerを要求し、
実際にholdoutした$q$だけを返すproviderをresource accountingへ接続した。
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.08,0.08,0.24)$ rad、
$\alpha_{m,b}=0.05/8$の$q=8$ candidateは各軸414 shot、1 shot RZ count 48135.0804となり、
round RZ cost $3.9855847\times10^7$の再計算が一致した。未検証$q=16$は拒否した。

これは1 snapshot・1 compiler・$q=8$のlocal dirty-worktree evidenceである。$q>8$、別分割、
proxy係数共分散、$q=8$物理信号・位相、複数round合計、状態準備、noise、実backend、
最終総costまたはimmutable CI evidenceではない。
変更後のlocal全test suiteは`478 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-01 virtual-Hadamard statistical failure note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4$で、finite-RTEの物理基底状態信号を用いた
仮想Hadamard測定を検証した。$(\beta_{\mathrm{PF}},\beta_{\mathrm{RTE}},
\beta_{\mathrm{stat}})=(0.08,0.08,0.24)$ rad、$\alpha_{m,b}=0.05/6$から得た各軸shot数は
389、390、391である。

厳密二項計算では、6軸のいずれかの座標誤差が許容量以上となる確率は0.0010363、
3 roundのいずれかの統計位相誤差が0.24 radを超える確率は$1.4775\times10^{-6}$で、
いずれも$\alpha_{\mathrm{tot}}=0.05$以内だった。10万回の周辺Bernoulli反復では座標失敗111回、
位相失敗0回で、片側95%上限はそれぞれ0.001299、$2.996\times10^{-5}$だった。

別に各shotへfresh IIDなRTE trajectoryを割り当て、計2340 trajectoryをsector内で直接作用した。
seed重複はなく、trajectory平均信号と解析信号の差は最大2.063標準誤差、全軸の条件付き測定数は
周辺二項分布の99.9%中央区間内だった。これはshort-roundの測定統計とfresh-IID実装のlocal検証であり、
全roundのRPE branch selection、最終位相復元、実backend、noise、状態準備、最終総costまたは
immutable CI evidenceではない。変更後のlocal全test suiteは`475 passed, 4 warnings`だった。
最初の全suite実行では既存の並列SQLite cache testが一時的な`database is locked`で1件失敗したが、
単独再実行と続く全suite再実行では通過した。4 warningは既存grouped-UWC test由来である。

## 2026-09-01 short-round signal・shot・compiled-cost connection note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4$で、finite-RTE信号検証、RPE shot式、
controlled time-evolution direct provider、状態準備なしHadamard interrogation providerを接続した。
物理full-$H$基底状態のPF信号半径は1から最大$1.11\times10^{-8}$のずれで、単位半径仮定と
実半径から得る各軸shot数は$q=1,2,4$で389、390、391と一致した。

同一compiler条件ではHadamard interrogationのRZ countはtime-evolution部分より各軸+2、
circuit sizeは+5で、CX count/depth、RZ depth、total depthは同じだった。
全roundで`round_cost=N_c g_c+N_s g_s`の再計算、scope識別、古典Monte Carlo標本数8を
量子shot数へ追加乗算していないことを確認し、専用payload validatorを通過した。

これは1 snapshot・1 compiler・$q\leq4$のlocal接続検証である。8 trajectoryのcompiled-cost
点推定を精密な候補順位または最終総costには使わない。仮想Hadamard測定は上記の別検証で
追加した。$q=8$ proxyの1 round接続は上記の別検証で追加したが、$q>8$、
全round集計、状態準備、noise、実backend、immutable CIは未評価である。
変更後のlocal全test suiteは`473 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
保存した接続結果JSONは専用validatorを再通過した。

## 2026-08-26 H4 follow-up・H5 system-size circuit-cost completion note

2026-08-25のH4 follow-upは全jobがreturn code 0で完走し、専用validatorを通過した。同一H4 chain、
距離1.0 Å、STO-3G、DF rank 12、固定snapshot、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしの範囲で、$L_D=0$の全Taylor patternを同一trajectoryで比較した
K1--K3 paired residualは全metric最大1.679%、主RZ 95%上側診断3.201%だった。$L_D=6$、
short-step 0.025、$K=2$の固定K1--K4を未使用$L=8$へ適用した点誤差は最大3.750%だが、
主RZ 95%上側診断は7.045%だった。同一trajectoryのpaired K1--K4 $L=8$構造検証は最大4.008%、
主RZ 95%上側診断4.902%だった。controlled $q=1,2$ affine modelの未使用$q=8$ holdoutは
全metric最大0.0529%だった。

H5 chain、距離1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、short-step 0.025、
$K=2$、同じcompiler条件では、$L=4,6,8$のpaired K1--K3構造残差が全metric最大1.665%、
K1--K4が0.551%だった。5%を満たす最小cluster長としてK1--K3を選んだ。独立calibrationは
最大RZ相対standard error 0.745%で停止し、all-order-0各長さ500、single-order-2各位置125の
独立full holdoutは全metric最大点誤差3.776%、主RZ最大z 2.009、予測側95%半幅1.459%だった。
事前の5%点誤差と2%予測精度を通過した。点wise正規近似95%上側診断7.461%は硬い受理条件ではなく、
rigorousな5%保証とは扱わない。

したがって、exact Hamiltonian snapshot・compiler条件ごとにK1--K3を較正して代表holdoutを行い、
不通過条件だけK4へ進む運用規則を採用する。回路costの広いpilot検証は一旦区切るが、$L>8$、
compiler/coupling/backend変更、新snapshot、controlled scope拡張または5%未満の候補差では再検証する。
これらはlocal dirty-worktree evidenceであり、full RPE、quantum shot、noise、実backend、最終総cost、
immutable CIまたは外部再現結果ではない。

整理後のlocal全test suiteは`472 passed, 4 warnings`だった。4 warningは既存grouped-UWC testの
complex-to-real castであり、今回の回路cost検証由来ではない。H4 paired K4、H5 paired、H5独立
calibration、H5独立holdoutの保存payloadは各専用validatorを再通過した。

## 2026-08-25 circuit-cost follow-up実装note

最初の追加batchはA--Cすべてreturn code 0で完走し、専用validatorを通過した。Aの独立K1--K3は
主誤差12.376%で不通過だが最大z 1.787で原因を分離できず、Bの独立paired K4は最大点誤差
1.281%だがRZ 95%上側診断5.714%、Cのcontrolled $q=8$は全metric最大0.0529%で通過した。

follow-up用に、複数order-2のpaired full/local-window残差、K4 500標本へのincremental resume、
固定K1--K3のL8 holdout、固定K1--K4のL8評価を実装した。変更後のlocal全suiteは
`470 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。follow-upのlive stateは
`artifacts/rte_cost_followup_batch/2026-08-25/status.json`へ保存し、成果物validator通過前は
追加科学結果として扱わない。

## 2026-08-25 circuit-cost追加batch実装note

同一H4 fixed snapshot上で、$K=2$の複数order-2 pattern、$L_D=6$の対応あり独立K4、
controlled $q=8$ holdoutを採取する実装とdetached batch runnerを追加した。
変更後のlocal全suiteは`469 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
数値jobのlive stateは`artifacts/rte_cost_data_batch/2026-08-25/status.json`に保存する。
完走済み成果物を専用validatorで検証するまでは、追加の科学結果や最終cost評価として数えない。

## 2026-08-24 connected-cluster short-step 0.030 transfer completion note

中断していたH4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、short-step 0.030、
finite Taylor cutoff `K=2`、同一Hamiltonian snapshot・compiler条件のtransfer holdoutを再開した。
固定calibrationに対し、未使用`L=4,6` full回路の全order-0を各1500標本、order-2が1回の条件を
各位置500標本とした。

全6 metricの最大絶対相対点誤差は3.541%、主RZ countは3.488%、RZ最大absolute z-scoreは
3.037、予測側95%相対半幅は1.784%だった。事前の点誤差5%と予測精度2%は通過した。
500/150標本の予備runでは全metric 7.138%、RZ 5.754%、予測半幅2.265%で不通過だったため、
当該超過は再開runで維持されなかった。再開時はcalibration相対標準誤差目標も1.0%から0.8%へ
締めており、改善をholdout標本数だけには帰属しない。一方、RZの点wise正規近似95%上側診断は5.738%であり、
残差ゼロまたはrigorousな5%保証とは扱わない。

再開には旧pattern checkpoint、v3固定sample chunk、既存SQLite metric cacheを併用した。8000回路中
1250件がpersistent cache hit、6750件がmissで、3 workerの実経過時間は393.2秒だった。生成artifactは
内部fingerprint・chunk意味検査を通過し、一時checkpointは残っていない。この結果はdirty local
worktree evidenceであり、別compiler、coupling map、系サイズ、full RPEまたは最終総コストへは
一般化しない。

## 2026-08-24 connected-cluster lightweight operation implementation note

既存の主1500/375 holdout結果を変更せず、compiled-cost処理をoffline calibration、transpileを
呼ばないprediction、固定calibrationのtransfer holdoutへ分離した。v2 calibration/transfer schema、
pattern単位task/checkpoint、same/different基底の直接条件付きsampling、cache状態に依存しない
deterministic-work Neyman配分、完全な数値回路・compiler・backendをkeyとするSQLite metric cacheを
実装した。さらにv3では各patternの標本列を固定indexのsample chunkへ分割し、chunkごとの十分統計量を
atomic checkpointへ保存して合成する。標本数を増やす際は完了済みfull chunkを再利用し、末尾partial
chunkの置換と新規chunkだけを計算する。same/different pairはclass別統計量を合成後に解析確率で
再重み付けする。旧v2 pattern checkpointは標本数まで完全一致する場合だけ読み取り再利用する。
production後の実現相対標準誤差が目標未達なら、分散に基づき不足層へ再配分するadaptive roundも追加した。

同じevent identityを複数short-step時間でcompileする角度不変性validatorも追加した。既存H4固定
snapshot、`L_D=3`、short-step 0.020/0.025/0.030、cluster長1--3、各pattern 2標本のsmokeでは
全6 metric差0だった。ただし低標本のimplementation smokeであり、manifestへ科学的artifactとして
登録せず、角度を除外したstructural cache reuseも有効化しない。別`L_D`、境界coverage、
compiler/coupling条件で検証するまで、short-stepごとに数値回路keyを分離する。

H4のpilot 2・production cap 4 smokeはcold約39.8秒、別checkpointから同じSQLite cacheを使う
warm run約37.3秒だった。warm transpile missは0だが、回路構築とcanonical fingerprint計算が残る。
同一checkpoint再開は完了taskを読み飛ばす。checkpoint fingerprintにはtask実装versionも含め、
実装変更後に旧summaryを黙って再利用しない。固定chunk範囲、十分統計量の合成、標本数増加時の
full chunk再利用、旧checkpoint互換、worker数による科学的出力の不変性、chunk seed改変の拒否を
回帰testで確認した。変更後のlocal全suiteは`468 passed, 4 warnings`で、warningは既存grouped-UWC
test由来である。本節は実装能力の記録であり、
新しいcost精度、full RPE、
最終総コストまたは大規模性能の検証ではない。

## 2026-08-24 operational connected-cluster compiled-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`、`K=2`の
1 compiler条件で、Taylor次数条件付きのK1--K3 connected-cluster運用推定器を検証した。
order-0単eventは厳密列挙し、pairはsame/different基底で層別、pilotからRZ countのNeyman配分を
決めた。productionとholdoutのseedは分離した。

正確なDF Hamiltonian配列をNPZ snapshotへ固定し、未使用full回路を`L=4,6,8`で評価した。
全order-0は各長さ1500標本、order-2がちょうど1回は各位置375標本である。RZ countと全6 metricの
最大点誤差は2.936%、RZ最大absolute z-scoreは2.074、予測側95%相対半幅は1.537%だった。
事前の点誤差5%と予測精度2%は通過した。一方、点wise正規近似95%上側診断は5.724%で未達、
order-2 K1は要求1535に対して標本cap 1500へ到達した。したがって代表1条件の
「実用点誤差5%内の暫定候補」であり、rigorousな5%保証とは扱わない。

別processで同じ分子条件から再構築したholdoutは、元holdoutと最大11.68%、z 5.215ずれた。
これは主結果へ結合していない。compiled-cost検証の再現単位は分子条件だけでなく正確なDF snapshotとする。
generatorはpilot K1--K3、production K1--K3、holdout L4/L6/L8の9 taskをfingerprinted checkpointへ
保存し、中断後は未完了taskだけを再開する。

全order-0 RZの差は`L=4,6,8`で+2.59%、-0.35%、+2.94%と単調増加せず、order-2が1回の
RZ点誤差は1.02%以下だった。この条件ではK4を追加せず、別`L_D`、short-step、compiler/coupling
条件への移送holdoutで悪化した場合に再検討する。full RPE、量子shot、noise、実backend、
resource accounting接続、最終総コストは未評価である。

主artifact、snapshot、9 checkpoint、versioned source、2 generator、専用test 4件および
[`docs/rte_connected_cluster_cost_validation.md`](docs/rte_connected_cluster_cost_validation.md)を
追加した。fingerprintとsnapshot SHA-256は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

当該主artifact生成後に軽量運用APIを追加した最新local全test suiteは`468 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のconnected-cluster検証由来ではない。

## 2026-08-24 hierarchical compiled-cost holdout note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`の
同一Hamiltonian表現を3 workerへ渡し、`K=0`の未使用`L=8`、`K=2`の`L=4,6`、
controlled partial-S2の未使用`q=4`を独立seedで評価した。三artifactの`preparation_hash`は一致した。

`K=0`では`C2,C3,C8`を各2000標本とし、pair-onlyはcount/sizeで最大8.851%残った一方、
triple補正は全metric最大1.744%、最大absolute z-score 0.521だった。したがって、この条件では
count/sizeにtriple、depthにpairを候補とし、4-event以上の係数は追加しない。

`K=2` runを監査すると、1 eventのorder-2確率は0.0001063で、旧`C1,C2,C3,C4,C6`
全8000 event位置中order-2は1回だけだった。したがって旧4.113%値は`K=2`内部の根拠から外す。
Taylor次数を強制した独立較正/holdoutでは最大点誤差9.119%、最大z 1.651となり、500標本の
独立係数差引きは精度不足と確認した。一方、同一trajectory上で1--3 event局所窓と全回路の
差を直接取る対応あり検証では、`L=4,6`のall-order-0/order-2が1回の全条件・全metricで
最大点誤差1.373%、RZ countの点ごとの正規近似95%診断1.796%だった。最大z 7.535なので
小さい4-event以上の残差は非ゼロだが、代表条件の暫定5%を通過した。運用時は解析的order重みと
対応ありconnected-cluster係数を使い、独立500標本係数推定は使わない。

controlled `q=1,2`各300標本から別seedの`q=4`を予測すると、全metric最大点誤差0.307%、
最大z 0.925、点ごとの正規近似95%上側診断0.958%だった。この代表条件ではaffine `q` model候補を支持するが、
`q>4`、別`L_D`、compiler/coupling条件または最終resource accountingへ一般化しない。

versioned source、並列generator、専用test、五つのfingerprinted local artifactおよび
[`docs/hierarchical_cost_validation.md`](docs/hierarchical_cost_validation.md)を追加した。
fingerprint、source hash、seed分離は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

最新local全test suiteは`457 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のcost検証由来ではない。

## 2026-08-23 high-statistics and stratified RTE boundary-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`、short-step時間0.025の
同一Hamiltonian表現を3 workerへ渡した。独立2 seedについて`C2,C3,C4,C6`を各1000標本、
別のcalibration/holdout seedでfragment-pair補正を1500/1500標本評価した。三artifactの
`preparation_hash`は一致した。

1000標本runでpair補正の最大絶対相対誤差は8.07%、8.18%、最大absolute z-scoreは
2.96、3.02だった。系統差はcount/sizeで確認し、depthではpair-onlyの最大zは1未満だった。
triple補正は最大2.33%、3.73%、最大z 0.79、0.97だったが、`mu3`自体のabsolute z-scoreは
最大1.40なので、非ゼロを確定したとは扱わない。

same-fragment確率は0.7310604だった。different境界をゼロとするsame-only modelは別seedの
pair holdoutに対して最大誤差3.65%、最大z 2.59で外れた。same/different双方の条件付き補正を
解析確率で重み付けすると最大0.849%、最大z 0.587だった。したがってpair係数には少なくとも
二分類が必要であり、長いevent列ではdepthをpair候補、count/sizeをtriple候補とする。

`rte_boundary_pair_validation_v1` source、並列generator、専用test、三つのfingerprinted local
artifactおよび[`docs/rte_boundary_pair_validation.md`](docs/rte_boundary_pair_validation.md)
を追加した。受理閾値、他の`L_D,K`、compiler、controlled回路、resource accounting接続、
最終総コストは未評価である。artifactはdirty local worktree evidenceでありimmutable CI
evidenceではない。repository全体の`overall_status = not_reproducible_from_repository`は
変更しない。

最新local全test suiteは`451 passed, 4 warnings`だった。

## 2026-08-23 RTE boundary-corrected compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`でshort-step時間を0.025に
固定し、RTE event列のcompiled-cost cluster modelをcalibration/holdout分離して検証した。
`C1`は218 eventを厳密列挙し、`C2,C3`は各300標本で較正した。別seedの未使用`C4,C6`を
各300標本で評価した結果、六指標を通じた最大絶対相対誤差はevent単純和157.51%、pair補正
8.73%、triple補正4.06%だった。最大absolute z-scoreは48.21、1.50、0.52だった。
同一DF fragmentが隣接する確率は0.73106だった。

triple残差は全metricで自身の標準誤差より小さく、depthではpair補正を一様に改善しなかった。
したがってpair補正を次の最小model候補、triple項を高次境界効果の診断量として記録する。
受理閾値、他parameter・compiler条件への一般化、resource accounting接続および最終総コストは
未評価である。

source、generator、専用test 2件、versioned/fingerprinted JSON artifact 1件および
[`docs/rte_boundary_cost_validation.md`](docs/rte_boundary_cost_validation.md)を追加した。
最新local全test suiteは`449 passed, 4 warnings`だった。artifactはdirty local worktree
evidenceでありimmutable CI evidenceではない。したがってrepository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-23 random-circuit compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`K=0`で、
complete circuitのcompiled costと、部分回路を別々にtranspileしたコスト和をpaired比較した。
`r=1`のpartial-S2は218 trajectoryを完全列挙し、3部分加法モデルは最大0.987%過大評価した。
同じDF表現を共有した`r=2`の100標本では、event別加算がRTE occurrence一体compileを
48.30--57.58%過大評価し、paired differenceのabsolute z-scoreは15.86--17.79だった。
別DF表現の300標本replicateでも52.99--61.15%の過大評価を確認した。
したがって、個々のeventを独立加算するモデルは採用せず、RTE occurrence以上をcost proxyの
最小較正単位とするpilot判断を記録した。

source、generator、専用test 3件、versioned/fingerprinted JSON artifact 3件および
[`docs/random_circuit_cost_validation.md`](docs/random_circuit_cost_validation.md)を追加した。
関連する既存cost testを含む最新local実行は`53 passed`だった。artifactはdirty local worktree
evidenceであり、1条件だけのpilotである。controlled回路、実backend、量子shot、ノイズ、
全round RPE、最終compiled総コスト、または他の`L_D,delta,r,K,q`への一般化を検証したものではない。
また、同じ分子条件でもprocess間でDF `preparation_hash`が変わるため、主$r=1,2$比較は
Hamiltonian共有batchへ置き換えた。異なるhash間の絶対compiled costは直接比較しない。

したがって、repository全体の`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-19 current local approximation-validation note

研究内容と現在地の短い統合要約は
[`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)を参照する。

2026-08-19のdirty local worktreeでは、最終コスト評価の入力を検証するため、次の三つの
result setを追加した。

- finite-RTE signal、attenuation、radius、phase-boundのH4 grid検証
- PF誤差surrogate、論文Appendix D Eq. (D6)のCPU Qiskit摂動係数、理想QPE分枝の
  H4全`L_D`検証
- H2--H6の実行可能delta窓と、PF演算子を構築しないEq. (D6) state-action係数検証

対応する文書、source、test、dirty-worktree artifactはmanifestへ登録され、構造検査は
成功している。H4全12分割では、well-conditionedな4点でfitしたEq. (D6)係数と
支配固有位相係数の差が最大0.288%だった。H2--H5の実行可能窓では両者の上包絡差が
最大1.144%で、事前の2%条件を全系が通過した。H6（DF rank 11、$L_D=5$）ではEq. (D6)による
`C_use=0.02086663`を得た。local testは`444 passed, 4 warnings`だった。これらは近似手法と実装経路の
local evidenceであり、最終compiled cost、H12の係数、量子shot、ノイズ、または外部から
再現された科学的結論ではない。artifactはimmutable CI evidenceでもない。

したがって、下記auditの`overall_status = not_reproducible_from_repository`は変更しない。
旧DF screeningとprose-only UWCを使用禁止とする判断も引き続き有効である。

## 2026-08-02 implementation hardening note

The current worktree replaces the DF legacy overlap proxy with a
shift-invariant, state-specific survival-phase-bias estimator (cache schema 8,
definition v3). It records explicit estimator status and is marked
`is_rigorous_bound=false`. Legacy/unmarked Cgs tables are now rejected by the
analytic PR-bound screening entry point, so the stale rankings described below
remain invalid and cannot be silently regenerated from the new surrogate.

Finite RTE distribution validation/serialization, actual-circuit/backend cache
identity, exact-zero circuit pruning, pre/post build workload guards, bounded
metric-only LRU caching, online compiled-cost statistics, and rolling
provenance digests were also hardened. A non-scientific Level-5-R regression
fixture now freezes all 32 combinations of `q=1..4`, raw/boundary-optimized,
controlled/uncontrolled, and exact/Monte Carlo compiled-cost evaluation.

The follow-up memory hardening makes event, partial-S2 request, exact
trajectory, and Monte Carlo trajectory generation single-pass. Level-5-R
provenance retains an explicit bounded prefix (1024 records by default) while
rolling digests cover the full stream. Cache-independent total build,
transpile-request, and instruction-application plans are rejected before any
Qiskit builder is called and checked against actual post-build work. The
Level-5-R fixture was rerun without changing gate/depth/mean/standard-error
values; schema 2 adds runtime/PRNG metadata and a digest over all 32 result
streams. The local regression suite reports `224 passed` in the documented
Python 3.11 environment. This remains implementation evidence only; no H3--H14
scientific baseline was generated by this change, and no long RPE circuit,
quantum shot, GPU statevector, noise simulation, or backend job was run.

## 結論

**監査基準 commit [`cf285c0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cf285c0ac1e3d587df4a8eb6bee2279a12ced462) の内容だけでは、現在の DF screening / UWC 検証結果を外部から再現・追跡できません。**

ここで「外部から再現可能」とは、clean checkout から、commit 済みの入力と手順を使って結果を再生成し、その結果が公開済みの数値と一致することを確認できる状態を指します。本書は既存 artifact の棚卸しであり、新たな科学計算を実行した結果ではありません。

> **DO NOT USE:** 現在 commit されている DF screening JSON を、修正済みの結果または最終結果として引用しないでください。ファイル内の算術は整合していますが、その Cgs 入力は後の commit で基底状態の不整合を理由に削除され、screening は再生成されていません。

## ステータス一覧

| 対象 | commit `cf285c0` にある証拠 | 判定 | 読み方 |
|---|---|---|---|
| 旧来の高次 Trotter 評価 | 出力付き `abe_trotter_project.ipynb` と、`artifacts/trotter_expo_coeff_gr{,_original}/` 内の係数 pickle 計 540 個 | **historical** | README が説明する旧来の高次 Trotter 解析の成果。現在の DF screening / UWC の検証証拠ではない |
| DF reduced screening | `epsilon_total=1e-4` の JSON 1件。635候補、12分子の best を収録 | **DO NOT USE / stale** | 保存値の加算と best 選択は内部整合するが、元の Cgs 表が削除済みで再生成不能。protocol 上も shortlist 前の近似 screening |
| DF 最終評価 | protocol と実装 | **incomplete** | shortlist の explicit-`L_D` Cgs 再 fit、H14 `8th(Morales)`、`4th(new_2)` が未完了 |
| UWC | 実装説明と H2--H6 等の数値表を含む Markdown | **reported only** | 表が参照する machine-readable JSON は commit されておらず、表から元 run を追跡できない |
| テスト | 4ファイルに `test_*` 関数定義が28件。UWC note に過去の `26 passed` の記録 | **current result unknown** | `cf285c0` に対するテスト実行結果ではない。この変更で追加する manifest 構造検査も科学計算・全 test suite は実行しない |

## 証拠と監査結果

### 1. DF screening

対象 artifact:

- [`artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json`](artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json)
- [`Partial Randomized Study Protocol.md`](Partial%20Randomized%20Study%20Protocol.md)
- [`artifacts/partial_randomized_pf/README.md`](artifacts/partial_randomized_pf/README.md)

JSON 自体について確認できる範囲は次のとおりです。

- `candidates` は635件で、1件は `(molecule, PF, L_D)` の組です。
- `best_by_molecule` は H3--H14 の12件です。
- 全635候補で、保存値の `g_total` は `g_det + g_rand` と一致します（最大絶対差 0）。
- 12件の `best_by_molecule` は、それぞれ同じ molecule の候補中で最小の `g_total` と一致します。

これは **JSON 内部の算術と選択処理だけ** の確認です。入力データ、Cgs fit、物理モデル、または結果の科学的妥当性を検証したことにはなりません。

再現性を失っている直接の理由は次のとおりです。

1. JSON の `cgs_table` は `/home/AbeHiromu/Project/.../df_cgs_cost_table.json` という生成環境の絶対パスを指します。
2. commit [`98f960c` (`基底状態ずれてたので削除`)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604) は、集約 Cgs 表、split 表、index の計37ファイルを削除しています。
3. その後も上記 screening JSON は残っていますが、削除理由を反映した正しい Cgs 入力から再生成された artifact はありません。

また protocol は、この計算を候補を絞るための近似と定義しています。screening では anchor の `C_gs,D(p,L_anchor)` を各 `L_D` に使い回し、**最終評価では shortlist の各 `(p, L_D)` で Cgs を再 fit して `G_total` を再計算する必要があります**。同じ protocol には、次も未完了と記録されています。

- H14 `8th(Morales)` の anchor Cgs
- H3--H14 `4th(new_2)` の anchor Cgs 計算、cost table への merge、再 screening
- shortlist に対する explicit-`L_D` Cgs の再 fit

したがって、入力問題がなかったとしても現在の JSON は最終結果ではありません。

### 2. UWC

[`notes/uwc_current_implementation_and_results.md`](notes/uwc_current_implementation_and_results.md) には、H2--H6 grouped UWC、H3 time-grid 診断、theta sweep、simple shift の条件と数値表があります。一方、同文書が参照する次の出力を含む `artifacts/grouped_uwc_pf_qpe/` は commit `cf285c0` に存在せず、`.gitignore` でディレクトリ全体が除外されています。

- `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json`
- `H3_bliss_sector_scaling_diagnostics.json`
- `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json`
- theta sweep 表の元になった run 出力

したがって Markdown の表は「報告された数値」として読めますが、repository 内の canonical raw/summary artifact と照合することはできません。なお文書自身の結論も、現在の simple BLISS quadratic shift では grouped PF+QPE cost がほぼ低下していない、という限定的なものです。

### 3. テストと CI

commit `cf285c0` の `tests/` には、静的に数えた `test_*` 関数定義が28件あります。

- `tests/test_df_hamiltonian.py`: 5件
- `tests/test_df_partial_randomized_pf.py`: 9件
- `tests/test_grouped_uwc_comparison.py`: 7件
- `tests/test_uwc_preprocessor.py`: 7件

UWC note が保存している実行記録は `.venv/bin/python -m pytest -q` の `26 passed` です。これは後から追加されたテストを含む現在の suite に対する結果ではなく、実行 commit、依存環境、完全なログも記録されていません。監査基準 commit `cf285c0` には `.github/workflows/` もありませんでした。この変更では manifest と記載パスの構造検査だけを追加しており、科学計算または全 test suite の CI ではありません。このため、`cf285c0` の28定義が pass するとは本監査から主張できません。

## 再現を妨げているもの

- 修正済みの DF Cgs 集約表・split 表・index がない。
- stale screening JSON に入力 hash、生成元 commit、実行環境、実行 command がない。
- DF screening の修正後再実行と shortlist の explicit-`L_D` 再 fit がない。
- protocol に記載された H14 `8th(Morales)` と `4th(new_2)` が未完了。
- UWC の Markdown 表に対応する machine-readable run artifact がない。
- UWC artifact の保存先が `.gitignore` され、レビュー可能な canonical summary の例外設定がない。
- 現在の HEAD を対象とする自動テスト結果がない。

## 「検証完了」とするための条件

以下をすべて満たした時点で、DF / UWC の結果を repository から外部検証可能と扱います。

1. **DF 入力を修正して固定する。** ground-state のずれを修正した Cgs を再計算し、集約表、全 split 表、index を同時生成する。各表に molecule、PF、`L_D`、入力 Hamiltonian hash、生成元 commit、生成 command を記録し、相互の件数と hash を検査する。
2. **未完了の DF ケースを埋める。** H3--H14 `4th(new_2)` と H14 `8th(Morales)` の必要な anchor Cgs を生成し、同じ canonical table に merge する。失敗または除外する場合は、対象、理由、結果への影響を明記する。
3. **screening を再生成する。** 修正後の canonical table だけを入力として `epsilon_total=1e-4` screening を実行する。結果には相対的な入力 path、全入力の content hash、生成元 commit、command、依存環境、candidate 件数を保存する。`g_total = g_det + g_rand` と molecule ごとの best 選択を自動検査し、旧 JSON を stale として置換または明確に隔離する。
4. **最終 DF 評価を実行する。** screening の shortlist と選定規則を保存し、各 `(PF, L_D)` で anchor ではない explicit-`L_D` Cgs を再 fit して `G_total` を再計算する。最終表から各 fit の machine-readable artifact と入力 hash へ追跡できるようにする。
5. **UWC の根拠データを公開する。** Markdown に載せる全表について canonical JSON/CSV を commit し、条件、baseline、seed（使用時）、backend、入力 hash、生成元 commit、command を保存する。Markdown の値が artifact から自動生成または自動照合されるようにし、必要な summary だけを `.gitignore` の例外にする。
6. **clean checkout で検証する。** 固定した依存環境と文書化した command で、小規模な end-to-end 再生成および全 test suite を CI から実行する。結果 artifact を作った commit に対する成功 check を GitHub 上に残し、比較 tolerance と期待値を test または検証 script に固定する。

上記が完了するまでは、旧来の高次 Trotter artifact、DF screening、UWC 表を互いに独立した進捗資料として扱い、現在の partial-randomized DF/UWC の完成済み検証結果として一括して引用しないでください。


## 2026-10-07 H4新host source取得・環境解決案

[H4新host source取得・環境解決案](docs/research/track_a_h4_new_server_preparation_20261007.md)。指定commit/source19照合済み、環境18 version/45 RECORD差の解決案でSTOP。source修正・新tests・production未実施、入力/停止証拠別送待ち、allowed_cpus=[]/approved=false。旧科学結果・原稿・Track Bは保持。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](docs/research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](docs/research/track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。


## 2026-10-08 H4受領監査・独立最終review TECHNICAL FAIL

[統合入口](docs/research/track_a_h4_receipt_final_review_20261008.md)。origin79827016/SOURCEad57d163照合、source32不変、read-only live profile/資源/quotaを確認。
凍結NPZ6/freeze/native stop/controlは未受領、producer bytes/SHA manifestを含む最小転送手順を具体化。科学array読込/再生成0。
独立reviewはpidfd送信時ESRCH競合で後続cleanupが中断するP1を純mock再現しTECHNICAL FAIL。57人工PASSはこの競合を覆わない。
未受領NOT_EVALUABLE、CPU/env/observer/74804案のUNAPPROVEDと技術FAILを区別。sourceは修正せず再sealなし、flags false、carry20/165214360/5466.188392877579を保持し本計算STOP。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](docs/research/track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](docs/research/track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](docs/research/track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](docs/research/track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。


## 2026-10-09 H4一度起動・Matplotlib run外設定write拒否STOP

[実結果](docs/research/track_a_h4_authorized_launch_20261009.md)。immutable認可/fresh gate PASS後driver2740551/observer/12 workersを一度起動したが、第一candidate準備のOpenFermion→Cirq→Matplotlib config mkdirが保護guardに拒否されFAIL_CLOSED_STOP。実exit1、14 own identities残存0。newtranspile0/累積actual20、charged4428938712 bytes/wall5472.345380863175秒は返却せず維持。retry/入力再生成/次stage/GPU/共有設定変更なし。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](docs/research/track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。

## 2026-10-09 H4軽量高速化・限定同等性確認

[変更・限定検証・binding](docs/research/track_a_h4_lightweight_speedup_20261009.md)。driverの距離内共通準備を再利用し、one/DF block呼出を静的13×218→13、全prepareを218→10種類に削減。ledger deltaは変更entry/reservation各最大1だけを参照し、全件走査・saved-historyコピーを除いた。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38。限定48人工PASS、全218prep/代表8wrapper+dense256case1/代表4signal/ledger13fileの旧new bytes・digest一致。12workersはmock、単一test process内部thread1、science array/transpile/GPU/affinity/production0。
実Gaussian/旧compiler output/実速度・H4本体成功は未検証。monitor/caps/compiler/science/carry不変、旧partial/cacheと混合しない。
carry20/4428938712B/5472.345380863175s、actual74804既承認、worst charge12.261694GiB>承認10GiBは残る。未seal/approved=false/runtime_authorization=false、absolute_launch_command=null、本計算0。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[認可・source・一度の起動入口](docs/research/track_a_h4_approved_relaunch_20261009.md)。利用者の「これについては問題ないので再実行して」を、既存13GiB cumulative charge案と一度のmap再実行の承認として反映。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`/closure39、output改定schema/gate/実OutputBudget capとmarginのみ変更。legacy10GiB default・科学/compiler/その他caps・prepare再利用/ledger保存は維持。
限定22pure gate PASS、旧48speedup/library/cleanup/native証拠campaign再実行なし。carry20/4428938712B/5472.345380863175s返却なし、actual74804・新残74784。
worst13165893832B <= 新cap13958643712B、余裕792749880B。sealed/approved/runtime_authorization=trueの認可artifactへ再結合。
独立review/artifact固定後fresh SOURCE/profile/input/carry・CPU/memory/PSI/OOM/fs/block/inode/quota/unusedroot/one-shot合格時にそのまま一度起動。実run状態はruntime証跡へ記録。
既承認12workers CPUs2/4–6/8–15、driver16/observer18、thread1・observerAS256MiB/RSS64MiB/admission120.25GiB保持。自動retry/入力再生成/旧partial/cache/次stage/GPU/共有設定変更なし。

## 2026-10-09 H4再実行run02・owned process退出でSTOP

[実結果・native停止監査](docs/research/track_a_h4_approved_relaunch_20261009.md)。13GiB認可・独立review・fresh gate PASS後driver2865403/observer/12 workersを一度起動。最初のwrapper reservation後observerがowned process exitedを記録しFAIL_CLOSED_STOP、driver exit143、14own identities残存0を2回確認。
新reserved1/完了0/signal0。元worker/IPC例外は未永続化で根本原因未確定。carry21/8692723164B、残74783、wall保守的upper5766.582514658794s（post-stop監査待ちを含む）、正確な終了時刻未記録。one-shot/全課金保持、retry/次stage/共有設定変更/GPUなし。

## 2026-10-09 累積17GiB・74805を明示承認、一度のrun03起動へ

[認可・起動入口](docs/research/track_a_h4_production_run03_20261009.md)。利用者の「上限の変更を承認するので本計算に入って」に基づき、累積output13→17GiBとactual74804→74805を承認済みとして反映。
SOURCE6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b/closure44・科学/compiler/options不変。carry21/8692723164B/5766.582514658794s upper保持、消費/予約返却なし。
人工compile省略、既存32 pure gateと独立技術review/proof/benchmark campaignを再実行しない。environment/CPU/observer・72h/他caps採用済条件を保持。
新plan sealed/auth approved/runtime true・exact14CPUsへ再結合。別担当の最終review、immutable artifact、直前fresh gate/未使用one-shotの合格後、一度mapを起動する。
完了/STOP後終了、自動retry/次stage/入力再生成/旧partial-cache混合/GPU/共有環境・venv・他job変更なし。起動/PID/結果は外部runtime receiptで別記録。

## 2026-10-09 run03を一度実行、plugin cache書込み拒否でSTOP

直前fresh gate PASS後driver2930049/observer2930256/12workersを一度起動した。memory465.892GiB、FS412.393GiB、inodes220471919、quota KNOWN、14role physical coresの最大busy4.67%。
SOURCE6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b、最終認可artifact4e8d113533fd880840bb8f6570ae6a636e916615。
最初のscience-000022 reservation後、Qiskit generate_preset_pass_manager→level1→PassManagerStagePluginManagerでworker write guardが未予算cache/temp writeを拒否した。
worker-log-first-stop.txtに元worker errorとphase/PID/serialを保存し、observerへreport/fsync後にown-runをfail-closed停止。driver exit143。
全14identityを2回ABSENT確認、新予約1/完了wrapper0/signal0。exact終了時刻は未記録、worker exit codeは元原因保存時点nullのまま。
carry22/12956511264B/保守的wall upper6004.111340102032sへ更新、残actual74783。上限17GiB/74805は承認済みだが費用返却/resetなし。
first STOPはresource cap/5秒違反ではなくworker_responseでのwrite guard拒否。観測interval0.668673s、duration/stalenessはcommand STOPのためnull、driver phase numerical_serialization:sine。
既存venvのstevedore._cacheはcache missで~/.cache/python-entrypointsへmkdir/writeする。ディスクcacheを無効にしてメモリcacheだけを使うprocess限定修正を検討する。
停止原本/one-shot/journal/ledger/observer/worker原因logはprivate home evidenceに保持し、NPZ/科学runtime/cacheをcommitしない。
自動retry/次stage/入力再生成/共有環境・venv・他job変更/GPUなし。今回の一度の実行はFAIL_CLOSED_STOPで終了した。


## 2026-10-11 H4 encoding再利用：run10を継続

run10は変更せず、独立worktreeで小行列の値をkeyにcanonical textを再利用する実装を固定。
SOURCE 3beb3870369eeb8ef76b4ead965c886eba7d16e9 / closure80、candidate内cosine/sine・trajectoryで共有しcache最大4MiB/1024 entries。
12限定人工tests PASS、全bytes/digest・同じ回路のgate counts/depth一致。人工encodingのみ0.310→0.135秒（約2.29倍）、production全体の速度は未測定。
科学/compiler/回路構成/資源契約不変、追加transpile・実NPZ読込・新本番起動0。将来の新SOURCE採用ではseed/binding再結合が必要。
[実装・限定検証・差分review・資料入口](docs/research/track_a_h4_encoding_speedup_20261010.md)。


## 2026-10-11 H4 run10：高速化版への再開処理を準備

SOURCE 11bba8f0669a264c0e1ef1d71b4d9d14bb394a12 / closure85、dataset identity SOURCE5aa6c368を保持し新実行SOURCEを別記録。
COMPLETEのみledger/bytes/nativeで照合し、完了wrapperのbuild/compileと検証済みsignalの再計算を省略。未完了/孤立は再利用せずcarry保持。
18限定mock PASS、独立差分review PASS、3件のbinding/root検証漏れ修正。実worker/transpile/科学array/新launch0。
run10は継続し変更0。最終停止proof/carry/receiptは未確定、allowed[]/approvedfalse/runtimefalse/未seal/commandnull。
[資料入口](docs/research/track_a_h4_resume_preparation_20261011.md)。


## 2026-10-11 H4 run10→resume01：完了分を保持した高速化継続

利用者「引き継いで」でown run10停止・native6×2残存0を確認。SOURCE 941749b8d4aedb43fd5f47b824116082493df393 / closure85、dataset seed SOURCE5aa保持。
COMPLETE8150/signal142、carry7878/4320481456B/35474.12562095933s、未完了2を保持。worst74512≤74805、chargebound約12.16GiB≤17GiB。
既知sealed停止ログ名の限定修正と19mock PASS。科学/compiler/CPU4/32GiB/observer/pressure/17GiB/74805/72h不変。
最終binding独立差分review・artifact固定・fresh gate後、一度mapを再開する。実状態はprivate receiptへ記録。
[資料入口](docs/research/track_a_h4_production_resume01_20261011.md)。


## 2026-10-11 H4 inverse・命令/定義encoding再利用：resume01を継続

SOURCE def7d48f4af6c07f2d4e361fae868f3b01db8915 / closure89、独立worktreeでgeometry-owned基底inverseを一度保持し、小命令/定義のcanonical fragmentを値keyで再利用。
cache2MiB/2048 entries、key16KiB/384 nodes、definition16命令まで。large/lazy fallback・mutation/phase/control/ordering/nonfinite STOPを保持。
14限定人工tests PASS、旧/new bytes/digest・uncompiled gate counts/depth一致。人工inverse9→0回、encoding0.446→0.320秒（約1.39倍）、production速度未測定。
簡単なauthor差分review PASS、独立launch review/campaign反復0。sampling/seed/signal関数とcompiler/worker/observer/資源契約は不変。
稼働resume01のsource85件SHA一致、runを維持し反映0。新採用bindingはfalse/allowed[]/未seal/commandnull、将来のSOURCE/seed/carry等再結合が必要。
追加transpile・実NPZ読込・新本体起動0。
[実装・限定検査・軽量review・資料入口](docs/research/track_a_h4_inverse_instruction_speedup_20261011.md)。


## 2026-10-11 H4 resume01→resume02 対応：現在runを継続

SOURCE 05a37cebaf18fa28e3a1e8582b86f54e32ba2a66 / closure92。run10由来importとresume01新規COMPLETEを区別してbyte/source/digest/seed/metricsを照合し、次runで完了済みbuild/compileを省略する。
actual=祖先carry+当run予約、ID carry+1、journal sumは過去chargeを含むため二重加算しない。未完了予約・orphan費用と累積72hを保持。
24限定mock PASS、receipt CLI/cache-owner順序/閉じた認可scopeも確認。CLI test環境のPython2変数不足を補正し全試行証跡を保持。
軽量author差分review PASS、科学/高速化/compiler/worker/observer/資源上限は不変。実worker/NPZ/追加transpile/新launch0。
親の6 owned identityをread-only確認しclock630bytesを固定。現在resume01を止めずsource85件一致、停止proof/最終carry/receiptは停止後に確定する。
準備bindingはfalse/allowed[]/未seal/commandnull。将来の明示引継ぎ指示後にreceipt/binding/fresh gateを完成して一度移行する。
[実装・限定検査・binding・将来の手順・資料入口](docs/research/track_a_h4_resume02_preparation_20261011.md)。


## 2026-10-11 H4 resume02：高速化へ引継ぎ・72h STOP解除

SOURCE d0726e946f89f8c30a08d383fb8ff48b363629ec / closure95。利用者の明示引継ぎ・可能なら時間上限解除指示を根拠に、own resume01 driverのみpidfd SIGINT、native6×2残存0を確認。
全80,157files/2世代ledger/bytes SHAを照合しCOMPLETE29388/signal521、carry28731/8788058342B/104895.53687566705sを保持。旧未完了費用返却0。
新wall capnullをdriver/observer両方へ接続し、累積wall記録・trace2GiB/総17GiB/actual74805/5秒/ASRSS/pressureOOM/own-stopを保持。候補環境/compiler/科学/seed/CPU4・32GiBは不変。
12新mock PASS、旧24/人工compile/campaign反復0。独立最終binding・private launcher差分PASS、保守的actual74127、chargebound17573199062Bで承認内。
SOURCE無変更のartifact固定後、直前freshCPU/memory/PSI/OOM/FS/inode/quota/未使用roots/one-shotを合格して一度mapへ。実開始状態はprivate receiptへ別記録。
[資料入口・承認・独立review・起動条件](docs/research/track_a_h4_production_resume02_20261011.md)。


## 2026-10-15 H6 独立資源観測の修正

H6 pilot run02はnative正しさ検証中のcallback stalenessでSTOP、compile0。既存metadata launcherの独立資源観測へ修正し、SOURCEb73580dd…/closure13、限定84testsとGIL保持C約12秒の2process実動作検査PASS。追加本番role/process0、5秒・資源・科学/compiler条件維持、旧消費/unknown保持、subagent0。新run03へ結合して再実行。
[修正・検証・新実行binding](docs/research/track_a_h6_external_monitor_fix_20261015.md)。


## 2026-10-15 H6 /proc退出競合修正・run04-v3起動

H6 run03は全native検証後の退出/proc競合でSTOP、1 worker PASS/受理0・全6 reap。SOURCE2e733d8a…でlive欠測を成功扱いせず5秒内の再観測/reapへ修正、92限定＋最小2childの退出/FD検査PASS。旧消費保持。CPU gateで未起動2grantを廃止、5秒idle選定・fresh gate後にrun04-v3起動。科学/profile/compiler/資源上限不変、subagent0。完了は未主張。
[修正・検証・新実行の入口](docs/research/track_a_h6_proc_exit_fix_20261015.md)。

## 2026-10-15 H6 fixed-delta readiness

[準備資料](docs/research/track_a_h6_fixed_delta_t_dependence_readiness_20261015.md)。46人工/mockとAPI/byte binding PASS、新Tの科学処理0、実成功/独立review/CIを主張しない。SOURCE90a3c06…、approved/runtime falseでSTOP。
