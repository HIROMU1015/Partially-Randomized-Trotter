## 2026-10-19 PR/PF絶対RZコスト図：v2・5精度（保存値のみ）

[成果物・採用除外・再現方法](docs/research/track_a_pr_pf_absolute_cost_v2_20261019.md)。v2 snapshot c0808b25を基準に26比較・52点を描画。両端精度はH4/H6/H8独立確認とH2/H4/H6/H8探索n8の別パネル、中間3精度は探索のみ。H4は新DF10、H6中間PFはmodelを白抜き表示。H2 P未完了・H8詳細公開監査未照合・有限標本の限界を保持。PNG/SVG/PDF各5枚、exact表・採否・caption・SHAを保存。新科学計算/会計/順位計算0、local commitのみ、公開pushなし。次の作業なし、mandatory STOP。以下は各時点の履歴。

## 2026-10-19 共通実験プロトコルv2改訂案（未採用）

[v2改訂案・変更点・条件一覧](docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md)。**v2はDRAFTで実行承認を含まない。** v1（9036d8a8）を保全し、H2を追加対象として提案。最新829d697aのH2 F/FP/D/A66/S_STATS完了・P未sealを反映し、当初7b202a1aのA未完了は時点履歴に分離した。提案n80/12は未採用でP実行未承認。H4/H6/H8独立確認6条件、DF12/DF10・探索/独立cohort、solver/正規化差、未観測event・母平均未保証・H8公開監査の限界を整理。GitHub固定67資料を読取照合。主表/Methods/CEM/原図34ファイルと既存契約を保全。構造16件PASS、新規科学計算・数値解析0。正式採用はユーザー判断、完了後mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H4/H6/H8掲載用図表・caption・Methods（表示固定）

[掲載packet・Methods・公開証拠の限界](docs/research/track_a_h468_paper_display_methods_20261019.md)。利用者の指示で主表＝独立確認6条件、Figure2A＝legacy/探索、2B＝H8独立n48の配置を固定。英語掲載caption・Methods、入力/solver/正規化の8例外、H8公開証拠14項目、CEM/台帳対応を整備。A82後続完成をcaptionで補足し、原図34ファイル・保存数値・旧統合bundleを保全。H8固定PFの公開receiptとPR詳細監査未検証を区別。構造新13＋既存11件PASS、新科学計算/数値解析0、最終freeze/詳細科学レビュー未開始。公開確認後mandatory STOP。以下は各時点の履歴。

## 2026-10-19 H4/H6/H8論文証拠統合（既存結果のみ）

[採用結果・証拠統合・Figure 2更新案](docs/research/track_a_h468_paper_evidence_integration_20261019.md)。H4 DF10 Pを正式採用（ε=.05はNOT_REPRODUCED/PF低費用、.0005はREPRODUCED/PR低費用）。H6/H8と独立確認6条件の主表案を整理。GitHub固定10資料＋9 SOURCE、47資料・SOURCE87hashを照合し、旧manifest/CEM/ledgerを保存した新versionを作成。H4 DF12 legacy・探索n8/保存FP n32を混合せず、H8 n48は既存Mechanismと同一証拠のalias。H8詳細公開監査と全入力生成規則の同一性は未確認。Figure 2原本34ファイル不変、数値更新は別承認。新規科学計算・統計解析0、最終freeze/詳細科学レビュー未開始。文書統合後mandatory STOP。以下は各時点の履歴。

## 2026-10-18 A_COST 3系完了・保存B/D解析結果

[結果・測定表・来歴](docs/research/track_a_pr_cost_batch_completed_results_20261018.md)。H4/H6/H8 A_COST各16候補×n8、計384軌道・768軸・48代表controlが完了。保存D3件は計7,160行/120比較、B3件は限定診断まで完了（無条件平均null）。H8は元16代表を再利用、SOURCE87f1f0d5…/15workerで残112軌道を約56分、21:14:13 JSTにnative exit0・全回収。H4は1Å/legacy DF12、H6 DF19、H8 DF28で系列を分離。新Aのn8を用いた新総費用・ランキングは未実施、旧STOP/未完了予約/他研究は保全。以下は各時点の履歴。

## 2026-10-18 H8 A_COST：候補単位の並列投入・引継ぎ準備

[実装・検査・未seal草案](docs/research/track_a_h8_pr_candidate_parallel_readiness_20261018.md)。SOURCE `87f1f0d528478ad212ba98159db06eec3c78433e`。代表軌道のnative/control合格ごとに同候補の残り7軌道を投入し、元128taskのseedを保持する継続adapter。人工24＋既存承認境界18＋無効CLI2件PASS。現在のrunは保全・継続、新方式launch0、有効grant0。最終receipt受領/静止観測後のsealと切替指示は未実施。以下は履歴。

## 2026-10-18 Codex 1：A_COST3系を並列起動・保存B/D会計完了

[起動・H8 D修正・scope別SOURCE](docs/research/track_a_pr_cost_batch_execution_20261018.md)。利用者の実行/修正再実行承認によりAを6/6/10 workerで開始、稼働継続中。保存B3件の限定診断とD3件は完了（H8 Dは修正後SOURCE8ff002dd…、元PR88行再現・native exit0/reap）。初期SOURCE31ef62a1…の旧STOP/消費を保持し、他の実行snapshot・高次PF研究・共有環境は変更しない。追加科学stageへ自動進行しない。以下は履歴。

## 2026-10-18 Codex 1：A_COST・B/D 9契約の技術seal（実行未認可）

[9契約の状態・統合承認案](docs/research/track_a_pr_cost_nine_contracts_seal_20261018.md)。SOURCE `31ef62a12407010e9c1a07f9b37facedb8dff5fe`。A_COST3件・既存D_ACCOUNTING3件は技術実行候補、B3件は条件付き費用の限定診断まで準備済み。完全層別の無条件比較は証拠未結合で未確定。9 byte gate・新人工21検査・無効CLI10件PASS、旧40検査は再実行なし。全grant無効/CPU[]、実preparation復元・compile・sampling・科学再会計・B数値解析0。独立cloneを使用し、既存SOURCE/結果/他job不変。以下は履歴。

## 2026-10-18 Codex 1：S2 PR A_COST・既存B/D会計の準備

[技術報告と未結合事項](docs/research/track_a_pr_cost_integration_readiness_20261018.md)。SOURCE `33c7231fc7c8f3940106bd8fe473e3db775fbcb0`。H4/H6/H8のA各16候補と29固定fileをmetadata照合。人工40検査PASS、9個別契約は無効grant・未seal。実分子signal/sampling/compile/再会計/ランキング0。高次PF・理論資源比較・他jobを変更せずmandatory STOP。以下は各時点の履歴。

H8追加D完了：B0 p15/q8 S2は両軸446,288 RZ（誤差0.216644%）、B0+I p16/q4 S4は両軸713,432 RZ（0.409930%）。登録4軸2%未満、native exit0・最終gate PASS・mandatory STOP。新fit/費用再会計なし。詳細 `docs/research/track_a_h8_dba_20261018.md`。以下の稼働記録は履歴。

H8 DBA先行起動：SOURCE49879a9c…でDの2構成4軸を独立grant・構成間並列で実行中。fresh gate PASS、人工7件PASS。実測の最終PASSは未確認。B/A未起動、H4保留。詳細 `docs/research/track_a_h8_dba_20261018.md`。


H8 DBA：固定D2構成4軸の先行実行準備。保存予測3117248/7・4973552/7、構成間並列・軸内cosine判定後sine、人工7件PASS。B/Aは未seal。詳細 `docs/research/track_a_h8_dba_20261018.md`。

H6 DBA実行更新：A境界16構成はnative exit0・最終gate PASS・mandatory STOP。D2軸/B希少層32軸は時点付き稼働記録。人工18件PASS、旧STOP保持・新費用会計なし。詳細：`docs/research/track_a_h6_dba_20261018.md`。

H6 DBA限定並列検証：`docs/research/track_a_h6_dba_20261018.md`。D2軸・B2群32軸・A16 signal構成。新SOURCE/grant、stageごとSTOP、旧結果保全。
## 2026-10-18 H8：追加6精度の保存値検証完了

[結果・実測/model区分・来歴](docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md)。SOURCE3156b036…、利用者の準備後実行承認に基づくC専用one-shot。保存anchor1,520行を再現後、新2,280行/12比較、native exit0・最終gate PASS、wall0.832秒。通常ε=.00075で+12.77%、.000625で−0.50%（有限標本の点比較）。人工164件PASS、新compile/sampling/signal/fit0。他のA/B/D/E/Fは未実行、条件付き実行承認は保持し、各単位のSTOPを維持。以下は各時点の履歴。

## 2026-10-18 Track A：対象を絞った並列検証計画（未承認）

[具体案・再利用・依存関係](docs/research/track_a_targeted_parallel_validation_plan_20261018.md)。PR境界16構成、希少層4群、新ε6点、対照3構成、H8 T2点、H6/H8距離各1点を15の個別承認単位に分離。metadata DAG/CLI、18固定file監査、人工21＋既存135件PASS。production adapter未seal、科学実行0。完了RARE/SELECTION等は再利用しmandatory STOP。以下は各時点の履歴。

## 2026-10-18 H8：希少層境界・選択効果の保存値解析完了

[結果・来歴・限界](docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md)。RARE 40条件（32境界/8不適格）とSELECTION元4精度の探索的bootstrap2000を別grantで各一回完了、両native exit0・mandatory STOP。人工135件PASS。未知費用はnull、母平均/厳密bias/selection-adjusted coverageなし。PR sampling/compile/fit/shot再計算0、旧結果/ledger/STOPは保持。 以下は各時点の履歴。

## 2026-10-18 H8：s4・複数2次・選択後効果の技術準備（未承認）

[統合報告](docs/research/track_a_h8_rare_tail_selection_readiness_20261018.md)。完了s2 run02（b21e90b8…）と既存正本を36固定file/closure9でmetadata照合。count条件付きsampler・統計schema/人工bootstrap・mock wrapperを実装、人工73件PASS。未承認4stage/INVALID grant、production dispatch未接続。実H8 sampling/build/compile/統計/感度/shot・費用再会計/新順位は0。旧STOP/control376/376・既存成果を保全し、追加検証なしの選択肢を残してmandatory STOP。以下は各時点の履歴。

## 2026-10-18 Track A H4〜H8：横断証拠監査（対象結果2026-10-17）

[証拠整理・metadata検査](docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md)。19系列・222保存値/状態・32固定ファイル、12比較・13不足を別表に記録。実測/model/有限標本を分離し、新科学計算・再会計・ランキング・統計解析は0。H8希少次数pilotは別チャットの未完了作業として収録せず、既存SOURCE/grant/ledger/他worktreeを変更しない。新しい科学的採否・新規性判断なし。以下の記録は各時点の履歴。

## 2026-10-17 H8：主要6軸実測反映の限定再会計完了

[結果・監査](docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)。SOURCE aa984d98…、保存3800行/40比較、科学dispatch1、native exit0・全回収・最終gate PASS。旧32/PR/SE/shot不変、winner変更0、model推定は残存。追加compile/signal/sampling/fit/retry0、mandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8：主要6軸反映の限定再会計準備・公開認証待ち

[準備・来歴・次の手順](docs/research/track_a_h8_six_axis_reaccount_readiness_20261017.md)。主要6軸結果ce6cd8ea…のremote一致、45入力/元scalar・native証拠、保存3800行のshot joinを照合。SOURCE aa984d98…/closure7、人工30件PASS。利用者の一回実行承認受領済みだが、新SOURCE pushのHTTPS認証失敗により有効grant/科学dispatch0。通常/非相殺の費用・ランキング・削減率更新は未実施。手動push後に既存承認でfresh gateと一回実行へ。旧結果/STOP/他job保持。以下は過去各時点の記録。

## 2026-10-17 H8：主要3構成6軸の直接compile完了

[結果・native回収・限定監査](docs/research/track_a_h8_key_compile_completed_results_20261017.md)。RZ/軸はB0+I p15/q4 S4が668,636、p11/q1 S2が42,268、B0 p15/q4 S2が223,940（各cosine/sine同値）。全6軸のモデル誤差0.216〜0.827%で2%未満PASS、22:15:38 JST正常終了。旧完了1軸＋新5軸をSOURCE別に結合。新run wall24分23.886秒、最大単一process RSS9.643 GiB、全回収・最終gate PASS。PR削減率の再会計は未実施・別承認。以下は各時点の履歴。

## 2026-10-17 H8：並列準備へ引継ぎ・稼働中

[引継ぎ・再利用・稼働証拠](docs/research/track_a_h8_key_compile_parallel_handover_20261017.md)。SOURCE 41e466e13f32e9224bb37790cc16f0ff6621c399。検証済みQPY2本と実測cosine1軸を再利用、残る2構成を並行準備。旧所有runのみ停止・native回収・61証拠不変。人工39件PASS、compile固定順/各軸2% gate維持。

## 2026-10-17 H8：現runを保持した準備工程の並列化

[実装・検証・契約](docs/research/track_a_h8_key_compile_parallel_preparation_20261017.md)。準備を3構成並行・最大6 workerへ拡張し、wrapper生成も軸workerへ移した。人工32件PASS、SOURCE 7f8d74660b2914a1223d0ac7990dde8d4645032e。compileは固定順直列・各軸2%以上でSTOP。新方式の科学起動0・無効grant・未seal。旧runの25 immutable filesはSHA一致し、稼働継続。速度/peak RSSは未測定。以下は各時点の履歴。

## 2026-10-17 H8：主要3構成・6軸の直接compile準備（未承認）

[主要3構成・6軸の実行準備](docs/research/track_a_h8_key_compile_readiness_20261017.md)。SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e`。保存入力11件のbyte照合、人工22件PASS。準備2並列・compile直列/各軸2%未満、own-process cleanupを実装。実H8 build/control/compile/費用再会計0、無効grant・CPU未指定・未seal。実compileと後続再会計は別承認。以下は各時点の履歴。

## 2026-10-17 H8：非相殺費用比較の一回実行完了

[結果・監査](docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md)。SOURCE ac932974a0f250be0df5720e1b6dad06f9ff9535で非相殺costを一回完了、1900行/20比較、paired44再現・union6 n32 PASS。driver/launcher exit0・全回収。184基準PR削減率（倍率1）16.62/34.68/35.13/39.28/−29.09%、全条件モデル条件付き。追加計算0、mandatory STOP。 以下は各時点の履歴。

## 2026-10-17 H8：非相殺COST_COMPARISON準備・未承認

[資料入口](docs/research/track_a_h8_noncancellation_cost_readiness_20261017.md)。新cost-only adapter/runner/契約と人工26件を準備し、旧非相殺56＋Phase C49もPASS。B2最終4軸・保存shot1900行・通常Phase Cを結合。SOURCE ac932974a0f250be0df5720e1b6dad06f9ff9535、plan f171099baeb4fdd56fb40748292db1c70074479b711702f360c9d88921fc96ad、未承認・未seal・実H8費用再解析0。 以下は各時点の履歴。

## 2026-10-17 H8 Phase C：5精度費用再会計完了・モデル条件付き比較

[結果・監査](docs/research/track_a_h8_phase_c_run01_results_20261017.md)。通常signed bias/P=0、旧32・強化184・PR確認union6/n32を1900行・一回再会計。driver/launcher exit0・全回収・最終gate PASS、本体0.355秒。184基準は全条件モデル推定を含み、PR削減率（倍率1）はε=.05/.01/.005/.001/.0005で−7.14/33.09/31.31/22.96/−51.14%。全精度優位を主張せず、全格子実測・厳密bias・母平均保証なし。新signal/compile/sampling/fit/retry0、科学判断へ戻してmandatory STOP。以下は各時点の履歴。

## 2026-10-17 H8 Phase C：修正後SOURCE固定・一回実行承認・公開認証待ち

[資料入口](docs/research/track_a_h8_phase_c_execution_binding_20261017.md)。SOURCE931755c4…/plan79d624b9…へ再結合。利用者の一回実行承認受領済み、人工49件・入力12件・B2登録4軸受入PASS。公開認証確認に失敗したため有効grant/科学attempt0、未使用output/lock保持。公開後に既存承認でfresh gateと一回実行へ。旧SOURCE・旧plan・B2/他worktreeは保持。以下は各時点の履歴。

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

[固定contract](scripts/h8_phase_b2_recovery_contract.py)／[dispatcher/witness](scripts/h8_phase_b2_recovery_witness.py)／[保存STOP checker](artifacts/resource_applicability/track_a_h8_b2_recovery_run01_stop/2026-10-17/verify_saved_stop.py)。

## 2026-10-17 H8 B2：停止記録修正・2 worker並列準備、科学再実行は未承認

[修正・並列化・受入契約・資源](docs/research/track_a_h8_phase_b2_recovery_parallel_readiness_20261017.md)。B2 run01 MEMORY_PRESSURE/STOPを保持し、残るB0I p21q8 S4の2軸用に修正・並列準備を実装。left/rightとcosine/sine検証・fingerprint/QPY保存を各2 worker、compileは各軸2%判定を挟む直列1。検証済み回路をQPYへ保存し、親の大回路を解放して1軸ずつ読込。PSI raw/対象groupをcleanup前保存、独立witnessでCLI終了後もnative waitを記録、marker集計をaxis_*へ修正。31 source/265 provider/34,273 runtime/16 inputsをseal、34人工testsと2 qubitの4人工compileで指標・fingerprint一致。5物理core/112.3125 GiB案、無効grantのみ。旧687保護/124worktree・B2原45files保持、H8新compile/fit/有効grant0。旧B0/旧B2 STOP・V2 NO_GO、C未承認を維持、公開後mandatory STOP。以下は各時点の履歴。

[contract](scripts/h8_phase_b2_recovery_contract.py)／[並列control・QPY](scripts/h8_phase_b2_recovery_parallel.py)／[adapter](scripts/h8_phase_b2_recovery_adapter.py)／[dispatcher/witness](scripts/h8_phase_b2_recovery_witness.py)／[launcher](scripts/launch_h8_phase_b2_recovery.py)／[人工tests](tests/test_h8_phase_b2_recovery.py)／[provider fixture](tests/check_h8_phase_b2_recovery_synthetic_provider.py)／[bundle](artifacts/resource_applicability/track_a_h8_phase_b2_recovery_readiness/2026-10-17/README.md)。

## 2026-10-17 H8 Phase B2：契約STOP・自動再実行なし

[実測・判定・終了監査](docs/research/track_a_h8_phase_b2_run01_results_20261017.md)。H8/1Å/STO-3G/DF28/T=.8、S4の固定4軸を直列・一回実行。実compile entry3、保存判定2軸（PASS2）。SOURCE244c45e7…、model8420b9f2…を固定。STOP理由は結果報告と原terminalに保存（MEMORY_PRESSURE）。B0I cosineは中断、sine未起動、未settle予約1を保持し、後続compile・retry/refit0。driver native exit−9、launcher報告exit1（外側native wait欠測）、全owned process回収。旧684保護files/123worktree・B0原53/B1原34 rawを保持。新fit/signal/再会計0、元B0 STOP/V2 NO_GOを維持。C未承認、公開後mandatory STOP。以下は各時点の履歴。

[固定contract](scripts/h8_phase_b2_contract.py)／[native adapter](scripts/h8_phase_b2_adapter.py)／[serial launcher](scripts/launch_h8_phase_b2.py)／[人工tests](tests/test_h8_phase_b2.py)／[保存結果checker](artifacts/resource_applicability/track_a_h8_phase_b2_run01/2026-10-17/verify_saved_result.py)。

## 2026-10-17 H8 Phase B2：S4の固定4軸・直列検証を実装、実行未承認

[実装・固定条件・資源・承認範囲](docs/research/track_a_h8_phase_b2_execution_readiness_20261017.md)。B1正本83a3c4a6…/model8420b9f2…を固定し、B0 p15q1・B0I p21q8 S4の両軸を順番に検証するadapter/launcherを準備。各軸2%以上で次軸を起動せずSTOP。SOURCE244c45e7…、21 source/265 provider/34,273 runtime/15 inputをseal、新19＋既存100＝119人工testsと2 qubit S4接続11項目PASS（人工4compile、実H8 0）。3物理core・80.25 GiB、無効grant/CPU[]/command null、fresh gate未発行。旧683保護filesと別チャットの許可済み1worktree更新を区別、更新後122worktree不変。B1・元B0 STOP・V2 NO_GOを保持、新fit/再会計/有効grant0、公開後mandatory STOP。以下は各時点の履歴。

[contract](scripts/h8_phase_b2_contract.py)／[runtime](scripts/h8_phase_b2_runtime.py)／[adapter](scripts/h8_phase_b2_adapter.py)／[driver](scripts/run_h8_phase_b2.py)／[launcher](scripts/launch_h8_phase_b2.py)／[prepare](scripts/prepare_h8_phase_b2.py)／[tests](tests/test_h8_phase_b2.py)／[人工provider接続](tests/check_h8_phase_b2_synthetic_provider.py)／[bundle](artifacts/resource_applicability/track_a_h8_phase_b2_readiness/2026-10-17/README.md)。

## 2026-10-17 H8 Phase B1：全4層PASS・費用model固定・mandatory STOP

[結果・model・検証・保全](docs/research/track_a_h8_phase_b1_run01_results_20261017.md)。H8/1Å/STO-3G/DF28/T=.8。新B0保存8軸と既存B0I16軸で4独立S2 fit。B0のp21/q8独立holdoutは両軸0%、B0Iは各8 training anchorの残差0%。係数は各層(a,b,d)=(1592,1314,25271/7)。SOURCE6fb8a0a7…、model8420b9f2…、計算本体wall.092秒/native driver・launcher exit0/全回収、SOURCE19/runtime34,273/input82照合。35人工tests PASS、旧681保護/119worktree不変。元B0 run02 STOPは保持、新compile/signal/再会計0、S4は未実測仮説、B2/C別承認。以下は各時点の履歴。

[fit/STOP](src/trotterlib/h8_phase_b1_fit.py)／[実行入口](scripts/h8_phase_b1_execution.py)／[人工tests](tests/test_h8_phase_b1_execution.py)／[evidence](artifacts/resource_applicability/track_a_h8_phase_b1_run01/2026-10-17/README.md)。

## 2026-10-17 H8 B0：終了監査修正・保存8軸の読取専用audit完了

[技術判定・修正・証拠](docs/research/track_a_h8_b0_saved_anchor_audit_20261017.md)。H8/1Å/STO-3G/DF28/T=.8、p14/21・q1/8のB0 S2保存8軸を照合し、欠測・hash/lineage不一致なし。再compileを要求する技術的欠損はなく、B1入力候補として構造的に再利用可。元run02のPROCESS_THREADS/契約STOP・過去の違反PID未記録を保持。管理Gitはthread1/optional lock無効、監視は違反identity/exe/thread数をcleanup前保存。新14＋既存83の97人工testsを固定PythonでもPASS。旧681保護/118worktree不変、科学compile/fit/新grant0、B1別承認・公開後mandatory STOP。以下は各時点の履歴。

[audit runner](scripts/audit_h8_b0_saved_anchors.py)／[metadata/Git helper](src/trotterlib/h8_b0_saved_anchor_audit.py)／[人工tests](tests/test_h8_b0_saved_anchor_audit.py)／[evidence](artifacts/resource_applicability/track_a_h8_b0_saved_anchor_audit/2026-10-17/README.md)。

## 2026-10-16 H8 B0 run02：8軸を保存、終了時のthread監視でSTOP

[保存測定・停止箇所・保全](docs/research/track_a_h8_b0_anchor_run02_stop_20261016.md)。H8/1Å/STO-3G/DF28/T=.8、B0 S2のp14q1/p21q1/p14q8/p21q8各cosine/sineを4 worker・2 waveで一回取得。8 worker全exit0/reap、TRAIN6/HOLDOUT2保存。約80.76分後、終了時のworktree確認中にPROCESS_THREADSでSTOPし、driver exit−9・launcher exit1、全回収。8軸の保存と取得契約の正常完了を区別し、fit/残差/再会計/自動retry0、旧B1 STOP・V2 NO_GOを維持。以下は各時点の履歴。

## 2026-10-16 H8 B0：CPU選択・起動前STOP記録を修正、再起動なし

[修正・診断・条件維持・人工検証](docs/research/track_a_h8_b0_cpu_gate_fix_20261016.md)。H8 B0 run01のCPU gateを修正。新SOURCE db559a51…、承認前の3固定window候補probe、全選択coreのcounter/負荷率/自身のCPU使用、科学起動前STOP receipt、試行域の再利用拒否を追加。raw5%・6物理core/176.25 GiB・4構成8軸/4 worker/2 waveは不変。新11＋既存27 tests PASS、source15/provider265/runtime34,273/input8照合。run01原因は遡及断定せず、旧677保護/116worktreeを維持。新compile/fit/有効grant/取得再起動0、公開後mandatory STOP。以下は各時点の履歴。

[CPU提案probe](scripts/probe_h8_b0_anchor_cpus.py)／[修正tests](tests/test_h8_b0_cpu_gate.py)／[公開bundle/checker](artifacts/resource_applicability/track_a_h8_b0_anchor_cpu_gate_fix/2026-10-16/README.md)。

## 2026-10-16 H8 B0 run01：fresh CPU gateで科学起動前STOP

[停止点・証拠・未実測・保全](docs/research/track_a_h8_b0_anchor_run01_stop_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。SOURCE572a337d…の4構成8軸取得を利用者承認で一回起動したが、fresh CPU idle checkがPHYSICAL_CORE_BUSYで拒否。driver/worker・fresh receipt・output/control/lock・科学予約・compileすべて0。自動retry/CPU再割当0。失敗coreの詳細sampleとlauncher exit codeは未捕捉のまま保持。旧676保護/115worktree不変、SOURCE14/provider265/runtime34,273/input8照合PASS。旧B1 STOP/V2 NO_GO・B1/B2/C別承認を維持し、公開後mandatory STOP。以下は各時点の履歴。

[STOP evidence/checker](artifacts/resource_applicability/track_a_h8_b0_anchor_run01_stop/2026-10-16/README.md)。実行sourceコードと旧結果を変更していない。

## 2026-10-16 H8 B0：8軸取得の実装・seal完了、実行未承認

[実装・gate・人工検証・承認範囲](docs/research/track_a_h8_b0_anchor_execution_readiness_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。固定4構成8軸・4 worker/2 waveのadapter/launcher、SOURCE/input/profile・資源監視/回収を実装。SOURCE 572a337d…、14 own/265 provider/34,273 runtime/8 inputのbytes照合PASS。17新unittest＋旧計画10、2 qubit fixture8項目PASS（人工4 compile・実H8 0）。6物理core/176.25 GiBとfresh gate必須、無効grant・CPU[]。旧B1 STOP/V2 NO_GO・旧673保護/114worktreeを維持。科学起動/fit/残差/再会計0、公開後mandatory STOP。以下は各時点の履歴。

[contract](scripts/h8_b0_anchor_contract.py)／[runtime](scripts/h8_b0_anchor_runtime.py)／[adapter](scripts/h8_b0_anchor_adapter.py)／[driver](scripts/run_h8_b0_anchors.py)／[launcher](scripts/launch_h8_b0_anchors.py)／[prepare](scripts/prepare_h8_b0_anchors.py)／[tests](tests/test_h8_b0_anchor_execution.py)／[provider fixture](tests/check_h8_b0_anchor_synthetic_provider.py)／[bundle](artifacts/resource_applicability/track_a_h8_b0_anchor_execution_readiness/2026-10-16/README.md)。

## 2026-10-16 H8 B0：新anchor最小取得計画・未承認

[候補・来歴契約・並列資源・STOP](docs/research/track_a_h8_b0_anchor_acquisition_plan_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。新B0 S2はTRAIN p14q1/p21q1/p14q8とHOLDOUT p21q8の各cosine/sine、最小4構成8軸を提案。設計座標のみのdet−49、実fit/compile/再会計0。4 worker/thread1・2 wave、後続B1で各軸holdout誤差2%以上STOP。265 provider/7入力照合・10人工tests PASS。旧B1 STOP・B0I16軸再利用・V2 NO_GOを維持。実adapter/launcher未seal、無効grant案のみ。旧673保護/113worktreeを保全し公開後mandatory STOP。以下は各時点の履歴。

[計画/receipt guard](src/trotterlib/h8_b0_anchor_plan.py)／[checker](scripts/check_h8_b0_anchor_plan.py)／[人工tests](tests/test_h8_b0_anchor_plan.py)／[公開bundle](artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/README.md)。

## 2026-10-16 H8 Phase B1：anchor照合完了・来歴不足でSTOP

[照合・契約・資源・STOP判定](docs/research/track_a_h8_phase_b1_anchor_readiness_20261016.md)。H8/1Å/STO-3G/DF28/T=.8。保存32実測軸を照合し、B0I16軸はprofile/preparationへ結合、旧B0の16軸はruntime全体・cost用basis未結合で全B1 STOP。H6係数移送・実rank/fit/compile/再会計0。82既存証拠、8人工tests PASS、無効契約/grant案とB2事前選択4軸（予測null）を公開。旧673保護/112worktreeを保全、V2 NO_GO、公開後mandatory STOP。以下は各時点の履歴。

[metadata guard](src/trotterlib/h8_phase_b1_contract.py)／[checker](scripts/check_h8_phase_b1_readiness.py)／[人工tests](tests/test_h8_phase_b1_readiness.py)／[bundle](artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16/README.md)。

## 2026-10-16 H8 Phase A run01：152候補完了・mandatory STOP

[全152 signal gzip](artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/H8_signal_complete_v1.json.gz)／[保存結果checker](artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/verify_saved_result.py)。

[結果・適格性表・資源/来歴/保全監査](docs/research/track_a_h8_phase_a_run01_results_20261016.md)。SOURCE fc463d9e…、H8/1Å/STO-3G/DF28/T=.8/q1/2/4/8。追加S2 96＋S4 56候補、B0/B0Iを評価。wall72.293秒・35 worker全exit0/reap・peak RSS6.846 GiB、132候補が少なくとも1条件で経験的適格。中断/科学retry0、費用fit/compile/再会計0、PR費用優位の新判定なし。原worker SHA欄1欠測は公開時の別SHAと区別。旧671保護/111他worktree不変、V2 NO_GO・B1/B2/C別承認、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8 Phase A：実装・人工検証完了、科学実行は未承認

[plan](scripts/h8_strengthened_contract.py)／[export復元](scripts/h8_strengthened_export.py)／[signal](scripts/h8_strengthened_signal.py)／[runtime gate](scripts/h8_strengthened_runtime.py)／[runner](scripts/run_h8_strengthened_phase_a.py)／[launcher](scripts/launch_h8_strengthened_phase_a.py)／[未承認準備](scripts/prepare_h8_strengthened_phase_a.py)／[人工tests](tests/test_h8_strengthened_phase_a.py)／[2 qubit fixture](tests/check_h8_strengthened_synthetic_provider.py)。

[H8 adapter・保存操作列復元・並列制御・契約](docs/research/track_a_h8_phase_a_implementation_readiness_20261016.md)。追加152候補は不変。22 primitive tasks/1,650 probesと13 prefix tasks、最大16 worker。28人工tests＋2 qubit fixture8項目PASS、provider265/runtime34,273/input11を結合。実H8復元・signal・数値再解析・compile・新有効grant0、別途明示承認とfresh gateが必要。V2 NO_GO・旧grant/ledger・V1/H5/H7を保持し、公開後mandatory STOP。以下は各時点の履歴。

## 2026-10-16 H8対照強化の準備・V2現行残差設計のNO_GO終了

[metadata checker](scripts/check_track_a_h8_strengthened_plan.py)／[人工guard tests](tests/test_track_a_h8_strengthened_plan.py)／[未承認計画](artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16/execution_plan_UNAPPROVED_v1.json)。

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

[track_a_v1_phase_a_parallel_readiness_20261016.md](docs/research/track_a_v1_phase_a_parallel_readiness_20261016.md)。V1は最大16worker・44限定人工tests PASS、SOURCE/input/profile静的照合一致。新科学計算・新grant0、未承認・mandatory STOP。V2は指定commit7e416705…のH6/H8計832理論行を読み取り専用照合し、重複生成・計算を保留。以下は各時点の履歴。

# プロジェクト案内

## 2026-10-16 Track A V1 実行準備

[報告・実装・人工テスト・証跡](docs/research/track_a_v1_h6_deterministic_readiness_20261016.md)。科学計算未実施、未認可draft、mandatory STOP。

## 2026-10-16 Native DFモデル：H6/H8 metadata接続監査

[主要報告](docs/research/track_a_native_df_h6_h8_metadata_binding_20261016.md)、[schema adapter](src/trotterlib/native_df_metadata_binding.py)、[人工runner](scripts/check_native_df_metadata_binding.py)、[tests](tests/test_native_df_metadata_binding.py)、[軽量証跡](artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/README.md)。
代表6trajectoryのbasis記述/hash・event identityを照合、20人工tests PASS。PFはA相当no-reuse、RTE tailはB相当reuseを区別。full PF primitive/registry・全component lawは未取得で、全実H6/H8費用への無条件適用は不可。新理論費用・期待値・数値再会計・科学計算・compile・grant・原稿改訂・subagents0、公開後mandatory STOP。

## 2026-10-16 Native DFモデル：Qiskit 1.3.0接続検証

[主要資料](docs/research/track_a_native_df_qiskit_connection_validation_20261016.md)、[接続runner](scripts/check_native_df_qiskit_connection.py)、[参照loader](scripts/native_df_qiskit_validation_support.py)、[tests](tests/test_native_df_qiskit_connection.py)、[証跡](artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/README.md)。
実分子評価・保存H6/H8再解析・sampling・transpile・server接続・新grant・原稿改訂0。公開後mandatory STOP。

## 2026-10-16 Native DF理論ゲート数モデル（人工検証のみ）

[数式・SOURCE・制限](docs/research/track_a_native_df_theoretical_gate_model_20261016.md)、[カウンタ](src/trotterlib/native_df_gate_model.py)、[人工runner](scripts/check_native_df_gate_model_synthetic.py)、[tests](tests/test_native_df_gate_model.py)、[証跡](artifacts/resource_applicability/track_a_native_df_gate_model/2026-10-16/README.md)。
新しい実H6/H8数値評価・sampling・compile・server操作・grant発行は0。既存結果・科学SOURCEは保持し、公開後mandatory STOP。

## 2026-10-15 H6 native B0+I全12構成：結果

[結果・全表・終了監査](docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)。H6 native B0+I全12構成の必要取得が完了。新12軸＋旧6軸再利用、全条件不適格6軸は未測定。SOURCEac6d28e3…、P=0/RZのPR確認union8/n32点費用は登録B0/B0I/full S2/S4最良より5精度で29.35〜56.14%低い。経験的u・点選択・matched B0未取得の限界保持。wall2326.546秒、native wait0/12worker全exit0・reap、保存表2607条件PASS。新科学計算/新grant/原稿改訂/共有環境変更0、科学採否はGPTへ、mandatory STOP。 以下は各時点の履歴。

## 2026-10-15 H6 optimized run04再実行認可

[再実行・binding・監視入口](docs/research/track_a_h6_family_optimized_run04_20261015.md)。SOURCEac6d28e3…、親重複走査削減/cleanup修正・61人工PASSを採用し、2worker/各thread1で新runへ。科学/compiler/資源/PSI条件不変、run03予約6/unknown保持。主担当照合/subagents0、直前fresh gate後だけ起動。以下は過去の各時点の記録。

## 2026-10-15 H6 fingerprint並列化・cleanup修正

[修正・人工61件・再実行案](docs/research/track_a_h6_family_fingerprint_parallel_fix_20261015.md)。親の重複走査をworker前後SHA証跡へ集約し、停止後のsignal割込によるreap中断を修正。旧証拠保持、2worker案、資源/PSI条件不変、今回の再起動0。以下は過去の各時点の記録。

## 2026-10-15 H6 font-cache Timer起動STOP修正

[原因・修正・binding](docs/research/track_a_h6_family_font_seed_fix_20261015.md)。SOURCE99abc20a…、空MPLCONFIGDIRのMatplotlib警告TimerでThreads2になる経路を特定。表示font metadataだけをbyte/hash固定してown-runへseedし、thread1/5s/科学条件を維持。人工49件＋seed済みprovider import診断PASS。前2回の科学予約/compile0を保全し、fresh gate後に修正one-shotを起動する。主担当のみ/subagents0。以下は各時点の履歴。

## 2026-10-15 H6 full-family startup STOP修正

[停止証拠・修正・再結合](docs/research/track_a_h6_family_startup_cache_fix_20261015.md)。SOURCE84d236cf…、初回科学予約0/native wait2を保全。Numba cacheをown-run tempへ閉じ、thread/affinityの診断とimport phaseを分離。元停止の内訳は未確定、46人工tests PASS、主担当照合/subagents0。既存科学/provider/compiler/最大18compileと資源は維持し、新output/grant・fresh gateで起動確認する。以下は各時点の履歴。

## 2026-10-15 H6 B0+I 全12構成の実行承認

[実行承認・最終binding](docs/research/track_a_h6_native_b0i_full_family_launch_20261015.md)。利用者の指示で新9構成のscreenと適格な最大18軸取得を承認、旧6軸を原SOURCEで再利用。SOURCE731826b3…は不変、主担当によるbyte/profile/input/carry照合PASS、subagents0。driver63/worker62,61,58,53,52,50、max6並列、AS32/RSS9GiB、wallnull・累積CPU24h。起動直前fresh gate合格時だけone-shot実行し、完了/STOP後に証拠を保持、失敗時は既定範囲内で修正。以下は各時点の履歴。

## 2026-10-15 H6 native B0+I 全12構成の実行準備

[全12構成・旧6軸再利用・新規最大18軸の実装と契約案](docs/research/track_a_h6_native_b0i_full_family_readiness_20261015.md)。SOURCE731826b3…、44人工tests PASS、別担当の限定技術監査PASS_WITH_LIMITS。旧3構成は元SOURCE2e733d8a…の実測としてREUSE_WITH_DECLARED_LIMITS。新9の全10条件screen後、全selected correctness合格時のみ最大6並列compileへ進む経路を準備した。旧4h実績を残し、新wall-null/CPU別予算・資源は未承認案。CPU[]/false/未seal/commandnull、新実H6数値・native build/probe/compile・grant・venv/他job変更0、mandatory STOP。以下は各時点の履歴。

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

## 2026-10-15 Track A D0 one-shot保存値再会計・完了STOP

[結果・完全一致・6metric/P・終了監査](docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)。SOURCE fa75c799…、旧precision1,088/費用2,659件exact PASS後、5精度×factor1/10×6metric/Pを一度再会計。ε=.0005/P=0のPR/full-PF RZ点比はH4固定panel .429664、H6確認union8 .496333。未確認CX_depth/P候補と標本・政策差を明記。wall2.223秒/peak54.14MiB、exit0、SOURCE/入力/venv/他job/H8変更・新signal/trajectory/compile0、科学判断未採択・mandatory STOP。以下は各時点の履歴。

## 2026-10-14 Track A D0：保存scalar再会計の技術準備

[入力・code・人工検査・実行契約の入口](docs/research/track_a_d0_saved_value_reanalysis_preflight_20261014.md)。H4 native fixed16／H6 exploration92／確認union8を分離し、ε=.0005・6指標・P感度を準備した。33限定人工tests PASS、SOURCE fa75c7997e135da7b7a0c4e23949e47252bd7a2e。実データの新再会計・新科学計算・H8変更・新grant0、別実行承認待ちSTOP。以下は各時点の履歴。

## 2026-10-14 Track A追加検証：技術調査・設計草案

[設計入口](docs/research/track_a_resource_evaluation_expansion_design.md)でH4 geometry/native条件差、native B0+I不足、奇数鎖の旧charge+1/triplet既定とNa≠Nb primitive制限、精度/T/metric/Pの再利用範囲を整理した。
保存headroom診断と静的pilot予算のみ。ROHF/UHF・科学比較契約は未採択、新科学計算/H8変更/新grant/原稿改訂0、DESIGN_DRAFT_NOT_AUTHORIZATION、公開後STOP。以下は各時点の履歴。

## 2026-10-14 Track A 正式共通研究ノート v1

server2298c4c…とlocal8e843989…を利用者の指示で正式統合。H4全面結果後GPT原reviewをexact公開し、旧M1/PM/AX・H4/H6/H8・失敗・未解決を固定証拠へ接続した。
[正式入口](docs/research/track_a_shared_research_status.md)、[sync policy](docs/research/track_a_environment_sync_policy.md)、[科学review判断](docs/research/track_a_scientific_review_decisions.md)。
情報の正本化であり科学source/raw/実行環境の統合や新計算認可ではない。原稿改訂/新科学計算/H8操作0、公開後MANDATORY_STOP。以下は当時の履歴。

## 2026-10-14 Track A ローカル照合・統合候補（未採択）

server固定2298c4c…とローカル保存履歴を照合。265資料実体/93内容の所在・hash・scopeを整理し、M1/PM/AXと旧H6 STOP/export履歴を候補へ追加した。
[統合候補](docs/research/track_a_shared_research_status_candidate.md)、[差分](docs/research/track_a_environment_sync_differences.md)、[local inventory](docs/research/track_a_local_inventory.md)、[source/result参照](docs/research/track_a_local_sync_reference_index.md)。
server入口・科学原本・旧STOP/grantは不変。正式共通正本への採択なし、新科学計算/compile/sampling/H8変更/新grant0。公開後mandatory STOP。以下は各時点の履歴。

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

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](docs/research/track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](docs/research/track_a_h4_worker_bootstrap_run02.md)と[完了報告](artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](docs/research/track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](docs/research/track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](docs/research/track_a_h4_geometry_resource_observer_fix.md)と[source bundle](artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](docs/research/track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](docs/research/track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](docs/research/track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

## Track A 原稿保留とH4 geometry拡張の準備

利用者の新しい指示により原稿作成・投稿先検討はいったん保留。
[H4 geometryと要求精度の全候補map設計案](docs/research/track_a_geometry_precision_extension_proposal_v0.md)と
[別JSON](docs/research/track_a_geometry_precision_extension_proposal_v0.json)を現在の準備入口とする。
218 template、1点12,464 wrapper、追加6点74,784 wrapperの案。距離・host・worker・新source/authorizationは未固定。
サーバーCPUを第一候補に同一synthetic fixtureで比較する設計で、性能確認・本計算はまだ実行していない。
[サーバー側への準備指示](docs/research/gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)は既存server環境を優先し、
環境確認・CPU benchmark・契約草案までを依頼する。旧local version一致を強制せず、compiler差を別layerにする。
原稿v0.1/v0.2・図・旧結果/STOPは保存し、科学計算STOP。H6/H8・追加96・Track B統合は別判断。
以下は各milestone当時の履歴であり、旧認可を新条件へ使い回さない。

## Track A 通し原稿v0.1・主図完成（2026-10-05）

最新は[原稿・補足・監査の入口](docs/manuscripts/README.md)。保存証拠だけから4主図と補足図1、
日本語通し原稿、再現性付録、claim audit、投稿可能性review依頼を作成した。
[表示script](scripts/resource_applicability/build_track_a_manuscript_figures.py)と
[bundle照合script](scripts/resource_applicability/verify_track_a_manuscript_bundle.py)、専用21 synthetic/local testsで追跡する。
9入力のcommit blob/hashと22生成assetを照合。新科学計算0、旧result/status/manifest・Track B不変。
原稿作成時点はlocal uncommitted。利用者の指示で原稿bundleのみをcommit/pushする。
投稿可判定は未実施。次は完成原稿review、科学計算STOP。以下は各milestone当時の履歴。

## Track A PM-2後review採用 原稿化へ

現在の入口は[主張・証拠対応表](docs/research/track_a_post_pm2_claim_evidence_map.md)と
[原稿・主要4図の設計](docs/research/track_a_post_pm2_manuscript_design.md)。
PM-2結果commit 5a1adffad780f0ec4272f5e8bb94713f9ff0f2bcを根拠に、固定DF・二次PF・
canonical finite-RTE・所定shot規則の有限signal事例研究として閉じる方針を採用した。
近接discard反証、B2内の精度依存、元5構成のtransferを中心に、既存結果・POSTHOC・解釈を分ける。
今回は対応表と図表設計まで。図生成・通し原稿は未実施、新科学計算0、Track B変更0。
旧result/status/manifestは不変で、科学計算のmandatory STOPを維持する。以下は各stage当時の履歴である。

## Track A PM-2保存値解析完了 研究方針review待ちSTOP

[PM-2結果と照合](docs/pr2_pm2_precision_resource_result_validation.md)から
`artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/`の結果・監査へ辿れる。
source `324435d`不変、ε=0.05の223候補を再現し、302点・67,346行を照合した。
developmentのprimary点最小は精度要求によりB2 q4/q2/q1へ変わる。M2元5構成は別domainで、厳密winnerは認定しない。
pre/post62 local tests passed、新しい科学計算0。利用者指示でresult commitへ収録するPOSTHOC local evidence。
`PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`、mandatory STOP、研究判断null、次段未認可。
以下の準備・source固定・旧stageの記述はmilestone当時の履歴である。

## Track A PM-2解析source固定 本解析は明示指示待ち

[保存値解析実装](docs/research/pr2_pm2_precision_analysis_implementation.md)のsourceを
`324435d77b6642dbd44e8d1f178420daf62e77ed`で固定した。stdlib-only解析module、future runner、専用synthetic testsの一組で読む。
`PM2_ANALYSIS_SOURCE_FROZEN_AWAITING_USER_LAUNCH`、62 local synthetic tests passed、fail/skip0。
実データのreference gate・精度走査・順位・P envelope、新signal/sampling/compileは未実行。
source/準備blobを照合した[固定後監査](artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/source_freeze_v1.json)を参照する。
以下の準備・旧stage記述は各milestone当時の履歴として保持する。

## Track A PM-2契約準備完了 解析は未認可

[PM-2精度と資源境界契約](docs/research/pr2_pm2_precision_resource_contract_v1.md)を固定した。
`src/trottertracks/resource_applicability/pm2_precision_contract.py`、
`scripts/resource_applicability/run_pr2_pm2_precision_contract.py / run_pr2_pm2_preparation_tests.py`、
`tests/tracks/resource_applicability/test_pm2_precision_contract.py`、
`artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/`を一組として読む。
全218 development候補とM2元5構成を別集合にし、POSTHOC、ε=0.005〜0.1、α_axis=0.025、軸別費用、共通P>=0を固定する。
保存JSON4件のidentity・field coverageだけを検査した。precision sweep/ranking/P envelopeは0。
`PM2_PRECISION_CONTRACT_FROZEN_ANALYSIS_NOT_AUTHORIZED`でSTOP。以下はmilestone当時の履歴として保持する。

## Track A PM-1実行完了：研究方針review待ちSTOP

最終承認・利用者の明示launch後、[PM-1結果照合](docs/pr2_pm1_discard_result_validation.md)を完了した。
H4 linear 1.00 Å、STO-3G、DF rank12、8 qubits、T=0.8、B0 rank4/5 × q=1/2/4/8、
delta=0.8/0.4/0.2/0.1、r=K=0の8構成が全件accuracy適格。8 signals/16 wrappers、CPU1、wall141.614秒。
最小の新B0 rank5・q1はG_RZ=229,718,060、保存B2 rank3・q1・r4・K2のpoint値の1.75659倍、
旧B0 rank6・q1より9.34%低い。これは固定development比較でありgeneral method optimumではない。
source134・plan・authorization・runner manifestを照合し、pre/post201 local tests passed、fail/skip0。
random/held-out/GPU/quantum shots/retry0。結果・監査は
`artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/`のlocal evidenceを利用者指示でresult commitへ収録する。
**statusはPM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW。mandatory STOP、研究判断null、次段未認可。**
旧draft/finalization/準備の記述はmilestone当時の履歴として保存し、現在地はこの節を優先する。

最終更新：2026-10-05

## Track A post-M2レビュー資料

Track Aの最新解析は[PM-0 POSTHOC証拠帰属](docs/research/pr2_post_m2_evidence_attribution.md)。
`src/trottertracks/resource_applicability/pm0_evidence_attribution.py`、
`scripts/resource_applicability/run_pr2_post_m2_evidence_attribution.py`、
`tests/tracks/resource_applicability/test_pm0_evidence_attribution.py`、
`artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/`を一組として読む。
保存JSON/sourceだけを再集計し、source-bound科学コードや旧M1/M2証拠は変更しない。PM-1以降は未認可。

Track Aの次段準備は[PM-1近接discard契約](docs/research/pr2_pm1_nearby_discard_contract_v1.md)。
`src/trottertracks/resource_applicability/pm1_discard_contract.py / pm1_discard_execution.py`、
`scripts/resource_applicability/run_pr2_pm1_discard_contract.py / run_pr2_pm1_discard.py`、
`tests/tracks/resource_applicability/test_pm1_discard.py`、
`artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/`を一組とする。
sourceはcommit `fd7552e`、sealed planと134 source blobの照合まで。本計算・authorizationは未認可。
[GPTへの準備bundleレビュー依頼](docs/research/pr2_pm1_preparation_external_review_request_fd7552e.md)に
固定identity、読む資料、停止条件をまとめた。利用者の別指示によりレビュー資料のみcommit/pushする。

このファイルは、人またはGPTがリポジトリ全体を読むときの入口である。研究内容の正本、
実装、検証コード、結果データ、発表資料を区別し、古い研究経路を現在の結論として読まない
ための案内をまとめる。

PR-2の最新状態は、旧S0 STOPとdevelopment-only S2結果を保持し、matched-accuracy resource-map研究の
compile-free M1-Aを`SELECTION_LIMITED`で停止した後、`PROCEED_BOUNDED_COMPILE_EXPANSION`として
M1-B1を実行・検証した段階である。旧16-cell selectorは「proxyではactual frontierを完全に保持できなかった」
監査結果として保存する。M1-Aでaccuracy適格だったB2/B3 194 cellを32 trajectory・Re/Im二軸、B0/B1 16 cellを
二軸で測る12,448-wrapper mapは全件完了した。actual six-metric ParetoはB2 rank 3、q=1の2件で、
primary RZ point minimumは`B2-rank3-q1-r4-K2`、状態準備cost感度のlower envelopeも全てB2だった。
研究判断は`CONTINUE_RESOURCE_STUDY`。その後、development actual Pareto 2件とB0/B1/B3代表を合わせた
5構成、primary RZ、6指標Pareto、10% materiality、重大cost underestimate、4 terminal statusを
M2 held-out transfer契約へzero-compute固定した。入口は
`docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`と
`docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`と
`docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`と
`docs/pr2_matched_accuracy_m1_b1_result_validation.md`と
`docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`。追加96 trajectory、held-out H4 1.30 Å、
transfer、S3は当時未実行・未承認だった。M2外部reviewの修正要求は
`docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md`へ反映し、
usable B2だけをsupportとratioに使うv2 source/planをcommit固定した。science実装の入口は
`docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md`である。
actual source `2978e2f`、別authorization `90a9f24`、最終review承認と利用者指示を経て、M2を一度だけ実行した。
最新は[結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)：H4 1.30 Å、STO-3G、DF rank12、
`T=0.8`の固定5構成、196 wrapperが完了し、B2二件がParetoに残って`TRANSFER_SUPPORTED`。
result commitへ収録するlocal execution evidenceで、固定構成のtransfer以外へ一般化しない。現在はmandatory STOP、研究方針全面review待ち。
追加96、held-out再探索、S3、別geometry/分子へ進まない。

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
5. [`docs/research/prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)
   事前検証カタログのうち、実施したID・work packageと文書、artifact、runner、testの対応表。
6. 数値を使う場合だけ、対応する `docs/*_validation.md` と `artifacts/` のJSONを確認する。

`docs/research/研究ノート/` は意思決定の時系列記録であり、現在の仕様ではない。過去の暫定値が
後日のノートや研究概要で変更されている場合は、最新の研究概要と規範文書を優先する。

## 現在の研究段階

2026-09-25現在、A0後のP-B/P-C/P-Aテーマ選定と、各候補の停止点まで完了した。P-Bは現H4 gridで
実用的signal差がなく停止した。P-A v1はblind transferを通過したが、形式化とforced-support
Taylor-order-2検証でinterval分割・一区間baselineとの差が全30 taskで0となり、interval claimを停止した。
P-Cは0.80--1.20 Åの局所pilotを通過したが、事前登録した0.70--1.60 Å tracking・breakdown検証では
追跡prefixが独立prefixと全点同一で、1.40/1.60 Åのcoefficient予測誤差が33.653%/123.245%となった。
pair予測、continuity診断、mechanism discriminationも固定gateを通らず、
`stop_pc_current_h4_family_as_primary`となった。

従って、A/B/Cに確認済みの主研究候補はない。P-Dのpilotと現実化gateは通過したが、S0/S1の
公平再最適化ではB1b/B2/B4が全scopeで同じnew fourth、`delta=0.2,R=16`を選び、B2のB4 regretは0、
限定K4でも選択は変わらなかった。finite補正が選択を変えるCase C/Dの証拠は得られていない。
outer-stageだけのB1aは一段延長後も`m_D=128`上限へ達したため、一次分類Case Bに
`undetermined_boundary`を付け、GO判定を出さず停止した。P-Dの研究方針とbaseline設計を再検討し、
H12、長RPE、compiled総costへはまだ広げない。
その後、固定S1 artifactの事後再解析で主baselineのB1b/B2/B4一致を確認し、P-D S2を停止した。
R3も一般multi-fidelity法との差分を固定できず`STOP_R3_NO_METHOD_DELTA`となった。2026-09-26には、
有限RTE打切り誤差をHadamard複素信号の位相方向と半径方向へ分ける新候補についてFR-0を完了し、
事前登録済みFR-1を固定2×2 toyで実行した。補正後演算子$A_{\rm corr}$と実際の平均
$A_{\rm mean}=A_{\rm corr}/\mathcal B$を区別した境界は495適用recordで違反0、semantic・負時間・K4・
非対称配置も通過した。一方、利用可能な$\underline\rho=0.8$でのG2は不通過で、真の$\rho$を使う
場合だけ改善した。その後のFR-R1aでは正scalar処理後の片側差が0、FR-R1bでは同情報のstrict gainが
8件あったが固定予算の片側認証差が0となり、現行判断は`MECHANISM_ONLY_NO_PRACTICAL_GO`である。
FR-R2は開始せず、2026-09-27に研究を候補探索から理論成果の完成へ切り替えた。C1/C2を中核、C3を
条件付き応用とし、先行研究との定理単位の照合と証明義務T1--T4を終えるまで新しい数値計算を行わない。

それ以前の中心課題は、DF Hamiltonianを決定論部分とランダム部分へ分けたpartial-$S_2$について、有限RTE、
RPEの信号半径・測定回数、1 shot当たりのコンパイル後回路コストを接続することだった。PF係数、有限RTE、
ランダム回路コスト、短いRPE段の接続は限定条件で検証済みだが、最終総cost評価には達していない。
既存設定`CA/10`を暫定目標にすると$q_{\max}=32768$が必要で、従来の固定
$\delta=0.1,r=4,K=2$は長roundへ単純外挿できないことを確認した。その後、H4の
実行済み$\delta$窓でround別$(r_m,K_m)$を再探索し、$\delta=0.01,0.0125,0.02$に
行列検査を通るscheduleを構成した。さらに、短時間幅0.02--0.000390625で局所回路指標が
変わらないことと、イベント列長8、16、32への移送を検査し、$r\leq16$では1--3イベント、
$r=32$では4イベント補正を用いる中央RTEブロックcost proxyを接続した。この限定proxyでは
$\delta=0.02$が全6指標で最小となった。

2026-09-21に、広い検証backlogを一括実行せず、研究方向を選ぶGate S1を先に置く方針を採用し、
最初の`WP00 -> WP02 -> WP01-S`を実行した。H4 rank-12固定snapshotへPF入力を結び直し、
CA/CA/10/CA/100のround horizonを監査した。CA/10は既存3 schedule・56点行列検査を再利用できるが、
CA/100は既存3つの$\delta$がすべて経験的PF予算を超える。CA/10の条件付き比較では
$L_D=0$を拡張した解析的成分作用数proxyでscreen outし、$L_D=12$の点推定は$L_D=3$より
21.0%低かった。
ただし保守的な長$q$移送scenarioでは区間が重なるため、結論は未決定である。続くWP04では、
両候補に公平な$\beta$・$\alpha$再配分を与えると決定論endpointの点推定差は4.96%へ縮み、
5%・25%区間がともに重なることを確認した。共通の主要因は$\beta$、次いで$\alpha$再配分で、
$L_D=3$のcompiled-cost整合round schedule単独の利得は固定schedule比1.93%だった。WP03では
$C_D$、論文D6、支配固有位相係数だけを差し替えた18条件の選択が全て
$L_D=12,\delta=0.02$で変わらず、係数選択も区間重なりを解消しなかった。Gate S1では
これらを統合し、区間判定を「同点」でなく`undetermined`、最大の残存不確かさをfull controlled
interrogationの回路scope・構造とした。T4とT7を主軸、T1/T2/T5/T6を限定継続、T3を保留とし、
次の一件をWP06-aとした。WP06-aではsupport限定Gaussian completionが単一Z/ZZ eventのRZを
39--62%減らした一方、異なるsupportの長さ3列ではfull basis共有より15.0%増え、事前の5% triggerが
発火した。control・relative-phaseの現行方針は同値性を通過した。WP06-bでは独立trainingから、同一
元basisのsingleton runだけsupport限定へ置換するpolicyを固定した。未使用列長3, 6でRZ -10.67%、
CX -8.13%、total depth -2.25%、最大operator残差$1.34\times10^{-15}$だった。中央RTE差だけを既存
proxyの$q$ slopeへ加えたbridgeでは$L_D=3/12$の点順位が反転した。続くWP05-aでは選択policyを
complete controlled partial-$S_2$／Hadamard wrapperへ接続し、$q=1,2$較正から未使用$q=4$をRZ最大
2.29%で予測した。中央additive bridgeのfull-wrapper RZ残差も最大2.63%で5%基準を通過した。
固定WP04条件の点順位は$L_D=3$となったが区間は重なった。WP05-bでは$q=8$と比較対照
$\delta=0.01$へ拡張し、$\delta=0.02,r=32,q=8$の初回5%逸脱を独立32 trajectoryで再検証した。
再検証では選択policyのRZ誤差0.52%、全metric最大0.54%、full basisのRZ誤差0.83%となり、
5%基準を通過した。続くWP01-D/C07では$\alpha$・shot数を候補ごとに再最適化し、点推定で
$L_D=3$が$L_D=12$より13.92%低かった。5% local model区間は僅かに分離した一方、25%移送区間は
重なるため、頑健な方向判断は未確定である。G08で後半3 roundへのcost集中を確認し、M08の
$q=16,32$直接holdoutはselected RZ 2.466%、観測RZ最大3.286%で通過した。これらの実測幅による
再集計ではlocal区間が分離するが、直接domainは$q\leq32$で25%移送区間は重なるため、頑健判定は
変わらず、現比較は最終的な科学的優位性評価ではない。
2026-09-23のM06/L08では、同一trajectoryをoptimization level 2で再compileした。q=16,32 proxyは5%基準を通過したが、固定plan focused再集計の点推定差は8.42%、区間分離上限は1.881%となり、実測selected RZ discrepancy 2.340%で区間が重なった。従ってcompilerをまたぐlocal分離は未確立で、頑健判定は`undetermined_under_compiler_and_transfer_sensitivity`である。
続くN07/P03では不確かさをsampling、model bias、compiler、長q移送、状態準備、外部移送に分離し、状態準備をRZ相当/shotのパラメータとして再集計した。`L_D=3`は2,376 shot多く、共通準備costは常に点推定利得を縮める。点推定break-evenはopt1で約9,905万、opt2 focusedで約4,707万RZ相当/shotだが、opt2 focusedはP=0ですでに区間が重なり、compiler-robustな区間優位性は確立しない。WP11では11個のartifactをT1--T7へ統合し、T4/T7を主軸、T1を範囲変更、T3を保留とした。次の一件は`L_D=3`のopt2未測定`r=1,2,4,8,16`を埋めるall-r coherent opt2再最適化であり、外部instance pilotは棄却せずその後へ延期する。
固定条件、数値、成果物は
[`research_direction_prevalidation.md`](docs/research_direction_prevalidation.md)と
[`research_direction_ablation.md`](docs/research_direction_ablation.md)、
[`research_direction_pf_sensitivity.md`](docs/research_direction_pf_sensitivity.md)、
[`research_direction_gate_s1.md`](docs/research_direction_gate_s1.md)、
[`research_direction_structure_pilot.md`](docs/research_direction_structure_pilot.md)、
[`research_direction_sequence_policy.md`](docs/research_direction_sequence_policy.md)、
[`research_direction_full_scope.md`](docs/research_direction_full_scope.md)、
[`research_direction_full_scope_extension.md`](docs/research_direction_full_scope_extension.md)、
[`research_direction_decision_cost.md`](docs/research_direction_decision_cost.md)、
[`research_direction_late_round_proxy.md`](docs/research_direction_late_round_proxy.md)、
[`research_direction_compiler_transfer.md`](docs/research_direction_compiler_transfer.md)、
[`research_direction_uncertainty_break_even.md`](docs/research_direction_uncertainty_break_even.md)、
[`research_direction_wp11_synthesis.md`](docs/research_direction_wp11_synthesis.md)に記録する。

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
| [`partial_randomized_trotter_prevalidation_catalog.md`](partial_randomized_trotter_prevalidation_catalog.md) | 研究方向を選ぶための事前検証backlog | 実行層・Gate S1・条件付きbranchの索引。実施済み範囲との対応は[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)を参照 |
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

WP11が選択したM06-F計算経路は、`src/trotterlib/research_direction_full_opt2.py`、
`src/trotterlib/research_direction_full_opt2_completion.py`、
`src/trotterlib/research_direction_full_opt2_extension_analysis.py`、
`src/trotterlib/research_direction_proxy_lineage_reconciliation.py`、
`scripts/run_research_direction_full_opt2_compute.py`、
`scripts/run_research_direction_full_opt2_analysis.py`、
`scripts/run_research_direction_full_opt2_completion.py`、
`scripts/run_research_direction_full_opt2_extension_analysis.py`、
`scripts/run_research_direction_proxy_lineage_reconciliation.py`、
`tests/test_research_direction_full_opt2.py`、
`tests/test_research_direction_full_opt2_completion.py`、
`tests/test_research_direction_full_opt2_extension_analysis.py`、
`tests/test_research_direction_proxy_lineage_reconciliation.py`、
[`docs/research_direction_full_opt2.md`](docs/research_direction_full_opt2.md)を一組として読む。
初期36 taskとfresh-32拡張15 taskは51/51で完了し、両gate通過後のcoherent再最適化も完了した。
A0は新規compileなしで最新fresh `q=1,2` proxyを旧`q=16,32`固定holdoutへ再照合した。
compute resultは`artifacts/research_direction_full_opt2/2026-09-24/`、最終監査・解析は
`artifacts/research_direction_full_opt2/2026-09-25/`に置く。

2026-09-25からはA0完了後の研究テーマ選定を`P-B -> P-C -> P-A`で行う。P-Bは
`src/trotterlib/research_direction_signal_weight_pilot.py`、
`scripts/run_research_direction_signal_weight_pilot.py`、
`tests/test_research_direction_signal_weight_pilot.py`、
[`docs/research_direction_signal_weight_pilot.md`](docs/research_direction_signal_weight_pilot.md)を一組として読む。
現H4 gridでは案Bを進めるsignal差が得られず停止した。P-Cは
`src/trotterlib/research_direction_geometry_energy_difference_pilot.py`、
`scripts/run_research_direction_geometry_energy_difference_pilot.py`、
`tests/test_research_direction_geometry_energy_difference_pilot.py`、
[`docs/research_direction_geometry_energy_difference_pilot.md`](docs/research_direction_geometry_energy_difference_pilot.md)、
`artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/`を一組として読む。
固定H4 5 geometryの未使用geometry/delta差分bias予測gateを通過して案Cを候補として残し、
P-Aは`src/trotterlib/research_direction_joint_synthesis_pilot.py`、
`scripts/run_research_direction_joint_synthesis_pilot.py`、
`tests/test_research_direction_joint_synthesis_pilot.py`、
[`docs/research_direction_joint_synthesis_pilot.md`](docs/research_direction_joint_synthesis_pilot.md)を一組として読む。
未使用列長3、5、8でinterval-union DPが現行policy比pooled RZを7.19%減らし、全gateを通過した。
3 pilotの比較は`research_direction_theme_selection.py`と同名runner/test、
[`docs/research_direction_theme_selection.md`](docs/research_direction_theme_selection.md)へ固定した。
P-Aを暫定主題、P-Cを副候補、P-Bを現範囲で停止とした。続くscoped prior-art auditは
[`docs/research/pa_joint_synthesis_prior_art_audit.md`](docs/research/pa_joint_synthesis_prior_art_audit.md)に、
計算前のblind条件は
[`docs/research/pa_joint_synthesis_blind_validation_preregistration.md`](docs/research/pa_joint_synthesis_blind_validation_preregistration.md)に固定した。
実装は`src/trotterlib/research_direction_joint_synthesis_blind_validation.py`、
`scripts/run_research_direction_joint_synthesis_blind_validation.py`、
`tests/test_research_direction_joint_synthesis_blind_validation.py`を一組として読む。compile前dry-runで
48 holdout taskと6 operator probeのevent digestを
`artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25/`へ固定した。未使用H5 snapshotと
H4 opt2 compiler contextの2 stratumは48/48 holdout、6/6 operator probeまで完了し、両方で全6 gateが
通過した。結果とscopeは
[`docs/research_direction_joint_synthesis_blind_validation.md`](docs/research_direction_joint_synthesis_blind_validation.md)に記録する。
blind gate時点ではP-A v1を正式候補へ進めた。続く形式化は
`src/trotterlib/research_direction_joint_synthesis_formalization.py`、
`scripts/run_research_direction_joint_synthesis_formalization.py`、
`tests/test_research_direction_joint_synthesis_formalization.py`、
[`docs/research/pa_joint_synthesis_v1_formalization.md`](docs/research/pa_joint_synthesis_v1_formalization.md)を
一組として読む。DP最適性は有限proxy候補内で形式化できたが、全54 recordが1 run 1 segmentで、
interval分割と非零Taylor-order構造は未検証だった。P-Aを条件付き候補へ狭め、次はこの2点を明示的な
one-segment baselineに対して判別する。H12、長RPE総cost、coupling/noiseは次の必須作業ではない。

続く非退化mechanism validationは
`src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py`、
`scripts/run_research_direction_joint_synthesis_mechanism_validation.py`、
`tests/test_research_direction_joint_synthesis_mechanism_validation.py`、
[事前登録](docs/research/pa_joint_synthesis_mechanism_validation_preregistration.md)、
[結果文書](docs/research_direction_joint_synthesis_mechanism_validation.md)を一組として読む。
training/blindをDF fragmentで分離したforced-support order-2全30 taskでcandidateと一区間baselineの
plan・compiled metricが完全一致し、run内分割は0件だった。P-Aのinterval claimを停止してP-Cへ戻る。
H12、長RPE総cost、coupling/noiseは次の必須作業ではない。


続くP-C tracking・breakdown validationは
`src/trotterlib/research_direction_geometry_tracking_breakdown.py`、
`scripts/run_research_direction_geometry_tracking_breakdown.py`、
`tests/test_research_direction_geometry_tracking_breakdown.py`、
[事前登録](docs/research/pc_geometry_tracking_breakdown_preregistration.md)、
[結果文書](docs/research_direction_geometry_tracking_breakdown.md)、
`artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/`を一組として読む。
固定8 geometry・2 policyの16/16 taskを完了した。追跡prefixは全点で独立先頭3 fragmentと同一、
blind coefficientは3/5、pairは1/4だけが固定誤差基準内で、診断正解率は50%だった。
7 gate中3 gate通過でcurrent H4 familyのP-Cを主研究候補から停止した。
先行P-C pilotの訂正版exact-energy artifactも同時に参照する。

P-D energy係数・random-tail負担Pareto監査は
`src/trotterlib/research_direction_energy_tail_pareto.py`、
`scripts/run_research_direction_energy_tail_pareto.py`、
`tests/test_research_direction_energy_tail_pareto.py`、
[事前登録](docs/research/pd_energy_tail_pareto_preregistration.md)、
[結果文書](docs/research_direction_energy_tail_pareto.md)、
`artifacts/research_direction_energy_tail_pareto/2026-09-25/`を一組として読む。
development `L_D=3`とblind `L_D=4`で同じselection reversalを確認し、7 gate全てを通過した。
ただしexact two-block pilotであり、負時間RTEと内部`H_D`誤差の次gateを通るまでは
`P-D-conditional-candidate`として扱う。

P-D現実化Go/No-Goは`research_direction_pd_realization.py`と同名runner/test、
[事前登録](docs/research/pd_realization_go_no_go_preregistration.md)、
[結果文書](docs/research_direction_pd_realization.md)、
`artifacts/research_direction_pd_realization/2026-09-25/`を一組として読む。

続くS0/S1公平再最適化は
`src/trotterlib/research_direction_pd_fair_comparison.py`、
`scripts/run_research_direction_pd_fair_comparison.py`、
`tests/test_research_direction_pd_fair_comparison.py`、
[S0契約](docs/research/pd_primary_research_contract.md)、
[既知baseline](docs/research/pd_prior_art_and_baselines.md)、
[S1事前登録](docs/research/pd_s1_fair_comparison_preregistration.md)、
[結果文書](docs/research_direction_pd_fair_comparison.md)、
`artifacts/research_direction_pd_fair_comparison/2026-09-26/`を一組として読む。
B1b/B2/B4は同じ選択となりCase C/Dの証拠は得られず、B1aの`m_D`上限依存により
`stop_s1_undetermined_boundary_no_go_decision`で停止した。

その固定artifactの事後再解析は
`src/trotterlib/research_direction_pd_s1_posthoc.py`、
`scripts/run_research_direction_pd_s1_posthoc.py`、
`tests/test_research_direction_pd_s1_posthoc.py`、
[外部review](pd_s1_review_5c331f0.md)、
[事後計画](docs/research/pd_s1_posthoc_reanalysis_plan.md)、
[結果文書](docs/research_direction_pd_s1_posthoc.md)、
`pd_s1_posthoc_reanalysis_v1.json`を一組として読む。一次Case Bは保存し、B1b/B2/B4だけを
固定候補集合でCase A相当と事後解釈する。P-D S2は開始せず、R3も未採用である。

R3の次段判断は
[R3先行研究監査と条件付き最小研究契約](docs/research/r3_prior_art_and_minimal_contract.md)を読む。
広いsplit/error/cost最適化は既存研究との重複が強いためNo-Goである。R3-Sはselectiveな認証・棄却へ狭めて監査したが、一般certified multi-fidelity法との差分と
quantum-specific保証を固定できなかった。`STOP_R3_NO_METHOD_DELTA`でR3を停止し、数値pilotを開始しない。

finite-RTE phase/radius分離FR-1は
`src/trotterlib/finite_rte_phase_amplitude.py`、
`scripts/run_finite_rte_phase_amplitude.py`、
`tests/test_finite_rte_phase_amplitude.py`、
[FR-0契約](docs/research/finite_rte_phase_amplitude_contract.md)、
[scoped先行研究監査](docs/research/finite_rte_phase_amplitude_prior_art.md)、
[FR-1事前登録](docs/research/finite_rte_phase_amplitude_fr1_preregistration.md)、
[結果文書](docs/finite_rte_phase_amplitude_validation.md)、
`artifacts/finite_rte_phase_amplitude/2026-09-26/`を一組として読む。
33条件・99状態でG0/G1/G3/G4を通過し、G2は不通過だった。判定は
`GO_FR2_MECHANISM_ONLY`であり、利用可能情報による実用的GOではないため旧FR-2は開始しない。
その後のFR-R0で正scalar分離、情報層、強いbaseline、非一様系のdecision gateを正式契約へ固定した。
FR-R1a事後解析は完了し、正scalar処理後にFR固有の片側認証差が残らない
`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`となった。続くFR-R1b非一様4×4検証では、同情報FR境界が
norm境界より厳しい8 witnessを得た一方、固定予算の片側認証差は0だった。現行判断は
`MECHANISM_ONLY_NO_PRACTICAL_GO`で、強制停止中である。

FR-1後の再設計は、[提案文書](fr1_revised_research_plan_20260926.md)と
[FR-R0正式契約](docs/research/fr_revision_scalar_structure_contract.md)を読む。後者が現行の規範で、
正scalar分離、I0/I1/I2、共通$\gamma$と最適化比較、FR-R1の事前登録要件を固定する。
statusは`FR_R1B_COMPLETE_MECHANISM_ONLY_NO_PRACTICAL_GO_MANDATORY_STOP`である。
[FR-R1a実装](src/trotterlib/fr_revision_fr1a_posthoc.py)、
[runner](scripts/run_fr_revision_fr1a_posthoc.py)、
[test](tests/test_fr_revision_fr1a_posthoc.py)、
[FR-R1a事後計画](docs/research/fr_revision_fr1a_posthoc_plan.md)と
[FR-R1a結果](docs/fr_revision_fr1a_posthoc.md)、
[artifact](artifacts/fr_revision_fr1a_posthoc/2026-09-26/)、
[FR-R1b事前登録](docs/research/fr_revision_nonuniform_preregistration.md)、
[FR-R1b実装](src/trotterlib/fr_revision_nonuniform.py)、
[runner](scripts/run_fr_revision_nonuniform.py)、
[test](tests/test_fr_revision_nonuniform.py)、
[FR-R1b結果](docs/fr_revision_nonuniform.md)、
[artifact](artifacts/fr_revision_nonuniform/2026-09-27/)を一組として読む。
FR-R2は開始していない。

FR-R1b後の現行方針は、[研究主張・証明義務・完成原稿契約](docs/research/fr_research_claim_and_manuscript.md)
を読む。正scalar分離と方向依存補正をC1、同じ情報層でのstrict improvementと改善不能条件をC2、
resource designを条件付きC3として分離する。新規性とT1--T4の証明監査が終わるまでは、既存artifact
だけを用いて原稿を完成させ、FR-R2、H4/H12、長RPE、新しいgridへ進まない。

## ファイルの状態区分

### 現行の正本

- `docs/research/研究概要・現状.md`
- `docs/research/研究目的・研究課題.md`
- `docs/research/研究方法・解析手順.md`
- `docs/research/数値実験・評価計画.md`
- `VALIDATION_STATUS.md`
- `artifacts/validation_manifest.json`
- `docs/research/fr_revision_scalar_structure_contract.md`（FR-R0の現行比較契約）
- `docs/research/fr_research_claim_and_manuscript.md`（FR-R1b後の研究完成フェーズ契約）

### 現行実装・検証

- `docs/research/pr2_s0_s1_execution_amendment_v3.md`、
  `artifacts/pr2_s1_s3_preregistration/2026-09-28/pr2_s0_s1_authorization_manifest_v3.json`
  （S0実行、条件付きS1 correctness、S1後mandatory STOPの結果前許可）
- `src/trotterlib/pr2_s0_s1_validation.py`、`scripts/run_pr2_s0_s1_validation.py`、
  `tests/test_pr2_s0_s1_validation.py`（snapshot/identity/corrected-estimator gateとS1 correctness-only経路。
  S2/S3 commandは実装しない）
- `src/trotterlib/pr2_v4_s2_development_validation.py`、`scripts/run_pr2_v4_correctness.py`、
  `scripts/run_pr2_s2_development.py`、`tests/test_pr2_v4_s2_development_validation.py`
  （別snapshot系列のV4 correctnessとdevelopment-only S2比較のserial経路）
- `src/trotterlib/pr2_v4_s2_parallel_execution.py`、
  `scripts/run_pr2_s2_development_parallel.py`、
  `tests/test_pr2_v4_s2_parallel_execution.py`、
  `docs/pr2_s2_parallel_execution.md`（同じS2 cellと段階barrierを保つbounded CPU並列実行層。
  実H4 S2結果は`docs/pr2_v4_s2_development_validation.md`と
  `artifacts/pr2_v4_s2_development/2026-09-29/`で追跡する）
- `docs/research/pr2_matched_accuracy_prior_art_gate_v1.md`、
  `docs/research/pr2_matched_accuracy_resource_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_implementation_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`、
  `docs/research/pr2_matched_accuracy_m1_execution_authorization_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`、
  `docs/research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`、
  `docs/research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md`、
  `docs/research/pr2_m1_b1_preexecution_external_review_request_1228168.md`
  （S2後の新規性gate、M1前研究契約、zero-compute実装契約。候補identity、16-cell selector、schema、
  seed規則、追加prior-art gate、M1-A/M1-B hard barrier、compile-free M1-A予算を固定済み）
- `src/trotterlib/pr2_matched_accuracy_m1_contract.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_precompile_barrier.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_execution.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py`、
  `scripts/run_pr2_matched_accuracy_m1_contract.py`、
  `scripts/run_pr2_matched_accuracy_m1_precompile_barrier.py`、
  `scripts/run_pr2_matched_accuracy_m1_a.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1_contract.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1.py`、
  `tests/test_pr2_matched_accuracy_m1_contract.py`、
  `tests/test_pr2_matched_accuracy_m1_precompile_barrier.py`、
  `tests/test_pr2_matched_accuracy_m1_execution.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_contract.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_execution.py`、
  `docs/pr2_matched_accuracy_m1_a_validation.md`、
  `docs/pr2_matched_accuracy_m1_b1_result_validation.md`、
  `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/`、
  `artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/`
  （候補列挙、synthetic barrier、development-only dense signal M1-A、194+16 cellのbounded compile
  plan/source/result/validation。science runnerはcompile map完成後に研究四分岐を自動判定せずreview待ちで
  停止し、別validatorがcheckpoint/cache再集計と研究reviewを行う。このstageではheld-out未承認）
- `src/trotterlib/pr2_matched_accuracy_m1_b1_result_validation.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1_result_validation.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_result_validation.py`
  （保存済みM1-A/M1-B1 artifact、全checkpoint、candidate別SQLite cacheをread-onlyで検査し、
  actual Pareto、旧selector、fixed-q=8、proxy、状態準備感度を再集計する。分子snapshot/held-outは読まない）
- `docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md`、
  `docs/research/pr2_m2_transfer_contract_external_review_request_06b2c32.md`、
  `src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、
  `scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、
  `tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、
  `artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`
  （M1-B1の5構成、判定量、重大underestimate、4 status、196-wrapper上限をzero-compute固定する。
  v1/draftは履歴として保存し、正式v2はusable B2にPareto support/ratioを統一してcommit固定した。
  zero-compute contract runner自身はheld-outを開かず科学実行を認可しない。後続M2結果は下記の専用入口）
- `docs/research/pr2_s0_reproduction_stop_c644925.md`、
  `docs/research/pr2_s0_external_review_request_c644925.md`、
  `artifacts/pr2_s0_s1_validation/2026-09-28/`（development byte-level hash不一致による
  `STOP_INPUT_REPRODUCTION_MISMATCH`、S1未実行、held-out signal/cost/ranking未開封、および外部レビュー依頼）
- `src/trotterlib/` のDF、RTE、RPE、compiled-cost関連モジュール
- 対応する `scripts/run_*.py`、`tests/test_*.py`、`docs/*_validation.md`
- manifestに登録され、statusと限界が明示されたartifact

### 歴史的・補助的資料

- `abe_trotter_project.ipynb`：旧来の高次PF解析ノートブック
- `Partial Randomized Study Protocol.md`：初期計画と意思決定の履歴
- `codex-inst.md`：過去の実装依頼メモ
- `README_partial_randomized_pf.md`：旧screeningを含む実装経路の説明
- `docs/main_audit_20260801.md`：特定時点の監査記録
- `partial_randomized_trotter_validation_review_898da848.md`：commit `898da848`を対象にした外部GPTレビュー。
  現在のP-A/P-C/P-B判断より前の評価であり、一次証拠または現行仕様ではない
- `partial_randomized_trotter_research_redesign_20260925.md`：上記レビューを受けたテーマ再設計入力。
  pilotの着想と停止条件を確認する補助資料で、実施結果と現在の判断は正本文書を優先する
- `fr1_revised_research_plan_20260926.md`：FR-1後の再設計入力。採択済み部分の正本は
  `docs/research/fr_revision_scalar_structure_contract.md`を優先する
- `research_focus_and_completion_plan_16d4482_20260927.md`：FR-R1b後の完成方針を検討した入力資料。
  採択済みの主張階層、証明義務、停止条件は`docs/research/fr_research_claim_and_manuscript.md`を優先する

これらは削除していないが、現在の研究方針や最新結果を確定する根拠には使わない。

## GPTが回答・資料作成するときの確認事項

- 事前検証カタログの実施状況は、[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)から専用文書・artifact・testまで追跡する。
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

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持した。この契約修正時点では科学実行とheld-out accessは未認可だった。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)と
`src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py`、
`scripts/run_pr2_matched_accuracy_m2_transfer.py`、
`tests/test_pr2_matched_accuracy_m2_transfer_execution.py`、
`artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/`を対応させる。
契約v2と正式planはcommit固定済み。source/authorization/環境gateはheld-out読み込みより前に置き、
旧draftは履歴だけであり、正式planはCOMMIT_BOUND。source/plan固定後に別authorizationと最終reviewを経て
一回のM2を実行した。科学sourceは変更せず、結果後のmandatory STOPを維持する。

## PR-2 M2最終実行前レビューの入口

[実行authorization](docs/research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)と
[最終review依頼](docs/research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)、
`artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json`を一組として読む。
actual science sourceとplanを変更せず別authorizationをcommitし、review待ちで一旦停止した。
review承認と利用者指示を得た後に一度実行し、現在は`TRANSFER_SUPPORTED`後の研究方針review待ちである。

## PR-2 M2 held-out結果の入口

[結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)と
`artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/`のresult、runner manifest、complete marker、
launch/post-execution auditを対応させる。196 wrapper、5構成、source128、pre/post84＋134 testsを検査した
result commitへ収録するlocal execution evidenceであり、immutable CIではない。`.runtime`とone-shot registryはcommit対象ではない。
固定5構成のtransfer支持だけを解釈し、全status後STOP・追加科学計算未認可を維持する。


## H4 geometry 契約準備 v1（2026-10-06・local未commit）

[準備bundle](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md)は契約schema・zero-compute plan・pure JSON validatorと129合成検査の入口。
6距離、218 template/点、74,784 wrapper、最大12 workersを固定し、生成/seed/memory/wall/outputはreview待ち。
science source/runnerの追加ではなく、旧公開draft・source・科学結果は不変。本計算・port・commit/pushは未認可、STOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](docs/research/track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](docs/research/track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 run05：identity hash分割・5秒監視維持

[run05固定sourceと再実行binding](docs/research/track_a_h4_lazy_identity_run05.md)を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 新サーバー引継ぎ（2026-10-07）

[引継ぎ資料入口](docs/research/track_a_h4_new_server_handoff_20261007.md)に、固定sourceの取得手順、run05監視STOP、累積予算、入力・証拠の別転送、新hostの未承認事項をまとめた。今回の追加は軽量資料のみ。本計算・入力再生成・GPU・共有環境変更を開始しない。


## 2026-10-07 H4新host source取得・環境解決案

[H4新host source取得・環境解決案](docs/research/track_a_h4_new_server_preparation_20261007.md)。指定commit/source19照合済み、環境18 version/45 RECORD差の解決案でSTOP。source修正・新tests・production未実施、入力/停止証拠別送待ち、allowed_cpus=[]/approved=false。旧科学結果・原稿・Track Bは保持。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](docs/research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。

人工runner `scripts/resource_applicability/run_h4_monitor_fix_a_tests.py`、auditor `scripts/resource_applicability/audit_h4_monitor_fix_a.py`。実runnerを起動しない。


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

実装：[signal](src/trottertracks/resource_applicability/h4_geometry/signal.py)・[ledger](src/trottertracks/resource_applicability/h4_geometry/ledger.py)。限定[runner](scripts/resource_applicability/run_h4_lightweight_speedup_tests.py)・[tests](tests/tracks/resource_applicability/test_h4_lightweight_speedup.py)・[bundle](artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/README.md)。

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


## 2026-10-09 H4 worker例外保存・cleanup競合修正

[修正・39限定人工・未確認事項](docs/research/track_a_h4_worker_error_fix_20261009.md)。SOURCE d3bb388401bfd25d31c53e7f785b533a7753ddb3/closure41。bootstrap/unpickle/GC/IPC例外を保存→observer report→own STOP順へ固定し、driver pulseのpublication reader競合も修正。元production例外は未記録で原因未特定。人工IPC/cold import合格、追加transpile/科学入力読込/本体/実worker/GPU/affinity0。carry21・現13GiB/74804を保持、新runtime認可false・未sealでSTOP。


## 2026-10-09 H4 run03：人工compile省略・本計算で診断する方針

[新run準備・累積上限2件の追加承認案](docs/research/track_a_h4_production_run03_20261009.md)。利用者が追加人工compileを省略し、実本計算で問題を保存・停止して修正する方針を指示。元compiler原因未特定は受容済みの未検証事項として保持。
SOURCE6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b/closure44、32 pure binding tests PASS、回路build/compile/transpile/科学array/実worker/affinity/GPU/本体0。
run02 native14identity×2ABSENT・7原本hash・ledger予約1件/journalを結合。carry21/8692723164B/5766.582514658794s upper保持、失敗返却/resetなし。
次全mapworst16.232668GiB/74805、現承認13GiB/74804は不変。追加承認案は累積17GiBと74805だけ、environment/CPU/observer採用認可を再要求しない。
未使用run03/output/control/one-shot、draft approved/runtime false・allowed[]・未seal・commandnull。承認後最終binding/fresh gates合格なら一度map起動、完了/STOP後終了・retry/次stageなし。

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

## 2026-10-09 run03 plugin cache原因を修正、次の実行は未認可

[cache修正・実run STOP・限定metadata検証](docs/research/track_a_h4_entrypoint_cache_fix_20261009.md)。run03を一度実行したSOURCE6e68fd9bcc68e788db6f5d43eaa6a03866e53d3bと、新fix SOURCEcd162e9305143d81c28908b9732f8eef12cd6b89/closure46を区別。
stevedore disk mkdir拒否を同guardで再現し、private .disableをown-process XDGへ固定。8 tests/6 plugin metadata順序一致・write0、科学compile/回路/Qiskit import/実worker/追加本体0。
run03全14×2ABSENT、carry22/12956511264B/6004.111340102032s upper・現17GiB/74805保持。次全map20.203GiB/74806に対する21GiB/74806は未承認proposal。
次run/carry22/native proof/SOURCE seed/profile/binding/fresh gate未結合、false/allowed[]/未seal/commandnull、自動retryなし。


## 2026-10-09 H4：失敗分を次回の予算に加算しない

利用者の「失敗した過去分は残さなくてよい」により、次回の新規mapはcarry0（actual/charge/wall）で実行単体を上限の対象とする。
旧carry22/12956511264B/6004.111340102032sと停止証拠・入力は履歴として保持。全map静的見積り8.136939GiB/74784件は承認済み17GiB/74805以内、旧21GiB/74806増額案は不要。
SOURCE cd162e9305143d81c28908b9732f8eef12cd6b89は不変で、現validatorは消費済みrun03/CARRY21。次の新SOURCE/seed/plan/auth/reviewと未使用runをcarry0へ再結合する必要がある。
予算方針のみ承認済み、新one-shot指示/本体起動なし。flagsfalse/allowed[]/未seal/commandnull、科学/追加transpile/GPU/共有環境変更0。
[最新の予算方針・見積り](docs/research/track_a_h4_per_attempt_budget_20261009.md)。


## 2026-10-09 H4 run04：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE d31b51080665a7eea806ff1adc17f3e0d19151fc/closure49、library cache v2保持、新run04。純粋11 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU12workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](docs/research/track_a_h4_production_run04_20261009.md)。


## 2026-10-09 H4 run05：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE 9d1471aff5840a76fa1579f4e14e71e62b8497a3/closure52、library cache v2保持、新run05。純粋16 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU12workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](docs/research/track_a_h4_production_run05_20261009.md)。


## 2026-10-10 H4 run06：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE 697843fbbd2a2aa7224261aa26f8da141ca57687/closure55、library cache v2保持、新run06。純粋21 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU4workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](docs/research/track_a_h4_production_run06_20261010.md)。


## 2026-10-10 H4 run06もhost PSI0.18でSTOP、inactive修正案を固定

run06は4workersで約856.585秒、host PSI0.18/nonroot全0/OOM増分0/available約445GiB/role8GiB内でもmemory_pressure STOP。
4actual予約/完了0・signal0、6identity残存0を2回確認。STOPの発生元や正確な終了時刻は補完しない。
元D4 contractのmemory PSI full avg10>0即STOPを変更するには明示契約改定が必要。通常の修正/再実行認可をpressure変更に拡張解釈しない。
SOURCE41d2b61f640937255c5fa6cab9549ae67b83c462/closure58、旧55 source不変、inactive evaluator・16pure tests PASS。
提案はhost-only PSI<1%を完全なnonroot PSI0/OOM0delta/effective120.25GiB/fresh5秒/既存role8GiB guardで条件付き許容。nonroot>0/host>=1%/欠測等はSTOP。
threshold最適性/共有負荷安全性/本計算成功は未検証。production未接続、approved/runtime false・allowed[]・未seal・commandnull。
carry0/17GiB/74805/72h/科学/compiler/凍結入力維持。本体再起動0、GPU/共有環境/venv/他job変更0。
[最新入口](docs/research/track_a_h4_host_pressure_policy_fix_20261010.md)。


## 2026-10-10 H4 run07：host-only PSI例外の明示承認・本番接続

利用者が「この条件を承認して再実行」と明示承認。host-only PSI<1%は全nonroot0/OOM増分0/effective120.25GiB/fresh5秒でのみ許容、他pressure/欠測/8GiB/5秒等はfail-closed維持。
SOURCE1dbdd1be2133a59ab81c7b81a6074af0880f72a3/closure62、33pure tests PASS、初回失敗ログ保持。actual kernel baselineと原driver OOM/hierarchyをobserver子へ固定transport。
run06原STOP/費用/6×2ABSENT/10filesSHAを保持し、新run07/carry0/CPU4subset[2,4,5,6]/driver16/observer18で再実行。17GiB/74805/72h・科学/compiler/凍結6入力不変。
独立review・immutable artifact・直前fresh/未使用lock PASS後に起動。共有環境/venv/他job/GPU変更0。開始確認後chat終了可、observer監視は継続。
[最新入口](docs/research/track_a_h4_production_run07_20261010.md)。


## 2026-10-10 H4 run07 STOP原因監査：q2 workerのAS8GiB枯渇が最有力

run07はgeometry0.70/B0/L_D3/q1/delta0.8の2wrapperをcompile COMPLETEしsignal1保存後、q2/delta0.4/cosine workerのresponse EOFでSTOP。
last AS7.999725GiB/残288KiB、RSS7.648GiB、hostavailable438.6995GiB/PSI0/OOM増分なし。full dense256x2568qUnitaryGateのgenericQSDでq1すでに約273万operation。
AS制限による割当失敗が強く支持されるが、元native error/signal/compile-vsmetrics phaseは未記録。stderrDEVNULLとformatter二次MemoryErrorによるerrorframe欠落をsource/mock2caseで確認。
SOURCE1dbdd1be2133a59ab81c7b81a6074af0880f72a3/closure62不変。6予約(2COMPLETE4RESERVED)/signal1/charge4263870068B/6identities×2ABSENT・raw18fileSHAを保存。
新science/input/compile/worker/affinity/GPU/source/caps変更・再起動0。全map成功やmemory leakは断定せず、原partialは次attemptへ混合しない。
[原因監査入口](docs/research/track_a_h4_run07_stop_cause_audit_20261010.md)。


## 2026-10-10 H4 worker AS/RSS32GiBを明示承認・SOURCE固定

利用者の「上限を３２Gで修正して」でnewhost4workersを32GiB、driver8GiB/observer256MiB・64MiB/head16GiBを維持。両fresh/startup gateは152.25GiB。
親soft8でhardをchild準備まで保持し、workerは認可/checkout後に32、全ready後driverhard8。observerはtrusted driver PIDとnative登録workerを8/32で区別。
独立review P2 cleanup signal errorでの後続掃除skipを修正、最終53pure/mock PASS。実child/science/transpile/affinity/本体0。
SOURCE b652fff9d2907015d3b8b7b23ce8bf0fe4fce33a/closure65、旧SOURCE変更7/不変55+new3。科学/compiler/凍結入力/旧pressure/carry0/17GiB/74805/72h不変。
SOURCE/profile/input/carry/capsの準備binding、memory承認true・overalllaunchfalse/allowed[]/未seal/commandnull。fresh run/plan/auth/review/freshgatesは次回起動へ。
旧run07 one-shot/partial/cacheは保持し再利用しない。元native診断2gapsと全map完走は未検証。共有環境/venv/他job/GPU変更0。
[修正資料入口](docs/research/track_a_h4_worker_memory32_fix_20261010.md)。


## 2026-10-10 H4 run08：worker32GiBで明示再実行

利用者が本計算再実行と開始確認後chat終了を明示指示。SOURCE8c917a5af7943969d4abede8b0bb012880efeee0/closure67、新run08/carry0。
worker4 AS/RSS32GiB、driver8GiB、observer256MiB/64MiB、admission152.25GiB、CPU2/4/5/6・driver16・observer18。科学/compiler/旧pressure/凍結6入力/17GiB/74805/72h不変。
最新run07 native6×2ABSENT/元18fileSHA/cost保持、旧partial/control/one-shotを再利用しない。新11metadata/byte tests PASS、旧53campaignと人工compile再実行0。
独立reviewとimmutable artifact・直前SOURCE/profile/input/carry/CPU/memory/pressure/OOM/容量/inode/quota/未使用lock合格後、一度mapを起動。実状態はruntime receiptへ別記録。
共有環境/既存venv/他job/GPU変更0、home内だけ。全map完走は未検証、次stage未認可。
[起動資料入口](docs/research/track_a_h4_production_run08_20261010.md)。


## 2026-10-10 H4 run08 host1.26% STOP・限定猶予の未承認案

run08は約32分でhostPSI1.26%によりSTOP、全nonroot0/OOM増分なし/available約426GiB/role32GiB内。compile2/signal1、6 owned identities×2ABSENT、旧費用保持・次carry0。
SOURCE736bdc15ccfec6b2a715e1723850a481f0f2ff6b/closure69、旧production67不変、新inactive2files/14pure PASS。毎秒監視は維持しhost1〜5%だけ152.25GiB/nonroot0/OOM0/fresh5秒の下で最大30秒猶予、5%以上等は即STOP案。
既承認1%即STOPの変更なので利用者承認前は本番接続/再起動しない。flagsfalse/allowed[]/未seal/commandnull。科学/compiler/worker4・32GiB/共有環境/venv/GPU変更0。
[限定修正案・検査](docs/research/track_a_h4_host_pressure_grace_fix_20261010.md)。


## 2026-10-10 H4 run09：host PSI猶予の明示承認・再実行

利用者が「猶予案を承認して再実行」と承認。host1〜5%だけnonroot0/OOM同一/available152.25GiB/fresh5秒の下で30秒猶予、5%以上・30秒継続・nonroot/OOM/欠測は即STOP、毎秒監視継続。
SOURCE251993785ef1dab2a3891bdbb5d079f5d2184f4d/closure72、新16pure回帰PASS、旧14/53/11campaign・人工compile反復0。worker4/32GiB・driver8GiB・observer256/64MiB、CPU2/4/5/6・driver16・observer18不変。
最新run08原17fileSHA/6identity×2ABSENT/cost6・complete2保持、carry0/17GiB/74805/72h/科学/compiler/凍結6入力維持。新run09、旧partial/one-shot/control/output混合なし。
限定差分review・固定artifact・直前freshgates合格後一度map、開始確認後chat終了可。共有/venv/他job/GPU変更0。
[認可・検証・起動資料](docs/research/track_a_h4_production_run09_20261010.md)。


## 2026-10-10 H4 realistic runtime：Gaussian構造化の未承認候補

現dense256×256/8qgenericQSDで初期q1/q2回路273万/507万operations、全map72h完走は現速度で厳しい。run09を維持した別worktreeで最大28two-mode Givens+8numberphase候補を実装。
SOURCE0cfdca65ed4a19d05e0cd886c9bbfe2bc26bee57/closure74、旧production72不変、新2files。12人工数値PASS、fullFock256/exterior-minors/oldlog-expm/fermionic signs/controlledphase1e-12一致、Qiskitbuild/transpile/NPZ/worker0。
compiledgatecountsは旧denseと変わる新cost系列案、実speedupと全map72hは未保証。adoption未承認/flagsfalse、run09に接続しない。科学/compileroptions/資源・共有環境不変案。
[候補・数値検証・採用範囲](docs/research/track_a_h4_gaussian_structure_proposal_20261010.md)。


## 2026-10-10 H4 run10：Gaussian新cost系列の明示承認・切替再実行

利用者が「新方式へ切替・再実行」と承認。ownrun09を停止し8予約/4complete/2signal/charge4263948142B・26fileSHA・6identity×2ABSENTを保持、旧partial再利用0。
SOURCE5aa6c3685afb9c3afcfbdc36f80244da0a7fe525/closure77、新7paths接続12tests PASS、旧数学12campaign反復0。structured Gaussian最大28two-mode+8numberphase、vacuum/JW/相対位相保持、新semantics/seed/cost系列へ分離。
科学H4/PF/RTE/compileroptions/input/環境/CPU4/32GiB/pressure1/5/30/17GiB/74805/72h/carry0不変。実Qiskit/compile速度・72h完走は未測定、人工compileは省略して実本計算で確認。
限定差分review・immutable artifact・freshgates全合格後一度run10起動、開始確認後chat終了可、監視継続。共有/venv/他job/GPU変更0。
[承認・新cost系列・起動資料](docs/research/track_a_h4_production_run10_20261010.md)。


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


## 2026-10-15 Track A：H8 D0 driver／H6 native B0+I bridge

[最終技術実装](docs/research/track_a_final_pre_review_implementation_20261015.md)／[科学判断handoff](docs/research/track_a_final_pre_review_handoff_20261015.md)。SOURCE60eeb51c…、独立bounded H8 driverとnative B0+I bridge、人工41件PASS。元科学SOURCE/結果/grant/ledger/venv不変、実値再会計・科学build/compile0。future契約false/CPU[]/未seal/commandnull、採択・起動は別承認、mandatory STOP。


## 2026-10-15 H6 独立資源観測の修正

H6 pilot run02はnative正しさ検証中のcallback stalenessでSTOP、compile0。既存metadata launcherの独立資源観測へ修正し、SOURCEb73580dd…/closure13、限定84testsとGIL保持C約12秒の2process実動作検査PASS。追加本番role/process0、5秒・資源・科学/compiler条件維持、旧消費/unknown保持、subagent0。新run03へ結合して再実行。
[修正・検証・新実行binding](docs/research/track_a_h6_external_monitor_fix_20261015.md)。


## 2026-10-15 H6 /proc退出競合修正・run04-v3起動

H6 run03は全native検証後の退出/proc競合でSTOP、1 worker PASS/受理0・全6 reap。SOURCE2e733d8a…でlive欠測を成功扱いせず5秒内の再観測/reapへ修正、92限定＋最小2childの退出/FD検査PASS。旧消費保持。CPU gateで未起動2grantを廃止、5秒idle選定・fresh gate後にrun04-v3起動。科学/profile/compiler/資源上限不変、subagent0。完了は未主張。
[修正・検証・新実行の入口](docs/research/track_a_h6_proc_exit_fix_20261015.md)。

## 2026-10-15 H6 fixed-delta T依存性の準備

[実行直前準備](docs/research/track_a_h6_fixed_delta_t_dependence_readiness_20261015.md)：新scripts/h6_fixed_delta_*・prepare_h6_fixed_delta_v1.py、人工tests、固定bundle。実科学計算は未認可。


## 2026-10-15 H6 fixed-delta result

[結果・図・全表・監査](docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md)。H6固定δ=.2/固定5構成、T=.4/.8/1.6。新144/144 compile・PR64完了、T=.8確認32再利用。ε=.05〜.005ではPR点費用約50〜55%削減、厳しい精度ではS4優位またはPR不適格。outer-PF biasが主因、希少次数・母平均未保証。wall3117.158秒、native exit0/全74 reap・残存0。保存314条件/7算術tests/900公開paired metrics PASS、新科学計算・共有変更0、mandatory STOP。


## 2026-10-19 Track A mechanism runner：準備完了・数値実行未承認

[実装・人工検査・固定契約](artifacts/resource_applicability/track_a_mechanism_runner_readiness/2026-10-19/integrated_readiness_report.md)。主要32比較だけを固定し、残り212はmetadataのみ。SOURCE 5c256067db29a326bcaa0a0e244b38f6b87bd138。実科学数値解析0、有効grant0、mandatory STOP。


## 2026-10-19 Track A Mechanism：主要32比較のone-shot完了

[結果・費用分解・説明未解決](docs/research/track_a_mechanism_primary32_results_20261019.md)。SOURCE 5c256067db29a326bcaa0a0e244b38f6b87bd138。core32件はexact一致、説明は28比較UNRESOLVED／4比較欠測・完了0。保存SE転記のみ、retry0、mandatory STOP。


## 2026-10-19 Mechanism説明側：診断と修正準備

[原因・SOURCE・人工検査・未承認契約](docs/research/track_a_mechanism_explanation_diagnostic_readiness_20261019.md)。SOURCE b6716396acc5760fde7cf7cc6f2481b183a08f10。core不変、63 side field存在／1 side未回収、人工105件PASS、実科学数値再解析0・mandatory STOP。


## 2026-10-19 Mechanism説明側：one-shot完了

[結果・条件・監査](docs/research/track_a_mechanism_explanation_results_20261019.md)。SOURCE b6716396acc5760fde7cf7cc6f2481b183a08f10。63 side／31比較で説明完了、A82_OLD_PR.PRのみ既知欠測。元core32件不変、数値不一致・追加欠測0、mandatory STOP。


## 2026-10-19 Track A Mechanism：Figure 2準備・A82限定回収

[統合入口](docs/research/track_a_mechanism_paper_integration_readiness_20261019.md)。保存core32／説明31比較62軸を転記し、人工26テスト・人工描画PASS。A82原signalはreceipt SHA同一で回収、数値再解析0・旧欠測保持。H4 D完了／PF代表10/10不変を別来歴へ追記、旧PRE_D_P数値・P未完了・最終freezeなし。local commit後mandatory STOP、公開push0。


## 2026-10-19 Mechanism：Figure 2実図・H4 P統合・A82限定SOURCE

[入口](docs/research/track_a_mechanism_figure2_a82_readiness_20261019.md)。固定32比較／31比較の保存値からFigure 2A/B（SVG/PNG/PDF）。H4 P4,185軌道／8,370 compile、主10＋診断4点方向再現を別cohortで追記し、旧core・P_PENDING履歴不変。A82 SOURCE a8961c45…、描画3de17acd…、人工158PASS。実A82解析・新統計・sampling・compile・push0、最終freezeなし、mandatory STOP。


## 2026-10-19 Mechanism：A82.PR限定one-shot完了

[結果・統合・STOP](docs/research/track_a_mechanism_a82_validation_results_20261019.md)。SOURCE a8961c45d7fd0310c1e34c50ffd004dd58898247、PR cosine/sineだけ一回PASS。旧63 sideを継承してside検証64/64、比較因子は保存31比較62軸のまま。core32・図116点・H4 P別cohort・旧欠測履歴は不変。公開承認未確定でpushなし。新比生成・統計・sampling・compile・最終freeze・詳細科学レビューなし、mandatory STOP。


## 2026-10-19 Mechanism：A82比較因子の実行準備

[契約・SOURCE・人工検査](docs/research/track_a_mechanism_a82_factor_readiness_20261019.md)。SOURCE 06c94745f50345b2d93714760608d9d11b83e784、保存sideを読む1比較2軸の因子専用runner、人工38件PASS。SOURCE5/入力14/runtime68を照合、旧363 file不変。実科学因子0、有効grant0、比較因子31/32を保持。別途実行承認までmandatory STOP、公開承認不明でpushなし。既存図/cohort・H4/H6並行job・protocol v1・最終freeze/詳細科学レビューへ進まない。


## 2026-10-19 Mechanism：A82比較因子one-shot完了・説明32/32

[結果・全因子表・統合](docs/research/track_a_mechanism_a82_factor_results_20261019.md)。SOURCE 06c94745f50345b2d93714760608d9d11b83e784、A82一比較2軸だけ一回PASS、親/worker exit0・回収。旧31を転記し32比較64軸の説明因子が完備、side64/64を継承。core/side/他31再計算0、旧381 file SHA一致、図/cohort・欠測履歴を保全。公開承認未確定でpushなし、最終freeze/詳細科学レビューなし、mandatory STOP。


## 2026-10-23 H2共通native・allowance技術基盤

[報告・実装・非実行契約](docs/research/track_a_h2_common_native_allowance_preparation_20261023.md)。固定4候補のGaussian/native遅延recipe、mock adapter、診断schema、人工allowance、provenance gate。人工36テストPASS、150受領file/278 provider file/67 PR taskを静的照合。実科学backend未接続、科学計算・新u/N/Gなし、政策未採択、実行準備未完、mandatory STOP。


## 2026-10-23 H2共通production基盤

[実装・契約・検証](docs/research/track_a_h2_common_production_20261023.md)。H2共通native/allowance production接続、exact3分岐、人工72件PASS。科学実行0・binding/政策/C witness未完、mandatory STOP。

## 2026-10-23 H2 DIAGNOSTICS実行binding準備

[技術封印・実契約・採択資料](docs/research/track_a_h2_diagnostics_binding_20261023.md)。固定4候補のSOURCE/inputと静的runtime許可集合を封印し、隔離preflight PASS。人工98件PASS、科学import/実行0。政策・科学承認待ち、PR C witnessと比較継続loaderは未完。local commit後mandatory STOP、push/PR/mergeなし。

## 2026-10-23 H2 DIAGNOSTICS run01完了

[限定実科学診断・結果・保全](docs/research/track_a_h2_diagnostics_run01_20261023.md)。明示承認後one-shot正常終了、固定4候補のstrict <1e-9診断4/4、exact予算。compile/sampling/u/N/G/順位0、比較はPR identityと継続loader不足でBLOCKED。前段SOURCE/結果/STOP保持、mandatory STOP。

## 2026-10-23 H2受入済み診断COMPARE専用分岐

[実装・PR identity・契約](docs/research/track_a_h2_accepted_compare_20261023.md)。新COMPARE_FROM_ACCEPTED_DIAGNOSTICSはbundle_load1で元path8/ref4を再実行しない。新SOURCE46a3b33e…、人工138件PASS。公開402 archive metadata/134 compiled raw SHAを回収・結合したが、原config/full event/旧stream/実runtimeが不足しPR単価再利用はBLOCKED。新科学実行・live grant0、原SOURCE/結果保持、local commit後mandatory STOP。

## 2026-10-23 H2保存PR単価：原証拠回収準備

[限定調査・回収tool・判定](docs/research/track_a_h2_pr_identity_recovery_20261023.md)。基準4008ce45…、SOURCE9ed3c76c4…。主要依存16 byte一致・元SOURCE70 Git一致、原67 event/402 archiveのexact requestを整備。人工174件PASS、保全491項目PASS。O00002のcanonical locator/原stream/実loadは未取得、PR再利用・COMPAREはBLOCKED。新科学計算・grant・レビュー0、local commit後mandatory STOP。

## 2026-10-23 H2 PR原本回収・134軸identity判定

[回収・identity・別契約案・保全](docs/research/track_a_h2_pr_original_identity_20261023.md)。原本O00002の期待SHAをGPUA100-gleap02で再確認し、2,554原証拠を直接照合。67 eventファイル/6,400 event/402 archive、134軸のraw・単価receipt・central/control・共通Gaussian登録fieldを結合。Qiskit830 byte一致、NumPy/SymEngine ABI差・歴史seal liblzma1差分と実worker全load欠測を区別。完全stream/runtime identityは未成立、strict再利用0/134・COMPARE BLOCKED。人工200PASS、16,804 path前後不変。新科学実行/compile/u/N/G/順位/grant/公開push0、別政策・最小probeは未承認DRAFT、mandatory STOP。

## 2026-10-23 H2 PR代替同等性契約：静的検証

[契約・構造検証・runtime・保全](docs/research/track_a_h2_pr_equivalence_contract_20261023.md)。基準588d19544…から代替契約を未採択の別schemaとして実装。科学回路A/費用測定Bを分離し、全134保存QPYのinverse/reuse/diagonal mirror/control/phase/wrapperを全instruction消費でVERIFIED。Aは11/12条件、Bは2/3条件が全軸成立。共通diagonal絶対角度の独立bindingとcross-runtime compiler費用同等性は未証明。strict/代替正式認定とも0/134・COMPARE BLOCKED。新48＋既存200＝人工248PASS、16,804保護path不変。新科学計算/実H2生成/compile/会計/grant/公開push0、未承認最小案とmandatory STOP。
