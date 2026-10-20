# PR資源評価：希少RTE event・期待compiled費用の既存証拠調査

2026-10-21 JST。状態：**DOCUMENTARY_INVESTIGATION_COMPLETE / MANDATORY_STOP**。
対象repository：`HIROMU1015/Partially-Randomized-Trotter`。専用branch：`codex/pr-rare-rte-existing-evidence-20261021`。
基点：`80b7a35c4dbf683f952293eeb55944bc10b9efc8`。固定commitをGitHubから実取得し、Git objectと照合した。

## 1. 技術的な充足判定

既存証拠で評価できるのは、固定8条件の**保存有限標本の費用・paired SE・event観測**、旧条件における条件付きpilotと感度診断、有限RTE lawの実装、固定分解モデルの理論期待値、および条件付きcompiler静的事実である。指定Pの未観測support全体について、確率・対応する費用・compiler上界の仮定を一つに結合した既存の認定済み期待compiled費用上界packetは、今回確認した範囲では充足していない。

これは証拠の技術的分類である。期待費用の順位が保証されるかの最終科学判断、新たな数学的上界の採択、GPT/Claude詳細科学レビュー、最終evidence freezeは行わない。未観測値を0にせず、既存の不利な点結果も保持する。

|成果物|内容|
|---|---|
|[条件別対応表](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/condition_evidence.md)|H2/H4/H6/H8の各2条件、固定SOURCE、次数・層・費用・欠測・A/B/C/D|
|[上界・推定の分類表](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/cost_guarantee_classification.md)|数学的関係、実装制限、proxy、観測最大、経験推定の区別|
|[来歴一覧](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/provenance.csv)|固定commit/path/Git blob/SHA256/bytes/取得範囲|
|[追加検証候補](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/additional_verification_plan.md)|未実行の最小候補、必要な情報と別承認境界|
|[保全・構造・終了監査](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/completion_audit.md)|科学実行なし、旧証拠保全、検査の実範囲、STOP|

調査は既知repo内の文書・SOURCE・artifact索引と全refの関連commit履歴を対象にし、85固定fileを来歴台帳へ収録した。全blob読了やprivate原rawの新規取得、未発見資料の不存在は主張しない。[検索範囲と限界](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/search_scope.json)を参照。

## 2. 有限RTEの確率法則：定義と仮定の証拠

以下の式は既存規約・SOURCE・既存rare設計にある式の転記であり、新しいboundの導出や数値評価ではない。[CONVENTIONS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_conventions.md)、[RTE_SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trotterlib/rte.py)、[NATIVE_COST](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/pr_cost_native.py)。

### 2.1 Tailと次数0/2/4

`H_R = λ_R Σ_l p_l P_l`、`λ_R = Σ_l |h_l|`、`p_l = |h_l|/λ_R`。係数の符号はHermitian involution `P_l`へ吸収する。DF fragment全体を一つのinvolutionと仮定しない。exact I/Z/ZZ展開のλ_RとDF選択proxyは別物であり、identity scalarをdeterministic baselineへ移した版ではtailとhashもその政策へ結合する。

physical tail time `t_R`、event step数`r`に対して `Δt=t_R/r`、`τ=λ_R Δt`。今回の二次PRでは外側`q` step、`δ=T/q`、各stepのtail occurrenceにr eventがあり、`t_R=δ`、`R=q r`、`τ=λ_R T/(q r)`。`scripts/pr_cost_native.py:steps`がこの接続を実装している。qはTaylor次数でも量子shot数でもない。

偶数次数nについて

`a_n(τ)=|τ|^n/n! × sqrt(1+τ²/(n+1)²)`、
`B_K(τ)=Σ_{n=0,2,…,K} a_n(τ)`、`p_n=a_n/B_K`。

従って0/2/4のweightはそれぞれ `sqrt(1+τ²)`、`|τ|²/2! sqrt(1+τ²/3²)`、`|τ|⁴/4! sqrt(1+τ²/5²)`という既存一般式の対応である。K2のn4は支持外。K4で観測0のn4は支持内未観測として区別する。τ=0の高次weight0と、正のweightを浮動小数で失う事象も区別する。今回τやp_nを新たに数値化していない。

次数nを抽選後、rotation component lとproduct components l_1,…,l_nを元p_lで独立に選び、event確率は `(a_n/B_K) p_l Π_j p_lj`。実装はcomponent probability和を検証後に正規化している。`_make_event`の入力index0がrotation、適用順はproducts→rotation、位相`(-1)^(n/2)`、角`+atan(τ/(n+1))`である。符号付きproduct phase・rotation angle・identity phaseを費用上の無操作と見なさない。

### 2.2 複数event・trajectory層

同一tail/τ/Kを各eventでIIDに使うモデルでは、trajectoryの次数wordの法則は積確率、count `(N0,N2,N4)`はmultinomial、位置は同count内で一様である。既存資料のR=q rは**event総数**であり、文書によってoccurrenceの呼称がevent位置を指す場合がある。今回のP call台帳のoccurrenceは外側q単位なので、台帳と次数式の単位を混同しない。[H6_RARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)、[H8_RARE_DESIGN](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_h8_rare_tail_sampling_design_20261018.md)。

排反層はS0_ONLY=`N2=N4=0`、S2_SINGLE=`N2=1,N4=0`、S2_MULTI=`N2≥2,N4=0`、S4_PRESENT=`N4≥1`。旧資料のs2はS2_SINGLE＋S2_MULTI、s4は他のorder2の有無を問わないS4_PRESENTである。

既存式：`π0=p0^R`、`π2=(p0+p2)^R−p0^R`、`π4=1−(p0+p2)^R`。singleとmultiは既存count lawにより分割する。H8既存設計はlog重み・multinomial count・一様位置・元component法則で条件付きsampleを作り、s4をN4=1へ、multiをN2=2へ強制固定していない。heterogeneousなtail/time/r/Kでは共通pのmultinomialを無条件適用できず、各eventの積法則に戻る。

### 2.3 Normalization、数学的関係、検証範囲

`B_K E_K[U]`はK+1までのTaylor多項式であり、全trajectoryのnormalizationは同一条件なら`B_K^(q r)`、異種occurrenceなら`Π_i B_Ki^ri`。これはsampling費用の平均へ掛け直す重みではない。attenuationはsignal/shot会計側に別保存され、Taylor biasと区別する。[CONVENTIONS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_conventions.md)。

既存の`B_infinity ≤ exp(τ²)`、step Taylor residual、occurrence/roundの誤差伝播boundは、normalizationまたはoperator誤差に関する数学的関係であり、compiled RZ/CXの費用上界ではない。

|式がSOURCEにあること|その式を今回のPへ使うための条件・既存検証の限界|
|---|---|
|finite lawとcanonical fieldのvalidator|整数/even K、weight/B/残差、component確率和等の実装検査。binary64を厳密実数確率のinterval certificateにはしない|
|Hermitian involution・tail展開|Z/ZZ・sign/identity/basis規約と保存preparation binding。少数control/oracle PASSを全compiled分岐の上界検証へ拡張しない|
|独立抽選による積法則・multinomial|sampler/APIの生成規則に関するモデル。seed非衝突とcohort分離は数学的確率独立性の証明ではない|
|全層期待費用分解|各層の質量と対応するfull-wrapper条件付き費用が必要。次数0内でもcomponent/basisはランダムで、S0標本平均を既知母平均にしない|
|有限supportのexact列挙API|単event全列挙の実装・局所保存結果はあるが、指定Pのwhole-wrapper全trajectory列挙結果ではない|

## 3. 固定8条件の保存証拠

共通条件はlinear H-chain、隣接1 Å、STO-3G、tol-only DF1e-8、T=.8/P=0、通常signed bias・allowance倍率1。DF実rankはH2=3、H4=10、H6=19、H8=28。H2/H4のdense state solverとH6/H8の記録上eigsh、正規化・残差gateの差はv2のまま保持する。既存v2§1–9のH2 P未実行という記述は履歴で、§10の完了・受入が現行である。[PROTOCOL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_pr_resource_evaluation_protocol_v2.md)。

全Pでfull measured cosine/sine Hadamard wrapperのcompiled metricsが保存されている。Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing未指定、symmetric_directional、scalar一回・reference不回転、状態準備P=0。同じoptions名でもruntime/input/circuit identityの同一性を自動認定しない。

G_iは保存shotによる`N_cos C_i,cos + N_sin C_i,sin`。保存平均とSEの対象はこのpaired trajectory費用であり、単eventの費用、n本の合計費用、実量子shotの測定誤差、サーバーwall費用ではない。RZ以外にCX/depth/sizeも保存されるが、保存GはRZ×shotでありCX総期待費用を追加計算しない。

- H2：q1 n80ではS0 79、S2_SINGLE 1。q4 n12はS0のみ。公開projectionのlaw確率・未観測費用・分散はnull。RTE eventは2条件合計256の保存値を引用し、条件別event総数を新計算していない。[H2_REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/docs/research/track_a_h2_p_completed_20261019.md)、[H2_trajectory_receipts_projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/trajectory_receipts_projection.json)。
- H4 DF10：q1/K2 n1108でS0 1071・single37、q8/K4 n78でS0 71・single7。multiは両条件未観測、K4のS4も未観測。order0/2/4の保存件数とtrajectory数を別欄に置いた。[H4_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/result.json)、[H4_event_label_audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/event_label_audit.json)。
- H6 p4：q2 n48のeventはorder0=374/order2=10、q8 n104は3327/1。公開aggregateにはS2_SINGLE/MULTIのtrajectory別内訳がないので新たにgroup集計しない。K2なのでS4は支持外。[H6_result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json)。
- H8：p8/q2とp7/q16の各n48はorder0が384/3072、order2/4観測0。探索・旧確認・pilotとのpoolingなし。p8 replica0の同じ確認cohortの完了1本を再利用した実行SOURCEと、後続実行SOURCEを区別する。[H8_REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md)、[PACKET_01_results_and_source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a6246f03c6297438ab624fc62e5061a58e53145f/artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/01_results_and_source.json)。

### H8公開監査の更新を過去記述に優先する

e6eea538の結果公開時・v2本文ではPR詳細監査や原profile未確認という履歴が残る。a6246f03のredacted packetには元result/worker/native/seed/grant/ledger/STOP/input/runtime対応が追加されている。匿名化により受領者が原PID/seed/private bytesを再照合できる範囲は限定される。コード条件、保存receipt、原bytesの再監査は別の証拠である。[PACKET_09_missing_evidence](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a6246f03c6297438ab624fc62e5061a58e53145f/artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/09_missing_evidence.csv)、[PACKET_10_source_conditions](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a6246f03c6297438ab624fc62e5061a58e53145f/artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/10_source_conditions.md)。

5f267e96の補足により**原契約内/profile JSONが取得済み**となり、canonical JSON digest `622ce4a8…`と両scopeのruntime map対応が確認済みという状態へ更新されている。standalone historical fileのSHAと同じdigest domainではない。古い「profile未取得」を現在の欠測として繰り返さない。一方PR回路本体は調査範囲で0/192軸取得、fingerprint再導出と旧compiler output bit同一性は未確認のまま。今回原private契約全体やhistorical runtimeを再取得・再実行したわけではない。[PROFILE_01_original_profile_receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f267e96ea313f373eb78a75040a43fac3a2ee1b/artifacts/resource_applicability/track_a_h8_original_artifacts_final_search/2026-10-20/share_candidate/01_original_profile_receipt.json)、[PROFILE_05_missing_status_changes](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f267e96ea313f373eb78a75040a43fac3a2ee1b/artifacts/resource_applicability/track_a_h8_original_artifacts_final_search/2026-10-20/share_candidate/05_missing_status_changes.json)。

## 4. 再利用する既存研究と移送の限界

|既存成果|評価済みの対象|今回への利用・限界|
|---|---|---|
|H6 rare-order解析 eee1343f|旧p5、q1/2/4 r4・q8 r2、K2/K4、探索576/確認256の保存metadata|解析確率、histogram欠測、条件付き逆転感度、native instruction guardの限界を再利用。今回p4/r4 Pの確率・費用へ移送しない。[H6_RARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)|
|H6 s2 pilot|p5/q8/r2/K2、n8全てN2=1、条件付き平均・既存層別点推定|既存の取得と解析を再実行しない。今回p4/r4とは別法則・別回路。[H6_S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/references/track_a_h6_conditional_s2_pilot_result_v1.md)|
|H6 DBA B完了|p5/q8/r2、K2 multiとK4 s4各n8|取得済み。K4 multiは未取得、今回Pへの直接代入不可。[H6_DBA](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/648184eb2a6f1c2406df17c1796eb5cc195908a8/docs/research/track_a_h6_dba_20261018.md)|
|H8 s2 run02 b21e90b8|p7/q4・q8/r4/K4、各n8、実現は全てN2=1|whole-wrapper実測とlaw receiptがある。p8/q2またはp7/q16を被覆しない。[H8_S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b21e90b8368645a5066c06a2a0a2ceb2850d162e/docs/research/track_a_h8_conditional_s2_run02_results_20261018.md)|
|H4/H8 DBA B＋後続B診断|H4はlegacy p3/q4,8、H8はp7/q4,8のmulti/s4各n8|S4/multiが常に未取得という説明は誤り。公開B診断は完全な層質量receiptと通常標本joinが不足し無条件平均null。H4はDF12をDF10へ読替えない。[B_DIAGNOSTIC](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/b_diagnostic_scope_summary.json)、[B_COMPLETION](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_pr_cost_batch_completed_results_20261018.md)|
|H8 rare sensitivity/selection cf8e93ee系|p7/q4,8の40条件中32境界、union6 n32の探索的bootstrap2000|未知multi/s4平均は変数/null、S0/single標本平均と一部model対照に条件付く。後続B取得で費用情報が増えても自動で母平均保証へ更新されない。[H8_RARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md)|
|H4 MC defa5f6d、表示295b93d5|今回DF10独立Pと同じ標本、保存累積平均・bootstrap・S0/single費用|同一cohortなのでそのまま参照可能。bootstrapは未観測multi/S4を生成せず、保存min/maxは全support上界でない。[H4_MC](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/295b93d50662a0f8af3ec469df43099a3f04af60/docs/research/track_a_h4_pr_cost_variability_20261019.md)|
|finite-RTE exact/MC費用API、旧H4/H5 cluster検証|単event exact期待値、短sequence全体compileと加法比較、局所calibration/holdout|full Pの全trajectory exact期待値ではない。event別compile和はwhole compileと非加法で、proxyの5%や点wise正規診断を厳密boundにしない。[EVENT_API](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_compiled_event_cost.md)、[COST_HISTORY](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/rte_compiled_cost_validation_summary.md)|
|理論A/B・native mixedとcompiled比較|旧H6/H8 832理論行、144 law、H4/H6固定7候補比較|既存有限law理論期待値はある。特定分解・basis再利用・phase幅に条件付きで、実compilerの上界ではない。[THEORY_REUSE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_v2_existing_theory_reconciliation_20261016.md)、[THEORY_COMPARE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e454b91185ed7b0f92a3790f800ba5d65d5f156/docs/research/track_a_pr_theory_vs_compiled_completed_20261018.md)|
|100 scope PF単価統合 80b7a35c|重複scope100/unique40、確認union8/6 n32、PF単価政策の条件付き比較|独立Pの再評価でもrare-support証明でもない。PF価格側の補強をPR未観測層の解消へ読み替えない。[Q100](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/80b7a35c4dbf683f952293eeb55944bc10b9efc8/docs/research/track_a_pf_q_model_evidence_integration_20261020.md)|

[完了済みH4/H6/H8条件付きpilotの保存値一覧](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/historical_pilots.md)には、後続完了commit97d25cc4/648184eb/839c4030の平均・SEを追加した。準備時や初期sensitivityの「未取得」を後続取得の不存在と誤読しない。

## 5. 上界を混同しないための判定

[TIER_R](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf8c19057fd82c900c6234152e152e79cb52e707/artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md)の既存L1は、target/error map/routing/direction/scheduling追加がない固定pipelineで、**完全なbasis translation直後の総gate数T**をlevel1最適化が増やさず最終RZ≤Tとなる条件付きコード事実を確認したもの。native instruction数や理論RZ数をTへ置換できない。

L2は固定replacement mapのtemplate長加法性まで確認しているが、全native gate familyの選択rule・再帰展開・mutable SessionEquivalenceLibrary・runtime/Rust identity・全input alphabetの閉包は未確認。native controlled-RZZはCCX–CRZ–CCX経路であり、理論モデルCRZZのCX–CRZ–CXとは異なる。既存静的監査は数値Tier Rを評価・認定していない。この監査を今回の8条件へ機械的に継承しない。

`max_instructions`等のguardは実装上の制限、保存最大値は有限標本の記述量、理論A/B/mixedは解析的モデル、MC/層別/感度は経験推定または仮定付き評価として分類した。全supportが正常compileされることやfailure branchの費用定義が未結合なら、成功枝のcapだけで無条件費用を閉じない。詳細は[分類表](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/cost_guarantee_classification.md)。

## 6. 不足と次段階

Aは保存点・既存解析の引用および既存数学的関係の適用範囲の整理まで。指定Pの未観測寄与を数値評価するには、BとしてP固有のtail/law/normalization・層partitionと保存costの結合を準備する必要がある。今回Bは実行していない。

Cは、Bと既存metadata照合の後でも同一候補・同一compilerの必要層費用が不足する場合の候補である。新しい条件付きwhole-wrapper取得、保存回路がない場合の復元/生成/compileは科学実行の別承認を要する。有限pilotは全support上界の代替ではない。

Dは、厳密期待費用保証に必要なcompiler閉包・費用上界・丸め/確率独立性/統計推論の仮定が未確定な部分である。Dを新しい上界の採択で埋めない。8条件それぞれの具体的なB/C/Dは[対応表](../../artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/condition_evidence.md)に記した。

ユーザーが次に判断する内容は、保存有限標本benchmarkの記述で十分か、どの固定条件についてBの保存値解析まで進めるか、さらにCまたはDの検証を必要とするかである。Codexは新しい科学計算・詳細科学レビュー・最終freeze・pushへ進まず、専用ローカルcommitと終了監査後に**mandatory STOP**する。
