# RPE総資源評価：shot・誤差配分側の接続準備

2026-11-10 JST。**静的な技術調査のみ。新しいN・signal・総費用の数値再解析は実施していない。公開・照合後mandatory STOP。**

再利用するshot算術・decoder・会計器は既にある。一方、現在のH4 DF10／H6 DF19の有限時間ε・bias・B・Nは、そのままRPE全ラウンドのenergy精度・成功条件にはならない。必要なのは新しい汎用solverではなく、対象state／phase／radiusの資格、round時間と回路構成の対応、今回採択する誤差・失敗確率配分である。

基点は共通interfaceの固定commit `1f5ea971f8ceac54f2dc00e71824ab8dabf95837`。既存接続branchは別作業との競合を避けて前進させず、`codex/resource-eval-rpe-shot-error-connection-20261110` に本報告だけを保存する。q依存価格モデルのfit・調査、アルゴリズム改善候補の再調査は担当外とした。

## 1. 再利用できるコードと既存条件

| 再利用先 | 役割・適合条件 |
|---|---|
| [rpe_resource_accounting.py][account]：`_hoeffding_shots`、`RPEErrorAllocation` | 座標余裕εと軸別αからceil付き十分N、PF/RTE/statの位相予算と軸別failure配分。独立した±1測定の算術を再利用できる |
| 同：`evaluate_rpe_round_candidate`、`build_rpe_resource_summary` | finite-RTE・PF・N・費用・unionの旧接続。**有効PF固有状態・未減衰単位半径モデル**とcost providerを伴う。DF10/19へそのまま呼ばない |
| [track_d_original_h_accuracy.py][original]：`OriginalHTarget`、`OriginalHRoundEvidence`、`assess_round`、`assess_energy` | 元Hへのcorrected operator誤差を一回含めてraw誤差・radius・phase・十分Nへ接続。元H厳密ground／alias／law／IIDの明示前提が必要。数値pilotとrigorous boundを区別する |
| [c1_stage3_accuracy.py][accuracy]：`_assess_raw_signal`、`_assess_energy_interval` | 上記の共通幾何・shot・horizon算術。内部helperだけを裸で呼んでstate資格を省かない。own-DF用`assess_round/assess_energy`と元H adapterは別のDF会計 |
| [rpe_target_round_horizon_validation.py][horizon]：`required_rpe_round_horizon`、[rpe_four_round_phase_validation.py][decoder]：`reconstruct_rpe_phase` | energy目標とbase時間からdyadic horizon、連続roundのnearest-branch復元。decoderへ真のE0を渡さない |
| [research_direction_ablation.py][allocation]：`_weighted_alpha`、`evaluate_ablation_scenario`、[research_direction_decision_cost.py][decision]：`_grid_search` | 旧18-roundのα・β・r/K共同選択。固定座標余裕で重みは`2 E[C]/ε_coordinate²`。旧runner全体はHamiltonian準備や価格providerを起動し得るため実行しない |
| [d0_frozen_shot_rule_v1.py][finite-shot]：`axis_headroom`、`u_aware_shots`、[run_pf_common_run.py][qualify]：`qualify_record` | 現H4/H6の有限時間signal精度用。bias・u・log B margin・headroom・整数／log-domain・三状態判定を再利用。ただしRPE energy／branch資格を与える関数ではない |
| [h46_long_time_signal.py][signal]：`evolve_batch/qualify`、[h46_full_h_pure_rte.py][pure]、[h46_full_h_t20_opt.py][t20-code] | 保存DF10/19の有限mean、raw/corrected、scalar、B、allowance。次回必要な時間・構成に限り再利用候補。`context/setup`等は科学処理を含むので今回import・実行していない |
| [track_d_resource_interface.py][interface-code]：`ScientificInputs/ControlProfile/MethodPrices`、`aggregate_supplied`、[c1_stage3_accuracy.py][accuracy]：`aggregate_totals` | 科学条件とcontrol価格を分けた既存のΣNμ会計。`SavedCDF8Study`は固定旧結果のread-only入口。現adapterは**X/Y共通のscalar N**であり、一般の軸別Nをまだ扱わない |

次の数値は**固定報告からの引用**で、今回の正式条件として採択しない。

| 既存研究 | round・時間 | 配分・資格 |
|---|---|---|
| [legacy H4 DF12・最終coherent opt2／V2][legacy] | linear H4/1Å/STO-3G、4電子sector70、Partial LD3／Full LD12、δ=.02、m0..17、q1..131072、T_m=.02q。Partialは保存r/K、Full r0/K0 | εE=.00015936001019904 Ha、βRPE=.4、αtotal=.05。cost感度α、保存された選択β/r/K/Nを固定。元資源式は単位半径仮定。V2は物理保存平均で条件付き成功を確認し、半径等の未保証を保持 |
| [Track D B4-2][b4] | 同H4/1Å/STO-3G、number-CDF8／元順LD2、元H ground仮定、sector36、S2/s1/r32/K0、h=.1、m0..12、T_m=.1·2^m | εE=.00159 Ha、βtotal=π/6、βstat=π/12、αaxis=.01/26、αtotal=.01。保存十分N合計14,738。ideal finite-RTE IID/exact native、B1数値operator norm＋guardに条件付き。最小N・actual成功保証ではない |
| [Track D B/C全round会計][bc] | 上記B4の26軸と同じ全13round、Strong／whole-path Reference、状態準備費除外 | B6b理想測定同値性の前提下で保存Nを共通使用。保存A/F/S価格を使用。長q proxy・全体不確かさ未資格を保持。CDF8のN／価格／総値をDF10/19へ移送しない |
| [現DF10/19・pure RTE固定点][light]／[T=2.0評価][t20] | linear H4/H6・1Å・STO-3G、tol-only DF10/19、保存state／fragment順／weighted-Hermitian completion。T=.8又は2.0、pureは決定論blockゼロ | 共通u floor=1e-9、αaxis=.025、log B margin／headroom政策。εはcomplex signalの精度で、energy精度ではない。biasとallowanceは経験的、保存Nは有限時間用 |

SOURCEの役割も区別する。legacy carrierは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、V2は`78c49d0b5f1d71e63b5959e78813237923bbc648`。B4結果は`07d18bfe1d5145d6f8de026c0d994d5cb1c783c8`、B/C結果は`8828b298975c88a2c0d1fd57ff6ed905233840f5`（科学実行SOURCE `b09b7e7abaa3b817c26657974a62c3bd3047af89`）。DF10/19の公開carrierは`9b83b96b83f62142c9d87858dc24a94db7ac5d61`／`7a8bc7f132e166390c943e6b32c92cc6abe1baf4`。各carrierをcleanな元実行SOURCEと同一視せず、各SOURCE_INDEX・既存実行履歴を残す。

## 2. RPE全ラウンドに必要な入力

| 入力 | 各roundで必要なもの・確認点 |
|---|---|
| 共通target | 元H又はown-DFのどちらのenergyを狙うか、εE、base時間h、energy prior／alias、共通Eの位相規約、decoder、最大m。13/18roundはこれらから決まる過去例 |
| 科学binding | H／snapshot／DF・rank政策／sector／state、fragment順、係数閾値、D/R/drop、一体項の所属、scalar/identity、law・normalization・SOURCE |
| 時間・構成 | m、`q_RPE=2^m`、`T_m=h q_RPE`、round内PF step数`q_PF`、`δ_m=T_m/q_PF`、公式・順序・負時間piece、RTE event数／piece、K |
| 信号 | 同じ入力stateのreference、corrected mean、raw mean、Re/Im、bias分解。reference自体の位相が共通Eに従う資格と基準半径下界が別に必要 |
| normalization | 全roundの`L_m=log B_m`、そのmargin／上界、`A_m=exp(-L_m)`。paired TaylorのKは最大偶数event次数、numeratorはK+1次。負時間pieceにも元の積を用いる |
| 誤差allowance | corrected信号の軸別u、reference側誤差を含むか、PF／切捨て／有限RTE／DF／projection／state／native・compile／finite-bit lawの範囲・単位・資格。観測bias＋uをrigorous boundに昇格させない |
| phase・radius | 正のraw半径下界、deterministic位相誤差、βPF/RTE/stat又はtotal/stat配分、branch条件。旧式のstrict dyadic条件はβtotal<π/3 |
| failure・測定law | 軸別α_m,bと全体αtotal、軸内fresh independent trajectory／測定、occurrenceの積law。保存価格用paired cohortやseed非重複だけではIIDの証明にならない |
| 費用との対応 | 同じ構成・law・whole-interrogation単位のmethod/axis別期待C、compiler／cut／C1／phase、cohort・model IDと適用時間域、幅・共分散・外挿資格 |

**Bとbiasの単位を揃える。** 現行paired-Taylorの保存定義は `z_raw = z_corrected / B`。±1測定の平均はrawであり、補正済みcomplex値をBernoulli平均にしない。Bにはround全体のevent積を含め、qやRを二度掛けない。Fullのtail-freeではB=1。Closed P3のweighted estimator/rangeは別lawであり、同じ「B」という名前だけで代入しない。

有限時間会計の既存式は、各軸bについて

\[
H_b=\epsilon_{\rm signal}/\sqrt2-|\Delta z_{{\rm corrected},b}|-u_b,
\quad
N_b=\left\lceil {2B_{\rm upper}^2\over H_b^2}\log{2\over\alpha_b}\right\rceil.
\]

`H_b>0`に加えて元のu/headroom判定・log-domain条件を維持する。このεsignalをεEへ名前だけ変えて流用しない。

B4系の元H接続は、単位複素数`exp(-iT_m E0)`に対するcorrected operator誤差e_mが与えられる場合に、raw誤差を

\[
D_m=A_m e_m+d_{\rm implementation,raw}+d_{\rm law,raw},\quad
\rho_{m,\rm lb}=A_m-D_m,\quad
\beta_{m,\rm det}=\arcsin(D_m/A_m)
\]

とし、`ρ>0`、`βdet+βstat≤βtotal`の下で

\[
\epsilon_{m,\rm coordinate}=\rho_{m,\rm lb}\sin\beta_{m,\rm stat}/\sqrt2,
\quad
N_{m,b}=\left\lceil{2\over\epsilon_{m,\rm coordinate}^2}\log{2\over\alpha_{m,b}}\right\rceil
\]

を使う。これらは**既存SOURCEの式の整理**で、今回の数値評価ではない。DF10/19の保存軸biasはoperator normではないため、`OriginalHRoundEvidence.operator_error`へそのまま入れられない。state-specific diskを用いる場合は、軸bias/uからのdiskとreferenceのphase/radiusの正当化を含む小さい専用adapterが必要になる。

全体では`Σ_m,b α_m,b≤αtotal`と全roundのphase／branch条件を確認する。energy幅は、元H誤差を信号に含めるlaneなら`βtotal/T_M`、own-DF laneなら別に資格付けたDF energy差を一回加える既存方式。DF差を信号とenergy reserveの双方へ二重加算しない。

### αaxis=.025の位置付け

`.025`は既存有限時間一候補のRe/Imに配る座標失敗上限。RPE全round・全候補・全εのfamilywise条件ではない。RPEで一様配分を採るなら`α_m,b=αtotal/[2(M+1)]`であり、非一様なら全軸の和を管理する。union bound自体にround間独立は不要だが、各軸のHoeffding／二項モデルの独立trial条件は必要。

αを変えると同じbias・Bでも十分Nは再算定対象になる。βを変えると座標余裕とhorizonも変わり得る。固定信号・固定βでのα変更ならsignal再計算は不要だが、NとΣNCの再解析には別途実行許可が必要。

## 3. 現H4 DF10／H6 DF19と価格担当への接続

最も重要な名前の衝突は**保存有限時間のqとRPEのqが違うこと**である。現`candidate.q`は固定TのPF step数。本報告では`q_PF`と呼ぶ。RPEの`q_RPE=2^m`と区別し、時間が同じという理由だけで両者を同一視しない。有限時間gridには非dyadic qもあり、T=.8／2.0の集合も連続dyadic horizonではない。

既存`FH20_…` IDは改名しない。価格担当には次の対応表を渡す仕様とする。今回はschemaの整理までで、新しい数値joinや価格表生成はしていない。

| 共通欄 | 対応・責任 |
|---|---|
| `source_commit, candidate_id` | 保存IDをその固定SOURCE／carrierと組にする。既存`FH20_…`はT=2.0 namespaceなので、別時間へIDだけを流用しない |
| `system, H/DF/state binding, family, split` | DF10/19とDF12/CDF8を区別。B3のprefix0は一体項をDに残す。pureのprefix0はDゼロなのでfamilyとpreparation/lawも必須 |
| `T, q_PF, delta, formula, order, R_outer, K, s` | 現保存欄`T,q,delta,formula,R_outer,K,s`を対応付ける。pure S2で`D=∅`のときのみ保存signalはs=q_PF·RとKで縮約可能。**価格同値性は保証しない** |
| `schedule_id, m, q_RPE, base_time` | 新RPE schedule側の欄。既存時間と構成が完全一致するときだけ保存signalへ対応させ、未評価roundは欠測のままにする |
| `target_id, allocation_policy_id, axis` | energy/state/lawのtargetとα/β配分を分離。Nはこのpolicyに従属。Re↔cosine↔X、Im↔sine↔Yを測定符号込みで対応 |
| `N_m,b, shot_origin, signal_origin` | 保存有限時間N／保存B4 N／新規RPE十分Nを区別。今回、新Nは空欄。MC価格標本数n、coherent step数、event数をNへ重ねて掛けない |
| `control_method, cost_model_id, cohort_id, compiler, scope, coverage` | q価格担当から受領する欄。Cは1回の測定付きHadamard wrapper期待資源で、PF誤差係数C_PFと別記号。価格modelのqがどちらのqかも宣言する |
| `C_m,b,g, uncertainty, width` | RZ/CX/各depth/sizeを資源gごとに受け取る。Nと結合後の`G_g=ΣN_m,b C_m,b,g`は将来の許可範囲。logical幅は最大値で、depth和は実機wall timeではない |

保存済みIDの具体例（既存best_configurationsのε=.0005行の構成を転記。新RPE候補の採択ではない）：

| 保存ID | 系／時間／構成 | 保存価格cohort |
|---|---|---|
| `FH20_21a628443351f066ab13` | H4 DF10、T2.0、q_PF8、R448、K2、s3584、pure S2 | `FULL_H_T20_EXPLORATION_V2`。独立H4確認cohortは別に保持 |
| `FH20_6cb9f6bff5bd73350a39` | H6 DF19、T2.0、q_PF32、R448、K2、s14336、pure S2 | 同探索cohort。H4確認標本とpoolしない |

T20のSOURCE_INDEXのbinding内`T=.8`は入力reference由来の履歴で、評価時間は別欄`evaluation_T=2.0`。結合にはcandidate.Tを使い、H/state bindingの時刻欄だけで取り違えない。snapshot/stateの保存hashは[lightの入力要約][input-summary]と[T20 SOURCE_INDEX][t20-index]を参照する。

現在の保存stateは数値固有状態であり、既知receiptに`ground_state_certified=False`がある。小さいresidualやstate hashは重要だが、これだけで元H厳密ground・全時間unit radius・共通EのRPE保証を認定しない。元H/own-DFの選択と、state／DF／projection差の扱いを今回改めて決める必要がある。今回receiptの該当metadataを読んだのみで、vector・eigenpairの再解析はしていない。

既存interfaceの`aggregate_supplied`はX/Y等N、元H targetは全round一様αという現制約を持つ。非対称u/biasや非一様αを扱うなら、旧CDF8 reader・旧guardを変えず、新namespaceでpolicy表と軸別Nを渡す最小拡張が必要。すでにある[共同最適化準備][joint]もこの制約を記録しており、同じsolverを重複実装する必要はない。Strong／ReferenceのN共用は同測定law・normalization・全使用domainの同値性が前提で、DF10/19には未移送。

## 4. 保存値で済む範囲と未確定の科学条件

**今回、保存値とSOURCEの静的読取だけで済んだもの：** 旧13/18-roundの設定・式・保存N、現DF10/19の候補ID・時間・law・信号欄・u/B政策・cohort・compiler・SOURCEの対応、既存テストの対象確認。保存数値の再集計、再ranking、phase変換、N算定はしていない。

**次回、保存値だけで計算可能な範囲：** 同じT・state・law・q_PF/R/Kの保存signalとallowanceを用いたα変更・座標余裕／Nの算定、資格が揃った後の凍結価格とのΣNC。ただし、これは新しい数値再解析であり実行許可が必要。現保存有限時間N自体をRPE十分Nとして読替えることはできない。

**追加signal／誤差評価が必要な範囲：** 採択したT_mに一致する保存値がないround、q_PF／R/K／分割／state変更、元Eへのreference位相・半径／誤差資格が不足するround。既存`.8/2.0`信号から位相を冪乗・補間して補わない。Hamiltonian生成は既存保存入力を使えば通常不要。stateの条件を変更する場合は独立した範囲として扱う。

未確定なのは、(1) energy targetが元Hかown-DFか、state／DF／projection誤差と理想／actualの主張水準、(2) εE・αtotal・β配分・base時間・prior・horizon、(3) 比較する有限候補とround別構成の自由度、(4) 一様／非一様αとX/Y等配分の採否、(5) 主資源と価格modelの適用domain・許容不確かさ。既存13/18round、α=.05/.01、CA/10等を今回の正式条件に自動採用しない。

## 5. 最小限の数値評価計画（未実行・未承認）

1. **ユーザー／研究方針担当**が上記未確定条件と有限候補を決める。**q価格担当Codex**の成果を固定commit・model ID・科学binding・時間／qの定義・coverage付きで受け取る。本報告は価格modelの採択や再fitを要求しない。
2. **Codex**は既存shot helper・decoder・会計器に薄いpolicy入力を接続する。人工fixtureでraw/corrected、q二種、軸別ceil、α和、欠測round、単位・DF一回、等N制約を確認する。旧実行契約・固定readerは保持する。
3. **実行許可後の最初の科学評価**は、H4/H6各一つの保存構成・保存T（例としてT2.0）で、同じsignal/bias/u/log Bを使う一roundのradius／phase／N接続に限定する。新H生成・sampling・build/compileなし。state/reference資格が不足すれば条件付き又は未資格として止め、単一roundから全RPE energy成功や費用優位を主張しない。
4. その結果と時間coverageを報告し、全roundへ進める実行範囲を確定する。必要な追加T_m／構成のfinite mean・誤差評価だけを既存`evolve_batch`等で行う。保存値と完全一致する点は再evolve不要。原本を保持したfresh出力を使い、normalization・scalar・phase・radiusを確認する。
5. 全roundの十分N・α union・branch／energy条件が揃った場合だけ、同じ候補／policy IDで価格担当の凍結Cへ結合する。軸別`ΣNC`、各資源とpeak幅、価格coverage／不確かさを報告する。未価格化点・未知law/実装誤差・未観測tail費用を0補完しない。公開・照合後mandatory STOP。

初回は小さい保存scalar表だけで判別する。全roundの新signalが必要な段階になってから、既存のbatch化・共通spectral作用・pure s等価性を適合範囲で使う。重い計算を別AIへ委譲せず、恣意的なCPU/RAM capを設けず、共有環境・他worktreeへ介入しない。

## 確認方法と停止状態

主エージェントがGit固定blob・既存報告・SOURCEテキスト・保存JSON/CSV欄と、既知receiptのstate metadataを読み取った。既存テスト（`check_track_d_original_h_accuracy.py`、`test_rpe_resource_accounting.py`、`test_h46_full_h_pure_rte.py`、`test_h46_full_h_t20_opt.py`等）は対象を静的確認したのみで、今回のtest実行・科学的再検証として数えていない。研究moduleのimport、Hamiltonian生成、eigenpair／signal再評価、trajectory sampling、回路build/compile、価格fit、N／総費用の新算定、詳細科学レビューは全て0。

文書の固定commit/pathリンク21件のGit object取得を確認し、共通interface仕様とT20報告の2資料は接続済みGitHubから取得した全文bytesもローカル固定blobと一致した。内部絶対path・token patternの非混入と数式表示の対応を確認した。

今回の変更は本報告と入口索引のみ。科学的証拠のstatusを変えないので、科学manifest／研究概要や旧SOURCE・raw・結果・grant・ledger・marker・mandatory STOPを書き換えない。公開時には新branchのremote SHAと報告本文のbytesを照合する。**公開照合後mandatory STOP。**

次の担当：**ユーザー／研究方針担当**は§4の条件と§5の限定評価の採否・実行範囲を判断する。**q価格担当Codex**は自身の依頼範囲の成果を共通欄へ対応させる。**本担当Codex**は新しい実行許可まで科学計算・詳細レビューへ進まない。

[account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trotterlib/rpe_resource_accounting.py
[original]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trottertracks/track_d_original_h_accuracy.py
[accuracy]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trottertracks/c1_stage3_accuracy.py
[horizon]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trotterlib/rpe_target_round_horizon_validation.py
[decoder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trotterlib/rpe_four_round_phase_validation.py
[allocation]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trotterlib/research_direction_ablation.py
[decision]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trotterlib/research_direction_decision_cost.py
[finite-shot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/scripts/d0_frozen_shot_rule_v1.py
[qualify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/scripts/run_pf_common_run.py
[signal]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/scripts/h46_long_time_signal.py
[pure]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/scripts/h46_full_h_pure_rte.py
[t20-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/scripts/h46_full_h_t20_opt.py
[interface-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/src/trottertracks/track_d_resource_interface.py
[legacy]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[b4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md
[bc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md
[light]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9b83b96b83f62142c9d87858dc24a94db7ac5d61/artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/REPORT.md
[t20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/REPORT.md
[input-summary]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9b83b96b83f62142c9d87858dc24a94db7ac5d61/artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/input_and_validation_summary.json
[t20-index]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json
[joint]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/docs/research/pr_rpe_joint_integration_20261110/NEXT_EXECUTION_PROPOSAL.md
