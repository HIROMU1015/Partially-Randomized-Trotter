# A1 新科学実行契約草案：保存DF・1-round公平比較

2026-11-06 JST。**DRAFT／未採択／未実行。** 本書の作成・公開はtarget採択や科学実行許可ではない。
対象・式の正本は[A1_NEW_FINITE_TARGET](A1_NEW_FINITE_TARGET.md)。過去の[契約草案][OLD_DRAFT]と[A1-0接続scope][OLD_SCOPE]の適合部分を再利用する。

## 1. SOURCE・inputを固定する方法

元core/adapterの候補は `6381f816306a6f98d6ea3abc13c1eb5ee8bdf9e6`。そのままの科学SOURCE候補のbytes一覧は[SOURCE_CANDIDATES.json](SOURCE_CANDIDATES.json)。これは候補の同定であり、将来追加するrunnerのSOURCE固定を完了した意味ではない。
将来の実行許可には、採択する本草案の固定SHA、NPZ SHA256、N1等の許可scopeを指定する。許可内で小adapterを実装・人工検証し、**実入力を投入する前**に実際のrunner/全依存/数値環境を新commitへ固定・公開照合する。同じ調査や承認発行を繰り返す必要はない。
Python3.12.3/NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0と既存CPU import helperを第一候補とし、実際のqiskit-nature等の依存版・wheel/BLAS情報・thread設定も記録する。NumPy choiceと価格compilerの検証済みprofileを変えるなら、別profileの対応確認を実入力前に行う。
共有venv/package/cacheは変更しない。runtime再構成は自分専用環境に限る。CPU/RAM/workerへ恣意的な低上限を新設しない。

科学inputは新しい `artifacts/o2_a1_new_target/<run-id>/` 内に固定する。source_commit、source hashes、NPZ/H/state/partition/preparation/tail/catalog/law/policy/compilerのidentityを結ぶ。
新identity値は今回未取得。N1取得値を同じcampaignの正本にし、旧hashや前回診断hashを期待値として埋めない。code/input変更、数値環境差、結果後の再ランキングは自動採用しない。

## 2. 比較と固定control

全方式は[一次target](A1_NEW_FINITE_TARGET.md#3-一次target実lawselected-full-interrogationの複素信号)の同じμ_X/μ_Y、w=W=1を推定する。
候補labelはa=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`。N1でIDだけでなくfragment origin、非identity性、sign/support/basisの意味とpositive actual-grid massを確認する。意味のずれ・欠落・zero supportは報告してSTOPし、別labelに差し替えない。

| control | 定義 | exact-Qでの既知平均 |
|---|---|---|
| g1 | 1[n=0,L0=a] | ν1=Q0 Qa |
| g2 | 1[n=2,L0=a] | ν2=Q2 Qa |
| g3 | 1[n=2,L0=a,L1=b] | ν3=Q2 Qa Qb |

joint momentは既存 `ControlBank.joint_mean`。g1g2=g1g3=0、g2g3=g3、g_j²=g_j。これは未知量子共分散を与えるものではない。
B0はraw平均、B1-Oはn0/n2の正層を合成、B1-OLは次の5層、A1は独立trainで凍結したλによる \(X_b=Y_b-\lambda_b^T(g-\nu)\) のproduction平均。

| B1-OL層 | 質量p_h |
|---|---|
| n0,L0=a | Q0 Qa |
| n0,L0≠a | Q0(1−Qa) |
| n2,L0≠a | Q2(1−Qa) |
| n2,L0=a,L1≠b | Q2 Qa(1−Qb) |
| n2,L0=a,L1=b | Q2 Qa Qb |

exact-Qで真に零質量の層だけ除く。未観測の正層を削除・再正規化しない。全正層の和1、相互排反と被覆を検査する。
3controlはこの5層の各層内で定数であり、飽和層平均に同じCV定数を加えても質量合成後は相殺する。order-onlyとの差をCV独自利益として二重計上せず、B1-OLを主要な強い対照にする。

## 3. 条件付き取得案

**第一候補はactual-grid整数質量による直接条件付き取得**。canonical unconditional generatorのSOURCEは変えず、B1用の小adapterだけを追加する提案である。
labelの格子count m_i=M Qi（M=2^53）をそのまま用い、許容label集合Aに対し、0,...,sum_A m_i−1の一様整数を累積countで割り当てる。強制a/bは直接置く。条件外positionとrotationは元label law、orderもその層の条件通りとし、SOURCEのrotation-first配列へ戻して `_make_event` を呼ぶ。
整数一様生成の無偏性・endpoint・overflowを人工countで検証し、浮動小数点で確率を再正規化した `choice` を厳密条件付きlawの代用にしない。
Q(ω|h)への条件付けであり、B1の重みは層平均合成のp_hだけ。eventのnominal probabilityを追加importance weightにしない。既存親研究のdyadic proposalへ変更しない。

既存草案のcanonical metadata rejectionを選ぶ場合は、各stage開始前に受理quota M_hを固定し、attempt cap=ceil(2M_h/p_h)を保持する。prod capは独立cal後に決める。棄却は測定前、古典生成費を全て記録する。
直接取得とrejectionの選択はoutcome前のユーザー判断事項。結果後の有利な切替えは禁止。直接取得の最低限の実装が未完成なら科学samplingはSTOPする。実行可能な直接取得を無視して棄却費だけで層別を不利に認定しない。
attempt capは完走確率保証ではない。未完了・失敗campaignと費用も保存し、完走したcampaignだけを選ばない。

## 4. 候補の精度・confidence・測定予算

以下は**採択前の具体案**。旧RPEのN=510、ε/α、半径から流用した数ではなく、科学値を代入して評価した結果でもない。

| 項目 | 提案値・規則 |
|---|---|
| 一次target座標精度 | ε_X=ε_Y=1/4。複素誤差は座標保証から扱い、energy精度とは呼ばない |
| 同時失敗予算 | 4方式×2axis全体でα_diag=1/20。18roundのα配分を継承した意味ではない |
| 方式・axis予算 | A=α_diag/8。δ_cal=A/4、δ_prod=A/2、δ_price=A/4 |
| B1の層別配分 | H正層へδ_cal,h=δ_cal/H、δ_prod,h=δ_prod/H。価格保証を行う場合の層別配分もoutcome前に固定 |
| A1 train | 1024 fresh shots/axis、一回。既存intercept付きOLS、3 slopesのみ凍結 |
| B0 train | 0。不要なCV学習費を負担させない |
| B1 pilot | 64 conditional shots/正層/axis、一回。層varianceとfull価格の実測だけを配分に使用 |
| cal | η=1/8、n_cal=max(2,1+ceil(2 log(1/δ_cal,cell)/η²))。変数・層ごとの独立固定数 |
| prod | 各方式・axis合計≤8192。一回、freshで固定N。十分Nがcapを超える場合は未完了STOP |
| CV range guard | D_CV≤4W。違反時clip/再fit/feature削除をせず、この候補を不成立と記録 |
| 数値engine allowance候補 | b_engine≤ε_b/8を共通の受理目標とする。値は後日の根拠付き取得が必要 |

α/ε/η/train/pilot/cap/allowanceはユーザーが採否を決める。採択時に一つへ固定し、測定後の緩和・増量・seed選別をしない。
価格confidenceを付けない初期案ではδ_priceは未使用のまま残す。train品質の検定をしたとは言わず、λを条件付けたcal/prod保証を用いる。

## 5. 有界confidenceと有限配分

λは既存 `fit_and_freeze` のminimum-norm OLSで取得し、binary64 slopesをexact Fractionにして凍結する。rank不足・rare control未観測を記録し、学習できたことにしない。
出力rangeは既存 `FrozenCV.output_range` のbox enclosure、D_CV=2W+sum|λ_j|。B0/未補正の層はD=2W。sample最小最大でrangeを狭めない。
νを丸める場合は \(b_{center}=\sum_j|\lambda_j|e_j\)。exact-Q/有理数処理のモデル内でのみ0とできる。

独立calの不偏sample variance s²から
\[
V_U=\min\{D^2/4,(\sqrt{s^2}+D\sqrt{2\log(1/\delta_{cal,cell})/(n_{cal}-1)})^2\}
\]
を用いる。[Maurer–Pontil v1 Theorem10][MP]のrange変換であり、IID、有界、固定n、λ凍結を要する。s²=0を真variance0とはしない。
共通b_engineにstate/sector/basis ordering、probability数値誤差、bit生成誤差、必要ならcompile等を含め、
\(e=\epsilon_b-b_{engine}-b_{center}>0\)を要求する。
state residualだけからこの上界を発行しない。uniform errorまたは同じ保証を与える事前上界が不足なら測定へ進まない。旧opt2 bias/半径で穴埋めしない。

fresh productionに対する二側bounded Bernsteinの保守的十分数とHoeffdingを全方式に認める。
\[
N_B=\max(1,\lceil\log(2/\delta_{prod,cell})[2V_U/e^2+2D/(3e)]\rceil),\quad
N_H=\max(1,\lceil D^2\log(2/\delta_{prod,cell})/(2e^2)\rceil),\quad N=\min(N_B,N_H).
\]
選択はproduction前。√/log/ceilingは外向き評価で十分性を保つ。これらは既知不等式の利用案であり新定理・新Nの数値結果ではない。[一次資料][MP]
cal失敗とprod失敗をunionする。補正outcomeは非binaryであり、旧V2二項tailを適用しない。

B1の具体的配分案：pilot価格平均 \(\widehat C_h\) と独立calのV_U,hを使用する。各層のε_hに対し上式でN_hを求め、sum p_h ε_h≤e、N_h≥1、sum N_h≤8192を満たすものからprod予測価格sum max(1,Ĉ_h)N_h最小を選ぶ。
候補集合は、等ε_h=eと、正整数j_hのsum j_h=J=64に対するε_h=e j_h/(J p_h)という有限集合を第一候補とする。tieは層順の辞書順。比較対象はこの有限集合内であり、真の最適Neyman配分とは呼ばない。
独立calに条件付けてε_h/N_hを凍結し、fresh層平均に適用する。層ごとの失敗をunionし、座標誤差をsum p_h ε_hへ戻す。pilot/calをprod平均へ混ぜない。
B0にもvariance-sensitive routeを認める。この初期案では全方式を同じcal→Bernstein/Hoeffding routeへ固定し、結果後に別confidence方式へ切り替えない。

## 6. State・measurement engine・独立性

全H・全4電子sectorのstateをengineへ一度bindする。再利用payloadはvector/sector/H hashと残差等を照合する。未対応ならN2で限定solveを許可する必要がある。
数値vectorと対象groundの差を保証するには最低固有空間の同定・分離等が必要で、残差の小ささだけをstate保証としない。必要な検査はN2 scopeに含めるが、engine内の情報をfit/配分へ漏らさない。
N2の具体的な受理経路案は、同じ全H/sectorのdense参照または既存matvecから、構成誤差込みのground固有値区間[l0,u0]と第一励起の下界l1、Rayleigh値の上界r_Uを検証することである。l1>u0により一意groundの分離が検証できれば、励起重みを min(1,max(0,(r_U−l0)/(l1−l0))) で上から押さえ、座標state誤差をその平方根の2倍で上界化できる。これは最低成分と励起成分のRayleigh値分解に基づく保守評価であり、今回は数値を代入していない。
単なる浮動小数点eigenvalueを検証済み下界とせず、dense/matvecの組立誤差とspectral enclosureを含める。必要なenclosureがない場合は条件未閉鎖としてSTOPし、残差toleranceだけで代替しない。solverと照合の許容値は採択したb_engine予算に従う。full molecular wrapperを使う科学検査はN3へ含め、N2だけの許可では起動しない。

各path・axisのfull selected回路に対するBorn確率をengineが計算し、その**一つのfresh bitだけ**と必要な古典metadataをestimatorへ渡す。事前計算したpath mean/variance表や真E0をpilotの代用にしない。
確率計算誤差と、53-bit U<p̂によるBernoulli生成なら座標誤差≤2/Mというgrid項をb_engineへ含める。範囲外確率を無根拠にclipしない。
有限-bit wordモデルと理想Bornモデルの接続を記録する。理想IIDの仮定をPCG64の決定論的独立性の証明と呼ばない。

method×axis×stage×stratumに固定した独立stream/trial領域を割り当て、pathとmeasurement-noiseのdomainも分ける。seed由来規約・衝突検査・範囲を保存する。nameが異なるだけでは独立性を保証できない。
`Batch/require_disjoint/evaluation_inputs` はdata再利用の検知に再利用する。joint RNG独立性の証明器ではない。X/Yは別fresh shotであり、一つのbitを両軸へ使わない。

## 7. 回路・価格・全費用

第一候補は `support_run_le_1`、`boundary_optimized`、ordinary controlled。新preparationから `register_support_restricted_bases` で生成・保存する。
policyはoutcomeなしでrun threshold=1を事前指定する。新policy宣言・新basis定義・新preparationのhashからfingerprintを作り、既存引数`training_fingerprint`へ渡す。これは新たな価格学習をした意味ではなく、旧WP06b training fingerprintを新targetへ流用しない。
selected/full-basisの数値同値の検査・許容差と、一次targetそのものの定義を区別する。policy適用が未対応ならSTOPし、full-basisへ無断fallbackしない。

compilerはQiskit1.3.0、basis=`rz,sx,x,cx`、optimization_level=2、seed17、backend/coupling/layout/routing指定なし。
主資源はcompiled RZ count候補。full no-state-preparation interrogationはD half-steps・中央RTE・制御・constant/identity/event phase・ancilla・X/Y readoutを含む。wrapperを二重課金せず、状態準備だけ除外する。
`selected_full_prices(...axes=(axis,))` と既存builders/compilerを再利用する。同じ数値回路/profileのcache hitでもshot価格は毎回課金する。旧8 trajectory平均、旧proxy、旧cacheのidentity不明なentryを価格真値へ移さない。

初期案は**実現campaign費用**を主報告とする。方式別に全train/pilot+cal+prodのfull RZを合計し、共通setup、ν取得、fit、補正、条件付き取得/棄却、compile/cache、engineの古典時間・peak memoryを別欄へ記録する。
同じデータを共有した場合も、実験全体の取得費と各方式の単独運用費を分ける。共通preparation/setupを一方式だけに課さず、ν/層質量等の方式固有setupは必要な方式に計上する。価格欠測・T/native未取得はUNKNOWNであり0/RZ換算にしない。
RZと古典秒等の異なる単位を無換算で加算しない。換算率が未採択なら費用vectorと補助座標を報告し、RZの改善をwall-time/総金銭費用の改善とは呼ばない。engineの古典simulation費は検証実費として、量子algorithmのestimator online費と分ける。
価格cacheは新run専用。共有cacheを書き換えない。`TrialCost/account_costs` と外部の古典費記録を再利用する。

実現費用差を期待費用改善のconfidenceとは呼ばない。期待費用の保証を要求する場合は、全pathに対する有効な価格rangeと独立価格取得の有限scope・精度・δ_priceをoutcome前に別途固定する必要がある。
Nがcal依存なので、無条件にE[N]E[C]とはしない。凍結後の条件付きprod費用を用いる。初期案では母平均価格・期待総費用の保証は未閉鎖のまま明記する。

## 8. 必要な実装差分と科学call（すべて後日許可が必要）

| 段階 | 呼出し・出力 | STOP境界 |
|---|---|---|
| 技術実装 | 新target用薄いrunner/serializer、exact conditional adapter、confidence/有限配分、engine/価格のbindingを人工入力で確認し、新SOURCE固定 | 実NPZを先に投入しない。旧guardの削除で代替しない |
| **N1：最小の次回推奨scope** | 同NPZ load一回→LD3 split/preparation一回→完全payload保存→同じconfig/CDF/Q/support/ν/joints/2層/5層。新policy basis構成・対応確認を含むがfull interrogationなし | fresh出力・新identity・policy/lawの不整合、候補label/support不成立でSTOP。完了後もSTOP |
| N2：stateとengineの接続 | N1固定payload、全4電子sectorの保存state検証または一回の限定solve、state/数値/bit-grid error budget、layout・符号/phase接続検査 | H/state精度・oracle隔離・正残余の不足。複数grid/PF有効state探索なし |
| N3：有限測定とfull価格 | 採択された全方式/axisのtrain/pilot→cal→N/層配分凍結→prod。一shot一fresh path/bit。必要なpath/axisだけselected wrapper build/不足compile | range/N/正層quota/attempt/予定外call。全K2 path価格列挙や別roundなし |
| N4：任意の追加接続 | 新Qとalgebraic reference、PF/finite-RTE/selected/元DFへのbias・半径、期待費用confidence等 | 今回の1-round mean比較に自動追加しない。別に目的と必要callを定める |

N1だけの許可でN2/N3へ進めない。一方、ユーザーがN1–N3の有限campaignをまとめて許可した場合は、同じ許可範囲内の段階ごとの再承認を求めず、各受理条件を満たして進める。
旧 `a1_law_validation.run` は旧hash guardを持つため新target runnerとして起動しない。`diagnose`や元opt2全runnerも再実行しない。

| 再利用資産 | 使い方・不足 |
|---|---|
| df_partial_randomized_pf / df_partial_s2 / df_rte_tail / rte | 同じ保存DFからsplit・preparation・eventを生成。元の固定SOURCEは保全し、計算済pre-threshold/hash payloadの返却だけを新SOURCEに追加して固定 |
| a1_connections / a1_control_variates | actual CDF、exact law/moment、OLS/range、raw/strata、bit binding、費用を再利用 |
| rpe_hadamard_interrogation / df_rpe_hadamard_compiled_cost | full wrapper・metric形式を再利用。全benchmark runnerは起動しない |
| research_direction_full_scope / sequence_policy | `_finite_inputs`相当とselected policy部品を再利用。paired full/selected compileは主測定へ流用しない |
| research_direction_full_opt2 | 価格profile・構成の設計参照。旧physical/training fingerprintを要求するworkerを直接使わない |
| a1_law_validation.preparation_payload / shared_cpu | payload保存とCPU namespaceを再利用。pre-threshold完全payloadにはextractionの読み取り専用返却差分が必要 |
| 既存A1/RTE/RPE tests | 該当人工部分を将来差分検証へ再利用。現在29PASSの履歴は新target測定成功の証拠ではない |

現接口は有限confidence、直接条件付き取得、state測定engine、source-bound新target orchestrationを完備していない。小adapterが必要な理由はここにあり、新しい承認管理・汎用pipelineは追加しない。

## 9. STOP・判定・結果保存

実行前：新target/sector/ε/α/予算/取得法/資源指標/理想モデル/許可scopeが未採択、実SOURCE未固定、N2/N3に必要なbias/独立性/価格接続が不足なら該当段階を開始しない。
実行中：H/source/law/wrapper/phase不一致、非有限値、bit/weight範囲外、trial混入、正層欠落、range guard、十分Nやquota/call超過、input破損でSTOP。実際の共有資源不足や他者への支障では自分のjobだけ安全停止する。
旧hash不一致だけでは新contractを失敗にしないが、新contract内部のidentity不一致を隠さない。結果後のfeature/seed/精度/δ/r/K変更、retryや別roundへの救済は行わない。

SOURCE/input、full catalog/basis/identity、actual CDF/ν/層質量、policy/profile/state binding、全bit/trial/価格、OLS/rank、cal/variance上界/N選択、費用と未完了理由を分離保存する。vector等のengine秘密はestimator入力へ含めない。
報告は「設計」「実装/人工テスト」「local科学取得」「条件付きconfidence」「未取得」「最終科学結論」を分ける。

継続材料は、全正層・同じmean精度/confidenceを満たし、CVのtrain/cal/range/全費用込みで強いB1-OLに対しても改善が残ること。実現費用差なら一campaign内の記述的比較として報告し、期待優位とは言わない。
不改善は学習費・range拡大・cal費・層別への還元などがshot/資源節約を相殺する場合。unknown価格や未完了比較、oracle漏洩、信号不一致は判断保留/無効であり不改善の科学結論と混ぜない。
sample varianceだけの低下、rawだけへの勝利、旧Nとの比較でCV独自利益や全PR/RPE優位を認定しない。

**今回のmandatory STOP**：3文書の設計・commit/公開・remote本文/blob/tree照合まで。新target採択、新準備/CDF/state/RNG/測定/compile/variance/N/費用評価、重要科学レビュー、C1解凍は行わない。

[OLD_DRAFT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a2fcbe212ebd66b563dd1413768e5569bf5fdfed/docs/research/o2_a1_server1_handoff_20261106/A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md
[OLD_SCOPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50f4b43ca8b27d8e5cfc337df3e4740fd65c61a1/docs/research/o2_a1_0_connections_20261106/A1_0_SCIENCE_RUN_SCOPE.md
[MP]: https://arxiv.org/pdf/0907.3740v1
