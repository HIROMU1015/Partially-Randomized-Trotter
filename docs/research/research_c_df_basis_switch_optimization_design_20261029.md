# 研究C：Basis-switch-aware DF設計と第三DF検証準備

2026-10-29 JST。起点`2e8d48e77737bd5a6a1135147ebe99eb65c45bae`、独立branch `codex/research-c-basis-switch-third-df-20261029`。正式共通指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3を確認した。本書は依頼された技術設計と人工検証の報告で、重要GPT科学レビューや新しい実分子評価ではない。

**推奨順序は、固定exact-groupのJ-mass surrogateを候補として保持し、保存γ=0の第三DFを先に検証すること。新目的・η・scaleは採択しない。** 新SOURCEには純粋PoCと第三DF専用adapterを用意した。第三DF自身のPF/RTE・shots・native32・価格予測を接続する未承認上限493-call契約を準備したが、科学実行はしていない。

## 1. 出発点と再利用

[直前148-call](research_c_df_basis_surrogate_repair_execution_20261029.md)は、保存直鎖H4・1Å/STO-3G、通常number CDF8/元順LD2、partial ordered S2 δ=h=.1/s1/r128/K0、M4 Proposed/tail-P4各32件の結果である。RZ held-out MAEはA/B/C順にM4=221.4073/305.6085/227.7980、P4=287.4242/261.5380/273.5792。登録CはSURROGATE_NOT_QUALIFIED。AとRZのPearson .9844/.9733はcohort内記述量で、DF間の予測性能・因果効果の証明ではない。Cの特徴・判定条件を変えて救済しない。

| 確認した既存資産 | 再利用・今回の差分 |
|---|---|
| `c_pr_aware_df.py`／`c_pr_aware_df_pilot.py`、[元設計](research_c_pr_aware_df_optimization_design_20261026.md) | 一般J→非identity Z/ZZ係数、smooth mass、ffsim loss/JAX AD/既存fit。新fitterやoptimizerは作らず、PoCは係数写像だけ再利用 |
| `c1_ffsim_corrected_codec.py`／`c1_real_orthogonal_codec.py` | 実Schur・列符号gauge、qualified x0、ffsim pack/unpack。人工AD/既存回帰で再利用。今回実分子codec/fitなし |
| Stage A γ0/.1の[30-call](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md) | 保存γ0 exportとDF予算PASSの来歴を第三候補へ固定。別s/rのPF/shots/nativeは転用しない |
| M1の寄与分解、M2同J/U決定論、M3/M4、M4-Native128 | 価格とshotsの分離、sector reference、RPE/affine費用、P0/full512を再利用。旧runnerは起動しない |
| `c_df_structural_support.py`、P4 [21-call](research_c_df_structural_support_execution_20261028.md)／[449-call](research_c_df_structural_pr_performance_20261028.md) | 固定support/parameter injectionと、候補固有PF→RPE→native32のcall graph。第三γ0にP4の疎J制約や144 atom条件を課さない |
| `c_df_basis_surrogate.py`／v2／reanalysis v2 | exact-key run、law/trajectory descriptor、overlap。既存key生成`_matrix/_spin_lift`をそのまま再利用。旧99/103 SOURCE不変 |
| B1/B2、M4-native、P4-native、`StrongBuilder`／CAR／P0 | sampler、actual adjoint、complex X/Y、full512、metrics、Qiskit排他を再利用。新native moduleは候補ID/seed/出力/descriptor接続のorchestration差分のみ |
| [固定U・凍結価格weighted mass案](research_c_pr_aware_df_optimization_design_20261026.md#7-第二候補固定u凍結価格weighted-mass) | 既提案。非負固定重みの加算massはrun分布/whole-P0を直接表さない。今回そのfit/価格calibrationなし |
| [RCDF/SCDF接口調査](research_c_compressed_df_interface_survey_20261024.md)、[構造案](research_c_df_structural_design_20261028.md) | 既知の係数正則化・support・shiftとの重複を維持し、独立新規性を主張しない |

外部一次資料の限定再確認では、[CohnらのC-DF時間発展](https://arxiv.org/abs/2104.08957)、[RC-DFの正則化表現](https://arxiv.org/abs/2212.07957)、[SCDFの対称性shiftとnorm低減](https://arxiv.org/abs/2403.03502)が既存である。これらの測定/qubitization用norm・費用を今回のPR tail λ/native RZへ同一視しない。本作業は網羅的先行研究レビューや文献初出の認定ではなく、既存設計と実装の重複確認に限定した。

## 2. Exact law：Q1/Q2

元frame順とsplitを固定し、atomごとに`c_a=L_a(J)`、`m_b=Σ_{a:key(a)=b}|c_a|`、`λ=Σ_b m_b`とする。Identityは既存の決定論phaseへ抽出し、このλに入れない。同じkeyでも別frameのatomを勝手に相殺・再結合しない。

λ>0で

\[
q_b=m_b/\lambda,\quad P_{same}=\sum_bq_b^2,\quad
T=(r-1)(1-P_{same})=2(r-1)\sum_{b<d}q_bq_d.
\]

最後のpair形は近い集中状態での`1-P_same`の桁落ちを避ける等価式で、law変更ではない。r=128なら係数127。Frame別q_lとkey集約q_bを区別し、Uの近接度・frame名・projector同値をbasis identityにしない。

**固定partition上ではTはJの非identity mass比の関数で、Uの幾何を直接含まない。** 非zero係数の領域で

\[
\partial T/\partial c_a=-2(r-1)\lambda^{-1}(q_{b(a)}-\sum_dq_d^2)\operatorname{sgn}(c_a),
\qquad \nabla_JT=L^T\nabla_cT.
\]

同じλでもmassを集中するとTは下がる。従って既存のλ-only γ項と実質的に異なる。ただし次数0、`T(αJ)=T(J)`（α≠0）であり、λの絶対値やattenuationを抑えない。固定Uでも一般に非凸で、非負固定重みのsmooth mass二乗の凸性を移送できない。全Uも動く元CDFの大域最適性は保証しない。

λ=0のexact tailは既存のNO_RANDOM_TAILとしてq={}、P_same=null、random event/switch期待数0。これは任意方向からの連続極限ではなく、別分岐である。λ→0の正規化gradientは悪条件になり得る。r=1は切替0、basis groupが1個でも切替0。これらから未知のhead価格を0と推定しない。

## 3. Smooth surrogate・scale・零係数

既存の全係数位置に`w_a=sqrt(c_a²+ε²)`を与え、`mε,b=Σ_{a∈b}w_a`、`Mε=Σ_b mε,b`、`qε,b=mε,b/Mε`、`Dε=1-Σ_b qε,b²`とする。PoCはhypotとscaleによる再規格化を使い、小さいλの二乗underflowを避ける。

`∂w_a/∂c_a=c_a/w_a`、`∂Dε/∂c_a=-2(qε,b-Σqε²)c_a/(Mε w_a)`。固定partitionで滑らかだが**exact sampling lawではない**。零係数もε massを持ち、固定zeroが多いframeにもghost確率が入る。今回PoCは既存Mεと同じ全係数slotを使い、data-dependent support cutoffを追加しない。将来固定maskを使うならその式・初期scale・support政策を別途固定する。

全J=0でもMε>0かつ複数groupのDεは正になる。Exact zero-tailをsmooth値から判定しない。一般対称/負係数/非PSD Jを許し、JAX ADへは既存ffsim parameter unpackを使う。Rank8/norb4の既存形式は回転48＋対称J80＝128変数である。

既存scaleは`f*=max(f0,1e-14 Ha²)`、`M*=Mε(J0)>0`。Dε自体は無次元で、`Tε/(r-1)`に等しい（r>1）。`Tε/Tε,0`は初期切替0で破綻するため既定にしない。二乗mass項も無次元で、単位の異なる数をそのまま足さない。εは将来初回なら既存1e-8 Haを候補とするが、新η/正則化scale/目的の最終式は今回採択しない。

## 4. A/B/C候補と推奨：Q2/Q4

| 候補 | 仕様案・利点 | 欠点・扱い |
|---|---|---|
| A：切替のみ | `f/f*+ηDε`。現exact-key runに直接対応するJ-mass集中を検証可能 | λ増加を制御しない。PF/attenuation/shots/headも不明。総費用目的として採択しない |
| B：mass＋切替 | `Fγ+ηDε`を候補として保持。γとηを別の無次元重みとして記録する | 非凸・初期値依存。Smooth/exact値を両方報告。γ=.1等の旧設定を新fitへ自動移植しない |
| Bの別候補 | `(Mε²-Σ_b mε,b²)/M*²=2Σ_{b<d}mε,b mε,d/M*²` | 原点で比を取らないが、massと集中を同時に変える別目的。非凸、ε biasもある。候補式の比較に留める |
| C：native価格proxy | 既知価格があれば`H+Σ_b E[R_b]G_b+中央Z/ZZ+phase`等を検討 | Head/basis価格は未取得/null。Whole-P0、相対境界、depthは単純加算できない。旧回帰係数を普遍価格やηとして流用しない |

現段階では**Bを最小人工候補として残す**が、fitへの採択は保留する。第三DF検証を先に行う理由は、同一DF内のrunと価格の関係を、異なるJ/Uへ移せるかが未検証だからである。登録不適格の動的3列Cとは別の「目的候補C」であり、旧Cの復活ではない。

λ悪化を無視したT低下は採用理由にならない。J/Uが変わるとPF誤差、RTE、`τ=λh/r`と`A_m=(1+τ²)^(-r 2^m/2)`、phase-room、ceilを含むshots、head/central価格も変化する。将来は同じ物理予算を満たした上で、候補自身の全13roundと条件付き`C_total=W μ_kernel+S`まで評価する。Ha単位DF誤差と無次元operator/signal誤差を根拠なく単純加算しない。

## 5. U最適化とexact共有：Q3

一般に異なるUのpartitionは近傍でも全別groupで、固定labelのswitch項にU gradientはない。完全一致する二frameを少し別々に動かすと、同一groupが分裂し、λと各massが同じでもTは`2(r-1)m_i m_j/λ²`だけ跳ぶ。人工微小回転でこの不連続を確認した。

列符号はdensity projectorを、列置換はJも整合して変換すればtensorを保持できる。一方、nativeの宣言tuple/key・recipeは別になり得る。旧gaugeをcanonicalizeせず、U近接やprojector同値でsharingを増やさない。修正codecの列符号gaugeとnative key同一性も別の検証事項である。

整合的な滑らかさを得る案は、固定groupごとに一つのUを共有する構造制約`U_l=V_g(l)`、またはU固定でJだけを動かす局所問題である。前者は独立回転の自由度・表現集合を変え、異なる初期Uを単に束ねれば元DFを保持しない。同basisのJをまとめるとHamiltonianは線形に再構成できてもPF frame順/分割は変わり得る。今回これらの制約adapter・merge・新fitは実装/採択しない。

Overlap `n-Σ|U_l^T U_k|⁴`は滑らかな幾何対照Bとして保持するが、小さいことから現tailのcompiled価格低下は導けない。新しいapproximate sharingや相対Gaussian政策は導入しない。

## 6. 第三DFのprovenanceと役割：Q5/Q6/Q7

対象は保存Stage Aの**通常CDF γ=0、rank8、元順LD2**。

- Export：`artifacts/research_c_pr_aware_df/2026-10-27/run01/baseline/final.npz`。
- SHA256：`f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02`。
- Fit SOURCE：`e1085c95f5c01b68566e17d543ace5832cb94061`。
- 元P-H4 SHA：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。Runnerはその来歴を持つStage A h/g/scalar snapshotを使う。Snapshot bytesのhashと元packetのhashを混同しない。
- Headerだけ確認：J/Uともfloat64 `(8,4,4)`、x/jac `(128,)`。**今回数値decodeしていない。** [Provenance](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/provenance.json)。

Stage A報告のBaseline sector誤差7.018855038843621e-9 Ha／guard3.278781198621627e-9 Ha、DF予算PASSは引用として保持し、再計算しない。Optimizer successとgradient・物理DF予算は別判定のまま。保存γ0は比較に適する第三候補だが、同一H4/距離/basis/rank、関連初期値・最適化系列である。統計的に独立なDF母集団標本とは呼ばない。

新しく得られるのは、未観測DFのown-law切替構造・PF/RTE/減衰/条件付きshots・同P0 native価格、既存2DFからの価格移送誤差である。32 trajectoryは一つのDFの中の反復であり、DF32件ではない。DFが3個になっても一般化・因果効果・head寄与の分離・別分子への移送は未確立である。

## 7. 外部検証の事前固定

旧cohort内4-foldとは別に、**M4 Proposed＋tail-P4の64行で訓練し、γ0の32行をDF単位でholdout**する。新Cは入れず、mean/A/Bだけを使用する。元64行のregistered descriptorとmetricsを保存run02から読み、event再解析・QPY census・compileを繰り返さない。

各6 metricでNumPy OLS rcond=1e-12を使う。訓練は共通の一つのintercept、未中心化/未scaleのraw特徴と価格単位、各DF32行で等重み。第三DFのinterceptを推定すること、cohort別interceptを教師なしで転用すること、特徴/scale/係数を第三価格後に調整することは禁止する。訓練snapshotは**第三J/Uの読込・価格アクセスより前**に保存する。

主指標は第三DFのRZ MAE、副指標RMSE・signed mean prediction error・6metrics・観測/予測sample mean。登録診断PASS案は、AのRZ MAEがglobal meanとBより厳密に小さく、rankがfullでA負予測0。Rank不足はUNRESOLVED、負予測はclipせず記録する。失敗を新特徴やRMSEで救済しない。これは今回新しい移送仮説の未承認契約であり、旧C判定の変更ではない。

第三DFで平均価格予測がずれる場合、固定head/basis/係数構造の差を含み得るが、head価格を同定したことにはならない。AがPASSしても、切替を操作した因果効果や新fit GOには自動的につながらない。

## 8. 未承認493-call契約・順序：Q8

[契約](research_c_df_basis_third_contract_unapproved_20261029.json)はUNAPPROVED_PROPOSAL。[専用runner](../../scripts/run_research_c_df_basis_third.py)は既定でhash/metadata/packagesだけのpassive preflightを行い、別途の明示的科学許可後に限り`--execute`を使用する。承認証明・新grantは不要。

保存直鎖H4・1Å/STO-3G、Nα=Nβ2/sector36、通常number CDF8/LD2、元g補正κ・核scalar一回・元frame順。δ=h=.1/s1/r128/K0、m0..12、CA=.00159 Ha／DF reserve=CA/8、own-DF oracle・状態準備除外を固定する。βtotal=π/6、βstat=π/12、αaxis=.01/26、αtotal=.01を既存pipelineから継承する。

| 段階 | 内訳・再利用 | 上限 |
|---|---|---:|
| T：訓練固定 | 保存64行load1＋identity1＋6metric共通intercept訓練6 | 8 |
| A：第三DF reference | 既存P4 referenceの63（候補固有general-Jへ変更）＋exact law/geometry descriptor1 | 64 |
| RPE | 3lane資格3＋13round×3=39＋energy3＋shots summary3 | 48 |
| Price/GO | affine3＋M4/P4との2仮定×2損益分岐4＋GO1 | 8 |
| B：native32 | 既存native325＋trajectory descriptor32 | 357 |
| C：DF holdout | 6metric評価6＋主RZ判定1 | 7 |
| 最終結果 | 結果/未資格/mandatory STOP | 1 |
| **合計** | **最大** | **493** |

Native325の内訳はCAR1、full raw factor9、sector照合1、独立reference context1、law照合1、Gaussian9、head relative4、shell読込6/検証3、32件それぞれsample/build/reference/raw gate/compile/compiled gate/adjoint/X-Y/metricsの9種=288、calibration1、native結果1。QPYはraw/compiled/各actual adjointの128件。全512次元raw/compiled operator、相対位相、unitarity、actual adjoint、complex X/Yを既存B2で数値検証する。新compiler/samplerなし。

Tを最初に固定し、Aが登録十分条件を満たさなければRPE/native/holdoutを実施せず73-call以内で終了する。A通過後も条件付きenergy/shots/Wが未資格ならnativeを行わず129-call以内。Native GOは観測価格や損益分岐の正負でcohortを選択しない。SOURCE/input/表現/ground/law/compile correctness FAILでは依存callを止め、並行開始済み作業を回収して部分証拠を保存する。

3laneはevidence、ideal_exact_native、assumed_raw_signal_budget。後二者は既存の反実仮定で、未知の合成/law誤差を実際に0/既知と認定しない。Evidence laneの未資格はnull。Native gateのcomponent/信号allowanceは旧1e-10/2e-10、normalization1e-12等をそのまま使用し、事後緩和なし。保存DF guardもbinary64数値pilotで、厳密enclosureではない。

Native sampleは新master2026102907/cohort、SOURCE digest＋γ0 J/U hash＋role/axis/indexに結合した既存PCG64、32件非paired、各128 K0 events。将来の各occurrence独立を維持し、一つのsample反復で費用を過小評価しない。P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initially-zero=false、既存head relative・tail exact-key共有・cross-head-tail融合なし。静的U gauge変更や未観測のliteral/shared削減なし。

期待総費用は有限32のkernel sample meanを使う条件付きserial点評価。Sはancilla head/readout shellのinterrogationごとの費用で、DFのone-body/F0/F1 headはkernel内に含む。状態準備は除外。全roundで一つの校正平均を共有したSEは記述的なものに限り、独立13校正と扱わない。全law/cost support/actual CA/保証付きshots/母平均優位は未資格。

Fresh `artifacts/research_c_df_basis_third/2026-10-29/run01`、retry/resume0、新fit/optimizer/rank探索/DF誤差再計算0。今回はこの出力を作成せず、passiveだけを確認した。新しい候補用のreference/native orchestrationは、P4/Proposedの旧role/gateを変更する代わりに薄い別moduleとした。旧数値kernelの再実装はない。

## 9. 人工検証・SOURCE・保全

純粋PoCはexact q_basis/q_frame、零tail、微小λ、負/非PSD J、固定zeroへのε bias、同basis/異basis、符号/列置換gauge、微小U変化の不連続を確認。解析coefficient gradientを有限差分照合し、実ffsim pack/unpack＋JAXでsmooth項のgradientを照合した。固定group項の回転パラメータgradientが0であることも確認。Optimizerはmockで遮断し、iterationは0。

第三DF adapterは人工Hamiltonianでown reference/13round/RPEを実行し、native価格/operator engineをmockした493-call end-to-endで、新32seed/JU/SOURCE binding、QPY128、descriptor32、holdout、段階STOPを確認した。別に既存の人工full512 raw/compile/adjoint/X-Y回帰を再利用。これは実H4の科学的PASSではない。

新21＋既存110＝**131 unique tests PASS**（evaluation56／138.64秒、AD75／5.32秒）。[検証記録](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/verification.json)、[evaluation log](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/evaluation_tests.log)、[人工AD log](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/fit_artificial_tests.log)。ローカルテストでありCI/外部再現ではない。

[SOURCE112](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/SOURCE.json)と専用[環境lock](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/environment.lock.json)を固定した。EvaluationはQiskit1.3.0、AD専用環境はffsim0.0.83/JAX0.4.38でQiskit2.4.2を含むがP0に使わない。専用venv/.cacheだけを変更し、共有依存はread-only。32 affinity CPU、空きRAM約55 GB、低loadを確認し、CPU/RAM/worker/wallに新たな人工上限は追加していない。既存小行列testのprocess内BLAS設定は保全した。

契約SHA256：`b3af0d017bb9e018380b30f031c4814dbd28c7dc1ee84f0b6d5b4715f4b4833a`。SOURCE manifest SHA256：`fa65695c5f8f3c402c7618e831296d59f006d1f6e4a8a7a4653adf86ab445912`。保存input93件。科学SOURCE・input・条件を変更する場合は旧契約を上書きせず差分を明示する。

旧SOURCE103・旧原本/STOP1667件をhash照合。Main workspaceのdirty変更には触れず、旧runner再起動・旧承認残枠流用なし。今回の実H4 decode、旧trajectory再解析、新PF/RTE、sampling、実分子build/compile、shots/価格新集計、新fitは**すべて0**。

## 10. 将来の最小1-fit案と判断：Q9

第三DFのA移送が未達/UNRESOLVEDなら、現在のswitch-only特徴を新目的へ採択しない。Head/basis固定費の未分離性、Bの適合、rank/外挿を確認するための別設計を研究Cが判断する。特徴追加やDF探索を自動起動しない。

第三DFで登録診断が有望でも、次に必要なのは目的式・ε/scale/η・U政策・初期値を**結果後の新提案として**固定した別契約である。最小案は保存H4/CDF8/LD2を維持し、固定Uまたは固定partitionのJ-mass項を既存ffsim loss/JAX/修正codec/SciPy L-BFGS-Bへ加える1 fit。U自由ならswitch項はUを直接制御しないことを明記する。Optimizer条件は旧options=None等の再利用を候補とし、今回採択しない（旧tol=1e-8はffsim分解初期化のtolで、SciPy収束tolの指定ではない）。

一度のfitのsuccess/gradient、初期・最終tensor/sector norm＋guard、exact/smooth λ/q/Tを別々に記録する。物理DF予算未達・correctness FAIL・未収束等はSTOPとし、初期値への自動fallback、rank変更、η探索、retry、追加fitをしない。予算PASS/T低下だけでは性能GOにせず、当該新J/UのPF/RTE/attenuation/条件付きshots/nativeを別途評価して`Wμ+S`と比較する。この一連の科学処理は今回未承認である。

次の担当は**ユーザー／研究C**が493-call案の可否を判断し、**Codex**は固定SOURCE/契約について別途明示的に許可された場合だけ実行する。新cost-aware fitや詳細GPT科学レビューの許可とは分離する。技術成果をcommit/pushしremote SHA/tree・SOURCE・旧原本を照合後、**mandatory STOP**。
