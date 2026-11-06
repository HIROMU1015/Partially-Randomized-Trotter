# Track D：サーバー既存資産とPR＋RPE総費用への接続調査

2026-11-06 JST。参照commit：`7d0dc55199c49a3f49a9ba93323843d8a336cce1`。
専用branch：`codex/track-d-pr-rpe-static-survey-20261106`。

**同じCDF8 endpointの保存32 trajectoryと13-round資料を再利用する入口がある。ただしReference／強いsymmetric-directionalの全path期待価格、元H厳密基底状態での必要shots、全round総費用を結んだ証拠はまだない。** 固定word02のCX・depth削減を全RPEの費用優位へ外挿しない。

本書は既存SOURCE・JSON・索引・Git objectの静的調査である。新しい科学計算、保存数値の再解析、sampling、回路生成、QPY decode、compile、科学moduleのimport、test／runner実行、新規実装は行っていない。数値は既存欄の引用で、平均・SE・shot数・価格比・費用合計を新たに算出していない。後述の検証は**提案のみ、未承認・未実行**である。

## 1. 指示・調査範囲・証拠の読み方

適用される共通／repositoryのAGENTS.md、PROJECT_MAP.md、正式共通指示 `research_project_instructions_unified_2026-10-25.md`（特に§8.1/8.2/8.4/10/10.3）を確認した。研究概要・現状、研究目的・方法、VALIDATION_STATUS／manifest、[研究統一索引][INDEX]、[3環境統合][THREE]、[サーバー2固定監査][S2AUDIT]と[取得限界][S2GAPS]、既存保存先一覧を入口にした。正式共通指示は利用可能なlocal原本を読み、repository内の架空の公開リンクを作っていない。

本人の対象repositoryに保存された既知の固定資料と、ユーザー指定の既存保存先一覧に限定した。既存worktreeを切り替えず、参照commitから本書用の分離worktreeを作成した。他hostへのSSH、未知領域・他ユーザー・履歴全体の広域探索、共有Python環境の変更は行っていない。主エージェント自身の確認であり、別担当の独立科学レビューではない。

| 区分 | 本書での意味 |
|---|---|
| 今回確認 | 当該サーバーで本文／JSON欄／Git object metadataを実際に読めた |
| 既存検証の引用 | 元報告の数値correctness・統計判定。今回再実行・再認証していない |
| 構造上の適用候補 | SOURCEと既存Track D条件に適合する構造。全pathのcompile後保証・母平均優位は未確認 |
| 未確認 | 対応未結合、未取得、未検証を区別。不在や費用0へ置換しない |

既存のgrant／ledger／marker／STOP、固定SOURCE・科学条件・入力・raw・結果を変更しない。現行mainに収録された同名moduleを、過去runの固定SOURCEと自動的に同一視しない。[収録とraw所在の案内][COLLECTION]

## 2. 確認できたSOURCE・Hamiltonian・保存path

すべてrepository相対path。以下のcarrierは保存資料の所在であり、科学実行SOURCEとは役割を分ける。

| 資産・条件 | 今回確認した所在・SOURCE | 再利用できる範囲／境界 |
|---|---|---|
| **Track D実H4固定word02**：linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、8 modes、number-CDF8、元順LD2、S2/s1/r32/K0、h=δ=.1、1 h-step | [RESULT][DRESULT]、`artifacts/track_d_h4_reference_validation/2026-11-06/run01/` のSOURCE／conditions／result／STOP。科学SOURCE `b21b5faff1fcbcabbb7f674c4b6d8818b8bb08ea`、保存carrierは本書参照commit | 同じword・cutの任意system入力に対するX/Y observableの数値検証、3方式の全wrapper価格。全law平均・N・全RPEは未評価 |
| **元C固定word pilot**：同じCDF8、4本の構造固定word、各32 events | 科学SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、結果carrier `3649b75af191916daf44abb91e14d34c43b83e8f`。[元報告][CWORD]と[保存run][CWORDSAVED]。`samples/word02_events.json`、`references/word02.npz`、`circuits/word02_{raw,compiled,raw_adjoint,compiled_adjoint}.qpy` | events／CAR reference／位相付きrawを再利用。4本はPR分布のIID標本ではない。別wordのReference実装・compile価格はない |
| **元H入力とCDF8 endpoint** | 元C SOURCEに `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz` と `artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz`。当該Git objectの存在・byte数を確認（下表） | 元h/g/constant、corrected κ、固定J/Uを再生成せず取得できる。今回はNPZの科学配列を展開していない |
| **同endpointのPR／ENTRY共通r32下流比較**：H4/1Å/STO-3G、CDF8、LD2、S2/s1/r32/K0、δ=.1、m0..12、q1..4096 | 科学SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、実装 `703938aba8cc38080e29e75acde8e0edc8dce868`。[報告][C32]、`artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/{A,B,C}/` | A/law、B/references・round別資料、C/samples/sample00..31_events・metrics・correctness JSONが現checkoutにある。CのQPY／独立referenceは[元結果を含む固定carrier][C32RAW]へ戻れる。Referenceとsymmetricの価格ではない |
| **update005/r32独立640標本**：同じ分子・CDF8/LD2/r32/K0だが**別endpoint** | 科学SOURCE `62fa4c0144b9e36f9d96b7dff67628051946632d`、結果carrier `459c8472d66689bf149a60fd6cc7311ff5237c65`。[報告][C640]と`artifacts/research_c_r32_price_statistics/2026-11-03/run01/` | native校正・paired測定・共有SEの実装資産。candidate SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。Track D endpointと混合不可。登録結果はUNRESOLVED_PRECISION_TARGET。追加samplingへ進める許可ではない |
| **legacy DF12の18-round RPE**：H4/1Å/STO-3G、8 qubits、4電子sector70、Partial LD3／Full LD12、δ=.02、m0..17、q1..131072、Partialはround別r1..32/K2、Full r0/K0 | [M06-F carrier][M06] `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json`、V1/V2/V3/R1後続資料 | round tuple・保存N/α/β・decoder・価格proxy・不確かさ評価を再利用候補とする。CDF8/LD2/.1/P0のデータではない。元実行はinitial `26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c`、fresh32 `7753eb71844e429facdb6f32c2def6e37379541f` のdirty provenanceを持ち、clean carrierへ付け替えない |
| **Track A legacy H4・保存8-event path**：H4/1Å/STO-3G、DF12、generation prefix6＋one-body、B2/K2/R8/q4、T=.8、δ=.2/r2 | [前回移送調査][TRANSFER] §4、`artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B2_K2_rep0_trajectory.json`。SOURCEはbase `b2e1bf65e21893b6c617223b42313623d3186f12`＋157-file未commit bytes凍結 | ordinary／symmetricのpaired比較資産。全8保存eventは実際にはorder0。D準備・全closure回収が未確認なので最初のReference母平均検証へ選ばない |
| **Pauli-JW／一般mixed／return集約** | [既存Gap表][GAPS]、[Track D適用調査][APPLICABILITY]、固定サーバー監査 | 実装・費用・weight規約の参照先。一般Pauli pathは真空ray保存を仮定できず、同じReference経路へ一括移送しない |

今回の限定Git metadata照合の例を示す。数値payloadの再評価ではない。

| 固定ref／path | Git blob | bytes |
|---|---|---:|
| `ba0e2bf2…`／上記元H input.npz | `4607bed9c83d4d176be1d3d29a43d1e5e1adbf3b` | 2922 |
| `ba0e2bf2…`／上記PR endpoint.npz | `1e1da718b82edf28a9779c76cf26407e1e58e3b7` | 2538 |
| `3649b75a…`／固定word02_raw.qpy | `7a162a9ba675c400f272472534ffba97cdee20ae` | 276034 |
| `459c8472…`／640標本 primary.json | `b7b8f9e2da85504a3587f2de010926385bcb6aa9` | 2869 |
| `4591b8d7…`／上記coherent_analysis JSON | `fa6f6e154fd60e135a9e6901a186976ad8bf34be` | 335692 |

Track DのSOURCE.jsonに保存された元143 SOURCE・16入力の照合結果は既存実行の証拠として引き継ぐ。今回143 fileすべての再hash、QPY意味論検査、全raw再監査を行ったとはしない。元H packet SHA256は `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、Track D／共通r32 PR endpointは `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。

共通r32の32 events JSONは、全件がこのendpointとSOURCE manifest `103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739`を記録していた。保存cohortは `near_practical_PR_r32_fixed32_v1`、master=`2026110304`、PCG64、paired=false／adaptive=false。新方式比較で同じeventsを使うことが新たなpaired比較案であり、元cohortが既にReferenceとpairedだったわけではない。上記carrier内の全32本×4 QPYと32 reference、計160 blobの存在を今回metadataで確認した。decode／回路照合はしていない。

## 3. 真空Referenceの適用範囲・path位相

[既存Track Dの適用条件][APPLICABILITY]とSOURCE本文を照合した。Referenceはfull-Fock真空、systemは**元Hamiltonianの厳密基底状態を仮定**する。状態準備費は評価対象から除外する。

| path族 | 静的に確認できる条件 | 未確認・適用境界 |
|---|---|---|
| CDF8のnumber-density head、signed rotated Z/ZZ tail | number-conserving Gaussianとactual inverse、Z/ZZの各因子は真空rayを保存する。scalar／抽出identityは位相として一度。K>0の積も各因子が同条件なら構造上は候補 | 実検証済みはK0のword02のみ。adapterはN8/CUT16/h=.1/r32/K0/word02を明示固定し、任意law／任意qを受ける実装ではない |
| legacy DFのone-body／DF-squared／Z/ZZ path | RZ/XXPlusYY Gaussian、同basisのactual inverse、normal-ordered headとsigned productの真空作用を追跡可能 | legacyの準備版・scalar／identity規約・cutを固定する必要。旧SOURCEの全closureや新Reference compile後同値性は未確認 |
| 個々のPauli termやcustom/mixed path | 全Hamiltonianの粒子数保存だけでは各sampled pathの真空保存を導けない。既存H6例にはXXYYを含む | 全wordでの特別な相殺や別referenceの可能性は未調査。一般的不可能性は主張しない |
| return集約・importance weight付きpath | Referenceで測定部分を置換できても、label／weight／normalizer／trial規約を別途保持する必要 | 集約ラベルがunitary pathか、weight range、zero-return／post-selectionの扱いまで同じであることを未確認。通常IID finite-RTEのNを転用しない |

一つの保存ordered pathを `Vω=AωBω` に分け、actual `Aω†` を用いる既存gadgetでは、補正前coherenceに `〈r|Vω†|r〉` が掛かる。`Vω r=exp(iχω)r` とancilla `P(χω)`が必要である。これは既存Referenceのfresh-state一shotのX/Y対応であり、新しい一般定理やcoherent QPEの保証を提示するものではない。[保存信号対応][SIGNAL]

χには元scalar・抽出identity、各event.phase、rotationの係数符号、product因子の真空固有値、露出したGaussian／global phaseを含める。物理Hの真空energyだけでrandom pathのχを代用しない。Cの `StrongBuilder` はrelative Gaussianの真空位相を解析追跡する。`basis_change_operations=[]` はC providerのbasisが恒等という意味ではなく、固定U／full-frame providerが必要である。

保存word02ではscalar=`2.29310124732`、identity=`0.041217410629575235`、χ=`−0.23343186579495756` rad。signed角の正負が交互で、保存χがscalar位相と一致している。他pathにもこの値を固定できるとはしない。χ欠落negative checkの不一致も既存結果に保存されている。[conditions][DCOND]／[結果][DRESULT]

**round全体の接続が別途必要である。** 1 h-stepのReference gadgetがdirect controlled-Vと同じ単発observableを与えても、そのgadget自体をq回直列に並べて同じRPE信号になることは実証していない。systemのbranch作用・back-actionまで `diag(I,V)` と同じとは限らない。全roundのordered trajectory `Vω,m` を一つのA/Bに分けて一回のgadgetにする構成、または別構成の対応を明示しなければならない。独立にsampleしたAとB、同じ短pathの無断反復、途中測定への置換は元lawを変更し得る。

## 4. sampling law・weight・normalization

以下は[rte.py][RTE]の `_paired_order_weight`、`finite_rte_distribution`、`_make_event`、`iter_sample_rte_events` とC `configure` の定義の転記であり、今回分布を生成していない。

- `τ=λR δ/r`。有限cutoff Kは非負偶数、order `n=0,2,…,K`。
- paired weightは `w_n=|τ|^n/n! × sqrt(1+(τ/(n+1))²)`、`B_K=Σ_n w_n`、order確率は `w_n/B_K`。
- atomは係数絶対値に比例する正規化分布から選び、符号はinvolution／applicationへ保持する。order nでproduct n個とrotation1個、`event.phase=(-1)^(n/2)`、rotation角 `atan(τ/(n+1))`。
- `event_probability` と `event_coefficient` は別欄で、前者は後者/B_K。既にpからsampleした通常MCでpやLCU係数を再乗算しない。全path確率と価格期待値、weighted estimatorのweightを混同しない。
- 同じlawで独立event／occurrenceを構成した場合のnormalization・attenuationは既存規約で会計する。Referenceのunitary／observableへBやattenuationを乗せない。通常の信号はattenuated raw meanであり、補正済みcomplex値を±1測定平均へ入れない。

同CDF8の保存lawはλR=`0.5000442435909978`、216非identity atom、確率和1、ideal finite-RTE IID、implemented finite-bit law error=`null`。word02の各event normalizationは`1.0000012209184224`。これらは[保存law][CLAW]／[conditions][DCOND]の既存欄である。

SOURCEはPCG64／NumPy choiceとseed policyを持つが、有限bit実装の到達確率をideal lawと同一とは認定していない。seed非重複も数学的IIDの証明ではない。量子shotごとにfresh trajectoryをsampleする条件と、古典的価格標本数nを分ける。一つの固定wordを多数shotへ再利用した混合分布を、無条件平均の単一二項分布とみなさない。[測定接続資料][MEAS]

通常のbounded ±1 outcomeで、**全対象pathに対し**ReferenceのX/Y条件付き分布、p、normalization、fresh-shot規約が保存されるなら、同じ測定平均・shot設計を使うための接続条件になる。これは条件整理であり、今回の全law実装保証ではない。weight付きestimatorではrange／分散・normalizerも保持する必要があり、単発observable一致だけでN保存を主張しない。

## 5. 全RPE総費用への接続状況

研究の既存目的関数は、資源指標gごとに

\[
G_g=\sum_{m=0}^{M}\sum_{b\in\{X,Y\}}
N_{m,b}\,\mathbb E_{\omega\sim p_m}
[C^{\mathrm{no\text{-}prep}}_{g,m,b}(\omega)]
\]

である。Cは1 shotのfull interrogationであり、event単価・kernel単価とは境界が異なる。固定word02の価格を全ωの期待値へ代入しない。[研究目的][OBJECTIVE]

| 接続要素 | 再利用可能な保存証拠 | 今回の目標へ不足する部分 |
|---|---|---|
| Hamiltonian／ordered law | 同CDF8の元h/g/scalar、J/U、216 atom、保存events、CAR reference | 元SOURCE版・signed順序・phaseを保つ全対象pathへのReference対応。別endpoint／legacyと混ぜない |
| 1 h-step価格 | word02の3方式×X/Y全wrapper、同endpoint32本のordinary kernel価格 | 同じ32本上の強いsymmetric／Reference全wrapper paired価格。固定4-word平均やordinaryとの差を代用品にしない |
| round schedule | 同CDF8ではm0..12/q1..4096、r32/K0/.1、既存PF/RTE operator error・attenuation・round資料 | 元H厳密groundへのstate／radius／phase予算対応、全roundの新方式構成と価格。保存own-DF groundでのNを自動採用しない |
| 必要shots | CDF8 PR ideal laneの保存14738 shots、failure=.01、αaxis=.01/26、βtotal=π/6、βstat=π/12。legacy18-roundの別N・decoderもある | 同じenergy精度・成功確率、alias条件、元H入力、synthesis／law誤差の資格。C evidence laneはUNQUALIFIED_ROUNDSでN／費用nullを保持 |
| 総費用会計 | Cのfenced serial composition、`W μkernel+S`、共有校正SE。legacyのfull-wrapper affine qモデルとR1 | Referenceのwhole-round gadgetとfenced serialのどちらを比較するか固定。q倍で消える／増える交換・融合・phase費を未確認のまま扱わない |
| 不確かさ | 既存calibration、paired軸・policy、共有anchor、近似区間・model residualの分離 | 新方式差のpaired価格不確かさ、未観測path／tail、long-q systematic、law移送。観測min/maxをtrue supportにしない |

### 既存の13-round CDF8会計

[共通r32報告][C32]のPR endpointはTrack Dと同じだが、signal／Nはown-DF ground oracleである。保存kernel平均RZ=`7519.09375`（n32）、ideal laneのW=`12539808`、shell込み条件付き総RZ=`94288043542.0`。これをReference価格や元H groundの新評価値とは呼ばない。比較対象ENTRYは別J/Uと別lawであり、Reference対symmetricの同path比較ではない。

既存Cの `serial_cost` は、保存kernelとhead／readout shellを別の費用境界として結ぶ。1 shotの全長一体compile価格ではなく、depthもfenced serial scheduled sumである。Track Dのfull-wrapper価格をこのkernel欄へそのまま入れるとshellを重複計上し得る。同じcalibrationを13 roundへ共有したSEを13個の独立校正として縮めない。

### 既存の18-round legacy評価

旧概要の「全round未評価」だけでは後続状況を表せない。[V2][V2]には保存物理平均によるPartial／Fullの `CONDITIONAL_SUCCESS`、[V3][V3]と[R1][R1]には全18-roundの条件付きRZ点費用がある。一方、Reference／symmetricの比較ではない。

| 項目 | 既存報告の範囲 |
|---|---|
| 保存総shots | Partial13588、Full11162。CDF8のNへ流用しない |
| 保存点費用 | Partial `1263314146772.9375`、Full `1327822299806.0` RZ。約4.858%の条件付き点選好だが全保存scenario区間は重複 |
| compiler | Qiskit1.3.0／opt2／seed17／rz,sx,x,cx／topologyなし。Track Dのopt1とは別 |
| 直接価格域 | Partialの各rでq1/2較正・q8 holdout、r32だけ別q16/32。Fullはq1/2とq16/32。最終tupleとの一致はPartial3 round／Full4 roundだけで、会計はそこでさえq1/2予測を使用 |
| 長q | q>32の価格は未検証外挿。A0 r32/q16,32を最終r1/q16,32へ転用不可。m15..17の最終r32もq32768..131072 |
| 後続STOP | R1後は費用順位決定目的のR2/R3追加compileを非推奨。旧局所精密化STOPを保持する。Track D計画を口実に自動再開しない |

後続m17には[ideal categorical平均の包含結果][M17]もある。SOURCE replay成分／IID uniform53／ideal未compile gate／保存stateのnormalized rayに限定した局所十分条件で、全体は `RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`。float仮想測定・実PRNG列・compiled回路へのbridgeは未解決で、[後続方針][M17NEXT]を保持する。CDF8やReferenceの全law精度保証には移さない。

V3/R1は元LOCALで個別paired価格vectorを未発見としている。今回も**本調査checkout内の**initial／fresh32の既知tasks・checkpoints・worker_results計6directoryは存在しなかった。これは全サーバー／全archiveでの不存在ではない。既存保存先情報からprivate原本・未知hostを新たに探索せず、共分散を復元済みとは報告しない。

## 6. RZ・CX・depth・qubitsと追加費用

[固定H4結果][DRESULT]の数値をそのまま転記する。条件は§2のword02、event16後cut、1 h-step。P0：Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、topology-free、位相保持compile。X/Yは別々にcompileし、下表の値は両軸で同じだった。

| full wrapper | RZ count | CX count | RZ depth | CX depth | total depth | qubits | size |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary＋既存共有・融合 | 6473 | 3074 | 2945 | 2530 | 6542 | 9 | 12812 |
| **symmetric-directional＋既存共有・融合** | 4482 | 1794 | 1682 | 1200 | 3952 | 9 | 9273 |
| Reference | 4721 | 1734 | 858 | 555 | 1955 | 17 | 9562 |

| 資源・条件 | 含まれる／今後区別すべき影響 |
|---|---|
| RZ | 強い対照より増加。P(χ)、inverse、cutで失う融合等を含む。RZ countだけではT count／合成時間を決められない |
| CX | 固定wordで減少。削減余地は既にdiagonal-only制御・symmetric化・basis共有を行った対照から測る |
| depth | 二つのlaneでの実行機会とcut／CSWAPの費用を含む。logical depth、serial depth総和、実機時間、T-depthは別の量 |
| qubits | 追加8-qubit reference register、全17対direct9。全shotsにqubits数を足したものを必要物理qubitsとは呼ばず、peak幅と必要ならqubit-timeを分ける |
| CSWAP | 前後2 cascade、各8対、計16の論理CSWAPを既存価格に含む。同じancillaの共有があり、無償のfanoutや全対同時実行を仮定しない。cat ancilla追加は未実装の別条件 |
| routing | 保存価格はcoupling/layoutなし。17-qubit配置・ancilla接続・SWAP追加でCX/depth差が変わり得る。機種／結合制約を採用する場合だけ同じ条件で再評価する |
| 合成精度 | MPFR160はdeclared phase蓄積であり、全評価160-bitでもfault-tolerant合成保証でもない。RZ/P(χ)のT合成精度、全roundへの誤差配分、Gaussian近似・routing後誤差は別に定義する |
| reset／readout | fresh-shotのsystem・reference・ancilla初期条件を保つ。状態準備費除外は維持しつつ、reference幅・維持／resetを評価する場合の費用境界を明記する。保存結果にその実機価格はない |

同じenergy精度・成功確率で比較するには、価格gごとの総費用とNを揃えて評価する必要がある。主指標・複数資源の重みが未採択なので、CX／depth改善からRZや全資源の単一winnerを宣言しない。

## 7. 再利用可能なコード・検証資産

下表は静的に確認した入口であり、今回実行した処理ではない。元runでは固定SOURCEを使い、現checkoutとの似たAPIを暗黙に置換しない。

| 既存資産 | 再利用する目的 | 必要な最小差分・境界 |
|---|---|---|
| [Track D H4 adapter][DADAPTER]、`scripts/run_track_d_h4_reference.py`、`tests/track_d_h4_reference_checks.py` | actual inverse、control=1制限、2 cascade、observable／state-action検査、χ負対照、full-wrapper指標 | word02固定を一般cohort・whole-roundへ移す差分が必要。旧runnerを新条件でそのまま起動しない |
| 元C SOURCEの `c_fixed_word_pilot`／`c_pr_integrated_performance.native(fixed_words=...)` | 保存eventsとCAR referenceの対応、P0、raw/compiled/adjoint/X-Y保存 | 固定版ではfixed_wordsあり、現carrierの同名nativeにはない。正しいSOURCE／依存組を使う |
| `c_pr_aware_df_stage_b::{prepare_arm,configure,build_kernel}`、`c1_stage3_native_bridge` | 元g corrected κ、number-CDF、signed atom、既存tail/control・scalar | 新DF fitやground solveを標準にしない。元frame順・J/U・precisionを保つ |
| `c1_h4_strong_baseline.StrongBuilder`、`vacuum_phase`、`c1_stage2_native::{runs,combined_diagonal}`、`ax2b_native_df_v5.append_directional_diagonal` | 同じrun grouping・diagonal fsum・relative Gaussian位相を保つ強いsymmetric対照 | head内部融合、tail basis共有、cross-head/tail relative fusion=falseを共通化。弱いheadへ戻さない |
| `c1_h4_native_fair.compile_checked` → `c1_h6_compiler_precision` → F2位相保持unroll | 共通P0・global/relative phase保持 | 全arm同条件。legacy opt2とCDF8 opt1価格を混合しない |
| `c1_h6_state_action.StateActions`、保存full256 CAR reference | 17-qubit全Operatorを作らず、reference/ancilla=0でのsystem入力列・observable確認 | 有限state検査・全system入力検査・全17-qubit operator保証を区別。新計算には別許可 |
| [rte.py][RTE]、`pr_dictionary_validation/circuits.py`、`df_partial_s2_repeated.py` | law、sign/order、actual inverse、既存basis共有、q回のordered request | samplerの再実装は不要。C providerとlegacy preparationの型・identityを確認する |
| `c_pr_aware_df_m3::{serial_cost,calibration_uncertainty}`、`c1_stage3_connection.compose_serial_cost`、`pr_dictionary_validation/accuracy.py` | 13-roundの既存shot／phase条件、fenced serial費用、共有SE | full-wrapper入力とのshell重複に注意。state_model=own_DF_ground_state_oracleの変更は新評価条件 |
| `rpe_resource_accounting.py`、`df_rpe_hadamard_compiled_cost.py`、`rpe_four_round_phase_validation.py` | N×期待価格、axis／round会計、decoder、fresh-IIDと価格nの区別 | ordinary legacy専用のprovider／単位半径仮定をReferenceやCDF8へ自動移送しない |
| M06-F／V3／R1、18-round measurement adapter、m17包含／bridge | 長q価格domain、共有anchor不確かさ、alias／state／measurementの境界 | full runnerはH準備・再探索等を含む。保存値を読むために起動しない |

既存人工tests、scalar／符号／actual inverse／Yの回帰資産は将来の限定変更へ利用できる。本静的調査ではそれらも走らせていない。

## 8. 既存データだけでは評価できない部分

1. 同一law上のReferenceと強いsymmetricの**paired全wrapper期待価格**。現在あるのはword02一点とordinaryの有限校正である。
2. Referenceのwhole-round構成と長q費用。1 h-step価格のq倍、旧affine係数、中央RTEだけのproxyでは埋まらない。
3. 同CDF8 endpoint・元H厳密groundに対する全roundのphase／radius、必要Nとbranch成功条件の対応。保存own-DF vectorの高精度や小DF差だけから同一と認定しない。
4. 全対象pathのχ・actual adjoint・Gaussian／compile精度とimplemented-law誤差。固定wordのPASSや未compile categorical包含で代用しない。
5. 母平均価格のtail／support、paired差と共有校正の不確かさ。32本や640本の観測極値を上界にしない。
6. 実routing、T合成、reference registerの維持・reset等を含む資源モデル。これらを未取得のまま0と扱わない。今回の状態準備費除外は維持する。

## 9. 次に必要な最小検証案と実行条件

**最初から新しいsamplingや最大qの直接compileを行う必要はない。** 以下は最小の接続案で、採択・新規実装・科学実行をまだ行っていない。後続の順序は依存関係を示し、各小工程で承認を取り直す手順を新設するものではない。

### A. 同CDF8・保存32 trajectoryでのpaired移送pilot

- 対象をTrack D endpoint、H4/1Å/STO-3G/Nα=Nβ=2、元順LD2、S2/s1/r32/K0、δ=.1の**1 h-step**へ固定する。`PR/C/samples/sample00..31_events.json`をそのまま使い、価格で標本を選ばず、新sampleを足さず、4固定wordや別update005の640本をpoolしない。
- 初回cutは既済のcomplete event16後を共通にする案。pathごとに既存metadataと位相規約でχを扱い、強いsymmetricとReferenceを同path・同P0・同fusion機会・同X/Y全wrapperで比較する。新規部分は既存adapterの固定word制限を外して保存cohortへ接続する最小差分だけとする。
- 元raw／CARとの位相付き対応、actual A†、真空ray、X/Y induced observable、compile前後の対応を確認した上でpaired価格を取得する。元H ground vectorの新solveはこのobservable検査には不要。ordinaryの追加compileを主比較の必須条件にせず、必要なreferenceとして既存保存物を利用する。
- 全32本なら主比較は2方式×2軸×32の最大128 wrapper compileという**提案scope**。同値な軸価格でも独立標本に数えない。個々の17-qubit全Operatorを生成する設計にはしない。既存state-actionと必要な入力領域を使う。
- RZ/CX/depth/qubitsを別々に記録し、対応あり価格差の有限cohort記述と、母平均推定の資格を分ける。既存32本の来歴・selection条件を保持し、母平均claimに不足なら不足を報告する。sampling追加は自動実施しない。

これにより「word02特有か」「同じ保存cohortでどの指標の差が残るか」を調べられる。これだけでは全lawの保証・同精度全RPE優位を認定しない。

### B. 元H入力・round構成・Nの接続

同じ13-round資料の再利用を先に選び、legacy DF12へ表現を切り替えることを最小案にしない。元H厳密groundの仮定を保ち、保存DF／PF／RTE operator誤差と既存accuracy関数からstate／radius移送を扱えるか確認する。新しいnorm・signal・N・成功率の算定が必要なら、それを科学実行範囲に含める。保存own-DF stateの無断代用、単位半径の自動仮定、ground solveによる暗黙の補完を行わない。

元H向けに有効な保存条件を再利用できる場合は、新方式によるpathごとの測定分布保存と結び、既存Nを保持できる範囲を明記する。保持できない場合は同じεE・αtotal・alias規約・誤差配分の下で必要なNだけを評価する。C親研究のCA=.00159 Ha／failure=.01は候補であり、legacyのCA/10／.05と混ぜない。ユーザー／GPTが目標を選ぶ。

full-round `Vω,m` のA/B分割、CSWAP回数、scalar一回、actual inverse、全qのordered lawを定義する。既存fenced serialを使うならその費用境界を明記し、whole-round gadgetと同じ信号を与えることを確認する。再samplingが必要なroundやK>0を使う設計は、新しい対象lawとして事前に指定する。

### C. 必要な価格domainだけを検証して総和へ接続

A/Bで残った回路族に限り、既存cost provider／境界補正・holdout基盤を再利用する。短qの較正と独立holdoutを先に選び、使用する長qへの移送根拠が不足する部分を明示する。legacy q1/2 affine係数やq>32外挿をReferenceへ流用しない。長qでの直接compileや新samplingは、主資源指標・必要精度・既存保存価格での判別可能性を踏まえて対象を絞る。

価格nと量子N、paired軸、shared calibration、model discrepancy、implemented／ideal lawを区別して、同じεE・成功確率を満たす方式だけに `Σ N E[C]` を適用する。routing／T合成評価は、その資源モデルを主張する場合にだけ追加する。初回pilotと同時に全機種・全精度を走らせる計画は不要である。

### 実行前に確定する条件

| 項目 | 次の担当が確定する内容 |
|---|---|
| 研究判断 | **ユーザー／GPT**：主指標（RZ、CX、depth、必要なら明示した複合目的）、共通εE・成功確率、pilotの目的とclaim水準、whole-round構成を選ぶ |
| 実装範囲 | **Codex**：別途の指示後に既存adapter／強い対照／SOURCE依存を最小差分で接続する。今回は実装0 |
| 科学実行許可 | **ユーザー**：対象SOURCE・入力・cohort／round・compile／統計評価・出力・STOPを特定して明示許可する。チャットの許可で足り、新しいgrantや承認証明を要求しない |
| 比較・精度 | 同じJ/U・κ・scalar／identity・sampling law・ordered path・state仮定・wrapper・fusion機会・compilerを保持。初回は既済P0、component1e−10／signal2e−10／rtol0／位相fitなしを候補とし、長roundへの十分性は別確認 |
| 保全・実行 | 旧SOURCE／入力／raw／grant／STOPを保全し、新しい出力と来歴を分離。旧one-shot・retry/resume・mandatory STOPを新namespaceで迂回しない |
| 資源 | 明示許可後に実空きCPU/RAM等を確認し、独立path／axis・cache・既存state-actionを適切に並列化。人工的な低い資源capを設けず、共有設定・共有venv・他jobを変更しない |
| 失敗・不確定 | SOURCE／入力／位相・correctness不一致や公平なbaseline構築不能は不足を報告して停止。未決区間は同等性や失敗とみなさず、無断の追加標本・精度緩和・retryをしない |

## 10. 文書検査・公開範囲・mandatory STOP

公開するのは本Markdownのみ。内部host/IP、内部絶対path、資格情報、非公開raw、指定保存先TSVの本文、承認原文は公開しない。既に公開されたrepository相対path・固定commit・科学入力hashと必要な出典だけを記す。既存の研究正本・索引・manifest・SOURCE・結果・STOPは更新しない。新しい科学結果・実行経路を追加する作業ではないためである。

文書のUTF-8・相対リンク・固定参照・変更path・`git diff --check`を確認する。参照H4 RESULTはGitHubからも取得した。専用branchへcommit／push後、remote ref SHAと本書の取得本文／Git blobを照合し、その実測結果を最終チャットで報告する。文書検査PASSを科学性能の検証と呼ばない。

**公開・照合後mandatory STOP。次担当はユーザー／GPT**：本書の不足と資源トレードオフを踏まえ、継続の要否、目標精度・成功確率、主資源指標、A〜Cの許可範囲を判断する。Codexは新しい科学計算・実装・sampling・compile・詳細科学レビューへ進まない。重要科学レビューを開始する場合の許可は、科学実行許可とは独立である。

[INDEX]: ../RESEARCH_RESULTS_INDEX.md
[THREE]: ../pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md
[COLLECTION]: ../main_collection_20261103.md
[DRESULT]: ../track_d_h4_reference_validation_20261106/RESULT.md
[DCOND]: ../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/conditions.json
[CWORD]: ../research_c_fixed_word_pilot_execution_20261104.md
[CWORDSAVED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_fixed_word_pilot/2026-11-04/run01
[C32]: ../research_c_near_practical_r32_downstream_execution_20261103.md
[C32RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/C
[C640]: ../research_c_r32_price_statistics_execution_20261103.md
[M06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md
[TRANSFER]: ../track_d_real_pr_transfer_static_20261106/REPORT.md
[GAPS]: ../pr_repository_claim_evidence_gap_matrix_20261101.md
[APPLICABILITY]: ../track_d_reference_control_feasibility_20261106/REFERENCE_APPLICABILITY.md
[SIGNAL]: ../track_d_reference_control_feasibility_20261106/SIGNAL_PRESERVATION_MAP.md
[RTE]: ../../../src/trotterlib/rte.py
[CLAW]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json
[MEAS]: ../rpe_18round_measurement_connection_20261103/IMPLEMENTATION_CONNECTION_REPORT.md
[OBJECTIVE]: ../研究目的・研究課題.md
[V2]: ../pr_rpe_v2_branch_success_20261103/REPORT.md
[V3]: ../pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md
[R1]: ../pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md
[M17]: ../rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md
[M17NEXT]: ../rpe_m17_virtual_measurement_bridge_preflight_20261104/NEXT_STAGE_DECISION.md
[DADAPTER]: ../../../src/trottertracks/track_d_h4_reference.py
[S2AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105/REPORT.md
[S2GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105/SERVER2_EVIDENCE_GAPS.md
