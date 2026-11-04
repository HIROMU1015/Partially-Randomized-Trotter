# 研究C：Direct m0 Hermitian oracle target・bridge 技術準備（2026-11-05）

既存producerが指定した固有値問題は、**保存Hの下三角を複素共役で補完し、対角の虚部を除いた行列**である。これは `H_eigh` の特定であり、研究Cの `H_target` の採択ではない。保存Hの差の大きさ・原因、実H4のground距離・X/Yは今回も未評価。技術準備のみ完了し、次の科学契約は **NOT_READY／非実行／未承認** とする。

起点は `75fe82e9d627bfb0ceca450b96459144e6627914`、独立branchは `codex/research-c-hermitian-oracle-bridge-readiness-20261105`。直前の15-call契約は3 attempts／2 COMPLETE／1 FAIL、残12 calls転用禁止、retry/resume=0のSTOPを維持する。[直前報告](research_c_direct_m0_bound_execution_20261105.md)・[固定契約](research_c_direct_m0_bound_contract_20261105.json)・[旧STOP](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01/STOP.json)。

## 範囲・原本

固定条件はLinear H4・1.00 Å・STO-3G、Nalpha=Nbeta=2、number CDF8／元順LD2、partial ordered symmetric S2、s1/r32/K0、h=0.1、m=0／1 occurrence、sector36→system256→controlled512、own-DF ground oracle／状態準備費用除外、Qiskit1.3.0／P0／opt1／seed17、Arb192、既存Gaussian／relative fusion、MPFR160 declared phase、既存RPE・CA・confidence政策。PR endpoint SHA256は `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。これらを変更しない。

共通運用指示の原本 `/home/abe/Project/Partially Randomized Trotter/research_project_instructions_unified_2026-10-25.md` の§8.1・§8.4・§10を実際に確認した。このファイルは起点commitには未収録であり、固定公開版を読んだとは主張しない。原本hashとproducer照合は [producer_provenance.json](../../artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/producer_provenance.json) に記録した。

今回の実H4へのアクセスはSOURCE／契約／provenanceの読取り、ファイルbytes hash、NPY headerのdtype・shapeの確認に限定した。**NPZの科学配列payloadは読み出していない**。非対称性診断、固有値計算、Arb oracle、X/Y、sampling/build/compile、校正、Stage C、actual CA、重要科学レビューはいずれも0。

## 保存Hの生成経路

固定入力は次のとおり。

|役割|パス|SHA256|
|---|---|---|
|保存sector H|`artifacts/research_c_near_practical/2026-11-03/run01/physical/PR_update016.npz`、array `H`|`ebdebe8f2c3f156c8202834181e143298d667490b3f33fc611dc0fab97b976cd`|
|保存ground候補|`artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/ground_candidate.npz`|`40fb36a7b236a527eea158e7682fe7baf15daa2c52252e96b16ff913f06deef8`|
|sector基底|`artifacts/research_c_near_practical/2026-11-03/run01/physical_reference.npz` の `sector_basis_indices`|固定契約のsaved_inputs参照|

NPY headerだけから、Hは `<c16`／(36,36)、sector_stateは `<c16`／(36,)、energyは `<f8`／scalar、C orderと確認した。complex128で格納されることと、非zero虚部が実際に存在することは別である。

生成は `c_near_practical.run` のphysical callback → `Service.request('assess',...)` → `c_u_step_control_pilot.evaluation_worker` → `c_pr_integrated_evaluation.Evaluation.assess`。評価workerは固定 `input.npz` のh・g・constant、当該PR J/Uを読み、`SectorCAR(4)` と `rank_assessment` を使う。`Evaluation.assess` は `generators=np.asarray(G), H=sum(G)` を `np.savez` へ渡す。

`SectorCAR.generators` の定義は、

\[
\kappa=h-\tfrac12\operatorname{einsum}(\texttt{prrq->pq},g),\quad
G_0=\sum_{pq}\kappa_{pq} E_{pq}+E_{\rm nuc}I,
\]
\[
D_i^{(t)}=\sum_{pq}U_{pi}^{(t)}\overline{U_{qi}^{(t)}}E_{pq},\quad
G_{t+1}=\tfrac12\sum_iD_i^{(t)}\left(\sum_jJ_{ij}^{(t)}D_j^{(t)}\right),\quad
H=\sum_{t=0}^{8}G_t.
\]

ここで `E_pq=a†_(2p)a_(2q)+a†_(2p+1)a_(2q+1)`。補正は**元g**から作り、核反発は一度だけ加算する。sectorは既存excitation tableによるspin-preserving CARの制限であり、回路の途中をsectorへ射影する操作ではない。`PhysicalSector.spin_sector` は36個のoccupation indexをsortし、alpha/betaはinterleaved、ground vectorはこの保存順序に対応する。system埋込みでは既存 `c_direct_m0_runner.embedding` が8-bitをreverseしてQiskit順へ移す。基底の順序をsortし直してvectorだけを据え置いてはいけない。

元の**分子Hamiltonian**（h/g/核反発からのphysical reference）と、当該J/Uによる**own-DF Hamiltonian**を区別する。`SectorCAR.physical` は既存OpenFermion physical Hamiltonianをbit permutation後にsectorへ制限し、DF誤差比較に使う。ground producerへ渡ったのはこのphysical referenceではなく当該PRの保存 `H=sum(G)`。

producerはJ/Uをbinary64 real、Jをexact transpose一致、Uを直交性の数値toleranceで検査する。`rank_assessment` は `max|H-H†|<=1e-10` を数値的に検査するが、厳密一致を強制せず、三角コピー・平均化も行わない。κのeinsum、U積、sparse行列積、密度の加重和、frame・全Hの逐次加算には浮動小数点丸めがある。Hermitian性を数学的に意図する式でも、独立な行列要素の演算経路によりbit一致は保証されない。

これは**原因の候補箇所をSOURCEから特定した**という意味である。実H4について「差が小さい」「丸めだけが原因」「無視できる」とは判定していない。どの演算が今回の不一致を生んだかは未確定。

Hの生成契約は `C_NEAR_OBJECTIVE_PRACTICAL_COMPARISON_II_O_V1`、SOURCE実装commit `9e9fc8391590aeb7f9d36b76d90b5a25bbe2fe5e`、base `2459e018ea2276c359d6fe96253fbcad0bfd65bb`。元SOURCE SHA256は `6442b2061e0be7533344d96a13e5059d738af343ffa98122841a405e0457e8fb`。ground側は `C_NEAR_PRACTICAL_TWO_ENDPOINT_COMMON_R32_DOWNSTREAM_V1`、実装 `703938aba8cc38080e29e75acde8e0edc8dce868`、base `2ceb0a8825785ebd4f21241d8ef0f20cb321bf69`、SOURCE SHA256 `103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739`。関連producer filesの現bytesが各保存SOURCEのhashに一致することを確認した。契約hash・個別file hashはproducer_provenanceに残した。

## 数値ground producerとNumPyの固有値問題

呼出しは `c_near_practical_r32.prepare_arm` → `ground(sector['H'])` → `c_pr_aware_df_stage_b_runner.ground` → `MolecularBackend._ground` → **`np.linalg.eigh(matrix)`**。UPLO指定はない。`_ground` はcomponent toleranceを検査後、最小の `values[0]` と `vectors[:,0]` をそのまま保存する。後続 `ground_gate` は元Hに対するbinary64 residual、norm、alias windowを検査する。residual閾値1e-10、normalization閾値1e-12は数値的gateであり、exact inertia・gap・state-distance certificateではない。Producerの保存とgate通過を、厳密固有状態の保証に読み替えない。

固定NumPyは1.26.4。既存環境の実ファイルを確認し、公式v1.26.4 SOURCEとも照合して分岐を確認した。UPLO既定値L、下三角側gufunc、complex128入力は `D->dD`、LAPACK `zheevd` を使う。対角虚部は用いず、固有値は昇順、固有vectorはcolumnとして返る。保存後の再正規化やphase/sign固定はproducerにない。LAPACKによる数値的正規化は、保存vectorのexact norm=1を意味しない。[NumPy 1.26 eigh仕様](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.eigh.html)、[v1.26.4 Python SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/linalg/linalg.py)、[v1.26.4 LAPACK接続](https://github.com/numpy/numpy/blob/v1.26.4/numpy/linalg/umath_linalg.cpp)。

数学的には、保存された各real/imag binary64 componentをexact rationalとして `A=H_saved` と置き、

\[
(H_{\rm eigh})_{ij}=\begin{cases}
\Re A_{ii}&i=j,\\ A_{ij}&i>j,\\ \overline{A_{ji}}&i<j.
\end{cases}
\]

これで既存producerの**入力に対応する理想的Hermitian固有値問題**は一意に定まる。LAPACKがこのexact問題をexactに解いたという主張ではない。UPLO=Uなら上三角を採用する別の定義になる。未使用三角に非対称成分があれば一般にL/Uは異なり、どちらも平均 `(A+A†)/2` とは異なり得る。

## H_saved・H_eigh・H_targetの比較

|候補|契約・vectorとの関係|一意性・構成可能性|追加bridge／科学計算・RPE影響|
|---|---|---|---|
|H_saved|保存bytesそのもの。旧Direct m0はexact real symmetryでFAIL|exact binary64成分として一意。非Hermitianかもしれない|Hermitian ground/gapとして扱わない。H_eighとの差のノルム診断は可能だが、それ自体はtarget採択の根拠にならない|
|H_eigh|既存数値oracleが入力した固有値問題に最も直接対応。研究保証対象への正式採択は未決定|固定H＋NumPyのL規則で一意。再SCF・再fit不要|当該matrixへのoracle certificate、literal/head/referenceのtarget対応が必要。採択しても旧STOPは解除されない|
|元のHermitian Hamiltonian|当該own-DFの理想式か、元分子のphysical Hかを先に指定。後者はown-DF policyと異なる|h/g/J/Uと演算子式は残るが、exact係数・κ・積・基底・Uの意味を固定する必要がある|H_eighとのHermitian差分boundか、選択targetでの直接certificate。physical Hへの移行を無断で行わず、DF/reference誤差とRPE energy targetも再bindする|
|保存係数から定義する別のHermitian target|候補を明記すれば再現可能だが、契約への整合は別判断|例：Hermitian κと対称Jによる厳密CAR多項式。Uが近似直交でも密度のHermitian性と対称Jの対和でHermitian性を示せる。ただし前提は実入力で未検証|exact/intervalで構成しH_eighとの差、gap、state距離を資格付け。literal Gaussian/relative-fusionおよびheadとの同一性は別に検査|

元の「Hermitian Hamiltonian」という言葉だけでは、物理H・理想own-DF式・保存係数をexactとした式を区別できない。特に保存κを使うか、保存h/gからexact κを作るか、近似Uをどの演算子に解釈するかを決める必要がある。SOURCEがあることだけで一意な研究targetが確定するわけではない。

平均化や片三角completionを研究targetとして採択していない。adapterのcompletionは **eigh semanticsを表現する明示opt-in** に限り、旧runner・固定SOURCE・契約へ接続し直していない。

## Rigorous bridge

### 行列差

各componentをexact rationalとして、任意の行列差Dに対し

\[
\|D\|_2\le\|D\|_F=\sqrt{\sum_{ij}[(\Re D_{ij})^2+(\Im D_{ij})^2]}
\]

をArb192のoutward sqrtで包囲する。これは非Hermitianな `H_saved-H_eigh` にも有効。`H_target-H_eigh` をWeyl／gap perturbationに用いる際は、両側を厳密Hermitianとして検査する。Frobenius boundは保守的であり、実問題でgap条件を満たすとは未判定。

### 隔離・residual・距離

`A` を明示したHermitian問題、保存vectorを未変更のv、`u=v/||v||`、`mu=<u,Au>`、`r²=||(A-mu)u||²` とする。shift σは数値計算で提案してもよいが、**証明はexact rational LDL inertia** による。`A-σI` の負固有値数=1、他は正、かつmu<σなら、`lambda0<σ<lambda1`、gap下界 `g=σ-mu>0` を得る。

\[
\|P_{\rm exc}u\|^2\le r^2/(\sigma-\mu)^2,\qquad
\|v-e^{i\theta}g_A\|\le |\|v\|-1|+\sqrt{2r^2/(\sigma-\mu)^2}.
\]

norm誤差もArb192で包囲し、原vectorをnormalizeして置換しない。global phaseは距離のために整合させるが、両経路の期待信号には影響しない。縮退・隔離不能・mu条件不成立・overlap bound>=1・zero LDL pivotは資格なしでSTOP。near-degenerateでもexact条件で判定し、任意の浮動小数点toleranceでgapを作らない。zero pivotはnonsingular行列でも起こり得るため、「必ず縮退した」とは解釈しない。

実対称なら既存 `c_direct_m0_oracle.qualify_ground` をそのまま再利用する。非実Hermitianなら

\[
R(A)=\begin{pmatrix}\Re A&-\Im A\\\Im A&\Re A\end{pmatrix}
\]

のexact実対称行列に既存LDLを適用する。固有値は2重なので負数=2・正数=2n-2を要する。これは物理的ground縮退ではない。vectorは `(Re v,Im v)` としてnorm・Rayleigh・residualをexact評価する。旧oracleが非実Hを拒否する仕様は変更しない。数値proposalや保存energy値を証明へ昇格させない。

### 別targetへの移送

厳密Hermitian A/B、Aのgap下界g、`||B-A||<=epsilon` に対し **g>2epsilon** を要する。このときBのgroundは一意、gap>=g-2epsilon。Bのexcited eigenvalueは `lambda1(A)-epsilon` 以上なので、Aのgroundに対するB residualから

\[
\min_\theta\|g_A-e^{i\theta}g_B\|
\le\sqrt2\,\epsilon/(g-\epsilon)
\]

を得る。保存vのA-ground距離にこれを一度だけ加算する。条件を満たさなければ移送資格なし。B自体で保存vを直接資格付けできる場合は、そのcertificateを使う方法もある。同一matrixなら移送項は0。いずれの方法もBの正式採択や既存reference identityを自動的には確立しない。

### Direct m0・X/Yへの接続

既存 `c_direct_m0_runner.embedding` のisometryと既存CARの基底対応を再利用する。保存v→採択target groundの距離etaを**初期state radiusへ一度**入れる。controlled forward/inverseとideal head/X/Y shellはunitaryなので、その状態誤差は増幅しない。既存 `c_direct_round_signal.native_action` は各native演算のArb丸めを別項δとして足す。actual inverseは同じrealized forwardへの対応を必要とし、独立再samplingはしない。

非unitary literal referenceはnorm上界aを必要とし `E'<=aE+δ`。norm=1や信号範囲[-1,1]を仮定せず、中心cのX/Y観測は各軸 `2||c||E+E²` で包囲する。X/Y signed区間を保持し、複素差は両軸からoutward sqrtで包囲する。unit initial state同士への独立補助式なら、同じmap M・norm<=bについて各軸 `2b² eta`。この補助式は未正規化中心へそのまま適用しない。

Oracle state誤差、native synthesis誤差、reference target移送は別のedgeである。初期radiusにetaを入れた上でoracle分をsignalへ再加算しない。既存Stage Pのoperator上界は直接state作用を評価した同じedgeへ重ねて加算しない。B2はhead/target/状態/word/scopeが一致するedgeだけ接続し、必要なhead/reference bridgeを無条件に削除しない。`H_target` が変わるなら既存referenceとの差・RPEのenergy対象を明示し、初期state距離だけで全reference差が解決したとは扱わない。m0固定1wordの技術接続は、fresh IID期待biasや全law資格を与えない。

## 最小実装・人工検証

新規 [c_hermitian_oracle_bridge.py](../../src/trottertracks/c_hermitian_oracle_bridge.py) はdecoded exact成分だけを受け、L/U明示completion、exact Hermitian候補検査、SOURCE/input/target identity、行列差bound、既存oracle再利用、複素realification、ground移送、state→signal補助boundを提供する。ファイル読込み・H4 runner・target採択・科学retry機構を持たない。係数から実H4行列を再生成する機能や区間係数Hamiltonianの一般solverも追加していない。

[prepare script](../../scripts/prepare_research_c_hermitian_oracle_bridge.py) はJSON提案を読むpassive入口だけであり、科学ライブラリやNPZ/QPYを読み込まず、science/resume/retry引数を拒否する。旧固定SOURCEのCalls／STOP／contractは変更していない。新しい科学run managerやapproval証明システムも作っていない。

**新規人工34 PASS。既存関連93 PASS／4 FAIL。合計127 PASS／4 FAIL、新規failure 0。** 直前実行commitのworktreeでも同じ4 FAIL／93 PASSを再確認した。失敗は旧pre-executionテスト4件が既に存在する `science-run01` に対して `FRESH_NO_RETRY_RESUME` を受けるもの。旧rawを隠す／削除する、SOURCEやテストを修正する、といったPASS化は行わなかった。[validation.json](../../artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/validation.json) とbaseline/current pytest原本参照。

|依頼の検証項目|対応|
|---|---|
|1–6：対称／微小・大非対称／L/U／completion／一致不一致|exact差分と既知構成、人工NumPy eigh residualとの対応|
|7：非実成分|複素off-diagonalの共役、対角虚部無視、real dtypeへの複素入力拒否、複素ground certificate|
|8–12：residual／gap／距離／縮退・near-degenerate／phase|exact既知ground、未正規化誤差、2^-100 gap、zero pivot、4 global phase、別targetへの解析的ground距離|
|13–17：target不一致／信号／非unitary／wrong SOURCE入力／無断対称化|hashと構成規則のbinding、移送の拒否、非unitary観測半径、平均化を候補に混入させる入力の拒否|
|18–20：未承認実行／retry-resume／STOP|passive CLI拒否、NOT_READY、旧STOP維持。既存回帰もpermission・固定条件・控えめな人工m0 full pipelineを検査|

検証環境は既存Python3.11.1／NumPy1.26.4／python-flint0.8.0。人工2〜3次元と既存人工回帰であり、大規模性能や実H4 oracleの成功を示すbenchmarkではない。検査時のCPU32、RAM62GiB／available約52GiB。支配的な新処理はexact LDLとFraction bit成長で、実36次元ならreal path、複素なら72次元realification。独立な実験cell以外のpivotを無理に並列化せず、新規科学契約を固定するときに現資源・bit成長・必要cacheを確認する。任意の小CPU/RAM上限やwall capは設けない。共有環境や他jobを変更していない。

## 次の限定科学検証：未承認・NOT_READY

[非実行契約](research_c_hermitian_oracle_bridge_contract_unapproved_20261105.json) は、固定identity/input、Arb192、数式、失敗partial STOP、retry/resume=0を保持する。技術 [SOURCE.json](../../artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/SOURCE.json) は技術snapshotであり、将来の科学SOURCEや実行許可ではない。

順序は次のとおり。各段階を自動で一括実行しない。

1. **提案A（最小の診断）**：新fresh scopeで保存HとH_eighのexact差分・ノルムだけを評価する。metadata/input binding、arrays load、L規則構成、rigorous normの論理roleを必要とし、target採択・eigensolve・X/Yを含めない。今次は実行していない。
2. **ユーザー／研究Cの判断**：H_eighを保証対象とするか、元のown-DF式／別係数targetとするか。physical Hへの移行は別条件変更。基底、係数・κ・Uの解釈、literal/head/referenceとの対応も固定する。診断Aはこの判断の前に別契約として切り出せる。
3. **提案B**：選択したscopeで保存vのinertia／residual／norm／phase-aligned距離を資格付けする。shift proposalが必要なら独立にcallへ含める。gap／zero pivot失敗ならSTOPし、別shiftや別targetへretryしない。
4. **提案C（必要な場合だけ）**：別Hermitian targetへのrigorous差分とgap条件を評価する。Bでtargetへ直接資格付けするなら、不要な移送計算を追加しない。
5. **提案D**：oracleとreference対応が資格を得た後、別SOURCE・新fresh・別許可のDirect m0 X/Y契約を準備する。旧15-callを再開せず、旧12 unitsも転用しない。

現時点では **H_target=null、logical_calls.total=null、scientific_SOURCE=null、fresh_output=null**。上記は必要roleと依存関係であって、実行可能なcall数・実行scheduleではない。選択target・参照edge・段階境界が未確定のため、恣意的な総call数は登録しない。Codexが段階を限定し、固定SOURCE／call contents／新規outputを具体化した後、そのscopeへの新しい科学実行許可が必要となる。今回の技術許可や以前の「実行してよい」を、この未確定scopeへ転用しない。

## 保全・公開・STOP

既存科学SOURCE、保存NPZ、旧契約、raw、grant/ledger/marker/STOPは無変更。並行Compiled→raw作業のcheckout・cache・lock・processも変更しない。関連index4件と新規成果物だけをcommitする。baseとの差分と固定manifest/input hashの保全結果は [preservation.json](../../artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/preservation.json)、技術STOPは [STOP.json](../../artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/STOP.json) に記録する。

公開後はremote SHA/tree、報告全文、主要blobを固定commitで照合し、[public_verification.json](../../artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/public_verification.json) に残す。これらは成果物の照合記録であり、permissionの証明ではない。

**MANDATORY STOP。** 次の責任はユーザー／研究Cによるtarget・参照条件の選択、Codexによる選択scopeの具体的な新規契約準備。科学実行・重要科学レビューへ自動進行しない。
