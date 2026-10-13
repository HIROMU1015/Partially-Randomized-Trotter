# 量子化学相対時間発展block：方法差・実装費用の追加技術監査 v1

2026-10-13 JST。基準commit **`25cbc6e79480230afb841201b6917b987cc2a3d7`**、独立branch `qc-relative-block-method-cost-audit-20261013`。
[受領指示原本](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/instruction_intake.md)、[基準の事前検証](qc_relative_block_prevalidation_v1_20261013.md)、[完了済み方針レビュー](quantum_chemistry_ground_energy_algorithm_direction_review_2026-10-13.md)に従う。

**技術結論：現constructorは既知THRIFTの相対積、二準位の条件付き対角化、可逆関数評価へ構成的に還元できる。強い対照へ同じ最適化を与えると、独立した生成手順・費用差は特定できなかった。**
これは調査対象の構成についての `REDUCES_TO_KNOWN` 判定であり、全先行研究の網羅、新規性の最終認定、全実装の最適性・不能性の証明ではない。
詳細function backendへ投資するPhase 2の入口条件を満たす具体的差を確認できず、backend実装・追加synthetic・pilotへ進まない。既知primitiveの再利用、条件付き費用式、未解決事項までを整理した。[GPT判断用資料](qc_relative_block_method_cost_handoff_v1_20261013.md)。研究継続・縮小・終了の判断はGPTへ戻し、今回そのレビューを開始していない。

## 1. 範囲・証拠の種類

有限spin-orbital basis、実係数・粒子数保存の電子Hamiltonianを

\[
H=E_{\rm nuc}+A+\sum_{j=1}^{M}B_j,\quad
A=\sum_i\epsilon_i n_i+\sum_{i<k}J_{ik}n_i n_k
\]

と分ける。Jは符号付き。B_jはdisjoint supportのsingle/double monomialと随伴、又は同じsingle transitionのdensity-assisted項をまとめたもの。複素係数、一般の多transition group、実分子adapterはこのconstructorの保証対象外。
既存13 syntheticは手書き5mode以下の構成確認で、実分子の性能・移送結果ではない。geometry/basis/DF rank・policy/L_D/PF窓はそのsyntheticに適用なし。
今回はその保存reportと生成source SHAを読むだけで、13件を再実行していない。新分子、固有値、数値的事後評価、科学batch、compile、T合成、新syntheticは全て0。
[機械可読status](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/audit_status.json)と[証拠identity](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/base_evidence_identity.json)を参照。

`SUPPORTED` は明示した数式・静的構成の技術的支持、`REDUCES_TO_KNOWN` は既知手順への具体的対応、`UNDETERMINED` は比較・実装・証拠が未閉鎖。文献の既知性と完成した本回路の実装検証を区別する。

## 2. 一次資料・sourceの対応表

資料IDと取得箇所・版・取得失敗は[文献台帳](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/literature_scope.json)。以下の判定は、一次資料の記述と本構成を照合した**監査側の推論**を含む。

| ID・論点 | 共通する部分／具体的な対応 | 異なる部分・差の前提 | 根拠 | 未確認事項・技術判定 |
|---|---|---|---|---|
| A1 相対積・時間依存PF | H0=A、αH1^j=B_jでTHRIFT Eq.8の因子がそのままC_j | 化学densityを選ぶ特殊化。新PF・次数改善は追加していない | [L1 THRIFT](https://arxiv.org/html/2403.08729)、基準module `relative_matrix` | 同精度step数は未評価。相対積は `REDUCES_TO_KNOWN` |
| A2 spectator sector | 保存occupationをprojectorに分解し、各sectorで小行列を対角化 | Patelのfermionic one-body可解classへdouble excitation全体をそのまま入れない。doubleはoccupation対への二準位encodingを追加 | [L2 III.2–3](https://arxiv.org/html/2305.18251v2)、`detuning_plan` / `valid_encoded` | 一般class全部のcoherent回路費用を確認したとはしない。sector手順は `REDUCES_TO_KNOWN` |
| A3 励起pair・追加control | number-preserving二状態にU(2)、single/doubleをoccupation対へ移す | JWの長いparityとmode順・routingを明示。定数supportのCNOT変更を独立算法としない | [L3 §2.2/3.1–3](https://arxiv.org/html/2106.13839v2)、[OpenFermion gate一覧](https://quantumai.google/reference/python/openfermion/circuits/gates)、`encode_pair` / `parity_plan` | OpenFermionの局所gateを全mode parity・QPE control込みの無料gateにしない。`REDUCES_TO_KNOWN` |
| A4 affine detuning | quadratic occupation energyの二pattern差は定数＋spectatorの線形和 | 特殊な疎な差係数なら加算数が減る。元Jが疎／差が疎という実入力証拠が必要 | 本資料§3、`detuning_plan`、[L4 関数算術](https://arxiv.org/html/1805.12445) | 全枝表を使わない構成は `SUPPORTED`。同じdescriptor/adderを既存法にも与えられ、方法差は `REDUCES_TO_KNOWN` |
| A5 函数をcompute–rotate–uncompute | occupation関数を可逆に計算して位相又は可変回転に使い、消去する | 係数関数とatan2/hypotの領域・guardが本用途固有。別用途の定数を移せない | [L5 §2.1–2.2.2](https://arxiv.org/html/2212.14030v3)、[L6 §4.1/Fig.14–15](https://arxiv.org/pdf/1204.0567)、L4 | 回路primitiveと資源modelは既知。完成した本関数backendは `UNDETERMINED` |
| A6 共通平均位相の相殺 | dbar Iが両指数で消える。強いTHRIFT回路でも代数的に同じ因子を生成できる | outer e^{-itA}・E_nuc位相は残る。素朴global A反復だけを比較相手にしない | §3、L1、`relative_matrix`、`df_trotter.ops.append_diagonal_primitives` | 相殺は `SUPPORTED`、独占的な構造利益は `REDUCES_TO_KNOWN` |
| A7 共通近似Δと誤差抑制 | 同じ近似Δを使うinteraction-picture積には§4のboundが成立 | 両指数へ独立した近似を入れる方式には継承しない。既存compressed法は同じ共通値を使用可能 | §4の直接導出、L1のinteraction-picture定義、`precision_bound` | 本用途への誤差配分は `SUPPORTED`。現候補だけの方法差は `REDUCES_TO_KNOWN`、文献全体での初出は `UNDETERMINED` |
| A8 定数ロード・加算 | occupation制御で定数wordをロードし、普通のadder、アンロード | 基準の全gate制御adder上界より小さい条件付き会計。全対照へ適用 | [L7 Cuccaro Table1](https://arxiv.org/pdf/quant-ph/0410184)、[L8 §3](https://arxiv.org/html/2001.09580)、Qualtran固定source AddK | §5の設計上界は `SUPPORTED`、独立な新手順は `REDUCES_TO_KNOWN`。compile実測ではない |
| A9 共有θ・線形Zの直接生成 | θを±で再使用、線形Δの最後のZはdensity-phase列にもできる | 同じ共有・fusion・精度設計をknown/native対照へ許す | §4–5、`angle_recipe`、`ops.py`のdiagonal primitives | 条件付き構成は `SUPPORTED`、独占利益は `REDUCES_TO_KNOWN` |
| A10 DF/Gaussian native | orbital/Givens変換→diagonal density→逆変換という既存対照 | splitが違うので一block単価だけで性能を比較しない。full Hとtruncation誤差を揃える | [L9](https://arxiv.org/html/1808.02625)、`ops.apply_df_block` / `apply_one_body_gaussian_block`、`two_body.py` | 既存強対照の成立は `SUPPORTED`、総費用順位は `UNDETERMINED` |
| A11 controlled evolution・clean workspace | 入力不変registerで算術、valid∧QPE controlで回転、逆算 | spectator測定による古典分岐はcoherent QPEと同じ機能でない | §3、L6 controlled QVR、基準`structural_ir`、`ops.py`のancilla phase | 静的cleanliness条件は `SUPPORTED`、全gate-level backend検証は `UNDETERMINED` |
| A12 独立な生成・費用差 | 同じencoding、係数、算術、関数、制御・合成をstrong compressed法へ割り当てられる | 別の時間／空間trade-off、少数group、更新cache、関数不要方式を具体化・比較できれば別検討 | §3–7、基準sourceと取得Qualtran source | 現constructorの差は未特定：構成は `REDUCES_TO_KNOWN`、全代替実装の最適性・将来差は `UNDETERMINED` |

L5のnilpotent transition E+E†のSVDだけで、detunedなA+B_j全体がXへ対角化できるとは主張しない。densityのZ成分は残り、次節の条件付きSU(2)が必要。論文の格子ゲージ資源値を化学blockへ移していない。
L2のprojector和は任意の追加構造がなければ指数費用になり得るが、アフィン構造のある本例に同じ下限を適用できない。測定後にCPUで角度を選ぶmeasurement回路はQPE用coherent対照から分ける。

## 3. 還元を具体化する

支持S=P∪Qのchi0/chi1をu/vとし、spectatorは不変とする。

\[
\Delta_j(s)=\Delta_{0j}+\sum_{l\notin S}d_{jl}n_l,\quad
d_{jl}=\sum_{i\in S}J_{li}(u_i-v_i).
\]

支持外同士のdensityとscalarは差で消える。Δ0は支持内の一体・density差。z_jは**非zeroのd_{jl}数**、z_j≤N−|S|。厳密zero除去を許すが、新しい閾値で係数を捨てるとmodel/implementation誤差の追加が必要。
各pairのA+B_jはdbar I+ΔZ/2+bX、無効occupationではB_j=0。pivot CNOTによるencoding E、valid q、parity符号を除くD=bの正振幅g≥0、rho=sqrt((Δ/2)^2+g^2)、theta=atan2(g,Δ/2)とすれば

\[
C_j|_s=R_Z(-t\Delta)R_Y(\theta)R_Z(2t\rho)R_Y(-\theta).
\]

rightmostから実行し、parity負なら両端Z conjugationを加える。EとE†、valid=0でI、g=0又はt=0で厳密Iを含める。
strong-known構成を「同じΔ/gを計算、同じθ/rhoを計算、同じ四回転、同じ逆算」と具体的に定義でき、現候補のIRの各stageを一対一に写せる。係数アクセス・classical descriptorのO(MN)会計と、最大O(poly(word,degree)) workspaceも同じ。
**数学的同値だけでなく、現提案の回路生成手順自体もこのknown compositionへ写る**。ただし、この対照の全backendも本repoで完成したとはしない。

旧`mechanisms.conditional_mixed`は二spectator枝のe^{-it(A+αB)}を生成し、spectatorの平均phaseを含む。今回のC_jそのものではない。`candidate_c`は二mixedとe^{itA}を組み合わせる。枝列挙版より安くする可能性と、strong compressed法より安くする主張を区別する。旧sourceをimport・実行していない。

不変なspectator、encoded非pivot bitからΔ/g/parity/validを作るので、target回転後も逆算入力は同じ。各jの算術をQPE control外で行っても、control0枝はencode/decodeとcompute/uncomputeが相殺する。算術費用は両枝共通の固定回路として請求する。
別のB_kがspectatorを変更するため、全jの角度を最初に一度だけcomputeし、全時間stepで再使用できるとは仮定しない。incremental更新cacheは新設計・更新誤差・uncompute・保存幅の証拠が必要な未着手案。

## 4. 共通精度・共有角度：静的な保証と留保

この節は保存入力の新数値評価を行わない直接導出。bmax≥max(|b|,|btilde|)とする。V_{Δ,b}(s)=e^{isΔZ/2}bXe^{-isΔZ/2}について

\[
\|V_{\Delta,b}(s)-V_{\widetilde\Delta,\widetilde b}(s)\|
\le |b-\widetilde b|+b_{\max}|\Delta-\widetilde\Delta||s|.
\]

unitary Duhamelを0→tで積分すると、基準の係数bound

\[
\|C-\widetilde C\|\le |t|\eta_b+\tfrac12 b_{\max}t^2\eta_\Delta+\epsilon_{\rm impl}
\]

が得られる。Δで割らない。両指数で同じ近似Δ、phase保持、零branch保持が条件。dbar誤差を両指数に同じ値で入れれば相殺するが、外側A・scalarや元Hamiltonianの近似誤差は別。

同じ近似θを±で使う場合、W(θ)=Ry(θ)Rz(2t rho)Ry(−θ)はaxisがθで回る。Pauli axis差のnormは2|sin((θ−θtilde)/2)|なので

\[
\|W(\theta)-W(\widetilde\theta)\|
=2|\sin(t\rho)|\,|\sin((\theta-\widetilde\theta)/2)|
\le\min(1,|t\rho|)\eta_\theta.
\]

したがって同じ理想近似係数のblockに対し、共有角度のfunction errorには

\[
\epsilon_{\rm impl}\le
\min(1,|t\rho|)\eta_\theta+|t|\eta_\rho+\eta_\chi/2+\sum\epsilon_{\rm synthesis}
\]

という十分boundを使える。χは最後のZ角度−tΔの**追加の角度評価誤差**。rho・χを先に正しい値から変更し、その後共有θを変更するtelescoping順を使う。rhoは既に近似した係数の理想値で、sup_s rhoでuniform化する。係数ΔのηΔをχにも二重に請求しない。RY二回の独立合成誤差は相殺を仮定しない。
これは共通θを使う全対照にも適用できる技術的corollaryで、独立した新規性・資源利益の認定ではない。実function丸めをこのηへ外向きに閉じたbackendはまだない。

term jの全occurrenceに対しS2_j=Σ_l t_{jl}²、S1_j=Σ_l |t_{jl}|。z_j差係数と定数をf_j fractional bitへnearest roundingするならηΔ_j≤(z_j+1)2^{-f_j-1}。detuning予算εD_j>0を満たす十分条件は、分子データを代入しない式として

\[
f_j\ge\max\!\left(0,\left\lceil\log_2\frac{b_{\max,j}(z_j+1)S2_j}{4\epsilon_{D,j}}\right\rceil\right).
\]

分子が0なら対応項の誤差は0、logは評価しない。Σ_j εD_j、Σ_j ηb_j S1_j、全function/synthesis予算、PF/model/A誤差を合計する。係数を既に丸めたHからΔを導出する場合は、その伝播boundを使い二重計上しない。
guardは丸め後の|Δ0|+Σ|d|を**全compute枝**で覆う。二進modular adderは可逆でも、overflowが有効branchに出れば目標detuningではない。word w_jにはsign・integer・fraction・guardを全て含める。
関数側はzero detection、quadrant、rhoのrange、最小非zero量、division入力域、内的rounding/garbage、angleのmoduloとancilla phaseを別に閉じる。θの入力感度を無条件に係数boundと同じにしない。
B=αB0でb=O(α)でも、ηΔ固定の誤差はO(α ηΔ t²)。理想p次THRIFTのlocal O(α²t^{p+1})を保持したいならηΔ=O(α t^{p−1})という追加精度条件が生じ得る。小αだけでword/function費用が減るとは言えない。

## 5. 再利用可能primitiveと条件付き資源台帳

[固定Qualtran取得台帳](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/external_primitive_source_index.json)：`quantumlib/Qualtran` commit `cf5164977d62fc8f3fe6f8815e92ee029d11d504`の4ファイルをread-only取得し、SHAを保存。install/import/compileはしていない。latest docsとこのcommitの版同一性は仮定せず、費用採用時は固定sourceを優先する。

| primitive | 取得source/原典の状態 | この用途への扱い |
|---|---|---|
| Add/AddK | Addの分解、AddKのconstant load–Add–unloadを確認。Addはtemporary AND/adjointを使う | constant load方式は使える。signed word/range、測定feedforward、逆算の条件を選ぶ |
| Product | concrete bitsizeのcomposite分解とcall graphあり | integer primitive再利用候補。signed/fixed-point truncationやworkspaceの追加を含める |
| Square/実数multiply/inverse | 各種資源式あり。取得Square、実数multiply/inverse classにはcomposite分解がない。inverseには大きいworkspaceのTODOも残る | 資源式を完成したgate-level関数と呼ばない。generic quantum算術の可能性は原典で支持、drop-in費用はUNKNOWN |
| ArcSin | classical numpy参照とcall graph。取得classには回路分解なし | atan2/hypotのbackendではない。既定parameterのdocs値を本関数の費用へ転記しない |
| QvrZPow / QvrPhaseGradient | 固定sourceにcomposite分解あり。phase-gradient版は追加state/registerとprecision選択を使う | phase oracle e^{iγx}はtarget RZそのものではない。target eigenvalueとenableを含むphase差へ変換し、state preparation/uncomputeを請求する |

L4のHorner・(inverse)sqrt、L5のNewton、[Quantum CORDIC](https://arxiv.org/html/2411.14434)は関数生成の既知選択肢。division/domain、quadrant、cleanliness、uniform誤差までこの用途へ接続した証拠が必要。原典の別domainの定数はこのconstructorに適用しない。
[Gidney adder](https://arxiv.org/html/1709.06648)のmeasure-and-fixupはworkspaceの測定と補正でsystem coherenceを維持する構成であり、spectator測定とは異なる。uncompute 0Tを全算術に適用しない。feedforwardなしのunitaryモデルと混ぜず、全方式へ同じresource modelを与える。

### 5.1 detuningのknown constant-load設計上界

w≥3、d_lをw-bit二の補数整数に符号拡張、fresh定数word Lとaccumulator D、clean carryを使う。
spectator n_lからd_lの1bit位置へCNOTしてL=n_l d_lをロード、普通のCuccaro modular Add(L,D)、同じCNOTでLを0へ戻す。AddはLを復元するので、occupationがsuperpositionでもこのアンロードが成立する。逆算は逆順。
L7の1 Addは2w−3 Toffoli、5w−7 CX、2w−6 X、ancilla1。h_lは整数d_lのw-bit Hamming weight。**一係数のcomputeとuncompute合計**は

\[
N_{\rm Toff}=4w-6,\quad N_{\rm CX}=10w-14+4h_l\le14w-14,\quad N_X=4w-12.
\]

Δ0のload/clearに≤2w X、全z_jでToffoli z_j(4w_j−6)、CX Σ_l(10w_j−14+4h_l)、追加workspace2w_j+1（accumulator、定数word、carry）。他のflag/function registerは別。
全gateを直列に置く保守的depthは≤22z_jw_j+2w_j。routing/T分解depthは別。これは静的な特定設計の上界であり、最適値・lower bound・compile測定ではない。
基準の≤22z_jw_j Toffoli上界は全adder gateを制御する粗い方式で、有効な旧記録として保持する。今回のknown constant-load方式はそれを使う必要をなくすが、両方式を比較した測定のspeedup倍率はない。**strong compressed法にも同じ方式を割り当てる。**

### 5.2 その他とUNKNOWN

| 部分 | 根拠を持つ限定会計 | UNKNOWN・閉じる条件 |
|---|---|---|
| pair encoding/decode | 2(k−1) CX、k=2/4、pivot Xの往復≤2X | routingとmode permutation |
| parity | p_j spectator bitsならcompute/uncompute2p_j CX＋Z conjugation2CZ。p_j≤N−k | groupの追加符号/zero判定 |
| valid/QPE flag | r=k−1のpattern一致はr=1でCX、r=2/3でAND ladder。valid compute/erase≤6 Toffoli、valid∧QPE control往復2 Toffoli | grouped gのzero/sign predicate、さらに追加するcontrols |
| affine group振幅 | z_g係数なら同じconstant-load費用とrange word | 全doubleを少数groupへまとめる根拠はない |
| function compute/erase | F_jを一回forwardの費用とする。unitary逆回路なら合計2F_j。θは一度生成して±に使用 | F_jのToffoli/T/CX/RZ/depth/ancilla、guard、時間scaleの費用はUNKNOWN。sqrt/division/Horner全backendを新設しない |
| 四可変回転 | angle word a_j、bitwise CRZ＋basis Clifford、enable∧angle bitの往復。設計envelope≤8a_j RZ、8a_j CX、8a_j Toffoli | 合成T、angle全域誤差、routing、phase-gradientとの同品質比較 |
| 最後のlinear Z | −tΔは定数＋z_j spectator phaseとして直接生成可能。valid/QPE enableありで定数CRZ一回、各n_l∧enableのcompute/eraseとCRZ | angle-register方式と直生成のどちらが安いかはz_j/a_j・合成精度次第。全対照へ与える |
| outer A/scalar | density位相≤N+N(N−1)/2、E_nucはQPE ancilla上のphase。ops.pyはancilla-relative phaseを明示 | sparse/fusion/同一係数共有と全controlled T/depth費用 |
| full block/step | 各部分を同じ指標で加算し、逆算・全M・stage multiplicityを含める | total T/CX/depth/workspaceはUNKNOWN。未実装部分を0にしない |

7T/Toffoliの厳密unitary分解を選ぶなら、全logical T=7N_Toff+Σ T_RZ(η)+その他で会計できる。Gidney/measurement modelならその固有会計を全対照で使う。T、CX、depth、qubit、古典CPU秒を一つの数へ無断で足さない。Q_jのピークはserialなblock間でworkspaceを再利用可能でも、function内部のgarbage/pebblingを含む最大値が未閉鎖。
bitwise回転の上界やknown adder上界はfull mixed blockのlower boundではない。これらだけを使って「追加算術で必ず負ける」とも断定しない。

## 6. 強い比較とbreak-evenの具体的条件

same target H、model予算、unitary/controlled機能、precision/error norm、成功率、native gate/resource modelを一致させる。全法へ厳密group化、commuting fusion、basis/Givens共有、係数zero/疎性、共通近似、回転共有と合理的precision配分を許す。各方式の適用できる最適化だけを実際に会計し、無効なgroup化を強制しない。

| baseline B | 同品質全step G_B | 現候補に対する条件 |
|---|---|---|
| excitation PF・S2/S4等 | outer density＋全excitation/parity/control/synthesis、stage/fusion後の費用 | 普通の定振幅励起はθ/rho算術を必須としない。追加算術は必要step低下等で回収する |
| known THRIFT | mixed access＋outer A、正負time/adjoint/段数を含む | 素朴global A反復は参考対照。optimized相殺版・局所版も検討対象 |
| strong compressed SU2＋算術 | §3の具体的known construction、§4–5の同じ最適化・費用 | 現構成を選べるため、本候補より高いと仮定する根拠がない。新独立差は未特定 |
| DF-native PF | G_DF=G_1body+Σ_l(2G_basis,l+G_density,l+G_control,l)、sharing後の費用 | full HのDF再構成・rank policy・errorを固定。L、ρ_l、Givens/diagonal数、PF誤差定数は未取得 |

M_effはexact grouping後のtransition数、m_jは一stepのoccurrence数。全jがdenseならM_effがO(N⁴)、z_jがO(N)となり、detuning加算だけでもO(N⁵w)のこの設計になり得る。これはモデル全体の最良lower boundではない。
DFのGaussian回路は条件付き角度が全occupationに依存するこの構成と別のrepresentation。元H・精度が同じときの全step/全taskで比較し、fragment数やrankを推定で埋めない。

費用単位uをT/CXなど一つに固定し、G_Rを候補のcontrolled step、G_Bをbaselineのcontrolled step、r_R/r_Bを同じ時間T・deterministic予算に必要なstep数とする。古典前処理時間を量子gateへ無断で変換しない。
量子準備費用Q_R/Q_B（phase-gradient状態等）がある一interrogationでは

\[
Q_R+r_RG_R<Q_B+r_BG_B.
\]

量子準備が共通ならG_R/G_B<r_B/r_Rがbreak-even。古典前処理はP_R/P_Bを別に比較し、仮に全費用を共通金額等へ換算するなら、その規則を明示してP_R+Q_R+r_RG_R<P_B+Q_B+r_BG_Bとする。

G_R=G_A+Σ_j m_j(H_j+2F_j)、H_jは§5のdetuning/group、encoding、flag、parity、回転・合成等の**function以外の全controlled費用**。同じuで許されるfunction compute＋eraseの予算は

\[
\sum_j2m_jF_j<\frac{r_B}{r_R}G_B+\frac{Q_B-Q_R}{r_R}-G_A-\sum_jm_jH_j=:\mathcal B_F.
\]

右辺≤0なら、指定したH_jを持つこの設計はfunctionをどれだけ安くしても勝てない。H_jがupperしかない現段階では、upperを代入した右辺≤0は一般的不能を証明しない。実費用又は必要なlower boundを使う場合にだけ除外判定になる。
均一F、occurrence数W=Σ_jm_jならF< B_F/(2W)。単なるsmall mixed blockではなく、励起数と全stageで許容function単価が縮む条件を示す。全F/H/G/rの数値は未取得。

同次数pで、validな比較boundが仮に

\[
E_R\le K_R\alpha^2T^{p+1}/r_R^p,\qquad
E_B\le K_B\alpha T^{p+1}/r_B^p
\]

なら、そのboundから選ぶstep（ceilingを無視した設計近似）はr_B/r_R=(K_B/(K_Rα))^{1/p}。共通準備での条件は

\[
\alpha<\frac{K_B}{K_R}\left(\frac{G_B}{G_R}\right)^p.
\]

**このα閾値はconditional**：K、同次数のformula、implementation/model誤差予算、precision依存G、ceilingを含む実stepで再評価が必要。小α用に最適化したPFやDFに同じα一次boundが必ず成立するという主張ではない。強いBが同等のα²条件を持てばα利益は消える。THRIFTの既知理論だけから本分子のr低下を推定していない。
αは指定Bの実coupling scaleで、Aへidentity shiftを入れて見かけ上小さくしない。RHF参照の特定matrix elementが0でも、全Fock空間のBが0又は小さいとは言えない。分子の軌道対称性でzero/疎性があるか、densityが支配的か、長いJW/routingが費用になるかは実係数を確認する境界。新係数は今回生成しない。

## 7. Ground-energy taskと、判断を変える追加情報

mixed block→全step→同精度時間発展→ground-energy推定を分ける。QPEの全powers/interrogationsで§4のS1/S2/合成誤差を集計し、同じbase stepをpowerに使うならそのphase意味論・alias intervalを保持する。
一つのunitary norm差から、ground eigenphaseの枝対応・prepared overlapを確認せずenergy誤差を割算で認定しない。元Hのmodel誤差、PF/effective Hamiltonian bias、係数/算術/合成、phase-estimation統計を別に配分する。
full taskはΣ_q N_rep,q(G_prep,q+G_controlled-evolution,q+G_measure,q)に初期資源準備等を加える。同じprepared stateを許してもground overlap・成功率は未評価、測定costやrepeat数を0にしない。
本relative/PFはdeterministic unitaryで、PR/LCU normalization係数やshot増加を導入していない。将来PR等を組み合わせるなら、その成功振幅・normalization・statistical contractが別に必要。今回のPR hybridは未実装・未認可。

| 未解決事項 | なぜ重要か／判断を変える証拠 | 次の範囲と認可境界 |
|---|---|---|
| known constructionを超える差 | method deltaなしのbackend完成は既知法の実装にもなる。具体的に異なる操作列、同機能の費用証明、適用範囲が必要 | 研究仮説・追加投資の重要科学判断はGPT。今回自動採択しない |
| function費用/precision | B_Fに収まるか、uniform誤差・zero・control・garbageが閉じるかで実装可能性が変わる | primitive設計を特定する作業は別scope。full backend/compile/T benchmarkは今回進めない |
| M_eff/z_j/L/ρ_l/coupling/commutator | sparsity/group数、rank、K、stepが量子利益を左右する | 新分子生成/FCI/固有値/科学・compile実行は別明示認可 |
| energy task | time-evolution勝利がground-energy全費用利益になる条件 | state preparation、ground branch、alias/statistics/controlを固定する新契約が必要 |

基準の[H2/H4未認可pilot案](../../artifacts/qc_relative_block_prevalidation/2026-10-13/pilot_design_NOT_AUTHORIZED.json)は保存するが、起動条件・実行認可は追加しない。現在の差のないconstructorを分子で測るだけでは、独立した新方法かという判断は変わらない。
継続するならGPTが「既知法の化学実装・費用研究として扱うか」「別の具体的差を先に求めるか」を判断した後、その目的に必要な最小scopeを新たに固定する。第二候補へ自動移行しない。

## 8. 技術完了と停止

資料・固定sourceの静的照合、直接数式導出、既知primitiveの費用会計を完了。新science/数値的事後評価/compile/eig/synthetic/旧13再実行0、GPTレビュー0。元source・結果・認可・STOP、N1 STOP/N2 HOLD/N3 CLOSEDを維持する。
global manifestの既存 `relevant_commits[7].description` 欠落FAILを保持し、本taskの保存照合PASSと区別する。独立CI・外部科学再現の証拠ではない。
[static metadata checker](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/audit_tools/check_static_integrity.py)はsource/証拠SHA、旧manifest、変更scope、追加リンクを検査する。[保存結果](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/static_integrity_check.json)。科学sourceのimport・数値述語の評価・synthetic再実行を行わない。
Mandatory STOP：次は方法差・追加投資に関するGPTの重要科学判断。そのレビュー開始は明示承認が必要で、計算実行承認とは別。現constructorの継続・終了、中心テーマ採択は本監査では決定しない。
