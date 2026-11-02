# 量子化学相対時間発展block：構成・方法差・費用の事前検証 v1

2026-10-13 JST。[完了GPT方針レビュー](quantum_chemistry_ground_energy_algorithm_direction_review_2026-10-13.md)第4–7・10–12節に従うCodex技術資料。
**構造と限定syntheticは確認した。独立した方法差と同精度の資源利益は未確定。科学pilotは未認可・Mandatory STOP。**
目的は元の量子化学Hamiltonianの基底エネルギー計算の低コスト化。THRIFT/PRの採用や、一つの短時間blockの安さを成功条件にしない。
N1停止/N2保留/N3 closure、旧Track A/Bのsource・結果・認可・STOPを維持する。今回は新たなGPT詳細科学レビューを実施していない。

## 1. 先行手順・既存sourceとの対応

| 対照 | 一次資料の確認範囲 | 本constructorとの差・留保 |
|---|---|---|
| THRIFT | [Bosseら、arXiv HTML](https://arxiv.org/html/2403.08729)定理1 Eq.8–13、Fermi–Hubbard資源appendix | 相対積C_jとseedは既存構成。mixedアクセス費用を含める必要がある。分子での優位は同論文から継承しない |
| occupation条件付き可解性 | [Patel–Yen–Izmaylov v2](https://arxiv.org/html/2305.18251v2)III.2、Eq.33–35 | sector依存対角化は既知。projector和の表示と、succinctなcoherent arithmetic circuitの具体費用は別。測定後の古典分岐をQPEに使わない。著者の全方式が必ず指数費用とは断定しない |
| fermionic excitation | [Arrazolaら v2](https://arxiv.org/html/2106.13839v2)2.2、3.1–3.3 | occupation対のU(2)/Givens、追加制御は既知。今回のpair encodingを新規性とはしない |
| 関数・可変角計算 | [Häner–Roetteler–Svore](https://arxiv.org/html/1805.12445)IV・appendix、[Quantum CORDIC](https://arxiv.org/html/2411.14434)導入・可逆化 | polynomial/function arithmeticとcompute–rotate–uncomputeは既知。arcsin回路の費用をatan2/hypotへ流用しない。poly域分割数・workspace・丸めも費用 |
| 低rank/DF-native | [Mottaら](https://arxiv.org/html/1808.02625)Givens/diagonal pairの構成 | 同じHをbasis rotation＋diagonal interactionで実装する強い対照。rank/truncation別の文献倍率や漸近経験則を本候補へ転記しない |
| sparse access | [Berryら著者PDF](https://cs.uwaterloo.ca/~cleve/pubs/2005SimulateSparseQph.pdf)第V節 | 行列entryを返す手順の費用は既知の重要点。A+B_jはdiagonal＋transitionであり、一般に1-sparseそのものとは呼ばない |

既存`mechanisms.conditional_mixed`は二つのspectator枝を列挙し、atan2/hypotとRY–RZ–RYを使う既知toy構成。今回のアフィン算術は枝の列挙を避けるが、その変更だけで新しいアルゴリズムとは認定しない。
`df_trotter.ops.apply_df_block`はbasisのinverse、diagonal squared作用、basisを適用する。`two_body.py`にはnormal ordering、chemist/physicist permutation、一体・scalar correctionがある。[静的source index](../../artifacts/qc_relative_block_prevalidation/2026-10-13/existing_source_static_index.json)。これらのsourceをimport/実行・変更していない。

今回の具体案は既知SU(2)＋可逆算術へ構成的に対応する。global A生成と全枝表の回避、相殺に対応する精度配分は費用差の候補だが、強い既存compressed構成にも同じ最適化を許す。
**現段階で独立した非自明な方法差を特定していない。** 未発表性の網羅確認・一般的不能の結論ではない。原レビューの新規性認定を代行しない。
出版社NatureのTHRIFT/DF直接取得は失敗し、上記arXiv HTMLを確認した。版同一性を仮定しない。[文献確認範囲](../../artifacts/qc_relative_block_prevalidation/2026-10-13/literature_scope.json)。原典全文をartifactへ大量複製していない。

## 2. Hamiltonian契約と圧縮された係数生成

実対称一体・実Hermitian二体、粒子数保存、有限spin-orbital basisを出発点とし、

\[
H=E_{\rm nuc}+A+\sum_j B_j,\quad A=\sum_i\epsilon_i n_i+\sum_{i<k}J_{ik}n_i n_k
\]

とする。Jは符号付き、非負density条件はない。複素励起係数・spin-changing変換は本v1の実装対象外。
Hermitian monomialと随伴を一度だけ登録し、densityとdensity-assisted singleをnormal orderingで区別する。全ijのprefactor/反対称化を二重計数しない。実分子adapterは未実装で、実integralを生成・読込していない。

励起のcreated supportをP、annihilated supportをQ、|P|=|Q|=1又は2、P∩Q=∅とする。chi0はP occupied/Q empty、chi1はその逆。spectator sは前後で不変。
支持内occupationをu_i,v_iとすると、

\[
\Delta_0=\sum_{i\in S}\epsilon_i(u_i-v_i)+\sum_{i<k\in S}J_{ik}(u_i u_k-v_i v_k),\qquad
 d_l=\sum_{i\in S}J_{li}(u_i-v_i),\quad
 \Delta(s)=\Delta_0+\sum_{l\notin S}d_l n_l.
\]

spectator同士のdensityとscalarは差で消える。descriptorの生成は一励起O(|S|N+|S|²)、全M励起でO(MN)。全差係数を保存すればO(MN)メモリになるため、元Jの行差を参照するdescriptor／オンデマンド生成も比較する。既存の入力係数取得・保存コストは消さない。
同じp→qを持つdensity-assisted項は、振幅g(s)=g0+Σ_l g_l n_lを持つ一つの二準位へ厳密にまとめ得る。これは既知conditional構造への整理で、全double励起を少数groupへできるとは仮定しない。通常PFにも同じgroup化を与える。

各有効対でA+B_j=dbar(s)I+Δ(s)Z/2+b(s)X。相対積は

\[
C_j(t)|_s=e^{it\Delta Z/2}e^{-it(\Delta Z/2+bX)}.
\]

無効occupationでC=I、平均dbarは両指数間で相殺する。外側seedのe^{-itA}とE_nucのcontrolled scalar phaseは残り、費用・位相を必ず会計する。

## 3. Pair encoding、parity、coherent controlとuncompute

Sのpivot pを選び、他のk∈SへCNOT(p→k)を適用する。両有効patternはn_k xor n_pが同じになり、必要ならpivotのXでchi0をlogical0へ合わせる。
valid flagは他の|S|−1≤3 invariant bitの定数pattern一致から計算する。density-assisted occupancy判定もspectator controlとして加える。flagはlogical rotationで変化しない。

Jordan–Wigner parityは、ordered monomialを右から適用するladder符号を基準にする。spectator maskは「そのmodeより上のladder index個数が奇数」のbit集合、support内符号と先行flipは固定定数へ入れる。
parity bitをP(s)とすればC_{Δ,−g}=Z C_{Δ,g} Z。正振幅reference blockの前後でparity-controlled Zを適用する。group振幅の符号も追加Z制御へ含め、0は厳密skip branch。

順序はencode→valid/parity→Δ（必要ならg）→angles→valid AND QPE control→relative SU(2)→逆順uncompute→decode。
算術をQPE control外でcompute/uncomputeしても、control0枝のsystem作用はIとなる。ただしその計算費用は両枝の共通回路へ計上する。spectatorを測定しない。
Δ/g/parityはspectatorの関数、validはencoded invariant bitの関数なので、回転でlogical bitが変わっても逆計算の入力は不変。この不変性をsyntheticで照合した。実際の全workspace gateのcleanlinessをcompile済み回路で検証したわけではない。

## 4. 回転と非線形関数の具体案

rho=hypot(Δ/2,g)、theta=atan2(g,Δ/2)、g≥0について、column-vectorで

\[
C=R_Z(-t\Delta)R_Y(\theta)R_Z(2t\rho)R_Y(-\theta).
\]

右からのchronological順をIRに保存する。θ registerを同じ値で±使用する。g=0又はt=0は全回転skipでIを保つ。C(-t)をC(t)†と同一視せず、adjointはゲート順と符号を両方反転する。

指数枝表なしの関数計算案：固定小数の加算/乗算、restoring division・square root、比較とquadrant flag、Horner評価を可逆化して使う。
atanの入力をx=min(|Δ|/2,g)/max(|Δ|/2,g)≤1へ正規化（0/0はskip）し、u=x/(1+sqrt(1+x²))≤sqrt(2)−1、atan x=2atan uを使う。
K項のalternating Taylor近似の切捨ては2|u|^(2K+1)/(2K+1)以下（K項をuからu^(2K−1)まで採用）。K=O(log(1/ηtheta))で済み、2^b要素のtableを仮定しない。quadrantのπ/2・πも所定誤差で保持する。
これは**数学的な構成案**。内的丸めの外向き伝播、guard bit数、gate-level sqrt/division/Horner backendは未実装・未認証。ライブラリのmath.atan2/hypotはsynthetic referenceであり、無料のquantum oracleではない。
有理化された最小非zero振幅、係数range、tのbit長を含めて作業精度を選ぶ必要があり、Δ=0での良いmatrix条件数と関数回路の安さを混同しない。

## 5. 係数・算術・合成誤差を分ける

同じ近似Δを両指数に使い、|Δ−Δtilde|≤ηΔ、|b−btilde|≤ηbなら、interaction-picture Duhamelから

\[
\|C_{\Delta,b}(t)-\widetilde C(t)\|\le\tfrac12 b_{\max}\eta_\Delta t^2+|t|\eta_b+\epsilon_{\rm impl}.
\]

V_Δ(s)の差は2|b sin((Δ−Δtilde)s/2)|≤|b||Δ−Δtilde||s|。実時間の向きも|t|で扱い、Δで割る仮定はない。
f fractional bitの各差係数のnearest roundingなら、ηΔ≤(1+n_s)2^{−f−1}。同じ二進固定係数を両側へ使用することが条件。一般の原h/J近似は先に差へ伝播させ、係数誤差を二重計上しない。
可逆加算はguard/rangeを持つ整数加算で行い、overflowを許さない。discarded bitを非可逆に消さず、garbageを保持して逆算する。

関数近似の独立なangle誤差をηtheta、central/final Zのangle誤差をηphi,ηchiとすると、rotation normのtelescopingでε_impl≤ηtheta+(ηphi+ηchi)/2+Σε_synthesisという粗い十分bound。
これは近似係数で定めた理想relative blockへの関数/合成誤差であり、係数変更のηΔとは分ける。controlled blockも同じsystem norm差だが、捨てたglobal phaseは別の誤りになる。
短time弱励起で直接積の粗い|t|ηΔより係数精度を緩め得るが、ε_implを緩めたという意味ではない。B=αB0でもηΔ固定ではO(αηΔt²)が理想THRIFTのα²誤差を支配し得る。
負時間・高次seed・全QPE powersには各occurrenceの|t_l|/t_l²/合成誤差を足す。外側PF/A、DF/model、state overlap、統計誤差は別。

## 6. 資源見積り：実測を作らず、未閉鎖費用を残す

w=整数/符号/guardを含む算術word、a=angle word（時間によるrange bitも含む）、K=関数近似degree、M=登録励起数。
[CuccaroらPDF Table1](https://arxiv.org/pdf/quant-ph/0410184)のmodular adderはw≥3で2w−3 Toffoli、5w−7 CNOT。外部controlを足す際、C3Xをclean ancilla一つで3 Toffoliへ、controlled CNOTをToffoliへ置換すると、制御加算は≤11w Toffoliという保守的上側**設計見積り**を得る。定数adderの最適化やgate-level validationをした数ではない。

| 構成部分 | ゲート/深さの見積り | 補助qubit・古典費用 |
|---|---|---|
| detuning compute＋uncompute | n_s係数で≤22 n_s w Toffoliの単純adder envelope、Clifford追加。ripple serial depth O(n_s w) | accumulator＋定数word＋carry、O(w)。係数読み出し/設定費用も別途 |
| affine amplitude/group | 追加O(n_s w) arithmeticと逆算 | 同じgroup化をnative PFへ許す。ゼロ/符号のflag費用 |
| angles＋逆計算 | sqrt/division/Hornerでschoolbook O((K+1)w²) arithmeticを基礎に、guard/domain依存を残す。数値定数・最適depthは未取得 | garbageを保持する単純可逆化ではO(poly(w,K)) workspace。無断でO(w) ancillaとしない |
| 4 variable rotations | bitwise方式なら最大8a single-qubit RZ＋8a CX、enable AND angle bitのcompute/uncompute最大8a Toffoli、basis Clifford追加 | control・valid enable flagとscratch。各CRZは2RZ/2CXというprimitive分解。routingとsynthesisは追加 |
| pair/parity/valid | encode/decode2(|S|−1) CNOT、parity計算/消去O(N) CNOT、valid/QPE enableの小定数Toffoli | flag/scratch、mode順序routing。相対phase保持 |
| outer A | occupation phase N＋N(N−1)/2項の素朴upper。controlとscalarを加える | 相殺した各Cの平均位相と混同しない |

論理TはToffoliの選んだ同一分解とRZの同じ合成保証から別に換算する。例えば7T/Toffoli conventionなら7×Toffoli countに合成Tを足すが、古典CPU秒・CX・qubit数をその値へ加算しない。
未実装function evaluatorのT/depthはnullのまま。RZ8aの上側見積りだけでfull circuitが安いと認定しない。

| 比較方式 | 一stepの構造的費用 | 公平な比較に必要なもの |
|---|---|---|
| native excitation PF | AのdensityとMの励起、parity/control/rotation。算術を常に要しない | 同じgroup化・fusion・PF orderの最適化 |
| explicit-global-A THRIFT | Aを各mixedで生成/逆生成する素朴経路、O(MN²)diagonal操作を含み得る | 参考対照。これだけに勝って新規性を認めない |
| compressed-known/candidate relative | O(M[Nw+(K+1)w²])+outer A、全逆計算・合成を含む | 既存SU2＋算術にも相殺/precisionを与える。本候補とのmethod delta未特定 |
| DF-native PF | 各fragmentのbasis/inverse＋density作用、素朴upper O(LN²) | L/rho、model誤差、Givens sharing、精度、前処理を同じHで固定。文献のrank縮約を無料にしない |

M=O(N⁴)のdense励起を各O(Nw)で扱うと、係数加算部分だけでもO(N⁵w)になり得る。少数group・全励起共通cache・低rank利益は未証明で、あるものとして差し引かない。
一step費用G_R/G_Bが高いなら、同じ要求品質で必要step r_B/r_Rがそれを上回ることがbreak-even条件。前処理を含める場合はP_R+r_R G_R<P_B+r_B G_Bを同一ゲート指標又は明示した換算規則で評価する。
現データではrもG_functionも未取得なので数値break-evenは未算出。αをidentity shiftで小さく見せない。

## 7. 限定syntheticの結果と限界

[module](../../src/trottertracks/representation_exploration/qc_relative_block_v1.py)、[固定synthetic runner](../../scripts/check_qc_relative_block_synthetic_v1.py)、[tests](../../tests/test_qc_relative_block_v1.py)、[結果](../../artifacts/qc_relative_block_prevalidation/2026-10-13/synthetic/report.json)。
13 tests PASS。手書き符号付きdensity、5mode full-Fock32以下、single/double/density-assisted、正負timeを独立ladder/scale-Taylor referenceへ照合した。geometry/basis/DF rank/L_D/PF窓はsyntheticには適用なし。登録外の構成検証で、実分子性能や新しい科学移送結果ではない。
零結合、零/近零/負detuning、fermionic parity、無効pattern、粒子数、control-relative phase、共通平均位相相殺、coeff rounding、adjoint順序、uncompute入力不変性、異常入力を検査した。
固有値/SVD関数をguardし、SciPy/Qiskit/OpenFermion/PySCFをimportしない。referenceのfull matrixを大系primitiveの無料oracleにしない。
CPU32/available RAM約51.5GiBを確認し、small matrix・Fractionの固定小suiteなのでCPU1/各thread1、AS2GiB・RSS検査512MiBを選択。spawn/GPUによる加速を主張しない。wall約.135秒、child RSS約40.6MiB、科学wall/CPU capなし。local検証でimmutable CIではない。
コードのangle_recipeは古典reference、structural_irは設計IR。分子adapter・可逆sqrt/division/Horner・native T合成・gate-level workspace cleanupは未実装。回路compile benchmark0。

## 8. 最小pilot案と今の停止点

[未認可pilot設計JSON](../../artifacts/qc_relative_block_prevalidation/2026-10-13/pilot_design_NOT_AUTHORIZED.json)。H2(0.74Å)/H4(linear1.00Å)、STO-3G、spin orbital4/8、ν2/4、frozen coreなし、実RHF軌道、元one/two-bodyを全項保持する候補。
H4は既存projectで使われたdevelopment geometryでblindと呼ばない。H2だけの特殊可解性を成功条件にしない。新integral/input SHAはなく、科学runner・source freeze・execution authorizationも未作成。
将来のdense referenceはphysical system/controlの512以下に限る。巨大算術workspaceをfull statevectorで無償に検査しない。実gate-level validation backendもlaunch前に閉じる。
前処理、descriptor、operator/control/clean workspace、同precisionのnative RZ/CX/T/Toffoli/depth/ancillaを比較し、証拠付きaccess費用とstep減少の両方が必要な判断を作る。
提案上限は分子integral2job、FCI事後reference2job、primitive24query、wrapper compile168record、2workers/thread1・aggregate RSS8GiB・出力64MiB/log4MiB、attempt1/retry0、wall/CPU capなし。現在の認可ではこれらの計算を一件も実行しない。

最終energy taskは一つのS(h)のpowers、位相alias interval、ground overlap/branch、成功率、状態準備費用を閉じる必要がある。JSONのenergy .001Ha/success .95等は後段設計候補で、未採択・未認証。
今回の最小pilotはprimitive成立と費用の判断用で、short-time誤差からground-energy全費用の勝利へ飛ばない。PR hybridは必須とせず未実装・未評価。

**現在の境界**：構成は既知部品へ対応し、独立した差を未特定。関数回路のuniform precision・具体T/depth/workspace費用も未閉鎖。単なるsource/testの追加ではなく、元レビュー第12節の方法差・coherent access/費用に関する重要判断の材料が揃った。
次はGPTが、相殺に対応する精度・構成に独立寄与を求める価値があるか、既知法の特殊化として区切るか、別候補を検討するかを判断する段階。レビュー開始は改めて明示承認が必要で、Codexで主研究採択・第二候補の自動起動を行わない。
科学batch/compileは未認可、Mandatory STOP。量子化学ground-energy低コスト化という研究目的は保持する。

## 公開・再取得確認

固定payload/source snapshot `c45e8a98cfa4bcdbe44c930fb96c15b6bea84eb9`、branch `qc-relative-evolution-prevalidation-20261013`。
[技術結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c45e8a98cfa4bcdbe44c930fb96c15b6bea84eb9/docs/research/qc_relative_block_prevalidation_v1_20261013.md)、[判断引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c45e8a98cfa4bcdbe44c930fb96c15b6bea84eb9/docs/research/qc_relative_block_review_handoff_v1_20261013.md)、[source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c45e8a98cfa4bcdbe44c930fb96c15b6bea84eb9/src/trottertracks/representation_exploration/qc_relative_block_v1.py)、[synthetic結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c45e8a98cfa4bcdbe44c930fb96c15b6bea84eb9/artifacts/qc_relative_block_prevalidation/2026-10-13/synthetic/report.json)、[未認可pilot案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c45e8a98cfa4bcdbe44c930fb96c15b6bea84eb9/artifacts/qc_relative_block_prevalidation/2026-10-13/pilot_design_NOT_AUTHORIZED.json)。
GitHub独立cacheから変更27blob/1,232,977bytesと旧source5blob、追加relative links31件を確認し、source/test/runnerのSHA3件がsynthetic生成時snapshotと一致した。
[公開receipt](../../artifacts/qc_relative_block_prevalidation/2026-10-13/publication/receipt.json)。remoteで同じtestsを再実行する完了要件は加えず、科学計算/compile0。
cacheはshallowだが必要なblobを取得でき、local object alternatesは使っていない。これは公開保存照合でimmutable CI/外部科学再現ではない。
元116 manifest entries/科学artifact/STOPを保持し、global manifest既存description欠落FAILは修正していない。
