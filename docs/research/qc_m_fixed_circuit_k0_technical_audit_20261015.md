# M案K0：固定回路のh²係数修正PF・限定技術監査

2026-10-15 JST。Aの[既存成果引継ぎ](qc_onebody_he1_archival_handoff_20261015.md)を作成した後、ユーザー指定のBを一回実施した。**M-aのcounterterm・射影／係数生成の中核は既知手順へ還元できる。DFに限定した独立の生成上の優位とCS2後の費用利益は特定できず、M案全体は `UNDETERMINED_NOT_READY`、K0 gateは不成立で停止する。** K1、数値射影、pilot準備・本格実装へ進まない。M案の科学的採否・終了を決める報告ではない。

## 1. 入力、固定対象、調査範囲

[受領した完了GPTレビュー](../reviews/quantum_chemistry_pf/qc_post_he1_research_direction_scientific_review_20261015.md)の400行・42,646 bytesを原文保存した。SHA256 `326622956c9eca9d28fc8cc9d500ad3cf1d45e7b2809ab06eea23aac1c750fca`。[原文identity](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/review_identity.json)。レビューが参照するClaude別原文のhashは著者による報告値と区別し、Codexの独立取得・計算結果にしない。

基準commitは `0e923f73754d1ebd168bf21a1b9e73971946e603`。H-E1監査、第一候補の方法差・h境界報告、native DF sourceの12固定本文を読み取り確認した。[固定参照identity](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/reference_identity.json)。過去のraw科学結果を再認証・再集計していない。

研究入力は元Hの係数、固定PF順序・frame・native grammar、許可された情報access。厳密量子基底状態は利用可能だが、E0・古典RDM・gap・励起resolventは設計入力として無料にしない。ここではM-a／構造保存型を優先し、M-bの変分探索、DF等価回転、PR/Track A/B、旧C1〜C4へ移行しない。

## 2. 数学的成立範囲（係数未代入）

### 2.1 形式展開と次数

固定した有限次元Hermitian fragmentについて、同じlog branchの原点近傍で

\[
S_2(h;\mathbf F)=\exp[-ih\{H+h^2D(\mathbf F)+h^4E(\mathbf F)+O(h^6)\}],
\quad H=\sum_\ell F_\ell
\]

と定義する。一般に `F_l(h)=F_l+h²W_l+h⁴R_l+O(h⁶)` がeven analyticで、各stepで同じHermitian係数・palindromic配置を用いる場合、BCH多項式への代入から

\[
H_{\rm eff}^{M}
=H+h^2V+h^4T+O(h^6),\qquad
V=D+\sum_\ell W_\ell,\quad
T=E+D'_{\mathbf F}[\mathbf W]+\sum_\ell R_\ell.
\]

ここで `D'[W]` は各fragment位置にラベルを付けたFréchet微分。**合計WだけではTを決められない。** 同じWでも配分が違えばD'と回路費用が違い得る。原レビューの固定Wの式はR=0の場合として整合する。新しい演算子係数を数値取得した検証ではなく、既知BCHの形式的代入である。

evenなF_l(h)を固定して反転すれば `S_M(-h)=S_M(h)⁻¹`。同じh・同じ係数のr回反復では `S_M(h)^r=exp[-irh H_eff^M(h)]`。異なるhで作ったkernelや適応更新を一つのH_effで表さない。各stepのfragment和は一般にHと異なり、**第一候補の「各分割の和がHを保存」と同じ構成ではない**。対象energyは元Hのままで、その近似積分器にcountertermを入れる。

V=0を演算子として実現できれば形式的に4次となる。Vのground期待値だけを消してもoperator 4次にはならない。一般二体fragmentのnested double commutator Dは最大4-bodyまで含み得るので、1/2-bodyだけの合法spanで全Dを消去できるとは限らない。有限hのradius・remainder・実装誤差は別条件である。

### 2.2 CS2後に何を変える必要があるか

非縮退・孤立groundの解析branchに対し、`Q=I−|0><0|`、`R0=Q(H−E0)⁻¹Q` と記すと、標準摂動展開は

\[
E_M(h)=E_0+h^2\langle V\rangle_0
+h^4\{\langle T\rangle_0-\langle0|V R_0 V|0\rangle\}+O(h^6),
\]
\[
|\langle0|0_M(h)\rangle|^2
=1-h^4\langle0|V R_0^2 V|0\rangle+O(h^6).
\]

E0とR0は解析上の記号であり生成入力にしていない。scalar補正後に効くのは `r_M=<T>−<V R0 V>` とoverlapである。`W=−d I` は元PFへのscalar補正と同じで、Tの非scalar部分もoff-diagonal Vも改善しない。古典energy後処理を対照に認めれば、その対照にcontrol位相gateを課す理由はない。物理回路にscalarを埋め込む場合はrelative phaseを価格に戻す。

非scalar WにはVの励起couplingやTを変える**機構の可能性**がある。ただしHS残差減少は、この二量の同時改善を保証しない。例えば混合項の大きさが減っても `<T>` との相殺が弱まり、補正後の絶対誤差は増え得る。保証gap下界gがあれば混合項には `0≤<V R0 V>≤||QV|0>||²/g` 等を用いられるが、gの取得、T、実overlap、全gate費用はその式から無料に得られない。M-bに必要なこれらの取得法は現案にない。

## 3. Native DF grammarを分ける（K0-2）

読んだsourceは[ops.py](../../src/trotterlib/df_trotter/ops.py)の `df_squared_diagonal_primitives`／`append_diagonal_primitives`、[model.py](../../src/trotterlib/df_trotter/model.py)の `DFBlock`、[partial-S2](../../src/trotterlib/df_partial_s2.py)の `_append_block`、[第一候補の静的会計](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_resources_v1.py)の `phase_counts`／`circuit_counts`。いずれも読取りのみ、import・回路生成・実行なし。

| 契約 | 合法な修正 | 同じ回路として言える条件 | 限界 |
| --- | --- | --- | --- |
| G1：frame固定のdensity pair-phase | `U_l†[c+Σu_p n_p+Σv_pq n_p n_q]U_l` の係数を変更 | 同じframeと同じ有効な位相slotを使うなら基底変換を維持可能。各slotの角度をh³だけ変更できる | 高level `DFBlock(eta,lam,U_ops)` は任意Jを入力しない。一般density係数へのadapterを今回実装したわけではない。lower-level RZ/RZZ/relative phase表現は既にある |
| G2：平方型の構造保存 | `λ_l L_l(h)²`、`L_l=Ση_lp n_p` | λのみの変更は厳密な平方型。eta変更も固定frameなら平方型を保つ | 任意のG1係数はrank-one平方ではない。eta変化は接方向の制約とh⁴項を持つ |
| 非固定frame／高body-rank W | 一般には追加のGaussian更新や多体phase等が必要 | 元のgrammarへのclosureと同価格を別途示す必要 | 「同gate数」の前提を認めない。再対角化・新DF・compileで救済しない |

固定λで `L(h)=L+h²δL` とする場合、

\[
W_l=\lambda_l\{L_l,\delta L_l\},\qquad
R_l=\lambda_l(\delta L_l)^2.
\]

Rを落とすと、今回狙うCS2後のh⁴項そのものを欠落させる。lambdaも変えるなら交差項を含めて展開が必要。平方型の接空間へ射影したWと、有限hで実際に使う平方型を同一視しない。

G1の位相は `n_p=(I−Z_p)/2` による既知変換でRZ/RZZへ戻せる。I成分はcontrolled時にancilla相対位相となる。**sourceはゼロ角を省く**：RZZ表の全pairを一度列挙しても、append時にexact-zeroをskipする。RZも係数合算後のゼロで省略する。したがってdensity係数の非zero数と実際のPauli位相slot数も同一ではない。

実効Pauli角が `h a_s+h³b_s+…` のとき、元のゼロslot `a_s=0` に `b_s≠0` を入れればgateが増える。元非zero slotのみを使い、全frame・符号規則・fusionを保ち、偶発的ゼロ・特殊角・合成精度を別扱いした**抽象固定slot会計**ならslot数不増は条件付きで言える。Clifford+TのT数、深さ、任意角の単価まで不変とは言えない。

第一候補の会計にはfixed-N scalar除去、n／nn primitive、基底共有とquery境界fusionがある。lower-level RZ/RZZ実装と異なる資源単位なので、旧P1/P2価格をM案へ無断流用しない。強対照にも同じ合法slot・control・scalar後処理・fusionの自由を与える。

## 4. M-a射影は何を計算するのか（K0-4）

合法な**ラベル付き**方向 `T_(l,a)` を固定し、`T_a=Σ_l T_(l,a)`、`W_l=Σ_a x_a T_(l,a)` とする。制約なし線形spanなら、normalized sector trace `τν` で

\[
G_{ab}=\operatorname{Re}\tau_\nu(T_a T_b),\quad
b_a=\operatorname{Re}\tau_\nu(T_a D),\quad
\min_x\tau_\nu[(D+\sum_a x_aT_a)^2]
\ \Longleftrightarrow\ Gx=-b.
\]

これは既知のleast-squares／HS射影。`Σn_p=ν I`、`n_p²=n_p`、frame間の重複などでGはsingularになり得る。物理的な合計Wの射影が一意でも、ラベル配分は一意とは限らない。指定metricの最小norm解などのtie規則を加えるのは標準線形代数であり、D'・費用を最適化したことにはならない。support／box制約を加えれば単純な直交射影ではなく制約付き問題になる。

**指数次元のsector行列の全列挙は必須ではない。** 固定body-rankのnumber-conserving monomialをCARでnormal orderし、異なる添字について

\[
\tau_\nu(a^\dagger_{i_1}\cdots a^\dagger_{i_r}a_{j_r}\cdots a_{j_1})
=\frac{\binom{N-r}{\nu-r}}{\binom N\nu}\det[\delta_{i_pj_q}]
\]

を使える。r>νでは0、重複添字はCARで処理する。spin数を別々に固定するsourceの場合は、保存される各spin sectorの組合せ因子を用いる。これは一様sector traceであり、未知のphysical ground-state RDMを取得する式ではない。

Nをmode数、mを入力fragmentの明示fermion monomial総数、kを方向数、tを一方向の最大monomial数、M_DをDの展開項数とする。二体H・固定BCH次数ならDのbody rank≤4、raw `M_D=O(m³)`、G1/G2の方向はrank≤2で `t=O(N⁴)` という粗い上界を置ける。既知CARと上のtraceを用いた素朴な係数生成の演算回数は `O(M_D+k t M_D+k²t²+k³)`、保存量は `O(M_D+k t+k²)`。全Dの保持を避けるstreaming/cacheやcompact BCHは対照にも許す。この上界は固定精度演算・固定body-rankの静的解析であり、実測費用・最良計算量・bit complexity・安いproduction実装の保証ではない。Gramのconditioning、係数丸め、証拠精度は別に必要。

Dがsectorを保つcommutatorの和なら `τν(D)=0`。identity方向だけのHS射影は0で、ground期待値dを消すscalar correctionと異なる。Dが二体fragmentから来ることや、traceが係数から計算できることだけでは、ground energy・CS2残差の取得を1-RDMへ閉じられない。Dのground期待値は一般に4-RDM、Tの明示期待値はさらに高いbody rank、励起混合項はresolventを要し得る。今回それらを計算・測定しない。

## 5. 強い一次文献への対応（K0-1）

6一次本文の関連式を確認した。[文献identity・確認節](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/literature_registry.json)。新規取得4本文、同日HE1で固定したCPF/SPRINT本文2件をhash照合して再利用。要旨だけから同値性を認定していない。

| 一次資料と確認式 | 既知として対応する部分 | M案について認定できないこと |
| --- | --- | --- |
| [Blanes–Casas–González–Thalhammer、modified potentials 2023](https://doi.org/10.4208/cicp.OA-2022-0247)、§2.2 Eqs.2.7–2.13、§3.1 | `exp(−h b V−h³c[V,[T,V]])`、可解なmodified potentialの吸収、BCH係数・stage費用、processed kernel。Mのh²counterterm原理に直接対応 | 論文はparabolic／imaginary-time Schrödinger等。位置potentialでのcommutator可解性を一般DFへ移さない。実時間ではanti-Hermitian generatorの符号へ戻す。既知のeffective-order構成を新原理にしない |
| [Blanes–Casas–Murua、2401.01722v3](https://arxiv.org/pdf/2401.01722v3)、§3.2 Eqs.3.4–3.7 | commutator flowが元flowと可換なら既存slotへ併合。modified potential、order条件と演算費用 | double commutatorが原density frameで対角・二体であるとは限らない。flow可解性と量子controlled価格は別 |
| [Mansuroglu–Fischer–Hartmann、2306.07208v2](https://arxiv.org/pdf/2306.07208v2)、Eq.2、Definition1 Eqs.4–6、Prop.2、Eq.12 | 固定した指数列の角度を古典BCH近似・trace Gramで最適化。G1の `θ_s(h)=θ_s^0(h)+h³z_s` は固定slot角度自由度の部分族 | 公開したC(θ)はdegree2までで、symmetric S2のdegree3誤差Dをそのまま評価する目的関数ではない。M-aのS2三次BCH／固定sector版が同論文に完全一致すると断定しない |
| [Corrected Product Formulas、2409.08265v4](https://arxiv.org/html/2409.08265v4)、Eqs.4–6,24–27、§5 | BCH corrector、symplectic／symmetric／composite、分解誤差と追加費用 | generic correctorを各DF slotに無償吸収できるとは限らない。pure conjugationはisospectral。前回監査の表示Eq.28留保は維持し再審査しない |
| [SPRINT/GRADE、2606.30741v1](https://arxiv.org/html/2606.30741v1)、chemical grouping／near-integrable、Appendix E.3 Eqs.96–102 | 化学fragment・norm階層・PF構成・processingとnative費用の比較 | Mの合法spanを使ったCS2後の利益を自動供給しない。SPRINTへ同じ係数・control/fusion情報を認める |
| [Carenaら、2203.02823v2](https://arxiv.org/pdf/2203.02823v2)、Improved Hamiltonians Eqs.1–5、Circuit Design | spatial lattice spacing aのdiscretization誤差を追加Hamiltonian項で改善し、その回路を構成 | spatial `a²` countertermとPF timestep `h²` countertermは別の誤差源。Mがこの研究に完全還元されるという根拠にはしない |

**限定的に還元できるもの**は、modified-equation counterterm、固定合法slotの係数変更、三次BCH残差の線形least-squares射影、sector traceのCAR評価である。これらを同じ情報で行う強対照は、M-aと同じnormal equationを生成できる。第三次BCHへの拡張やDF向け係数表現が未掲載というだけでは、独立した生成上の効率改善にはならない。

ただし「既存一論文がこのDFの全手順・全保証を既に述べた」とする同値証明は得ていない。一般Mが既知法に尽くされること、未掲載の組合せが歴史的に初出であることのいずれも認定しない。

## 6. 同情報の強対照、技術判定と停止

強対照には元S2、元CS2、M-a+同品質scalar補正、S4、同じ合法slotでの既知係数最適化／counterterm fitting、条件を満たすnear-integrable/SPRINT/CPF、既知外挿を置く。許すframe・slot・h機会、誤差・failure、state情報・取得費用、controlled phase/fusionを揃える。新しい比較実験や費用値は作成していない。

| K0軸 | 結果 |
| --- | --- |
| K0-1 先行研究 | counterterm・係数自由度・HS/CAR生成のcoreは `REDUCES_TO_KNOWN`。特定DF全手順の歴史的完全同値は未認定 |
| K0-2 grammar | G1吸収とG2接方向・h⁴項は `SUPPORTED`。slot不増はsupport保存等に `CONDITIONAL`、T価格不変は未保証 |
| K0-3 数学 | 形式展開・scalar対照・摂動機構は条件付きで `SUPPORTED`。HS減少からCS2後の誤差・overlap・費用改善は導けない |
| K0-4 情報・生成費用 | 固定body-rank／係数accessの下でsector Gramを多項式演算で記述可能。標準手順の特殊化であり、本方法だけの計算量差は `NOT_IDENTIFIED` |
| K0-5 独立性 | 同情報の既知counterterm/係数fitに対し、独立した生成・保証・費用改善規則は `NOT_IDENTIFIED` |
| M案全体／gate | `UNDETERMINED_NOT_READY`、`gate_pass=false`。一般的no-go・正式な科学的終了ではない |
| K1・M-b探索・pilot・production | 未実施・未承認。K0の未解決を理由に自動延長しない |

残る不足は、**(i) sourceの有効support・価格へ閉じた具体的なラベル配分規則、(ii) 同情報の既知fitで得られない生成または保証の差、(iii) scalar補正後のT・励起混合・overlapまで含む利益の根拠**。重要本文の未取得が停止原因ではない。「さらに文献を探せば」「M-bを最適化すれば」を次の作業にせず、[GPT判断資料](qc_m_fixed_circuit_k0_gpt_handoff_20261015.md)へ戻す。

[文書identity・保全監査](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/document_identity_audit.json)はbyte/hash/JSON/linkの技術確認であり、数値結果の再認証やCIではない。係数生成の粗い演算回数では固定rankのnormal-order処理を定数として数え、辞書・sort等の管理費を別扱いする。旧sourceを実行して計測した上界ではない。

今回は新H・分子・eig・RDM／期待値・PF・数値射影／最適化／比率計算・保存数値集計・compile/T合成・科学tests/checker・pilot/production・新GPT科学レビュー0。旧科学source・結果・消費済み認可・STOP、manifest既存FAIL、第一候補の正式状態、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template CLOSED、Track A/B/C1〜C4を変更しない。独立branch `qc-m-fixed-circuit-k0-audit-20261015` へ公開・GitHub再取得照合後 **Mandatory STOP**。

## 公開照合・停止

Payload `9d79967b4ff16addc0e82ad7ba05c62eb4673b03` をGitHubから独立再取得し、23変更blobs／2,882,751 bytes、原文42,646 bytes／SHA、5,177旧保護files、12固定参照本文を照合した。[receipt](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/publication/receipt.json)、[Mandatory STOP](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/publication_stop.json)。164旧manifest/non-result/既存FAIL、親root baseline445pathのstatus・bytes一致。公開メタデータは別commitで保存・再取得し、科学計算・保存数値集計は再実行しない。K0 gate=false、K1未開始、第一候補正式状態とM未採択を維持して **Mandatory STOP**。
