# 新C：PR × Control-Free Phase Estimation 限定技術・先行研究調査

2026-11-06 JST。実施環境：**SERVER 1 / GPUA100-gleap-01**。
専用branch：`codex/new-c-control-free-survey-server1-20261106`。
専用worktree：`/tmp/new-c-control-free-survey-server1-20261106`。
SOURCE読取基準：`2397c5ac511148423ae89f25f74f1de967d1d39a`。

**既存PR＋fresh-shot RPEをC-1/C-2へそのまま差し替えられる資格は確認できない。** 主な問題は、振幅平均と確率平均の不一致、固定groundでのC-1の測定情報の退化、絶対energyの基準、C-2の虚時間normalizationと情報取得費である。これは限定的な適合性の整理であり、新C全体の科学的GO/NO-GO、新規性、改善率の判定ではない。対応表・再利用入口・最小検証とSTOPは [APPLICABILITY_MATRIX.md](APPLICABILITY_MATRIX.md) に示す。

## 1. 範囲・研究前提・保全

元の物理Hamiltonian Hの厳密基底状態 |g⟩を供給し、元stateの準備費を除外して、同じE₀を同じenergy精度・成功確率で求める比較を扱う。RZ・CX・T・depth・追加qubitは分ける。今回は新しい分子、geometry、basis、DF rank、L_D、delta窓、精度、round数を採択していない。旧H4条件を新Cへ自動適用しない。

厳密stateの供給は、古典ground vector、E₀、gap、state-preparation unitaryとそのinverse、ground projector、任意の追加state・重ね合わせの準備oracleを無料供給する仮定ではない。追加測定、normalizer・reference情報の取得、古典処理も課金対象として残す。[目的・範囲][objectives]、[新A/B固定調査][AB]の比較規約を継承する。

実施は文書・SOURCEの静的読取、必要箇所だけの一次文献照合、指定2文書の作成・公開照合に限る。科学SOURCEをimportせず、test/runner、科学計算、保存値の数値再解析・再ranking、sampling、quantum simulation、回路build/compile、価格取得、T合成は実施していない。phase retrieval solver、state準備回路、推定器、optimizerを実装・設計していない。以下の式は既知定義への直接代入・恒等式であり、新しい数値実験や非自明な保証の提案ではない。

SERVER 2の新A pilotには接続・介入せず、本branchは固定新A/B静的調査から分岐した。新A pilotの結果は今回取得していないため、存在・完了・性能を認定しない。新A/Bと旧Track C・C2（特にC2-C）を別系列として扱い、旧STOP、SOURCE・raw・result・grant・ledger・markerを変更しない。共有設定・パッケージ・他jobを変更していない。新しいgrant・承認証明・管理基盤は作成しない。

## 2. 固定資料と一次文献の確認深度

適用AGENTS、PROJECT_MAP、研究概要・現状全文、規範文書とVALIDATION_STATUS/manifestの関連箇所、および正式共通指示2026-10-25を確認した。共通指示の公開収録版はSERVER 1の読取原本と `cmp` で一致した。[正式指示][instructions]

| 固定資料 | 本調査で使った範囲 |
|---|---|
| [3環境統合 / 3571477][I] | REPORTの証拠階層・再利用位置づけ。全原runの再監査ではない |
| [PR＋RPE文献調査 / e8b19ff][L] | O6とO5、SOURCESのoracle/位相推定器条件。O6は主にQSP-PEを扱っており、C-1/C-2の適合保証を既に与えていた文書ではない |
| [Track Cプレフライト / 4b625e5][P] | finite-RTE、diagonal-only control、phase・radius・RPE会計、再利用testと強baseline |
| [C2限定調査 / 6f6c084][C2] | 元H/state/情報・総費用の比較規約と旧停止境界 |
| [新A/B静的調査 / 2397c5a][AB] | 新Aの単一shot reference干渉式、actual adjoint・path phase、canonical QPEとの条件差 |
| [status][status]・[manifest][manifest] | 有限RTEのlocal evidenceと後続の固定行列・理想gate・IID条件付き保証、未資格な元H表現包含・長round価格を区別 |

| 一次文献 | 実際に照合した箇所・確認深度 |
|---|---|
| **C-1** Clinton et al., *Quantum Phase Estimation Without Controlled Unitaries*, PRX Quantum **7**, 010345 (2026), [DOI][C1] | **出版社最終PDF**の§I B、§II A/B、§III、App.A/B/Cの必要箇所。VPR入力・支持/独立性・不定性、2Dのdummy/正値性、shot/古典処理と保証の限界を確認。[公開PDF][C1pdf]。著者版2410.21517v2と最終版の全文差分照合はしていない |
| **C-2** Yang et al., *Phase-Sensitive Quantum Measurement without Controlled Operations*, PRL **132**, 220601 (2024), [DOI][C2paper] | 出版情報と[著者版2308.10796v2][Yang]の本文Eqs.(1)–(9)、Supplement A.1–A.3、B、Cの関連箇所。虚時間近似・normalizer・投影・零点・費用を確認。出版社版との全文差分は未照合 |
| 周辺対照 Schiffer et al., *Hardware-Efficient Quantum Phase Estimation via Local Control*, PRX Quantum **6**, 040348 (2025), [DOI][localdoi] | [著者版2506.18765v2][local]の§II、III.1、App.A/F。local controlと完全control-free variantを区別するための限定読取 |
| PR原著 Günther et al., *Phase estimation with partially randomized time evolution* | [2503.05647v2][PR] App.A Eqs.(A18)–(A31)。線形なHadamard信号とnormalization。有限化の実際の規約はSOURCEを優先 |
| 新Aの基礎 Rowe et al., *Split-Evolution Quantum Phase Estimation for Particle-Conserving Hamiltonians* | [2604.14921v2][SE] §II Eqs.(1)–(3)と新A固定調査。reference/CSWAPとphase retrievalを同一方式としない |

出版社ページ・固定arXiv版を用いた限定検索であり、網羅的な先行研究調査・新規性審査ではない。C-1のPDF本文はweb経由で取得できた。shellの出版社PDF直取得は403だったが、公式配信APIでPDF取得もできた。PDF抽出用のローカルtoolが無かったため、本文照合にはwebの出版社PDF解析を用いた。環境へのpackage追加は行っていない。

## 3. C-1の要求条件

### 3.1 Vectorial phase retrieval（VPR）

C-1は `f₁(t)=⟨Φ|exp(+itH)|Φ⟩` に加え、`f₂(t)=⟨Φ|exp(+itH)|s⟩` と、`|f₁|², |f₂|², |f₁+f₂|², |f₁+i f₂|²` を用いる。追加state |s⟩と、相対位相を定めた |Φ⟩+|s⟩、|Φ⟩+i|s⟩の準備、|Φ⟩への投影が必要である。実際のnormalized superpositionの測定確率をこれらへ戻す際はそのnormも必要。直交時の簡単な準備例を任意のentangled groundへ移さない。[C-1 §II A, App.B Eq.(B3)][C1pdf]

理想的な一意性は、対応するDFTの多項式が共通rootを持たないことと、幅が標本数の半分以下の有限支持条件などを伴い、なお自明な不定性を残す。有限時間のspectral leakage、信号零点、測定noiseを含む安定性は別問題。C-1のrelaxed eigenvalue problemの小さい残差・gap診断を、指定成功確率のenergy certificateと同一視しない。[C-1 §II A, App.B][C1pdf]

R本のsecondary signalなら測定系列は `1+3R` 本で、N時間点の各shotに掛かる。relaxed固有値問題の変数数は `(R+1)N` であり、C-1はN固定でRを増やした場合の古典実行時間のcubicな増加も述べる。支持探索・memoryも別途必要である。一般の要求精度・失敗確率からN/R/shotを決める処方を本PRへ適用できる保証は未確認であり、論文の実験設定値を必要shotとして採択しない。C-1の基底state用GHZ型準備の説明は特定のcomputational-basis入力例である。今回のgroundについて、その回路・費用・取得可能性は未資格である。

### 3.2 Two-dimensional phase retrieval（2DPR）

C-1は `f(t,z)=⟨Φ|exp(+itH) exp(+izH_D)|Φ⟩` を測る。dummy Hamiltonian H_Dの実装、Hとの可換性、入力上の非自明な第二方向の変化、時間反転の共役対称性、適切なwindowでの実・非負DFTを利用する。単に時間点を二次元に並べる方式ではない。[C-1 §II B, Eq.(6), App.C][C1pdf]

二次元のgenericな一意性は、一次元の退化例を除外する条件であり、HIOの全入力に対する収束・noise下のenergy保証ではない。N×Mのgridでは第二時間方向Mの分だけ測定が増え、dummy evolutionと反復FFT・再始動も費用となる。FFTを使う通常の1反復の計算量と、必要反復数・成功率を保証することは別である。C-1 §IIIは実用条件での一般的な保証・資源scalingを未解決としており、RPEのshot数を流用する根拠にはならない。

### 3.3 不定性と固定groundへの直接代入

以下は**本研究条件への静的な代入結果**であり、C-1の数値結果ではない。以下ではrepositoryに合わせ `exp(-itH)` の符号に統一する。

`H|g⟩=E₀|g⟩` として、C-1のΦを供給groundに固定すると

$$
f_1(t)=e^{-iE_0t},\qquad
f_2(t)=e^{-iE_0t}\langle g|s\rangle.
$$

全secondary signalはtargetに比例し、直交stateならゼロである。normalized interferenceの確率も時間によらず、E₀の位相を識別しない。独立したspectral情報が生じるとは言えず、genericな一意性条件から絶対energyの識別は導けない。特に「groundの精度が良いほどVPRに有利」とは推論できない。

2Dでも、文献の順序のまま

$$
f(t,z)=e^{-iE_0t}\langle g|e^{-izH_D}|g\rangle
$$

なので、絶対値からH方向のE₀は消える。非縮退groundと `[H,H_D]=0` ならH_Dもground上でscalarとなる。縮退していてもH方向の因子は同じである。さらに、固定粒子数sectorでdummyを粒子数に選ぶだけでは非自明な第二方向を作れない。原文の複数sectorを含む入力例を置換しない。

これらは**同じgroundをbra側に使う文献の直接構成**についての制約である。追加readout/stateや別の許された情報アクセスを加える可能性まで否定しない。ただし、その資産、同じE₀への接続、費用は新たに示す必要がある。PRの各pathではgroundが固有状態でないため確率が変化し得るが、その変化を元HのE₀と同定する保証は別である。

C-1の不定性には全体位相、energy軸のshift、反転/共役が含まれる。`f(0)=1` は時間信号の定数phaseを固定できても、線形phase `exp(-ict)` を固定しない。`H→H+cI` のuncontrolled確率は不変である。既知のHamiltonian定数項cを最後に足すことと、未知の残りの絶対energyを測定で特定することを区別する。独立に得たenergy anchor、追加のphase-sensitive測定、または十分な公開priorの情報内容・費用が必要であり、E₀自体をanchorとして渡すのは循環である。[C-1 §I B, App.A, Eq.(B4)][C1pdf]

## 4. C-2：虚時間近似が要求するもの

C-2は `G(z)=⟨ψ'|exp(-iHz)|ψ⟩`, `z=t-iβ` のlogが解析的な領域で、`∂t arg G = ∂β log|G|` を利用する。`t±ih` の**未正規化振幅**を短い虚時間近似と実時間発展・投影から求め、既知の初期phaseから積分する。近傍のzero-free/下界条件、差分・積分・real/imaginary-time近似・統計誤差を伴う。離散零点への拡張も追加手順と条件を要する。[本文Eqs.(1)–(9), Supplement A/C][Yang]

必要資産は、local項に対するstate依存の近似unitary、normalizer `c_j^±=||exp(±hH_j)|ψ⟩||`、readout stateのunitary inverse、両符号の測定と古典積分である。短い相関長/local Hの条件が効率性に関係し、一般の相関stateではlocal tomographyや古典線形問題の費用がある。DF表現のZ/ZZがbasis共役で書けることだけでは、空間locality・短相関長・安価なtomographyを満たさない。stateを供給されることだけからnormalizerやunitaryを取得済みとしない。[Supplement A.3.1][Yang]

**固定groundでの代入結果**：全Hの厳密虚時間発展は

$$
e^{\pm hH}|g\rangle=e^{\pm hE_0}|g\rangle.
$$

normalized stateは同じgroundなので、理想実時間発展後のground survival確率は1であり、虚時間側のenergy情報はnorm `exp(±hE₀)` にある。このnormを無料で渡せば求める情報を先取りする。local項ごとのnormalizer・期待値を測定して情報を得る経路は排除されないが、取得shot・回路・古典処理を含めて比較する必要がある。その極限が既知のenergy期待値測定に還元されるかもレビュー対象である。normalized stateの不変性をC-2一般の不可能性と取り違えない。

文献の条件下で、系サイズn、積分区間長t、虚時間幅h、実時間Trotter幅τ・次数p、各時間点の測定数Mを用いると、phase誤差はITEで `O(n t h²)`、real-time近似で `O(n t² τ^p)`、統計で `O(I t/(h√M))` と整理される。Iは測定確率の逆平方根を時間平均した量で、小確率が高shot負担を生む。これらは同論文の条件付き誤差scalingであり、有限confidenceの全energy成功率や本PR用のMを指定する式ではない。追加のnorm取得誤差、時間点数、投影、tomography、積分処理も会計する。[本文Eqs.(6)–(9), Supplement A/B][Yang]

C-2本文の略号RTEは**real-time evolution**であり、このrepositoryの**randomized Taylor expansion**とは違う。文献のTrotter近似誤差式をPRの確率平均へそのまま移さない。新Cでは、PRのreal/imaginary-timeで一貫した信号、normalizationとlog微分を含む誤差接続は未資格である。

## 5. PRとの整合性：一次momentと測定確率

### 5.1 既存SOURCEが対象にしている信号

`a_ω(t)=⟨g|U_ω(t)|g⟩`、proposal p、weight wとして、対象は線形な複素信号

$$
\mu(t)=\mathbb E_p[w_\omega a_\omega(t)].
$$

既存canonical finite-RTEは正probabilityでsamplingし、paired Taylorの符号をevent unitaryへ含める。1 short-stepは次数K+1までのpaired polynomialをnormalization B_Kで割った平均で、occurrenceごとの積Bを用いて `Q_raw=Q_corr/B`、`Z_raw=Z_corr/B` と分ける。finite Kのcorrected operatorも一般にexact exponentialではない。異種occurrenceは順序・signed time・個別Bを保持する。[rte.py][S-rte] `finite_rte_distribution`, `_make_event`, `finite_rte_operator_moments`, `compose_finite_rte_occurrences`、[PR App.A][PR]

fresh-shot Hadamardは各pathのRe/Imを測るので期待値が線形である。対してcontrol-free survival測定をpathごとに行い平均すると、w=1の場合でも

$$
\mathbb E|a_\omega|^2-|\mathbb E a_\omega|^2
=\mathbb E|a_\omega-\mathbb E a_\omega|^2
$$

というvariance項が残る。一般wでもaをwaに置き換えた恒等式が成立する。probabilityへ古典weightを掛けるだけでは、異なるpath間の振幅cross termは生まれない。`U_ω` と `-U_ω` は同じsurvival確率だが、振幅平均では区別される。これは直接の恒等式による反証材料であり、sampling実験を行った結果ではない。

VPRのinterferenceも、同じrandom pathで測れば `E[a_ω conjugate(b_ω)]` を含み、`E[a_ω] conjugate(E[b_ω])` ではない。重ね合わせstateの準備だけでPRのcoherent meanのVPR入力になったとは扱えない。pathを跨ぐ干渉・coherentなLCU等の追加accessを仮定する場合、その実現・制御/ancilla・normalization・shot費用は別問題である。今回そうした推定器は設計していない。

### 5.2 scalar・weight・時間構造

- **符号とscalar**：event符号、component符号、constant/extracted-identity phaseはSOURCEに保持される。uncontrolled確率はpath scalarに不感なので、phase metadataの保存だけでは欠けた測定情報を復元しない。weight側とgate側の二重配置を避ける。[DF builder][S-builder]、[partial S2][S-partial]
- **normalization**：既知の正Bを掛けても上記のmoment差は消えない。raw attenuationを半径で課金する表現と、corrected測定のrangeで課金する表現を二重計上しない。signed/complex weightは確率でなく、complex weightならRe/Imの混合・range/momentまで必要。
- **unitary条件**：各U_ωがunitaryでも平均operatorは一般に非unitary。`E[UρU†]` というchannelから `E[U]` は決まらない。C-1の固定Hermitian Hによるpositive spectrum、2D共役対称性、C-2の同じ解析関数のlog微分は自動継承されない。
- **時間点の整合**：時間点ごとにfresh law、r/K、PFを変える平均は、単一のunitary groupの標本列とは限らない。固定seed/固定unitaryのpowersを使うなら別のsignalであり、元groundのbranch・E₀へのbiasを示す必要がある。有限PFやrandomizationが作ったleakageを新しい物理spectral情報として使わない。
- **保証の対象**：[SOURCE保証][S-guarantee]のgenerator/finite-bit/phase/radius/aliasは固定保存行列・理想gate・IIDモデルの範囲で読む。`ground_certificate` は保存state・energy・gap情報を使うoffline照合であり、新Cへ渡せる無料oracleではない。decoderの公開spectral boundと、検証用の真値を区別する。

## 6. 周辺のlocal-control方式と新Aとの違い

Schiffer et al.のsequential Hadamardは、既知のgate列の各prefix間の複素積を、単一gateのcontrolとgroundへの投影から測る既知法である。全prefixの測定、途中の小さいoverlap、state unpreparationの費用を伴う。固定unitary trajectoryへの文献手順の適用余地はあるが、複素信号を再構成してから平均する費用・biasと、確率を平均してから非線形再構成する方式は異なる。後者への保証は与えられていない。[§III.1, App.A][local]

同論文App.Fの完全control-free variantも、既知のprefix複素振幅と追加Pauli-rotation測定・例外処理を必要とする。どのprefixでも未知phaseを無条件に得るという主張ではない。元stateのコピー供給と、ground projectorを実装できるunitary inverseは違う。これらはC-1/C-2の直接置換とは別の既知法の適用候補であり、今回PR用に実装・資格確認していない。

新Aはreference/CSWAPによって**量子干渉でpathの複素信号**を得ようとする方式である。[SE-QPE][SE]のcanonical条件と、[新A固定調査][AB]のfresh-shot単一block条件を分ける。既知phaseのpathwise reference固有作用・actual adjointを要し、追加register/CSWAP/補正を課金する。新Cのphase retrievalとは入力・測定・保証が違う。新A pilotの未取得性能は空欄のまま比較する。

## 7. 強baselineと総費用への因果経路

| 比較対象 | 保持する強さ・未取得事項 |
|---|---|
| **Baseline 1：既存PR＋fresh-shot RPE** | basis非制御、diagonal/中央actionだけのcontrol、同basis共有、境界相殺、scalar/relative phase保持、適用可能な精度保持compilerを含む。古いgeneric全gate controlを唯一の対照にしない |
| **Baseline 2：新A reference干渉** | 同じ元state/target/成功条件で、追加register・CSWAP・phase補正・routing・実inverseを含める。今回の基準は固定静的調査であり、pilotの価格・優越は未取得 |
| **Candidate：新C** | secondary/重ね合わせstate、投影、dummy/ITE、全時間点・measurement family、tomography/normalizer取得、phase retrievalと必要な追加処理を含む。control-freeという名称は改善証拠ではない |

資源kごとの会計は、candidateの実際の時間grid Jと測定family A_jについて

$$
G_k^{(C)}=\sum_{j\in J}\sum_{a\in A_j}
 n_{j,a}\,\mathbb E[C^{\rm no\ original\ prep}_{j,a,k}],
\quad k\in\{\mathrm{CX,RZ,T,depth}\}
$$

と整理できる。元groundの準備以外の追加state操作・測定・postselection失敗も含める。追加qubit peak、古典再構成のtime/memory、情報取得・前処理・price calibrationは別記する。shot加重depthはwall timeではなく、RZ個数もT数ではない。全方式へ同じalias prior、情報・精度/失敗配分・compiler最適化機会を与える。

期待する因果経路は、**残っているcontrol費の減少 → 追加state/測定を含む単価 → 識別可能で安定な同target信号 → PR bias・attenuation・復元誤差 → 必要shot・全grid総資源**である。どれか一つの改善では最後まで接続しない。外部論文の回路depth比較を本DF SOURCEの総T/CX改善へ置換しない。

復元後に正当な同じ複素信号へのdisk ηと基準半径下界ρが得られた場合に限り、`η<ρ` から位相ずれ `arcsin(η/ρ)` 等へ接続できる。PR/PF/finite-bit/compiler/synthesisのbiasと、測定noise・復元conditioning・zero/alias・energyへの変換を分ける。全点の同時confidenceと復元失敗を総失敗予算へ入れる。C-1の数値noise実験やC-2の漸近誤差式から、既存RPEの整数shot・branch保証が自動的に得られるわけではない。

## 8. 科学レビューへ渡す問いと追加性質の候補

1. **測定target**：許されるaccessでPRの一次複素meanまたは同じE₀への認証可能な別signalを得られるか。二次momentしか得られない案を除外できるか。
2. **固定groundの情報**：C-1の直接構成の退化を解消する追加state/readoutは取得可能か。絶対energyのshift・反転をどう固定し、そのanchorをbaselineにも与えるか。
3. **C-2の契約**：locality/相関・normalizer・tomography・projectorは何から得るか。groundでのenergy情報を無料のnormに隠していないか。
4. **時間構造**：finite-RTEの非unitary mean、時間点依存のr/K/PF・Bを含め、文献の支持・positive spectrum・解析性を保てるか。保てないなら、既存定理の適用を外した追加研究となることを認識しているか。
5. **誤差と成功**：PR biasと復元誤差を分離し、同じenergy精度・成功確率を示す有限confidence接続は可能か。solver残差だけで認定していないか。
6. **費用利益**：強baselineのcontrolは既に軽量化されている。それでも全測定・追加state・古典費を支払う利益の機構があるか。最小人工確認で判別できることと実H4が必要なことを分けられるか。

既知VPR/HIO/QITE/local-controlを実装すること自体は既知法の応用接続である。独立した追加課題となり得るのは、上の条件を満たす**PR meanとの測定整合、固定groundで取得可能な情報とanchor、有限noiseでのenergy保証、総費用への接続**である。これらの性質が成立するとも、世界的に新しいとも本調査は判断しない。具体的な新推定器や証明案は作成していない。

最小確認案と反証時のSTOPは比較表§4にまとめた。今回は人工simulationさえ実施していない。取得accessと識別可能性・mean・保証の問題が解決する前に、実H4、長round、native priceを走らせる理由は示されていない。将来それらが必要となれば、ユーザーが条件・主資源・最小範囲を選び、必要な実装・科学実行を別途明示許可する。

## 9. 文書検査・公開照合・mandatory STOP

公開差分は本書とAPPLICABILITY_MATRIXの2文書に限定する。固定SHA/path、関数・testの所在、相互リンク、差分の空白を静的に検査する。科学的採択・新しい検証結果を追加していないため、既存の研究正本・validation manifest・旧索引は変更しない。今回の文書検査を科学計算、test PASS、immutable CI、外部再現の証拠と呼ばない。

commit/push後にremote branch SHAと2文書のremote内容（Git blob/取得bytes）をlocal commitと照合し、結果は最終チャットに報告する。この段落は照合手順を記したもので、未実施時の成功宣言ではない。

**公開照合を完了した時点でmandatory STOP。次の担当はGPT／ユーザー。** 新しい回路・phase retrieval・入力状態準備・推定器の設計実装、科学計算、科学的GO/NO-GOレビュー、旧研究の再開は自動開始しない。

[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md
[L]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md
[P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/track_c_o3_technical_preflight_20261106/REUSE_AND_BASELINES.md
[C2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3/docs/research/track_c2_limited_investigation_20261106/REPORT.md
[AB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/research/pr_reference_signal_static_investigation_20261106/REPORT.md
[objectives]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/research/研究目的・研究課題.md
[instructions]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/VALIDATION_STATUS.md
[manifest]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/artifacts/validation_manifest.json
[C1]: https://doi.org/10.1103/7qcr-znl2
[C1pdf]: https://journals.aps.org/prxquantum/pdf/10.1103/7qcr-znl2
[C2paper]: https://doi.org/10.1103/PhysRevLett.132.220601
[Yang]: https://arxiv.org/html/2308.10796v2
[localdoi]: https://doi.org/10.1103/zlp8-rz9n
[local]: https://arxiv.org/html/2506.18765v2
[PR]: https://arxiv.org/pdf/2503.05647v2
[SE]: https://arxiv.org/pdf/2604.14921v2
[S-rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/rte.py
[S-builder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_rte_qiskit.py
[S-partial]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_partial_s2.py
[S-guarantee]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/track_c_source_guarantee.py
