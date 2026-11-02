# finite-RTE研究：定理単位の先行研究監査と理論主張の独立レビュー

- レビュー日：2026-09-27
- 対象：`HIROMU1015/Partially-Randomized-Trotter`
- 指定ブランチ：`all-r-coherent-opt2-reoptimization`
- 固定commit：`71169d817c165b76a9e25fc6f6a16ade28ffe069`
- 主対象：C1／C2、および証明義務T1–T4。C3は評価対象の中心にしない。
- 最終推奨：**`TECHNICAL_NOTE`**
- 実施範囲：指定資料・保存済み証拠・一次文献の読解、定義と数式の独立な代数的検算、必要箇所のコード静的読解。新しい数値実験、対角化、sampling、compile、テスト実行、repository変更は行っていない。

## 1. 結論

**現時点の具体的な理論内容は、有限RTEの正しい解析・診断をまとめるtechnical noteとして整理することを推奨する。新しい一般手法または情報理論的最適性を確立した研究としては、まだ扱わない。**

C1の平均演算子、正scalar、物理信号半径の関係は、必要な仮定を補えば正しい。しかし、RTE／LCUの既知の平均式、正scalarの因数分解、標準的な複素平面の幾何から導かれる。現行FR-0の非可換積boundも、順序を保つ積展開・unitary共役・Cauchy–Schwarzによる直接の帰結である。

C2は次の三つを分ける必要がある。

1. **一般演算子について、`||Q||`と信号半径下界だけでnorm-diskを一様にstrict改善することはできない。** 第6.2節でdisk境界を達成する構成を示す。
2. **Hermitian構造などの追加情報を使ってnorm-diskより鋭くすることは可能だが、その一般的な機構には既知のq-numerical range／weak-valueの結果が直接適用される。** 第6.3節に解析的な代表例を示す。
3. **有限paired-Taylor RTEが実際に作れる誤差だけに限定した、情報別の最適性・識別不能性は未確定である。** 現原稿では、その最適性の対象となる情報写像と許容クラスがまだ定義されていない。一般演算子の極値を、そのままfinite-RTEの極値とすることはできない。

したがって、「数式が正しく、8例でnorm対照より小さい」という事実から、C2/T3の独立した新規性へ進むことはできない。一方、文献に同じ文章が見つからないことや、反対に一般的な数値域理論が存在することだけで結論も出していない。本レビューでは、実際に既存式へ還元できる部分と、還元できたとまだ言えない部分を区分した。

**T1–T4の現行の具体化についての主判定は、T1＝`DIRECT_COROLLARY`、T2＝`DIRECT_COROLLARY`、T3＝`UNRESOLVED`、T4＝`DIRECT_COROLLARY`である。** T2をI0での非自明な一様strict改善と読む場合は、その部分は`FALSE_OR_COUNTEREXAMPLE`となる。T4の判定は、本原稿で具体的に与えられたTaylor familyと一様剰余に対するものであり、T3で必要となる任意の極値Qのfinite-RTE実現性を証明したことは意味しない。

ここで`DIRECT_COROLLARY`は「同じ名前の定理がそのまま出版されている」という意味ではない。既知の結果または初等的な恒等式から、同じ仮定の下で本レビューに記した導出により得られ、現状では独立した理論上の差分を主張する根拠がない、という判定である。

既存の`MECHANISM_ONLY_NO_PRACTICAL_GO`は変更しない。今回の推奨は研究文書の位置付けについての判断であり、過去の実験gateの再分類ではない。

### 1.1 読解したrepository資料

以下はすべて固定commitのファイルである。特に断らない限り、節番号は各ファイル内の節番号を指す。

| ID | ファイル | レビュー上の用途 |
|---|---|---|
| R01 | `PROJECT_MAP.md` | 正本・履歴・結果の関係 |
| R02 | `docs/research/研究概要・現状.md` | 現在の停止点と主題の状態 |
| R03 | `research_focus_and_completion_plan_16d4482_20260927.md` | 完成条件、二固有値例、研究上の非目標 |
| R04 | `docs/research/fr_research_claim_and_manuscript.md` | C1/C2、I0/I1/I2、T1–T4、最終分類 |
| R05 | `docs/research/finite_rte_phase_amplitude_prior_art.md` | 既存のscoped prior-art audit |
| R06 | `docs/research/fr_revision_scalar_structure_contract.md` | 正scalar・情報層・強い対照の契約 |
| R07 | `docs/research/finite_rte_phase_amplitude_contract.md` | finite-RTE平均、積bound、位相／半径命題 |
| R08 | `docs/finite_rte_phase_amplitude_validation.md` | FR-1結果 |
| R09 | `docs/fr_revision_fr1a_posthoc.md` | scalar処理で説明された旧改善 |
| R10 | `docs/fr_revision_nonuniform.md` | FR-R1b結果・限界 |
| R11 | `VALIDATION_STATUS.md` | 文書段階と数値検証段階の区別 |
| R12 | `artifacts/validation_manifest.json` | 証拠の対応・状態・集計値 |
| R13 | `artifacts/fr_revision_nonuniform/2026-09-27/fr_revision_nonuniform_r1b_v1.json` | 保存済みsummary、gates、witnessの照合 |
| R14 | `src/trotterlib/fr_revision_nonuniform.py` | scalar最適化の意味の静的確認 |

固定commitの閲覧基点：
<https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/71169d817c165b76a9e25fc6f6a16ade28ffe069>

対象原稿：
<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71169d817c165b76a9e25fc6f6a16ade28ffe069/docs/research/fr_research_claim_and_manuscript.md>

## 2. 重大な問題点

### 2.1 平均演算子と平均channelを混同する表現

**該当：R04 §2.1の「paired samplingを平均した有限RTE channel/operator」。**

本研究のHadamard信号で用いるのは

\[
M=\mathbb E[U_\omega]
\]

という平均振幅演算子である。一方、random-unitary channelは

\[
\Phi(X)=\mathbb E[U_\omega XU_\omega^\dagger]
\]

であり、一般に`M X M†`とも一致しない。

最小の区別例は、`U_ω=+I,-I`を等確率で選ぶ場合である。このとき`M=0`だが`Φ=id`である。channelの固有位相が保存されても、平均演算子の信号は消失し得る。

**修正案：** 対象を「independently sampled unitary sequenceの平均振幅演算子」に固定する。channelに関するGuらの定理を使う箇所には、対象と仮定を移すための別の議論を必要とする。

### 2.2 I0/I1/I2の定義が完成原稿と数値契約で異なる

**該当：R04 §3と、R06 §6。**

| ラベル | 完成原稿R04 | 既存検証契約R06 |
|---|---|---|
| I0 | full relative errorの`||Q||`と`ρ_min` | 係数、`||h||`、Taylor remainder、`B_K`、局所norm区間 |
| I1 | QのHermitian/anti-Hermitian部、符号、sector、交換等 | 安価な代数的spectrum、involution、対称性、証明済みinterval |
| I2 | directional expectation、moment、状態部分空間 | denseの真のspectrum・ρ・extrema等のoracle情報 |

これは用語上の小さな問題ではない。特に`||Q||`というfull productの情報は、局所誤差の積boundと同じ入力ではない。状態情報が保証付きで外部から与えられる場合と、denseの正解から読み出す場合も異なる。

**修正案：** 「情報の内容」と「取得経路・費用」を別々に定義する。情報写像`I(x)`、その値を共有する許容問題の集合、正scalar選択に利用できる情報を明記する。それまでは、旧I1の8件を新I1に関する一般定理の証拠と同一視しない。

### 2.3 I0のstrict改善と、finite-RTE固有の改善が区別されていない

**該当：R04 §6のT2「I0/I1/I2ごとにnorm diskより厳密に強い十分条件」。**

一般の`Q`について、norm上界とsignal floor以外を使わないI0モデルではnorm-diskの接点を達成できる。したがって、I0で非自明な一様strict改善を与えるという読み方は偽である。第6.2節で構成を示す。

一方、`Q`がfinite paired-Taylorから来るという約束を使うなら、それは許容演算子クラスの制限である。この制限を明記せずに、一般演算子の不可能性またはstrict改善をfinite-RTEへ移してはいけない。

**修正案：** T2を「I0では何が最良か」「追加構造のどの情報が改善を許すか」に分解する。T3とT4を結び付ける部分では、反例が実際にfinite-RTEクラスへ属するかを証明する。

### 2.4 `phase/radius certificateがstrictに強い`の意味が曖昧

**該当：R04 §1.2、§3 C2、R10のstrict-gain解釈。**

位相上界が小さいこと、物理半径下界が大きいこと、両方を同時に改善することは別である。現行FR矩形boundは、位相を改善しても半径下界を悪くする場合がある。第6.4節に解析例を記す。

**修正案：** `strict phase improvement with a valid radius lower bound`と、`simultaneous phase/radius Pareto improvement`を区別する。8 witnessが支えるのは、登録された比較における位相上界のstrict差であって、一般的な両指標の優越ではない。

### 2.5 q-numerical rangeのcontainmentから最適性は出ない

**該当：R04 §4.2およびT3。**

containment自体は正しい。しかし、実際の`U,Q,ψ`は同じRTE列により結び付いている。q-numerical rangeでは固定Qに対してoverlap条件を満たす任意の二ベクトルを動かすため、実現集合を広げている。

また、各局所誤差`d_K(τh)`がnormalであっても、異なるframeの因子の積からできるfull `Q`は一般にnormalではない。normal-matrixの定理をfull `Q`へ直接適用するには、追加の証明が必要である。

**修正案：** 上界を得る緩和と、同じ情報で達成可能な最適値を区別する。`ρ_min`だけを知る場合は、原則として`ρ∈[ρ_min,1]`にわたる集合または最大値を扱う。固定overlapの値域に単に`ρ_min`を代入してよいとは限らない。

### 2.6 scalar除去は、一般の`h^4`の先頭項を全て除去しない

**該当：R04 §4.4の「r^-3の共通radial driftを除き」、§4.5の「分離する前の先頭位相項」。**

一般のHermitian `h`では`h^4`はscalarではない。正scalarで除けるのはその共通成分だけである。また、正scalarは信号位相を厳密に不変にするので、二固有値例の先頭位相係数は「除去後」にも残る。

**修正案：** `h^4=cI`となる特殊場合と非一様な場合を分ける。二固有値例の符号は、原稿の`exp(-iTh)`規約なら第6.6節の正符号に固定できる。「規約に応じた符号を除いて」のまま定理にしない。

### 2.7 “OPT”という数値ラベルは最適性証明ではない

**該当：R06 §4.2／§7、R14の`_optimize_fr_gamma`。**

静的読解では、norm側は有限spectrumのenvelopeに対する候補点を解析的に列挙している。一方FR側は、候補breakpointの区間ごとに`minimize_scalar(method="bounded")`を呼ぶ。区間ごとのunimodality、全停留点の網羅、外向き丸めを伴うinterval認証の証明は、この実装部分にはない。

これは返されたγでのcertificateが直ちに不正という意味ではない。**ある有効なγで強いnorm対照を上回る存在例には、FR側の大域最適性は不要**である。しかし「全γでこれ以上改善不能」「FRの大域最適値」は別途証明なしには言えない。登録探索区間も`[0.5,1.5]`であり、全正scalarに対する最適性とは異なる。

この指摘により過去のgateは変更しない。数値最適化の出力とT3の数学的最適性を区別するための注記である。

### 2.8 文献の書誌・スコープの修正が必要

**該当：R04 §5および末尾参考文献。**

- `arXiv:2111.10430`はXiantao Liの **Some Error Analysis for the Quantum Phase Estimation Algorithms**。原稿にある **Theory of Quantum Simulation with Product Formulas** ではない。
- `10.1080/03081089808818538`の正題はChi-Kwong Liの **q-numerical ranges of normal and convex matrices**。原稿の **The q-numerical range of a matrix** と一致しない。
- SPRINTは **Symmetry-Protected Randomized near-Integrable Trotter**。原稿§5の「signal-processing型resource-efficient phase inference」という説明は対象を取り違えている。

この三点は名称だけの修正ではない。何を既知として比較するかが変わるため、第3節の対象・仮定・結論に合わせる必要がある。

## 3. 定理単位の先行研究対応表

### 3.1 読解と版管理の範囲

以下では、実際に確認した定理・式だけに番号を付ける。論文に番号付き定理がない場合は式・節を示す。文献全文を取得できなかった場合は、書誌確認と定理の確認を区別する。

arXivの取得では、同一のversionless URLについてテキスト化結果とPDF画像が別の版を返す場合があった。特にPR論文ではv2のテキストとv1の画像が混在したため、**下表のPRのA18–A41は、先頭に2026-07-13／v2と表示されたテキストの番号**に固定した。v1画像のA3等をその番号と混ぜていない。Hu–Jinはversion指定HTML v2で定理を再確認した。

Li (1998)とTsing (1984)は、出版社／著者側の書誌は確認できたが、当該原文の定理本文を取得できなかった。したがって、その二文献の未確認定理を理由にC2を直接系と断定していない。代わりに、全文が読めたLi–Poon–Szeの著者公開原稿により、q-numerical rangeの具体的な既知還元を確認した。

### 3.2 一次文献対応

| ID・文献 | 確認箇所 | 仮定 | 結論 | C1/C2との対応・直接性 |
|---|---|---|---|---|
| W1 Günther et al. | Appendix A.2、A18–A29、Lemma A.2 (A30–A31)、Lemma A.3 (A41) | `H=Σp_l P_l`、`p_l`は確率、`P_l²=I`、独立sample、指定のRTE／partial-PF構成 | normalizationを掛けたHadamard平均がexact evolutionまたはexact-tail PFの信号に一致。normalizationの上界も与える | C1の出発点は既知。有限cutoff版は同じ和をtruncateする。state-conditionedなstrict最適性までは述べていない |
| W2 Wan–Berta–Campbell | Lemma 2、Appendix CのC2–C5、Appendix E.2 Theorem 4と証明 | 正規化Pauli和のrandomized LCU、有限cutoff M、Theorem 4では`r_j≥|t_j|`、複素係数`F_j` | paired Taylor LCU、normalization、有限打切りによる複素推定量biasをnorm・telescoping・Taylor tailで制御 | finite/truncated RTEと複素biasを扱うこと自体は既知。C1とT1の基本手法に直接近い。位相専用の情報制約下最適性ではない |
| W3 Gu et al. | Theorem 1、Supplementary IのS3–S8 | unitary **channel**へ弱いnoise channelを合成。Kraus演算子がHermitian。摂動論が成立する分離・小ささの条件 | noisy channelの固有位相はnoise強度の一次で不変。縮退空間の一次補正もHermitianとして処理 | Hermitian摂動による一次位相保護は既知。ただし指定状態の有限時間`arg<ψ|M|ψ>`を一様に保護する定理ではない。C2をそのまま直接系とはできない |
| W4 Ogawa–Kobayashi–Tomita | §II.A、Eqs. (3)–(7)、Table I。番号付き定理なし | `N(θ)=I+θC+O(θ²)`、一般C、非零pre/post overlap | normalized transition amplitudeの一次応答をweak valueで表す。modulus微分は実部、phase微分は虚部 | 原稿4.2の感度解釈は既知。Hermitian Cでもweak valueは複素で位相が一次で動き得る。有限積remainderの具体的評価とは分ける |
| W5 Li (1998) | 出版社書誌・abstractまで。定理番号未確認 | normal／convex matricesのq-numerical range | 当該クラスの値域の記述 | 原稿の正題を訂正する。全文未取得なので、この文献単独で直接還元を判定しない |
| W6 Tsing (1984) | 出版社・機関repositoryの書誌まで。定理番号未確認 | overlapを拘束したbilinear form、C-numerical range | 制約付き行列要素の幾何を扱う基礎文献 | 表現自体は古いが、全文未確認の特定定理を本レビューの結論に用いない |
| W7 Li–Poon–Sze | 著者公開全文、Theorem 3.1、Corollary 3.3、直前のq-range／Davis–Wielandt関係式 | quadratic operator `A²+αA+βI=0`、固定overlap。境界達成にnorm attainment条件 | quadratic operatorのq-numerical rangeを2×2圧縮の楕円diskへ還元。有限次元では対応する最大normが達成される | Hermitian二固有値例のstrict改善と最適性はこの既知幾何の直接系。一般の非可換RTE full Qがquadraticとは限らない |
| W8 Yi–Crosson | Lemma 2 (Eq.12)、Lemma 3 (Eq.18)、Theorem 1 (Eq.22) | unitary PF、effective Hamiltonian、対象subspace、gap条件。改善には一次固有値補正の消失等 | 固有値ずれと固有vectorずれを分離し、条件付きでTrotter／QPE誤差を改善 | unitary spectral errorと状態依存改善は既知。非unitaryなTaylor平均の有限時間信号／同情報最適性とは対象が違う |
| W9 Xiantao Li | Lemma 1 (Eqs.23–24)、Theorems 2.2、3.1、4.1 (Eqs.26、38、56) | 不完全固有状態のresidualとgap、近似unitaryの小さい整合誤差、random-unitary法の独立性・集中条件 | QPE成功確率をresidual、gap、近似誤差、ランダムstep数に結び付ける | 状態情報の取得・近似がQPEへ与える影響は既知。C2の有限RTE signal certificateを同じ出力として与えてはいない |
| W10 Casares et al. (SPRINT) | Appendix F、F8–F14、F17。該当箇所は番号付きcertificate定理ではない | randomized ordering／PFの摂動展開、小さいstep、cumulant近似とspectral解析 | 平均信号のspectral shift、damping、重み変化を解析 | 位相と振幅の区別は既知。近似展開をfinite-RTEの全域厳密certificateや情報理論的最適性と同一視しない |
| W11 Hu–Jin | v2 §II.1 Theorem 1 (Eq.6)、Theorem 2 (Eq.11) | `du/dt=-(A1+iA2)u`、Cartesian decomposition、存在するtime-ordered evolution。算法設定ではA1の正性等も使用 | phase-first／amplitude-firstのinteraction-frame恒等式 | nonunitary evolutionの分離は既知。一般のamplitude因子は正scalarでなく、指定信号の位相不変を意味しない。C1への概念的近接はあるが出力は異なる |

### 3.3 正確な書誌情報とURL

**[W1]** Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl, Aram W. Harrow, *Phase Estimation with Partially Randomized Time Evolution*, **PRX Quantum 7, 020332 (2026)**. arXiv:2503.05647。使用した式番号は取得したv2本文に基づく。

- <https://doi.org/10.1103/ynxb-p2xq>
- <https://arxiv.org/abs/2503.05647>
- <https://arxiv.org/pdf/2503.05647>

**[W2]** Kianna Wan, Mario Berta, Earl T. Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, **Physical Review Letters 129, 030503 (2022)**. arXiv:2110.12071v2。

- <https://doi.org/10.1103/PhysRevLett.129.030503>
- <https://arxiv.org/abs/2110.12071>
- <https://arxiv.org/pdf/2110.12071>

**[W3]** Yanwu Gu, Yunheng Ma, Nicolò Forcellini, Dong E. Liu, *Noise-Resilient Phase Estimation with Randomized Compiling*, **Physical Review Letters 130, 250601 (2023)**. arXiv:2208.04100v2。

- <https://doi.org/10.1103/PhysRevLett.130.250601>
- <https://arxiv.org/abs/2208.04100>
- <https://arxiv.org/pdf/2208.04100>

**[W4]** Kazuhisa Ogawa, Hirokazu Kobayashi, Akihisa Tomita, *Operational formulation of weak values without probe systems*, **Physical Review A 101, 042117 (2020)**. arXiv:1912.10222。

- <https://doi.org/10.1103/PhysRevA.101.042117>
- <https://arxiv.org/abs/1912.10222>
- <https://arxiv.org/pdf/1912.10222>

**[W5]** Chi-Kwong Li, *q-numerical ranges of normal and convex matrices*, **Linear and Multilinear Algebra 43(4), 377–384 (1998)**。

- <https://doi.org/10.1080/03081089808818538>
- <https://www.tandfonline.com/doi/abs/10.1080/03081089808818538>
- 著者の業績一覧：<https://cklixx.people.wm.edu/pub.html>

**[W6]** Nam-Kiu Tsing, *The constrained bilinear form and the C-numerical range*, **Linear Algebra and its Applications 56, 195–206 (1984)**。

- <https://doi.org/10.1016/0024-3795(84)90125-3>
- <https://hub.hku.hk/handle/10722/156092>

**[W7]** Chi-Kwong Li, Yiu-Tung Poon, Nung-Sing Sze, *Elliptical range theorems for generalized numerical ranges of quadratic operators*, **Rocky Mountain Journal of Mathematics 41(3), 813–832 (2011)**。確認箇所は著者公開原稿のTheorem 3.1およびCorollary 3.3。

- <https://doi.org/10.1216/RMJ-2011-41-3-813>
- 著者公開全文：<https://cklixx.people.wm.edu/quadra.pdf>

**[W8]** Changhao Yi, Elizabeth Crosson, *Spectral analysis of product formulas for quantum simulation*, **npj Quantum Information 8, 37 (2022)**。

- <https://doi.org/10.1038/s41534-022-00548-w>
- 公開全文：<https://www.nature.com/articles/s41534-022-00548-w>

**[W9]** Xiantao Li, *Some Error Analysis for the Quantum Phase Estimation Algorithms*, **Journal of Physics A: Mathematical and Theoretical 55, 325303 (2022)**. arXiv:2111.10430v3。

- <https://doi.org/10.1088/1751-8121/ac7f6c>
- <https://arxiv.org/abs/2111.10430>
- <https://arxiv.org/pdf/2111.10430>

**[W10]** Pablo A. M. Casares, William Maxwell, Danial Motlagh, Hitarth Choubisa, Zy Niu, Ignacio Loaiza, Jonathan E. Mueller, Arne-Christian Voigt, Juan Miguel Arrazola, Stepan Fomichev, *Theory and practice of Trotter product formulas for quantum chemistry*, **arXiv:2606.30741v1 (2026)**。

- <https://arxiv.org/abs/2606.30741>
- <https://arxiv.org/pdf/2606.30741>

**[W11]** Qitong Hu, Shi Jin, *Quantum Simulation of Non-Unitary Dynamics via Amplitude-Phase Separation*, **arXiv:2602.09575v2 (2026)**。定理番号はv2 HTMLで確認。

- <https://arxiv.org/abs/2602.09575>
- 版指定全文：<https://arxiv.org/html/2602.09575v2>

### 3.4 直接還元の具体的な範囲

**W1/W2からC1へ：** even Taylor termと次のodd termを組にし、係数の符号をsampled unitaryへ吸収すると、有限和のnumeratorは`P_(K+1)`、確率正規化は`B_K`となる。正scalarを任意に因数分解することは代数的な再表示であり、新しいsamplerではない。

**W4から一次位相機構へ：** `N(θ)=I+θQ`とpre/postselectionを本信号の`ψ,U†ψ`に合わせれば、一次位相はweak valueの虚部である。Hermitian Qだから虚部がゼロになるのではない。前後の状態が一致する等の条件が必要である。

**W7およびq-rangeの定義からC2の一般的な部分へ：** 固定overlapの値域をnorm diskより小さく拘束すること、その二固有値例での最適性は既知の幾何に属する。第6.1節の積剰余と組み合わせれば、全行列のq-rangeを厳密計算せずに局所値域の和で囲うことも、定義から導ける。

**直接還元できたとまだ言えない部分：** 固定されたfinite-RTE生成規則、同じ情報写像、同じscalar規則の下での達成可能集合そのもの。その最適性は単なるq-range containmentより強い要求である。ただし、現原稿ではその要求を具体的な定理にしていない。

## 4. 数式・定義の監査

### 4.1 finite-RTE平均と正規化

局所Hamiltonianを

\[
h=\sum_\ell p_\ell P_\ell,\qquad p_\ell\ge0,\quad\sum_\ell p_\ell=1,
\quad P_\ell=P_\ell^\dagger,\quad P_\ell^2=I
\]

とする。符号は`P_l`へ吸収できる。`τ=λs`のようなdimensionless timeを使う場合は、`h=H_R/λ`との対応を固定する。単なる`||h||≤1`と、実装で使用する確率分解の指定は区別する。

偶数`K≥0`について

\[
P_{K+1}(-i\tau h)=\sum_{n=0}^{K+1}\frac{(-i\tau h)^n}{n!},
\qquad
B_K(\tau)=\sum_{\substack{n=0\\n\text{ even}}}^{K}
\frac{|\tau|^n}{n!}\sqrt{1+\frac{\tau^2}{(n+1)^2}}
\]

と定めれば、指定のpaired samplingで

\[
\mathbb E[\widetilde U_K(\tau)]
=\frac{P_{K+1}(-i\tau h)}{B_K(\tau)}
\]

となる。[W1,W2] 負時間の符号は多項式とsampled unitaryに保持し、確率正規化には絶対値を用いる。cutoff Kは通常Taylor次数K+1を残すという規約である。

各occurrenceのdrawが独立なら、非可換であっても順序を維持して

\[
\mathbb E[\widetilde U_N\cdots\widetilde U_1]
=(\mathbb E\widetilde U_N)\cdots(\mathbb E\widetilde U_1)
= A_{\rm corr}/\mathcal B
\]

とできる。相関samplingにこの因数分解は一般に成立しない。等確率の±Iを同一乱数で2回使えば、`E[U_ω²]=I`だが`(E[U_ω])²=0`である。

### 4.2 scalarの扱い

`D_j=U_j†A_j−I`、実数`c_j`、`γ_j=1+c_j>0`として

\[
\widehat D_j=\frac{D_j-c_jI}{\gamma_j},\qquad
A_j=\gamma_jU_j(I+\widehat D_j)
\]

は厳密な恒等式である。`Γ_c=∏γ_j`より

\[
A_{\rm corr}=\Gamma_c\widehat A_{\rm corr},\qquad
A_{\rm mean}=(\Gamma_c/\mathcal B)\widehat A_{\rm corr}.
\]

`Γ_c/𝔅`は解析上取り出した物理的な信号倍率であり、samplerの減衰を物理的に除去したことを意味しない。controlled平均では`diag(I,A_mean)`のcontrol-1 branchだけに残る。全systemのglobal scalarとして消せない。

`γ_j=0`なら再中心化不能、`γ_j<0`ならπの位相が付く。複素scalarを使えばさらに位相が変わる。正scalarという制約は証明に不可欠である。

### 4.3 weak-value型の比とq-numerical-range containment

`z0=ρe^{iφ}≠0`、`u=ψ`、`v=e^{iφ}U†ψ`なら

\[
\langle v|u\rangle=e^{-i\phi}\langle\psi|U|\psi\rangle=\rho,
\]

\[
\frac{\widehat z_{\rm corr}}{z_0}
=1+\frac{\langle v|Q|u\rangle}{\rho}.
\]

符号・共役は原稿の定義で正しい。[W4]の一次応答の形に加え、この比の書換え自体はexactである。

\[
W_\rho(Q)=\{\langle y|Q|x\rangle:
\|x\|=\|y\|=1,\ \langle y|x\rangle=\rho\}
\]

と定義すれば、実際の比は`1+W_ρ(Q)/ρ`に含まれる。数学文献との内積規約の違いは、実overlap `ρ≥0`と本定義を明記して吸収する。q-numerical rangeのqは反復回数qとは無関係である。

affine変換は

\[
W_\rho(\alpha Q+\beta I)=\alpha W_\rho(Q)+\rho\beta
\]

であり、identityの移動量にρが付く。この点はscalar recenteringの幾何と整合させる。

### 4.4 norm diskと絶対半径

`||Q||≤E`、`ρ≥ρ0>0`とする。`E<ρ0`なら

\[
\left|\frac{\widehat z_{\rm corr}-z_0}{z_0}\right|
\le E/\rho\le E/\rho_0=\eta<1.
\]

よって

\[
\Delta\phi\le\arcsin(E/\rho_0),\qquad
|z_{\rm obs}|\ge\frac{\Gamma_c}{\mathcal B}(\rho_0-E).
\]

相対比の`1−η`を、ρを掛けずに観測半径へ使わない。実際のρが既知なら`(Γ_c/𝔅)ρ(1−E/ρ)`である。`ρ0`だけの場合は上記の下界を使う。

signal floorは分母の安定性を保証するが、`A_corr`側の信号がゼロにならない条件も必要である。normの場合は`E<ρ0`、FRの場合は後述の`L>0`がその十分条件になる。

### 4.5 branch規約

位相誤差は

\[
\Delta\phi=\left|\operatorname{Arg}
\bigl(z_{\rm corr}\overline{z_0}\bigr)\right|
\]

という円周距離とする。個別のprincipal argumentを単に引くと、branch cutの両側で人工的な2π差が生じる。正scalarによる位相不変性は、両信号が非零であることを条件とする。

matrix logarithmを使う漸近展開では、`τ=0`から連続なbranchを指定する。noncommuting productのlogを、根拠なく各因子のlogの和へ置き換えない。

### 4.6 P3展開と二固有値の符号

\[
P_3(-ix)=1-ix-\frac{x^2}{2}+i\frac{x^3}{6}.
\]

原点から連続なlogに対し

\[
\log P_3(-ix)
=-ix-\frac{x^4}{24}-i\frac{x^5}{30}
+\frac{x^6}{72}+O(x^7)
\]

であり、原稿の係数はこの範囲で正しい。局所相対誤差は

\[
e^{ix}P_3(-ix)-1
=-\frac{x^4}{24}-i\frac{x^5}{30}
+\frac{x^6}{72}+i\frac{x^7}{252}+O(x^8).
\]

logとrelative errorは最初の差が8次からなので、4–6次が一致しても同じ関数と混同しない。

二固有値例のsigned phaseは第6.6節で検算する。原稿の規約では`r^-3`係数の符号は正である。`T,a,b`を固定しない一様主張には、bounded domainとsignal floorの追加が必要である。

## 5. T1–T4判定表

| 項目 | 主分類 | 判定対象と理由 | 成立する部分／残る部分 |
|---|---|---|---|
| **T1** | **`DIRECT_COROLLARY`** | 現行FR-0のfull ordered productに対するnorm remainderとphase/radius bound | 第6.1節の導出で非可換性を保って成立。新しいorder-sensitiveな改善やsharp boundを得たわけではない |
| **T2** | **`DIRECT_COROLLARY`** | 現行FR boundが登録norm-diskよりstrictになる、明示的な十分不等式 | 第6.3–6.4節で導ける。一般的なI1のstrict改善は既知q-rangeからも得られる。I0で非自明な一様strict改善まで要求する部分は`FALSE_OR_COUNTEREXAMPLE`。情報最適なcertificateの主張は未定義 |
| **T3** | **`UNRESOLVED`** | finite-RTEに限定し、同じ情報値を共有する問題族での限界・最適性 | 一般I0ではdiskがsharpであり既知幾何の直接系。Hermitian二固有値の限界も既知。finite-RTE全体の情報写像・許容クラス・scalar固定規則が未定義で、現原稿の最有力新規性主張はまだ命題になっていない |
| **T4** | **`DIRECT_COROLLARY`** | 原稿に具体的にあるP3反復、二固有値finite-RTE例、bounded Hermitian familyに対する一様remainder | 第6.5–6.6節でfunctional calculusと解析的Taylor remainderから導ける。一般のrank-one極値Qやq-range全体をfinite-RTEで実現する部分は未証明であり、T3の未解決な達成可能性問題に属する |

### 5.1 分類を強く読みすぎないための注意

- T2の`DIRECT_COROLLARY`は「現FR式の比較条件は作れる」という意味であり、どのI1/I2情報にも対応する最適certificateが完成したという意味ではない。
- T4の`DIRECT_COROLLARY`は「一様剰余が一般に難しくて証明不能」という問題ではないことを示す。finite-RTE由来の任意のQが自由に選べることは示さない。
- `POSSIBLY_NOVEL`は今回は付けない。唯一残る候補はT3のRTE制約付き最適性だが、その定理自体が具体化されていないため、現時点で新しい定理候補と評価するより`UNRESOLVED`とする方が正確である。
- sourceに同じ式がないことは`POSSIBLY_NOVEL`の理由にしない。

## 6. proof skeletonまたは反例

以下では、初等的に確認できる小命題には導出を与える。これらはC2/T3全体の証明完了を装うものではない。未解決なfinite-RTE最適性は第6.7節で明示する。

### 6.1 T1：順序を保つ非可換積とFR bound

**仮定。** 有限個のunitary `U_j`、局所corrected factor `A_j=U_j(I+D_j)`を考える。scalar再中心化後なら`D_j`をその再中心化誤差に置き換える。deterministic factorも`D_j=0`として列に含めてよい。

\[
U=U_N\cdots U_1,\qquad A=A_N\cdots A_1,
\qquad V_{j-1}=U_{j-1}\cdots U_1,
\]

\[
\widetilde D_j=V_{j-1}^\dagger D_jV_{j-1}.
\]

**積の恒等式。** 因子を順に移動すると

\[
U^\dagger A=(I+\widetilde D_N)\cdots(I+\widetilde D_1).
\]

交換を仮定せず、元の順序を保つ。したがって

\[
Q=U^\dagger A-I=\sum_j\widetilde D_j+R,
\]

\[
\|R\|\le R_2:=\prod_j(1+e_j)-1-\sum_j e_j,
\qquad \|Q\|\le E:=\prod_j(1+e_j)-1,
\]

ただし`||D_j||≤e_j`。このboundは高次の順序付き積をsubmultiplicativityで抑えただけなので、任意の非可換列にvalidだが、順序依存の相殺を利用してはいない。

**信号への移送。** `D_j=F_j+iG_j`、`||F_j||≤a_j`、`||G_j||≤b_j`、`a=Σa_j,b=Σb_j,s=Σe_j`とする。正規化状態ψに対し

\[
|\eta\rangle=(U^\dagger-z_0^*I)|\psi\rangle,
\quad \langle\psi|\eta\rangle=0,
\quad \|\eta\|=\sqrt{1-|z_0|^2}.
\]

よって

\[
\frac{\langle\psi|A|\psi\rangle}{z_0}
=1+\langle\psi|Q|\psi\rangle
+\frac{\langle\eta|Q|\psi\rangle}{z_0}.
\]

`ρ≥ρ0>0`、`κ0=√(1−ρ0²)/ρ0`とすれば、最後の項は絶対値`κ0E`以下。Hermitian／anti-Hermitian成分の期待値を実部と虚部へ振り分けて

\[
L=1-a-R_2-\kappa_0E,
\qquad Y=b+R_2+\kappa_0E
\]

を得る。`L>0`なら

\[
\Delta\phi\le\arctan(Y/L),
\qquad |z_{\rm obs}|\ge(\Gamma_c/\mathcal B)\rho_0L.
\]

ここまでの仮定・順序・剰余は明示でき、現行命題の基本構造は正しい。

**既知q-rangeとの関係。** 定義からunitary invarianceとMinkowski包含

\[
W_\rho\!\left(\sum_j\widetilde D_j+R\right)
\subseteq\sum_j W_\rho(D_j)+\mathbb D(0,R_2)
\]

が得られる。各局所値域をCauchy–Schwarzで矩形に囲えば、上記型のFR certificateへ至る。これは、full Qの数値域を計算しなくても局所情報から囲える、という点も含む。局所化だけを独立した新しい原理とするには、さらに既知の囲いを超える具体的な結果が必要である。

### 6.2 T2/T3：一般I0でnorm-diskはsharp

**対象クラス。** 全てのunitary U、unit ψ、`||Q||≤E`、`|<ψ|U|ψ>|≥ρ0`を許す。`0<E<ρ0≤1`とする。これは一般演算子クラスであり、finite paired-Taylorという制限はまだ課していない。

overlapが`<v|u>=ρ0`となるunit vectorsを選び、`U†u=v`となるunitaryを選ぶ。任意の`|ζ|≤E`に対して

\[
Q=\zeta|v\rangle\langle u|
\]

とすれば

\[
\|Q\|=|\zeta|,\qquad
\frac{\widehat z}{z_0}=1+\frac{\zeta}{\rho_0}.
\]

したがってdisk全体が達成される。`η=E/ρ0`として接点

\[
\zeta=\rho_0\{-\eta^2+i\eta\sqrt{1-\eta^2}\}
\]

を取れば、位相は`asin η`に等しい。`ζ=−E`で絶対半径の下界`ρ0−E`も達成される。

**結論。** normとfloorだけを知る一般I0では、norm-diskより一様に小さい位相上界も、一様に大きい半径下界も保証できない。I0のstrict改善を要求する部分には、この反例がある。

**適用境界。** このrank-one Qがfinite-RTEから出ることは示していない。したがって「finite-RTEでもdiskが常に最適」という結論へは移せない。

同じ構成で`ρ0=ε²`、`ζ=iε`とすると、operator perturbationが0へ近づいても位相は`atan(1/ε)`へ動く。signal floorのない一般一様保証がないことも分かる。

### 6.3 T2/T3：Hermitian二固有値の改善は既知の楕円幾何

具体的な既知還元として、Hermitian二固有値演算子`Q=cI+dZ`を考える。q-numerical rangeの楕円定理[W7]、またはその2次元の直接導出により、`W_ρ(Q)`の中心は`ρc`、実軸半径は`d`、虚軸半径は`d√(1−ρ²)`となる。

`c=0`、`0<d<ρ≤1`のとき、補正比の領域`1+W_ρ(Q)/ρ`は、中心1、実軸半径`d/ρ`、虚軸半径`d√(1−ρ²)/ρ`の楕円diskである。接線の幾何から

\[
\beta_{\rm ell}
=\arctan\frac{d\sqrt{1-\rho^2}}{\sqrt{\rho^2-d^2}}
\]

となる。一方、norm-diskは

\[
\beta_{\rm disk}=\arcsin(d/\rho)
=\arctan\frac{d}{\sqrt{\rho^2-d^2}}
\]

なので、`0<ρ<1`ではstrictに改善し、`ρ=1`では正の実軸上に留まって位相は0である。

これは「Hermitian情報があればstrictに強くできる」「二固有値ではその範囲がsharp」という内容を、既知理論だけで与える。したがって、この一般的な主張をC2/T3の新規性とすることはできない。

一方、任意のnoncommuting finite-RTE full QはHermitianでもquadraticでもない。これを一般のfull productの最適性へそのまま適用するのも誤りである。

### 6.4 現行FRのstrict条件と、phase/radiusの非同時改善

第6.1節で`E<ρ0`かつ`L>0`とする。FR位相boundがnorm-diskより小さいための必要十分な数値不等式は

\[
\frac{Y}{L}<\frac{E}{\sqrt{\rho_0^2-E^2}}.
\]

これは二つの既存boundを比較した直接の帰結であり、この条件を書くこと自体は新しい定理ではない。

半径もstrictに改善するなら、別に

\[
\rho_0L>\rho_0-E
\]

が必要である。両者は同値ではない。

例えば単一のHermitian相対誤差で`a=e=E`、`b=R2=0`、`ρ0=4/5`とすると、`κ0=3/4`で

\[
L=1-7E/4,\qquad Y=3E/4.
\]

十分小さいEで位相boundはnorm-diskより小さくなるが、FRの半径下界は

\[
\rho_0L=4/5-7E/5
\]

で、norm-diskの`4/5−E`より小さい。この例は「位相をstrict改善したからphase/radiusの両方をstrict改善した」とする推論を否定する。

実装では有効な二つの位相上界のmin、半径下界のmaxを採ることはできるが、それも標準的なboundの組合せである。

### 6.5 T4：P3のoperator展開と一様remainder

ここでは新しい数値計算ではなく、解析的な証明順序を示す。

**固定する成立域。** Hermitian h、`||h||≤H_*`、`|T|≤T_*`、`M=T_*H_*`を固定する。rは正整数とし、`r≥max(1,4M)`を取る。

1. 複素disk`|z|≤1/2`では
   \[
   |P_3(-iz)-1|\le\frac12+\frac18+\frac1{48}=\frac{31}{48}<1.
   \]
   したがって原点からのlog branchが解析的に定まる。
2. `f(z)=log P3(-iz)+iz`はこのdiskで有界。例えば
   \[
   |f(z)|\le M_0:=\log(48/17)+1/2
   \]
   を使える。
3. Cauchy評価により、`|z|≤1/4`で7次以上のremainderを定数×`|z|^7`で一様に抑えられる。例えば係数和から`2^8 M_0|z|^7`という保守的上界を選べる。
4. Hermitian hのspectral theoremによりscalar評価をoperator normへ移す。よって
   \[
   r\log P_3(-iTh/r)
   =-iTh-\frac{T^4h^4}{24r^3}
   -i\frac{T^5h^5}{30r^4}
   +\frac{T^6h^6}{72r^5}+O_M(r^{-6})
   \]
   が得られる。
5. 同じhの関数同士は可換なので、この段階の指数展開は正当である。二次以上の指数remainderを`e^x−1−x`型に抑えれば
   \[
   P_3(-iTh/r)^r=e^{-iTh}
   \left[I-\frac{T^4h^4}{24r^3}
   -i\frac{T^5h^5}{30r^4}+O_M(r^{-5})\right]
   \]
   となる。bounded normを固定したため、定数は行列次元に直接依存させる必要がない。

**phaseへの移送。** さらに`|z0|≥ρ0>0`を固定し、signalの相対補正が例えば1/2未満になるrを取れば、`Arg(1+w)=Im(w)+O(|w|²)`を同じbranchで一様に使える。

**非可換列へ広げる場合。** 異なる`h_j`を含む積に対してlogを加算してはいけない。局所remainderを先に抑え、第6.1節の順序付き積で合成する。Nもrとともに増える場合には、`Σe_j`および積remainderを明示的に追う。scalar再中心化後は`γ_j`が正で0から離れる条件も必要である。

この証明骨格で、原稿のP3展開の一様化は十分に成立可能である。ただし、通常の解析的Taylor展開とfunctional calculusの直接適用であり、これだけを独立新規性にはしない。

### 6.6 T4：二固有値の先頭項とfinite-RTE内部反例

`h|a>=a|a>`、`h|b>=b|b>`、`|ψ>=(|a>+|b>)/√2`とし、`d=(a−b)T/2`、`cos d>0`とする。

\[
z_0=e^{-i(a+b)T/2}\cos d.
\]

第6.5節より、normalized signal ratioの展開は

\[
\frac{z_{\rm corr}}{z_0}
=1-\frac{T^4}{24r^3}
 \left[\frac{a^4+b^4}{2}
 -i\frac{a^4-b^4}{2}\tan d\right]
-i\frac{T^5}{30r^4}
 \left[\frac{a^5+b^5}{2}
 -i\frac{a^5-b^5}{2}\tan d\right]
+O(r^{-5}).
\]

したがって、原稿のnegative-exponent規約でのsigned phaseは

\[
\boxed{
\operatorname{Arg}\frac{z_{\rm corr}}{z_0}
=\frac{T^4(a^4-b^4)\tan d}{48r^3}
-\frac{T^5(a^5+b^5)}{60r^4}
+O(r^{-5})
}
\]

である。第1項の符号は正であり、signed Tにも対応する。絶対位相誤差はこの式の絶対値であって、符号付き係数そのものとは区別する。

uniformな主張では`|a|,|b|,|T|`を有界にし、`cos d≥ρ0>0`、r十分大という条件を置く。`cos d>0`は便利な十分branch条件であり、円周距離を正しく扱う別branchまで否定するものではない。

**実際のpaired-Taylor class内の例。** identityを含まない表現として

\[
h=\frac{Z\otimes I+I\otimes Z}{2}
\]

を取り、固有値1の`|00>`と固有値0の`|01>`の等重み重ね合わせを使う。これはHermitian involutionsの確率和であり、paired-RTEの許容tailである。deterministic interleavingをidentityにすれば、r回のfinite Taylor列そのものになる。

`0<T<π`では

\[
\operatorname{Arg}(z_{\rm corr}/z_0)
=\frac{T^4\tan(T/2)}{48r^3}
-\frac{T^5}{60r^4}+O(r^{-5}).
\]

よって、単一固有値で位相誤差が`r^-4`でも、一般superpositionのfinite-RTE信号は`r^-3`になり得る。正scalarで割ってもこの位相は消えない。

これはfinite-RTE内部で実現する限界例として使える。ただし、標準Taylor展開と二分枝の干渉からの直接の帰結であり、この一例だけを新しい一般最適性定理としない。また、「全てのfinite-RTEでnorm-diskがsharp」とも示していない。

### 6.7 未解決なT3を定理にするために足りない定義

情報理論的主張をするなら、少なくとも以下を含む一つの数学的対象が必要である。

\[
\mathcal A(i)=\{x\in\mathcal C_{\rm RTE}:\mathcal I(x)=i\},
\qquad
\Theta^*(i)=\sup_{x\in\mathcal A(i)}
\left|\operatorname{Arg}\frac{\widehat z(x)}{z_0(x)}\right|.
\]

ここでxには、許すHamiltonian分解、時間・cutoff・反復、deterministic interleaving、入力状態、正scalar規則を含める。`I(x)`には、正確な値か上界か、局所量かfull-product量か、取得費用を含むかを区別して書く。

T3を成立させるには、上からのcertificateだけでなく、同じiを共有するfinite-RTEの列がその値を達成する、または任意に近づくことが必要である。第6.2節の一般rank-one構成も、q-numerical range全体も、その達成可能性を自動的には保証しない。

**現原稿には、この`C_RTE`と`I`がまだ固定されていない。** したがって、ここは「あと一つの不等式を証明すれば完成する穴」ではなく、新しい定理の対象を定義する段階にある。未定義な最適性に対し、証明可能・不可能や新規性を確定させない。

## 7. 既存証拠で言えること／言えないこと

### 7.1 保存済み結果の照合

R10、R11、R12の該当entry、R13の`summary`／`gates`／`gate_witnesses`を照合した。

| 項目 | 保存値 | レビューでの位置付け |
|---|---:|---|
| matrix conditions | 20 | 固定toy gridの実行範囲 |
| state rows | 61 | 状態行数 |
| method records | 610 | 登録手法×状態の記録 |
| applicable records | 227 | boundの適用条件を満たす記録 |
| soundness failure | 0 | 指定数値許容差内で反例未検出 |
| common-scalar strict witness | 8 | 同じscalarでの位相bound差 |
| optimized strict FR-gain witness | 8 | 登録最適化比較での位相bound差 |
| one-sided fixed-budget witness | 0 | 固定予算でFRだけ通る条件はなし |
| decision | `MECHANISM_ONLY_NO_PRACTICAL_GO` | 維持する |

- result fingerprint：`affac0ae8132450ccb2de3512b6a463a3f9d7b1ac8a6f12cc38303ac75e891d4`
- 文書記載のfile SHA-256：`e2a6f9326951fe67e979022dc733704e0c036d3342ab3cb317c5f77b885aeebc`
- このレビューでは上記hash値の一致する記述を照合した。raw bytesからのhash再生成や628 testsの再実行はしていない。
- artifactはdirty worktreeで生成されたlocal evidenceである。外部独立再現やimmutable CIと呼ばない。

### 7.2 言えること

**第一に、登録したtoyで実装の意味と数式の整合が数値的に確認された。** ordinary／controlled平均、負時間、cutoff対照、適用条件の処理が、保存された許容差内で整合した。

**第二に、正scalar-only処理で説明しきれない、登録FR式の位相上界の改善例がある。** ただし既存の全幾何学的certificateに対する優越ではなく、登録されたscalar-norm baselineとの比較である。

**第三に、固定予算での実用的GOは得られていない。** この判断はそのまま保つ。逆に、三つの任意予算で差がないことだけで、あらゆる用途で価値がないと証明されたとも言わない。

### 7.3 言えないこと

- 227件のpassによってT1の任意長・任意次元の定理が証明された。
- 8件によってT2の一般十分条件、T3の最適性、T4の全域一様remainderが示された。
- 数値上のstrict差によって既存q-range／weak-value理論からの独立性が示された。
- 物理半径の有効な下界があることだけで、phaseとradiusの両方をstrict改善した。
- supplied `ρ_min`を使ったことにより、そのcertificateを安価に取得・準備できると示された。
- `OPT`というラベルにより、すべての正scalarに対する数学的最適性を示した。
- 同じ8件が、非可換順序の最適化を証明した。

最後の点について、witnessは`ν=0,0.5`、`q=4,8`の四組が、それぞれ可換・非可換条件に現れた計8行である。対応するbound値は同じであり、現boundがlocal spectrum、factor数、共通のρ下界から作られていることと整合する。**非可換列でvalidだったことと、非可換順序を利用して改善したことは別である。**

### 7.4 数値証拠が証明義務を置き換えているか

R04には「passは証明義務を埋めない」と明記されており、その方針は妥当である。ただし「8 witnessはT2の存在例」という表現は、「数値計算上の存在候補」と限定するのがより正確である。厳密な存在証明にするには、対象となる式のanalytic比較または丸め誤差まで制御した検証が要る。

本レビューの第6節は、そのために新しいgridを回すのではなく、どの一般的な主張がすでに解析的に導けるかを整理したものである。旧数値結果やgateを書き換えるものではない。

## 8. 最終分類と次の一件

### 最終分類：`TECHNICAL_NOTE`

**1. 既知研究との差**

現時点で具体的に書かれているC1、非可換積のnorm remainder、weak-valueの比、norm-disk、代表的なHermitian strict改善、P3および二固有値例の一様化は、既知結果または初等的な導出で得られる。これらを正確に統合する価値はあるが、独立した新手法の主定理が成立したとする根拠には不足する。

文献全文が取得できなかった部分は明示した。その不足を「同じ定理は存在しない」と読み替えていない。また、Gu、Yi–Crosson、Hu–Jinの対象は本研究と完全には一致せず、それら一冊が本研究の全てを解いているとも判断していない。

**2. 成立しそうな中心定理**

現行FR-0を、正scalar・順序付き積・signal floor・円周位相・物理半径まで含めて正しく定式化したcertificateは成立可能であり、第6.1節の導出がその骨格になる。P3のbounded Hermitian familyに対する一様remainderと、一般superpositionで`r^-3`位相誤差が残るfinite-RTE例も成立可能である。いずれも現状の具体化では直接の帰結として扱う。

**3. 最大の未解決点または反例**

一般I0のstrict改善にはnorm-disk達成例がある。新規性候補として残るのは、fixed-informationのfinite-RTE到達可能集合に限定したsharpな限界だが、現原稿では情報写像と許容列がまだ定義されていない。これは一つの完成済み命題に残る技術的な補題ではなく、主定理そのものの具体化が未完という状態である。

そのため、`ONE_OPEN_ITEM`を理由に継続を自動許可するより、**現在証明できる内容はtechnical noteとして閉じる**判断を推奨する。将来、具体的な新定理が得られる可能性を否定するものではない。

**4. 次に行うべき作業一件**

> **現原稿を「有限RTEの平均演算子・位相／半径評価のtechnical note」へ一度だけ整理する。** その一作業の中で、channel表現と書誌・情報ラベルの不整合を直し、第6節の既知／直接帰結の導出を配置し、未定義の情報最適性は達成済みcontributionから外す。FR-R1bの結果と`MECHANISM_ONLY_NO_PRACTICAL_GO`はそのまま保存する。

新しい検証系列、数値grid、応用計算をこの判断の付属条件にはしない。technical noteとしての位置付けは採録を保証するものではなく、現在の成果を過不足なく確定するための推奨である。
