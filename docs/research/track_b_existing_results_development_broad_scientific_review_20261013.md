# Track B：既存成果を発展させる研究方針の横断的科学レビュー

- 作成日：2026-10-13（日本時間）
- レビュー開始承認：利用者の「レビューを開始して」
- 対象：選択肢②「既存成果の発展」。closed P5単独に限定せず、return集約、再配分、合成配置、保証、回路構造、限定negativeを含める。
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- 状態：**科学的レビュー完了。既存成果を用いた新しい発展RQと設計方針を選定。新しい方法の優位・独立新規性・論文成立を認定したものではなく、科学計算の実行は未認可。**
- 既存方針との関係：旧source、結果、個別closure、authorization、markerを維持する。新しい設計の検討と、旧runの再開を分ける。Track A・N3へ介入しない。

## 0. 結論

**②を進めることを支持する。** ただし、非採択だった候補を順番に再実行するのではなく、既存成果に残った構成・positive・限界を、別の科学的な問いの下で発展させる。

最も優先する研究軸は、次である。

> **有限RTEの代数的集約・乱択表現・有限精度実装を、同じ目標時間発展とconfidenceへ戻したとき、どの条件で総資源が減り、どの条件で利益が失われるか。その違いを、利用可能な情報から説明・予測できるか。**

この問いの核には、G系列の一般構成とclosed P5、R1の再配分・角度依存性、SPの合成負担の累積を置く。最初から全てを一つの新samplerへ合体させることはしない。closed P5は重要な検証例であり、必ず最良となる主役ではない。

成果の形は、(a)具体的な構成・生成可能性、(b)資源利益の定量的な成立条件、(c)同一taskの実装比較、のいずれか、または論理的につながった組合せでよい。「新しい数学primitiveを発明する」「全既知法に勝つ」ことだけを成功条件にしない。

**FRの位相・半径保証は、独立した第二の発展候補として評価する。** 数学的な改善が実際に残っており、予算選択が変わらなかったことと理論的価値がないことは別である。一方、位相taskを主軸のcomplex-signal taskへ混ぜず、FR-R2を自動再開しない。

DF基底変換・run-level実装改善は応用研究として候補に残す。ただし、今回の主軸を基底費用の話へ無理に結び付けない。generic whole補正・Gaussian積生成・単純相関法の既存判断は維持する。

| 判断対象 | 今回の決定 |
|---|---|
| ②の研究設計 | 継続を支持。既存成果に基づく発展RQを選定 |
| 第一の研究軸 | 有限RTEの表現・生成・有限精度実装による総資源利益の成立条件 |
| closed P5 | 中核的な既存構成・検証例。単独・普遍的なwinnerとしては未採択 |
| SPとの組合せ | 同一targetで接続できる場合の仮説。保存改善率を掛けない |
| FR | 独立理論候補。今回数値系列は再開しない |
| DF回路構造 | 別の応用候補。基底費用を主軸の採択理由へ強制しない |
| 新しい独立アルゴリズムの確立 | 未確立 |
| 次の担当 | Codex：採択RQに沿う技術具体化・不足証拠の切分け・最小比較計画を一束で準備 |
| 新規科学計算 | 別途明示承認が必要。今回実施なし |
| 旧系列のclosure・mandatory STOP | 維持。新しいRQの設計は旧実行の再開ではない |

## 1. 前回の一時休止判断から、何を変えるのか

前回の休止記録[R3]は、具体的な新機構がないまま新候補の実装系列を連続的に立ち上げない判断である。PR改善一般の不可能性や、既存成果を発展させる価値まで否定していない。

今回、利用者は②を明示的に選び、既存成果を広く評価するよう依頼した。したがって、評価対象は「今から全く新しい手法を一つ見つける」ことから、「既存の構成や限界から、解く価値のある次の問題を組み立てる」ことへ移った。

**変更するのは今後の研究設計の優先順位であり、過去の証拠の判定ではない。** 例えば、BMの同値性はそのまま、G5の固定class分離はそのまま、Gaussian候補の方法差未定義もそのままである。これらをpositiveへ書き換えない。

また、前回の回答はclosed P5の資源拡張を中心に挙げすぎた。今回、FRの同情報での境界改善、SPの非分離な配置効果、G系列の実装可能性そのものも比較すると、②の対象はより広い。これらを十分に比較せず、②まで一律に休止へ含めるのは適切でない。

ただし、「利用者が継続を望んだから有望と認定する」という判断でもない。以下で、各成果が支持する内容、研究として未成立な部分、必要な追加作業を分ける。

## 2. 証拠の取得範囲とレビューの水準

### 2.1 今回直接取得した主要資料

GitHubの固定commitから、G6–G10完成ノート、exact資源表への入口、P5・canonical generator source、R1.5帰属報告、SP-1結果・契約、G5終了認証、BF/BM、P-A機構判別、FR-R1b、DF補正closure、基底費用診断、休止記録を取得した。[R1–R16]

資料は、単にPASS件数や冒頭結論を見るだけでなく、target、比較自由度、保証、native会計、保存結果の意味まで確認した。特に、G7/G9のsourceを読み、P5とpartial-returnの係数・normalizerの対応を確認した。

### 2.2 今回していないこと

全branchの再調査、全eventのraw再集計、全synthesis列の行列検証、科学sourceのimport、repository tests、native compile、T合成、Hamiltonian・trajectory生成は行っていない。G10 exact tableは取得したが巨大な単一JSONで表示が部分的であり、全17行を新たに独立再認証したとはしない。数値表の主要値は、既存の保存表・報告の転記である。

原科学結果の数理・数値保証を全て再証明するレビューではない。既存の証拠の限定を保ち、研究方針を横断的に判断するレビューである。

### 2.3 区分

- **保存事実**：既存報告、固定source、結果表にある内容。
- **本レビューの解析**：既存式の展開、条件付きの会計式、設計上の含意。
- **新しい研究課題**：今後検証すべき問い。未実装・未実証。
- **研究判断**：どの問いへ次の労力を使うか。新規性・性能の証明とは別。

## 3. 既存成果を「成功／失敗」だけで分類し直さない

| 系列 | 残っている成果 | 旧判断が止めたこと | 発展先としての位置付け |
|---|---|---|---|
| G6–G10、closed P5 | 正値性、局所query、有限bit補償、少数群生成、固定nativeのT利益・trade-off | 一般fullが常に実用的、全PR/QPE優位、一般priorityの無条件主張 | 最もまとまった構成・理論・実装の出発点 |
| R1/R1.5 | 再配分後の角度・precision・biasと資源順位の変化 | 一部のPareto残留を全baseline勝利とすること | 表現変更の利益が何に由来するかの機構資料 |
| SP-1 | 少数wrapperでの大きな利益と、累積時の反転 | 一般D/R配置則、実DF-RTE、全compiled費用の主張 | finite-RTEと合成を接続する際の条件付き仮説 |
| G5/G10下界 | 指定した費用・辞書・confidence classでの排除・分離 | あらゆるsampler・実装・物理最低shot数の排除 | 比較可能なclassを明示する理論・判定手法 |
| FR-R1b | 同じ利用可能情報で位相上界が厳しくなる機構 | 固定予算での実用GO、FR-R2の自動実行 | 別の理論発展候補 |
| P-A/WP06 | run-levelのbasis選択利益、sequence依存性 | interval分割独自の寄与 | 応用・実装研究、比較対照の基盤 |
| BF/BM | finite objectiveの限定negative、同情報でのBCH同値性 | 同じ仮説のparameter拡張で新手法とすること | 強い対照・同情報原則を与える制約 |
| DF whole補正 | 条件付き高次化、少数軌道支持、局所費用分離 | generic wholeを主線として拡張すること | 数理成果は保持するが直近の優先度は低い |
| Gaussian積生成・単純時間相関 | 成立条件、既知法との対応、未取得費用の特定 | 方法差未定義のまま実装へ進むこと | 直近の新系列には戻さない |

上表は改善率を横に並べて順位を付けたものではない。各系列はtarget、fixture、oracle、compiler、precision、confidenceが異なるためである。

## 4. 第一の発展軸：有限RTEの表現と実装を、研究対象として扱う

### 4.1 RQを具体化する

主RQを、三つの連結した問いへ分ける。

**RQ-A：同じ有限時間発展を表す方法を変えると、何が改善するのか。**
ordinary、partial-return、closed P3/P5、一般local full、明示Pauli情報を使える場合のmatched CTSを、同じ有限多項式の表現として比較する。normalization、回転角度、event support、利用する情報、生成費用を分離する。

**RQ-B：その改善は、有限精度実装を戻したときにどこで残り、どこで反転するのか。**
R1のprecision依存、G10の頻出root回転価格、SPのsampling負担の累積を使い、実装の粗さと統計負担の交換を調べる。ただし、原SPのtoyの改善率を他fixtureへ移さない。

**RQ-C：同じ物理的時間発展へ接続した場合、どの条件が実用的な差を決めるのか。**
有限Taylor targetを比較するだけでなく、必要に応じてその多step積を同じ時間・精度・confidenceへ戻す。ここで最初からQPE基底状態エネルギーまでを必須としない。有限時間のcomplex signalというtaskでまとまる研究も正当である。

この三問は、単なる「もっと実験点を増やす」ではない。代数、情報取得、実装価格、物理時間への接続のどこに差があり、どこにないかを説明する研究である。

### 4.2 目指す貢献を限定する

「costとvarianceを掛けることが大切」は既知であり、論文の中心的新発見にはできない。importance samplingの最適lawも既知である。[P1]

狙うべきなのは、特定のfinite-RTE構成に対する、検証可能な次のような内容である。

- return集約の統計的利益が、入力の係数分布・時間刻み・次数にどう依存するか。
- full集約と低次数特殊化で、同じensembleの情報取得費用と予算の差がどこに生じるか。
- 既存の強いpartial-returnとの差が、物理的なstep数を戻したときに残る条件。
- 角度・合成precisionの変化で順位が反転する条件と、その頑健性。
- exact event表・true signal・未知normalizerを無料で使わず、どこまで意味のある比較ができるか。

このうち何を主要claimにできるかは今後の結果と近接文献との対応による。今回「新しい普遍的資源最適化理論」を確立したとはしない。

## 5. G系列とR1から分かった、具体的な発展余地

### 5.1 closed P5はpositiveを持つが、対照ごとに利益が違う

固定3-system-qubit provider、p=(1/5,3/10,1/2)、x=5/7、同じfinite P5の保存値は次のとおりである。全数値は百万単位の表示丸め。T/CXは別資源であり、1QにはTも含まれる。[R1, R2]

| 方式 | T切片 | CX切片 | 期待accepted calls K |
|---|---:|---:|---:|
| ordinary | 371.085420 | 21.904736 | 2.651626 |
| partial-return | 275.437399 | 16.998103 | 2.019804 |
| closed P5 | 272.239831 | 16.649261 | 1.951680 |
| matched CTS | 387.633441 | 10.391732 | 3.396808 |

資源式はG_j(h_j)=C_{0,j}+K h_jで、h_jは共通の準備・読出し単価である。切片だけで全PR費用とは呼ばない。

closed P5のT切片はordinaryより約26.6%、matched CTSより約29.8%低い。一方、partial-returnとの差は約1.16%であり、CXではmatched CTSの方が低い。これは「全法より30%改善」ではない。[R1]

**約1.16%を無価値とみなす必要も、次の研究の打切り閾値にする必要もない。** 単一の局所taskのG比を、多stepの改善率上限とする根拠はない。しかし、最も近い対照との差を省き、遠い対照との大きな差だけで研究を進めることも適切でない。

### 5.2 同じensembleでも、利用できるnormalizerが違う

G10のm=3/5では、general fullとclosedは同じidealの非zero event係数表現を共有する。ただし、local側はenvelope・zero-fillとnormalizer上界、closed側は少数群normalizerによる直接生成を使い、実proposal、受理率、weight、十分予算の扱いは同一ではない。[R1] 登録費用の差を、新しい理想演算子を構成した利益と読まず、また上界の精密化だけに全差を一意帰属させない。

したがって、この差は以下を分ける自然な対照になる。

1. 演算子・ensembleを変更した効果。
2. 同じensembleに対して、利用できる正規化情報が異なる効果。
3. zero-fill／直接生成の実proposal・weight、および有限bitの係数の違い。
4. その情報を取得する古典費用。

これは「より正確な上界が常に得」という主張ではない。正規化の精密化で量子試行を減らしても、その取得負担やonline費用が大きければ別のtrade-offとなる。古典秒数とT数を無次元で加算せず、別座標または明示した換算モデルで扱う。

m=7のfullとP5+tailは別ensembleであり、上の情報効果だけで同じ説明をしてはいけない。

### 5.3 R1の「normalizationは下がるがshotは増える」を使う

R1.5のx=1/8、native precision=1e-4では、再配分AのB²は低い側にあるが、bias上界は高く、十分shot数はordinary/PTSCよりわずかに多い。それでもper-shot T価格が低いためT総費用が下がった。一方、CXの順位は同じでない。[R4]

さらに同じrho half-angleのT数は、1e-3/1e-4/1e-6で40/46/70、比較するx half-angleは36/52/68である。precisionによって同じrepresentationの優劣が反転する。[R4]

この保存事実は、normalizationだけの目的関数では説明できない。次の研究では、B²だけでなく、残余統計幅と実装角度の費用を戻す必要がある。ただし、これは保存された特定合成列についての事実で、最適合成の一般定理ではない。

### 5.4 完全な集約を強制することと、選択肢を増やすことを区別する

ある表現が高価になったことから、「代数的情報を増やすと必ず損」とは言えない。元の表現も選択可能なまま候補集合を広げるなら、その集合上の最小費用は増えない。これは集合包含の初等的事実であり、新しい最適化定理ではない。

一方、候補の取得や選択に費用がかかるため、実際のアルゴリズムでその最小値を安価に達成できるとは限らない。full集約を必ず使うpolicyの敗北、情報を取得する価値、実装可能な選択則は分ける。

この区別を研究へ生かす場合も、既知optimizerやRA-D0の係数探索を名前だけ変えて新規としない。

## 6. 本レビューの解析：強い対照との小step差を先に見る

本節は既存sourceの式からの手計算による展開であり、新しい数値実験ではない。一般priorityや新定理のclaimではない。使用するnormalizerは、ideal canonical表現の係数L1質量であり、finite-bit・zero-fill・upper-bound予算と区別する。

正のp_i、Σp_i=1、χ=Σp_i²、固定P5、x→0^+とする。[R5, R6]

### 6.1 ordinary

ordinary paired表現の係数質量は

\[
 B_{\rm ord}(x)=\sqrt{1+x^2}
 +\frac{x^2}{2}\sqrt{1+x^2/9}
 +\frac{x^4}{24}\sqrt{1+x^2/25}
 =1+x^2+O(x^4).
\tag{1}
\]

### 6.2 partial-return

既存CanonicalGeneratorのpartial_return_tailでは、rootの実・虚係数が

\[
 a_0=1-\chi x^2/2,\qquad b_0=x-\chi x^3/6
\]

であり、distinct pair群とordinary高次数tailを足す。

\[
\begin{aligned}
 B_{\rm partial}(x)
 &=\sqrt{a_0^2+b_0^2}
 +\frac{(1-\chi)x^2}{2}\sqrt{1+x^2/9}
 +\frac{x^4}{24}\sqrt{1+x^2/25}\\
 &=1+(1-\chi)x^2+O(x^4).
\end{aligned}
\tag{2}
\]

### 6.3 closed P5

closed P5のrootはa_0=1−χx²/2+O(x⁴)、s_0=x+O(x³)である。従ってroot質量は1+(1−χ)x²/2+O(x⁴)。長さ2のparent群の総質量は(1−χ)x²/2+O(x⁴)、長さ4はO(x⁴)なので、

\[
 B_{\rm P5}(x)=1+(1-\chi)x^2+O(x^4).
\tag{3}
\]

従って、ordinaryからpartial-returnへ得た先頭次数の差と、partial-returnからclosed P5への追加差は異なる。

\[
 B_{\rm ord}-B_{\rm partial}=\chi x^2+O(x^4),
\qquad
 \log\frac{B_{\rm P5}}{B_{\rm partial}}=O(x^4).
\tag{4}
\]

(4)の第二式は追加差の符号や有限xでの大きさを固定しない。pの高次momentや全係数を要する。また、sourceにはP3のquartic比較を返す既存helperがあり、small-x展開一般を新しい着想として扱わない。[R5]

### 6.4 反復による増幅を正しく区別する

固定dimensionless総時間τをr等分しx=τ/r、各stepのideal canonical係数samplingが独立で、拒否なし・weight一定Bの場合に限定する。正規化weightの二次momentはB(x)^{2r}である。

\[
 \log\frac{B_{\rm P5}(\tau/r)^{2r}}
 {B_{\rm partial}(\tau/r)^{2r}}
 =O(\tau^4/r^3),\qquad r\to\infty\text{ with fixed }\tau.
\tag{5}
\]

つまり、「一stepの小さいnormalization差も、反復すれば必ず大きな利益になる」とは言えない。同じ総時間でstepを小さくすると、その差自体が速く小さくなる場合がある。

ただし、(5)は**normalization由来の理想momentだけ**の式である。native価格の離散変化、range項、bias、rounding、古典生成、別のstep数選択を含まない。G10の局所G比1.16%をr乗する式でもない。これは発展RQの具体的な比較軸を与えるもので、closed P5の非優位を証明するものではない。

## 7. SPを組み合わせる研究は、何を新たに問うのか

### 7.1 保存された利益の意味

SP-1のC、n=16では、NONE比の総T費用がD-only=1.3195、R-only=4.2817、DR=0.1026である。n=32のDRは3.1606となる。単一配置が不利でも同時配置で有利になる条件と、累積で不利に転じる条件が実在する。[R7]

しかし、D/Rにはgate数、角度、符号、外側coinの複数差がある。これは「ランダム部分にだけprobabilistic synthesisを適用すべき」などの一般則ではない。原対象はtoy coherent wrapperであり、finite RTEではない。[R8]

### 7.2 二つの保存改善率を掛けてはいけない

外側RTE表現をM=Σ_e a_e V_e、proposalをq_e、weightをa_e/q_eとする。各eventの**既に制御されたjoint回路**を、内側のsigned合成へ分解し、条件付きL1質量をΓ_eとする。実際の内側分岐を絶対係数で選び、±1測定Yを行う場合、

\[
 M_2=\sum_e\frac{|a_e|^2}{q_e}\Gamma_e^2,
\qquad
 R_{\max}=\max_e\frac{|a_e|\Gamma_e}{q_e},
\qquad
 \bar C=\sum_eq_e\,\mathbb E[C\mid e].
\tag{6}
\]

一般に、M_2は「外側のmoment×平均内側moment」へ分離しない。表現を変えるとa_eだけでなく角度、Γ_e、回路費用、合成誤差も変わる。従ってP5の改善率とSPの改善率の積から総利益を得ることはできない。

(6)は条件付き期待値からの会計恒等式であり、新しい定理の主張ではない。歴史依存の内側weightの場合はΓ_e一定の表記を使わず、条件付きweight二乗期待値へ戻す。

### 7.3 first momentとchannelを混同しない

元RTEはcoherent first operator momentを使う。system-only channelを同じにする合成を挿入しても、controlled枝間の位相が保たれるとは限らない。

SPの契約は、logical controlled作用をjoint-spaceへlowerし、共通fusionを行ってからPAIを配置する。これにより、ancillaを含むobservableの意味を扱っている。[R8]

将来両者を接続するなら、この対応を同じtargetで確認する必要がある。制御前system unitaryのglobal phaseを捨てない。既に制御された回路全体の共通phaseとの区別も保つ。誤差合成には元のsigned-channel/first-momentの条件を戻す。

### 7.4 最小の研究仮説

最初に問うのは、「P5とSPを一緒に使えば最良か」ではない。

> **表現を変えて得る統計的利益と、合成方法を変えて得る実装上の利益は、同じtaskで独立に足し合わせられるか。それとも、角度と条件付きweightを通じた相互作用が表現の順位を変えるか。**

比較が技術的に成立する場合だけ、表現2種類×合成2種類の対応する四条件を用意すれば、相互作用を識別できる。ただし、この「四条件」は設計構造の例であり、今回新しいfixture、seed、実行budgetを登録したわけではない。

同一taskでの費用をG_{r,s}とすれば、

\[
 \mathcal I=\frac{G_{1,1}/G_{0,1}}{G_{1,0}/G_{0,0}}
\tag{7}
\]

は表現効果が合成選択で変わるかの記述量になる。I≠1だけで新規性・統計的有意性・実用性を認めず、誤差幅、nativeの価格、共通費用を確認する。

SPを接続する意味が薄い、あるいは一般の既知probabilistic synthesisと全く同じ説明しか残らない場合、SPを主研究に無理に付けず、RTE表現の適用条件研究へ限定する。

## 8. 同じ物理taskへ戻すための数学契約

### 8.1 同じ有限targetなら、比較は段階的に行える

M_jが各stepの有限Taylor演算子、U_jが理想unitaryで、||M_j−U_j||≤ε_jとする。積のtelescopingから

\[
 \left\|M_r\cdots M_1-U_r\cdots U_1\right\|
 \le\prod_{j=1}^{r}(1+\epsilon_j)-1
\tag{8}
\]

を保守的に使える。各U_jのnormが1であることを使い、M_jをunitaryやCPTP contractionとは仮定しない。

同じM_jを表現する複数ensembleなら、独立step samplingで同じideal finite mean積を表す。共通の決定論blockを同じ位置に置く場合も、そのideal対象の一致は保持される。これは表現比較を始める上で有利である。

ただし、finite-bit版はexact rational targetが異なり得るため、それぞれの係数biasを戻す。native実装誤差も各stepのnormとweightへ結び付ける。

### 8.2 比較する誤差

以下を別項目として記録する。

1. 外側PFの近似と、内側Taylor打切り。
2. 表現・係数・確率の有限bit近似。
3. 回転、control、basisを含む実装誤差。
4. finite-shot統計誤差。

同じm/x/targetの比較では打切り部分を共通にできる。mやstep数を変える場合は、その共通性はなくなる。誤差上界の保守性が選択へ影響するため、truth-free保証と事後のexact診断を分離する。

### 8.3 使う資源会計

outer ensembleが正係数a_e、proposal q_e、Y²=1ならM2=Σa_e²/q_e、range=max a_e/q_e。axisごとに正の残余統計幅s_aを確保し、例えば既存の保守的Bernstein policyは

\[
 n_a\ge\left\lceil\log(2/\alpha_a)
 \left(\frac{2M_{2,a,+}}{s_a^2}+\frac{4R_{a,+}}{3s_a}\right)\right\rceil.
\tag{9}
\]

ここで「必要最小shot数」とは呼ばない。共通の評価規則での十分予算である。

\[
 G_j=C_{{\rm init},j}+\sum_a n_a\mathbb E[C_{a,j}]
\tag{10}
\]

へ実装費用を戻す。実際の量子shotを実行しなくても、所定の精度・confidenceに対する資源予算の研究はできる。ただし、何を実測し何をモデルから得たかを区別する。

T、CX、RZ、depth、古典時間は異なる座標である。Tと1Qの二重加算を避け、人工重みで後付けwinnerを作らない。使用可能情報や準備費用を変えるなら、条件を明示する。

## 9. 新規性と研究価値：既知のものを何と区別するか

### 9.1 既知の原理を新発見にしない

- 固定ensembleの費用と二次momentを最適化するimportance samplingは既知。[P1]
- 安い回路とsampling overheadを交換するprobabilistic synthesisは既知。[P3]
- Taylor表現をunitaryのstochastic combinationとして扱うことや、層ごとのsamplingも既知。[P2]
- 乱択した廉価回転により時間発展を構成するTE-PAIも近接する。適用範囲・observable・制御化・費用を揃えず比較しない。[P4]

例えばCauchy–Schwarzによる

\[
\left(\sum_eq_ec_e\right)\left(\sum_ea_e^2/q_e\right)
\ge\left(\sum_ea_e\sqrt{c_e}\right)^2
\]

およびc_e>0でq_e∝a_e/√c_eは、新しい主要結果にはしない。zero-cost、range項、bias、古典取得、integer shotまで含めた最適性がこの式だけで従うわけでもない。

### 9.2 既存構成の「組合せ」でも研究は可能

部品が既知であることと、対象に関する新しい結果がないことは同じではない。一方、部品を列挙しただけのpipelineも十分でない。

今回の研究を成立させるには、例えば以下のいずれかが必要である。

- finite-RTE固有の係数構造に対する、意味のある一般条件・保証を示す。
- 同じ演算子・情報・実装機会で、従来の比較では分からなかった資源順位の成立範囲を説明し、独立条件で確かめる。
- 必須の係数・normalizer・角度を、比較対象より有用なアクセス条件・費用で取得する具体的構成を示す。
- 既知法の適用研究として、有限時間精度まで含む再現可能な資源評価と新しい適用知見を示す。

これらを全て同時に達成する義務はない。一般定理が主成果なら、大量の分子benchmarkを必須にしない。実装上の定数因子利益が主成果なら、普遍的な漸近改善を必須にしない。

### 9.3 G5の結果を何へ使うか

G5は、固定dictionary・非負degree matching・所定precision・同じconfidence policyなどのclassを、強い保存CTS lawが下回った結果である。[R9]

この結果を、全ての表現や新しい合成列を禁止する根拠にはしない。一方、同じclass内の再探索を、別名の新手法として再開もしない。下界は、どの自由度を変えなければ既存の結果を超えられないかを明示する道具となる。

そのようなclass区分や資源下界を発展させる理論研究も可能だが、既存不等式を再掲するだけでは独立寄与は弱い。新しいclass、利用可能情報、有限precisionでの意味を明確にする必要がある。

## 10. 独立した第二候補：FRの保証を発展させる

FR-R1bは、同じ利用可能情報を使い、正scalar処理後にもFRがnorm境界より厳しくなる8条件を確認した。例えばν=0,q=8ではNORM=2.089763×10⁻⁵rad、FR=1.520264×10⁻⁵radである。[R10]

固定の予算10⁻²、10⁻³、10⁻⁴radでは認証できる条件が同じだったため、旧実用gateは不通過だった。**これは理論的な改善がゼロという結果ではない。**

発展RQは、次のように置ける。

> **同じ取得可能な情報の下で、finite-RTEの半径方向と位相方向を分けた保証が、どの入力クラスで通常のnorm保証より本質的に鋭くなるか。改善を得るために必要な半径情報は、どの費用で取得できるか。**

この問いは、FR-R2を再実行して、既存の二boundの間へ予算を後付けで置くこととは異なる。その操作は既存データで選択差を人工的に作るだけであり、独立した実用witnessにはならない。

理論研究なら、改善が生じる構造、tightness、必要情報、他のscalarly optimized norm boundとの比較を主成果にできる可能性がある。ただし、scalableな情報取得と近接文献への独立差は未確定である。

また、位相精度を良くすることとcomplex signal精度を良くすることは同じではない。positive scalarは位相を変えなくてもsignal半径・shot負担を変え得る。FRを主軸へ合体してP5の資源勝利を作ることはしない。

**今回の優先順位は主軸の次とするが、価値のない残り候補とは扱わない。** 主軸とは違う理論研究を希望する場合の、具体的な選択肢である。

## 11. その他の発展候補を比較する

### 11.1 DF回路構造・基底変換：応用研究として正当

WP06/P-Aには、run-levelのbasis共有・support-union選択による実装利益が残る。interval DP独自の寄与が識別できなかったことは、そのrun-level利益を消さない。[R11]

新しい基底費用診断では、固定8列の加法的RZの84.83%、CXの68.97%がbasis/逆basis由来だった。しかし、部分別compileの加算は全体compileよりRZ8.49%大きく、全eventがorder0である。最適化後の厳密占有率、母平均、削減可能率、T費用ではない。[R12]

したがって、既知compilerをDF-RTEへ適用したときの期待wrapper資源と適用条件を研究する選択肢はある。ただし、今回の利用者の意図に従い、基底費用をreturn集約研究や時間相関研究へ強制的に結び付けない。

### 11.2 BF/BM：そのまま再開する優先度は低い

BFは同じfamily・同情報・同budgetでのfinite objective固有の価値が得られず、BMは同じcompact BCH式へ還元された。[R13, R14]

このnegativeを、誤差や費用を無視する理由にせず、共通の強いbaselineへ利用する。別の評価規則を片側だけへ与えたり、cacheを候補だけへ許したりして、同じ方法を新しい候補に戻さない。

### 11.3 DF whole補正：数学的成果は残すが直近は選ばない

条件付き高次精度と少数軌道の支持は数学的な資産である。しかし、generic wholeとlocal Pauliはfinite meanが違い、保存CDとEも別fixtureである。既存の局所T差だけでは同taskのPR総費用を決められない。[R15]

将来、構造を使う別の実装や新しい保証が具体化すれば検討できる。現在の全T空欄やr6 phase対応を埋めることを、②の直近の主作業にはしない。

### 11.4 N3・Track A・相関法

N3は別taskであり、同じ「保証」という語だけでFRやPRへ混合しない。Track Aの既存の資源評価も独立した系列として維持する。新しい主軸は、Track Aの結果を別名で再発表することではなく、finite-RTEの表現・実装自体を比較する問いである。

時間相関は今回主軸にしない。既知法との対応監査は既に行われている。②を広く見ることを、全ての過去候補を再開する口実にしない。

## 12. 次に必要な仕事：一つのCodex準備作業にまとめる

### 12.1 目的

**採択した主RQを、既存証拠と実装に結び付く有限な研究計画へ具体化する。** 新候補を次々に探す監査ではなく、既存構成を研究として発展させる技術準備である。

### 12.2 作業範囲

1. **同じもの／違うものの固定表**：R1、G10、SPのtarget、ensemble、proposal、normalizer情報、native価格、bias、confidenceを対応させる。各系列内で比較し、数値を混ぜない。
2. **既存数理の再利用と必要な追加解析**：一般構成・finite-bit証明を重複再監査しない。本レビュー(1)–(6)のような新しい整理は、式の独立展開で確認し、適用前提と既存helperを記録する。登録matrix・samplerは動かさない。
3. **利用可能な入力と実装の対応**：P5/local/full/ordinaryの生成入口、保存native列、control位相、取得コスト、参照専用列挙を区別する。CTSに使えるPauli情報を候補にも公開し、逆に存在しないoracleを無料にしない。
4. **最小の前向き比較案**：同じ有限targetの表現比較を基本に、必要な場合だけ合成選択との相互作用を加える。物理時間への接続で必要な誤差と反復条件を一つの計画にまとめる。
5. **限定した先行研究とのclaim照合**：一般IS・probabilistic synthesis・SCUを再発見しないため、今回目指す命題や適用知見だけを対応させる。広域の文献検索を何度も繰り返すこと自体を完了条件にしない。
6. **実行承認準備**：必要な新しい取得が何かを明記し、fixture、主要指標、比較自由度、誤差配分、資源上限、結果分類を実行前に固定する具体案を出す。科学実行の承認は別に受ける。

最初の比較は既存のsmall explicit providerを機構検査用に使える。ただし、すでに開封したp/x/providerはdevelopmentであり、独立検証とは呼ばない。将来の確認条件は結果前に決める。今回、新しいseed、分子、閾値、上限を勝手に追加しない。

### 12.3 完了条件

- 何を発展させ、何を主張し得る研究かが一文で説明できる。
- 既存証拠で言えることと、新たに確認しなければ言えないことが分かれている。
- 比較のtarget・情報・誤差・実装費用の対応が明示される。
- 次の限定実験が何を識別するかが分かる。positiveを必ず得ることは要求しない。
- 原結果・旧closureを維持し、別の仮説と実行契約であることを記録する。

**新しい数学primitiveが見つからないことだけを、今回の計画の自動STOP理由にしない。** 今回は理論的な成立条件・適用知見も研究目的として選んでいる。ただし、得ようとする内容が全て既知の再掲で、新しい知見が具体化できない場合は、その事実を報告する。

### 12.4 今回開始しないもの

登録比較計算、native compile、T合成、新しいsampling、分子生成、Hamiltonian/NPZの再計算、旧runnerの再実行、authorization/marker作成は、このレビュー承認には含まれない。

技術source準備・登録外テストの具体的な認可範囲は、そのCodex作業指示で分ける。旧契約が要求する独立sourceレビューを黙って消さない。一方、同じ採択方針のmetadata・保存形式・テスト修正のたびにGPTの詳細レビューへ戻す必要はない。

## 13. 結果に応じた進路

| 得られた内容 | 適切な次の判断 |
|---|---|
| 表現・実装の差を説明する一般条件が得られた | 理論・方法研究として発展。実験は主張を識別する範囲へ限定 |
| 新手順はないが、同taskの有意義な適用範囲が分かった | 応用・数値研究として位置付ける。別の新アルゴリズムと偽らない |
| 同一targetでSP等との相互作用が明確に残る | その機構へ対象を絞る。全pipelineの包括最適化に膨らませない |
| closed P5が最良でないが、別の既存表現が説明可能に有利 | 研究の失敗としない。特定法の勝利より設計条件を主成果にできる |
| 保存値に近い挙動だけで一般条件・独立知見が残らない | 範囲を縮小し、既存ノートで閉じることも検討 |
| 必要な保証や情報取得が現実的でない | その制約を明示。未取得を0やoracle freeへ置換しない |

新しい事実や重要な科学的設計変更が現れた時点でGPTレビューへ戻す。準備commitができたという事実だけで同じ内容を再レビューしない。

## 14. 論文化の着地点

現在の資料は、技術ノートとしてまとまった成果を既に持つ。一方、「新しい一般PR法が強い既知法より総資源で優れる」という主張は未確立である。

発展研究の中心は、例えば次のようにできる。

> **有限RTE表現の代数的簡約と実装コスト：局所生成・正規化情報・有限精度が資源優位の成立条件をどう変えるか。**

これは仮題であり、投稿先や採録可能性を保証しない。独立論文にするには、既知一般論に留まらない構成、定量的条件、または新しい適用知見が必要である。逆に、全分子で同じwinnerになる必要はなく、どこで利益が反転するかが予測可能であれば、それ自体が重要な結果になり得る。

FRを選ぶ場合は、別の位相保証論文の筋になる。両者が数学的に結び付く証拠がない段階で一つの論文へ詰め込まない。

## 15. 最終的な科学判断と不確実性

**②の研究設計を再開することを支持する。第一の方向は、既存return集約・再配分を、表現・生成・実装・十分測定予算の関係として発展させる研究である。**

この判断の確かさは、層ごとに異なる。

- 既存成果に理論・positive・機構差が残ることは、固定資料で確認できる。
- それらを同じtaskと比較条件で再構成する研究課題は具体化できる。
- どの条件で新しい性能優位が出るか、どこまで独立した新規性があるかは、まだ未確定である。
- 論文化の十分性は、本レビューだけでは確定しない。

従って、これは「休止から全実験再開」への変更ではない。**意味のある発展RQを選び、その技術具体化へ進む**という変更である。

次の実務はCodexにまとめて任せ、GPTは科学的結果・保証・主要claimの判断へ集中する。旧結果・個別closure・mandatory STOPと、科学実行の別途明示承認は維持する。

---

## 参考資料：内部の固定証拠

本文の[R]は以下の固定資料を指す。記録された過去の未承認・STOP表示は、その時点の履歴と区別して読む。

- **[R1] G6–G10完成ノートv1.0**：対象・正値性・生成・finite-bit・P5・全17行・下界・限界。commit `22a0d032717b4139cdfa806850643e93ba25624f`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md
- **[R2] G10 exact資源表**：取得入口の確認。巨大single JSONの全量独立再監査は未実施。commit `e65e3c0680fff4cfc4243ed5d6c87429cd5ce753`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json
- **[R3] Track B一時休止記録**：既存判断の対象と再検討境界。commit `792af43ef237c0c5cd9b42ec102a15dfc3c60be4`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/792af43ef237c0c5cd9b42ec102a15dfc3c60be4/docs/tracks/algorithm_codesign/track_b_research_pause_record_20261013.md
- **[R4] R1.5保存値帰属**：precision-envelope、費用factorization、angle keyの離散効果。commit `64bb39fdcc4928d0f026b63f874c8bf4be2e69ab`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md
- **[R5] G7 generator source**：canonical ordinary/partial/closed、local full、quartic helper。commit `b9ed01455351628c9073748f5ba5751aa794b789`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g7_generator.py
- **[R6] P5 source**：群・DP・actual proposal・reference-only enumeration。同commit。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g9_p5.py
- **[R7] SP-1結果**：48 mask rows、synthetic scope、累積利益・反転。commit `9d2bb1fa439748b02084bd9fbc9b10a705328f8a`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
- **[R8] SP-1契約**：joint-space lowering、fusion、PAI、signed moment/bias。科学source `0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md
- **[R9] G5固定辞書終了認証**：指定classの下界と同一CTS law。commit `23adf61f9342abff20f059be69089e89f66f7797`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md
- **[R10] FR-R1b結果**：同情報strict gainとdecision relevanceの区別。commit `64bb39fdcc4928d0f026b63f874c8bf4be2e69ab`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/fr_revision_nonuniform.md
- **[R11] P-A機構判別**：one-segment対照、旧run-level positiveを保持。同commit。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/research_direction_joint_synthesis_mechanism_validation.md
- **[R12] DF基底費用診断**：加法的割合・all-order0・共通global phase caveat。commit `403204edf22e5712a71ac9ac7eebdbcd18626421`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/docs/diagnostics/rte_basis_cost_attribution_20261013.md
- **[R13] BF closure**：限定family・budgetでのnegative。commit `d55de044b8e956ba6292209a94bb081014dfdae2`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
- **[R14] BM同値性監査**：compact BCHと同情報scoreの一致。同commit。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
- **[R15] DF補正closure**：数理成立・局所T比較・別fixtureのCD/E。commit `4192077723770f4846cc391ea0da96c1e115f2fa`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4192077723770f4846cc391ea0da96c1e115f2fa/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md
- **[R16] 研究概要・現状**：FR、P-D、cost model等の位置付け・資料発見用。単独の要約を新しい一次実験としない。commit `64bb39fdcc4928d0f026b63f874c8bf4be2e69ab`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/research/研究概要・現状.md

## 参考資料：外部一次文献と確認の限界

- **[P1] Cugini, Atif, Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms**, arXiv:2603.13495v1（2026-03-13）。§II/Theorem 1、§III/Theorem 2、合成channel・情報費用の条件を確認。cost×second-momentの一般原理とbias不変を参照する。有限confidenceを含む全目的の最適解として拡張しない。
  https://arxiv.org/html/2603.13495v1
- **[P2] Peetz, Smart, Narang, Quantum Simulation via Stochastic Combination of Unitaries**, arXiv:2407.21095v2。Taylor/SCU構成、Pauliの収集、層別samplingの関連箇所を確認。全CTSが必ず全語列挙を要求するとしない。repositoryのmatched CTSは一つの具体的特殊化である。
  https://arxiv.org/html/2407.21095v2
- **[P3] Koczor, Sparse Probabilistic Synthesis of Quantum Operations**, arXiv:2402.15550v2。確率合成の廉価回路・sampling負担・precisionの関連本文を確認。新しい組合せの数値優位や一般phase保証を、この文献から無条件に移さない。
  https://arxiv.org/html/2402.15550v2
- **[P4] Kiumi and Koczor, TE-PAI: Exact Time Evolution by Sampling Random Circuits**, arXiv:2410.16850v3。固定角度の乱択・sampling overhead・実装の関連本文を確認。元のfinite-RTEとtarget・control・資源会計を対応させず、優劣の結論へ使わない。
  https://arxiv.org/html/2410.16850v3

### 版・取得に関する注意

PR原論文arXiv:2503.05647のabstractページは改訂版を示したが、今回取得したversionless PDFは第1頁画像にv1表示があった。明示v2の全文取得はこの経路では成功していない。従って、取得PDFをv2本文と称して引用しない。本レビューの具体式と比較根拠は固定repositoryの[R1,R5,R6]を正本とした。この版差は、既存全PR定理の新しい再認証を今回はしていないという限界であり、資料の内容を推測で補っていない。

文献全体の網羅的priority調査や、全定理の独立再証明は実施していない。新しい研究で主要claimを定める際には、そのclaimに必要な版・命題の限定照合を行う。未取得の図表や本文を、取得済みとして扱わない。
