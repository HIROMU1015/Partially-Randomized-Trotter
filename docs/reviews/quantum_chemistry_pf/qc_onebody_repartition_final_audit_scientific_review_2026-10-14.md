# 一体項再配分・1-RDM応答生成：最終方法差監査後の科学的レビュー

- 作成日：2026-10-14 JST
- 区分：ユーザーの明示的な開始承認に基づく重要科学レビュー。
- 対象：一体項再配分の第一候補について、継続・縮小・終了・第二候補への移行を判断する。
- 主要固定資料：`HIROMU1015/Partially-Randomized-Trotter`、`9e288089ca22e2729976ae75d096c4df00e49a7d`。
- 継承前提：厳密基底状態を利用可能とする。block-encoding・qubitizationを対象外とし、Track AのPR資源評価、Track BのPR内部改良を移管しない。
- 実行境界：本レビューでは新しい分子生成、固有値計算、数値期待値・RDM取得、最適化、PF性能評価、compile、保存値再集計、旧27検査の再実行を行っていない。

## 0. 結論

**「許可された最強compact BCH対照を上回る、独占的な応答生成高速化」という主張は採用しない。しかし、その監査だけを理由に、一体項再配分によるPF低コスト化の研究全体を終了する判断も採用しない。**

次は、同じ方法差監査を繰り返すのではなく、**一体項再配分が有限stepにおいて、同じ厳密状態情報を利用できる強い対照より有用なPFを作るかを、一つの限定pilotで判別するための準備**を優先する。研究テーマの正式採択・性能優位・論文化可能性の認定ではない。数値実行は別途明示承認を要する。

本判断は、未検証項目が残っているという理由だけの継続ではない。既に支持された、(i) Hamiltonianの厳密保存、(ii) 可解fragment classの維持、(iii) 固定先頭pairに対する1-RDM二次応答、(iv) 主要誤差を局所的に変える条件付きの明示的機構を根拠とする。一方、その機構の固有位相・overlap・回路費用への利益は未取得であり、主要誤差のスカラー補正という強い対照を越えられるかが最重要の判別事項となる。

| 判断対象 | 本レビューの判断 |
|---|---|
| 同じ恒等式・CAR・情報・cacheを許したB*との計算量差 | 差がないという限定監査を支持する |
| 閉式・特殊化の歴史的先行性 | 未確定。B*の構成だけから既報と認定しない |
| 数式と1-RDMによる応答生成の能力 | 保持する。能力の正しさと専有的優位を分ける |
| 独自の応答生成高速化だけを主題とする路線 | 採択しない。同じ監査を終了する |
| 再配分による有限stepの量子資源改善 | 未評価。一回の限定判別へ進む準備を推奨する |
| 第一候補のproduction開発・大規模探索 | 今は進めない |
| 第二候補のDF因子回転 | 保留。自動昇格させない |
| 新しい科学計算 | 未承認。今回のレビューは実行承認ではない |

## 1. 研究全体の目的と、直前の進行判断の修正

本研究の目的は、厳密基底状態を利用できるという指定条件の下で、量子化学Hamiltonianの基底エネルギー計算に使うPF・時間発展回路を低コスト化する構成法を検討することである。誤差評価器の古典計算量改善だけが目的ではない。

以前のC1では、固定macrocycleを増段しても、旧入力状態・有限catalog・grid・価格モデルのoracle診断に費用利益がなかった。その系列は終了済みであり再開しない。旧THRIFTでは理想PFのstep削減が追加実装費用を回収しなかった。その現templateも終了状態を維持する。今回の一体再配分は、元Hを変更せず、PF内のfragmentの割当を変更する別の構成である。

厳密基底状態の指定により、近似初期状態のfidelityを保証する研究や、基底状態準備アルゴリズムの開発に戻らない。元の厳密基底状態と近似PFの固有状態とのoverlapは別量として扱う。必要な1-RDM等を理想的な設計情報として用いる機構検証と、その情報取得回数まで含む運用資源評価を区別する。前者を開始するために後者の測定法を完全に完成させることは要求しない。

直前の進行回答では、「独立生成差がないため数値pilotを推奨しない」とした。この否定は強すぎたため、本レビューで修正する。**対照が同じ応答表を生成できることは、その表を用いた再配分によってPFの量子回路費用が改善しないことを意味しない。** また、同じ数学を用いたmatching constructionの存在は、特殊化の閉式が既に刊行済みだという証拠と同じではない。

Codexの停止自体は、当時のPhase A条件に忠実であり、問題ではない。修正対象は、私が設定した「生成器の専有的高速化がなければ物理的検証に進まない」という科学的な入口条件である。元監査・認可・STOPの履歴は上書きしない。

## 2. 今回取得・使用した資料

### 2.1 GitHub正本

| ID | 固定ref・path | 確認範囲 |
|---|---|---|
| G1 | `9e288089ca22e2729976ae75d096c4df00e49a7d` / `docs/research/qc_ground_pf_final_method_difference_audit_v1_20261014.md` | matching B*の操作列、入力・出力、計算量、限定性 |
| G2 | 同ref / `docs/research/qc_ground_pf_final_method_difference_gpt_handoff_v1_20261014.md` | 技術結論、未評価のfinite-hと研究判断の境界 |
| G3 | 同ref / `artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/literature_registry.json` | 原典・版・確認箇所・先行性未判定の記録 |
| G4 | 同ref / `artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/publication/receipt.json` | 冒頭の取得記録。payloadは`df11cbe1f5a7084065efbaf3addff59993666c4e` |
| G5 | `c0bb4350e599e557f7897e64222854d0fe45d620` / `docs/research/qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md` | 二分割・一般位置・先頭pairの数式、1-RDM表、既知法との関係 |
| G6 | 同ref / `artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/formal/word_identity_results.json` | 27件の保存PASSと、検査が保証しない項目 |
| G7 | 同ref / `docs/reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md` | 冒頭・研究前提・過去判断。全文は本会話添付の同名Markdownも読んだ |

G1のGit blob SHAは`00eb2c0f226d76754b94611206b93f3176b3b909`、G2は`1543ea5c68dc943771323c3ac8fd4138324764cc`。これはconnectorから受領したGit blob識別子であり、SHA256と同一視しない。G4に記載された全保護ファイルの照合を、今回自分で再実行したとは主張しない。

主要本文・handoff・保存検査にアクセスできたので、この科学判断に不可欠なrepository資料の未取得はない。全原rawデータ、全source、全過去branchを新たに独立監査したという意味ではない。

### 2.2 外部一次文献

| ID | 原典 | 今回の確認内容 |
|---|---|---|
| P1 | Choi, Loaiza, Izmaylov, *Fluid fermionic fragments for optimizing quantum measurements of electronic Hamiltonians in the variational quantum eigensolver*, Quantum 7, 889 (2023) | 一体項再配分、covarianceに基づく線形solve、Trotter誤差への応用提案 |
| P2 | Martínez-Martínez, Yen, Izmaylov, *Assessment of various Hamiltonian partitionings…*, arXiv:2210.10189 / Quantum 7, 1086 (2023) | §2.3、Eqs.37–42。既知の再配分と、spectral-range最適化の難しさから採用されたheuristic |
| P3 | Mehendale et al., *Estimating Trotter approximation errors to optimize Hamiltonian partitioning for lower eigenvalue errors*, Digital Discovery 4, 3540–3551 (2025) | 出版版II、Eq.8、誤差とQPE資源の接続。旧first-order版とは区別 |
| P4 | Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*, arXiv:2606.30738v1 (2026-06-29) | III.2とIV.1。grouped symmetric BCH、再帰、cache、期待値評価 |
| P5 | Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026-06-29) | II.Aの一体補正の配置、Gaussian/density回路、III.Bのfirst-pair norm階層。これらの部分だけを参照 |

P1–P5の一般理論から今回の閉式の初出性を推定しない。文献検索と関連節の照合では、この特定の1-RDM応答生成が明示された独立の先行手順を確認できなかった。しかし、不在証明ではなく、今回の探索で確認できた範囲にすぎない。

PDF本文は取得できた。webのページscreenshotは、P1/P2/P3/P5でInternal ErrorまたはCache missとなった。したがって図表の数値・図の細部を新たに読み取ったとは主張しない。本判断は取得した文章・式、GitHub資料、および以下に明示する解析的考察に基づく。P5のHTML v1は404だったがPDF本文を取得した。P2のversion指定PDFとP3のHTMLが取得できない経路もあったが、代替の公式PDF本文を取得した。これらを論文不存在や未公開と解釈しない。

## 3. `REDUCES_TO_KNOWN`の支持範囲

### 3.1 支持すること

G1のB*は、既知grouped BCHの再帰から変更前後の差を取り、変わらないtailを除き、ground交換子の期待値ゼロとone-body CARを使って、同じg,Qを生成する構成である。元提案のblack boxを呼ばない具体的なmatching procedureが記述されている。

この比較条件では、現法にだけ1-RDM・低メモリ・cacheの恩恵を認めて、B*を高body-rank・反復full-BCH計算へ縛ることはできない。両方とも供給されたa,γからdense arithmetic O(n^3)、追加保存O(n^2)、K個のdense候補の評価O(Kn^2)が可能であり、独占的な漸近高速化は支持されない。

O(n^3)は最適下界でもなく、実runtimeや定数係数のbenchmarkでもない。粒子数・spin block・疎方向等の節約も同条件で両法に許す。

### 3.2 支持しない一般化

G1の第3節は、B*で「前回の固定operator difference」を使用し、同じground恒等式とCAR簡約を許している。これは、**今回の研究で整理した特殊化を取り込める既知frameworkが存在すること**を示す。次は別の主張である。

- 特定の刊行物に、同じ局所応答式・入出力・再利用法が既に明記されていた。
- 既存の未変更softwareがその特殊化を自動生成する。
- 特殊化の証明・情報rankの整理には理論的記録価値がない。
- 同じ表から得られる再配分PFには、有限stepの費用利益がない。

これらは監査で認定されていない。前二者は先行性・実装機能の問題、後二者は科学的意義・性能の問題である。

公平なaccessとは、同じ入力情報や既知の合理的最適化を利用できることである。一方、提案で整理した核心的簡約までB*に組み込んだ比較は、**その簡約を用いる方式全体の価値や初出性を否定するための対照ではない。** この区別は、対照を弱くするためではなく、何を比較した実験かを誤らないために必要である。

### 3.3 記録の提案

旧statusは変更しない。その意味を新レビューに、次のように追補する。

> Matching specialization exists under the equal-identity/equal-access comparison. No exclusive response-generation complexity advantage is established. Historical priority and finite-step physical utility remain separate questions.

これは「新規である」という逆の認定でもない。基本変換は既知であり、独立論文としての非自明性が高いと評価したわけではない。

## 4. 一次文献から実際に分かること

P1のFFFは、元Hamiltonianを保った一体項の移動と線形方程式による測定最適化を与え、Trotter誤差への応用にも言及する。したがって移動自由度やlinear solveを本研究の発明とはしない。

P2は同じ再配分をEqs.37–39に明示し、Trotter誤差をspectral-range経由で最適化する難しさから、reflection/L1を用いるheuristicへ進んでいる。今回のstate-specific leading responseとその目的は異なるが、目的を替えるだけで新算法になるわけではない。

P3は対称PFの有効Hamiltonianと基底固有値摂動を利用したpartition評価を行う。厳密状態の使用やleading期待値自体に新規性を置かない。

P4は二次PFの三重交換子をO(L)個のgroup、四次PFをO(L^2)個のgroupへ整理する。Lはfragment数であり、1-RDM行列の軌道数nとは異なる。P4のO(L) groupsと本応答のO(n^3) matrix arithmeticを、そのまま比にしてspeedupとはしない。P4の再帰・groupingは今回の特殊化の基礎でもある。

P5は一体補正を一体fragmentへ集めるか、density fragmentに残すかが回路構造へ影響することを具体化している。このため「density frameが同じだから総gate費用は変わらない」とは扱わない。

本研究に残る可能性は、これら既知原理を否定して新しい名前を付けることではなく、**特定classで低情報rankの応答を閉じ、使える再配分手順と有限stepの有用性を一体として示すこと**である。まだ完成した寄与ではない。

## 5. 数学的に保持する内容と、限定された追加考察

### 5.1 定義

H=A+B+T、Aはone-body、Bは最初のdensity fragment、Tは以後のfragmentの和。固定S2でAが最外側、Bがその直内側にあるとする。XはBのdensity frameのone-body diagonalで、[X,B]=0。

A→A+X、B→B−XによりHを厳密に保つ。scalar・normal-order一体補正・sectorを含めた元Hに対する等式である。もともとのDF切断誤差を消したという意味ではない。

S_x(h)=exp[-ih(H+h^2 D_x+O(h^4))]と書く。主要係数d(x)=〈D_x〉0についてG5の式は

\[
D_X-D_0=\frac1{12}[A,[A,X]]+
\frac18[H,[X,A]]+\frac1{24}[X,[A,X]],
\]
\[
\Delta d(X)=\frac1{12}\langle[A,[A,X]]\rangle_0+
\frac1{24}\langle[X,[A,X]]\rangle_0.
\]

後者はexact eigenstate上での期待値の等式で、operatorから[H,[X,A]]自体が消えるわけではない。Tの明示期待値は式に残らないが、γは全Hの状態に由来し、物理的にtailが無関係なのでもない。

### 5.2 本レビューでの短い代数照合

二分割Strangのleading項を

\[
\mathcal C(P,R)=\tfrac1{24}[P,[P,R]]+\tfrac1{12}[R,[P,R]]
\]

とする。変更されるのは外側二項

\[
\mathcal C(A,B+T)+\mathcal C(B,T)
\]

のみである。外側はH固定として

\[
\mathcal C(A,H-A)=\tfrac1{12}[H,[A,H]]-\tfrac1{24}[A,[A,H]].
\]

差を展開し、内側の変化は[X,B]=0を用いて

\[
-\tfrac1{12}[B+T,[X,T]]+\tfrac1{24}[X,[X,T]]
\]

となる。外側の二次[X,[X,T]]項と相殺し、Jacobi恒等式で整理すると5.1のoperator differenceになる。この手計算の確認は、旧27件のプログラム検査を再実行したものではない。

### 5.3 1-RDM応答と限界

A=dΓ(a)、X=dΓ(x)、x diagonal、γ_rs=〈a_s†a_r〉0なら

\[
d(\mathbf x)=d_0+\mathbf g^T\mathbf x+\mathbf x^TQ\mathbf x,
\]
\[
g_p=\tfrac1{12}\operatorname{tr}\{\gamma[a,[a,n_p]]\},\qquad
Q_{pq}=\tfrac1{24}\operatorname{tr}\{\gamma[n_p,[a,n_q]]\}.
\]

one-body CAR閉包から全て1-RDM収縮となる。g,Qは変化分を与え、絶対d0は別取得である。一般DFでQはPSDとは限らず、全粒子数方向はnullとなる。二分割のPSD定理を一般DFのglobal convexityへ移さない。

実係数xについて

\[
\mathbf x^TQ\mathbf x=-\frac1{12}\sum_{r<s}
\operatorname{Re}(a_{rs}\gamma_{sr})(x_r-x_s)^2
\]

と書ける。符号付き重みのLaplacianであり、個々のweightの符号は仮定しない。全Nの移動が誤差を変えないこともこの式から整合する。

### 5.4 局所的な改善方向が存在する条件

以下は本レビューによる、既存二次応答からの解析的帰結であり、新しい分子結果・新規定理の初出認定ではない。

gauge・symmetryを守る許容線形部分空間への直交射影をPとし、v=P gとする。d0≠0、v≠0で、その方向の原点近傍が許容されると仮定する。

\[
\mathbf x(t)=-\operatorname{sgn}(d_0)t\mathbf v
\]

に沿って

\[
\left.\frac{d}{dt}|d(\mathbf x(t))|\right|_{t=0}
=-\|\mathbf v\|^2<0.
\]

したがって、この条件下では主要誤差の絶対値を少し減らせる。全入力で条件が成立する保証、有限stepの勝利、最良PFへの優位ではない。v=0、d0=0、許容境界、取得intervalが符号を定めない場合は別扱いである。この条件は旧C1の異符号kernelの存在を要求しない。

### 5.5 情報誤差を後から評価できる形

O_x=(1/12)[a,[a,x]]+(1/24)[x,[a,x]]とおく。仮に古典γの取得誤差がSchatten trace normでη以下なら

\[
|\operatorname{tr}\{(\widehat\gamma-\gamma)O_x\}|
\le\eta\|O_x\|.
\]

粗いが明示的な条件として

\[
\|O_x\|\le\tfrac13\|a\|^2\|x\|+
\tfrac16\|a\|\|x\|^2
\]

が得られる。これらはnorm不等式による条件付き会計であり、今回ηや測定費用を取得していない。厳密量子状態が供給されることと、古典γの有限shot推定が誤差ゼロであることを混同しない。同時に、初期の理想情報pilotをこれで禁止することもしない。

## 6. leading改善より重要な対照：scalar-corrected S2

### 6.1 なぜ必要か

d0が設計側に取得可能なら、元PFの推定値からh^2 d0を引く後処理が可能である。これには新しい多数体補正回路を必要としない。d0の取得誤差・branch整合性は必要だが、未知E0を入力しているわけではない。

再配分法だけに正確なd0を渡し、この安い対照を禁止すれば、主要誤差低減の価値を過大評価する。そのため、元S2と再配分S2のそれぞれに同じ種類のscalar correctionを許す。

### 6.2 有限stepとoverlapの解析

以下は固定分解・同じlog branch・孤立した非縮退基底branchに関する標準摂動論からの整理である。この条件をユーザーの全対象への追加制約として扱わない。縮退がある場合はP0D_xP0のスペクトルと入力の関係を別に扱う。

\[
H_{\mathrm{eff},x}(h)=H+h^2D_x+h^4F_x+O(h^6),
\]
\[
E_{0,x}(h)=E_0+h^2d_x+h^4c_x+O(h^6),
\]
\[
c_x=\langle F_x\rangle_0-
\sum_{n>0}\frac{|\langle n|D_x|0\rangle|^2}{E_n-E_0}.
\]

scalar correction後はh^4 c_xが最初に残る。さらに

\[
1-|\langle\psi_{0,x}(h)|\psi_0\rangle|^2
=h^4\sum_{n>0}\frac{|\langle n|D_x|0\rangle|^2}{(E_n-E_0)^2}
+O(h^6).
\]

ここに励起スペクトルを記したのは機構の説明のためであり、全スペクトルを無料のgenerator入力として要求するものではない。未来の小系pilotではtruth評価側でfinite-hのenergy・overlapを照合し、候補生成側には漏らさない。

**1-RDMが与えるのはd_xの変化であり、一般にc_xやoverlapではない。** 特に[H,[X,A]]は対角期待値ゼロでも非対角行列要素があり、有限stepのoverlapや高次誤差へ関わる。その項を「無関係」として捨てたgate性能評価をしてはいけない。

## 7. 科学的に残る可能性と、現状の障害

### 支持された有望要素

- 元Hを厳密に保ち、active-space削除・近似Hによる新しいmodel biasを入れない。
- density frameを維持でき、one-body Gaussianとdiagonal phaseという既存の回路classで実装できる。
- 先頭pairの主要誤差応答を、全高body-rank演算子の反復取得ではなく1-RDM表へ閉じられる。
- 条件付きで主要誤差を減らす方向を、oracleのfinite-h探索なしに生成できる。

これらは能力であり、B*にも取り込める。だから能力自体が無意味になるわけではないが、専有的速度差と表示してはいけない。

### 未解決の障害

- d0の取得費用は一般に残る。
- 元one-bodyが疎・対角なら、Xの追加でGaussian変換が高価になる場合がある。
- 合法なbasis fusion、linear density phase、scalar/control、回転合成費用は数え直す必要がある。
- 小さいd_xはh^4残差やoverlapの改善を保証しない。
- 完全な既知再配分法を同じ目的と情報で最適化すると、同じPFへ到達し得る。
- 一件の有利な分子結果だけでは、独立新規性・一般移送・論文化十分性にならない。

## 8. 比較対象は二層に分ける

### 8.1 方法差・生成器の比較

刊行済みの手順、現在の実装、今回の特殊化を加えたB*を区別する。B*に対する理論的計算量差はないと記録し、再度それを証明するbenchmarkを主目的にしない。implementationの一致確認は正しさの検査であり、優位性の実験ではない。

既存のgeneric BCH実装が遅くても、未最適化実装との比較だけで普遍的な計算量改善をclaimしない。一方、実装を拡張して同じ閉式に到達できることを、閉式の刊行済みの証拠にも変換しない。

### 8.2 生成されたPFの物理・量子資源比較

必要な核対照は、元S2と同費用細分化、再配分S2、各々のscalar-corrected版、S4、既知の外挿、適用可能な既知再配分である。大きな全手法catalogを義務化せず、比較対象の役割と対象classを固定する。

FFFはmeasurement目的が違うため、元FFFのvariance最小化結果とPF-error最小化を混同しない。L1/reflection等の既知再配分は、許されるX自由度と利用情報を明示して比較する。S4に再配分を許す場合も、S2専用response式をS4の誤差式として使わない。

## 9. 情報・量子・古典費用の公正な会計

**同じ情報を取得できる権利と、全方式に同じ情報取得費用を必ず負担させることは違う。**

再配分にγとd0が必要で、scalar-corrected元S2にはd0だけが必要なら、後者へ不要なγ取得を強制してはいけない。同じ共通資源単位で表現できる場合の例は

\[
C_{\mathrm{repart}}=C_\gamma+C_{d_0}+Q_{\mathrm{repart}},\qquad
C_{\mathrm{corr}}=C_{d_0}+Q_{\mathrm{corr}}.
\]

古典時間と量子gateを足す換算が未固定なら別資源として出す。外部からγが既に供給される理想比較なら、その条件を宣言し、後からγの取得費用も含む一般優位へ格上げしない。

leading項が支配し一step単価が固定に近い場合、同じsimulation時間の費用比較にはc_step sqrt(|d|)が現れる。scalar correction後にh^4係数が支配するなら、対応するproxyはc_step |c_x|^(1/4)となる。これらは説明用で、整数step・合成・overlap・confidence・有限h全体の費用を置き換えない。

量子回路部に利益が得られた場合、その差額を「追加情報取得へ使える予算」として示せる。差がない場合は、情報取得法を高価に実装しても現モデルの不利を救えない。ただし設計上界・価格仮説による負の差を、全最適回路への不能下界にはしない。

この研究で行うのは新しい再分割法の構成・component資源の評価であり、Track AのPR全round・normalization・rare-tail実験を移管することではない。

## 10. 他の選択肢との比較

| 選択肢 | 利点 | 障害 | 判断 |
|---|---|---|---|
| 第一候補を今すぐ全面終了 | 追加費用なし | 生成器同等と物理効果を混同した終了になり得る。現候補はfinite-h未評価 | 今回は選ばない |
| 1-RDM特殊化を技術noteだけで固定 | 正確な成果を残せる | 研究目的の量子費用改善は未検討のまま | 最終的な代替着地点として保持 |
| 同じ方法差監査をもう一回 | 限定scopeの再確認 | 同じB*と同じ式で結論は変わらない | 行わない |
| 限定finite-h・強対照比較を一回準備 | 未取得の物理的価値を直接判別できる | 新規性・有利さは未保証 | 優先する |
| DF因子回転へ直ちに移行 | 別のH保存自由度 | 既知自由度、不変な目的量、frame変化の費用。現時点でより有望という新証拠なし | 第二候補のまま保留 |
| 新しい時間発展公式を全面探索 | 新原理の可能性 | 具体的な構成・低費用accessを欠いたまま候補数を増やす危険 | 現在は開始しない |

DF因子回転の既知不変性は前回レビューの適用範囲を保持する。個々の平方fragmentを順序付きで指数化するPFまで不変ではない。ただし、その事実だけで第一候補より高い優先度を与える根拠にはしない。

## 11. 次のCodex作業：一つの限定検証を実行可能にする準備

本レビューに基づく次作業は、**結論追補・限定selector/collector・強対照・実行前契約・独立checker・人工fixture・公開を一つの準備単位としてまとめること**である。今回の承認は数値実行ではない。

### 11.1 結論の追補

旧監査を上書きせず、独占的応答生成speedupを採用しないこと、歴史的先行性は未確定であること、次の検証目的は有限stepのPF有用性であることを追補する。B*との一致をもう一度「負け」と数えない。

### 11.2 対象の最小化

前回draftの「少なくとも二つの非可換density fragmentを持つ、既存の固定Hermitian分子DF Hamiltonian一件」を再利用候補とする。これは新たに特定分子・距離・rankを指定するものではない。既存packのidentityと回路の対応を確認して、実行前に固定する。

更新は最外側Aと最初のdensity B、XはB frameにdiagonalという現在のclassに限定する。frame順序・fragment選択を結果に応じて変えず、非隣接・同時多fragment・DF因子回転へ拡張しない。

### 11.3 結果前に固定する科学条件

- 元H・scalar・normal-order・DF切断policy・粒子/必要spin sector。
- 厳密基底状態の供給方法と、generatorへ渡すγ、d0またはinterval。E0とfinite-h正解はtruth側に隔離。
- 許容Xのgauge・symmetry・bounds・方向数、候補生成・tie・zero fallback。無制限の数値最適化を行わない。
- h grid、energy bias目標、branch/overlap扱い、合成precision、共通の費用モデル。
- 元S2、再配分S2、各scalar-corrected版、S4等の対照と、それぞれの情報取得費用。
- 小効果・数値境界・missing・branch不確定時の扱い。旧C1の5%gateを無断流用しない。
- source/input/runtime identity、各call上限、memory/output/log、one-shot/retry/STOP。

許容幅やeffect thresholdはCodexが根拠を持って計画へ具体化する。主要な科学比較・実行範囲の変更が必要になった場合だけ判断を戻し、通常の実装修正・人工fixtureのたびに詳細GPTレビューを要求しない。

### 11.4 最初に判別する三つの問い

1. **応答と実際のfermionic実装が対応するか。** 形式27検査は保持し、数値取得後はone-body表と独立なdirect contraction、H保存、sector、phase規約を照合する。これは新しい数値検査なので別認可後である。
2. **候補生成がfinite-hで有効か。** γ/d0による選択をsealし、同じ候補を有限hの基底branch・overlapで評価する。結果を見てXを選び直すoracle評価は運用成功として扱わない。
3. **強い対照に対する費用差が残るか。** 特にscalar-corrected元S2とS4を戻す。回路の変化と必要step・overlapを分離し、positiveなら情報費用の余地も提示する。

理想段階で差がない場合に無条件でcompileを大量実行しない。段階的な打切りを用いるなら、分岐条件と各段階の上限を最初の実行契約に固定する。既存実行認可は流用しない。

## 12. 判定とSTOP

| 結果 | 科学的な扱い |
|---|---|
| responseが変わらない、または許容領域で主要改善がない | その固定class/inputでの制約として保存。新しい入力探索へ自動延長しない |
| uncorrected S2だけ改善し、scalar correction/S4を戻すと利益なし | 主要誤差最適化の機構は残るが、現方針の低費用算法としては区切る根拠になる |
| finite-hで利益があるが追加回路/情報費用を回収しない | quantum/情報の交換として保存。無料情報の一般優位を主張しない |
| 同精度の強対照に対して意味のある差が残る | その限定scopeの結果として、前向き移送と論文claimの次レビューへ進む。直ちに新規性認定しない |
| branch、roundoff、missing等で比較不能 | UNDETERMINED。閾値変更・retry・別namespaceによる救済をしない |

これは一般PFや全分子の不能性を判定する実験ではない。否定結果でも、同じ第一候補の救済scanを無期限に継続しない。肯定結果でも、初出性と運用可能性・一般性を別に評価する。

## 13. 論文化の着地点

現時点で認定できるのは、固定条件の数学的構成と、同条件matching implementationの存在までである。独立論文に十分だとは認定しない。

あり得る寄与は、第一pairに対するstate-specific応答の情報rankを閉じる理論、その能力を使う具体的なrepartition手順、適切なPF対照との有限step・回路費用の関係を一体として示すことである。既知frameworkへ組み込めることと、新しい結果として記録する価値がないことは同じではない。

ただし、基本自由度・二次最小化・既知BCH評価の名前替えだけなら新算法としない。閉式も既報と確認された場合は、その部分を先行結果として帰属する。性能が一件改善しただけなら、それを一般算法や量子優位にしない。利益が残らなければ、正確な技術noteと限定された否定結果として固定する選択がある。

新規性を完全に証明してから一切の性能検証を許す、という入口条件は採用しない。限定実験と文献照合を適切に組み合わせて、寄与が成立するか判別する。ただし新規性未確認を隠したり、既知法に不利な制約を付けたりしない。

## 14. 最終的な担当と保全

**次の担当：Codex。** 一回のfinite-step判別に向けた準備・実装・人工検査・実行前契約・GitHub公開をまとめる。通常のsource完成だけで同じ詳細レビューを要求しない。実データのγ/d0/固有値/性能/compileの新規取得には、別途の明示実行承認が必要である。

新しい重要結果が得られた後に、ユーザーの別開始承認を受けてGPTが科学的継続判断を行う。今回のレビューでC1–C4を再開せず、N1 STOP、N2 HOLD、N3 CLOSED、THRIFT現template CLOSED、Track A/Bのsource・結果・認可・STOP、既存global manifestのFAILを変更しない。

GitHubへ本レビューを無断pushしていない。本ファイルは新しいGPT科学レビュー記録であり、旧技術監査の原文を改変するものではない。

## 15. 参照リンク

### GitHub

- G1: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/docs/research/qc_ground_pf_final_method_difference_audit_v1_20261014.md
- G2: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/docs/research/qc_ground_pf_final_method_difference_gpt_handoff_v1_20261014.md
- G3: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/literature_registry.json
- G4: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/publication/receipt.json
- G5: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/docs/research/qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md
- G6: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/formal/word_identity_results.json
- G7: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/docs/reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md

### 一次文献

- P1: https://quantum-journal.org/papers/q-2023-01-03-889/pdf/
- P2: https://arxiv.org/pdf/2210.10189
- P3: https://doi.org/10.1039/d5dd00185d
- P3 PDF: https://pubs.rsc.org/ja-jp/content/articlepdf/2025/dd/d5dd00185d?page=search
- P4: https://arxiv.org/html/2606.30738v1
- P5: https://arxiv.org/pdf/2606.30741

以上のリンクと固定refを使って根拠を追跡できる。今回のレビューは先行研究の全証明を査読したものではなく、重要な比較箇所の照合と、明示された独自の解析的考察・研究投資判断である。
