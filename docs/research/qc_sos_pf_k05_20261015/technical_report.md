# SOS-PF K0.5追加技術調査報告

2026-10-15着手、2026-10-16完了（JST）。文献・静的数学のみ。K0固定commit `58d8672702284d73321cc67a26c4640adaddde5f` を基準とする。[索引](README.md)／[原文対応表](literature_correspondence.md)。

**K0判定は `UNDETERMINED_DATA_REQUIRED` のまま。** 今回は研究採否を決めない。新しい生成規則や有用な分子用step保証を確立したとは主張しない。

## 1. K0から増えた証拠と変わらない点

| 区分 | K0.5での整理 | 証拠・限界 |
|---|---|---|
| 追加確認 | Kang–SuはSOSの時間発展だけでなく、正規化されたHamiltonianの誤差へ戻す逆向き評価も持つ | v3 §2.5 Lemma 7／Cor.8。一般energy保証への接続は§3で説明 |
| 追加確認 | 一般のground-energy task保証に、励起gap下界は必須ではない | operator誤差からQPE出力への安定性。特定branchのoverlap保証とは別 |
| 追加確認 | 有限時間二次PFの明示的な交換子対照が既にある | Childsほか、1912.08854v3 §5.1 Eq.150／Prop.16 Eq.152 |
| 追加具体化 | K0 Cauchy剰余の支配条件、入力情報の取得価格、固定次数と増大次数の違い | §4–6の記号的整理。数値評価なし |
| 未解消 | Kang–Su出版版とarXiv v3の完全な数学的差分 | 出版本文取得403。書誌・検索索引の確認を本文取得としない |
| 変更なし | 独立生成規則、利用可能な化学certificate、総費用改善が未確立 | K0の科学状態を更新する証拠は得ていない |

記号を固定する。対象sectorで `H'=H−LI=Σ_j F_j`、`0≤F_j≤f_j I`、`L=E_SOS`、`E_g=E_0−L`、`Λ=Σ_j f_j`。`E_g` はSOS下界からのoffsetである。`g_0=E_1−E_0` は同じsectorの励起gapであり、別の量である。Hamiltonianの近似factorization誤差は、PF誤差と別に元Hへ戻して加算する。

## 2. A：Kang–Suが実際に包含するもの

### 2.1 回路と生成の入出力

[Kang–Su v3](https://arxiv.org/html/2510.06316v3#S6) のCor.27は、**既に与えられた** `H=Σ_k B_k†B_k, B_k=Σ_j A_jk` を入力にする。各 `A_jk` のHamiltonian block encodingを足し、Lie group commutatorで積を作り、外側のPFで和を作る（Eqs.245–253）。従って、係数から良いSOS分割を発見する最適化法と、この分割から回路を生成する規則を混同しない。

Hermitianの場合はCor.29／Fig.13が平方のHamiltonian QSVT実装を与える。§6.3はDFTHC型のHermitian一体平方へ適用する。直接のnative回路で、軌道回転後の対角一体平方を位相gateへ展開するK0の実装と、入出力演算子が対応する。しかし内部回路・精度配分・access価格は同一ではない。方法名やancilla数の差だけで新規分割算法としない。

### 2.2 誤差・有限step・依存量

Cor.27の誤差はoperator normである。`n_K`（平方数）、`n_J`（各生成子の項数）、`α_∞=max_k(Σ_j||A_jk||)^2`、nested commutator量を使う。Eq.252は内側の和・積・外側の和の誤差を含むが、固定次数pでも定数を隠したO評価であり、最終query式は次数を上げた漸近評価である。これを、そのまま実数の最大stepを返す証明書とはしない。

一方、同論文に明示的な有限時間評価がないという理解も誤りである。Lemma 9 Eq.74にはHermitian `J,K` のgroup commutatorについて三次交換子normと `τ^3/2` を使う明示上界がある。これは**積を生成する内側subroutine**の保証であり、分子用の外側PF・回転・QSVT・読み出し全体を一つの数値certificateにしたものではない。

§6.3／Appendix C.2は粒子数ηを保つ一体係数行列のcommutatorからsector上界を構成する。固定次数なら係数から古典的に評価する道がある。`E_g`、`E_1−E_0`、ground期待値は§6の主要simulation boundの入力でない。従って、**SOS offsetが小さいほど有限hのground bias／overlapが鋭くなる、という保証まで§6からは確認できない**。

### 2.3 accessと費用の照合

| 項目 | 原文での契約 | 本候補へ移すときの条件 |
|---|---|---|
| 基本access | controlled Hamiltonian block encodingsとその逆・分数時間 | 直接指数化が可能か、Cor.22のfractional scaling価格が必要かを区別 |
| generic SOS | Cor.27は2 ancillaを報告 | 各queryをnative gateへ展開する価格は別に必要 |
| Hermitian平方・化学 | Cor.29／§6.3は1 ancillaを報告 | QSVT次数・各回転精度・制御回路費用を含める。低ancillaだけで低総費用としない |
| 化学の構造 | Eq.275はrank／copy／base／mode数とsector commutator量に依存 | 密な分割を低rankの実装価格で数えない |
| 粒子数 | すべての中間演算がsectorを保存 | 一体生成子に落ちない一般Gram-SOSへ自動転用しない |
| QPE | §1.6は利用可能性を述べ、§2.5は演算子誤差への変換を論じる | エネルギー窓・alias・読出し精度・成功確率の予算が別途必要 |

reported ancillaは論文のarithmetic contractでの値である。別方式との比較には、phase-estimation register、合成workspace、制御primitiveの数え方を揃える必要がある。今回それらのnative compileは実施していない。

### 2.4 原文の定数を数値certificateに使う前の注意

次の二点はK0の既知性確認より細かい照合である。論文全体の反証・erratum認定は行わない。

1. §6.3では一体項と平方内のidentity shiftを省いている。外側のscalar shiftはcommutatorを変えないが、`(Q+cI)^2=Q^2+2cQ+c^2I` の交差一体項は一般に他fragmentと可換ではない。完全な元Hのcertificateでは、この項を一体残余へ戻して、分割・norm・交換子を同じ規約で評価する必要がある。原文のscaling議論を完全Hamiltonianの厳密同値証拠としない。
2. Eq.271のnested anticommutatorからnormへ移る際には `||{A,B}||≤2||A||||B||` の係数を保持する必要がある。深さpでは、表示された `1/2` と合わせて `2^p` の単純上界係数が現れる。例えば `[A²,B²]={A,{B,[A,B]}}` の直接評価は `4||A||||B||||[A,B]||` を与える。Eq.274の表示量を係数1の厳密上界として採用することは、今回独立認証していない。root後の因子は高々2なので漸近scalingと有限定数の問題を分ける。出版版での記法／修正は未照合。数値certificateへ転用するなら、まずこの段階を再導出して安全な係数を固定する。

また `α_comm` の全次数supと、固定次数pでの有限個の係数行列計算を分ける。後者が多項式的でも、項列は概ね `n_K^(p+1)`、展開数もpとともに増える。無制限に次数を上げた最適query式に、同じ古典計算価格を付けることはできない。

### 2.5 v3と出版版

v3 PDFの表紙は2026-09-01。ACM／CrossrefはDOI [10.1145/3816436](https://doi.org/10.1145/3816436)、掲載巻号7(4)、1–86、online publication 2026-08-13を対応づける。出版版は日付上v3より前であり、arXiv登録上のjournal referenceだけで本文同一性を判断できない。

ACM PDF、fullHtml、epdfの取得は403。検索索引には同じSOS回路・化学応用の記述があるが、完全本文・全数式は取得できなかった。従って**v3から出版版への定理番号、even-QSVT、平方化、誤差定数の差分はUNVERIFIED**。v1とv3の差を出版版との差に代用しない。ここは今回の取得限界として残す。

## 3. B：励起gapなしで既に閉じる保証

### 3.1 逆logとWeylによるenergy評価

Kang–Su [§2.5 Lemma 7](https://arxiv.org/html/2510.06316v3#S2.SS5) は、Hermitian generatorの正しいlog branchで `||J||,||K||≤1−ξ<1` なら

\[
\|J-K\|\leq\xi^{-1}\|e^{-iJ}-e^{-iK}\|.
\tag{A}
\]

K0の `S_2(h)=e^{-ihH_eff(h)}` について、`h≠0`、`J=hH'`、`K=hH_eff` の両norm条件を**別途認証**できれば、simulation誤差εから `||H_eff−H'||≤ε/(ξ|h|)`。Weylの固有値摂動により最小固有値の差も同じ上界を持つ。励起gap下界は不要である。これは既知のoperator保証の直接帰結であり、新規性を主張しない。

ただし次を含意しない：厳密ground入力がH_effの最下固有状態に高確率で重なること、phase aliasの解消、任意に大きいhでの単一log branch。norm条件を仮定だけで済ませない。K0 Eq.13の領域なら `||H_eff||≤Λ+h²N_D+r_4` を使う粗い認証経路があるが、実用性は未評価である。

### 3.2 QPE task自体のgap-freeな安定性

厳密groundの量子利用という既存前提の下で、理想QPEはそのenergyを指定精度・成功確率で読む。各controlled evolution queryのoperator誤差をε_qとすれば、circuitのtelescopingにより最終state norm差は `Σ_q ε_q` 以下であり、測定分布の差もこの量で抑えられる。統計・窓・aliasの理想保証へこの失敗予算を加算できる。励起gapもH_effの個別固有vectorの追跡も必須でない。

これは**理想回路に近い出力分布を保証する経路**である。同じhのPF kernelを長時間反復する場合の誤差増幅を無料で取り除かない。`S_2(h)^r` のworst-case誤差はr倍まで蓄積し、QPE全体の予算へ戻す必要がある。新しいstate preparation研究へ移行するものでもない。

### 3.3 state residualと特定branchの違い

既知state上のresidualを `η_s=||(S_2(h)−e^{-ihE_g})|0>||` とする。`0<ω≤π` なら、spectral theoremから、対象位相から円周距離ω以上離れたPF位相への総重みは `η_s²/[4 sin²(ω/2)]` 以下。この式は既存のspectral分解の直接適用であり、η_sの上界だけを与えた場合も同じ評価が使える。

これは近傍の**位相集合**に重みがあることを示す。gapなしで一つのbranchへのoverlapを保証する式ではない。またη_s自体を取得可能な情報で鋭く上界する問題を残す。期待値だけの低energy条件を、低energy空間への完全support条件へ置き換えてはいけない。

### 3.4 既知の明示的な二次PF対照

K0と同じ順序の `S_2` に、[Childsほか §5.1 Eq.150／Prop.16](https://arxiv.org/html/1912.08854v3#S5.SS1) の二分割積分評価を順次適用する。`R_j=Σ_{k>j}F_k` として

\[
C_2=\sum_{j<M}\left\{\frac{\|[F_j,[F_j,R_j]]\|}{24}
+\frac{\|[R_j,[R_j,F_j]]\|}{12}\right\},
\qquad \|S_2(h)-e^{-ihH'}\|\leq |h|^3 C_2.
\tag{B}
\]

Hermitian fragmentなら実時間での明示評価であり、励起gap・SOS offset・Cauchyの複素log半径を必要としない。`r|h|=t` の反復誤差は `t h² C_2` 以下。sector保存ならsector normでよい。厳密normが難しい場合は、係数からの**認証上界**を代入する。

この対照はenergy固有の鋭いSOS-offset boundではないが、少なくとも既に閉じた有限時間保証である。K0のleading係数 `K_SOS E_g` が小さいことだけでは、この対照より認証stepが大きくなると結論できない。Eq.BからEq.AやQPEへ戻す際の仮定・価格も同じ予算で比較する。

### 3.5 低energy理論の適用範囲

[Mizuta–Kuwahara v1](https://arxiv.org/pdf/2504.20746v1) のTheorem 1は有限時間の低energy発展誤差を扱い、励起gapを仮定しない。Supplement Theorem S1 Eq.S5は、PSD・k-local・g_ext-extensiveな各H_qと `ΠH_qΠ≤ΔΠ` から `||Π[H_q,…,[H_1,H_0]]Π||≤q!(2k g_ext)^q Δ` を与える。ここで `g_ext` はextensivenessであり、励起gapではない。

密な一体平方を全supportの単一PSD項として扱う粗い特殊化はできるが、kやg_extの悪化を価格へ戻す必要がある。対角化できることだけで、全fragment共通の局所性や一定layer数を得たことにはならない。局所monomialへ展開したとき、各項のPSD性が失われれば、そのまま定理の入力にできない。

Hejazi低energy理論にもinteraction norm・mode degree・group数・漏洩の条件がある。Hejazi QPE論文とYi–Crossonの個別固有state解析にはgap条件が残る。これらを合わせても、非局所化学の有利な定数・漏洩・有限h・同情報費用が自動で閉じるわけではない。[具体的な式・仮定の対応](literature_correspondence.md)に記録した。

## 4. B：K0のCauchy＋Schur保証の入力と価格

K0 technical_audit §5.2 Eqs.13–17を変更せず参照する。非縮退groundに対し、`v=h²N_D+r_4`、`b=h²B_D√G+r_4`、認証 `G≥E_g` と `0<g≤g_0`、条件 `v<g/2` の下でbias／overlapを抑える。

| 必要情報 | 係数からの経路 | 未取得・費用上の条件 | 使用箇所 |
|---|---|---|---|
| 元H・sector・厳密再構成 | 係数一致・対称性ideal・PSD証拠の検査 | 有限精度の残差normと下界補正が必要。今回再認証なし | 全体の入力同一性 |
| PF-nativeなPSD分割とL | 既存DF、平方完成、Gram証拠から構成可能なclassがある | 独立した改善規則Rは未指定。一般SOSのLをnative化後へ流用しない | G・各F_j |
| `f_j,Λ,N_D,B_D,K_SOS` | 係数1-norm等の粗い上界は可能。固定次数の係数commutator上界も候補 | exact多体normは無料でない。鋭い値・bit精度・検証価格は未取得 | log半径・摂動量 |
| 上界U、`G=U−L` | 粗くは `G=Λ` も安全。検証可能なtrial determinant等からUを作る経路もある | 既存資料から有用なGを今回取得したとはしない。量子groundの存在と古典E_0の既知性は別 | offset型の係数bound |
| 励起gap下界g | 係数はgapを決めるが、安価な正の証拠が得られるとは限らない | 例えば認証 `L_1≤E_1`, `E_0≤U_0` から `L_1−U_0>0` が必要。ground用SOS下界LだけではL_1を与えない | Schur分母・特定branch overlap |
| dの補正値と精度 | 期待値推定または別の古典証拠 | RDM／量子測定・失敗確率・価格は未取得。今回は計算禁止 | scalar補正を使う場合のみ |
| energy窓・alias・query誤差配分 | 理論上はnorm上界と既知shiftを利用可能 | 必要な総時間・成功確率を含むtask contract未固定 | QPE/RPE結果の解釈 |
| step単価・認証古典費用 | 記号的gate数・rank・証拠サイズは会計可能 | 新分割の実際の値、合成価格、最終総費用は未評価 | 有用性・同情報比較 |

「係数のみから」は、情報として決まるという意味と、実用的な古典予算で鋭い証拠を得られるという意味を分ける。多体対角化すれば得られることを安価なcertificateの存在としない。正のgが未取得でもEq.Bのsimulation保証は使えるが、K0のSchur式へg=0を代入して救うことはできない。

## 5. B：明示剰余が支配する条件

`Λ=0` は全fragmentがゼロの厳密な例外として除く。`Λ>0`、`a=log(3/2)`、`c=log 2`、`x=|h|Λ/a<1` とする。K0の式を記号的に書き換えると

\[
r_4(h)=\frac{c\Lambda}{a}\frac{x^4}{1-x^2},\qquad
r_4(h)\sim\frac{c}{a^5}\Lambda^5 h^4\quad(x\to0).
\tag{C}
\]

これは新しい科学計算ではなく、既存式の依存関係の整理である。

- logの十分条件は `|h|<a/Λ`。Λが大きいと、実際のPFがよくても証明の窓は狭い。
- leading bound `h² K_SOS G` に対する比は、`K_SOS G>0` の場合 `r_4/(h²K_SOS G)=cΛ³ x²/[a³ K_SOS G(1−x²)]`。Gだけを小さくすると、この剰余が相対的に支配しやすい。
- 特定branchを保つには `h²N_D+r_4<g/2`。有用なoverlapには、さらに `(h²B_D√G+r_4)/(g−2v)` が小さくなければならない。孤立条件だけでは十分な成功確率を与えない。
- Schur補正は、hが十分小さければ `h^4 B_D²G/g` 型。gが小さい場合に「h⁴だから無視できる」とは言えない。
- exact frustration-free `E_g=0` ならground上のPFは厳密である（K0 §3.1）。一方r_4はゼロにならない。この差はCauchyのglobal norm評価の粗さを示し、実際の失敗を示さない。

uniform上界 `f_j≤f` では `Λ≤Mf`。K0のraw辞書から `N_D≤4Σ|κ| f³=M(M−1)(4M+1)f³/36` という粗い上界が得られる。従って、同じfでMを増やす粗い評価ではlog窓は `1/(Mf)`、孤立条件の二次項は概ね `h²M³f³` となる。これは最悪の符号無視評価であり、実際のcommutator・rank・fが分割とともに変わるため、M増加だけで優劣を決めない。

必要な進歩は、「式をもう一つ書くこと」だけではない。同情報で得られるcommutator・sector・低energy評価を用い、剰余・off-diagonal block・漏洩のいずれを改善できるか、残る定数と取得費用を閉じる必要がある。今回はその有用性の証拠を得ていない。

## 6. C：独立生成規則がまだ特定されていない理由

| 操作・構成 | 直接得られるもの | 独立した生成算法とする前の未閉鎖部分 |
|---|---|---|
| 既存DFのscalar PSD shift | 同じ分割回路のPSD表示と下界 | 既知scalar位相を戻せば実PFは同一。実bias改善機構にならない |
| 平方完成・一体残余のshift | 既存frame内の再表示 | 交差項・残余を戻した後のL、norm、回路単価まで必要。平方完成自体は既知 |
| SOS/BLISS・symmetry ideal | sector上で同じHとなる係数自由度、Gram証拠 | native平方classへの閉包と情報予算、既知法以上の選択規則が未指定 |
| Gram因子化・因子回転 | 同じPSD Gramから複数の平方表示 | 因子回転の存在は既知。どの回転をどの価格で選び、対照には何ができないか未提示 |
| 複素Gramの実化 | Hermitian成分と一体残余への分解経路 | rank増、残余下界、元sectorとの同値、回路費用の保存は別証明 |
| native一体平方の指数化 | 軌道回転と対角位相による回路 | 実装可能性はK0で閉じた部分。独立生成規則を選ぶ根拠ではない |
| 保証だけの改善 | 固定kernelの認証hを大きくできる可能性 | 実biasは変わらない。研究目的を分割生成から認証・選択へ変えるかはGPT判断 |

一般Gramのunitary因子回転は `ΣQ†Q` を保つ。Hermitian一体生成子の実orthogonal回転も `ΣQ²` を保ち得る。しかしfragment間commutator、各指数化費用、下界を含む選択問題は残る。同じ入力から既存手法が得られない出力であることは、今回まだ示していない。これは不可能性の証明ではない。

新しいRを主張するには少なくとも、入力classと許される情報、決定的な出力規則、元Hとの厳密同値／残差、PSD・sector・指数化、古典計算量と証拠精度、既知構成への還元関係、改善する量と失敗する条件を指定する必要がある。ground truthから良い回転を後選択する探索だけではこの契約にならない。

K0.5では新しい有望な独立生成則を特定していない。既知classの自由度を新方式として採択したり、未固定Rを数値探索で救う作業へ進んだりしない。

## 7. D：判断材料の統合と停止点

| 判断対象 | 静的に閉じた範囲 | まだ必要な証拠 |
|---|---|---|
| 新しい分割生成算法 | 既知DF/SOS/BLISS・平方完成・因子化との対応 | 独立したRと同情報baselineとの差。数値だけでは規則の欠落を埋めない |
| 有限step保証 | gap-free operator保証、逆log、QPE安定性、条件付きSchur | offsetを使う有用な定数、非局所化学の漏洩／剰余、必要情報の取得価格 |
| 実際の計算費用 | queryとnative gateの区別、記号的総会計 | 同一精度・成功確率での認証h、step単価、classical証拠費用。今回数値なし |
| 既存法への還元 | 一般SOS-PFはKang–Su、二次operator boundはChildsほか、固有値摂動はYi／Hejazi、低energy射影はMizuta等 | 統合した有用なchemistry certificateまでの包含は未確定。出版版完全照合も未了 |
| K1の情報価値 | Rを固定後なら、native化で利益が消えるか、剰余支配かを反証できる可能性 | K1の採否・目的・入力・gateはGPT／ユーザー判断。未認可草案は変更しない |
| 停止理由の区別 | 広い構成の既知性、保証の数学的未閉鎖、取得情報不足、費用改善機構未提示を分離 | いずれも今回の候補A正式終了判定に置き換えない |

今回の追加資料は判断材料を増やしたが、K0の判定を変える科学的根拠にはしていない。これ以上、未固定Rの周辺文献を広げても、版差分・鋭い係数・費用の未取得を直接解消しないため、限定調査を終了する。

**Mandatory STOP。K1・GPT詳細科学レビュー・研究採択・GitHub公開の開始を許可しない。Claude原本は未取得のまま。**
