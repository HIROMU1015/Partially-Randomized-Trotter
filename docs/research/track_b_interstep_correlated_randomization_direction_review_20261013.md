# Track B：ステップ間の乱択相関を用いるPR改善の研究方針レビュー

- 作成日：2026年10月13日（JST）。
- 開始承認：利用者の「その方針でレビューを開始して」。
- 対象：Partially Randomized Trotter（PR）の内部アルゴリズム。時間方向の相関を出発点とするが、採択・拡張を前提にしない。
- 正式な運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`。
- 参照repository：`HIROMU1015/Partially-Randomized-Trotter`。読取り対象の固定commitは第1節・文献表に示す。
- 実施状態：**基礎的な数理・既知構成との対応・次の作業方針は検討済み。新しい相関付きPR手法や数値pilotは採択しない。PRHS本文との完全な対応は、本文未取得のため保留。**
- 次の担当：Codexによる、近接する一次文献の版確認・式・保証・資源会計と既存PRの対応確認。新しい計算系列の立上げではない。
- 実行状態：新しいHamiltonian・行列・回路・trajectory生成、compile、T合成、repository tests、GitHub書込み、authorization・marker作成は0。以下の式は解析的な展開・恒等式であり、新規数値実験ではない。
- 既存generic whole補正・Gaussian候補のclosure、Track Bの自動的な新規実験立上げの休止、mandatory STOPは維持。Track A・N3へ介入しない。

## 0. 結論

**ステップ間相関を、現時点で次の新規PRアルゴリズムの主線として採択しない。一方、相関が不利・無価値・不可能だとも結論しない。**

今回の大きな更新は、構想に近い一次文献が具体的に見つかったことである。qSHIFTは履歴依存のjoint quasiprobability、PRHSは相関した時間スライスによる高次化を扱う。前者の関連本文は取得でき、後者は公式arXivの書誌・abstractまで確認できた。したがって「このrepositoryでは未検証」と「研究分野で未開拓」は明確に区別する。[P1–P3]

数理的には、次の三つを分ける必要がある。

1. **元のPR平均を変えず、経路の出現確率だけを変える。** 全経路importance samplingで可能だが、likelihood ratioの二次momentとrange、既知の費用最適化との重複が課題。
2. **平均も変え、高次の誤差相殺を狙う。** 可能な構成はある。ただし単純な二ステップ構成でも既知quasiprobabilityと対応し、負係数の正規化・多blockでのweight負担を戻す必要がある。
3. **別の測定・推定器を相関させる。** 同じPR内部の時間相関とは別の課題。古典乱数の共有だけで量子shot雑音が相殺するわけではない。

今回得た解析例は、単純な持続型相関の注意点と、相関一般を排除してはいけない理由を同時に示す。持続型Markov lawをそのまま使えば元のoperator meanが変わる。全経路補正をすれば、二状態の簡単な例でweightの二次momentが指数的に増える。一方、経路全体で元分布を混ぜる既知のdefensive mixtureならweightを有界にできる。従って、単一例から「相関は必ず指数的に悪化する」というno-goを作らない。

**直近では新samplerやcompilerを作るより、PRHS・qSHIFT・全経路importance samplingを現在のfinite-RTE PRと同じ比較対象へ対応させる限定技術確認が適切である。** 比較はqDRIFTより高次になったかだけでなく、現在のRTEの有限mean、normalization、coherent signal、実回路費用、同じ最終精度・confidenceまで戻す。

DF基底変換費用に関する以前の質問は独立した事実確認であり、このレビューの採択理由・主要改善目的には固定していない。ここでは回路費用を一般の実装依存量として扱う。

## 1. 使用資料・取得状況・レビューの境界

### 1.1 内部資料

| ID | 固定資料 | 今回確認した内容 |
|---|---|---|
| R1 | `321cd8c3066194fe98741cbebbcdba1515d1deaa` / `src/trotterlib/rte.py`、1594–1750行 | eventの次数、product→rotationの順序、phase、normalization、通常の独立抽選経路 |
| R2 | 同commit / `docs/hierarchical_cost_validation.md`、1–62行 | fresh IID trajectory下のpair/triple・反復回路費用予測。相関lawへの変更実験ではない |
| R3 | `64bb39fdcc4928d0f026b63f874c8bf4be2e69ab` / `docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md` | 現Gaussian候補のclosureと、性能未確定、他系列への不介入 |
| R4 | 添付 `track_b_overall_direction_and_research_pause_review_20261013.md` | 直前の全体休止判断と、既知部品の組合せを一律に研究対象外にしない方針 |

R1–R3はGitHub connectorで固定commitから実際に取得した。R4は会話に添付されたレビュー原本を使用し、このレビューでは新たなGitHub公開先を確認したとはしない。古い候補の全raw・全branch・全試験を再監査していない。

現在確認できたR1の通常samplerは、eventごとにorderを引き、指定分布からcomponent列を抽選する。これは履歴に応じて遷移kernelを変える実装ではない。ただし一つのsource経路と既存記録から、全branch・全過去案に相関実装が絶対に存在しないという不存在証明は行わない。

### 1.2 外部一次資料

| ID | 文献・版 | 取得の範囲・本レビューで使う内容 |
|---|---|---|
| P1 | Dubus, Cunningham, Roland, *New random compiler for Hamiltonians via Markov Chains*, arXiv:2411.06485v2 | HTML本文。連続時間Markov切替、channelを対象とする解析、特にbalanced schemeの範囲 |
| P2 | Lee, Choi, *qSHIFT: An Adaptive Sampling Protocol for Higher-Order Quantum Simulation*, arXiv:2604.26263v1、2026-04-29 | HTML本文・Appendix C.2。履歴依存quasiprobability、高次係数条件、負係数、古典連立方程式。全定理・数値実験の独立認証ではない |
| P3 | Cugini, *Pathwise Random Hamiltonian Simulation*, arXiv:2608.29756、2026-08-30 | **公式arXivの書誌・abstractのみ**。相関時間スライスとquasiprobabilityによる高次化の存在を確認。完全な式・sampling algorithm・誤差定理は未確認 |
| P4 | Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 | HTML本文§II、Theorem 1。全回路parameter vectorに対する費用×二次moment最適化 |
| P5 | Mohammadipour, Li, *MLMC-qDRIFT*, arXiv:2604.26865v1 | HTML本文§2、特に§2.3。fine/coarse couplingと実測shot雑音の区別 |
| P6 | Faehrmann et al., *Randomizing multi-product formulas for Hamiltonian simulation*, Quantum 6, 806 (2022) | 公式journalページ。signed/multiproduct型構成との近接性。全証明を新たに監査したものではない |
| P7 | Hesterberg, *Weighted Average Importance Sampling and Defensive Mixture Distributions*, Technometrics 37 (1995) | 公式出版社abstract。元分布を混合に含めてweightを制限する既知性。下記のbound自体は直接導出 |

PRHSについてabs/html/pdfの版付き・版なし経路、export系などを試したが本文を取得できなかった。web側ではcache/取得エラー、container側ではDNS解決失敗であり、論文非公開・未投稿・権限不足とは断定しない。他サイトの自動要約やAI批評は、定理・性能・誤りの根拠に採用しない。

PR原論文arXiv:2503.05647のversionless PDFも補助的に確認したが、解析テキストとスクリーンショットの版・日付・式番号に不整合があった。そのため、今回の版固定された数値比較・式番号照合には使用しない。既存RTEの具体的baselineはR1に置く。PDFの図表から新しい性能値を転記していない。

### 1.3 未取得部分が制限する判断

本文未取得のPRHSと、将来の候補が厳密に同じか、違うか、どちらが総費用で良いかは判断しない。一方、相関という自由度の意味、単純な候補の解析、実行より先に確認すべき事柄は、取得済み資料と以下の明示した数理条件から判断できる。

従って本書は、PRHSまで含む全手法比較を完了した最終新規性判定ではない。**研究方向の基礎レビューを実施し、新手法採択・新計算は保留した記録**である。

## 2. 「ステップ間相関」で何を変えるのか

### 2.1 相関の階層

次は同じものではない。

- 一つのfinite-RTE event内のcomponent列の依存性。
- 一つのランダムblock内の、複数short RTE occurrenceのjoint law。
- 決定論的部分を挟む別のmacro time step同士のjoint law。
- Hadamard回路の二つの枝での乱数共有。
- 異なる量子shot間、またはfine/coarse推定器間の共通乱数。

今回の中心は二番目・三番目である。一番目は以前のreturn集約等と接点があるが、その成功を時間方向の新lawの検証済み証拠にはしない。五番目は測定器・推定器の別設計を要することがある。

### 2.2 元のtargetを定義する

経路を\(\omega=(e_1,\ldots,e_m)\)、元のjoint lawを\(P(\omega)\)とする。独立な元設計なら\(P=\prod_t p_t(e_t)\)だが、以下のchange-of-measure自体は任意の既知Pに適用できる。

\(V_\omega\)は必要な決定論的部分・作用順・符号・control位相を含む一経路のunitaryである。canonicalな正規化Bが経路に依らない場合、元のfirst momentは

\[
A_P=B\sum_\omega P(\omega)V_\omega.
\tag{1}
\]

既存weightが経路依存ならBを\(W_0(\omega)\)に置き換える。物理targetは\(e^{-itH}\)等であり、\(A_P\)はその近似か、外側PFと内側finite-RTEを含む登録targetになる。

ここでのcoherent signalは\(\operatorname{Re/Im}\operatorname{Tr}(\rho A_P)\)。channel平均\(\sum P V_\omega\rho V_\omega^\dagger\)との一致で代用しない。[R1]

例えばIと−Iは同じsystem channelを与えるが、そのfirst momentは混合率で変わる。制御化するとscalar phaseが相対位相になる。

### 2.3 厳密な旧平均保持を普遍的な必須条件にしない

旧平均を保つ方法は比較しやすいが、それだけに研究を制限しない。旧平均を変えて新しいbiasを導入・相殺する方法でも、**共通の物理targetへの最終誤差とconfidence、総資源を比較できれば**研究対象である。

ただし、元のRTE保証をそのまま継承できる場合と、新たに証明する場合は分ける。新しいvarianceやbiasを旧sourceの値で埋めない。

## 3. 単純な持続型相関は旧平均を保存しない

以下は初期screening用の解析例であって、登録H4/H6の実験ではない。

\(Q_i^2=I\)、\(Q_i\) Hermitian、確率\(p_i\)、\(R=\sum_i p_iQ_i\)とする。一eventを

\[
U_i=cI-isQ_i,\qquad c=\cos\theta,\quad s=\sin\theta
\]

とする。最初のlabelはpで選び、次を

\[
K_\eta(j\mid i)=(1-\eta)p_j+\eta\delta_{ij},\qquad0\le\eta<1
\tag{2}
\]

で選ぶ。各時刻の周辺分布はpのままだが、同じlabelを維持する傾向がある。

\(A=\sum_i p_iU_i=cI-isR\)とすると、

\[
\begin{aligned}
\mathbb E_\eta[U_2U_1]
&=(1-\eta)A^2+\eta\sum_i p_iU_i^2,\\
\sum_i p_iU_i^2
&=(c^2-s^2)I-2icsR.
\end{aligned}
\]

従って

\[
\boxed{\mathbb E_\eta[U_2U_1]-A^2=-\eta\sin^2\theta\,(I-R^2).}
\tag{3}
\]

RはHermitian contractionなので\(I-R^2\succeq0\)。一般には\(O(\eta\theta^2)\)の差がある。\(\theta=\arctan\tau\)ならcanonical K0回転の構造に対応するが、normalizationは別に戻す必要がある。

この式は「周辺分布が同じだからPR meanが保存される」という仮定を排除する。反面、**物理targetに対する最終誤差が必ず悪化すること、全相関法の劣位を証明した式ではない。** 元approximationのbiasとの相殺もあり得る。K2/K4や決定論的Dを挟んだ\(U_jDU_i\)の評価を、この二因子式で代用しない。

## 4. 全経路再重み付けは正しいが、統計負担が変わる

### 4.1 exact change of measure

新しいjoint lawをqとし、\(P(\omega)>0\)の全経路で\(q(\omega)>0\)とする。weightを

\[
W_q(\omega)=B\frac{P(\omega)}{q(\omega)}
\tag{4}
\]

にすれば

\[
\sum_\omega q(\omega)W_q(\omega)V_\omega=A_P.
\tag{5}
\]

これは経路中に非可換な決定論的因子があっても成立する。パスを並べ替えず、同じ\(V_\omega\)を使って確率だけ変えるからである。

同じ各経路の近似回路\(\widehat V_\omega\)を使うなら、実装誤差の平均も元と同じになる。これは未知の実装誤差が解消することではない。self-normalized estimator\(\sum WX/\sum W\)へ無断で変更せず、通常の\(n^{-1}\sum WX\)を使う。weightのclip・経路削除も別のbiasを生む。

この全経路IS自体は、P4の一般parameter-vectorに対する結果の範囲に入る。「一stepから全pathへ広げる」だけを新規性にはしない。

### 4.2 ±1 Hadamard測定の場合

\(Y\in\{-1,+1\}\)、\(X=W_qY\)なら、

\[
M_2=\mathbb E_q[X^2]
=B^2\sum_\omega\frac{P(\omega)^2}{q(\omega)},\qquad
W_{\max}=B\max_\omega\frac{P(\omega)}{q(\omega)}.
\tag{6}
\]

M2は二次momentで、varianceは\(M_2-|\mathbb EX|^2\)である。真signalを無料で利用して最小shotへ置き換えない。

経路依存の旧weightなら、M2の分子は\(P^2|W_0|^2\)、rangeは\(\max|PW_0/q|\)となる。異なるmeasurement outcome規約ならその規約で計算する。

### 4.3 二状態持続kernelの正確な診断

\(p_1=p_2=1/2\)、経路長m、式(2)のkernelを考える。stayは\((1+\eta)/2\)、switchは\((1-\eta)/2\)。元Pは独立な一様列である。

一遷移のlikelihood factorはstayで\(1/(1+\eta)\)、switchで\(1/(1-\eta)\)。その二乗のq平均は

\[
\frac{1+\eta}{2(1+\eta)^2}
+\frac{1-\eta}{2(1-\eta)^2}
=\frac1{1-\eta^2}.
\]

従って

\[
\boxed{
\sum_\omega\frac{P(\omega)^2}{q(\omega)}
=(1-\eta^2)^{-(m-1)},\quad
\max_\omega\frac{P(\omega)}{q(\omega)}
=(1-\eta)^{-(m-1)}.
}
\tag{7}
\]

一方、期待switch数は\((m-1)(1-\eta)/2\)。例えば\(\eta=1/2\)ではswitch数は半減するが、M2倍率は\((4/3)^{m-1}\)、range倍率は\(2^{m-1}\)になる。

これは単純な解析controlであり、実DF費用、最適sampler、任意相関法の下界ではない。switch回数を実回路T/CX単価と等置もしない。

### 4.4 resetだけで蓄積を消せるわけではない

短いblockを独立に引き、そのlikelihood correctionを全blockで掛ける場合、一blockのratio二次momentが\(\kappa>1\)ならb blockで\(\kappa^b\)になる。短block化には設計上の意味があり得るが、resetという名称だけで総normalizationを一定とすることはできない。

## 5. ただしweightの指数増大は相関一般のno-goではない

任意の生成可能な相関経路law rに対して、**経路全体を一度選ぶ**混合

\[
q_\lambda=(1-\lambda)P+\lambda r,\qquad0\le\lambda<1
\tag{8}
\]

を使えば

\[
\frac{P}{q_\lambda}\le\frac1{1-\lambda},\qquad
\sum_\omega\frac{P(\omega)^2}{q_\lambda(\omega)}
\le\frac1{1-\lambda}.
\tag{9}
\]

boundはmに依らない。これは既知のdefensive mixture原理である。[P7] 各stepで小さい混合を反復する式(2)とは違い、一つの経路をPかrかから引く混合を使っている。

従って「相関を付けたら必ず指数的なweightが必要」とは結論しない。ただし、実際のq密度とP密度を求める費用、rの生成、finite-bit probability、Cの変化は残る。**boundが有限であることから資源改善は出ない。** 式(9)だけと期待費用の混合式を組み合わせた保守上界では、元Pより良いことを通常認証できない。

この混合を新しいPR手法として命名・採択しない。現在の判断では、全経路ISを比較する際の既知の安全な対照である。

## 6. 「何を最適化するか」は既知結果と区別する

### 6.1 全経路のcost–moment最適化

\(C(\omega)>0\)を実際の一経路wrapper費用とする。非加法的なcompiler効果を含めても、Cauchy–Schwarzにより

\[
\left(\sum_\omega q(\omega)C(\omega)\right)
\left(\sum_\omega\frac{P(\omega)^2}{q(\omega)}\right)
\ge\left(\sum_\omega P(\omega)\sqrt{C(\omega)}\right)^2.
\tag{10}
\]

等号のlawは\(q^*(\omega)\propto P(\omega)/\sqrt{C(\omega)}\)。P4のTheorem 1は個々のgateだけでなく、一般の回路parameter vectorを対象とする。非加法的全path costを引数にすること自体は、その定理外ではない。

最小のcost–moment積と元平均費用の比は

\[
\frac{(\mathbb E_P\sqrt C)^2}{\mathbb E_PC}
=1-\frac{\operatorname{Var}_P(\sqrt C)}{\mathbb E_PC}.
\tag{11}
\]

費用分布が狭ければ純粋なISの改善余地は小さい。しかし、この条件は実費用について確認しなければならず、IID labelだから、または有限標本SEが小さいからといって成立を仮定しない。未観測の高費用経路もある。

q*を式で書けても、全経路Cの取得と正規化、確率密度の評価、samplingが安いとは限らない。指数的経路表を無料で使う対照も、候補だけに安いcost oracleを与える比較も行わない。C=0がある場合はそのmassを別途扱い、無条件に\(1/\sqrt C\)を用いない。

### 6.2 finite confidenceまで戻す

一軸の目標幅をε、bias上界をb、残る統計幅を\(s=\varepsilon-b>0\)、失敗確率をαとする。独立な**全trajectoryをshotごとに**引くなら、\(|X|\le W_{\max}\)、variance上界M2を使った保守的なBernstein十分予算の一例は

\[
n\ge\left\lceil\log(2/\alpha)
\left(\frac{2M_2}{s^2}+\frac{4W_{\max}}{3s}\right)\right\rceil.
\tag{12}
\]

係数4/3は\(|X-\mathbb EX|\le2W_{\max}\)を用いたもの。これは新しい最適shot定理でも、固定pilotの新契約でもない。既存のconfidence規則を変えるなら別途明示する。

期待量子費用は\(G=\sum_a n_a\mathbb E_q C_a\)。\(\mathbb E_q[W^2C]\)だけを総費用とみなしたり、回路時間の削減率をshot込み改善率としたりしない。classical generation、lookup、normalizer、synthesis取得費用も、量子資源とは区別して戻す。

shot間まで相関させる場合には式(12)の独立性も失う。path内相関とshot間相関を混同しない。

## 7. 平均そのものを変える高次化には、具体的な既知構成がある

### 7.1 二ステップの解析的な例

ここは一般Hermitian生成子を用いた説明用構成であり、実finite-RTEのatan角度や既存DFの登録計算ではない。\(\overline H=\sum_i p_iH_i\)、\(\|H_i\|\le1\)とし、

\[
F(h)=\sum_i p_i e^{-ihH_i},\qquad
B_2(h)=2F(h)^2-F(2h).
\tag{13}
\]

\(K_2=\sum_i p_iH_i^2\)とおくと

\[
F(h)=I-ih\overline H-\tfrac12h^2K_2+O(h^3),
\]

\[
2F(h)^2-F(2h)
=I-2ih\overline H-2h^2\overline H^2+O(h^3)
=e^{-i2h\overline H}+O(h^3).
\tag{14}
\]

また同じ二因子の語で

\[
B_2(h)=\sum_{i,j}a_{ij}e^{-ihH_j}e^{-ihH_i},\quad
\boxed{a_{ij}=2p_ip_j-\delta_{ij}p_i.}
\tag{15}
\]

\(\sum a_{ij}=1\)、row/column和はpである。ただしaは一般に確率ではない。\(0<p_i<1/2\)の対角係数が負になる。

この展開は実際に二次項を相殺するので、「相関を使っても高次化できない」という結論は誤りである。一方、これはmultiproduct／quasiprobability型の既知機構に近く、二項の場合はqSHIFT Appendix C.2の係数と一致する。[P2, P6]

\(p_1=p_2=1/2\)の場合は対角係数0、二つの順序に各1/2で、通常の順序反転randomizationに対応する。任意多数項での低コストのpositive samplerが得られたわけではない。また\(H_i^2=I\)等の特殊関係がある場合、形式wordが独立だと仮定した一意性は主張しない。

### 7.2 negative coefficientの費用

\(\Gamma_2=\sum_{i,j}|a_{ij}|\)として\(|a|/\Gamma_2\)から引き、\(\Gamma_2\operatorname{sign}(a)\)を掛ける。一様L項でL≥2なら

\[
\Gamma_2=3-\frac4L.
\tag{16}
\]

同じblockをb回独立に使い、全blockのsigned weightを掛ける素朴な構成では、weightの絶対値は\(\Gamma_2^b\)、二次momentは\(\Gamma_2^{2b}\)。高次化でblock数を減らす利益との釣り合いが必要である。

この式も、全qSHIFT・PRHSの実装が必ずこの倍率になるという主張ではない。別の高次係数、時間依存、正規化、推定器、block配置は別に調べる。

### 7.3 論文の式を黙って修正して使わない

取得したqSHIFT v1 HTMLでは、本編Eq.(12)のoff-diagonal係数とAppendix C.2のoff-diagonal係数にfactor 2の表示差がある。前者の表示式をそのまま足すと一般には1にならず、後者はその節の制約と式(15)に整合する。

本レビューは、この不整合を記録して**Appendix C.2との限定的な係数対応**だけを使う。原PDFの版・著者の訂正・実装上の採用式を確認せず、全qSHIFTの理論や数値結果が無効と断定しない。また本編を無言で補正して「文献全体が検証済み」ともしない。次の技術照合で式の正本を固定すべき具体点である。

### 7.4 PRHSとの関係は本文取得後

PRHSのabstractは、M correlated slicesとquasiprobability、任意次数への拡張を明示する。[P3] この一般方向は今回の候補と非常に近い。ただし、現時点で式(15)がPRHSの低次数特殊化だと確認したわけではない。原本文から、係数、l1 norm、block反復、sampling費用、targetと誤差定義を照合する必要がある。

論文が報告するequal-query-costの高精度結果を、coherent PRのshot-inclusive native/T優位へ転用しない。

## 8. 既存PRに接続する場合の本質的な比較条件

### 8.1 qDRIFTに勝つことと、現RTEに勝つことを分ける

現sourceは次数をpairしたfinite RTEを使用している。新候補がqDRIFTの一次誤差を改善しても、現在のcanonical finite RTEや許された既存対照を超えたことにはならない。[R1]

比較すべきなのは、共通物理時間・target・入力情報の下での誤差と費用である。少なくとも外側PF、内側近似、probability/weight有限精度、native/synthesis、統計誤差を区別する。論文のchannel次数をRTEのfirst-operator次数へ同じ文字のまま移さない。

### 8.2 channel法を一律に除外しない

P1・P2は主としてchannel／Heisenberg observableの定式化である。Hadamard first momentへ直接等置できないが、\(H_j\mapsto |1\rangle\langle1|\otimes H_j\)などの制御付き拡張を使い、ancillaを含めたchannelを解析する方法は候補になる。

これは無料の移植ではない。制御gate、scalar phase、アクセス可能な演算子、誤差norm、ancillaを含むobservable、全資源を確認する必要がある。P1のbalanced schemeについての結果を、自由な任意transition-cost kernelの定理として使わない。

### 8.3 既存compiler共有は両側へ与える

R2は、独立に引いた列でも一体compileに非加法的な効果があることを扱った資料である。新lawだけにboundary fusionを与え、iid側をevent費用単純和へ戻して利益を作らない。

同じ理由で、変更したのがsamplingか、compilerか、targetかを分ける。決定論的blockを挟む境界では、相関したlabelだけを見てgateが相殺するとは仮定しない。

## 9. 量子shotと古典的な経路varianceを分ける

fine/coarseの同じ乱数を使い、理想的な二つの期待値が近づいても、実際に別々に測る±1 outcomeが自動的に相殺するわけではない。

共有した古典経路ωの下で別測定が条件付き独立なら、各conditional meanを\(\mu_f(\omega),\mu_c(\omega)\)として

\[
\operatorname{Var}(Y_f-Y_c)
=\mathbb E[1-\mu_f^2+1-\mu_c^2]
+\operatorname{Var}(\mu_f-\mu_c).
\tag{17}
\]

第二項が小さくても第一項は残る。これは実測varianceについての全分散公式であり、新規実験ではない。

MLMC-qDRIFTはこの問題自体を§2.3で扱い、augmented difference-stateと非unitary mapの実装へ進めている。[P5] 従って「seed共有でshot削減」という素朴案は採択せず、一方「量子MLMCは不可能」とも結論しない。今回は別の推定器研究を同時に立ち上げない。

## 10. 研究方針の比較と採否

| 方向 | 今回支持する内容 | 現時点で採択しない理由／残る問い |
|---|---|---|
| 持続型Markovをそのまま使用 | 周辺分布を保つ相関は作れる | 旧meanを変える。新biasを評価しないまま費用比較できない |
| そのMarkov lawを全経路補正 | exact meanは保持できる | 単純例でM2/rangeが指数増大。方法自体は既知IS |
| defensive mixture等の有界weight | 相関一般のno-goを避けられる | 既知構成。生成・cost・有限confidenceの実益は未確認 |
| signed correlated blockによる高次化 | 明示的な二次相殺は成立 | 近い既知係数がある。多block正規化と現finite RTE対照が必要 |
| arbitrary moment-preserving coupling | operator関係を使えば余地があり得る | 具体的な安価なnull direction・sampler・改善機構が未定義 |
| fine/coarse等の推定器相関 | 経路varianceを減らす別の考え方 | shot雑音・非unitary実装を含む別task。今回は拡張しない |

**採択するのは、一次文献と既存PRの数学・資源の対応を限定して確認する作業である。** 新しい相関law、source、fixture、性能閾値、pilot予算は採択しない。

有望候補を必ず一つ出すことを完成条件にしない。既知部品を使うこと自体、定数因子改善しかないこと、漸近定理を持たないことを理由に一律却下することもしない。次に必要なのは、具体的な生成手順・保証・アクセス・費用の差である。

### 10.1 このRQで今言えること

研究として残す問いは、

> 元のfinite-RTE PRのcoherent taskと、同じ利用可能情報の下で、時間方向の相関が生む回路／bias改善が、weight・range・古典生成・実装精度の負担を上回る、具体的に生成可能な構成があるか。

これは研究課題の整理であり、新手法を採択したものではない。「既知Markov chainのパラメータをgrid searchする」ことを直ちに最小pilotとしない。

### 10.2 研究としての着地点

具体的な差が残れば、生成手順・保証・適用条件・同taskの資源削減を主要寄与にする可能性がある。既知法の直接適用だけなら、応用上の独立した問いを明示して選び直す。何も差が残らなければ、Track Bの実験休止を維持してよい。

今回の式(3)、(7)、(9)、(15)だけで独立論文の新定理や新PR法が得られたとは判断しない。これらは、誤った比較を早期に避け、近い先行構成へ対応させるための解析である。

## 11. 次のCodex作業：限定的な一次資料・式・資源会計の照合

### 11.1 目的

PRHS本文という具体的な未取得資料と、qSHIFTの保証・式・sampling会計を確認する。以前の133 branch横断調査をやり直すのではなく、一つの技術作業単位としてまとめる。

### 11.2 指示として使用できる作業範囲

1. arXiv:2608.29756（PRHS）の版付き原本・利用可能な補足を取得し、著者・版・日付を確認する。取得不能なら試行先と原因を区別し、abstractから定理を再構成しない。
2. arXiv:2604.26263v1（qSHIFT）は§III、Appendix B/C、特にEq.(12)とC.2の係数表示の違いを、原PDFまたは訂正版と照合する。無言で式を修正して採用しない。
3. 両手法について、生成するjoint law、実際のpositive sampling probability、signed/likelihood weight、normalizer、有限bit処理、history更新、全blockのmeanと二次moment・rangeを分けて記録する。
4. 保証targetがunitary first momentか、channelか、特定observable/stateか、normは何かを原文の式番号付きで示す。coherent PRへ移すために制御付き拡張が必要なら、その追加条件を記録する。
5. query/gate countと、実際の±1等の量子measurementまで含めたshot数を区別する。古典的path expectationの分散を実measurement varianceと置き換えない。
6. 現RTE source[R1]、全経路IS[P4]と同じ情報を使って比較し、既知構成への還元部分と、未定義・未取得部分を一枚の対応表にする。高次qDRIFTとの比較だけで現PRへの優位としない。
7. 本レビューの簡単な解析例は、独立した式の展開・source/文献の対応として確認してよい。新しい登録行列生成、分子計算、sampling、native compile、T合成、scientific runner/testsは行わない。
8. 結果は「既知手順への対応」「追加の具体差あり」「必須資料不足」を区別する。研究採択やclosure解除、実行認可をCodexが代行しない。
9. 文書化するなら小さい取得・対応記録で十分。必要なURL・版・section・hashを残し、転載権を確認せず他者論文PDFを公開repositoryへpushしない。既存dirty差分・source・result・markerを保全する。
10. 一束の報告でSTOP。細かいsource変更ごとにGPTレビューを要求せず、数理保証・方法差・本番pilot採否に重要な判断が必要になった時点だけ戻す。

この作業は、論文取得を理由に未承認の実験を実行する認可ではない。技術照合後に具体的な候補が残った場合だけ、Codexが最小の識別検証案を提示し、科学的な範囲の採択とユーザーの別途実行承認へ進む。

### 11.3 次の判断条件

- **追加の研究差が具体化する場合**：対象、最も近い対照、数式または実行手順、期待される改善と支払う負担、反例になり得る条件が揃ったところでGPTが採択を判断する。
- **既知法の適用に留まる場合**：その事実を報告し、勝手に分子benchmarkへ移らない。応用研究を選び直すかは別判断。
- **資料が取得できない場合**：その部分は保留。既知性・非優位の確定根拠にしない。存在するURLと実取得可能性を分ける。
- **単なるparameter違いしかない場合**：新計算を立ち上げない。過去の終了候補へ条件を足して成功例を探す運用に戻らない。

新しい改善率閾値・締切・分子数・seed・計算時間上限は本レビューで追加していない。新実行を設計する段階で必要な資源範囲を定め、旧凍結契約には触れない。

## 12. 最終判断と既存方針への影響

| 論点 | 判断 |
|---|---|
| 時間方向相関という自由度 | 未否定。一般に有用・無用とは判定しない |
| 周辺分布を保つだけで旧mean保持 | 成立しない。式(3)の反例 |
| path reweightingによる旧mean保持 | 成立。ただし統計・費用を再評価する |
| weight指数増大が全相関に不可避 | 認定しない。式(8)–(9)が反例となる既知構成 |
| 簡単な二ステップ高次化の独立新規性 | 認定しない。qSHIFT C.2・multiproduct型構成と近接 |
| PRHSとの完全な方法差 | 原本未取得のため保留 |
| 相関samplerの新規実装・pilot | 今回は採択しない |
| 次の担当 | Codex：近接一次文献の版・式・保証・総資源対応を限定確認 |
| Track Bの自動的な実験立上げ休止 | 維持。今回の読取り・解析は休止に矛盾しない |
| generic whole・Gaussian等の既存closure | 維持 |
| Track A・N3 | 変更・統合・終了指示なし |
| 登録科学実行 | 別途明示承認が必要、現時点で未認可 |
| mandatory STOP | 維持 |

**総括：相関そのものを新規性・資源利益の代わりにしない。正しいmean、高次化の代償、実際のshotと回路費用までを同じtaskへ戻す。現在は新手法を無理に作るより、近接する既知構成との具体的な対応を一度完了させる地点である。**

## 付録A：正のunitary混合についての限定的な二次条件

これは今回の全相関法のno-goではなく、適用前提を明記した解析上の注意である。

固定有限確率pω、\(U_\omega(0)=I\)、滑らかなunitary path

\[
U_\omega(t)=I-itK_\omega+t^2L_\omega+O(t^3)
\]

を考える。unitarityより\(L_\omega+L_\omega^\dagger=-K_\omega^2\)。正重み平均が\(e^{-itH}+O(t^3)\)なら\(\mathbb E K_\omega=H\)、\(\mathbb E K_\omega^2=H^2\)で、

\[
\mathbb E(K_\omega-H)^2=0.
\]

各項はpositive semidefiniteなので、正確率の各pathで\(K_\omega=H\)。つまり、この条件下で二次までunitary first momentを一致させるには、一つのpathのfirst-order generator自体がHを再現する必要がある。全項を一度ずつ含むrandom-order sweepなどはこの条件に反しない。

時間依存の確率・support、t=0でもidentityでない稀なRTE path、B(t)>1、signed weights、state-specificな量だけの保証へこの議論を広げない。ましてPR改善全般の不可能性とはしない。

## 付録B：qSHIFTのsampling会計に関する読み方

qSHIFT Appendix B.2の表示式は、pathごとの量子期待値を確率変数として二乗する部分を含む。そこから直ちに、実際に一回測定した±1 outcomeのvarianceと同一であるとは扱えない。原文のsamplingの定義、数値実験がexact path expectationsか測定sampleか、必要な量子shotを改めて区別する必要がある。

これは本レビューが当該論文の全complexity claimを無効と認定したことではない。式の表示差と同様、**本研究の資源式へ取り込む前に確認すべき具体的な項目**である。本書の式(6)、(12)、(17)はそれと分けて導出した。

## 参考資料と取得先

### 内部資料

- [R1] RTE event・sampler source（固定commit）  
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py
- [R2] 階層compiled-cost modelの拡張holdout検証  
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/hierarchical_cost_validation.md
- [R3] Gaussian積生成候補の終了記録  
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md
- [R4] `track_b_overall_direction_and_research_pause_review_20261013.md`。会話添付の原本。今回確認した内容は特に第0–2節。

### 外部一次資料

- [P1] Benoît Dubus, Joseph Cunningham, Jérémie Roland, *New random compiler for Hamiltonians via Markov Chains*, arXiv:2411.06485v2.  
  https://arxiv.org/html/2411.06485v2
- [P2] Sangjin Lee, Sangkook Choi, *qSHIFT: An Adaptive Sampling Protocol for Higher-Order Quantum Simulation*, arXiv:2604.26263v1.  
  https://arxiv.org/html/2604.26263v1
- [P3] Davide Cugini, *Pathwise Random Hamiltonian Simulation*, arXiv:2608.29756（2026-08-30）。公式abstractのみ取得。  
  https://arxiv.org/abs/2608.29756
- [P4] Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1.  
  https://arxiv.org/html/2603.13495v1
- [P5] Pegah Mohammadipour, Xiantao Li, *MLMC-qDRIFT: Multilevel Variance Reduction for Randomized Quantum Hamiltonian Simulation*, arXiv:2604.26865v1.  
  https://arxiv.org/html/2604.26865v1
- [P6] Paul K. Faehrmann, Mark Steudtner, Richard Kueng, Maria Kieferová, Jens Eisert, *Randomizing multi-product formulas for Hamiltonian simulation*, Quantum 6, 806 (2022).  
  https://quantum-journal.org/papers/q-2022-09-19-806/
- [P7] Tim Hesterberg, *Weighted Average Importance Sampling and Defensive Mixture Distributions*, Technometrics 37(2), 185–194 (1995). 出版社abstract確認。  
  https://www.tandfonline.com/doi/abs/10.1080/00401706.1995.10484303

本書の新規作成は研究判断の記録であり、既存repositoryのstatus、結果、契約、source、authorization、markerを変更したものではない。
