# G6：return集約・非列挙生成の独立数学監査

2026-10-10 JST。利用者が採用した[GPT G5 review](../../research/track_b_G5_research_direction_review_20261010.md) §19の技術監査。
旧G5の性能比較とは別の探索的数理記録。以下の証明と、新しいoff-domain形式テストを区別する。
「独立」はレビューのchecker・旧RA evaluatorを用いない再導出・再実装を意味する。外部機関の再現、査読済み定理、immutable CIではない。

## 1. 仮定と対象

有限の正の有理数p_i、sum p_i=1、L>=1、奇数m>=1、0<x<=1、sigma=±1。
零質量labelは入力段階で除外できる。x=0は厳密なidentityとして別扱い。
Q_iはHermitian involution、Q_i^2=I。異なるlabel間の関係は一切利用しない。
Q(u)=Q_i1...Q_ilはoperatorの左から右の順序であり、circuitの時間順では逆になる。
同一隣接labelの除去で得る形式reduced wordをuとする。実際のQが追加関係を持っても、形式恒等式の写像は成立する。
奇数語Q(u)自体がinvolutionだとは仮定しない。

R=sum p_i Q_i、対象は**固定finite polynomial** M=P_m(-i sigma xR)である。
exp(-i sigma xR)へのTaylor remainderやPR全体のerror budgetはここで認可・評価していない。

## 2. 集約恒等式：PASS

P_n(u)をn回IID labelのraw wordがuへreduceする確率とする。
reduceは長さのparityを保存するため、n>=l=|u|、n-l偶数だけが寄与する。

\[
a_u=\sum_{\substack{l\le n\le m\\n-l\ {m even}}}
(-1)^{(n-l)/2}\frac{x^n}{n!}P_n(u),\qquad
M=\sum_u(-i\sigma)^{|u|}a_u Q(u).
\]

有限Taylor展開のraw wordを同じreduced wordへまとめるだけである。
(-i sigma)^n=(-i sigma)^l(-1)^((n-l)/2)が符号の根拠。
非可換な語をreorderせず、coherent signalに必要な相対位相も保持する。

## 3. 挿入上界：PASS（等式ではない）

chi=sum p_i^2。n>=|u|で

\[
P_{n+2}(u)\le(n+1)\chi P_n(u).
\]

長さn+2のraw wordでreduced長<=nなら隣接iiが存在する。
例えば最初の隣接pairを決定的に削除すると、uへreduceする長さnのraw wordになる。
逆に各長さnの語のn+1 gapsへ任意iiを挿入する。すべての対象語を少なくとも一回覆う。
挿入後の語の確率重みは、元の語の重みにp_i^2を掛けたもの。
全挿入のmultisetを足すと上界が得られる。多重生成を等式と扱うことはできない。
P_n(u)=0の場合も右辺が0になり、同じcoverから左辺0が従う。

## 4. Short-step非負性：PASS

c_n=x^n P_n(u)/n!とすると

\[
\frac{c_{n+2}}{c_n}\le\frac{\chi x^2}{n+2}\le\frac12,
\quad c_l=\frac{x^l}{l!}\prod_{j\in u}p_j.
\]

有限alternating sumなので、最初の二項と最初の一項が下界・上界になる。
末尾が一項だけの場合も同じ弱い下界を使用できる。

\[
\frac{x^l}{l!}p(u)\left(1-\frac{\chi x^2}{l+2}\right)
\le a_u\le\frac{x^l}{l!}p(u).
\]

従ってすべての形式語のa_u>0。一般x>1への延長は主張しない。
「集約すれば常に正」はこのshort-step条件なしには未証明である。

## 5. 形式母関数：PASS、ただし既知free-product式の特殊化

Q_i^2=Iだけで作るfree product of Z2のCayley treeを用いる。
F_iは根からi-labelled隣接頂点へのfirst-passageの形式確率級数。
別label jへのexcursionとreturnを繰り返し、最後にiへ一歩進む分解から

\[
F_i(z)=\frac{zp_i}{1-z\sum_{j\ne i}p_jF_j(z)},\quad F_i(0)=0,
\qquad G(z)=\frac1{1-z\sum_i p_iF_i(z)}.
\]

F_i^2などを消去した数値root選択をせず、z=0での形式解を次数順に求める。
根からuへの唯一のsimple pathをfirst-passageで通過し、最後のreturn分を付けると

\[
\sum_{n\ge0}P_n(u)z^n=G(z)\prod_{j\in u}F_j(z).
\]

確率係数のproductはcommutativeだが、Q(u)の順序は保持する。
analytic transience、infinite-length sampling、resolventの数値branchを必要としない。
L=1はF_1=zp_1=z、G=(1-z^2)^(-1)というrecurrent caseを含む。

これはAomoto–Kato §1 Eq.(1.6)、Lemma 1.1のGreen function multiplierをZ2へ特殊化した式である。
彼らのp_bar_i=p_i/2、spectral variable zeta=1/zを用い、resolventをzで割ると上のGとなる。
Z2単因子のmultiplierはp_i/zeta、他因子のreturnによるshiftはsum_(j!=i)p_jF_j。
従ってF_i=zp_i/(1-z sum_(j!=i)p_jF_j)へ戻る。これは**本監査の対応導出**であり、
当該論文にfinite Taylor/RTE rejection algorithmがそのまま掲載されているとの主張ではない。
[一次本文 pp.62–65](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)。

## 6. Parent pairing：PASS

偶数長reduced uをparentにし、uが空なら全i、空でなければi!=first(u)をchild iuとする。
各奇数reduced wordは、その最初のlabelを除いた唯一の偶数suffixを持つ。

\[
s_u=\sum_{i\in C(u)}a_{iu},\quad d_u=\sqrt{a_u^2+s_u^2},
\quad\phi_u=\arctan(s_u/a_u),\quad q(i|u)=a_{iu}/s_u,
\]
\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_u Q_i}Q(u).
\]

involutionのEuler identityでcos phi=a/d、sin phi=s/d。
d_u E_i[V]はparentのa_uと全childのa_iuを厳密に再構成する。
sum_i q_i=1がparentを一回だけ含める根拠。s=0なら純parentを扱う。
非可換語全体を回転generatorにしない。回転は単一のQ_iに対してのみ必要。
Controlled実装ではparent位相(-i sigma)^l=(-1)^(l/2)を捨てられない。

## 7. Ordinary envelope・acceptance：PASS

t_l=x^l/l!、b_l=sqrt(t_l^2+t_(l+1)^2)、B_ord=sum_(even l<=m-1)b_l。
§4からa_u<=t_l p(u)、s_u<=t_(l+1)p(u)。従って

\[
d_u\le b_l p(u),\quad B_{new}=\sum_{u\ {m even,reduced}}d_u\le B_{ord}.
\]

第二不等式は、各lのreduced語のIID質量が1以下という事実を使う。
全parentやB_newを実際に列挙・計算する義務はない。
a_u>=t_l p(u)/2、b_l<=sqrt(2)t_lより、各有効parentのacceptanceは

\[
A_u=d_u/(b_l p(u))\ge1/(2\sqrt2)>1/4.
\]

x>0,m>=3では空parentのa_empty<1であり、少なくとも一箇所のenvelopeがstrictになる。
従ってideal normalizationのstrict減少自体は、数値のwinner探索をしなくても導ける。
これはnative cost、古典取得費用、総資源のstrict改善を意味しない。

## 8. Zero-fill estimatorと未知normalizer：PASS_WITH_BUDGET_LIMIT

一回の試行でlをb_l/B_ordから選び、長さlのraw IID語を生成する。
**raw語がreducedでなければzero trialにする**。reduce後の別parentへ移してはいけない。
reduced parent uをA_uでacceptし、childをq(i|u)から選ぶ。
accept後だけcoherent-axis Hadamard outcome Y=±1を測定する。その他はX=0。
accept時X=B_ord Y。平均の分母はaccepted countではなく**全試行数**である。

\[
\Pr(u\ {m accepted})=d_u/B_{ord},\quad
Z=B_{new}/B_{ord},\quad
E[X]=\Re\text{ or }\Im\operatorname{Tr}(\rho M),
\]
\[
E[X^2]=B_{ord}B_{new}=B_{ord}^2Z,\quad |X|\le B_{ord}.
\]

empty-parent寄与からZ>=a_empty/B_ord>=(1-chi x^2/2)e^(-x)>=1/(2e)。
期待試行数はboundedだが、すべての拒否・局所queryのclassical costも課金する必要がある。

axis remaining error s、ell=log(2/alpha)の場合、standard Bernsteinの十分条件は

\[
N\ge\ell\left(2B_{ord}^2 Z_+/s^2+4B_{ord}/(3s)\right),
\]

ここでZ_+は認証済み上界。未知Zを真値として無料使用してはいけない。
追加取得なしならZ_+=1とする。別のZ推定を行うなら、その古典cost/confidenceを別契約へ入れる。
もしZが別途既知なら、ceilを無視した理想的な期待量子呼出し数NZは
ell(2B_new^2/s^2+4B_new/(3s))となるが、今回の実装済みshot plan・性能証拠ではない。

ZeroFillで全試行を数え、Discardでは追加normalizationが要る原理自体は既知である。
[Cugini et al. §IV.1–IV.2](https://arxiv.org/html/2603.13495v1)。

## 9. m=3・state・負例診断

\[
a_\emptyset=1-\chi x^2/2,\quad
a_i=p_i[x-x^3(2\chi-p_i^2)/6],
\quad s_{jk}=x^3(1-p_j)p_jp_k/6\quad(j\ne k).
\]

従ってtan phi_(jk)=x(1-p_j)/3。m=3 formulaだけを新規性とはしない。
full degree-3 return吸収の符号を保持する。
ABBAは空、CBBAはCAへreduceするので、逐次reduceするならlast labelだけではstateが不足する。
ABCABCは一般free-product語として非空なので、odd wordをinvolution扱いする変換は不正。
raw proposalのreduce先への再配分、accepted-only平均、挿入上界を等式とする誤りをfocused testsで診断した。

## 10. 証明と検算の区別・結論

§2–8が一般命題の再導出であり、有限fixtureの一致が証明の代用品ではない。
16 focused testsでは独立なfirst-adjacent-pair deletionをoracleに使い、生成kernelのstack/recurrent seriesと比較した。
raw形式語3,965、母関数係数3,794、positive集約503、挿入上界178、parent envelope170、
sigma±1の形式平均12、digital event重み63の照合。分子・DF・量子matrix・circuit・native synthesisは0。

**結論**：固定short-stepのideal数式に反例は見つからず、上述の仮定のもとで成立する。
ただしfree-product母関数とzero-fillは既知原理である。
[prior-art監査](g6_prior_art_and_method_delta_20261010.md)と
[有限bit・access監査](g6_finite_bit_and_access_audit_20261010.md)を併読する。
独立新規性、native成功、総費用優位は未確定。G6後mandatory STOP。
