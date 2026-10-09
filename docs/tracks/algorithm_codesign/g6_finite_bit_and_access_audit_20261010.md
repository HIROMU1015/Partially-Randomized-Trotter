# G6：finite-bit、非列挙性、control・費用の契約監査

2026-10-10 JST。対象・数学仮定は[独立証明](g6_independent_mathematical_audit_20261010.md)。
本回実装はstdlibの[局所prototype](../../../src/trottertracks/algorithm_codesign/return_aggregation.py)。
実機sampling、circuit、matrix、native angle synthesisは行っていない。
ideal irrational sampler、有限bit approximation、未実装native実行を区別する。

## 1. Access契約

| Access | 必要な内容・費用 | 本回の状態 |
|---|---|---|
| p table | 明示されたL個の正のrational p_iと総bit長。入力read自体Omega(L) | 小さいoff-domain形式値のみ |
| Q label | Hermitian involutionとQ_i²=I。追加代数関係は使わない | 形式labelとしてのみ使用 |
| controlled Q_i | control位相を含む厳密なoracle、又はstrict error/cost証明 | **MISSING**。system Q_i oracleだけで無料に得られるとしない |
| controlled exp(-i sigma phi Q_i) | 可変phi、signed時間、strict relative phase、workspace・error・native cost | **MISSING**。Q_iから安価に生成できるとは仮定しない |
| root/係数query | local a_u,s_u,d_uの有理算術とroot interval | 実装・focused testsあり |
| finite-bit probability/weight | full-support dyadic proposals、補正weight、zero trialを含むrange/bias | 局所packet・deterministic bit mapping・rational weightまで実装 |
| native error/cost取得 | 局所word依存角度に対するcost/errorをonline取得するか、別の一様証明を使うか | **MISSING**。全語のcost表を暗黙に使用しない |
| confidence budget | known range/variance上界、bias、初期化/測定/拒否の費用 | ideal boundは証明。実taskのbudgetは未採択 |
| global B_new/Z | 全parent normalizer | **不要**な生成法。ただし未知Zをshot削減へ使わない |

Pauli取得costの下界、DF固有access advantage、暗黙入力をsublinearに読むalgorithmは証明していない。

## 2. 算術量とbit量

f_i[n]=[z^n]F_i、C[n]=sum_i p_i f_i[n]とすると

\[
f_i[n]=p_i\mathbf1_{n=1}+\sum_{b=1}^{n-2}f_i[b]
(C[n-1-b]-p_i f_i[n-1-b]).
\]

右辺は既知の低次数係数のみ。全iを計算した後C[n]を保存する。
G[n]=sum_(a=1..n-1) C[a]G[n-1-a]、G[0]=1。
直接のj!=i loopならO(L²m²)だが、prototypeは上記Cの再利用で**O(Lm²)** rational operations、O(Lm) coefficients。
一つのparent wordのproduct queryはO(lm²)、その全childはcached parent seriesに各F_iを一回convolveしてO(Lm²)。
一試行の有効parent queryはO(m³+Lm²)。raw word判定はO(m)、word/stateはO(m log L) bits。
prototypeは固定個のqueryだけを保持し、past wordから全宇宙tableを蓄積しない。

共通分母D=lcm denominator(p_i)、input bit長H_pとすればlog D<=H_p。
確率級数のn次係数は分母D^nで書け、numeratorのbit長もO(n H_p)に抑えられる。
有理xのbit長H_xを含むa_uは分母がD^m den(x)^m m!を割る。
従って係数bit長はO(m(H_p+H_x+log m))。
schoolbook exact integer arithmeticも含めてpolynomialだが、**rational operationsの数をbit operationsや実費と同一視しない**。
native Q/rotation取得はこの係数算術に含まれていない。

## 3. Finite-bit proposal：exact irrational lawとは区別する

理想のb_l/B_ord、A_u、d_uは一般にirrationalなので、有限bitでそのままexact sampling/weightを実装したとは言えない。
G6は以下のbounded approximation構成を別に記述・実装した。

1. order、label、acceptance、childの各確率をfull-support dyadic値にする。
2. 局所eventの**実際のdyadic proposal確率pi**を計算する。
3. d_uの有限rational近似d_tildeを使うtarget係数alpha_tildeをpiで割り、rational weightを付ける。
4. d_tilde-dによるoperator mean biasを明示的に課金する。確率丸め自体はproposal補正で相殺する。
5. native angleはsymbolic tangent s/aを渡すだけ。ここではsynthesisしない。

具体的にideal event係数はalpha_(u,i)=d_u a_iu/s_u。s=0ならalpha_u=d_u。
dyadic order h_l、label h_i、accept h_A、child h_(i|u)から

\[
\pi_{u,i}=h_l\prod_{j\in u}h_j\ h_A\ h_{i|u},\qquad
\widetilde\alpha_{u,i}=\widetilde d_u a_{iu}/s_u,\qquad
\widetilde W_{u,i}=\widetilde\alpha_{u,i}/\pi_{u,i}.
\]

非reduced raw proposalとaccept拒否はzero。これらも全試行数へ含める。
q_iのdyadic lawとweightでpi W=alpha_tildeが**exact rational**に成立する。
そのmeanはM_tildeであって、一般には元Mに厳密一致しない。
この点を「有限bitでexact有限Taylor平均」と過大に書かない。

## 4. Support・range・biasの一般上界

各dyadic massがidealの(1±eta)内、0<eta<1、|d_tilde-d|<=rho dとする。
order、l個label、accept、childを掛けるので

\[
\pi_{u,i}\ge(1-\eta)^{l+3}\alpha_{u,i}/B_{ord},\quad
|\widetilde W|\le\frac{(1+\rho)B_{ord}}{(1-\eta)^{m+2}}.
\]

pure-parentにはchild factorがないので同じ保守的上界を使える。
coefficient L1 errorとoperator errorは

\[
\sum_{u,i}|\widetilde\alpha_{u,i}-\alpha_{u,i}|
=\sum_u|\widetilde d_u-d_u|\le\rho B_{new}\le\rho B_{ord}.
\]

B_ord<=exp(x)<3であるため、未知B_newなしにrange<=3(1+rho)/(1-eta)^(m+2)、bias<=3rhoとできる。
finite lawのvarianceがideal B_ord B_newに**そのまま一致するとはしない**。
必要なら上の既知rangeから保守的な十分shot数を立てる。今回は実taskへのshot割当を実行していない。

## 5. 必要bit長・bounded construction

short-step下の一様support下界がある：

- ideal order q_l>=t_(m-1)/3。
- labelsはp_min>0。
- valid-parent acceptance A>=1/(2sqrt2)>1/4。
- child q(i|u)>=p_i/2>=p_min/2（存在するchildのみ）。
- d_u>=t_l p(u)/2>=t_(m-1) p_min^(m-1)/2。

これらのlog inverseはO(m(H_p+H_x+log m))。従って相対誤差eta,rhoを与えたroot enclosureとdyadic丸めは、
入力bit長とlog(1/eta),log(1/rho)にpolynomialなbit数で構成できる。
prototypeの`root_interval`はinteger isqrtで分母2^Kの証明区間を一回作る。
probabilityはmidpoint normalizationとlargest remaindersを使い、全qについて
q_tilde>=(1-eta)q_hi、q_tilde<=(1+eta)q_loを検証する。
acceptanceはlower endpointを下向きに丸め、同じsupport下界を検証する。
必要precisionが不足すれば**fail closed**。ランダムなbit追加・sampler retryはない。
任意入力へのH,K自動選択器、random bit source、quantum measurement interfaceは本回実装していない。
固定H bitsからindexへのmappingまでが実装範囲である。

形式fixtureはK=256、H=160、eta=1/1000、rho=1/1000000を結果前固定。
これらは新しいscientific precision条件ではなく有限bit意味論のoff-domain test設定。
63 eventでpi W=alpha_tilde、positive support、range上界、global L1 budgetをexact rationalで確認した。
全wordを使うのは**独立oracle側の小さいtestだけ**。生成prototypeには全word enumeration、B_new、global cost tableがない。

## 6. Angleとcontrolled implementationの未閉鎖点

tan phi=s/a<=2x/(l+1)<=2。数学的なangle enclosureは、
atan r=2 atan[r/(1+sqrt(1+r²))]と、絶対値<2/3のalternating atan seriesを使えば、
log(1/tau)にpolynomialなprecisionで構成できる。
これは数理上の有限precision処理可能性であり、native gate列・T数の保証ではない。
角度近似error tauなら、involution回転のstrict operator差はtau以下。
相対位相を含むnative event error deltaとcoefficient biasは分けて戻す。
例えば一様deltaが別途認証済みなら、operator mean差<=rho B_ord+(1+rho)B_ord delta。
coherent測定のbiasへ移すときはtask固有のfactor/conventionも契約化する必要がある。

phi_uは長さだけで決まらない。母関数のproductからlabel count histogramとfirst labelに依存し得る。
fixed Lならhistogram数はmのpolynomialでも、Lとmを同時に増やすと候補数は大きくなり、
角度が衝突してO(m)種類だけになる保証はない。
online synthesisで全dictionary取得を避けても、毎trialの取得・compile費用が利益を消す可能性は残る。
controlled Q、signed rotation、parent phase Z、workspace、preparation/readoutを含むnative費用は**未検証**。

## 7. 今回の到達範囲

| 項目 | 判定 |
|---|---|
| 理想数式、local normalizer-free生成 | 一般証明＋局所prototypeで確認 |
| finite-bit proposalとfinite rational weight | bounded approximation構成、off-domain意味論で確認 |
| 有限bitでexact元mean | 主張しない。irrational coefficient/angleを近似するbiasを課金 |
| whole controlled circuit、strict phase/native synthesis | 未実装・未検証 |
| native angle取得込みの総資源改善 | 未判定 |
| 長時間・multi-step PR/QPE、DF・分子への接続 | 未認可・未評価 |
| 新規性・主method採択 | GPT/利用者判断待ち |

**mandatory STOP**。一般算術のpolynomial性を大系の実用優位と呼ばない。
実装比較を追加する必要性・scopeは、本資料を受けたGPT/利用者の別判断へ戻す。
