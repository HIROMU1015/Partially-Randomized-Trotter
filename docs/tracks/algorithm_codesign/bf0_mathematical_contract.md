# BF-0 mathematical contract — Track B-F

日付: 2026-10-04  
status: `PROCEED_BF0_DESIGN_AND_NOVELTY_AUDIT` / `DRAFT_FOR_REVIEW`  
実行authorization: **なし**。本書は数学上の比較契約の提案であり、algorithm採択・BF-1実行許可ではない。

関連: [claim監査](bf0_prior_art_claim_matrix.md)、[BF-1予算付き提案](bf1_minimal_pilot_proposal.md)、[BF-0外部レビュー依頼](bf0_external_review_request_20261004.md)。参照元、commit、未commit資料のidentityはclaim監査のprovenance表に記録する。

## 1. 狭いRQと証拠境界

固定したDF表現、split、native kernel、有限RTE、coherent-signal taskの下で、**同じ五stage四次family、同じ探索自由度、同じ評価予算**を通常PF設計と部分ランダム化task設計へ与えたとき、randomized tailの有限化を目的関数へ戻すことで、精度を満たす資源選択に意味のある差が生じるか。

係数が異なること、stage数が少ないこと、高次PFを適用できることだけではmethod deltaにならない。今回の五stage compositionと比例allocationは既知の構成である。候補差分はそのtask固有の設計問題に限り、最終的にはcompiled resourceによる検証を必要とする。BF-1のcomponent-action proxyだけでその最終claimを成立させない。

Track AのM1/M2はsource-bound local evidenceである。H4 1.00 Åはdevelopment条件、H4 1.30 Åは開封・採点済み条件であり、Bのheld-out、blind validation、independent replicationではない。本契約はAのstatus・contract・artifact・runtime/cacheを変更または再利用しない。Bの独立held-outは未定である。

## 2. Native ideal kernelと操作順

固定したidentity抽出規約の下で

\[
H=cI+\sum_{\ell=0}^{L_D-1}D_\ell+R,\qquad
R=\lambda_R\bar R,\quad\|\bar R\|\le1
\]

とする。DF fragmentの順番、基底、rank policy、tailの係数表、\(\lambda_R\)の定義とscreening thresholdは入力contractに含める。\(D_\ell\)はnativeに実装する各DF blockであり、\(e^{-it\sum D_\ell}\)を無料のexact oracleに置き換えない。

操作を**時系列の適用list**で定義する。list \((F_1,\ldots,F_n)\)の作用素は\(F_n\cdots F_1\)である。native second-order kernel \(S_2(h)\)のlistは

\[
e^{-ihD_0/2},\ldots,e^{-ihD_{L_D-1}/2},
e^{-ihR},
e^{-ihD_{L_D-1}/2},\ldots,e^{-ihD_0/2}.
\]

scalar phase \(e^{-ich}\)も保持する。controlled evolutionは\(\operatorname{diag}(I,U)\)であり、system上のglobal phaseはcontrolに対するrelative phaseとなるため、捨てない。抽出済みtail identityとの二重計上もしない。

外側time stepは\(h=T/q\)。一step内で下記の五つの\(S_2(w_jh)\)を時系列に適用し、同一stepを\(q\)回繰り返す。\(q\)はouter-step数、\(R_{\mathrm{bud}}\)は一outer-step当たりのRTE microstep総予算であり、Hamiltonianのtail \(R\)と区別する。

## 3. 五stage四次familyとdomain

\[
\boldsymbol w=(a,b,c_0,b,a),\qquad c_0=1-2(a+b),
\]

\[
2a+2b+c_0=1,\qquad 2a^3+2b^3+c_0^3=0.
\]

これはself-adjoint second-order kernelの既知の四次composition条件であり、新しいorder条件ではない。理想的なexact exponentialsとexact係数について、固定\(T\)で通常のglobal PF errorは\(O(q^{-4})\)である。有限RTEを挿入した全algorithmが同じ次数を持つとは主張しない。

review用domain提案は実数係数、\(\max_j|w_j|\le2\)、上の等式制約を満たすすべての枝とする。七stage・複素係数・processor・kernel自体の再設計は含めない。exactなゼロstageは許す。したがって三stage Yoshida compositionもゼロを挿入した境界対照として含む。

domainの一変数表示は\(s=a+b,d=a-b\)として

\[
d^2=\frac{15s^3-24s^2+12s-2}{3s},\quad
a=\frac{s+d}{2},\quad b=\frac{s-d}{2},\quad c_0=1-2s.
\]

\(s\ne0\)、右辺が非負、\(d\)の両符号、全係数capを満たす枝を残す。\(c_0\)のcapから\(-1/2\le s\le3/2\)だが、この区間全体がfeasibleではない。今回、枝端点の数値探索や係数最適化は行っていない。

future implementationはexact定義と実際に使用した数値係数を両方保存する。\(\eta_1=|\sum w_j-1|\)、\(\eta_3=|\sum w_j^3|\)について少なくとも\(10^{-12}\)以下というscreenを提案するが、それだけで四次精度やsignal biasを保証しない。係数丸め、operator評価、数値誤差のsignalへの寄与を別途\(u_{\rm num}\)へ計上できなければ実行gateを閉じない。制約違反をpenaltyで許容してpositive resultを作らない。

## 4. Exact fusion → finite insertion → circuit simplification

主案 **Construction F: fuse then randomize** の手順は次の順に固定する。

1. exact PFの時系列listを展開する。
2. exactなzero-time factorを除き、隣接して同じgeneratorを持つexponentialだけを時間加算で融合する。消去によって新しく隣接した同一generatorにも同じ規則を適用する。
3. 残った各tail occurrenceのsigned time \(t_j\)に独立な有限RTEを挿入する。
4. 個々のsampled circuitに、controlled phaseを含めてexactなgate cancellation等を適用する。

同じgeneratorであることは入力の同一identityで判定する。commutatorが小さいという理由で非隣接factorを交換しない。丸められた近ゼロをexact zero扱いしない。係数のexact定義に基づくtime ledgerと浮動小数点へlowerしたtimeを区別し、exact取消の根拠を保存する。\(D_0\)のstage境界融合と、\(L_D=0\)やexact相殺などで露出するtail融合を区別する。後者ではtail occurrence数・normalization・allocationも変わる。

**Construction S: randomize each stage** は、各stageのtailを先に有限RTE化し、その後sampled circuitにexact simplificationだけを許す別algorithmである。一般に

\[
P_{K+1}(-it_1R)P_{K+1}(-it_2R)
\ne P_{K+1}[-i(t_1+t_2)R].
\]

microstep数を含めた有限作用素でも同じ注意が必要である。FとSの差をcompiler optimizationと呼ばない。BF-1はFを主構成とし、Sは数学上の対照として保持する。F/S比較runを自動追加しない。

通常の\(L_D>0\)・非zero stageではtail間にDF blockが残る。\(L_D=0\)、zero stage、exact境界相殺等では、全\(q\)stepを含むlistを先に簡約する必要がある。この退化caseへ、一stepの\(B\)を機械的に\(q\)乗する式を流用しない。BF-1の固定入力ではstationaryな一step templateを使えることを実装前reviewで確認する。

## 5. 現行finite-RTEの意味論

現行sourceの\(K\)は**保持する最大の偶数event次数**である。\(K=2\)の補正後first momentはTaylor次数3まで、\(K=4\)は次数5までを含む。genericな\(P_K\)記法とのoff-by-oneを避ける。

tail occurrence \(j\)のsigned time \(t_j\)、microstep数\(r_j\ge1\)について

\[
\tau_j=\lambda_Rt_j/r_j,\quad
A_k(\tau)=\frac{|\tau|^k}{k!}
\sqrt{1+\frac{\tau^2}{(k+1)^2}},\quad
B_K(\tau)=\sum_{k=0,2,\ldots,K}A_k(\tau),
\]

\[
p_k(\tau)=A_k(\tau)/B_K(\tau),\qquad
P_{K+1}(X)=\sum_{n=0}^{K+1}X^n/n!.
\]

component sampling、event phase \((-1)^{k/2}\)、signed rotation \(\arctan[\tau/(k+1)]\)も現行規約を保存する。negative stageでは\(\tau\)の符号を失わない。\(B_K\)だけが偶関数である。

補正後occurrence meanと物理的なunitary sampleのmeanは

\[
M_j=[P_{K+1}(-it_jR/r_j)]^{r_j},\qquad
\mathbb E U_j=M_j/B_K(\tau_j)^{r_j}.
\]

全tail sampleはoccurrence/microstep間で独立とし、exact DF factorsと\(M_j\)を同じ時系列で合成した作用素を\(M\)とする。全体の

\[
\log B=\sum_{j\in\text{all occurrences}}r_j\log B_K(\tau_j),
\qquad\mathbb E U_{\omega}=M/B
\]

を用いる。stationary templateが確認された場合に限り、一step内の和を\(q\)倍できる。異なる\(t_j,r_j,K_j\)に共通の\(B\)を仮定しない。\(B=1\)または\(e^{\tau^2}\)という上界を実際のfinite normalizationへ置換しない。

sourceの式から、\(K\ge2\)では\(\log B_K(\tau)=\tau^2+O(\tau^4)\)、\(K=0\)では\(\tau^2/2+O(\tau^4)\)となる。したがってBF-1のleading normalization対照は係数1を使う。この展開とabsolute-tail-time model自体は新規claimではない。

\(\|\bar R\|\le1\)の下で、microstepのTaylor remainderには

\[
e_j= e^{|\tau_j|}\frac{|\tau_j|^{K+2}}{(K+2)!},\qquad
E_{\rm tail}=\exp\left(\sum_jr_j\log(1+e_j)\right)-1
\]

という保守的な全composition error boundを使える。DF factorsはunitaryとして扱う。これは理想PFに対する有限tail置換誤差であり、PF bias、統計誤差とは別である。固定\(r_j\)でK=2を使うと一般にtail誤差はglobal \(O(q^{-3})\)となり得る。理想compositionの「四次」をそのまま有限algorithmのラベルにしない。

## 6. Allocationの自由度を制限する

F簡約後の一step内の非zero tail timeを\(t_j=h\gamma_j\)、その数を\(n\)、\(\Gamma=\sum_j|\gamma_j|\)とする。\(R_{\mathrm{bud}}\ge n\)でなければinfeasibleである。

既知の連続allocation \(r_j\propto|\gamma_j|\)を基準に、BF-1のinteger rule案は

\[
x_j=(R_{\mathrm{bud}}-n)|\gamma_j|/\Gamma,\qquad
r_j=1+\lfloor x_j\rfloor
\]

から始め、残り\(R_{\mathrm{bud}}-\sum r_j\)を\(x_j\)のfractional part降順に一つずつ配る。tieはoccurrenceの時系列index順とする。zero tailは削除して\(r_j=0\)。\(\Gamma=0\)ならRTEを用いず、\(B=1\)とする。これはlower boundを守る再現可能なrounding ruleであり、integer-optimal allocationとは主張しない。

共通K、同じ一step allocationを全\(q\)stepに適用する。各\(r_j,K_j\)の独立探索は行わない。退化して全step融合が必要になる入力はBF-1対象外とし、契約変更reviewを必要とする。

連続allocationでのleading log-normalizationは

\[
\log B\approx\frac{\lambda_R^2T^2}{qR_{\mathrm{bud}}}\Gamma^2.
\]

従って\(\Gamma\)だけを小さくする設計や、この既知式とPF errorのtradeoffを再表示するだけでは新規性を満たさない。finite polynomial bias、有限\(B_K\)、integer allocation、native DF workを共通taskで調べる必要がある。

## 7. Common task、accuracy、resource

事前固定したnormalized state \(\psi\)に対する

\[
z=\langle\psi|e^{-iTH}|\psi\rangle,\qquad
\nu=\langle\psi|M|\psi\rangle
\]

のcomplex coherent signalをtaskとする。Hadamard testの各axis \(a\in\{\Re,\Im\}\)で\(Y_a\in\{-1,+1\}\)を観測し、estimatorは\(B\overline Y_a\)。そのmeanは\(\nu_a\)である。noise-free algorithmic approximationの比較であり、QPE/RPEのenergy/statistical errorと混ぜない。

total signal tolerance \(\epsilon_{\rm sig}\)、failure probability \(\alpha\)について、axis tolerance \(\epsilon_a=\epsilon_{\rm sig}/\sqrt2\)、\(\alpha_a=\alpha/2\)を固定する。\(b_a=|\nu_a-z_a|\)、数値誤差の保守値\(u_a\)から

\[
s_a=\epsilon_a-b_a-u_a>0,\qquad
N_a=\left\lceil\frac{2B^2}{s_a^2}\log\frac{2}{\alpha_a}\right\rceil
\]

を同じHoeffding規則で割り当てる。全axisのunion boundでcomplex error \(\le\epsilon_{\rm sig}\)を確率\(\ge1-\alpha\)で目指す。signal-amplitudeのoracle varianceから都合よくshotsを減らさない。feasibility marginが数値誤差以内なら`UNRESOLVED_NUMERICAL_MARGIN`である。

最終的に閉じたい資源目的は

\[
G_{\rm compiled}=\sum_aN_a\,\mathbb E_\omega C_a^{\rm compiled}(\omega).
\]

BF-1ではまず\(G_{\rm action}\)という**native component-action proxy**を使う提案とする。exact DF exponential一つを一action、RTE microstepを\(\sum_kp_k(k+1)\)actionsとして数える。簡約後のlistからdeterministic actionsを数え、scalar/control/wrapper cost、state preparation、rotation synthesis、fault toleranceは別欄で未評価とする。proxyをRZ数、compiled cost、最終総costと呼ばない。異なるDF blockやtail componentの重さを同一とした近似の限界を残す。

## 8. 設計情報の階層とoracle leakage guard

| 階層 | 設計時に許す情報 | 成果の呼び方 |
|---|---|---|
| I0 formula-only | exact order条件、係数、stage list、\(\Gamma\)、記号的normalization式、公開された一般norm bound | formula-level設計 |
| I1 development-Hamiltonian accessible | 入力DF係数から定義した\(\lambda_R\)、fragment/basis構造、state-independent commutator/error surrogate、analytic work | instance-aware設計 |
| I2 oracle-assisted development | full matrix/eigensystem、exact ground state/energy、exact coherent signal、exact finite bias | oracle benchmark・機構確認 |

特定入力から評価した\(\lambda_R\)やcommutatorはI1であり、formula-onlyに混入させない。BF-1のO/L/F全armは同じI2 accessを持つことを表示する。I0/I1-only selectorの実装・保証がなければ`NOT_ESTABLISHED`とする。

I2で探索した係数、feasibility、threshold、seed、surrogate tuningをI0/I1の設計情報へ逆流させない。後日のtest入力や既に開封したM2から係数を調整しない。I2最適化後のtransferが良くても、大系で使えるdesign algorithm、blind validation、独立再現の証拠とは扱わない。I0/I1版を主張するには、情報集合・bound・trainingとevaluationの分離を別reviewで固定する必要がある。

## 9. 現行sourceが支持する範囲と未閉鎖事項

`rte.py`にはheterogeneous occurrenceのmean/normalization compositionがある。一方、`DFPartialS2StepRequest`は`pf_label='2nd'`を要求する。repeated-S2 builderの既存境界融合は、同じsecond-order kernelの繰返しについての能力であり、任意のsigned五stage compositionのDF controlled wrapper検証ではない。

`product_formula.py`の係数listはcenter-first half-listである。現行`morales_8th_list`はMorales v1由来の17-stage formulaで、v3 Table Iの21-stage formulaとは別identityである。`new_4th_m2_list`の八桁係数をexact order条件成立とみなさない。

BF-1前に閉じるべき事項は、入力/stateのidentity、係数精度と誤差guard、全枝探索の列挙仕様、Fの退化case排除、signed-time/controlled-phaseのadapter仕様、ordinary/leading/finite objectiveの実装照合、published baselineのidentityとfairness、数値budgetと実行authorizationである。現時点では文書上の定義であり、これらを実装・数値検証済みとは扱わない。

BF-0報告後はSTOPし、三文書のreviewへ戻る。science runner、signal評価、trajectory生成、compile、GPU操作、共通API変更、commit/pushを本書から認可しない。

## 10. 2026-10-04利用者reviewに基づく進行順序

現在の中心仮説を保ち、BF-0三文書の外部review → 必要な最小修正 → BF-1事前登録・authorization → 一回限りのBF-1 → mandatory STOP → 研究BのRQ・新規性・着地点の全面再評価、の順に進む。BF-0 reviewで重複、公平性、oracle情報、意味論、自由度、判定規則の重大な問題が判明した場合は、BF-1を実行せず修正または縮小・停止する。

問題が閉じた場合、BF-1は仮説にmethod deltaとdecision relevanceがあるかを判別する最小pilotである。最良PFの探索や、研究方針の追加の全面再設計を先に行う段階とはしない。一回限りの実行のscopeと上限は結果前に固定し、成功・negative・inconclusive・中断のいずれでも終了後に停止する。BF-2、B-S、held-out、oracle-free design ruleへは、結果後の全面再評価なしに進まない。

この進行方針の承認と、外部reviewの通過、実行authorizationは別である。現在の三文書はreview用草案のままで、BF-1実行を認可しない。
