# G7：予算・controlled access・限定費用の結果前契約

採用された[GPT G6 review](../../research/track_b_G6_scientific_review_20261010.md) §12を技術契約へ落とす。
研究判断はGPT/利用者。G5閉鎖、G6原証拠、旧authorization・marker・STOPを保持する。
入力・精度・keyを結果後に増やさず、一束完了/失敗でmandatory STOP。

## 入力と証拠境界

Hermitian involution `Q_i²=I`、正の有理p、`sum p=1`、`R=sum p_i Q_i`、同じ有限平均
`P_m=sum_(n=0)^m (-i sigma x R)^n/n!`。geometry/basis/DF rank/splitは該当しない。
G6から既知のdevelopment形式入力のみで、held-out/分子/native全costの証拠ではない。

| 入力 | p | x | m | 用途 |
|---|---|---|---|---|
| P3_control | (3/7,4/7) | 2/5 | 3 | 三次closed formと局所生成の同値・費用対照 |
| P5_general_order | (1/5,3/10,1/2) | 5/7 | 5 | 一般次数の最小implementation economics |

ordinary、partial-return + ordinary tail、全P3 closed form + ordinary tail、full returnの4方式。
sigma=+1で費用取得。sigma=-1は同取得sequenceのactual adjointを使い、off-domain testsで符号を確認する。
同じcontrol・残りerror・readout・provider条件。CTS/全known methodsに対する最終比較は未実施。

## Uを全parent表なしに構成する

`t_l=x^l/l!`, `b_l=sqrt(t_l²+t_(l+1)²)`。G6の局所係数はa_u、s_u、
`d_u=sqrt(a_u²+s_u²)`。G6のpositive short-step envelope `d_u<=b_l p(u)`を使う。
これはx<=1、odd mに対する既存独立証明の範囲内である。

raw IID wordが隣接同一labelを含まない質量をR_lとする。reduce後の長さ分布とは異なる。
末尾iの質量をv_(l,i)とすると、`v_(1,i)=p_i`、`R_0=1`、
`R_l=sum_i v_(l,i)`、`v_(l+1,i)=p_i(R_l-v_(l,i))`。
末尾i以外からiを付加する全場合を分ければrecurrenceが従い、O(Lm)有理演算で求められる。

`U=d_empty+sum_(l=2,4,...,m-1) b_l R_l`。
各次数のenvelopeを加算して `B_new=sum_u d_u<=U<=B_ordinary=sum_l b_l`。
平方根はinteger-isqrtによる外向き有理区間。**U.hi**を予算に戻す。
root local queryとR_l recurrenceのみを予算に使い、全parent和/観測variance/信号を使わない。
G6 kernel前計算を含め有理演算O(Lm²)、bit complexityは別。
m>=3,x>0ではempty-rootの厳密減少からU<B_ordinary。
m=1は等号、L=1はU=B_newの特殊例。一般mでU<B_partialは主張しない。

## 強い三次対照の独立導出

chi=sum p_i²、mu_k=sum p_i^k、`a0=1-chi*x²/2`。
partial-returnのroot odd massは `b0=x-chi*x³/6`。
raw二次wordのdistinct部分は `t2 p_j p_k`、odd extensionは全IID child。
これは `R²=chi I+D`, `R³=chi R+R D`を代入した同じP3である。

全集約ではroot child iのmassは
`a_i=p_i*[x-x³*(2chi-p_i²)/6]`、その和は `s0=x-(2chi-mu3)x³/6`。
distinct even parent `(j,k)`には
`a=t2 p_j p_k`, `s=t3 p_j p_k(1-p_j)`, `tan(phi)=x(1-p_j)/3`。
childは `i!=j` に比例p_i、kも `k!=j` に比例p_k。したがって

`B_P3=sqrt(a0²+s0²)+sum_j t2*p_j*(1-p_j)*sqrt(1+[x(1-p_j)/3]²)`。

first-j groupだけを生成すればO(L)で三次canonical samplerを構成できる。
m=5では次数4/5のordinary paired groupを足して**同じP5**にする。
partialもP3部分+同じtail。三次だけの吸収を一般Green法の差として扱わない。

三次、L>=2、x>0では `B_new<U<B_partial<B_ordinary`。
rootでは `chi-mu3>0`、tailでは `1-p_j<1` が厳密差を与える。
独立な平方根級数展開（**m=3限定**）は

- ordinary：`1+x²-7*x⁴/72+O(x⁶)`
- partial：`1+(1-chi)x²+(-7/72+chi/18)x⁴+O(x⁶)`
- full：`1+(1-chi)x²+(-7/72-chi/6+mu3/4-mu4/36)x⁴+O(x⁶)`
- partial-fullのx⁴係数：`sum_i p_i²(1-p_i)(8-p_i)/36 > 0`。

## 有限bitと十分な試行数

K=256 root bits、H=160 probability bits、eta=rho=10^-12。
根のmidpointを一度丸め、exactな有理within-group lawを掛ける。
dyadic proposalは全positive supportでrelative errorを認証し、不足なら停止する。
fullはraw-reduced検査→local query→dyadic acceptance→child。
raw nonreducedと不受理は測定前zero。zeroをreduce後parentへ再配分しない。
productionは全31parent表を読まず、固定サイズkernelと一つのlocal packetを使う。

理想係数alphaに対し、full proposalは
`pi>= (1-eta)^(m+2) alpha/B_ordinary`、`alpha_tilde<= (1+rho)alpha`。
order一回、raw labels最大m-1回、acceptance一回、child一回を安全にm+2へ押さえる。
canonicalもgroup/labels/childの同上限を使える。
`kappa=(1+rho)²/(1-eta)^(m+2)` とすると

- full：`m2_plus=kappa*B_ordinary.hi*U.hi`
- canonical：`m2_plus=kappa*B_arm.hi²`
- range：`W_plus=(1+rho)*B_envelope.hi/(1-eta)^(m+2)`。

これはdigital variance=理想varianceという主張ではない。
exact digital acceptance/reference m2は費用照合だけに使い、Nを下げない。
すべてのnormalizerは `<=exp(x)<3`、coefficient L1 biasは `<=3rho`。
controlled rotationはstrict error<=2 epsilon_Rz、epsilon_Rz=10^-6。
全方式共通の保守的axis biasを `2*[3rho+3(1+rho)*2epsilon_Rz]` とし、
`s=1/200-common_bias>0`。exact controlled-Q oracleの条件付き保証である。
Taylor truncationは同じ有限targetの比較なので今回のaxis budgetへ混ぜない。

8 rows×2 axes、alpha_axis=1/320、familywise<=1/20。
epsilon_axis=1/200ならcomplex error<=sqrt(2)/200<1/100。
`ell>=ln(640)` を `9log2+log(5/4)` のatanh級数96項とpositive remainderで上向き認証。
`N=ceil(ell*[2m2_plus/s²+4W_plus/(3s)])`。|X-E X|<=2W_plusのBernsteinを使う。
期待量子callsはN*actual digital acceptance、**hard capはN**。二つを混同しない。
128 trials/rowの固定SHA256 bitstreamはinterface/CPU診断。独立uniform bitsの確率法則は
予算の実装契約であり、固定traceからaccuracy保証を推定しない。

## controlled-Q providerとnative Rzの接続

conditional primary providerはexact controlled Hermitian involution Q_iとactual adjoint。
その物理T/CX/1Qは非負の未知symbol。無料の分子providerを仮定した総costとはしない。
helperで `F=H CQ_i H`、`F^dag controlled-Rz(2sigma*phi) F` を実装する。
helperを0へ復元し、outer=1で `exp(-i sigma phi Q_i)`、outer=0でI。
controlled-Rzは Rz(sigma phi),CX,Rz(-sigma phi),CX。
各accepted eventはchild provider2 calls、helper H4、CX2、Rz2。
even-parent phase i^lはl=2 mod4でouter Z。全方式同じadjacent Q cancellation。
wordのoperator順序はcircuit timeでは逆順。system以外のworkspaceはouter+helper=2。
小synthetic 8×8 testはstrict位相、signed time、helper resetを確認する。全circuit compileではない。

24 positive atan(tangent) keyを全方式からdeduplicateして結果前固定。
既存pygridsynth2.0.0、fixed package/source identity、共通seed0、dps100、up_to_phase=false。
strict interval Frobenius upper<=10^-6、request epsilon/4。
negative Rzは取得positive sequence全体のactual adjointで、W scalarを保持する。
合成失敗・timeout・precision不足ならretryしない。

primary Tは `2N E[T_Rz/trial]+sum_i 2N E[Q_i calls/trial]*T_Q_i`
および共通state-preparation費用 `2N Z*T_prep` の条件付きaffine vector。
Re/Im readoutは各accepted callにつき1Q2/3。1Q/CX/workspaceは補助。
全trialの古典生成、kernel前計算、unique key cold acquisition、support reference enumerationを
別々に保存する。共通key cacheの取得をproductionの非列挙性能と呼ばない。

## Launch・停止・監査

[machine contract](../../../artifacts/track_b_g7_budget_control_preparation/2026-10-10/contract_v1.json)、
[key inventory](../../../artifacts/track_b_g7_budget_control_preparation/2026-10-10/synthesis_key_inventory_v1.json)。
§12.3の委譲された技術範囲をauthorization receiptへ正確に記録する。
新source commitをclean HEADとして固定し、source manifestとruntime/protected893pathsを照合。
fresh markerをexclusive create後、24 keysを一度だけ取得。run1/retry0。
wall1200s/CPU900s/RSS512MiB/AS1536MiB、per-key wall30s/CPU20s、output16MiB。
失敗prefixから勝敗を解釈しない。全8row完了のみcomplete status。
新5%/10%成功閾値なし。exact conditional cost differenceと未指定provider/prep costを渡す。
旧G6結果とmarkerを変更せず、新規性・総分子cost・次scienceは未認可のままSTOP。

GPTレビュー§15のreproducibility directoryはローカルに渡されていない。
その期待値/スクリプトをコピーせず、式とoff-domain形式算術を独立に導出した。
