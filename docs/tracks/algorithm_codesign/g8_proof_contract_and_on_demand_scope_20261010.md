# G8：非列挙native取得・有限provider誤差の結果前契約

採用[GPT G7 review](../../research/track_b_G7_scientific_review_20261010.md) §11.1–11.3の技術作業。
新しい独立branchで一束を固定する。G7の原結果、source、contract、authorization、marker、STOPを変更しない。
旧G5閉鎖とTrack A、rootの未commit資料も保持する。完了/失敗後mandatory STOP、研究判断はGPT/利用者。

## 固定対象とmetric

同じ `P_m(-ixR)`、`R=sum p_i Q_i`、Hermitian involutions、G7の4 production laws。
P3_control p=(3/7,4/7),x=2/5,m3とP5_general_order p=(1/5,3/10,1/2),x=5/7,m5。
既知developmentのみ。新geometry/basis/DF/split/Pauli辞書/providerは指定しない。
primaryは条件付き期待T/provider/preparation vector。CPU/cache/1Q/CXを別に記録する。
新materiality cutoffを作らず、割合だけでGO/STOPを決めない。

## 保存値・数学監査

G7 saved costsからexact fractionsで `T_full/T_base=(K_full/K_base)*(T_full/K_full)/(T_base/K_base)`
と各affine差の符号を照合する。display CSVの丸め値をsign認証に使わない。

fullの理想acceptanceはB_new/B_ordinary。digital order/label lawの上向きrelative boundを
各因子へ掛け、dyadic acceptanceは理想acceptance以下なので、
`z_digital <= min(1,(1+eta)^(m+2)*U.hi/B_ordinary.lo)`。
child lawを加算すると1。raw nonreduced rejectionは受理を増やさない。
global B_new/event表を予算へ与えず、root local queryとraw-reduced recurrenceでUを取得する。

独立uniform bits・固定lawでM回試行したaccepted count AにBernsteinを適用する。
`v=M*z_upper` とすると、`C=ceil(v+sqrt(2*v*t)+2*t/3)` は
`Pr(A>C)<=exp(-t)`。sqrtは外向き有理区間、hard cap Mも維持する。
G7レビューのcoarse z=.864、t7によるcapは**数学自己検算として別記**し、元G7のfailure/Nを変更しない。
P5 empty-root closed formもoff-domain raw-word polynomialで独立に確認する。

## finite-provider parameter契約

各providerがunitaryなstrict controlled-Q近似で、全labelについて
`||CQ_tilde-CQ||<=delta`、逆はactual adjointと仮定する。relative phaseを無視しない。
実物理providerは選ばず、このdeltaの達成費用も取得しない。
event eの誤差はhelperを含むjoint operatorのtelescopingから
`error_e<=2*epsilon_Rz+sum_i n_(e,i)*delta_i`。
target-coefficientで重み付けした `sum_e alpha_tilde_e*error_e` をbiasへ戻す。
proposalで重み付けしたexpected query countをbias上界へ代入しない。

共通normalizer<=exp(x)<3と `n_e<=m+1` により、各axisの保守的biasは
`b_delta=2*[3rho+3(1+rho)*(2epsilon_Rz+(m+1)delta)]`。
`s_delta=1/200-b_delta>0` を要求。
許容domainは `0<=delta < [(1/200-b_0)/(6(1+rho)(m+1))]`。
このsymbolic rangeと、固定一つの仮想値 `delta=epsilon_Rz=10^-6` を保存する。
新precision gridや都合のよいphysical providerを探すものではない。
helper leakage・controlled relative phaseもstrict joint errorに含まれる。
G7の理想providerでのaffine優位を、finite-error provider実証へ遡及変更しない。

## 新failure配分・予算

16 axesの推定failureは `16*(49/16000)=0.049`、8 rowsのresource failureは
`8*(1/8000)=0.001`、合計0.05。canonical rowsの余ったresource配分も結果後に再利用しない。
`t=9`、positive rational Taylor部分和で `exp(9)>8000` を検査する。
Re/Im epsilon_axis=1/200、complex<1/100を保持する。
`ell>=log(2/alpha_axis)` はatanh96項の上下有理区間から上向き取得。

G7同様のm2/range上界から `N=ceil(ell*(2m2_plus/s_delta²+4W_plus/(3s_delta)))`。
fullはU、canonicalはsmall group normalizerを使う。true signal/reference momentは使わない。
各rowについてM=2Nとz_upperからaccepted capを固定する。
capを超えるrunはprefixを採点せずabort。このcapはaccepted回数の保証であり、総T上界ではない。
native取得失敗・cache上限・技術的timeoutは別のtechnical abort。
このprototypeでその失敗確率が0だという一般保証は得られていない。

## On-demand経路と対称cache

productionへ渡すものはp,x,m,arm,有限bit/error/capのみ。
全parent表、global B_new、事前native key表、G7保存sequenceを注入しない。
live生成したeventがpositiveの場合のみsymbolic tangentからpositive Rzをstrict合成する。
pre-quantum zeroではcache/provider/readoutを要求しない。
negative primitiveはW scalarを含む取得positive列全体のactual adjoint。
provider/control mappingはG7と同一。literal conditional native descriptionを保存し、physical CQは未実装と明示する。

- row cache：全方式LRU8 entries/128KiB。
- acquisition memo：全方式共通、32 entries/1MiB。whole-bundle開始時は空。
- memoはlive requestからだけ増える。native合成は一tangent一回、memo eviction/resynthesisは行わない。
- rowのeviction missは共有memoから再取得できる。未知keyでcap到達ならSTOP。
- cold row cacheは空、warmは同じ128固定bitstreamを再生してrow cacheを保持する。
- isolated cold chargeは各rowで実際に要求したunique key取得時間の合計。共有memo hitを冷取得実測と呼ばない。

全方式で同じcache・backend・epsilon・miss上限。全方式128 cold+128 warm trials、計2048。
seedはG7と同じ `G7-development-bitstream-v1`。bitstreamはCPU/接口診断のみ。
この短いtraceからacceptance、variance、accuracy、総期待Tを頻度推定しない。
古典query/root処理、tangent key作成、miss取得、provider description、cache bytesを分ける。
128trace serialization/hashはrow測定にも含まれ、timerの短時間測定をgeneral throughputと呼ばない。
local query内部のGF/根計算/ratio構成は一つの計測区間、key文字列構成は別区間として定義する。

同runtime pygridsynth2.0.0・common seed0・dps100・up_to_phase=false。
strict interval Frobenius upper<=10^-6、request epsilon/4。backend探索、precision変更、retryなし。
global/row cacheのbyte capはserialized payload量で、Python object heap全体ではない。
実プロセスRSS/AS上限も併記する。

## 独立small-support/cost-aware補助診断

全production完了後にだけG7 saved result/cost tableをparseし、別moduleでsmall supportを列挙する。
新production cacheへbackfillしない。独立referenceの費用をproduction CPUへ混ぜない。
G7保存per-trial費用を新Nへ再予算化する場合は「保存費用を用いた条件付き感度」として保存する。
新G8 traceが全角度を取得したことや、全量子費用を実測したことにはしない。

§11.1のcost-aware診断は、全4 representationに同じ一つのoracle proposalを与える。
固定positive saved Rz cost `C_e=2*T_count` に対して
`q_e proportional to alpha_tilde_e/sqrt(C_e)` を256bit外向きsqrt/160bit dyadic lawへする。
元の各alpha_tilde、angle、eventを保持し、weightをalpha_tilde/qへ変える。
exact rational m2/rangeと同じs_delta/alpha_axisからNと期待Rz-only Tを算出する。
新RNG/synthesis/gridを実行しない。global表を要するI2 oracleであり、非列挙runtimeとは異なる。
production lawをこのproposalへ差し替えず、G7のprimary classificationも変更しない。

Cauchyから `(sum alpha_tilde²/q)*(sum q*C)>=(sum alpha_tilde*sqrt(C))²`。
固定Bernstein policyの二axis Rz Tには
`4*log(2/alpha_axis)*(sum alpha_tilde*sqrt(C))²/s_delta²` というleading-term下界がある。
logとsqrtの**下端**を使う保守的下界と、候補の有限N評価を別保存する。
これはrepresentation/原係数・price固定の政策下界であり、physical shot下界・全method最適性ではない。
provider/preparation Tを0にしたRz-only endpointでの診断で、一般総native優位を判定しない。

## Source・cap・停止

新source commit S、clean HEAD、固定runtime/source manifest/protected918pathsを照合する。
fresh G8 markerをexclusive createして一束だけ実行。旧G7 runner/markerは使用しない。
wall1200s、CPU900s、RSS512MiB、AS1536MiB、per-key30s/20s、32 synthesis calls、output16MiB。
合成miss失敗/cap/certificate不足ではretryしない。prefixを最終比較へ使わない。

成功statusは `G8_ON_DEMAND_PATH_AND_CONDITIONAL_PROVIDER_BUDGET_COMPLETE`。
意味は「固定traceのpathと条件付き数式の一束を完了」。全supportの合成termination、
全N quantum execution、physical-provider δ達成、総T削減、新規性成立を認証するstatusではない。
失敗は `G8_TECHNICAL_INCONCLUSIVE`。いずれもSTOPし、次の科学scopeはGPTへ戻す。

レビュー§15のselfcheck directoryはローカルに未提供。独立式/算術を使い、期待値コピーなし。
