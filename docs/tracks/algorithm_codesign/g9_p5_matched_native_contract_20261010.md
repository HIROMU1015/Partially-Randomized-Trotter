# Track B G9：P5特殊化・matched-native結果前契約

利用者がGPT G8 reviewを採用。§14.2–14.6の一束の数学/source/native確認を実施する。
基点はG8結果e4b410746aadcf03c955b5e961c672c04b220de5、別branch/worktree
`track-b-g9-p5-matched-native-20261010`。原G8や過去のSTOP/marker/sourceは変更しない。
完了・技術失敗・数学反例・scope mismatchのいずれでもmandatory STOPし、研究判断をGPTへ返す。

## 固定対象と比較

同じknown development `p=(1/5,3/10,1/2),x=5/7,m=5`、full operator P5。
3 system qubits、`Q0=Z0; V1=R_XX01(pi/4); V2=R_XX12(pi/4) R_ZZ01(pi/4); Qi=Vi† Zi Vi`。
右から作用する。新synthetic providerのforward接続であり、角度/p/x/mをheld-outとは呼ばない。
分子/DF代表性やPauli取得困難性のcontextではない。

primary directはordinary / partial-return+tail / closed-P3+tail / local full / closed-P5 full / matched CTSの6方式。
既知5 return方式のgeneric helperは接続用診断、5 row。計11 row/22 axes。
特殊化は同研究familyのfast path。旧G5のtoy/dictionary主線を再開したものではない。

## P5の独立再導出

IID質量をc_n(u)とする。非隣接一致word uの長さl+2 raw wordは、
隣接同labelの一pairをl+1個のslotへ挿入して得る。三重同labelの重複だけを各文字位置で一度引くと
`c_(l+2)(u)=p(u)[(l+1)chi-sum_t p_(i_t)^2]`。
これはfree-wordの順序を保つscalar数式で、演算子の交換を使わない。

左乗算walk再帰から`c2(empty)=chi`、`c3(i)=pi(2chi-pi²)`。
`c4(empty)=sum_i pi*c3(i)=2chi²-mu4`。
`c5(i)=pi*c4(empty)+sum_(j!=i) pj*c4(j,i)`に上一return式を代入すると
`pi[5chi²-4chi*pi²+2pi⁴-2mu4]`。Green evaluatorをこの導出には使わない。

`tn=x^n/n!`。
rootのa0/各child、長さ2(j,k)のA/S/child、長さ4のleading scalarは採用review§10の式となる。
root1、ordered pair L(L-1)、長さ4のfirst-label Lで**L²+1以下の群**。
初期化はpairのSをmu3からO(1)で求め、全(j,k,i) child表を作らない。
completion massは(previous label,remaining length<=3)のDP。これはraw-reduced列用で、一般stack returnの近似ではない。
群構築/rootの算術operationはO(L²)、bit費用は別。sample時のlocal conditionalはO(L)。

a0>=1-t2>0、A>=t2-3t4>0、各level2 child括弧>=t3-4t5>0。
root childは`pi[t1-2t3-2t5]`で下から抑えられ0<x<=1で正。
一labelでは非root群を省略できる。
root/group選択・word/child選択をdyadic160、rootを256bit outwardにし、alpha/qをexact rationalで補正する。
係数roundingの平均operator誤差は群ごとのrelative rhoをL1で集計。同じ理想ensembleであるがG8のrational係数とbyte-exactな同一性は要求しない。

独立first-adjacent deletion oracleで既知P5の31 parents/63 events、10 groupsを確認した。
off-domain L1/L2/L3/L4 fixture、Green局所式との同じ理想係数、DP、phase/direct/provider/importを23 focused testsで確認。
初期prototypeのInterval arity/import/inventory数誤りは取得・marker作成前に修正し、準備artifactへ記録した。
paid native取得とscience cost outcomeはsource固定前0。

## 明示providerとnative公平性

Pauliの独立symbolic取得はQ(sqrt(2))で正確：
`Q0=ZII`、`Q1=(IZI+XYI)/sqrt(2)`、
`Q2=IIZ/sqrt(2)+(IXY-ZYY)/2`。
source testはこれをVのanalytic matrixからの独立referenceと照合する。
Vのpi/4 rotationはparity ladder/Tで実装し、Vのscalar位相はV ... V†内だけで相殺する。
random eventのi^phaseはouter Zとして保持。negative Rzは取得sequence全体のactual adjoint、W scalarも保持。

各方式に同じdirect `V† controlled-P rotation V`を許す。
共通compilerはCZ->H CX Hの後にadjacent inverse cancellationのみ。T/T†/CX/1Qはliteral primitiveの加算。
provider Tは実装から数え、無料価格に置き換えない。全回路最適性や最良compilerとはしない。
workspaceはdirect外側ancilla1、helper診断2。
CTS real correctionのT=0も保持し、全caseをfull operator平均で比較する。簡単な状態signalへtargetを縮小しない。

matched CTSは[Peetz–Smart–Narang Theorem1](https://www.nature.com/articles/s41534-025-01168-w)と
[Supplementary Note5 Eq13–15](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41534-025-01168-w/MediaObjects/41534_2025_1168_MOESM1_ESM.pdf)のfinite P5特殊化。
実・虚Taylor部分をPauli収集し、even correctionはidentityも含め別event、odd部分は一つの角度にpairする。
現small providerで取得できるI1を全方式に公開する。channel equalityでfirst operator meanを代用しない。

## 誤差と予算

同じnative Rz strict epsilon=10^-6、rho=eta=10^-12、H160/K256。
providerはこのexact Clifford+T modelでdelta=0。G8の仮想delta=10^-6達成実験とは呼ばない。
全representationに`B<8`（Pauli l1(R)<=2,exp(2)<8）を使い、coefficient mean error<=8rhoを確認。
CTSのfield/angle/norm丸め誤差も同じ上限へ戻す。
共通bias `2[8rho+8(1+rho)*2epsilon_Rz]`、Re/Im各epsilon=1/200。
strict native event error<=2epsilon_Rzはcomposition上界。float matrix 1e-10はphase/順序diagnosticでありconfidenceの認証予算ではない。

22 axes alpha=49/22000で0.049、11 resource rows beta=1/11000で0.001、合計0.05。
logはoutward、accepted tail t=10（exp(10)>11000）、cap<=hard M=2N。
local fullはordinary envelopeとUを使用し、reference B_newやmatrix signalでNを縮めない。
closed P5はO(L²) normalizerを利用するcanonical fast path。CTSはcollected I1 normを利用。

primaryは二axis native T intercept + K*T_prep/readout、K=期待accepted calls。
prep/readout単価は全方式共通非負parameter、無料状態準備を勝因としない。
CX、1Q、workspace、classical構築/Pauli取得/reference費用は別座標。
IS追加探索はしない。任意総cost-aware proposalに対する優位性・最適性は主張しない。
G8 oracle scalar/混合下界をG9のnative最適性証拠へ転用せず、G8原分類も変更しない。

## One-shot・取得・資源上限

全6方式に必要な19 unique keysを結果前inventoryへ固定。
18は旧G7のangle/epsilon/tool/strict-phase/sequence hashを照合して再利用し、CTSの1 keyだけ新規取得。
primitive再利用は旧wrapper costの流用ではない。全eventへこの明示providerの共通compilerを適用して再会計する。
共有key容量32/1MiB、未知key/capでabort、再合成0、retry0。
全small event表はoperator/reference会計用。local/closed generatorの引数へ注入しない。

既存isolated SP05 runtime/pygridsynth2.0.0、同backend/seed/options/precision。
一process、wall1200/CPU900 s、RSS512MiB/AS1536MiB、per-key wall30/CPU20 s、
new synthesis call1、sequence20,000 chars、output16MiB、N/axis<=10^8。
source clean/full HEAD/critical hashes/runtime/保護ledger確認後、exclusive one-shot markerを作る。
source commitをremoteへ固定後、一回だけrunnerを呼ぶ。失敗・cap・guard違反はprefixを最終科学比較に使わない。

新p/x/m/provider/grid/seed/precision/backend、LP/v4、分子/DF/NPZ/GPU、実量子shots/trajectoryは禁止。
全作業完了/失敗後に保存値・provenanceだけを照合し、必要資料をcommit/pushしてmandatory STOP。
次の科学実行・新規性・主method・論文scopeはGPT/利用者へ戻す。
