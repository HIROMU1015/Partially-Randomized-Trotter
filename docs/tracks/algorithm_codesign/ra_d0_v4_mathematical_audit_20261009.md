# RA-D0 v4 独立数学監査

2026-10-09 JST。数学判定は **`V4_MATH_AUDIT_PASS`**、最終総合分類は
**`V4_EXACT_SOLVER_FEASIBILITY_UNVERIFIED`**。現環境でexact LP backendの実行を確認できていない。
主要な十分条件は以下の前提の下で成立し、人工有理数207件は207 PASS / 0 FAIL。
これはproduction実装や登録最適化の認可ではない。資料公開後mandatory STOP。

## 対象・入力・独立性

branch：`track-b-ra-d0-v4-mathematical-audit-20261009`。基点：T0.2
`d3a7cbb239487ddedf44699378f6c182c1fe5993`。
設計入力は `06575a3bc9d2354b829e0e9a6c21ad1512f77909` の
`ra_d0_v4_numerical_design_20261008.md`（SHA256 `5af8430d2d23adb0db0f9be87ecd1b851037984b7e9c1af9cb53f3218a1aa1a7`）。
[設計snapshot](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/inputs/gpt_design_20261008.md)と
[今回の監査指示](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/inputs/user_audit_instruction_20261009.md)を保存した。

数学は設計式を前提とせず下記の導出で検査した。
[監査専用script](../../../scripts/tracks/algorithm_codesign/check_ra_d0_v4_mathematical_audit.py)はstdlibのみを使い、
旧source/kernelsをimportしない。人工lawの認証は丸め前のXi/Gamma判定から独立に、
丸め後のq/y/zをinterval endpointsへ直接代入して計算する。
GPTの自己検算240件を今回の独立207件へ含めていない。

固定tableの読み取りは、saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、
x={1/8,1/4}、登録precision={1e-3,1e-4,1e-6}の静的preflightに限定した。
分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。
新しいscience input、angle、synthesis、operator/matrix/signal評価は行っていない。

## A. 定理の前提と固定tableの照合

前提は次のとおり。

- 有限個のgroupと非空のprecision集合、固定のcanonical ID順を使う。
- cのintervalは正かつ順序付きで、cbar>0。D endpointsは正しく順序付けされたenclosureである。
- 同group内のprecisionでD intervals、理想event、条件付き内部sampling、phaseが同一である。
- u>=0、exact normalizer、B2またはB3の構造等式が成立する。丸め時に再正規化や負成分clippingをしない。
- biasとcost係数は非負。workspace capを超えるvariantをactiveにしない。
- N=2^60、y>=1/N。B2でz>=0、sum z=yとし、Ymax rad<=2/Nをpreflightする。
- n、ell_upper、kappa_upper、error/accounting/capsは固定値である。結果を見た調整はしない。

固定tableのx二条件を独立に照合した。各21 columns、B2のalias展開後8 groups / 24 entries、
B3の7 groups / 21 entriesである。全三representationについてsum v=tがexactに一致した。
norm intervalは `c_minus^2 <= a^2+b^2 <= c_plus^2` と正値性を満たし、D enclosuresとの対応も一致した。
precision間のD・理想event・内部確率・phaseは同一、cost/dは非負、workspace除外は0。
Ymax rad<=2/Nは全groupで成立した。B2 alias分の余裕を落とさず、各GammaのB2/B3最大を使える。
全9 profiles/xのnorm metadataについても、precisionに依存しない同一性を48件照合した。

保存R1のordinaryとPTSC-K0の共有O0について、x二条件×precision三条件×sigma二符号の
12組を照合した。条件付きevent・phase・native IR/cost/error・workspaceが同一である。
B0のimplemented_coefficientはrepresentation normalizationに依存するためalias同一性の条件に使わない。
[静的照合の全値](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/rounding_bound_audit_v1.json)を参照する。

## B. 構造等式とmidpoint mean bound

B2ではU_rg=sum_p u_rgp=z_r、q*_rgp=cbar_rg u_rgpなので、
group mass M_rg=cbar_rg z_rがexactに成立する。
各rでsum_g v_rg=tなら、

    sum_r,g v_rg U_rg = sum_r z_r t = y t.

したがってdegree matchingは自動的に成立する。sum v=tが破れた入力にはこの推論を使えない。
固定tableではこの前提を確認済みで、追加degree制約は冗長である。
B3ではsum_g v_g U_g=ytを直接課す。precision配分に加えgroupごとのdegree自由度が残るが、
元の許容誤差付き数値クラス全体を表すparameterizationとは主張しない。

xiは元sourceと同じ、各degreeについてinterval残差両端の絶対値最大を加算した量とする。
構造等式を引けば、各端点の残差は

    sum_g U_g (cbar_g D_endpoint_kg - v_kg).

U_g>=0でtriangle inequalityを適用し、各degreeを加算すると、

    xi(q*,y) <= sum_g U_g sum_k max(|cbar D^- - v|, |cbar D^+ - v|) = Xi(u).

端点の向き・absolute value・degree和を含むinterval certificateのboundであり、
cbarを真のnormと見なしたものではない。真のoperator equalityを主張せず、元delta_numへ残差を戻す。
inactive z=0なら全対応u=0。cbar>0の下でM=0ならU=0でもある。
shareの除算はM>0のgroupでだけ行う。y>=1/Nは線形制約で、丸め後もy^N>=1/Nを保証する。

## C. 階層LRMの構成とsupport

正規化済みMについて、s_g=N M_g、f_g=floor(s_g)とする。
left=N-sum fは整数で、sumのfractional partsに等しい。
fractional partが大きい順にleft個へ1を足すため、K_gは非負整数、sum K=Nであり、

    Delta M_g = K_g/N-M_g,  |Delta M_g| <= 1/N.

M=0のfractional partは0。left>0なら十分な正fractional partsがあるため、zero groupを選ばない。
left=0なら追加counts自体がない。tieは固定ID順で処理できる。

M>0ではpi_gp=q*_gp/M_g=u_gp/U_g、sum pi=1。
内部のLRMも同じ証明で、

    epsilon_gp = K_gp-K_g pi_gp,  |epsilon_gp| <= 1,
    q^N_gp-q*_gp = pi_gp Delta M_g + epsilon_gp/N.

K_g=0なら内部countsを全0としepsilon=0にできる。M=0でも全0を直接設定する。
zero precision shareへ新しいcountsを与えない。正のcontinuous groupでもmassが極小ならK_g=0になり得る。
全joint countsを平坦化し、一つのN通りの一様整数へ対応させられる。
ここで構成したのは整数lawであり、trajectoryやruntime samplerを実行した証拠ではない。

## D. y/z roundingと元tau

nearest/half-upでは|y^N-y|<=1/(2N)。theta_r=z_r/yは非負でsum theta=1。
z^N_r=theta_r y^Nならsum z^N=y^N、|z^N_r-z_r|<=1/(2N)、inactive z^N=0。
zはlatent witnessであり、dyadic化を要求するphysical probabilityではない。

任意のc_endpointについて、

    |M^N-z^N c_endpoint|
    <= |M^N-M| + z |cbar-c_endpoint| + |z-z^N| |c_endpoint|
    <= 1/N + Ymax rad + c_plus/(2N).

z<=y<=Ymaxと正値normを使った。Ymax rad<=2/Nの前提なら元のtau=(3+c_plus/2)/N以下である。
endpoint両方を認証する元membershipと一致し、zero-mass representationにも成立する。

**上端の注意：** y<=Ymaxだけからy^N<=Ymaxは一般には従わない。
y=Ymax=1+3/(4N)ならy^N=1+1/Nで超える。
これはその部分推論の反例であり、固定tableの同時保証を破る反例ではない。
元sourceのYmaxは2/(sum t-delta_num)。固定tableでは各D columnのL1 envelope<2が確認でき、
丸め後のmean認証から

    y^N (sum t-delta_num) <= sum_k (Dq^N)_k <= 2

が従うため、y^N<=Ymaxも成立する。genericな別tableへの移送にはこの論拠か明示的な上端処理が必要。
監査で新しい上端roundingをproductionへ採用したわけではない。

## E. Gamma_xi/d/Qの独立導出

Dがgroup内で同じなので、各degreeのendpoint和の変化はsum_g D_endpoint_kg Delta M_gのみ。
target側の変化は-(y^N-y)t_kである。絶対値最大は端点変化量以下だけ増えるため、

    xi(q^N,y^N) <= xi(q*,y) + sum_g L_g/N + ||t||_1/(2N)
                 <= Xi(u) + Gamma_xi.

bias係数d>=0と内部誤差式を使うと、

    |sum_p d_gp(q^N_gp-q*_gp)|
    <= max_p d_gp |Delta M_g| + sum_p d_gp |epsilon_gp|/N.

groupを加算して提案Gamma_dを得る。costもC>=0に同じ証明を適用し、両向きの絶対差をGamma_Qで抑える。
inactive groupの寄与は実際には0だが、table固定のreserveへ残しても安全。
alias展開はgroup数を増やすためB2/B3の共通maxを使う。coalesce後のgroup数でB2を認証しない。
端点・tieの等号は非strictのまま受理する。LRMの1/N、1は安全な上界であり、全Gammaがtightとは言わない。

非負性を外すとmax+sumは負になる例があり、絶対差のboundではなくなる。
一般signed costにはabsolute coefficientsによる別boundが必要だが、今回は採用しない。
固定tableのcost/dは非負なので、この問題は発生しない。

## F. Mean・confidence・caps・workspaceの同時保証

H=ey-dq*-Xiとし、Gamma_h=e/(2N)+Gamma_d+Gamma_xi。
上記から、

    e y^N-dq^N-xi(q^N,y^N) >= H-Gamma_h.

H>=kappa_up+Gamma_hなら元confidenceを満たす。
kappaを下げていない。Xiはconfidence側で一回、Gamma_xiもrounding損失として一回現れ、
mean capを別にも課すことは漏れでも不正なconfidence緩和でもない。
保存dの意味は元sourceのphase-preserving実装bias上界のままである。

mean条件から、

    xi^N <= Xi+Gamma_xi <= y delta_num-delta_num/(2N) <= y^N delta_num.

resource条件から、

    G_R^N = 2n(C_R q^N+h_R)
          <= 2n(C_R q*+h_R+Gamma_R) <= b_R.

h_1Q=5/2、h_T=h_CX=0は元会計のまま。supportが増えても、使用可能variant全てのworkspace peak<=capを
preflightしてあればworkspaceも通る。期待workspaceで置き換えない。
実装時はこれらの構成保証に加え、旧定義の全certificateへ直接代入して再確認する。

## G. Fixed-n LP、nesting、比較下界

固定table/nに対してcbar、v、rho、全Gamma、kappa/capsは有理定数。
normalizer、構造等式、Xi、mean、confidence、capsは全てu/y/zの線形制約となる。
LP最小値は保守的continuous inner generatorの目的値であり、元クラス全体の最小値ではない。
同じ設計の丸め前後では|G_Q^N-G_Q*|<=2n Gamma_Qだが、これはglobal optimality保証ではない。
認証済みUと元outerの認証済みLを別々に保存しgapを報告できる。今回はU/Lを取得していない。

元の認証クラスではK1 subset K2 subset K3が成立する。
B1はB2の他representationをq=z=0とすることで埋め込める。
B2の同一implementation aliasesはcountsを加算してB3へ埋め込め、denominator・normalization・Dq両端・
bias・cost・active peak workspaceを保存する。共有O0のidentityは静的12組の照合で確認した。
再量子化は不要である。

**区別すべき点：** B2/B3でgroup集合が違うので、continuous点を写して別々にLRMしたlawが
同じになるとは限らない。新生成器のdecode像そのもののnestingを仮定しない。
certified B2 seedを再量子化せずB3へ渡す規約がこの相違を処理する。

decode(I2) subset K2、decode(I3) subset K3を認証する一方、lowerはK2を包含するO2から取得する。
したがってprimaryは元の `U_Q^B3,certified < L_Q^B2,outer,certified` のまま公平である。
inner最小値をB2 lowerにすること、B3だけreserveを弱くすることは認めない。

元sourceのnumerical B2 outer rows（robust=False）についても包含方向を確認した。
任意の元認証lawのq/y/zに、各degreeのinterval残差上界r_kを追加すれば、
outerのlower-endpoint/upper-endpointのmean rows、sum r<=y delta_num、
dq+sum r-ey<=-kappa、normalizer、resource rowsを全て満たす。
membershipのouter rowsは `M-z c_plus<=tau` と `-M+z c_minus<=tau` であり、
両端を認証する元membershipより弱い。q<=1、z<=y<=Ymax、r_k<=Ymax delta_numの有限boundsも満たす。
よって元クラスをouterから除外していない。この確認はsourceのread-only監査で、LP build/solveは0である。

## H. 反例とinfeasibilityの意味

主要十分条件の前提を全て満たす反例は見つからなかった。
次の人工有理数反例・注意例は保存した。

| 例 | 結論・必要な扱い |
|---|---|
| single group c=v=t=1、d=e-kappa、q=y=1 | 元lawはconfidence境界で認証可能。構造等式がu=y=1を固定し、正Gamma_hでinner全体が空になる |
| 同eventで1 countを低d・高cost variantへ移動 | mean/membershipを保ててもresource capを破る。confidenceだけのgreedy修復は十分でない |
| precision別D intervals [1,1] と [1,5]、q=(0,1)、v=cbar=y=t=1 | 最初のintervalから作ったXi=0はactual xi=4を抑えない。D同一性または別envelopeが必要 |
| 小分母N=8、c interval=[1/8,3]、cbar=25/16 | membership preflight不成立で元tau超過。tauを緩めて受理しない |
| cost=-1 | max+sum型Gammaが負となる。非負性が必要 |
| y=Ymax=1+3/(4N) | y上端だけの推論は失敗。固定tableではmean/L1論拠で閉じる |

[反例のexact値](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/rounding_bound_audit_v1.json)を参照する。
これらは科学的negative結果ではない。inner空の反例はv4の保守性を示す。

statusはCERTIFIED_PRIMAL、INNER_MODEL_INFEASIBLE_ONLY、CERTIFIED_ORIGINAL_CLASS_INFEASIBLE、
EXACT_ACQUISITION_FAILED、RESOURCE_LIMITを区別する。
inner用Farkasはinnerのみの証明で、元クラスの正常skipには使えない。
元クラスを包含するouterの正しいFarkasに限り元class infeasibleを言える。
outer/innerを含む全row・boundsを独立Fractionで確認し、solver statusだけを認証にしない。
既存seedがある場合のcaps付きupperは維持できるが、それだけでは改善witnessにならない。

## I. 独立synthetic検証と計算費用

実行：`/usr/bin/python3 -B scripts/tracks/algorithm_codesign/check_ra_d0_v4_mathematical_audit.py`。
人工39 fixturesと、六確率座標のdenominator-3 simplex全56点×補助N={2,3,4}の168例、合計207件。
小NはLRM上界の補助的integer arithmetic検査であり、本番N=2^60やdyadic契約を変更していない。
本番Nではinteger counts、finite midpoint/interval、inactive support、非zero mean丸め残差、
confidence/mean/resource boundary、workspace、alias、各反例を直接Fraction certificateで検査した。

**207 PASSは、期待した受理・拒否・反例検出が全て一致した意味である。全fixtureが実行可能という意味ではない。**
人工ell_upper=12と固定synthetic nのkappaは旧と同じ形のoutward upperを使う。
登録nのLPやscience taskのconfidence評価を行ったものではない。
数学的証明と有限個のproperty checksを区別し、後者だけで一般保証とはしない。

[全207件](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/synthetic_verification_v1.json)と
[実行記録](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/audit_execution_v1.json)：
wall約0.153 s、CPU約0.152 s、peak RSS 44,116 KiB。
監査専用CPU 30 s/address space 256 MiBのguardを使い、旧v3 production capは継承していない。
reserveは保守的で、境界lawを失う可能性がある。登録domainでのgenerator feasibility、optimality gap、
費用改善の実用性はまだ未検証である。静的Gammaが小さいだけで解の存在を推定しない。

## J. Exact backendと実装へのGO/STOP

SoPlexは有理入力を扱う候補である。公式7.0 exact-mode資料にはGMP付きbuild、exact parsing・solving・
final checkの設定が記載されている。これを将来使う場合もversion/build/parserを固定し、decimal/floatへ
落とした入力をexact入力と扱わない。[SoPlex 7.0 exact-mode](https://soplex.zib.de/doc-7.0.0/html/EXACT.php)

公式APIにはrational primal、dual、Farkas取得interfaceがある。取得可能なstatus・有限bounds・係数を含む
全rowの独立Fraction検証が必要で、APIの存在は成功保証ではない。[SoPlex source API](https://github.com/scipopt/soplex/blob/master/src/soplex.h)

公式siteでは6.0.3以降Apache-2.0とされ、build依存には別licenseがあり得る。
採用buildの依存inventoryは未確認である。[SoPlex official site](https://soplex.zib.de/)

PATH上のsoplex/esolver/qsopt_ex/sage/glpsolは見つからず、system Pythonと既存pyenv 3.12.3の
read-only module probeでも候補bindingsは見つからなかった。全環境で不存在とは断定しない。
新dependencyのinstall・production runtime変更は0。synthetic solver callも0である。
exact input parser、primal/dual/Farkas出力、backend実行時間は **UNVERIFIED_BACKEND** として未確認のまま返す。
[backend記録](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/exact_solver_feasibility_v1.json)を参照する。

**提案：数理上は次の限定backend検証を設計できる。現時点のproduction実装・登録実行はNO-GO。**
GPTが必要性・範囲を判断した後、承認済みisolated環境でexact rational round trip、独立certificate、
小規模off-domain timing/capsを閉じる必要がある。旧2秒/LPや旧total callsへの適合は未保証。
この監査からproduction source、authorization、one-shotへ自動移行しない。

## K. Provenanceと停止

旧v3/T0/T0.1/T0.2を含む135 protected filesのSHA256を実行前後で照合し、不変を確認した。
旧source/contract/authorization/result、consumed marker、R1/R1.5、Track A、過去STOPを変更しない。
旧marker SHA256は `88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
sourceや索引の既存tracked pathを編集せず、今回の新規監査pathだけを公開する。

registered solver=0、new synthesis/science=0、v4 production implementation=0、
new angle/precision、IS/CTS、circuit/matrix/trajectory/DF/molecule/NPZ/GPU、authorization/marker変更=0。
数値・研究上の旧分類を再分類しない。
[全証拠manifest](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/evidence_manifest_v1.json)を入口とする。
**mandatory STOP。次の判断をGPTへ戻す。**
