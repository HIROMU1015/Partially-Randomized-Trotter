# 有限Taylor RTEのreturn集約：構成とnative資源上の適用限界

構成・資源trade-off・限界ノート、初稿 v0.1。GPTレビュー日2026-10-10、資料化2026-10-11 JST。
本稿は利用者が採用した[G10 v3科学レビュー](../research/track_b_G10_v3_scientific_review_20261010.md)
を、既存証拠に対応する形へ整理したもの。独立新規性、投稿十分性、新たな科学実行は採択しない。
原結果の`G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`とmandatory STOPは維持する。

## 要旨

有限Taylor演算子の乱択unitary実装では、形式returnの集約、係数質量の削減、event supportの
削減、native総費用の削減を区別する必要がある。既存G6の一般構成は、involution関係だけを用いて
全形式returnを集約し、even-parent/child pairingと局所係数queryから生成する。低次数のclosed
P3/P5は同familyの安価な特殊化である。

G10 v3の固定synthetic入力では、各次数内の同一有限targetで比較すると、登録canonical
full returnは対応するclosed対照よりT切片・準備呼出し係数とも大きかった。特にm=7では
closed P5＋ordinary tailに対する追加native資源利益は得られない。固定full辞書とnative合成列、
同じ十分予算規則を保つproposalクラスでも、保存下界は0≤h≤970で対照の登録費用を上回る。
一方、P5系のliteral CTS対照へのT利益は残り、CTSにはCXの利点がある。

この記録は一般構成の数学的成立を否定しない。一般fullの性能探索はこの固定入力で区切り、
低次数実装と一般構成の関係、情報access、finite-bit補正、費用モデルによる適用限界を残す。

## 1. 対象と証拠境界

一般数理対象は、正の有理p_i、Σp_i=1、Hermitian involution Q_i²=I、奇数m、0<x≤1に対する
M_m=P_m(-iσxR)、R=Σp_iQ_i、σ=±1である。x=0はidentityとして別扱いする。
追加の代数関係を仮定せず、非可換語の順序とcontrolled相対位相を保持する。
既存の[一般数学監査](../tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)
と[finite-bit/access監査](../tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)が根拠であり、
今回その証明やsamplerを再実行したものではない。

G10のnative比較はp=(1/5,3/10,1/2)、x=5/7、m=3,5,7、3-system-qubitのdevelopment providerに限定する。
Q0=Z0、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)、Qi=Vi†ZiVi。
Pauli展開が取得できるI1文脈であり、分子geometry、chemistry basis、DF rank、split L_Dは適用外である。

各m内ではfull first operator momentを比較する。m間を同じexponential精度のランキングにしない。
Taylor remainder、multi-step PR、QPE/RPE、エネルギー精度、DF scaleの取得費用はscope外である。
finite-bit実装では近似係数・角度のbiasを別会計し、irrationalなideal lawのexact実装とは区別する。

原結果は`fcd3ea6217bc00b667180cec149a70102d75f07e`、取得性改善は
`e65e3c0680fff4cfc4243ed5d6c87429cd5ce753`、science S3は
`b9ed01455351628c9073748f5ba5751aa794b789`、実行HEAD A3は
`53a7bc4ca8051bfd76343e98f3122c0f198e37d0`。source-bound local evidenceであり、immutable CIや外部科学再現ではない。

## 2. 一般構成と低次数特殊化

P_n(u)をn個IID labelから得たraw wordが隣接同labelの除去でuへreduceする確率とする。
形式集約係数と演算子平均は

\[
a_u=\sum_{\substack{|u|\le n\le m\\n-|u|\ {\rm even}}}
(-1)^{(n-|u|)/2}\frac{x^n}{n!}P_n(u),\qquad
M_m=\sum_u(-i\sigma)^{|u|}a_uQ(u).
\]

既存監査ではshort-step条件下の非負性、既知free-product母関数の特殊化による局所query、
有限bit proposal補正を区別して扱っている。Green関数自体を新定理とはしない。

偶数reduced parent uのchild集合C(u)について、s_u=Σ_i a_iu、d_u=√(a_u²+s_u²)、
φ_u=atan(s_u/a_u)、child確率a_iu/s_uを用いると、

\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_u Q_i}Q(u)
\]

の平均がparentとそのchildを再構成する。回転は単一involution Q_iに対して行い、奇数語全体を
involution扱いしない。局所生成はglobal normalizerの全列挙を要求しないが、未知normalizerを
無料のshot削減情報として使えるわけではない。zero-fillも全推定試行数へ含める。

P3/P5のclosed構成はfamily内の低次数実装である。m=3/5の対応するfullとclosedはideal ensembleを
共有し得るが、生成・normalizer取得・予算の会計が異なる。m=7のclosed P5＋ordinary tailは
全returnを吸収するfullと異なるensembleで、P5後の追加集約の価値を判別する対照となる。
非列挙性だけをordinary RTEとの差とはしない。native Q_i、controlled回転、角度取得・合成の費用も必要である。

## 3. 共通予算と資源座標

固定axis誤差は1/200、alpha_axis=49/34000、全34 axesと17 rowsのfailure合計は1/20。
strict Rz error=10^-6、up_to_phase=false、root 256 bit、proposal 160 bit、eta=rho=10^-12。
sourceのbias控除後の残余幅sと既知moment/range上界を共通Bernstein十分予算へ戻している。
Nはaxis当たり試行数、受理率Zに対しK=2NZが期待量子呼出し数である。

主T費用はG_T(h)=T_0+Kh、h≥0。hは共通prep/readout単価であり、結果後に都合のよい単価を
採用しない。T_0は二axes分の期待native切片である。native T/CX/1Q、workspace、古典費用は別座標で、
Tを含む1Q countとTを独立量として単純加算しない。期待費用とtail/hard上限も分ける。
native実装はliteral direct loweringであり、全回路の大域compiler最適化結果ではない。

全17行の[正確な資源表](../../artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json)
と[独立event索引](../../artifacts/track_b_g10_v3_review_access/2026-10-10/README.md)を正本として辿れる。
原結果に物理量子shotsを実行した証拠はなく、Nと費用は固定taskの十分資源会計である。

## 4. 登録canonical比較

| 次数 | fullのclosed対照 | fullのT切片増加 | fullのK増加 | 適用範囲 |
| ---: | --- | ---: | ---: | --- |
| 3 | closed P3 | 約0.19329% | 約0.19329% | 同じP3、登録law |
| 5 | closed P5 full | 約0.64336% | 約0.64336% | 同じP5、登録law |
| 7 | closed P5＋tail | 約2.91155% | 約0.61484% | 同じP7、登録law |

差が両方正なので、各比較でfullの登録T費用は全共通h≥0で大きい。この主張はgeneral fullの
全入力・全backendでの不可能性を意味しない。

m=3ではpartial returnが低いT切片を持ち、closed P3は低いKを持つ。保存された正確な有理数から
両affine費用の境界はh≈287.5265937973となる。単価を選んで一つのwinnerに固定しない。

![m3 prep T sensitivity](../../artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/m3_prep_T_boundary.png)

図1：closed P3−partialの登録T費用差。横軸0–600は既存affine式の表示範囲で、新しい科学条件ではない。

## 5. 固定辞書sampling下界の意味

event演算子・正係数a_j・native T_j・合成精度・合成列・残余誤差s・failure配分を固定する。
full-support proposal q_jでweight a_j/q_jを用い、Σq_j≤1の残りはzero-fillとしてよい。
既知のCauchy–Schwarzと登録十分予算規則から、二axes費用には

\[
G(q;h)\ge\frac{4\log(2/\alpha)}{s^2}
\left[\left(\sum_j a_j\sqrt{T_j}\right)^2+h\left(\sum_j a_j\right)^2\right]
\]

という条件付き下界がある。原sourceは平方根・対数の下側有理近似を使う。
今回のローカル検算は保存済みlower fieldを用い、対数や最適proposalを再計算していない。

m=7の保存full下界と登録P5＋tail予算との差は表示値で

\[
L_F(h)-G_{P5+tail}(h)\simeq5,415,782.443240-5,537.20852461h.
\]

その分離端点は約978.0708852063。安全な0≤h≤970では有理数の差が正で、h=970でも
約44,690.174365 Tの余裕がある。これはfull側のproposalクラス下界と対照の実現登録予算の比較である。
端点は実行可能lawのwinner crossoverではなく、それより大きいhでfull改善の存在を示すものでもない。
canonical fullの不利は別のT/K両正差から全h≥0で残る。

![m7 lower and registered gaps](../../artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/m7_registered_and_lower_T_gaps.png)

図2：P5＋tailを基準とする二つの異なる差。青線は保存された全proposal下界、橙線は登録canonical費用。
横軸0–1200は表示範囲である。下界未分離と改善の存在を混同しない。

下界は特定の十分予算policyの費用を制約し、あらゆる量子推定の最低query数を証明するものではない。
別辞書、event統合、別合成列/精度、state-dependent variance、別推定法は比較クラス外である。
m=3/5へこの分離を一般化しない。T=0のCTS eventも削除せず、固定lowerに含まれている。

## 6. root価格、CTS trade-off、古典側の限界

m=7の空parent rootで、P5＋tailのtangentは2189372/2921709、fullは9011383315/12025404923。
保存native Tは各childで[136,138,140]から[140,142,144]へ変わる。rootは受理回路の約87.8%を占める。
追加集約が頻出回転の合成価格も変えた具体例である。これだけで全費用差の一意の原因帰属とはしない。

P5系はliteral CTSに対する登録T利益と固定CTS下界による条件付き分離を維持する。一方m=7で
CTSのnative CXは約10.393百万、P5＋tailは約16.657百万である。共通prep CX単価gを含む
CXの境界は約4.3354181512。literal CTS family全体への優位や全資源支配を主張しない。

![m7 separate native resources](../../artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/m7_native_resource_vector.png)

図3：P5＋tailを基準とするnative資源の別座標。共通prep gatesを加えていない切片とKの表示であり、
座標間を足して単一scoreにしない。

255対2,250というm=7 binding件数は参照supportの性質である。両構成のproduction生成は全表を
毎trial列挙する方式ではなく、件数比を時間・メモリscaling優位や一意回路数と呼ばない。
総runtime22.23秒、576固定interface trialsも方式別のscaling証拠ではない。

## 7. 維持するclaimと区切り

[Claim/evidence表](../tracks/algorithm_codesign/g10_v3_claim_evidence_map_20261010.md)に根拠と非claimを対応付けた。
return吸収、ordinary pairing、free-product式、cost-aware importance sampling、zero-fill、CTSには既知の
構成要素がある。これらの文献上の位置付けは採用したGPTレビュー§10とG6監査に依存し、
今回新たな全文priority監査を実施したとはしない。構成全体の独立新規性は未確定である。

G9の条件付き継続は、G10の「P5特殊化後にも一般fullに利益があるか」で区切る。
今回の固定native比較ではその追加利益は不支持であり、一般full性能主張を主役として拡張しない。
低次数構成の成果、一般恒等式と有限bit構成、native利益の条件と限界を既存証拠から整理する。
独立論文の主要新規claimを確定すること、別構成や比較・主評価を変更すること、一般化検証はGPT/利用者へ戻す。

この初稿は投稿承認ではない。m9、新p/x/provider/seed/precision、sampling最適化、合成、DF/分子/PR/QPE、
G11は未認可。旧結果・source・contract・authorization・STOP・markerとTrack Aを保持する。
**Mandatory STOP。**
