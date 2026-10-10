# G10-A：保存native費用と任意proposalの固定policy下界

採用[G9 v2 GPT review](../../research/track_b_G9_v2_scientific_review_20261010.md) §13-Aに基づく独立保存値監査。
入力はG9結果commit `c95736fd2990f5ef6dd1cb5866421fd78bb28687`、同p=(1/5,3/10,1/2)、x=5/7、
有限P5のfull first operator moment、指定3-system-qubit synthetic Clifford+T provider、direct実装。
geometry/basis/DF rank/split L_D/PF windowは適用外。known/development、独立外部再現やCIではない。
旧classification/source/contract/authorization/result/marker/STOPは変更しない。

## 独立確認と入力の違い

[保存値専用script](../../../scripts/tracks/algorithm_codesign/audit_g10_saved_cts_policy.py)はstdlib有理算術だけを用いる。
G9原result SHA256を照合し、digital rational event係数と実装費用から再導出した。
[別の費用照合](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/saved_native_recount_v1.json)で
945 direct bindingsの保存literal IRとprimitive T/T† countからT費用を再計数した。
回路build、行列、strict synthesis guard、sampler、solverは再実行しない。
理想Pauli係数はG9 sourceのQ(sqrt(2))、登録angleは有限rational tangent、下界に使う係数は保存digital値。
これらのbyte一致や、理想値についての下界へ無条件に読み替えることは要求しない。

## 固定policyの一般式

非負digital係数 a_e>0、full-support proposal q_e>0、保存native T_e>=0、共通prep/readout T価格 h>=0を考える。
zero-fillのpre-quantum zeroは量子費用0で、非zero事件のproposal和が1未満でも以下のCauchy不等式は成立する。
共通十分shot規則 n>=2 log(2/alpha) m2/s²、二axis合計G=2n sum q_e(T_e+h)から、

\[
G(q;h)\ge {4\log(2/\alpha)\over s^2}
\left(\sum_e a_e\sqrt{T_e+h}\right)^2.
\]

各pairで sqrt((T_i+h)(T_j+h))>=sqrt(T_i T_j)+h。
よって固定256bitの平方根下端を用いる独立有理算術で、

\[
G(q;h)\ge A_{\rm lower}+K_{\rm lower}h,\quad
A_{\rm lower}={4\ell_{\rm lo}\over s^2}\left(\sum a_e\sqrt{T_e}_{\rm lo}\right)^2,
\quad K_{\rm lower}={4\ell_{\rm lo}\over s^2}\left(\sum a_e\right)^2.
\]

log下端はrange reduction後の正atanh級数96 terms、root下端はinteger isqrt。
range項・ceilingを捨てる向きは下界。T=0を正の価格へ置き換えない。
これは固定policyの必要費用下界で、物理的必要shot数、attained IS optimum、実行可能なproposalではない。
既知[Cugini–Atif–Subaşı Theorem 1 Eqs.(9)–(13)](https://arxiv.org/html/2603.13495v1)の原理を
この有限confidence規則へ適用したもので、IS原理の新規性を主張しない。

## CTSレビュー下界の確認

全14 rotation事件が136 T、全10 real correction事件が0 Tであることを保存IRから確認。
rotation係数質量R>7/5、margin s<1/200、log(44000/49)>34/5を有理Taylor exp上界で確認した。
従ってreviewの

\[
G_{CTS}(q;h)>290017280+2132480h
\]

を独立に確認した。保存closed P5はG(h)<255860637+1834258hなので、全共通h>=0で分離する。
[exact証拠](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/saved_policy_audit_v1.json)には
Rと各有理下界、閉形式側の保存T/Kを保持する。粗い下界の符号だけでなく、全固定辞書へ同じ一般式を適用した。

| 固定G9辞書 | 任意proposal T切片下界の表示値 | 保存closed P5との全h>=0分離 |
|---|---:|---|
| ordinary | 347,918,490.395 | 成立 |
| partial_return_tail | 258,117,355.764 | 成立 |
| closed_P3_tail | 266,899,436.516 | 成立 |
| full_return | 255,172,895.627 | この下界では未分離 |
| closed_P5_full | 255,172,895.627 | この下界では未分離 |
| matched_CTS | 305,097,828.828 | 成立 |

表示値は説明用で、判定は保存exact Fractionの不等式とK下界を使う。
ordinary/partial/P3/CTSには固定precision・価格・policy内の分離が残る。
full local/closed P5の同familyは未分離であり、非改善やsampling最適解の一致を証明したものではない。
G9登録結果の数値やclassificationを新しいprimary outcomeへ書き換えない。

## 主張しない範囲

identity吸収・Pauli regrouping・既知定数寄与除去・stratification・precision配分・別confidence・別compiler・別providerは外。
canonical CTSへの約29.77%を最適sampling後の保証削減率と呼ばない。
partial/P3の分離も新しい実行可能lawを取得した結果ではなく、この固定event辞書に対するpolicy下界である。
m7/general degree、新規性成立、実分子/DF、PR/QPE総cost、algorithm採択へ一般化しない。
