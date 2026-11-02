# RA-D0 T0.1 GPT handoff — active-support diagnostic only

2026-10-08 JST。**`T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED`**でmandatory STOP。
指定projectionとprojected-law quantizationを各一回実施した。
solver/runnerは0回。旧v3/T0の分類、result、消費済みmarker、authorizationを保持している。

固定S=`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`、
A=`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`、
R=`35f8b949079f15d0348bc082b916324870da7246`。
T0基点=`72192b3475d59f5c56370cb4068d5659789f0ef4`。
branch=`track-b-ra-d0-t01-active-support-projection-20261008`。
対象は`P1_ANCHORS:1/8:767135:minimum:T`一件だけ。

[監査全文](ra_d0_t01_active_support_audit_20261008.md)、
[新しい独立stdlib script](../../../scripts/tracks/algorithm_codesign/audit_ra_d0_t01_active_support.py)、
[projection結果](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/projection_result_v1.json)、
[certificate比較](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/certificate_comparison_v1.json)、
[verification](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/verification_v1.json)、
[全出力manifest](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/evidence_manifest_v1.json)を参照する。

## 結果

ordinaryだけがactive（lambda=1）。PTSC_K0/Aはinactive（lambda=0）で、
precision sharesを定義せず全weights/latent zを0に保った。
active ordinaryの保存precision sharesとinterval midpointによる共通正規化を使用した。
continuousのsimplex、latent sum、midpoint group equality、precision shares保持はexactに成立。
fixed dyadic samplerも共通denominator=2⁶⁰、sum q=1、fixed z=lambda×fixed yを満たす。

| point | membership | mean margin / 判定 | confidence margin / 判定 | workspace |
|---|---|---|---|---|
| 保存N3 | FAIL | +9.84594×10⁻¹³ / PASS | −4.46407×10⁻¹⁷ / FAIL | PASS、1 |
| continuous projection | PASS | +9.84638×10⁻¹³ / PASS | −6.07779×10⁻¹⁹ / FAIL | PASS、1 |
| fixed dyadic projection | PASS | +9.84638×10⁻¹³ / PASS | −9.16086×10⁻¹⁹ / FAIL | PASS、1 |

fixed点の未充足条件はconfidenceのみ。continuous点でも既にconfidenceがFAILなので、
dyadic roundingだけを原因としない。閾値やnを変更せず負のmarginを記録した。
schema/source reconstruction上のtechnical blocking issueはない。

## 限定的に言えることと未決事項

今回のactive-support規約により、T0のzero-mass適用不能は解消し、
同じ保存nominal designからmembershipを通すdiagnostic pointを構成できた。
しかし四certificate同時PASSには至らない。
**数値modelそのものの不成立とは解釈しない。**
このpointは新B2 minimum、最適解、RA-D0性能結果ではない。
資源費用はpoint診断として保存するだけで、改善量を論じない。
B3の優位性、新algorithm採択、新規性、repair後のrun成功は未判定。

次に数値実装を再設計する情報価値・必要性・範囲を、confidence marginとmean保存、
構造保存parameterization、dyadic lawを含めてGPT側で判断する。
Codexは修正source、v4 authorization、再実行を作らず、科学仮説・candidate・baselineを変更しない。
今回の診断は次stageのauthorizationではない。

## 不変と回数

旧marker SHA256=`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
旧v3/T0 source/contract/authorization/result/marker/監査scriptと旧R1証拠はbyte-identical。
元N0/N3再構築は旧法のread-only verificationで、新projection/量子化の各一回とは別に記録した。
solver、B2/B3/Farkas、runner、new synthesis/science/circuit/matrix/trajectory/DF/molecule/NPZ/GPU、retryはすべて0。

**mandatory STOP。次の数値実装の再設計・研究方針判断はGPT側へ戻す。**
