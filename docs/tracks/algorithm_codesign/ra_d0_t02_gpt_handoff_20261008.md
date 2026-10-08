# RA-D0 T0.2 GPT handoff — one saved point, post-hoc diagnostic

2026-10-08 JST。**`T02_FULL_CERT_PASS`**でmandatory STOP。
T0.1 fixed dyadic lawのordinary O2内で、指定delta=2⁻⁴⁰を`O2:1e-4`から`O2:1e-6`へ一回移動した。
移動countsは2²⁰=1,048,576、common denominatorは2⁶⁰。量子化・solver・RA-D0 runnerは0回。

固定source S=`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`、
authorization A=`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`、
v3 result R=`35f8b949079f15d0348bc082b916324870da7246`、
T0=`72192b3475d59f5c56370cb4068d5659789f0ef4`。
branch基点はT0.1=`5cf56e4a5949d64c24eac127bac0c223d431df87`。
branch=`track-b-ra-d0-t02-exact-certificate-20261008`。
対象は`P1_ANCHORS:1/8:767135:minimum:T`一件のみ。
saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x=1/8。
分子geometry/basis/DF rank/split/PF delta窓は適用外。

[監査全文](ra_d0_t02_exact_certificate_audit_20261008.md)、
[独立stdlib script](../../../scripts/tracks/algorithm_codesign/audit_ra_d0_t02_exact_certificate.py)、
[診断前fixed contract](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/fixed_shift_contract_v1.json)、
[point結果](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/weight_shift_result_v1.json)、
[全certificate](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/certificate_comparison_v1.json)、
[resource vectorと増分](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/resource_delta_v1.json)、
[verification](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/verification_v1.json)、
[全出力manifest](../../../artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/evidence_manifest_v1.json)を参照する。

## 確認できたこと

対象2列の理想D intervalsはcanonical bytes / exact Fractionで同一、workspace peakはともに1。
logical event label/word/phase、prototype/degree/a/b/directionの一致も保存tableから確認した。
精密列のsaved d upperが小さく、source weightは指定移動量以上だった。
table index 13/14、B2 variable index 4/5はIDから解決してassertした。

| point | sampler | membership | mean | confidence margin / 判定 | workspace |
|---|---|---|---|---|---|
| T0.1 fixed | PASS | PASS | PASS | −9.1608554379127114×10⁻¹⁹ / FAIL | PASS、1 |
| T0.2 shift | PASS | PASS | PASS | +4.7597036654694233×10⁻¹⁷ / PASS | PASS、1 |

y/z、他のq、全group mass、全Dq interval endpoints、xi upperはexactに不変。
inactive PTSC_K0/Aのq/zも0のまま。
confidence差は`delta*(d_source−d_destination)`とexact一致。
resource増分はT=`795518995/8796093022208`、CX=`0`、1Q=`4108007925/17592186044416`。
各座標で`2*n*delta*(C_destination−C_source)`とexactに一致した。全値はpoint診断である。

## 解釈の境界とGPTへの判断事項

**同じ保存一件について、membershipを通したlawの精度配分を指定量だけ変えると、
固定confidenceまで満たせることを確認した。**
deltaは旧失敗点を見た後に選ばれており、事後診断である。
T0.2 PASSをRA-D0 v3成功、B2 minimum取得、B2/B3優劣、RA-RTE新規性、汎用repair完成へ拡張しない。
runtime samplerを実行した証拠や新しい科学実験ではなく、保存tableと固定certificate上の構成である。

B2/B3中心比較は依然未実施。今回の単点構成をもとに数値実装を一般化する必要性・範囲と、
研究継続の情報価値をGPT側で判断する。Codexは科学仮説・candidate・baseline・閾値を変更しない。

## ProvenanceとSTOP

125 protected入力のbytes/hashes、旧source/contract/authorization/result/marker/T0/T0.1 scriptは不変。
旧marker SHA256=`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
v3=`D0_TECHNICAL_INCONCLUSIVE`、T0=`T0_TECHNICAL_INCONCLUSIVE`、
T0.1=`T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED`を保持する。
technical blocking issueはなし。
solver/runner/new synthesis/science/circuit/matrix/trajectory/DF/molecule/NPZ/GPU/IS/CTS/retryは全て0。
追加authorization、marker変更、alternate delta/pair、iterative repair、v4 sourceは作成しない。

**mandatory STOP。数値実装の一般化と研究継続判断をGPT側へ戻す。**
