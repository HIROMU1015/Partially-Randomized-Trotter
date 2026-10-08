# RA-D0 T0.1 active-support projection — read-only diagnostic

2026-10-08 JST。限定分類は **`T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED`**。
指定projectionを一回、projected continuous lawへの固定dyadic変換を一回適用した。
continuous/fixed両点でmembership、mean、workspaceはPASS、confidenceはFAIL。
**RA-D0数学modelの不成立、B2 optimum、性能改善を示す結果ではない。**
元v3は`D0_TECHNICAL_INCONCLUSIVE`、T0は`T0_TECHNICAL_INCONCLUSIVE`のまま保持する。

## 固定入力と処理範囲

- source S：`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`。
- authorization A：`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`。
- one-shot result R：`35f8b949079f15d0348bc082b916324870da7246`。
- T0 commit / branch作成基点：`72192b3475d59f5c56370cb4068d5659789f0ef4`。
- 独立branch/worktree：`track-b-ra-d0-t01-active-support-projection-20261008` /
  `.worktrees/track-b-ra-d0-t01-active-support-projection-20261008`。
- 対象一件：`P1_ANCHORS:1/8:767135:minimum:T`。
  saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x=1/8、n=767135。
  固定tableの既存3 precisionだけを使い、別x/n/objectiveは扱わない。
  分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。

[projection contract](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/projection_contract_v1.json)
を診断前に保存した。SHA256は
`1c1256b30eb5457ab1b823b6bed80edd05ddc764292e79d0047244b3d41e0607`。
これはdiagnostic規約であり、RA-D0実行authorizationではない。
[新しい独立script](../../../scripts/tracks/algorithm_codesign/audit_ra_d0_t01_active_support.py)
はstdlibだけを使い、既存T0 script、v3 kernels、NumPy/SciPy/backendをimport・変更しない。

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/tracks/algorithm_codesign/audit_ra_d0_t01_active_support.py
```

## Active-support規約とexact identities

raw saved zのrelative sharesはordinary=1、PTSC_K0=0、A=0。
activeはordinaryのみ、inactiveはPTSC_K0/Aである。
inactiveの6 groupではprecision sharesを定義せず、unnormalized/continuous/fixed qとzを0に保つ。
0除算は行っていない。active普通法のO0/O2はいずれもpositive raw group massなので、
保存precision sharesをそのまま使う。

各active groupで既存interval midpointを使い、
`q_tilde=lambda*midpoint*pi`、`W=sum(q_tilde)`、
`q_proj=q_tilde/W`、`y_proj=1/W`、`z_proj=lambda/W`を一回だけ構成した。
continuous qの総和=1、zの総和=y、group mass=z×midpointはすべてexactに成立。
active within-group precision sharesもcontinuous pointでexactに保持している。
fixed pointは既存roundingのためprecision sharesの厳密保存を主張しない。

固定変換はdenominator=2⁶⁰、q largest remainder / index-order tie、
y nearest / half-up、negative q reject、fixed z=lambda×fixed y。
common denominator=2⁶⁰の整数countsが総和2⁶⁰であることを確認した。
Fractionは約分されるため、各表示分母が文字通り2⁶⁰であるという意味ではない。

元N0/N3の再構築は旧法のread-only verification。
旧N3 rounding reconstruction=1を、新しいprojection application=1 /
post-projection quantization application=1とは分けて記録している。
代替projection、反復repair、candidate探索は行っていない。

## 三点のcertificate比較

固定tau、D interval、mean cap=y×10⁻¹²、n=767135の既存ell/kappa、workspace cap=1を使用する。
confidence marginは`y/200 - d*q - xi_upper - kappa_upper`。
計算はすべてFraction、表示のみdecimal近似で、符号はexact値から決める。
continuous点のdyadic grid適格性は要求せず、fixed点はsampler gridも確認した。

| point | membership | mean | confidence | workspace |
|---|---|---|---|---|
| 保存N3 | FAIL | PASS | FAIL | PASS、peak=1 |
| projected continuous | PASS | PASS | FAIL | PASS、peak=1 |
| projected fixed dyadic | PASS | PASS | FAIL | PASS、peak=1 |

| point | mean margin | confidence margin |
|---|---:|---:|
| 保存N3 | +9.8459394223497014×10⁻¹³ | −4.4640730023364375×10⁻¹⁷ |
| projected continuous | +9.8463817026438357×10⁻¹³ | −6.0777900577738025×10⁻¹⁹ |
| projected fixed dyadic | +9.8463786198957500×10⁻¹³ | −9.1608554379127114×10⁻¹⁹ |

continuous点ですでにconfidence marginは負であり、fixed roundingだけの失敗とは言わない。
負のmarginを微小として無視せず、tau、tolerance、accuracy、nなどを変更しない。
fixed法の唯一の未充足条件はconfidence。schema/input/再構築上のtechnical blocking issueはない。

ordinary groupのmembership endpoint residualは以下。
inactive group residualは全点で0。全8 groupの値とPASS/FAILを保存している。

| group | 固定tau | 保存N3 residual | continuous residual | fixed residual |
|---|---:|---:|---:|---:|
| ordinary/O0 | 3.0391410822573796×10⁻¹⁸ | 3.9473196826838043×10⁻¹⁷ | 4.9231908513219178×10⁻¹⁰¹ | 1.3816868054283835×10⁻¹⁹ |
| ordinary/O2 | 2.6054762855654438×10⁻¹⁸ | 1.5719418360794930×10⁻¹⁹ | 4.9231908513219178×10⁻¹⁰¹ | 1.4800226041303912×10⁻¹⁹ |

continuous midpoint equalityとinterval endpoint検査は別であり、midpoint equalityだけでPASSとしたわけではない。
T/CX/1Qの費用は固定tableの費用から各点で計算して保存したが、
diagnostic pointの会計として扱い、B2 minimum、winner、改善量とは呼ばない。

## Verificationとprovenance

[input identity](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/input_identity_v1.json)、
[projection result](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/projection_result_v1.json)、
[certificate comparison](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/certificate_comparison_v1.json)、
[verification](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/verification_v1.json)、
[evidence manifest](../../../artifacts/track_b_ra_d0_t01_active_support/2026-10-08/evidence_manifest_v1.json)を参照する。

独立に再構築したN0変数/membershipと、N3 q/y/z、membership、mean residual、confidence、
resources、workspaceは旧v3/T0保存値と一致。
projection identities、inactive support、active shares、fixed counts/latent shares、
全三点の四certificateを確認した。verificationのPASSはfull certificate PASSを意味しない。

- 旧marker SHA256：`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
- 旧certificate gzip SHA256：`cde1364d5081ed743db10ae693e6e77560c0633c30b17c6c3e21d752daf144eb`。
- candidate table SHA256：`2f86169fc301ffa6b73f3cde7c8bd49939a4e7742d422c309bac181e5f09d0b6`。
- v3 contract SHA256：`e9422bab70389a41715d8fe817060b811ed492ea46a0a3a4a54f23d043f4afd1`。

元v3/T0 protected files、source、旧R1/R1.5証拠、authorization/result/markerは固定T0 commitとbyte-identical。
既存overview/index/source文書の固定bytesを保持し、本診断への入口は新規の本書と
[GPT handoff](ra_d0_t01_gpt_handoff_20261008.md)に置いた。Track A・共通APIは変更していない。

## 非claimとSTOP

この診断は、指定active-support projectionが保存一件のmembershipを通す一方で、
full certificateを通さないことを示す。数学modelの不成立やRA-RTEの有効/無効を結論しない。
B3/B2 comparison、新B2 minimum、資源削減、repair後one-shotの成功保証、新規性・最適性は主張しない。

solver、registered/synthetic LP、B2 minimum再取得、B3、Farkas、RA-D0 runner、
source v3/T0 script/authorization/marker変更、denominator/tolerance/tau変更、new synthesis/angle/precision、
science、circuit/matrix/trajectory、DF/molecule/NPZ、GPU、retryはすべて0。

**mandatory STOP。repair source、v4 authorization、新one-shotを作らない。
confidence marginを含む次の数値実装再設計と研究方針判断はGPT側へ戻す。**
