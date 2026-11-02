# RA-D0 T0 read-only numerical failure audit

2026-10-07依頼の固定pathを維持し、2026-10-08 JSTに公開準備を完了。
対象は`P1_ANCHORS:1/8:767135:minimum:T`の保存certificate一件だけ。
限定分類は **`T0_TECHNICAL_INCONCLUSIVE`**。監査のexact reconstruction自体はPASS。
raw nominal時点で既存membership幅を超過し、保存dyadic点にはconfidence不足もある。
指定projectionはzero-mass groupのため適用できず、repair可能性の判定には到達していない。
元v3分類は永久に`D0_TECHNICAL_INCONCLUSIVE`のまま保持する。

## 固定scopeと実装

- source S：`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`。
- authorization A：`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`。
- result R：`35f8b949079f15d0348bc082b916324870da7246`。
- 独立branch/worktree：`track-b-ra-d0-t0-read-only-failure-audit-20261007` /
  `.worktrees/track-b-ra-d0-t0-read-only-failure-audit-20261007`。Rから作成。
- [独立audit script](../../../scripts/tracks/algorithm_codesign/audit_ra_d0_t0_saved_failure.py)。
  `/usr/bin/python3`とstdlibのみを使用。RA-D0 source package・NumPy・SciPy・solverをimportしない。
- saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x=1/8、n=767135、T minimum。
  既存candidate tableの3 precisionのみをそのまま参照する。別x/n/objectiveを解析しない。
  分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/tracks/algorithm_codesign/audit_ra_d0_t0_saved_failure.py
```

固定sourceの`exact.py`、`lp.py`、`numerical.py`とnumerical policyを読み、必要な
scalar interval・LP row・rounding式を独立に転記した。saved nominal primalをFractionのまま代入する。
LP coefficient rowのsubstitutionは量子operatorのmatrix evaluationではない。
新coefficient、角度、sqrt正規化、入力は取得しない。confidenceのell/kappaだけは固定100桁ruleを再現する。

## raw nominalと全LP制約

24 q variables、y、4 mean auxiliaries、3 latent zを保存primalから抽出。
fixed B2 inner LPの2 equality rows、27 inequality rows、全32 variable boundsをexact arithmeticで評価した。
保存dualの全stationarity residual・correction・lower boundも独立row再構築と厳密に一致した。
これは元solverのoptimal flagを再評価する処理ではない。

| 項目 | 再導出値 |
|---|---:|
| simplex residual | 3.1780676180981349×10⁻¹⁷ |
| latent mixture residual | 0 |
| equality residual max | 3.1780676180981349×10⁻¹⁷（sum q=1 row） |
| inequality violation max | 7.0152279028132682×10⁻¹⁷（degree_0_lower、row index 2） |
| confidence inner row slack | −4.1265745909047099×10⁻¹⁹ |
| variable bound violation max | 0 |

保存mean auxiliariesは四つとも0だが、exact mean endpoint residualは0ではない。
mean rows・membership rows・confidence rowには微小な違反が残る。
これらの未スケール残差はsource `backend.py`の固定primal tolerance=10⁻⁹より小さいが、
HiGHS内部のscaled feasibility判定を再現したとは主張しない。
solverのstatus=0 / Optimalは保存値として保持する。
fixed certificateはsolver toleranceをvalidity基準にせず、別途exact arithmeticで判定している。

全rows、bounds、最悪row、confidence slack、saved dual/primal/resourcesは
[nominal residual audit](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/nominal_residual_audit_v1.json)に保存した。

## ordinary/O0 membershipの段階分解

すべて同じ保存点と固定intervalから再導出した値であり、依頼文のapproximate値を入力にしていない。
各groupについてmass、latent z、interval endpoints、residual、tau、ratio、residual−tauを保存した。

`ordinary/O0`の固定tauは3.0391410822573796×10⁻¹⁸。
denominator=2⁶⁰、dyadic unit=8.6736173798840355×10⁻¹⁹。

| stage | ordinary/O0 residual | residual/tau | 前stageからのresidual差 |
|---|---:|---:|---:|
| N0 raw nominal | 7.0698219394420570×10⁻¹⁷ | 23.262565797672061 | — |
| N1 q normalization only | 3.9162227734257691×10⁻¹⁷ | 12.885952535368712 | −3.1535991660162879×10⁻¹⁷ |
| N2 fixed y rounding / z rescale | 3.9162227734257691×10⁻¹⁷ | 12.885952535368712 | 0 |
| N3 actual fixed dyadic q | 3.9473196826838043×10⁻¹⁷ | 12.988273909784596 | +3.1096909258035139×10⁻¹⁹ |

raw nominalですでにtauを超えている。q normalizationは超過を減らすが、tau内には戻らない。
yは元から固定dyadic格子上にあり、latent sumもraw yと一致するため、この点ではN2は変化しない。
largest-remainder roundingはresidualを少し増やす。したがって、この点を
「rawはPASSでquantization後だけFAILする`T0_QUANTIZATION_DOMINANT`」とは分類できない。
量子化前の残差が大きいという記述的診断に限定し、一意なcausal percentageは付けない。

interval radius contribution `|z|×(c⁺−c⁻)/2`はN0/N3とも約4.9231908513219180×10⁻¹⁰¹。
endpoint residualはexactに`|mass−z×midpoint| + |z|×radius`へ分かれる。
この保存groupの10⁻¹⁷程度の超過は、保存coefficient interval radiusでは説明できない。
これはalgebraic interval accountingであり、因果寄与の百分率ではない。

[stage decomposition](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/membership_stage_decomposition_v1.json)
には全8 groupのN0–N3とstage間exact differencesを保存している。
N3のq/y/z、membership各flag、xi、bias、resourcesは元certificateと厳密に一致した。

## 保存N3点にある別のcertificate failure

N3はsampler PASS、membership FAIL、mean PASS、confidence FAIL、workspace PASS。
mean marginは約+9.8459394223497014×10⁻¹³。
固定mean residualを戻したconfidence marginは約−4.4640730023364375×10⁻¹⁷。
元certificateの`certified=false`は再構築と一致する。

ここで判定したのは**保存N3点**であり、projected pointではない。
membershipだけを直せばconfidenceも通るとは言わない。
単一pointの未認証は、RA-D0数学modelの不存在やB3/B2の性能関係を示さない。

## projectionを適用しなかった理由

添付の§9.1は「group massが0の場合は、新しい規約を発明せずtechnical diagnostic failureとして報告」と指定する。
この保存点では次の6 groupがexact mass=0、raw latent z=0である。

| representation | zero-mass groups |
|---|---|
| PTSC_K0 | O0、P2、P3 |
| A | A0、A1、A2 |

`pi=q/group_mass`が未定義となる。非zeroのordinary/O0、ordinary/O2のsharesと、
fixed interval midpoint、latent relative sharesは記録したが、zero groupのsharesを補わない。
inactive representationを省く規約も自動で導入していない。
projection application=0、projection後quantization=0。
projection後membership/mean/confidence/workspaceはすべて**NOT_EVALUATED**であり、FAILではない。
projection後resourcesは未取得。

[projection diagnostic](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/projection_diagnostic_v1.json)
は`NOT_APPLIED_ZERO_MASS_GROUP`。このformula applicability blockerにより、
最終T0分類は`T0_TECHNICAL_INCONCLUSIVE`となる。
structure-preserving repairの成功・失敗を判定したわけではない。

## provenance、非claim、STOP

[input identity](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/input_identity_v1.json)と
[verification](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/verification_v1.json)に
固定S/A/R、audit script SHA、protected入力と元result一式のhash、0回のexecution countsを記録した。
[evidence manifest](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/evidence_manifest_v1.json)が本監査の出力一覧。

- marker SHA256：`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
- certificate gzip SHA256：`cde1364d5081ed743db10ae693e6e77560c0633c30b17c6c3e21d752daf144eb`。
- candidate table SHA256：`2f86169fc301ffa6b73f3cde7c8bd49939a4e7742d422c309bac181e5f09d0b6`。
- contract SHA256：`e9422bab70389a41715d8fe817060b811ed492ea46a0a3a4a54f23d043f4afd1`。

元marker/result/certificate gzip/technical failure/source manifest/authorization/contractはRとbyte-identical。
source manifestの全97 critical files、旧R1/R1.5証拠も不変。
固定source文書の当時のstatusは書き換えず、新しい監査の入口は本書と
[GPT handoff](ra_d0_t0_gpt_handoff_20261007.md)に限定した。Track A、共通APIは変更していない。

今回のsolver・registered/synthetic LP・B2 minima再取得・B3・Farkas・RA-D0 runner・
authorization/marker作成変更・synthesis・science・circuit・matrix・trajectory・DF/molecule/NPZ・GPU・retryはすべて0。
既存法を修正・採択せず、denominator/tolerance/tau/solver settingsを維持。
RA-RTE有効性、B3の優位、retry成功、projectionの新規性・最適性を主張しない。

**mandatory STOP。repair source実装、v4 authorization、再実行へは進まない。
zero-mass時のprojection規約とconfidence marginを含む次のnumerical repair/redesign判断はGPT側へ戻す。**
