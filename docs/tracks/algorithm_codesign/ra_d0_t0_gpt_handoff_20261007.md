# RA-D0 T0 GPT handoff — exact saved failure only

2026-10-07依頼の固定pathを維持し、2026-10-08 JSTに公開準備を完了。
**`T0_TECHNICAL_INCONCLUSIVE`**でmandatory STOP。
元v3は`D0_TECHNICAL_INCONCLUSIVE`のまま。repair/runの成功を示す結果は取得していない。

固定S=`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`、
A=`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`、
R=`35f8b949079f15d0348bc082b916324870da7246`。
T0独立branchは`track-b-ra-d0-t0-read-only-failure-audit-20261007`。

[監査全文](ra_d0_t0_read_only_failure_audit_20261007.md)、
[独立stdlib script](../../../scripts/tracks/algorithm_codesign/audit_ra_d0_t0_saved_failure.py)、
[machine-readable verification](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/verification_v1.json)、
[全出力manifest](../../../artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/evidence_manifest_v1.json)を参照する。

## 保存点から確認できたこと

対象はP1最初のB2 T minimum、x=1/8、n=767135の一件だけ。solverを呼ばず、
saved double-derived Fraction primalを固定B2 inner LPの全制約へ代入した。
保存dualの全residual/correction/lowerと再構築が一致し、saved N3法の全certificate fieldsも一致した。

| ordinary/O0 | residual | residual/tau |
|---|---:|---:|
| N0 raw nominal | 7.0698219394420570×10⁻¹⁷ | 23.262565797672061 |
| N1 q normalization | 3.9162227734257691×10⁻¹⁷ | 12.885952535368712 |
| N2 fixed y / z scale | 3.9162227734257691×10⁻¹⁷ | 12.885952535368712 |
| N3 fixed dyadic q | 3.9473196826838043×10⁻¹⁷ | 12.988273909784596 |

tau=3.0391410822573796×10⁻¹⁸、2⁻⁶⁰=8.6736173798840355×10⁻¹⁹。
rawで既にFAIL。normalizationは−3.1535991660162879×10⁻¹⁷、y/z段階は0、
dyadic roundingは+3.1096909258035139×10⁻¹⁹の変化。
coefficient interval radiusの項は約4.9231908513219180×10⁻¹⁰¹。
causal percentageを付けず、nominalの時点に違反があるという診断に限定する。

保存N3はmembershipだけでなく**confidenceもFAIL**。
confidence marginは約−4.4640730023364375×10⁻¹⁷。
meanとworkspaceはPASS。membershipを修正すれば全条件が通るとは言えない。

## projection blocker

依頼§9.1により、zero-mass groupは新規約を補わずtechnical diagnostic failureとする。
保存primalではPTSC_K0のO0/P2/P3、AのA0/A1/A2がmass=z=0。
指定within-group shares `q/group_mass`は未定義。
そのためprojectionを実施していない。projection後membership/mean/confidence/workspaceは
すべて**NOT_EVALUATED**。performance/winner/resourcesの新結果はない。

したがって、`T0_SUPPORTS_STRUCTURE_PRESERVING_NUMERICAL_REPAIR`、
`T0_QUANTIZATION_DOMINANT`、`T0_DEEPER_NUMERICAL_INCONSISTENCY`の採択根拠にはしない。
数学modelの反証でも、repair可能性の確認でもない。

## GPTへ返す未決事項

1. inactive representation / zero-mass groupのprojection規約をどのように明示するか。
2. membership幅とdouble nominal residualのinterface、およびconfidence marginを含めた
   numerical realizationの設計を変更する必要性・範囲。
3. 元v3の消費済みmarkerとtechnical分類を保持したうえで、今後別の準備・検証に情報価値があるか。

これは判断事項の整理であり、新しい規約・repair・runを提案済み契約として採択していない。
Codexはrepair source、v4 authorization、再実行に進んでいない。

## 不変と実行回数

元marker SHA256=`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
元result/certificates/technical failure/source manifest/authorization/contractはbyte-identical。
旧R1/R1.5証拠・Track A・source/solver/tolerance/denominator/tauも不変。
T0のsolver、B2 minima、B3、Farkas、RA-D0 runner、new synthesis/science/circuit/matrix/trajectory/DF/molecule/NPZ/GPU、retryはすべて0。

**mandatory STOP。研究方針・次のnumerical repair/redesign・追加検証の必要性と範囲はGPT側で判断する。**
