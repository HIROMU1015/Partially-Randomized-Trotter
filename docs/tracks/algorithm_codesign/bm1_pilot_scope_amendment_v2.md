# BM-1案 v2 amendment：heuristic scope・cost・method gate

2026-10-05 JST。利用者のBM-0.5 reviewを反映する。
[BM-1 proposal v1](bm1_small_model_pilot_proposal_v1.md)はcommit `3b2d624adde979f7d8f983bc7fdbbec88f90c500`の
結果前提案として保持し、今回の差分だけを記録する。preregistration／実行authorizationではない。
**BM-0.5で現adapterのnew-method gateは不通過。BM-1を実行しない。**

## 1. 承認されたscope方針の反映

将来、別の明確なmethod deltaまたはapplication pilotが承認された場合も、初回は
**leading-order heuristic + I2 oracle evaluation**へ限定する。

- selectorはG_i/lambda_i/N、登録列とcostからなるI1だけを使う。
- exact finite signal・bias・oracle winnerは共通評価専用。selector・thresholdへ戻さない。
- selection regret/missを測る。leading modelの予測誤りも保存する。
- finite-time accuracy保証／certificate、BCH remainderの完成を初回claimにしない。
- numerical guardはI2参照の信頼性のため結果前固定する。heuristicへ科学的保証を付けない。

finite-RTE意味論、signed time、controlled relative phase、独立occurrence、phase一回という
algorithm同一性の義務は残る。certificateを要求しないことはsemantic／numerical検証を省く理由ではない。
BM-2でcertificateやcompiled resourceを扱うかも別判断、今回は未認可。

## 2. Primaryとsecondaryを修正

| 役割 | Metric／cost policy | Claim限界 |
|---|---|---|
| Primaryの構造cost | exact adjacent fusion後のnative logical block count。各deterministic DF/one-body指数blockと各tail occurrenceを1blockとして計数 | 物理gate costやshots込みのresource gainではない。tail内部r eventsも1occurrenceの単位なので限界を明記 |
| Primary evaluationの役割 | I2共通参照で、登録domain内の選択miss／count regretとcontrol機構を見る | oracleは選択に使わない。regretの定義・feasibility・thresholdは実行前に別freezeが必要 |
| Secondary stress test | v1の固定人工weights c_A=1,c_B=8,c_R=4 | 人工costだけの改善はmechanismに限定。DF-native／compiled resource改善を主張しない |
| Shot/action proxy | 実施を承認する場合だけsecondaryとして結果前固定 | 今回primaryへ含めない。新しいshot model探索を追加しない |

非退化a,b>=1、R occurrenceが残る、許可fusionのみというv1の条件では
`C_count(nested)=q[2{m(2a-2)+1}+2b+1]-(q-1)`、
`C_count(flat)=q[2(a+b)+1]-(q-1)`。
これは列文法からの記号count。zero/empty/identity、消えたR等のcaseは別規則が要る。
commuting generatorのreorderを片方にだけ許さない。basis transitionのactual costは未評価。

## 3. 同情報・同scoreの比較を明確化

compact側にもnestedの再帰構造、floor/internal分解、repeat reuse、同じDF backendとaggregationを許す。
combinedとseparatedのtriangle policyを別armに与え、その違いを独立methodと扱わない。
norm-onlyに勝っただけではnew-method GOにしない。

model recipes、6instances、m={1,2,4}、q={1,2,4}、最大72列、ideal/finite計144 reference案は
v1の未実行提案のまま保持する。このamendmentはdomain追加もpilot採択も行わない。
source、canonical RTE、selector、tie、numerical guard、regret/threshold、資源上限、seed、
authorizationは未freeze。旧BF基準・budgetを流用しない。

## 4. Gateと停止

[BM-0.5監査](bm05_equivalence_and_method_delta_audit_v1.md) §6のnew-method gateが実行前条件。
現adapterは代数同値で、同情報compactには得られないscore／reuseを示せていない。
よってBM-1 new-method routeの実行条件を満たさない。
applicationへの縮小、別delta、追加検証の必要性・scopeはGPT側が決める。
今回、science run／toy生成／NPZ／signal／trajectory／compile／GPUは0、mandatory STOP。
