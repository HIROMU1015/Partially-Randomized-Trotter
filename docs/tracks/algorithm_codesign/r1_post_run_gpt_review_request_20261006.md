# R1後GPT review：全登録結果から研究Bの判断へ

2026-10-06 JST。R1 one-shotは`R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`で完了し、
保存値監査PASS。**mandatory STOP**。この資料は研究方針の判断をGPTへ戻す入口であり、
追加実行・algorithm採択・新規性成立を認可しない。

## 最初に読む固定資料

1. [結果照合と全primary比較](r1_one_shot_result_validation_20261006.md)。
2. [全264 rows表示CSV](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、
   [summary／全24 primary比較](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/descriptive_summary_v1.json)。
3. [evidence manifest](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
   [保存値audit](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/saved_field_audit_v1.json)、
   [原result](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json)、
   [one-shot marker](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/one_shot_consumed.json)。
4. [source Sの結果前契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/docs/tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)、
   [Sのnative semantics](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/docs/tracks/algorithm_codesign/rte_reallocation_r1_native_semantics_v1.md)、
   [authorization Aの明示指示](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/411f08f768244fe87b600d82308c3851847fe9e4/docs/tracks/algorithm_codesign/r1_execution_authorization_receipt.md)。
5. 既知内容・定理scopeは
   [固定R0.5監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61dd534567fda5c7348fdc688814089eb26a3561/docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)、
   [固定R0 proof](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md)を参照。

結果commitのfull SHAと固定GitHub URLはCodexの公開後報告を使用する。
S=`d43d64a821a0249a0dfab12a2472bd3a72fdee74`、A=`411f08f768244fe87b600d82308c3851847fe9e4`。
source/preparation Pは別履歴として保持し、実行はclean HEAD=Aだけで行った。

## 保存結果が示す範囲

targetは2-qubit P₃(-iσx(3Q₀/4+Q₁/4))、m=3/K=2、x={1/8,1/4}、σ=±1、
native ε={10⁻³,10⁻⁴,10⁻⁶}。分子/DF/geometryはない。
126 keys / 264 rows / 132 controlled tasksが完了し、全task適格。science failureなし、retry0。
契約のaccuracy/confidence/caps/tool/guardは変更していない。source-bound local evidenceである。

primary distinct-basis controlledの各12 groupsで、AのB²はordinary/PTSC-K0の両方より低い。
ただしA/PTSC-K0のE[CX]は全12 groupsで高い。
G_Tはordinary比LOWER8/HIGHER4、PTSC-K0比LOWER4/HIGHER8。
G_CXはordinary比LOWER10/HIGHER2、PTSC-K0比LOWER4/HIGHER8。
全σ/precisionを掲載し、最良条件のみに絞っていない。
Pauli controlsにはcollected CTSを保存し、I0-style primaryと混ぜない。

G_T/G_CXはcoherent biasを引いた共通Bernstein sufficient-shot forecastとadditive primitive costの積。
measurement・finite signal・whole-circuit最適化済みcost・DF改善の測定ではない。
実際にPauli展開可能なtoyであり、実問題のI0/I1取得cost分離は未実証。
登録二符号や三precisionは独立validationではない。単一scalar materiality・自動研究GOはない。

## GPT側へ返す判断

1. normalization/native-event trade-offとrestricted theoremを、Track Bの主method候補へ進めるだけの情報とみなすか、technical resultへ縮小するか。
2. 同I0 PTSC-K0対照を含む全vectorを評価し、B²だけの利益や一条件の小さなratioを根拠にせず、有用な適用域・Pareto候補があるか。
3. synthesis precisionによるcost/guard/bias差への依存を、mechanism resultの限界としてどう扱うか。
4. Pauli CTS controlsと、Pauli reductionを使わないdistinct-basis実装contractから、algebraic accessに関して主張できる範囲を決める。
5. RQ・先行研究との定理/method差・論文最小着地点を再評価し、追加検証が必要ならその必要性・最小scopeを別途指定する。

これは新しい研究案の採択や次stage提案の自動承認ではない。
Codexは保存値照合・公開までで停止した。別geometry/degree/precision、IS/PAI、DF/分子、
trajectory/GPU、wrapper、辞書取得、新baseline探索を追加していない。
結果がpositiveに見える条件を含んでも、再実行やgrid拡張はしない。

## 役割分担と未実施

GPT：研究方針/RQ/新規性/論文着地点/追加検証の必要性・範囲。
Codex：承認済み固定specの一回実行、保存値監査、provenance、必要資料のcommit/push。
本run後のscience callsは0。旧BF/BM/SP/Track A証拠、source、authorization、marker、STOPは保持した。
最終cost優位・DF scale・oracle/access advantage・algorithm採択・immutable CI・外部再現は主張しない。
**mandatory STOP。追加science未認可。**
