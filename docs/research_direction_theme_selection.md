# P-B/P-C/P-A 研究テーマ選定

最終更新：2026-09-25 JST

## 結論

研究テーマ選定pilotと各候補の停止点を一巡し、**P-A（DF回路列のinterval joint synthesis）**、
**P-B（signal weight）**、**P-C（current H4 geometry family）**はいずれも事前固定した条件で
主研究候補から停止した。現時点でA/B/Cに確認済みの主研究候補はない。

P-A v1は未使用H5 physical snapshotとH4 optimization level 2の両stratumで事前固定した6 gateを
全て通過した。ただし後続形式化で、48 holdoutと6 probeの全件が1 source run当たり1 segmentとなり、
さらにforced-support order-2のtraining/blind全30 taskでも一区間baselineとplan・compiled metricが
一致した。statusは`stop_pa_interval_dp_as_primary_and_return_to_pc`である。scoped auditは網羅的な
新規性証明ではなく、run-level policyの既存改善は限定的実装結果として保持する。

P-Cは局所pilotを通過したが、後続tracking・breakdown validationで追跡prefixが独立prefixと全点同一、
stretch側blind予測が破れ、pair/diagnostic/mechanism gateも不通過だった。statusは
`stop_pc_current_h4_family_as_primary`である。これはgeometry差分一般の棄却ではない。

## 4問による比較

| theme | 既存baselineとの差 | 明確な差分 | 未使用条件 | 完成形を一文で表せるか | 判断 |
|---|---|---|---|---|---|
| P-A | full共有、event-support、現行`support_run_le_1`より安いrun-level union選択 | H5 -17.08%、H4 opt2 -6.60%は保持。interval分割増分はorder-0/2とも0件 | physical/compiler transferは通過したが、forced order-2 blindでも一区間baselineと同一 | 現行v1では独立interval差を示せなかった | 主研究候補から停止 |
| P-C | absolute energyでなくsigned PF errorの差を利用 | 局所二重holdoutは2.539%だが、stretch blindは33.653%/123.245% | 未使用geometryとdelta、held-out region | 局所相殺は保持するが追跡・診断で破れを予測できない | current H4 familyを停止 |
| P-B | energy biasとweight/signalを分離 | 実用的な選択差なし | 選択器を検証できるfailureなし | 反例が得られれば可能 | 現範囲で停止 |

P-Aは強い既存project policyへのrun-level改善、未使用列長、physical/compiler transfer、
operator同値性を示した。一方、明示的一区間baselineとの事前登録比較でtraining/blind全30 taskが
同一planとなり、interval部分の独立差分を示せなかった。P-Cは未使用geometry/deltaの予測gateを通過したが、
現時点では滑らかな局所補間を越えるbreakdown機構や、軌道・fragment追跡による設計可能性までは
示していなかった。後続検証ではtrackingがprefixを一度も変えず、stretch側breakdownをcontinuity診断が
検出できなかったため、P-Cもcurrent H4 familyから進めない。P-Bは一般仮説を棄却せず、現gridで
主題化する根拠がない。

## 次の判断点

P-Aの[非退化mechanism validation](research_direction_joint_synthesis_mechanism_validation.md)と
P-Cの[tracking・breakdown validation](research_direction_geometry_tracking_breakdown.md)まで完了した。
P-Cは7 gate中representation、delta holdout、nontrivial cancellationの3 gateだけを通過し、
固定停止条件が成立した。

当初の案を比較し続けるなら、未実施なのはP-Dの有限PF familyに対するenergy係数とrandom-tail負担の
Pareto監査である。代わりに、A/B/Cが止まった原因を踏まえてR3/R6/R8へ問いを再定義してもよい。
いずれも新しい事前登録を作ってから計算し、H12、長RPE総cost、追加$q>32$、
$\delta/r$境界拡張を自動的な次作業にしない。

## 証拠

- selection artifact：
  `artifacts/research_direction_theme_selection/2026-09-25/theme_selection_pb_pc_pa_v1.json`
- content fingerprint：
  `7daa49c3c30b453d69d52830d0889ce04104cc5a0f02a2c3083517486d575fbb`
- implementation：`src/trotterlib/research_direction_theme_selection.py`
- runner：`scripts/run_research_direction_theme_selection.py`
- test：`tests/test_research_direction_theme_selection.py`
- 専用test：`3 passed`
- 全suite：`582 passed, 2 skipped, 4 warnings`、失敗0

selection artifactはP-B/P-C/P-Aのcontent fingerprintとfile SHA-256を固定し、新しい物理計算や回路compileを
行わずに判断を再構成する。証拠statusはlocal dirty-worktreeであり、外部再現またはimmutable CIではない。
