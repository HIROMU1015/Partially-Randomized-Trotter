# H7中性開殻：限定pilot完了

**20/20 group・40/40 full-wrapper compile・16 PR trajectory・32 events、COMPLETE_MANDATORY_STOP。**

- [条件・結果・限界の説明](../../../../docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md)
- [6構成・精度別cost/shot/SE集計](cost_summary_v1.json)、[30 precision rows CSV](signal_precision_cost_v1.csv)
- [比較図PNG](cost_comparison_v1.png)・[SVG](cost_comparison_v1.svg)
- [40回路・240 metrics CSV](compile_metrics_v1.csv)、[20 groupの公開抽出](completed_groups_v1.json)
- [input/state/DF/native/6 signal](input_state_signal_v1.json)
- [終了・binding監査](completion_audit_v1.json)、[native exit/reap/残存0](native_completion_v1.json)
- [実資源集計](resource_summary_v1.json)、[callとpartial保守枠](call_totals_v1.json)、[attempt履歴](attempt_history_v1.json)
- [SOURCE270file](source_closure_v1.json)、[profile/input/carry/launch binding](launch_binding_v1.json)
- [監視修正33人工/mock検査](artificial_tests_33.txt)、[private fileのbytes/SHA](private_evidence_inventory_v1.json)

Linear neutral H7 / 1Å / STO-3G / ROHF / Nα4Nβ3 / 14modes / sector1225 / tol-onlyDF rank23・prefix6 / T.8・q1・δ.8 / S2・fullYoshidaS4・PRr2K2/K4。Runtime SOURCEd9fd36fc、元science/seed SOURCEf93ad35b。

ε=.05/P0のPR/fullS2総RZ点比はK2 .684194、K4 .706055（31.58%・29.39%減）。PRはε≤.01で不適格、ε=.01はfullS4だけ適格、ε≤.005は全6構成が不適格。各PR8trajectoryの限定panel、経験的numerical allowance、未観測tail・母平均上界なし。

run01退出監視STOPを保存し、late completionを含む12完了groupをbyte/native照合して受理。残8 groupを新runtimeで元seedのまま完了し、再開wall109.560秒。S4は初回550.097秒/HWM5.714GiBで完了し再計算なし。全attemptの記録済みwall合計986.556秒、会話・修正待ち時間を除く。

通常最大4group・内部thread1、S4単独。Preprocessing/spin/phase/hash/cos-sin reuse、既存compilerとvenvを維持。最終native exit0/全reap/cleanup0/残存0。Call上限46compile/19trajectory/38eventsは旧3 partial group予約を含み、logical40/16/32と区別。unknown actual=null、reset/refund0。

保存メタデータの追加監査は新SCF/DF/state/signal/trajectory/compile0。120軸shot算術と40回路のmetric/source/compiler/event/task/seed、元snapshot streaming SHAを照合。公開JSON/CSVで追跡でき、raw NPZ/runtime/cache/credentialsは収録しない。Private evidenceは `/home/AbeHiromu/projects/h7-handoff-evidence/20261016/`。

Full panelや追加科学計算へ進まずSTOP。これは主担当のローカル技術監査であり、別担当の独立レビュー/CI/外部再現ではない。
