# H5中性開殻：限定pilot完了

**COMPLETE_MANDATORY_STOP：20/20 group、40/40 full-wrapper compile、16/16 PR trajectory、32 RTE events。**

- SOURCE：`d86a4673d6ada701797d3e035f745d0d4faf5150`
- run ID：`h5-neutral-open-shell-pilot-20261016-run04`
- 実行時間：249.438秒（約4分10秒）、native exit0、全worker exit0/wait/reap、cleanup error0、own-run session残存0。
- [説明・条件・修正履歴](../../../../docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md)
- [完了監査](completion_audit_v1.json)、[native終了](native_completion_v1.json)、[資源観測集計](resource_summary_v1.json)
- [6構成の費用・shot集計](cost_summary_v1.json)、[40回路の6metrics/seed/hash CSV](compile_metrics_v1.csv)、[groupの公開抽出](completed_groups_v1.json)
- [ROHF/DF/state/native/6signal検証](input_state_signal_v1.json)、[SOURCE closure](source_closure_v1.json)、[launch binding](launch_binding_v1.json)
- [34人工/mock検査](artificial_tests_34.txt)、[旧attempt/carry保持](attempt_history_v1.json)、[private fileのbyte/SHA inventory](private_evidence_inventory_v1.json)

Linear neutral H5、1Å、STO-3G、ROHF共通MO、5e/Nα3Nβ2/10modes/sector100、tol-only DF1e-8/cutoff0/signed order/weighted projection、actual rank14/prefix4、T=.8/q1。B0/B0I/fullS2/fullS4/PRr2K2/PRr2K4の限定6構成。通常4group並列/内部thread1、S4単独、preparation/hash/cos-sin reuse。S²=.7500000000000001、state residual1.05e-15、独立H action差1.74e-14、native primitive最大差5.59e-15。

| 構成 | group数 | 平均RZ/回路（cos） | 平均CX/回路（cos） | ε=.05でのP0総RZ点推定 |
|---|---:|---:|---:|---:|
| B0 | 1 | 5966 | 2476 | 不適格 |
| B0+I | 1 | 5966 | 2476 | 2,018,876,502 |
| Full PF S2 | 1 | 18871 | 8202 | 388,818,084 |
| Full PF S4 | 1 | 55279 | 24246 | 861,080,983 |
| PR r2/K2 | 8 | 7269.25 | 2852.5 | 217,125,228.25 |
| PR r2/K4 | 8 | 7197 | 2834.375 | 215,276,664 |

ε=.05におけるPR/PF最良対照（fullS2）比はK2 .558424、K4 .553669（約44.2–44.6%の点削減）。PRはε≤.01で不適格。fullS4はε=.01/.005でも適格、ε=.001/.0005はこの固定panelの全構成が不適格。母平均・未観測次数・全候補最良・stateprep込みやenergy推定全体の優位性を保証しない。

S4のgroup wall122.802秒、worker peak RSS1.918GiB。全stage最大同時RSSの1秒サンプル値1.768GiB、最大観測間隔1.003013秒・観測処理.007409秒。peak HWMと同時RSSサンプル値は別の量で、監視の間のpeakを0や同時peakとして補完しない。

既定counterと明示Kのadapter不備を修正して新attemptを起動した。旧run03の完了S4 2compile、旧partial/未完了予約/unknown actual、run02入力消費はそのまま保持。run04の40compileに返却や過去partialの混合をしていない。源コード/環境全byteは実行後にも一致。科学比較条件・既存H4/H6/H8・共有venvの変更、GPU query/useは行っていない。

保存scalar/JSON/byteによる監査は追加SCF/DF/state/signal/trajectory/compileを行わない。raw NPZ・runtime/checkpoint/cacheはGit未収録。private evidenceは `/home/AbeHiromu/projects/h5-handoff-evidence/20261016/pilot-run04/`。

限定pilotは終了。H7・H5全面map・追加samplingへ自動進行せずSTOP。
