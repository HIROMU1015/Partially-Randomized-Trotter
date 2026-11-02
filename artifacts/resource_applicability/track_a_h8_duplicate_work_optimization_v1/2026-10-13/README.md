# H8重複処理削減の実装・人工検証

2026-10-13 JST。現H8ジョブを継続したまま別worktreeへ実装。本格H8・切替・retry/resume・追加scienceは起動しない。

- SOURCE `f7c461356eacea091261388af58b44dbb96121c1`、branch `track-a-h8-duplicate-work-optimization-20261013`。
- [設計・適用条件](../../../../docs/research/track_a_h8_duplicate_work_optimization_v1.md)、[準備status](preparation_status.json)、[source closure249＋validation2](source_closure.json)、[旧245source保全](source_preservation.json)。input/profile/compiler/carryは現計算の来歴を保持する。
- [44人工/mock tests](test_result.txt)、[実行上限](test_execution_plan.json)。canonical byte/digest一致・mutation/phase/control/非finite・人工16-mode X/Y、mock compiler input/metrics/seed/call一致。新科学array読込0・trajectory0・actual transpile0・新実worker0。
- [人工microbenchmark](synthetic_benchmark.json)：初回hash2.73倍、warm hash.95倍、状態検証約2.00倍。実H8全体の速度倍率は未測定。
- [最終整合確認](final_consistency.json)：249runtime＋validation2のSOURCE blob/hash一致、資料link一致、稼働中runの245source変更0・coordinator identity一致。中断後の再開時に44testsを繰返さず証拠を保全した。
- future adapter `h8_fast_cost_port_v1.cost_port`は新hashとbranch再利用に接続。並列waveはnormal最大8/heavyまたはunknown最大3のmetadata案、内部thread1・aggregate76GiB。allowed_cpus=[]/approved=false/runtime_authorization=false、absolute launch command=null。

起動中run SOURCE5c207fb8/worker/coordinatorを変更しない。現runnerを新portへ付け替えていない。将来のsource/profile/input/validation method/task/seed/carry/plan/auth/fresh gateが揃った別実行で適用する。共有環境・既存venv・H4/他job変更0・GPU0。
