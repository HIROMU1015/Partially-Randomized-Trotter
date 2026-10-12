# H6ハッシュ再利用・8worker：実装済み、現runは継続

SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、branch `track-a-h6-hash-reuse-eight-worker-20261012`。
[実装・科学条件・資源案・引継ぎ契約](../../../../docs/research/track_a_h6_hash_reuse_eight_worker_v1.md)。

- [251 source closure・旧245のblob/SHA不変](source_closure_v1.json)
- [環境/compiler/input/selection/task binding・現run生存時の起動拒否](implementation_binding_audit_v1.json)
- [101限定人工tests](limited_tests_v1.xml) / [CPU30/thread1/AS2GiB/wall180sの計画](limited_test_plan_v1.json)
- [hash-only人工比較](microbenchmark_result_v2.json) / [wall90秒の計画](microbenchmark_plan_v1.json)
- [限定差分確認・未検証事項・元run観測](local_review_v1.json)
- [8worker資源案・未seal・future grant条件](implementation_plan_v1.json)

人工hash部分は約7.28/9.75倍、全digest一致。実H6全体/8worker速度は未測定。8実workers、実science、追加transpile、新grantは0。
現4worker run（CoPID20241・SOURCE73018cbc）は停止/変更しない。旧245 source/compiler・凍結input/DF/archive/state・既存結果/seedと256確認taskを維持。
準備用processの設定だけを使用し、共有設定/venv変更0・GPU0・tmpはhome配下。

8worker提案：CPU20/21/24/25/26/27/28/29、coordinator22/23、内部thread1、各AS32GiB/RSS9GiB、aggregate RSS76GiB/jobRAM96GiB、fresh host248.25GiB/effective96GiB。現runへ適用しない。
完了whole groupだけを旧source・task/seed/event/compiler・両軸・native退出/cleanupで受理し、残りを新SOURCEへ登録する実装を追加。探索8と確認32をpoolしない。
元runが生存中なのでtechnical execution planは未seal。approved=false/runtime_authorization=false/allowed_cpus=[]、commandnull。切替時に旧STOP/cleanup、final reuse/carryと残task、partial予約、fresh資源/new exact grantを再結合する。
元4groupがpartialになる場合の追加full-task予約は最大compile8/trajectory4（compile1720/trajectory844案）となり得る。これは未来の例で、上限変更や実行承認ではない。budget返却/resetなし。

Git公開は認証待ち、現段階はlocal commitのみ。実runtime/checkpoint/cache/NPZを新commitへ追加しない。
