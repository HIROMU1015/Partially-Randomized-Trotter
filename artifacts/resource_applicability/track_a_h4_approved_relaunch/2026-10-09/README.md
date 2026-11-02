# H4明示認可・13GiB再実行入口

一つの入口は[固定SOURCE・認可・直前gate・起動状態](../../../../docs/research/track_a_h4_approved_relaunch_20261009.md)。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`、closure39、branch `track-a-h4-approved-relaunch-20261009`。

- [利用者認可](USER_AUTHORIZATION_v2.md)、[source blob/hash](source_freeze_v1.json)、[限定22tests](limited_tests_v1.json)。
- [plan](plan_authorized_v11.json)、[auth](authorization_v11.json)、[review](review_v11.json)、[binding](authorization_binding_v11.json)。
- [独立最終review](independent_relaunch_review_v1.json)、[固定runner argv](runner_argv_v11.json)。
- [library cache](library_cache_profile_v1.json)、[停止・carry receipt](stop_receipt_v9.json)、[commit対象](commit_inventory_v1.json)。

13GiB累積charge、carry20/4428938712B/5472.345380863175s、actual74804と全ての他cap/科学/compilerを固定。
worst13,165,893,832B <= cap13,958,643,712B、余裕792,749,880B。旧失敗費用を返却しない。
sealed/approved/runtime=trueは今回の認可artifactのsnapshot。独立review・commit後にfresh gate PASSなら一度起動し、完了またはfail-closed STOPで終了する。
自動retry・旧partial/cache・入力再生成・次stage・GPUなし。実run/PID/statusはhome runtime receiptと最終報告に記録する。
