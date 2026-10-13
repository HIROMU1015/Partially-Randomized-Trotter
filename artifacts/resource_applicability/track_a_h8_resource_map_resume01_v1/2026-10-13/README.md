最小上限補充 trajectory835/occurrence4211は利用者が明示承認済み。auth02を再結合しfresh検査PASS、approved=true/runtime_authorization=true。新run01の一回起動へ進む。以下の承認待ち記述は先行REVIEW時点の記録。

# H8 core：8 workerへのcomplete-only引継ぎ

SOURCE `c6692964c948071ed00363b5b711fbca8930ad61`、branch `track-a-h8-resource-map-resume01-20261013`。科学seedのsourceは元の `cb8a56c6b2a772827275f9200b02638030f33f72` を保持する。

旧runは利用者の「8ワーカーにして引き継いて」に基づき2026-10-13 23:28 JSTにown-run STOP。signal92/92、探索75/592をbyte SHA・native reap・task/seed/paired metric/compilerで受理し、残517を別output/新grant/one-shot lockへ引き継ぐ。全worker回収済み、coordinator native exit1・wait/reap済み、first STOP USER_STOP_REQUEST、cleanup error0。過去の消費/予約を保持し返却/reset0。

CPU20/21/24/25/26/27/28/29、coordinator22/23、各内部thread1。8 processの準備段階を各RSS1GiBで監視し、実event・静的wrapper instruction上界を報告後、coordinatorが9GiB normalまたは24GiB heavyを予約するまでnative buildを待機する。全体のworker予約72GiB/aggregate76GiB/job admission96GiB/AS32GiBを保持する。観測次数すべて0かつ上界200000以下（ordinary anchorは両controlの上界100000以下）だけnormal。rare/large/unknownはheavy。memory boundの数学的証明ではなく、各RSSと5秒fail-closed STOPが実上限を守る。heavy待機中もobserver/coordinatorの観測を継続し、parent退出時のown-run cleanupを保持する。最大8normal・混在時は予算内の並列数とし、常時8 heavyの同時実行を許さない。

SOURCE transitionは旧264のうちscheduler/contract/adapter3だけの変更と新metadata helper1件に限定し、旧科学/runtime261・profile・compiler・input・plan・science gridは不変。SOURCEが変わっても探索/確認のseedを旧campaign sourceで結合し、worker数/完了順に依存しない。旧complete receiptは旧SOURCEのまま保存して来歴を隠さない。新science/SCF/DF/state/signal再計算0、旧wrapperのtranspile再実行0。探索8と独立32の選定・top2/union<=8・精度/shot/native/control/normalizationも不変。

中断3 groupのheld予約compile6/trajectory3/occurrence3は保持する。compileは旧1708内に収まる（実登録592のcore最大1700＋held6＝1706）。最大8構成×独立32まで完走するための最小変更はtrajectory832→835、occurrence4208→4211。control376・他call/resource上限は維持する。carry lineage subtotal2289/1112は不変、compileの累積枠3997不変、trajectoryの累積枠は1944→1947案。現時点のgrantはapproved=false/runtime_authorization=false、上限変更の承認待ちで起動していない。

34 local artificial/mock PASS（12.51秒）、単一test process/内部thread1/AS2GiB/wall60秒/log16MiB。8同時workerはmockで、実worker stress/NPZ読込/transpileなし。旧41全件は再実行せず関連回帰だけ実施した。実8worker速度倍率/rare H8完走は未測定。fresh CPU/memory/PSI/OOM/filesystem/inode/quota/native318/input5/source closureをread-onlyで確認、metadata conditional technical review合格。実行時に再度fresh gateする。

private入口：`/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-resource-map-resume01-v1/`。未使用run01/control01、draft auth01。原runとpartial/cacheを削除しない。実行承認が揃えば新auth02のmanifest/grantを再結合し、既存launch_track_a_h8_resource_map_v1.pyを一度起動する。自動retry/next stageなし、全core終了またはSTOP後mandatory STOP、wall capなし。共有環境/venv/H4/他job/GPUへ変更なし。

[監査・資源・限定review](receipt_and_review.json)、[SOURCE closure](source_closure.json)、[限定tests](limited_tests.txt)。NPZ/runtime/checkpoint/cache/credentialsはGitに含めない。
