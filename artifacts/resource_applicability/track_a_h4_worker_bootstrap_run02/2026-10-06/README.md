# H4入力生成run02：worker bootstrap修正と一度の再実行

利用者の「では修正してもう一度実行して」に基づく。
CPU [3,5,6,7,8,9]、6 workers、own-run限定mask0x3e8を維持する。
6入力を一度生成しfreeze後STOP。fresh CPU/memory/pressure/OOM/容量/quota検査不合格なら起動しない。

SOURCE_COMMIT: `049e69919af16ad29a67a217dc7a407d6b1754a6`
source19 paths中の変更はgates.pyのrun02/audit参照とworkers.pyの-Pだけ。
科学inputs/execution/signal/circuit/compiler/resource/parallel/seed codeは不変。
-Pはscript directoryのsignal.pyによるstdlib signal shadowを防ぐ。
1実起動prefix import回帰test PASS。科学job/実transpile/GPUをテストでは実行しない。

run ID: `track-a-h4-geometry-v2-20261006-run02`
fixed output: `/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run02`
run01の失敗source/plan/認可・journal・driver logは保持する。削除/移設/同一run resumeはしない。

- [新source freeze](source_freeze_v1.json)
- [run02 plan](input_generation_plan_run02_v1.json)
- [CPU承認を保持したrun02 authorization](authorization_run02_v1.json)
- [利用者の再実行承認を転記したreview](stage_review_user_approved_run02_v1.json)
- [利用者承認の出所](user_reexecution_authority_v1.json)
- [bindingと旧run/output拒否検査](run02_binding_audit_v1.json)
- [軽量bundle manifest](artifact_manifest_v1.json)

容量条件3GiB/260000 inode、6 workersのfresh effective memory72GiB、
own driver/worker AS8GiB・RSS guard/headroom16GiB、wall72h/output10GiBは維持する。
新しいscientific designやsignal/compile stageの認可は作成しない。
実行logとfresh resource/結果はserver-local control `/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/h4-input-generation-run02-20261006T091149Z` に保持する。
科学NPZ/runtime/checkpoint/cacheはこのcommitに含めない。成果はinput freezeであり、mapや最終科学結論ではない。
