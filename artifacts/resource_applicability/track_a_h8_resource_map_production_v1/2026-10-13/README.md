# H8本格core resource-map：認可済み実装と一回起動入口

SOURCE `cb8a56c6b2a772827275f9200b02638030f33f72`。branch `track-a-h8-resource-map-production-20261013`。最新ユーザー指示で準備後の本格core開始まで認可済み、科学的比較条件の変更はなし。

- [実行契約・高速化・数値品質・予算・STOP](../../../../docs/research/track_a_h8_resource_map_production_v1.md)
- [92候補とselection](candidate_registry.json)、[manifest](approved_manifest.json)、[一回grant](approved_grant.json)
- [call/resource/carry](call_resource_budget.json)、[264 source/test closure](source_closure.json)
- [41限定検査](test_result.txt)、[限定差分review](limited_review.json)、[fresh資源/native](fresh_prelaunch.json)
- [科学call0のmetadata setup修正](setup01_metadata_stop.json)

Run ID `H8_CORE_RESOURCE_MAP_20261013_run02`。private入口 `/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-resource-map-v1/`。`control02/launch.json`/`coordinator.log`/`native_exit.json`、`run02/manifest.json`/`signal/observation_journal`/`signal/group_*/progress_*.json`、後続`exploration`/`confirmation`/最終`execution_terminal.json`。

92 signal→適格探索8→top2/精度 union<=8→独立32、core最大1708compile/832trajectory/4208occurrence。全PR/未測定を初めからheavyにし、actual zero-orderの登録p7だけpromotion候補。条件付き最大8worker、AS32/normalRSS9/heavy24/aggregate76/job96GiB、5秒fail-closed。q16/transfer/rare pilot/SCF/DF/state/他研究/GPU/automatic retry/次stageなし。科学的成功や母平均上界は起動・人工testsから認定しない。
