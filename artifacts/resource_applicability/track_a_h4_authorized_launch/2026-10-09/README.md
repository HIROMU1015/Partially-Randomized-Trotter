# H4 approved one-shot execution bundle

一つの入口は[認可・直前検査・実起動報告](../../../../docs/research/track_a_h4_authorized_launch_20261009.md)。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a` 不変、最新利用者指示を[認可](USER_AUTHORIZATION_v1.md)へ固定。

- [plan](plan_authorized_v8.json) / [auth](authorization_v8.json) / [review](review_v8.json) / [SOURCE33](source_freeze_v6.json)。
- [独立最終execution review](independent_execution_review_v1.json)、[binding](authorization_binding_v8.json)、[直接exec手順](EXECUTION_PLAN_v1.md)、[exact argv/env](runner_argv_v8.json)。
- [固定前監査](artifact_verification_v6.json)、[commit対象](commit_inventory_v6.json)、[bundle SHA](artifact_manifest_v6.json)。

sealed/approved/runtime=true、14 exact CPU roles、observer/環境採用とactual+20明示認可済み。他caps/carry/science/compiler options不変。
最終commit実SHA照合とfresh gate後に追加利用者承認なし一度起動。失敗なら不合格原因を記録して停止、retry/旧cache/次stage/GPU/共有設定変更なし。
SOURCE既存監視で継続し、通常cleanupとobserver強制STOP経路を区別して得られた実exit/own残存を確認する。
raw runtime/log/NPZ/checkpoint/cacheはGit外。起動実結果は別runtime receipt/入口追記で記録する。
