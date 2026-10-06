# H4の6入力生成stage容量・最終承認入口

**必要量3 GiB＋260000 inodes。観測available約3.615 GiB＋225817022 inodes。quota3種は非有効。判定：足りる。**
対象は6入力生成→freeze→STOPだけ。10GiBはcampaign capであり、今回stage開始の全量空き要件にはしない。
CPU [3,5,6,7,8,9]、6 worker、own-run taskset mask0x3e8は未承認案。approved=false、実行未認可。

公開直前の17:17:59 JST再観測でもavailable **3,880,157,184 bytes ≈3.613678 GiB**、
available inode **225,816,970**、quota3種非有効で「足りる」を維持した。
[公開直前容量観測](prepublication_capacity_observation_v1.json)も保存する。launch直前検査の代用ではない。

- [source由来容量計算・全仮定](../../../../docs/research/track_a_h4_input_generation_stage_storage_review.md)。
- [保存32配列・容量計算JSON](array_and_storage_estimate_v1.json)、[filesystem/quota観測](filesystem_quota_observation_v1.json)、[stage容量判定](storage_verdict_v1.json)。
- [初期quota照会履歴](quota_initial_query_v1.json)、[quota3種の状態照会](quota_status_query_v1.json)。
- [不変性・binding・容量監査](final_storage_audit_v1.json)、[read-only計算ログ](storage-review-attempt-01.log)、[全手順](READONLY_METHOD_v1.md)。
- [byte-identical plan](input_generation_plan_v2.json)、[byte-identical auth草案](authorization_proposal_v1.json)、[approved=false review](stage_review_proposal_v1.json)。
- [hash/fingerprint一覧](identity_summary_v1.json)、[manifest](artifact_manifest_v1.json)、[commit scope](publication_scope_audit_v1.json)。
- [一括最終承認資料](FINAL_APPROVAL_PACKET_v1.md)、[未実行absolute command](FUTURE_LAUNCH_COMMAND_v2.md)。

「10GiB全量確保案」は旧CPU proposal bundleへ監査履歴として保存し、上書きしない。
今回scientific arrays/NPZ/分子計算/本番/CPU許可/review承認は0。新source修正・大きなtestsも0。
launch直前に3GiB/inode/quotaとCPU/memory/pressure/OOMをfresh再確認し、不合格/不明なら起動しない。
独立最終review・利用者承認・明示launchが揃うまで、資料公開後もSTOPする。
