# H4 worker32GiB修正資料

[変更・実効上限・検証・次回起動条件](../../../../docs/research/track_a_h4_worker_memory32_fix_20261010.md)。
[actual SOURCE65](source_audit_v1.json)、[32GiB profile](worker_memory_profile_v1.json)、[利用者の承認](USER_MEMORY_AUTHORITY_v1.json)、
[準備binding](preparation_binding_v1.json)、[53限定回帰](test_result_v1.json)、[事前予算](test_plan_v1.json)、[保持した検査履歴](test_history_v1.json)、
[最終独立review v2](independent_worker_memory32_review_v2.json)、[初回review履歴](independent_worker_memory32_review_v1.json)、[SOURCE commit一覧](source_commit_inventory_v1.json)。
worker4/AS・RSS32GiB、driver8GiB、observer256MiB/64MiB、head16GiB、fresh available152.25GiB。科学/compiler/旧pressure/carry0/17GiB/74805/72h不変。
元native診断2gapsは残り、全mapの完走は未検証。sourceと軽量資料だけを固定、実runtime/NPZ/cache/credentialはcommitしない。本体起動0。
