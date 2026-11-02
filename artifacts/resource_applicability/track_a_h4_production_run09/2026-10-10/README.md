# H4 run09：承認済みhost-only PSI猶予

[scope・条件・検証・fresh起動](../../../../docs/research/track_a_h4_production_run09_20261010.md)、[SOURCE72](source_freeze_v1.json)、[利用者launch認可](USER_AUTHORITY_v9.json)、[猶予承認](USER_PRESSURE_AUTHORITY_v9.json)、[pressure v2](memory_pressure_profile_v2.json)、[契約改定](pressure_contract_amendment_v2.json)、[最新native停止proof](stop_receipt_v19.json)、[plan](plan_authorized_v19.json)、[auth](authorization_v19.json)、[review](review_v19.json)、[限定独立review](independent_authorized_run09_review_v1.json)、[最終binding](authorization_binding_v19.json)、[argv](runner_argv_v19.json)、[freshlauncher](fresh_launch_script_v9.json)、[16回帰](limited_binding_tests_v1.json)、[commit一覧](commit_inventory_v1.json)。
4workers AS/RSS32GiB・driver8GiB・observer256MiB/64MiB・head16GiB・admission152.25GiB、CPU2/4/5/6・driver16・observer18。
host1〜5%はnonroot0/OOM同一/available152.25/fresh5秒で30秒猶予、5%以上や30秒継続・nonroot/OOM/欠測は即STOP。起動前host<1%、観測は毎秒。
科学/compiler/凍結6入力/17GiB/74805/72h不変、carry0。旧partial/科学cacheを使い回さない。SOURCE/認可/freshgate固定後に一度map、開始確認後chat終了可。
