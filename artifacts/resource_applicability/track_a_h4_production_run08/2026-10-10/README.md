# H4 run08：worker32GiB本計算の認可資料

[実行scope・資源・fresh gate・限界](../../../../docs/research/track_a_h4_production_run08_20261010.md)。
[SOURCE67](source_freeze_v1.json)、[利用者認可](USER_AUTHORITY_v8.json)、[凍結入力/最新native停止proof binding](stop_receipt_v18.json)、
[sealed plan](plan_authorized_v18.json)、[auth](authorization_v18.json)、[最終review](review_v18.json)、[独立review](independent_authorized_run08_review_v1.json)、
[最終binding](authorization_binding_v18.json)、[fixed argv](runner_argv_v18.json)、[fresh launcher](fresh_launch_script_v8.json)、[11限定tests](limited_binding_tests_v1.json)、[commit一覧](commit_inventory_v1.json)。
worker4/AS・RSS32GiB、driver8GiB、observer256MiB/64MiB、head16GiB、admission152.25GiB。CPU2/4/5/6・driver16・observer18。
科学/compiler/旧pressure条件/凍結6入力/17GiB/74805/72h不変、per-attempt carry0。追加人工compile0。
新run08のSOURCE/profile/input/nativeproof/carry/認可/digestを独立review後固定し、fresh gates全合格時に一度execする。開始確認後chat終了可。
実runtime/NPZ/cache/credentialはこのbundleへ収録しない。全map完走・旧compiler完全同一性は未検証。
