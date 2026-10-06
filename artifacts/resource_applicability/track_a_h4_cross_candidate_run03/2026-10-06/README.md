# H4 cross-candidate run03

[資料入口](../../../../docs/research/track_a_h4_cross_candidate_run03.md)。

利用者認可：その修正を入れて、ワーカー数を増やして再実行して。SOURCE 79a552d517ee4d391629b1929ab90d2ba1dae451。

49 local metadata/artificial-job tests PASS、追加transpile0。旧run02全証跡保持。6凍結入力をsource lineage付きで再利用し入力を再生成しない。CPU12件・12 workers。累積10GiB/72h/74784 invocationsをresetしない。fresh preflight PASS後にのみ新runを一度起動しMAP_COMPLETE_STOP。

source_freeze_v1.json / input_reuse_and_prior_budget_v1.json / signal_compile_plan_v1.json / authorization_user_approved_v1.json / stage_review_user_approved_v1.json / binding_checks_v1.json / stage_storage_estimate_v1.json / test_attempt_2,3.log/xmlを一組として参照。production完了や科学的結論は未検証。
