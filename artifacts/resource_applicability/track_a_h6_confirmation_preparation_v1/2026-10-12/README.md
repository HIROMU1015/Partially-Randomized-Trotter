# H6独立32 trajectory確認：準備完了・実行未承認

[実行準備契約・科学条件・次の承認案](../../../../docs/research/track_a_h6_independent_confirmation_preparation_v1.md)。
起点/result `df99d6ee28942df8d12aac3c4db7cd202df55d77`、SOURCE `73018cbcc39803890974c7ee51401063e6c8daf6`。
原top2/精度のunion8を固定、各32 fresh trajectory＝256 group/512 wrapper。
実確認0、新approved grant0、approved=false/runtime_authorization=false/allowed_cpus=[]、absolute_launch_command=null。

- [候補固定・256 task/seed](sealed_v1/candidate_selection_v1.json)
- [source245/profile/input/探索592/carry/plan/output seal](sealed_v1/manifest_confirmation_v1.json)
- [一回grant草案（未承認）](sealed_v1/authorization_draft_v1.json) / [未実行launch草案](launch_request_draft_v1.json)
- [希少orderと固定32の事前限界](sealed_v1/rare_order_plan_v1.json)
- [69限定tests](limited_tests_v1.xml) / [所有・資源・wall/output計画](limited_test_plan_v1.json)
- [source/profile/input/seed/未使用control監査](preparation_audit_v1.json)
- [保存CSV別算術・literal seed・最終整合性確認](final_consistency_review_v1.json)
- [source closure/旧240維持](source_closure_v1.json)
- [準備時の資源/quota snapshot](preparation_resources_v1.json) / [block/inode/cache/journal容量案](capacity_plan_v1.json)
- [origin df99照合・non-force公開計画](publication_plan_v1.json)

4worker CPU20/21/24/25、coordinator22/23、AS32/RSS9GiB・aggregate40/jobRAM64・thread1・wall cap=null。
最小の累積上限変更案はtrajectory832→840。known578＋旧停止予約6＋新256＝840、未承認。compileは1188＋予約12＋新512＝1712で原cap内。
不明歴史のactual totalはnullを保持し、予約を実消費としない。失敗分返却/resetなし。
旧科学条件/全結果/source/venv/H4/共有環境/他jobは保全。旧compiler output bit同一性は未検証、既存roundoff承認条件を維持。
source/tests/軽量準備資料だけを公開。raw NPZ/runtime/cacheや認証情報を追加commitしない。

次の明示承認は、上記256/512の一回実行とtrajectory840案。fresh prelaunchと新exact grantが必須。
完了/STOP後mandatory STOP、自動retry/追加sample/H8/次stageなし。今回は実行しない。
保留したhash再利用・8worker改修は本準備へ混ぜず、別作業として保留している。
