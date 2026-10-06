# H4入力生成 CPU・launch最終案

`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`。
**CPU [3,5,6,7,8,9]、6 worker、利用者承認未取得。approved=false、本番未認可。**
異なる6 physical cores、NUMA node0、約3秒の受動sampleで各core busy0%。空き/予約/許可の保証ではない。
memory観測は約981.887GiB、PSI/OOM0。filesystem空き約3.717GiBは10GiB全上限分の容量条件に未達。

- [候補・根拠・科学未実行の説明](../../../../docs/research/track_a_h4_input_generation_cpu_launch_proposal.md)。
- [CPU/topology/NUMA/load/resource観測](cpu_resource_observation_v1.json)、[launch-context提案](launch_context_proposal_v1.json)。
- [byte-identical plan v2](input_generation_plan_v2.json)、[CPU候補入りauth草案](authorization_proposal_v1.json)、[未承認review](stage_review_proposal_v1.json)。
- [source/binding/resource監査](proposal_audit_v1.json)、[検査12件](gate-tests-attempt-01.json)、[全手順/失敗履歴](OBSERVATION_METHOD_v1.md)。
- [hash/fingerprint一覧](identity_summary_v1.json)、[manifest](artifact_manifest_v1.json)、[commit scope監査](publication_scope_audit_v1.json)。
- [未実行launch command](FUTURE_LAUNCH_COMMAND_v1.md)、[一括最終review/承認判断](FINAL_APPROVAL_REQUEST_v1.md)。

science SOURCE `9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a` とscience checkoutを移動・変更しない。
planは完全byte-identical、authはallowed_cpusだけ、reviewはauth digestだけ変更してapproved=falseを維持する。
CPU許可・独立最終review・明示launch・容量/fresh resource条件が揃うまで実行しない。
12 metadata tests PASS、追加transpile0/旧28件不変、taskset/worker/科学/GPU/共有環境変更0。公開後STOP。
