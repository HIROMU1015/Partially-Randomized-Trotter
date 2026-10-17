# H8主要6軸：準備3構成並列化の監査bundle

SOURCE `7f8d74660b2914a1223d0ac7990dde8d4645032e`。実装・人工検証のみ。新方式の実科学起動0、現在のrunは旧SOURCEのまま保全。

- [資料入口](../../../../docs/research/track_a_h8_key_compile_parallel_preparation_20261017.md)
- [未承認plan](execution_plan_UNAPPROVED_v2.json) / [無効grant](grant_INVALID_DRAFT_v2.json)
- [SOURCE/profile/input binding](source_profile_input_binding_v2.json)
- [旧/new closure blob・SHA](source_closure_diff_v1.json)
- [人工32検査](artificial_tests_v1.json) / [原log](artificial_test_output.txt)
- [主担当の技術確認](author_technical_check_v1.json) / [現run保全](live_preservation_v1.json)
- [軽量payload file receipts](file_receipts_v1.json)

準備は最大3構成・6 numerical worker、compileは直列・各軸2%以上でSTOP。7 coreと208GiB admissionは未固定の配置/見積りであり、実使用量や実行中の上限ではない。real speedup/peak RSSは未測定。旧grantや旧結果を新SOURCEの実行成功として扱わない。
