# H4 geometry compile並列source再固定

`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`。科学未実行、公開後STOP。
Branch `track-a-h4-geometry-parallel-source-20261006`。
起点review `4d5eba454dda06bc2735730cf7c3f132e456db29`、契約base `b662dbd72e49fa713a25c716f323843e547e973b`。

- [修正内容・合成検査範囲・未検証事項](../../../../docs/research/track_a_h4_geometry_parallel_source_implementation.md)。
- [起点/実行前budget照合](identity_preflight_v1.json)、[source段manifest](source_stage_manifest_v1.json)。
- [検査結果](test-attempt-01.json)、[ログ](test-attempt-01.log)、[新規invocation予約監査](synthetic_transpile_reservations.jsonl)。
- [source freeze audit](source_freeze_v1.json)：new actual SOURCE_COMMITのblob/hash/closureと変更対象外sourceの不変性。
- [全資料manifest](artifact_manifest_v1.json)、[commit対象・公開scope監査](publication_scope_audit_v1.json)。
- [次レビュー依頼](REVIEW_REQUEST_v1.md)。このbundleは入力生成plan/authを作成しない。

compileをadmitted worker数以下のbounded投入へ変更し、処理中ownerの重複予約/compileを防ぐ。
COMPLETEとidentity/digest検査後だけcache再利用し、logical order/axis pairing/weightを維持する。
failure時はown runを停止し、消費済み予約と部分ledgerを残す。retry/resume/worker補充/払い戻しなし。

最終111 tests PASS、fail/error/skip0。今回追加synthetic transpile3、旧25との累積28/64。
今回test失敗0、旧失敗記録と旧25件予約台帳は不変。実worker/production性能は未検証。
分子アクセス/科学入力生成/実signal・sampling・build・compile/GPU/本番起動/authorization発行/共有環境・他job変更0。
旧科学証拠・source/review bundle・契約を保存する。local synthetic証拠でありCI/独立外部再現ではない。
