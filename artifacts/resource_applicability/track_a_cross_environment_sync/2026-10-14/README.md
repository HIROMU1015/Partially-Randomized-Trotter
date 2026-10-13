# Track A server inventory / cross-environment synchronization snapshot

2026-10-14 JST。サーバー観測の暫定初版、ローカル未照合。[共通入口](../../../../docs/research/track_a_shared_research_status.md)、[棚卸し](../../../../docs/research/track_a_server_inventory.md)、[同期ルール](../../../../docs/research/track_a_environment_sync_policy.md)、[local handoff](../../../../docs/research/track_a_local_codex_sync_handoff.md)。

|File|内容・scope|
|---|---|
|`fixed_evidence_index.json`|19件の主要証拠のcommit、repository相対path、Git blob、bytes/SHA、固定GitHubリンク、fresh remoteからの到達性|
|`remote_branch_snapshot.json`|read-only `git ls-remote`で取得したmain/Track A heads。一時点の可動branch観測|
|`worktree_inventory.json`|登録67worktreeのHEAD/remote/dirty/family counts、限定changed file identity。branch未公開とcommit到達性を分ける|
|`external_review_inventory.json`|Git管理外レビュー4件とbyte-identical公開コピーのmapping|
|`science_identity_projection.json`|H4/H6/H8のinput H/DF/state/source/profile/seed hashと結果commit。科学arrayを含まない|
|`h8_readonly_snapshot.json`|run/source/seed、探索203/592、92 signals、8 active、checkpoint/complete metadata SHA、ledger known/reserved、source265一致。最終resultではない|
|`private_storage_presence.json`|private NPZ/runtime/venvの識別名・保存状況・記録hash・bytes。絶対pathと本体は公開しない|
|`worktree_table.md`|全worktreeの短い表とdirty summary|
|`verification.json` / `asset_manifest.json`|固定Git/公開bytes、JSON整合、内部リンク、保全scope、新科学計算0の照合|

既存のscientific inputs/array、raw trajectory、live checkpoint、ledger、grant、source、venvは変更していない。marker・小さいcheckpoint/complete・source bytes・Git metadataだけをread-onlyで確認した。
H8観測はnon-atomicなので進捗がその後変わる。現在のrunを停止したりfinal sealしたりしない。
既存公開raw pathを含むfile本体はコピーせず、新JSONはowner-relative識別名へsanitizeした。サーバー私有保存先への解決表はGit対象外。

共有ノートは正本の索引。claim/科学採択/新grantを発行せず、公開後mandatory STOP。再照合はstdlib script `scripts/resource_applicability/verify_track_a_shared_status.py` をrepository rootから`python3 -P -B`で実行する。元科学計算やH8監視を再実行しない。
