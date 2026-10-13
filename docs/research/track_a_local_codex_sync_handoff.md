# ローカル側Codexへの研究情報照合依頼

2026-10-14、サーバー側の暫定初版。**ローカルの現在状態は未確認。科学計算・無条件mergeは依頼しない。**

## 最初に読むもの

1. [共通研究入口](track_a_shared_research_status.md)。
2. [サーバー棚卸し](track_a_server_inventory.md)と[同期ルール](track_a_environment_sync_policy.md)。
3. [fixed evidence commit/blob index](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/fixed_evidence_index.json)。
4. H4実測結果、H6確認/pilot、H8最新resume02の各固定報告。科学レビューも必ず対応scopeを読む。
5. [研究ノート索引](研究ノート/README.md)と、各固定H6/H8 branchの`docs/research/研究ノート/README.md`。同名fileでもbranchにより現在地が異なる。

共通入口は元証拠の索引で、研究の全内容を新しく再定義していない。原稿は優先作業ではなく、v0.4は科学的採否前のdraftとして参照する。

## GitHubで照合する基準branch・commit

|用途|branch|サーバー観測remote HEAD|
|---|---|---|
|H4元geometry結果|`track-a-h4-geometry-results-20261013`|`7c4483378047b57f514c47dad98905e3f2a9a756`|
|H4 B0+I結果|`track-a-h4-b0i-map-parallel-run02-20261014`|`98cdb35cd08469714f20d6cc3b982d9c80908f46`|
|H4統合v0.4|`track-a-h4-b0i-manuscript-v04-20261014`|`47dc17a83e2fc9216e22b59317969952ef79028a`|
|H6探索|`track-a-h6-roundoff-resume01-20261012`|`df99d6ee28942df8d12aac3c4db7cd202df55d77`|
|H6独立確認|`track-a-h6-hash-reuse-eight-worker-20261012`|`321cd8c3066194fe98741cbebbcdba1515d1deaa`|
|H6 rare技術解析|`track-a-h6-rare-order-cost-analysis-20261013`|`eee1343f6e99aa7661892ddfbe1067e974e2b74a`|
|H6 s2 pilot|`track-a-h6-conditional-s2-pilot-preparation-20261013`|`e0e6473d7047158a7f0f788d026a7a89a01898b8`|
|H6確認後GPTレビュー|`track-a-h6-scientific-review-publication-20261013`|`0dd5c3d204dab9b731268a1aa2add621080f3d20`|
|H4 native anchor/H8 state|`track-a-h8-saved-df-state-preflight-20261013`|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|
|H8 feasibility|`track-a-h8-fast-retry21-20261013`|`a55da5c2b012912c485184845aa73ca053ac00b9`|
|H8 production旧設計|`track-a-h8-resource-map-production-20261013`|`eaeeefc38be454375d141fabfb8ae229498175b9`|
|H8 resume01|`track-a-h8-resource-map-resume01-20261013`|`96b9560e6bc02a71a5917db5d0cf911a69aec3de`|
|H8 resume02・H6 pilot後レビュー|`track-a-h8-resource-map-resume02-20261014`|`38d386c8732295e6fab5d6fcbd47fbe586dc207d`|
|AX2B過去採択資料|`track-a-ax2b-h4-post-review-20261010`|`054bb7b2b2ca788b47ebdc5c14c115f24c8b6379`|

今回の新共有branchは `track-a-cross-environment-sync-server-20261014`。その固定commit・公開状態は作業完了時のチャット/公開監査を参照する。作成前のremote snapshotにこのbranchがないことは正常である。
全remote headsは[read-only snapshot](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/remote_branch_snapshot.json)。branch名に「preparation」があっても結果を含む場合があるため、名前からstageを推測しない。

## ローカル側で確認し、差分表として返す項目

- Repository/登録worktreeのHEAD・branch・dirty/untracked・未push。既存作業をreset/checkout/cleanしない。
- 上記各branchの現在remote SHAと固定基準との差。SOURCE・result・科学レビュー・原稿を分ける。
- main/ローカルbranchの研究概要が、H4「B0+I未測定」やH8「未開始」を現在情報として扱っていないか。歴史的原文を削除せず最新固定参照を添える。
- H4元集合1812点B2と、B0+I追加後397/1415の対象集合の違い。H4独立cost確認/rare-lawの未検証を補完していないか。
- H6探索8本、独立確認32本、条件付きs2 8本の区別と、追加H6 sampling一旦STOPという後続科学判断。
- H8最新runのSOURCE/seed/grant scopeを旧production設計と混同していないか。共有snapshotは最終resultでない。新snapshotはH8担当から受領する。
- 原子間距離1Å/H4〜H8/H5-H7候補/精度/T依存性など、今回利用者が示した将来候補とローカル側の計画が一致するか。科学採択・新計算承認へ読み替えない。
- サーバー外にしかない研究ノート、GPTレビュー、追加結果、採択/撤回/amendmentの有無。path・commit/blob/SHA・stage・根拠を付して報告する。
- 各研究のDF/PF/control/lowering/compiler/reference/u/P/selectionが一致するか、差を明示して別系列として扱うか。

差分の返却format例：

|情報ID/相対path|サーバー基準commit/blob|ローカルcommit/blob/dirty|一致/差分/未取得|条件・時点差|提案する処理|
|---|---|---|---|---|---|
|固定result/report|fixed evidence indexを参照|ローカルで確認|未記入|実行SOURCEとresultを分ける|読取比較、原本保全|

機微絶対pathやcredentialを差分表へ含めない。ローカル確認が揃うまで暫定初版を統一版へ変更しない。

## 想定される競合・未取得情報

サーバーmainにはstaged deletion958件、旧monitor worktreeには未commit11件＋untracked18件がある。原因/意図は本taskで確定していないため、原本保全のまま担当へ確認する。
Git untrackedの大量runtimeは未公開の原証拠であり、未pushの軽量研究資料と区別する。
同名remoteがないHEADでも別公開branchから取得できる場合がある。唯一到達性を確立できなかった旧run03 commit a3248055は必要性を確認してから移行を検討し、今無断公開しない。
H8最終費用map/独立確認結果は観測時未取得。H4/H6の母平均winner・厳密u/旧compiler同一性は未検証で、同期で解消できる情報欠損ではない。

## 必要な場合の移行候補

優先は軽量public reports/projection/承認済み科学レビューとそのbyte manifest。Git管理外レビュー4件は公開byte-identicalコピーが確認済みで、再転送は通常不要。
独立科学再現が具体的に必要になったときだけ、凍結NPZ・generation/freeze・元state/DF receipt・source/profile・native終了証拠・対象rawを、exclusive受領先/bytes/SHA/容量と別認可の転送案にする。
private venv、認証情報、internal endpoints、live H8 checkpointの無条件copyはしない。環境・絶対directory位置を一致させるだけの移行は要件ではない。
このhandoffの後も新scienceやH8操作は開始せず、情報差分照合を完了して報告する。
