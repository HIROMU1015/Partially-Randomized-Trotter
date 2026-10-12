# N3 OCC v1 source準備：公開・remote再取得

2026-10-12 JST。専用branch `n3-occupation-transfer-v1-source-preparation-20261012`。
実装とsynthetic検査をsource commit `83f51cd8a4848572c66a5535c67d0d0a5f3868cd`へ固定し、
source-bound planとreadinessをpayload commit `6d877855656ee93c896228c6842c9928089c5b67`へ保存した。
この記録・receiptはその後のcommitで追加し、frozen source8ファイルを変更しない。

- [実装・検証報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d877855656ee93c896228c6842c9928089c5b67/docs/research/n3_occupation_transfer_v1_source_preparation_20261012.md)
- [レビュー修正を反映した追補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d877855656ee93c896228c6842c9928089c5b67/docs/research/n3_occupation_transfer_amendment_v2_20261012.md)
- [source-bound execution plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d877855656ee93c896228c6842c9928089c5b67/docs/research/n3_occupation_transfer_v1_execution_plan_20261012.json)
- [readiness audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d877855656ee93c896228c6842c9928089c5b67/artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/readiness_audit.json)
- [新8入力preflight（schemaだけ）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d877855656ee93c896228c6842c9928089c5b67/artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/source_bound_preflight.json)
- [STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d877855656ee93c896228c6842c9928089c5b67/artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/MANDATORY_STOP.json)

GitHubへpush後、別bare repositoryへGitHubからfetchして50blobs / 4,172,233 bytesを取得。
local byteとinventory SHA/size、source commitの8 blobs、入力・契約・planを照合PASS。
以前GitHubから取得したobjectsのcacheは使用したが、今回の新payload commitはfetch前に存在しなかった。
worktree objectsやalternatesは使用していない。完全にfreshなbase取得とは区別する。

remote exportのstdlib checkerで固定3syntheticの87,765 checksが保存auditと一致した。
新8入力は1692 schema checksのみで、predicate・eigensolver・ground・compile・科学batchは0。
旧12入力とT3の再探索も0。
source準備の24 synthetic testsはlocal PASSであり、remote再取得時にはsolver/generatorを再実行しない。
これはimmutable CIや外部環境での科学再現ではない。

基点の保護tree1842 entries、旧manifest91 entries、root既存1344 files/HEAD/dirty entriesを保持。
作業中に別Trackの新しいuntrackedレビューがrootへ追加されたため、追加として区別して保持した。
rootの既存dirty内容と別Trackのsource/STOPを変更・stageしていない。
既存global manifestの`relevant_commits[7].description`欠落FAILは保持する。

[機械receipt](../../artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/publication_receipt.json)に
全50fileのhash・remote exact audit・保全記録を保存した。
[再取得後の検査source](../../scripts/verify_n3_occupation_transfer_v1_source_publication.py)は
`--bare-repository`と`--export-directory`を指定して保存objectsを確認するだけ。
ネットワークfetch、generator、科学runner、eigensolverを呼ばない。

次は準備確認と明示実行認可の判断。同じ詳細科学レビューを再要求しない。
固定8入力の科学batchは未認可、Mandatory STOP、run1未作成、中心テーマnull。
N1追加探索停止・N2保留、実分子/多外部/一般二体への拡張停止を維持する。
