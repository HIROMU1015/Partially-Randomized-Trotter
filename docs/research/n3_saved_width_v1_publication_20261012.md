# N3 保存証拠・幅評価準備：公開とremote再取得

2026-10-12 JST、専用HIROMU1015 branch `n3-saved-width-v1-preparation-20261012`。
完了済み科学レビューを受領し、今回認可された証拠整理・式の技術確認・比較仕様準備・syntheticだけを実施した。
**実8入力の新しい保証幅は未評価。事後評価は別の明示認可待ち、Mandatory STOP。**

- frozen source11: `3f295df0263059ce16d866f84c1132f8bace0670`
- source fingerprint: `b190ea508afb9a10fad3d3dbfeb92d50f46ad5813baadd70965aa6fc5ad044d1`
- source-bound plan: payload `d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a`、SHA `bd3f5f4a21deafb227a6794d3277ca3eab1d9a30c610236d42294103e91e53dd`
- [準備報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a/docs/research/n3_saved_width_v1_preparation_20261012.md)
- [数学契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a/docs/research/n3_saved_width_v1_mathematical_contract_20261012.md)
- [固定比較仕様](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a/docs/research/n3_saved_width_v1_comparison_specification_20261012.md)
- [小さい証拠viewの入口](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a/artifacts/n3_saved_width_v1/2026-10-12/saved_evidence/README.md)
- [実行plan（権限false）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a/docs/research/n3_saved_width_v1_execution_plan_20261012.json)
- [STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0fcf12fd8d3fb01795dbc544caa8eeb33c0449a/artifacts/n3_saved_width_v1/2026-10-12/preparation/MANDATORY_STOP.json)

GitHubへpush後、別bareへGitHubからfetchし、949 blobs / 12,455,820 bytesをbatch export、local byte/SHA/sizeと照合PASS。
新payloadはfetch前に別bareへ存在しなかった。過去GitHub取得objectsのcacheは使い、local worktreeのobject copy/alternatesは使わない。
元大ledgerは既存のimmutable objectとして取得し、別の同内容の巨大JSONを追加公開しない。新viewは各JSON8192 bytes以内。

remote exportで`run_n3_saved_width_v1.py --preflight`を実行し、source11、原pack/ledger、全view tree、1250 pointer roundtripを確認PASS。
幅・PSD・E_forced・eigensolver・optimizer・科学batchの再実行は0。24 synthetic testsのlocal logは取得してbyte照合し、remoteで新しい数値テストを回していない。
同一local環境での保存整合性であり、immutable CIや外部環境の科学batch再現ではない。

original source8/plan/input/run1/result/STOP、1904保護tree entries、旧manifest94 entries、root1347 files/HEAD/既存dirty entriesの保全PASS。
既存global manifestのdescription欠落FAILを保持し、raw開発失敗logの末尾spaceは元logとして保持する。source/docsのwhitespace検査とは区別する。

[compact publication receipt](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/publication_receipt.json)と
[分割payload inventory](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/publication_inventory/index.json)、
[補助source inventory](../../artifacts/n3_saved_width_v1/2026-10-12/preparation/remote_additional_source_inventory.json)。
payloadをexport後の再確認は、`python scripts/run_n3_saved_width_v1.py --preflight`または`prepare_n3_saved_width_v1.py --verify --directory artifacts/n3_saved_width_v1/2026-10-12/saved_evidence`。
どちらもidentity/JSON-pointerだけで、新幅を計算しない。

次は固定scopeに対する一回の数値的事後評価の実行認可。同じ科学レビューを再要求しない。
認可後も新κ/τ/Z、eig、E_forced、新入力、元batch再実行、N1/N2再開、実分子・多外部・一般二体拡張は含めない。
元NO_ADDITIONAL_ACCEPTANCE_OBSERVED/2-base未達/STOPと中心テーマnullを維持する。
