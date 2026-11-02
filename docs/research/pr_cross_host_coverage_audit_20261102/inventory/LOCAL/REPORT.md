# PR研究 Coverage監査 Phase 1 — LOCAL既存証拠インベントリ

対象は `HIROMU1015/Partially-Randomized-Trotter`。観測環境は **LOCAL**。ユーザーが明示したread-only Phase 1のみを実施した。これは既存証拠と保存解析版の所在・metadata台帳であり、新規科学評価、詳細科学レビュー、coverage不足判定、主要claimの採択ではない。

観測snapshot: `2026-11-02T23:08:19+09:00`。開始時の主証拠root HEADは `e098c54c78f589055082f9cfc2b13de50c90ca94`。観測は非atomic。各固定Git objectの内容は不変だが、現在の可変worktree全体を凍結したとは主張しない。

## 固定仕様・規約

- [固定PROTOCOL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md)、同commitの[43列テンプレート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/INVENTORY_TEMPLATE.csv)をGitHub connectorで取得し、本文・列辞書・列順を確認した。schemaは `PR-COVERAGE-1`。
- [LOCAL用固定指示](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e440e5c06853599b32f678c27c01ab817c1df120/docs/research/pr_cross_host_coverage_audit_20261102/HOST_INSTRUCTIONS.md)も取得した。PROTOCOL/template/HOSTのGit blobはそれぞれ `3ea836949db06d1f077e8594b187840bf9e23971` / `c62cf3c286a539e41d75b5012f9e3da030a8b986` / `c7e886df38cb38464d52cd39777c0b6ade702268`。connector返却値とLOCAL固定Git objectが一致した。
- 主worktreeの実在する `AGENTS.md` と `research_project_instructions_unified_2026-10-25.md`（特に§8.1、§8.4、§10）を実際に読んだ。ルートから親projectsまでの各 `AGENTS.md` 候補も確認し、適用される親ファイルは発見されなかった。追加の証拠worktreeをcheckoutしておらず、その内容規約を読了とはしていない。出力treeは主証拠treeと別の専用Git/indexである。
- **Phase 0 provenance訂正注記**: 固定PROTOCOLには担当LOCALの記述が残るが、ユーザーの本指示による実担当は **SERVER 2 Codex**。固定仕様は編集していない。本Phase 1の担当はLOCAL。旧SERVER_A/Bを今回SERVER 1/2へ対応付けていない。

## 読取root・許可範囲・保全

| 公開用ID | 今回確認した範囲 | 限界 |
|---|---|---|
| LOCAL-EVIDENCE-ROOT-01 | ユーザー所有の主repository、対応remote、現在の入口文書、限定した未commit V4記録、固定Git object store | dirty状態を保持。現在checkoutだけを研究全体とは扱わない |
| LOCAL-REGISTERED-ROOT-SET | 既存のGit worktree登録から、存在と同一ユーザー所有を確認した242 rootの登録metadata | 主rootを含む。内容の全走査や各rootのraw完全監査はしていない。個々の所在逆引きはprivateに保持 |
| LOCAL-ARCHIVE-01 | 既知H6 exporter ZIPのmember一覧と小さなFILE_HASHES/依頼文書 | exporterを実行せず、NPZ等を生成しない。公開済みpacketへの既存対応を再利用 |
| LOCAL-ARCHIVE-02/03 | 既知representation/H4 geometry算術レビューZIPのcentral-directory一覧 | 既存の検算scriptを実行せず、全展開・数値検算をしない |
| LOCAL-PRIVATE-EVIDENCE-0001 | 主rootにある既存uncommitted V4 correctness JSON。mtime/size安定を確認してmetadata読取 | 非公開原本のパス・rawを公開せず、60条件の参照のみ匿名化。公開Git上の同ファイル存在とは扱わない |

全home/projects/tmpを横断した再帰探索はしていない。登録済みlinked worktreeは同一Gitとの対応を登録metadataで確認した。旧環境inventoryにある独立clone・bare・別保存領域は、公開匿名版だけでは現在の物理所在と全て結合できず未解決とした。無関係なrepository、他ユーザー、process/job、共有設定へアクセス・変更していない。SERVER 1/2へのSSH・転送・調査は0件。

証拠treeへのGit操作は `GIT_OPTIONAL_LOCKS=0`、`GIT_NO_LAZY_FETCH=1` を付けた `show/ls-tree/cat-file/rev-parse` 等の読取を使用した。科学SOURCE/input/raw/result/grant/ledger/marker/STOP、既存研究map・概要・validation manifestは変更していない。証拠treeのcheckout/reset/merge/clean/gc/prune、branch/worktree/archiveの削除・cleanupは実施していない。

## 再利用した固定カタログ

| 既存入口 | 版 | 今回の対応 |
|---|---|---|
| [PURPOSE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究目的・研究課題.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [METHOD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究方法・解析手順.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [SUMMARY](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究概要・現状.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [MAP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/PROJECT_MAP.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [STATUS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/VALIDATION_STATUS.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [MANIFEST](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/validation_manifest.json) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [LANDSCAPE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [CLAIMS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [TRACKA](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md) | `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | 取得済み。以下の系列/参照対応表へ接続 |
| [CARDS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md) | `a9bdcf1d44061daceffe896637e2c92fcb3353b4` | 取得済み。以下の系列/参照対応表へ接続 |
| [SOURCEINDEX](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/SOURCE_AND_PROVENANCE_INDEX.json) | `a9bdcf1d44061daceffe896637e2c92fcb3353b4` | 取得済み。以下の系列/参照対応表へ接続 |
| [ENVLOCAL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/local/LOCAL_INVENTORY_REPORT.md) | `fa55c162fd8fe82bd8ccd91f5fb7601d8513791e` | 取得済み。以下の系列/参照対応表へ接続 |
| [ENVA](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/server_a/SERVER_A_INVENTORY_REPORT.md) | `fa55c162fd8fe82bd8ccd91f5fb7601d8513791e` | 取得済み。所在候補のみ。他環境の現在実体は未確認 |
| [TLOCAL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/track_a_local_inventory.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。以下の系列/参照対応表へ接続 |
| [TSERVER](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/track_a_server_inventory.md) | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | 取得済み。所在候補のみ。他環境の現在実体は未確認 |

主rootの `PROJECT_MAP.md` を先に読み、`docs/research/研究概要・現状.md` を全文確認した。そのcheckoutの9/28時点の停止位置と、固定landscape/後続結果の時点差を区別した。固定版の目的・方法・status/manifest・landscape/claim表もmetadata抽出に使用した。固定32ファイルSOURCE catalog、70カードとSOURCE_AND_PROVENANCE_INDEX、既存58行study_indexの索引構造を再利用し、それらを独立実験の件数へ変換していない。

既存 `check_validation_manifest.py`、`inventory_pauli_hybrid_rte.py`、`document_pauli_jw_saved_summary.py`、`integrate_pauli_h2_h4_h6_saved_results.py` の読取・literal CSV転記・参照検査部分を静的確認した。CSV DictReader/DictWriterとJSON数値を文字列のまま読む方式、固定Git参照検査の考え方を再利用した。既存runnerは特定系列の出力先・科学モジュール・再会計へ接続するためimport/実行せず、本schema用の小さなstdlib-only一時adapterで転記した。新しい管理基盤・approval証明・grant発行は作っていない。

## 条件別に記録した保存証拠

[INVENTORY.csv](INVENTORY.csv) は **9,665 metadata観測行**、43列、UTF-8/RFC4180。これは実験数、独立run数、独立cohort数、科学的成功数ではない。DETAILS.jsonlは不要だったため作成しない。

行は元CSVの1-based `CSV_DATA_ROW`（ヘッダを除く）またはJSON pointerに結合する。固定commit/path/rowで元条件・metricを復元できる。PF/finite-RTEはstateとq/roundを分け、保存v4/v5を別解析版として保持した。H4 geometryは候補別のbias/B/価格取得を記録し、302精度への新しい再会計は行っていない。原CSVの空欄はNOT_RECORDED、元JSONの科学的nullは参照先で保持し、引用する場合は文字列 `"null"` とする。0で補完していない。

evaluationには元結果のfield/row参照を保存し、新しいN/G・平均・SE・比・fit・順位を計算していない。全metricをCSVへ複写せず、原値のexact文字列・定義・対象軸を保存版で参照する。`source_commit` は原SOURCE bindingとfull commitの存在を確認できる場合に限り記録した。dirty検証のgit_commitはclean scientific SOURCEへ格上げせず、code_refsに記録済みbaseと元provenanceを残した。元run名の無い場合はrun_idを発明せずNOT_RECORDEDとし、元cohortをsamplingへ残した。LOCAL-EXP/LOCAL-COND/LOCAL-EGは本観測の仮IDで、同一runや環境間identityを認定するIDではない。

scientific_statusは保存されたpass flag等の範囲でのみ正規化し、technical_status・正式受理・validityを別にした。探索の費用最小やexit0だけでは科学PASS/正式受理にしない。既存判定が行単位で未結合のものはUNKNOWNであり、新判定を発行していない。

| 系列 | 観測行数 | 固定保存表・行単位 |
|---|---:|---|
| `FINITE_RTE` | 96 | [h4_sto3g_d100_rank12_ld0_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld0_v1.json); RECORDED |
| `FINITE_RTE` | 96 | [h4_sto3g_d100_rank12_ld1_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld1_v1.json); RECORDED |
| `FINITE_RTE` | 150 | [h4_sto3g_d100_rank12_ld3_dt0p025_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p025_v1.json); RECORDED |
| `FINITE_RTE` | 150 | [h4_sto3g_d100_rank12_ld3_dt0p05_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p05_v1.json); RECORDED |
| `FINITE_RTE` | 150 | [h4_sto3g_d100_rank12_ld3_dt0p1_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p1_v1.json); RECORDED |
| `FINITE_RTE` | 150 | [h4_sto3g_d100_rank12_ld3_dt0p2_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p2_v1.json); RECORDED |
| `FINITE_RTE` | 150 | [h4_sto3g_d100_rank12_ld3_dt0p4_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p4_v1.json); RECORDED |
| `FINITE_RTE` | 96 | [h4_sto3g_d100_rank12_ld3_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_v1.json); RECORDED |
| `FINITE_RTE` | 96 | [h4_sto3g_d100_rank12_ld6_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld6_v1.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld0_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld0_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld0_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld0_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld10_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld10_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld10_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld10_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld11_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld11_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld11_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld11_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld1_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld1_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld1_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld1_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld2_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld2_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld2_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld2_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld3_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld3_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld3_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld3_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld4_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld4_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld4_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld4_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld5_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld5_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld5_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld5_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld6_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld6_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld6_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld6_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld7_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld7_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld7_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld7_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld8_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld8_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld8_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld8_v5.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld9_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld9_v4.json); RECORDED |
| `PF_DELTA` | 36 | [h4_sto3g_d100_rank12_ld9_v5.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld9_v5.json); RECORDED |
| `H4_GEOMETRY` | 1308 | [saved_candidates_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json); RECORDED_PRICE_BIAS_CONDITIONS; saved precision grids referenced, not expanded |
| `H8_SIX_AXIS_SAVED_REACCOUNT` | 3800 | [accounting_rows_exact.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/accounting_rows_exact.csv); RECORDED |
| `PR2_M1_B1` | 210 | [pr2_matched_accuracy_m1_b1_compile_map_result_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json); /compile_map |
| `PR2_M2` | 5 | [pr2_matched_accuracy_m2_transfer_result_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json); /candidate_results |
| `PR2_S2_B2` | 12 | [pr2_s2_development_resource_result_parallel_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json); /B2_candidates |
| `PR2_S2_B3` | 12 | [pr2_s2_development_resource_result_parallel_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json); /B3_candidates |
| `PR2_S2_DETERMINISTIC` | 2 | [pr2_s2_development_resource_result_parallel_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json); /deterministic_candidates |
| `DF_LONG_B2` | 320 | [best_configurations.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/best_configurations.csv); RECORDED |
| `DF_LONG_B3` | 160 | [best_configurations.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/best_configurations.csv); RECORDED |
| `DF_LONG_B3` | 40 | [fixed_nominee_confirmation.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/fixed_nominee_confirmation.csv); RECORDED |
| `PAULI_H2_CONFIRMATION` | 55 | [candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv); RECORDED |
| `PAULI_EVENT_EXTENSION` | 50 | [candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pauli_rte_event_extension/2026-10-26/candidate_precision_table.csv); RECORDED |
| `PAULI_H4_B0` | 530 | [candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_precision_table.csv); RECORDED |
| `PAULI_HYBRID_RTE` | 15 | [new_candidate_accounting.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pauli_hybrid_rte_comparison/2026-10-25/new_candidate_accounting.csv); RECORDED |
| `PAULI_H6_FAIRNESS` | 40 | [CANDIDATE_RESULTS.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv); RECORDED |
| `RPE_FOUR_ROUND_ACCOUNTING` | 4 | [h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_limited_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/rpe_four_round_accounting_validation/2026-09-18/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_limited_v1.json); RECORDED |
| `RPE_FOUR_ROUND_SIGNAL` | 4 | [h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/rpe_four_round_phase_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json); RECORDED |
| `RPE_SAVED_ROUND_SCHEDULE` | 56 | [h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_round_schedule_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/rpe_delta_round_schedule_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_round_schedule_v1.json); RECORDED |
| `PR2_LOCAL_V4` | 60 | LOCAL-PRIVATE-EVIDENCE-0001; RECORDED_REDACTED; original uncommitted retained |
| `H4_GEOMETRY_B0I` | 120 | [measured_candidates.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/measured_candidates.json); RECORDED |
| `H5_NEUTRAL_PILOT` | 6 | [input_state_signal_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/artifacts/resource_applicability/track_a_h5_neutral_open_shell_pilot_v1/2026-10-16/input_state_signal_v1.json); RECORDED_SIGNAL_CONDITIONS |
| `H7_NEUTRAL_PILOT` | 6 | [input_state_signal_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162c256ea39b4ab3dadde65a94b3eec73cea4720/artifacts/resource_applicability/track_a_h7_neutral_open_shell_pilot_v1/2026-10-16/input_state_signal_v1.json); RECORDED_SIGNAL_CONDITIONS |
| `PF_SIZE_SAVED` | 9 | [h2_h5_operational_c_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h2_h5_operational_c_v1.json); RECORDED |
| `PF_SIZE_SAVED` | 9 | [h2_h6_paper_d6_c_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h2_sto3g_d100_rank3_ld1_pf_d6_validation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h2_sto3g_d100_rank3_ld1_pf_d6_validation_v1.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h2_sto3g_d100_rank3_ld1_pf_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h2_sto3g_d100_rank3_ld1_pf_v4.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h3_sto3g_d100_rank5_ld2_pf_d6_validation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h3_sto3g_d100_rank5_ld2_pf_d6_validation_v1.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h3_sto3g_d100_rank5_ld2_pf_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h3_sto3g_d100_rank5_ld2_pf_v4.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h4_sto3g_d100_rank7_ld3_pf_d6_validation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h4_sto3g_d100_rank7_ld3_pf_d6_validation_v1.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h4_sto3g_d100_rank7_ld3_pf_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h4_sto3g_d100_rank7_ld3_pf_v4.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h5_sto3g_d100_rank9_ld4_pf_d6_validation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h5_sto3g_d100_rank9_ld4_pf_d6_validation_v1.json); RECORDED |
| `PF_SIZE_POINT` | 8 | [h5_sto3g_d100_rank9_ld4_pf_v4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pf_c_system_size_validation/h5_sto3g_d100_rank9_ld4_pf_v4.json); RECORDED |
| `PAULI_H4_PR_SEARCH` | 725 | [candidate_precision_table.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/candidate_precision_table.csv); RECORDED |
| `PR2_OLD_S0` | 2 | [pr2_s0_validation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_validation_v1.json); RECORDED |
| `PR2_REBASELINE_V1_V3` | 4 | [pr2_v1_v3_result_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json); RECORDED |
| `RPE_M06F_SAVED_MODEL` | 36 | [wp11_all_r_opt2_coherent_analysis_20260925_065827.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json); RECORDED |
| `COMPILED_ADDITIVITY` | 1 | [h4_sto3g_d100_rank12_ld3_dt0p1_r1_k0_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r1_k0_v1.json); RECORDED |
| `COMPILED_ADDITIVITY` | 1 | [h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_mc300_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_mc300_v1.json); RECORDED |
| `COMPILED_ADDITIVITY` | 1 | [h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_v1.json); RECORDED |

DF長時間は保存nomineeと確認表を記録した。全signal grid・全r/K感度表を新たな行へ全展開した意味ではない。Pauliは保存candidate×precision×cohortの表を転記し、確認集合minと結果前固定nomineeを再選抜していない。精度間の共有価格・cos/sinのpaired標本・signal再利用は独立観測増加とは扱わない。

## 旧索引19系列との対応と未解決箇所

次表の「未解決」は棚卸し上の参照結合・展開の状態。科学的coverage不足や追加計算義務の判定ではない。旧索引から辿った135の参照候補は固定treeの存在/読取状態を確認した。

| study_id | 今回の状態 | 継続して参照する固定資料 |
|---|---|---|
| `H4_GEOMETRY` | 候補1308の価格/bias条件を記録。保存302精度全表の行展開は未解決 | [7c4483378047](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md) |
| `H4_GEOMETRY_B0I` | 測定候補120を記録。精度再会計/6再使用軸のrun結合は未解決 | [98cdb35cd084](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md) |
| `H4_NATIVE_ANCHOR` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [0f96a2e45ba5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md) |
| `H5_NEUTRAL_PILOT` | 6 signal条件を記録。各価格標本/精度会計の詳細結合は未解決 | [bde58ba23f1f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md) |
| `H6_CONFIRMATION` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [321cd8c30661](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md) |
| `H6_NATIVE_B0I` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [34efc914b2f4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md) |
| `H6_STRENGTHENED136` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [f89c40af7328](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_reaccount_results_20261016.md) |
| `H6_P7_MEASURED_PAIR` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [f89c40af7328](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md) |
| `H6_FIXED_DELTA_T` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [043f58ac9336](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md) |
| `H6_RARE_SAVED_AUDIT` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [eee1343f6e99](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md) |
| `H6_CONDITIONAL_S2` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [0f96a2e45ba5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md) |
| `H7_NEUTRAL_PILOT` | 6 signal条件を記録。各価格標本/精度会計の詳細結合は未解決 | [162c256ea39b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162c256ea39b4ab3dadde65a94b3eec73cea4720/docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md) |
| `H8_ORIGINAL_MAP` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [681ec23c93fd](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md) |
| `H8_D0` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [3d910ee1de45](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md) |
| `H8_PHASE_C_SIGNED` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [6dc5bf0de01a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md) |
| `H8_NONCANCELLATION_COST` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [23d46e489af0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md) |
| `H8_KEY_SIX_AXES` | 原報告/索引・固定参照を確認。run/cohort/条件別の詳細展開・受理状態の個別joinは未解決 | [ce6cd8ead5c0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6cd8ead5c0311ec72165c04869e7df4050b75b/docs/research/track_a_h8_key_compile_completed_results_20261017.md) |
| `H8_CURRENT_SIGNED` | 保存accountingの条件行を記録（通常/非相殺・factorは元欄参照） | [96e69a01d1fa](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md) |
| `H8_CURRENT_NONCANCELLATION` | 同じ保存表の別bias_kindとして記録。通常と合算して独立実験にしない | [96e69a01d1fa](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md) |

E01〜E32のsource catalogは各固定commit/pathをLOCAL Gitで取得した。E03等の比較条件表、E29〜E31の既存レビューは所在/時点/判定出典として扱い、レビューを再実施していない。旧19系列と上記CSV行の観測単位は異なり、系列数から独立実験数を推定しない。

## 70研究カードとSOURCE索引の扱い

各カードを名前だけで自動算入せず、既存の目的・評価対象を参照した。改善方式のcoverageは除外する。SUPPORT_ONLY候補は「改善研究にある既存PR baseline/検証」だけの所在候補であり、改善方式全体の支持・受理ではない。今回条件を一意に結合できないものは実験行を発明せず、以下へ未解決として残した。

| カード | 今回の対応 |
|---|---|
| REP-01 Pauli groupingとDF回路基盤 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| REP-02 旧DF rank screening | 失効した旧DFの所在追跡。肯定証拠へ算入しない。修正後canonical入力は未解決 |
| REP-03 BLISS型UWC＋grouped PF | UWC原JSON/診断rawは既存限定探索でもNOT_FOUND。今回は原物の未発見を未実施とはしない |
| PF-01 Yoshida・Morales・新4次係数の比較 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| PF-02 Processed PF／YP8m8 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| PF-03 PF固有位相・D6係数とsurvival代理量 | PF_DELTA/PF_SIZE_POINT/PF_SIZE_SAVEDに記録 |
| PF-04 有限時間校正・第一研究S0–S4 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| PF-05 有限時間校正・第二研究Direction C | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| PF-06 FS-C旧入力回収とFS-R1校正follow-up | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| PF-07 BCH高次項・response・direct-scaling | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| CIR-01 WP06-a/b full/support-union基底計画 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| CIR-02 P-A interval-union DP | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| OPT-01 round別r/K・誤差/失敗確率配分 | RPE保存round/schedule/M06-Fを記録。旧WP全解析版の展開は未解決 |
| OPT-02 P-B signal-aware selection | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| OPT-03 P-C geometry tracking／energy差 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| OPT-04 P-D energy/tail-aware高次PF選択 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| OPT-05 R3 split／formulaのmulti-fidelity選択 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| ERR-01 FR 位相・半径分離 | 改善方式自体は範囲外。既存PR baseline/回路検証のSUPPORT_ONLY候補は索引を保全、条件別join未解決 |
| PR-01 PR-2 圧縮＋残差random補完 | PR2旧S0/新系列/V4/S2/M1/M2を記録。PM1/PM2全表の結合は未解決 |
| PR-02 PR-3 tail限定Richardson | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| PR-03 PR-1/4/5/6初期候補群 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EVAL-01 Track A PM-1/PM-2 precision資源地図 | M1/M2とH4保存候補を記録。PM2全精度表の展開は未解決 |
| EVAL-02 Track A H4–H8強対照と希少tail | H4/H5/H7/H8の条件記録と19系列参照対応。独立確認/希少層の全詳細は未解決 |
| EVAL-03 PR高次PF探索と次数逆転の既存証拠 | 高次PF/長時間の保存nomineeを記録。共通588/高次独立確認の全原条件結合は未解決 |
| RB-01 B-F finite-RTE係数設計 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-02 B-M DF multirate／compact BCH | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-03 BS block合成・SP synthesis placement | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-04 R0/R1 finite-mean再配分 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-05 RA-D0 LP・G1–G5固定辞書 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-06 G6 Return Aggregation／General Full | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-07 G7 conditional native economics | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-08 G8 on-demand synthesis／provider budget | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-09 G9 Closed P5 matched-native | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-10 G10 degree-matched native比較 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-11 DF希薄三次補正・generic whole | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-12 Gaussian積生成 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-13 D1–D3 finite-target・境界相殺・保証価格 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-14 Parent allocation／S4 KKT | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-15 M1-S shared ± labels | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-16 G6 finite-bit／B3 live API・古典費用 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-17 C1 G6–DF移植・H4 logical pilot | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| REP-04 N1 representation selector | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| REP-05 N2 signed charge・L1 fit | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| REP-06 N3 occupation／active-energy certificate | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| REP-07 THRIFT／mixed relative evolution | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-01 C1 macrocycle／符号相殺 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-02 C2 branch付きPF外挿 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-03 C3 self-energy／active space | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-04 C4 cost-aware proposal／whole path | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-05 一体再配分・exact-ground PF応答 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-06 H-E1・M-a/M-b PF小改変 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-07 SOS-PF生成・K0/K0.5 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-08 A2 RF0/RF1/EW2有限step相対保証 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-09 DF covariance縮約 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| REP-08 非DF Pauli対DF fixed-signal比較 | Pauli-JW candidate/precision/cohort表を記録。D1/D2/6局所Arbの詳細展開は未解決 |
| EXT-01 P1* bus/CAR固定word | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-02 J1 joint-loop・echo構成 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-03 Q1 integrable level insertion | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-04 集団遷移・pair collision・R1-P | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-05 G coherent adaptive subspace | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-06 固定条件付き測定 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-07 B-01 unknown-gap／energy情報 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| EXT-08 T1 logical accreditation・独立原理P1–P8探索 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| REP-09 core吸収・physical pair表現 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| RB-18 interstep correlated randomization | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-10 H1 先導固有値補正 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-11 H2 mixed correlation／filter | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-12 H3 frame／interaction picture | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-13 H4 copies／phase reference | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |
| QC-14 exact fermion encoding／controlled構成 | 範囲外：独立したアルゴリズム改善/新規提案または別repository専用研究。必要な既存PR baselineの個別対応は未解決 |

SOURCE_AND_PROVENANCE_INDEXのrepository/commit/path/roles/既存identityを再利用した。別repositoryの科学SOURCE・grant・STOPや実行許可を転用していない。外部PF repositoryのファイルはこの主repositoryの直接観測へ混入させず、カードからの二次参照に留めた。

## status/manifestと後続索引の対応

validation_manifestの登録行は結果だけでなく準備・契約・技術状態も含む。以下はresult_sets IDごとの棚卸し状態で、completion criteriaの採否・不足優先度を新評価していない。トップレベルの新しい個別study登録も、同一固定manifestへ戻ることで保持する。

| manifest result_set ID | 今回の対応 | 元status |
|---|---|---|
| `legacy_high_order_pf` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `historical_evidence` |
| `df_partial_randomized_screening` | 旧失効/原証拠未発見を出典付きで保持。肯定科学結果へ算入せず、条件が未確定のため新行なし | `invalidated_pending_regeneration` |
| `uwc_grouped` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `prose_only_missing_raw_artifacts` |
| `tests` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `medium_q_rpe_hadamard_benchmark` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `large_q_rpe_hadamard_affine_proxy` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_hadamard_proxy_resource_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_allocation_sensitivity_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_four_round_accounting_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_four_round_phase_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_target_round_horizon_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_delta_round_schedule_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_delta_compiled_cost_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `research_direction_prevalidation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_ablation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_pf_sensitivity` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_gate_s1` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_structure_pilot` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_sequence_policy` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_full_scope` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_full_scope_extension` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_decision_cost` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_late_round_proxy` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_compiler_transfer` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_uncertainty_break_even` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_wp11_synthesis` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_full_opt2` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `research_direction_signal_weight_pilot` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_geometry_energy_difference_pilot` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `finite_rte_signal_validation` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `rpe_round_cost_connection_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `rpe_hadamard_failure_validation` | 関連RPE保存round/modelを記録。全旧版/全rawへのjoinは未解決 | `source_present_no_current_ci` |
| `random_circuit_cost_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `rte_boundary_cost_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `rte_boundary_pair_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `hierarchical_compiled_cost_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `rte_order_stratified_cost_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `rte_connected_cluster_cost_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `rte_system_size_cost_validation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pf_delta_surrogate_validation` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `pf_c_system_size_validation` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `research_direction_joint_synthesis_pilot` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_theme_selection` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `research_direction_joint_synthesis_blind_validation` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_joint_synthesis_formalization` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_joint_synthesis_mechanism_validation` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_geometry_tracking_breakdown` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_energy_tail_pareto` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_pd_realization` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_pd_fair_comparison` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `research_direction_pd_s1_posthoc_reanalysis` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `finite_rte_phase_amplitude_fr1` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `finite_rte_phase_amplitude_fr_revision_r0` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `finite_rte_phase_amplitude_fr_revision_r1_plans` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `finite_rte_phase_amplitude_fr_revision_fr1a` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `finite_rte_phase_amplitude_fr_revision_fr1b` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `finite_rte_phase_amplitude_fr_completion_contract` | 改善方式のcoverageは範囲外。既存PR baselineのSUPPORT_ONLY候補は未解決 | `source_present_no_current_ci` |
| `pr2_pr3_minimal_pilot` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_s0_s1_validation` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `pr2_new_series_v1_v3` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `pr2_v4_s2_development` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_resource_contract` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_implementation_contract` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_preexecution_amendment_v2` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_execution_authorization` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_a` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_b1_contract` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_b1_execution_source` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m1_b1_result_validation` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m2_transfer_contract` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m2_transfer_contract_revision_v2` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m2_execution_source` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m2_execution_authorization` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_matched_accuracy_m2_transfer_execution` | 関連条件をINVENTORYへ記録。各行のUNKNOWNと原保存版の制限は保持 | `source_present_no_current_ci` |
| `pr2_track_a_pm0_evidence_attribution` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm1_nearby_discard_preparation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm1_execution_authorization_draft` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm1_authorization_finalization` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm1_discard_execution` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm2_precision_resource_preparation` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm2_precision_analysis_source` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr2_track_a_pm2_precision_resource_analysis` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `source_present_no_current_ci` |
| `pr_pf_s2_recomputed_20261022` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `LOCAL_COST_PAIRS_WITH_FAILED_AUXILIARY_DIAGNOSTICS_MANDATORY_STOP` |
| `pr_pf_analytic_vs_circuit_CNG_20261022` | 未解決：索引・保存statusを確認。条件/run/保存解析版の詳細展開は未完 | `SAVED_CNG_ACCOUNTING_COMPLETE_N_INDEPENDENT_MISSING_CONDITIONAL_PHYSICS` |

現在の研究結果統一索引も確認した。H4 P3 Stage3 run03（保存snapshot60515a33…）のDF–Pauli matched比較は関連する既存検証の参照候補だが、今回の条件別joinは未解決。Stage4設計を実施済み結果へ昇格させていない。研究B/B2-D、研究C、Track Bの改善方式は範囲外とし、当該系列内のPR baselineの必要性・同一性は後続照合事項として残す。

## 重複・独立確認・回復・失効の候補関係

| 対象 | 元資料にある関係と今回の扱い |
|---|---|
| 固定Git carrier・archive・複数branchのコピー | commit/path/blobまたは既存result IDで所在候補を結ぶ。同じ値・同じbranch名だけで独立実験にしない |
| PF v4/v5、dirty baseと後続文書 | 別保存解析版。生SOURCE差・入力identity・実行時刻が未結合ならRERUN/独立cohortを確定しない |
| PR2旧S0→rebaseline | 旧STOP_INPUT_REPRODUCTION_MISMATCHを保持。新入力系列はCONDITION_VARIANT候補で、旧STOP解除・成功再解釈ではない |
| M1→M2 | developmentで固定した候補のgeometry transfer。移送先の再最適化・独立全探索としない |
| H4 geometry→B0I | 比較集合の条件変更。再使用6軸と新234軸の旧記録を保持。240独立新実験と数えない |
| H6 p7、H8六軸置換 | 単価の保存再会計版。旧価格標本・shot/signalのREUSES候補。model単価を実測へ一括置換しない |
| H8独立主表とMechanism cohort | 既存資料がaliasと記す。コピー/引用候補であり追加独立確認へ加算しない。詳細公開監査とのjoinは未解決 |
| 長時間B2/B3探索と確認 | 固定nominee/cohortの関係を原記録付き候補として保持。B3確認表は探索表と別cohortだが、3環境間のcanonical identity・非重複を今回確定しない |
| B3中断・復旧 | 原provenanceは106 PASS再利用、後続133記録再利用、一部event-only再構成を記す。CONTINUES/RECOVERS/REUSES候補。元STOP・元byte欠落の制約を保持し、新独立標本へ加算しない |
| Pauli追加event/Hybrid確認 | 条件変更・追加確認・共有target/Nを区別。5精度は価格標本を共有する。同一seed再実行や確認後minを独立の結果前固定確認としない |
| 旧DF screening/UWC | 失効したDF旧入力37ファイルのhistorical identityと、修正後canonicalの未解決を区別。UWCは原JSON/診断raw未発見。後続結果を根拠に失効値を復活させない |

全環境の同定・canonical重複排除・独立run/cohort数の確定はPhase 2へ残す。CSVの各元参照は一意だが、別表からの引用・同じsignal/priceの再使用が科学的に独立であることを保証しない。

## 未発見・未アクセス・未解決と探索の限界

- `NOT_FOUND`: 既存GAP-001のUWC原JSON/theta sweep/診断raw、GAP-008の修正後canonical入力等は、[保存済み限定探索](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dfdfa6460f9e7d25a9595132a672e8de2a5dc7e/docs/research/pr_algorithm_improvement_landscape_20261023/local_environment_search_20261023/SEARCH_REPORT.md)の状態を引用。今回新しい広域探索を実行した意味ではない。旧無効化37ファイルのGit存在と、修正後入力不明を混同しない。
- `NOT_ACCESSIBLE`: SERVER 1/2の現在のfilesystem/raw、許可対象外の保存領域は今回読取範囲外。GitHub/LOCALにある報告の写しを他環境の現存確認と呼ばない。旧SERVER_A/Bの対応はUNKNOWN。
- `UNKNOWN`/未解決: 242登録rootのうち内容未展開の領域、旧独立clone/bareの現在対応、run時の完全SOURCE差分、入力/state/reference/lawの科学的identity、seedの全非重複、raw/QPY全量、正式受理の行単位join。未調査の可視領域をNOT_ACCESSIBLEとはしていない。
- `NOT_RECORDED`: 行の限定selectorにない元run IDや空欄。全資料に存在しないという断定ではない。元ファイルの別object/contractにある情報でも行へ未結合ならUNKNOWNを保持し、missing_fieldsで限定範囲を示す。
- 大gridの未展開: 長時間B2のcandidate_precision_accounting/r_K_sensitivity、B3のR_K_sensitivity/matched controls、旧H6/H8全探索・全希少層、H4の302精度/PM2、共通PF588、高次独立確認、D1/D2/Arbの全局所条件、旧connected-cluster/全WP解析版は固定索引へ対応付けたが、全行をこのCSVへ展開していない。これは未実施や科学的coverage不足の判定ではない。
- B3公開ebe9e2aa…と旧資料が記すlocal最終f794a2aa…の差、H4親native exit/未丸めnormalization/旧event、H8確認詳細、古いSOURCE packetの公開/保全差は未解決。これを解くためのRNG replay、復元再計算、raw再構成は実施しない。

## 整理・branch削除・archive記録の利用

LOCALの既存TSVは列定義を確認した。`remote_branch_cleanup_inventory_local_20261102.tsv`（745行）、handoff/result各372行、`remote_branch_archive_result_20261102.tsv`（354行）から、branch名不在と固定commit喪失を区別する所在追跡方法を再利用した。現在の保持先をここで再認定したり、削除・cleanupを再実行したりしていない。

`server2_repository_inventory_20261102.tsv` とLOCAL crosscheck（各485行）は、LOCAL保存の過去metadataとして列だけを確認した。SERVER 2のrun/raw現存、job状態、権限、現在のHEAD確認として転記しない。絶対path・内部host・IPは公開しない。

既存[KEEP保全監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa55c162fd8fe82bd8ccd91f5fb7601d8513791e/inventory/branches/github_branch_cleanup_execution_20261027/KEEP_REF_PRESERVATION_AUDIT.md)と旧LOCAL inventoryを読んだ。旧保持監査は369科学参照commit、archive/bundle/保持依存等への導線であり、今回の科学的再検証や独立障害域backupの証明ではない。新しいref復旧・archive全展開・全payload hash・cleanupは0件。

## 公開用匿名化・技術的整合性確認

公開差分はこの `INVENTORY.csv` と `REPORT.md` のみ。内部host/IP、ユーザー名付き絶対path、資格情報、非公開raw、元TSVのprivate所在列、実行環境の詳細を含めない。必要なroot/証拠IDの逆引きはLOCALの非公開一時作業記録に保持し、commitしない。公開済みfixed commit/path/元IDへのリンクは保存する。

主agentによるstdlib-only検査で、43列と固定列順、UTF-8/CSV roundtrip、JSON構文・条件objectの必須key、ID一意性、元CSV row/JSON pointerの解決、固定Git blob/commit存在、公開refの形式、元literal欄、欠測token、数値0/null非変換、秘密・絶対path非混入、許可差分2ファイル、主証拠treeのHEAD・active branch・既存tracked status保持、および並行変更との差分を確認する。これは科学runner/会計checker/独立科学レビューのPASSではない。metadata検査はPASS_METADATA_ONLY：9,665元row/pointer解決、20,787条件literal欄と21,616保存metric literal欄、74固定blob参照を確認した。RFC4180のCRLFはGitのcr-at-eol設定をコマンド内だけで有効にして差分検査した。commit SHA、remote SHAと公開本文の一致は公開後の最終チャットに記す。

snapshotは元資料を新しい科学SOURCEへ固定し直すものではなく、既存raw、grant、消費済marker、one-shot/retry条件、STOPを全て維持する。metadata処理にCPU/RAM/workerの人工上限は設けず、固定Git資料はキャッシュした。重い科学処理を起動していない。

観測中に別系列の研究C branchとCodex補助refの更新、他環境TSVの未追跡追加を検出した。本作業のGit mutationは独立出力repositoryだけであり、それらへ変更・巻戻し・凍結要求はしていない。他環境TSVの内容も読まない。主証拠HEAD/active branchと既存tracked status、固定Git blobの存在は保持を確認したが、全ref/全untrackedの観測前後完全一致は主張しない。

## Mandatory STOP・次の担当

**LOCAL Phase 1は、専用branchへのcommit/pushとremote SHA・公開CSV/REPORT本文照合後にmandatory STOPする。未解決を残した状態で終了し、SERVER 1/2、Phase 2/3、coverage不足A〜E分類、詳細科学レビュー、新規科学計算へ自動移行しない。**

次の担当はユーザー。必要なら対象環境/Phaseを指定した別の明示指示を出す。本LOCAL CodexはSTOP後に追加調査・他環境棚卸し・統合・科学実行を開始しない。
