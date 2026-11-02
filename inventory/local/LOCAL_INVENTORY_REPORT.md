# ローカルPR研究棚卸し — 公開用匿名化コピー

2026-10-26 JST。environment=LOCAL。**PUBLIC_REDACTED_COPY。詳細な棚卸し原本とは異なる。整理の実行許可ではない。**

対象：`HIROMU1015/Partially-Randomized-Trotter`。既存4成果物から公開用コピーを作成した。原本は変更・削除せず非公開で保持する。今回の公開作業は新しい棚卸し・科学計算・数値再解析・詳細科学レビューを含まない。下記件数・保持判断は元の観測snapshotを引き継ぎ、本公開作業の作業領域やその後の並行作業は加算していない。

## 集計と単位

| 項目 | 件数 |
|---|---:|
| 実在・ユーザー所有を確認した非bare worktree | 389 |
| 主common directoryに登録されたworktree | 330 |
| その他の独立cloneのcheckout | 59 |
| Git common directory（独立リポジトリ） | 85 |
| 対象repoと対応を確認できたcommon directory | 80 |
| 名前・用途候補のみのbare（対応未確定） | 5 |
| 非bare clone本体 / bare保管領域 | 60 / 25 |
| KEEP_ACTIVE | 2 |
| KEEP_EVIDENCE | 302 |
| 保護対象（上記2分類の和） | 304 |
| REMOVE_CANDIDATE_AFTER_APPROVAL | 0 |
| HOLD_UNKNOWN | 85 |
| dirty worktree（tracked変更あり / untrackedあり） | 31（9 / 31） |
| detached HEAD | 22 |
| local branch ref / 固有branch名 | 490 / 420 |
| remote-tracking ref（独立clone間の重複を含む） | 2795 |
| その他ref | 51 |
| GitHubの現行branch / 固有tip SHA | 776 / 775 |

common directory数とworktree数は異なる単位である。bareはworktree数へ含めない。clone間の同名branchを別refとして保持し、CSVにcommon directoryを付けた。これらを独立科学研究数や実験数とは解釈しない。HOLDも今回の操作対象から除外する。

**削除候補0件、削除可能確定0件。** HEADの公開・cleanは確認できても、現在の利用終了、固定SOURCEの絶対path、private成果・rawの別保管、他環境の依存解消を確認できないためである。整理余地がないという意味ではない。後述の集約検討群はHOLDであり、削除対象リストではない。

## 観測範囲・判断の限界

元snapshot：metadata観測2026-10-26 15:59–16:00 JST、GitHub heads観測同日15:59 JST。登録worktree、local branch、cached remote ref、GitHub heads、既存研究索引と保存領域metadataを照合した。Linux上で可視のユーザー所有研究領域に限定し、全filesystem・他namespace・他環境を網羅したとは主張しない。

元調査ではGitの任意index更新と暗黙fetchを抑止し、raw全件hash、科学runner実行、SSH、削除、統合を行っていない。tracked SOURCE/controlの空参照は不存在証明ではない。観測は非atomicであり、別作業によるworktree・branch追加も実際に検出してKEEP_ACTIVEへ反映した。processの限定的な可視性を利用終了の証拠にしない。

本公開版の各CSVは元CSVと同じ行数・分類を持ち、環境内IDで対応する。LOCAL_BRANCHES.csvはLOCAL_BRANCH、REMOTE_TRACKING、GITHUB_BRANCH、OTHER_REFをrecord_typeで区別する。GitHub行のenvironment=LOCALは一覧を取得した環境を表す。IDは研究ID、認証情報、アクセス権ではない。環境間照合はbranch・SHA・study・公開SOURCE参照を用い、IDだけで同一保存領域を推定しない。

## 匿名化・非公開とした情報

- ユーザー名を含む絶対パス、内部保存先、clone／共通Git領域の実体名をworktree／repository／dependencyのIDへ置換した。元pathとIDの逆引き対応は公開しない。
- private raw、未追跡・ignored成果物の具体的な配置・ファイル名、内部export／実行logの配置、symlink・venv・外部依存の詳細、object storageの内部metadataを除外した。存在・entry数・保持分類・依存種別・未確認状態は保持した。
- OS/kernelの詳細、マウント配置、内部process・実行環境構成は公開しない。ホスト名・IP・認証情報・token・credential付きURL・外部非公開データの内容は含めない。
- 対象GitHub repositoryとの対応を確認していない別remoteのref名と、内部運用refの詳細は公開しない。該当行の種別・件数・整理分類は残す。
- 科学文書・SOURCE・公開済みcontrolへのリンクは、観測した公開GitHub headsから到達するcommitにあり、versioned pathの存在を確認できたものに限定した。未公開commitのSHA・保護判断は残すが、そのファイル名・内容・ローカル所在は公開しない。
- 本書は原本の構成・集計・保全上の結論を再利用して要約した公開版であり、詳細原本を置き換えない。原本の補助repo一覧・集計JSON・公開作業のprivate対応表はこのbranchに含めない。

## Git・研究証拠の保全上の要点

`all-r-coherent-opt2-reoptimization`の元local HEAD `e098c54c78f589055082f9cfc2b13de50c90ca94` は同名remoteより2commit先だったが、別の公開tipの祖先として確認できた。local mainは同名remoteより1commit後方だった。これを今回変更したという意味ではない。

`track-b-cross-scientific-review-record-20261019` のHEAD `3bd4eb9fadb1be69e39f48903293391acdac931f` は元snapshotの全GitHub headsから到達しない固有commitを1件持つため保護する。過去のpush、tag、PRの到達状況までは判定していない。GitHub branch不在と未公開commitを同義としない。

主Git本体は多数worktreeとB3関連refの共有依存を持つ。linked worktreeと独立cloneが混在するため、directory単位の一括整理は不可。dirty・detached SOURCE・private結果・grant／ledger／marker／STOP・refなしbareのobject packを保持する。共有venvとsymlink参照先も利用終了・依存解消の確認待ち。公開HEADからGit管理外の現在状態を復元できるとは扱わない。

同一HEADの8群を重複科学証拠とは認定しない。同一HEADでも独立cohort、異なるSOURCE、別の実行出力・停止履歴を持ち得る。旧Track A／PR2の未commit変更と過去の棚卸し記録を区別し、過去の削除許可を新たな対象へ転用しない。

サーバーBのB3はユーザー申告の実行中状態を前提に保護する。B3関係branch・SOURCE・結果・共有参照に変更を加えず、別研究のpause記録をB3停止の根拠にしない。GitHub branchは他環境でも利用されるため、local checkoutがないことを削除根拠にしない。

## dirty領域（内部所在を匿名化）

以下はGit status entry数でありdirectory内部の全file数ではない。具体的な未追跡・ignored pathは非公開とした。

| worktree_id | tracked変更entry | untracked group entry | 分類 |
|---|---:|---:|---|
| LOCAL-WT-0001 | 27 | 91 | KEEP_ACTIVE |
| LOCAL-WT-0022 | 21 | 19 | KEEP_EVIDENCE |
| LOCAL-WT-0043 | 7 | 3 | KEEP_EVIDENCE |
| LOCAL-WT-0047 | 8 | 31 | KEEP_EVIDENCE |
| LOCAL-WT-0048 | 8 | 62 | KEEP_EVIDENCE |
| LOCAL-WT-0049 | 8 | 45 | KEEP_EVIDENCE |
| LOCAL-WT-0050 | 9 | 16 | KEEP_EVIDENCE |
| LOCAL-WT-0051 | 8 | 48 | KEEP_EVIDENCE |
| LOCAL-WT-0052 | 8 | 65 | KEEP_EVIDENCE |
| LOCAL-WT-0210 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0211 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0213 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0215 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0216 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0231 | 0 | 3 | KEEP_ACTIVE |
| LOCAL-WT-0293 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0294 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0296 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0298 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0300 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0301 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0303 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0304 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0309 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0313 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0317 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0341 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0345 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0348 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0358 | 0 | 1 | KEEP_EVIDENCE |
| LOCAL-WT-0365 | 0 | 1 | KEEP_EVIDENCE |

## GitHub上の代表成果候補

branchは可変参照、下記commit/pathは固定参照。科学的採択や新たなevidence freezeを行った表ではない。今回数値結果の引用・再集計はしていない。

| 研究単位 | 代表branch／固定commit | 入口・保全先 |
|---|---|---|
| DF・Pauli統合 | `codex/df-pauli-evidence-paper-plan-20261026` / `6e3436a16b13e539b0faa2a0a3286f6ab3549543` | `RESEARCH_SYNTHESIS.md`, `EVIDENCE_MATRIX.csv`, 原稿・図表計画 |
| 長時間・高次DF PF/PR | `codex/h46-long-time-pf-pr-20261025` / `155529e20a8fac39e00786c40b57830b71203c4f` | `REPORT.md` と同snapshotの結果参照 |
| Pauli-JW総合 | `codex/pauli-jw-research-summary-20261026` / `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` | `docs/research/pauli_jw_resource_evaluation_summary.md` |
| Track A研究データ索引 | `731b3d6f3572c6aa7950a579b68251fec9a5218b` | `docs/data_collection/2026-10-24_track_a_research_data/README.md`, study_index, source_catalog |
| CEM・ledger・既存原稿参照 | `09e71910ee49e0d9d3a2f2f410249f6623022c0d` | `artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/claim_evidence_matrix_v3.csv`, `evidence_ledger_v5.csv`, `paper_manifest.json` |
| 研究Bの既存成果・一時保留 | `codex/research-b-final-pause-handoff-20261026` | `docs/research/research_b_final_outcomes_and_pause_handoff_20261026.md`。サーバーB B3の実行状態とは別 |
| C1/C4など研究C | LOCAL_WORKTREES.csvの研究C分類群 | 実装・preparation・science・resultsを統合せず個別SOURCE／STOPとvenv依存を保持 |

Track A索引のstudy_index58行、source_catalog80参照、およびEVIDENCE_MATRIX43行の構造・参照を利用した。研究数の新規カウントや科学結論の再評価は行っていない。既存CEM/ledgerはsource_catalogの固定参照として再利用し、その全行内容を新たにレビューしたとは主張しない。統合synthesisも入口・証拠参照を中心に閲覧しており全文の科学レビューではない。

## 次の担当と停止点

1. サーバーA担当Codex：ローカルのpush完了後、`codex/pr-environment-inventory-20261026`を取得し、同様に公開情報を確認したサーバーAの棚卸しを`inventory/server_a/`へ追加commit・通常pushする。他のdirectory・既存履歴を上書きしない。
2. 統合担当Codex：ローカルとサーバーAの棚卸しを照合し、branch／SHA／study／SOURCE／独立cohort／保存依存を確認する。privateなpath解決は各環境の非公開原本で行う。
3. ユーザー：保全先・利用終了・復元条件が確認できた具体的な整理案について、実際に実施する対象を別途判断する。本公開版は削除やmainへのmergeを許可しない。
4. サーバーB担当：今回は操作対象外。後日、別途指示に従い`inventory/server_b/`へ追加する。B3の稼働環境に干渉しない。

公開ファイルはLOCAL_WORKTREES.csv、LOCAL_BRANCHES.csv、LOCAL_STORAGE_DEPENDENCIES.csv、LOCAL_INVENTORY_REPORT.mdの4点。本branchは公開棚卸し資料の受け渡し用であり、詳細原本・科学SOURCE・rawの保管branchではない。

**専用branchのcommit／通常push／remote SHA照合後にmandatory STOP。worktree削除・統合・prune・branch削除・force push・mainへのmergeは実施しない。**
