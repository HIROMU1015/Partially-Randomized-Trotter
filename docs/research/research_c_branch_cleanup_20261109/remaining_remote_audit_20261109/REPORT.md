# 残存remote branch棚卸し・候補分類（2026-11-09）

**開始時317本、最新棚卸し297本。KEEP 131本、HOLD 166本、追加統合候補0本、削除候補0本。今回、新たに削除可能と確定できたbranchは0本。**

これはユーザー指定のGit技術監査であり、詳細科学レビューではない。科学計算・保存数値の再解析・回路build/compile・branch統合・削除・force push・prune・gcは実施していない。新しい管理branchも作成しない。既存管理branch `codex/research-c-branch-cleanup-20261109` に本追補だけを公開する。

## 全件表と判定の意味

- [最新297本の分類表](branches.csv)：full remote SHA、分類・理由、保全refとSHA、祖先到達性、未統合commit数、worktree状態、過去台帳・依存関係。
- [branchごとの根拠JSON](branch_evidence.json)：公開済み文書・SOURCE/STOP等のpath例、branch名参照の固定snapshotと行、履歴包含先一覧。
- [過去削除209件の照合](completed_deletions.csv)、[開始後のremote変化](remote_changes.csv)、[技術検査と制約](checks.json)。

| 最終分類 | 本数 | 意味 |
|---|---:|---|
| KEEP | 131 | main、明示保持・保全先、使用終了が確認されていない実在checkoutの枝 |
| HOLD | 166 | 過去HOLD、環境・用途・依存不明、未公開差分や独立履歴の保全未確定 |
| 追加統合候補 | 0 | 統合対象として確定できた枝なし。後述の系列案は調査提案 |
| 削除候補 | 0 | 削除条件をすべて満たした枝なし |
| 合計 | 297 | GitHub実remoteの最新観測 |

名称・日付だけで決めていない。KEEPも「今後永久に必須」という科学的判断ではなく、現在確認できる保持根拠を示す。HOLDでは不明点を残し、保全済みという理由だけで削除候補へ変更しない。

## 316→317→296→297と過去台帳の照合

前回の研究C削除後snapshotからTrack Dの38本を差し引いた316本に対し、開始時には `codex/pr-reference-v6q-prep-20261109` が追加されていた。整理記録の追記によるSHA更新もある。開始時の実測317本を採用した。

監査中、別作業がTrack C/O3の21本を削除し、統合保全先を `a8b4c6f02a3cd7dc58c300af11c8b20f9011d11a` から `b3c81c2677988c7b9491eeee31fbcc0f9b29b078` へ更新した。この21本の不在と全元tipの祖先到達性を確認し、その時点の296本へ反映した。**この21本は本監査の削除実績ではない。** [当該完了記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_branch_cleanup_20261109/DELETION_RESULT.md)を参照。

| 既存整理 | 記録上の削除 | 今回の扱い |
|---|---:|---|
| 研究C、2026-11-09 | 56 | 全56本不在・保持refから祖先到達。再候補化なし |
| Track D/O2、2026-11-09 | 38 | 全38本不在・統合先/O2独立保持先へ祖先到達。再候補化なし |
| Track C/O3、監査中の別作業 | 21 | 全21本不在・統合先へ祖先到達。再候補化なし |
| 全体整理、2026-11-02 | 94 | 93本不在、1本再出現。記録された元SHAは全94件保全 |

その後の公開照合で `codex/df-full-h-pure-rte-light-20261109` が別作業により `9b83b96b83f62142c9d87858dc24a94db7ac5d61` で公開され、297本となった。実在checkoutと観測済みjobに結び付くためKEEPへ追加した。今回の監査がこの研究枝を作成・実行・公開したものではない。

分類表は本追補の公開直前snapshotである。整理branchの表内SHAは `0ea56f39b09503889d741d179528f2b5e8d67f38` であり、本追補の文書公開により当該整理branchだけが直系子commitへ進む。公開後の先端SHAは最終報告と公開commitのURLで示す。

旧94件のうち `track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019` が現在 `77126767c00585d0967558c0756907df2d69d1ff` で存在する。**HOLDとし、過去の削除許可・候補指定を現在の再削除へ流用しない。**

旧台帳のKEEP・HOLD・cross-environment保持理由も取り込んだ。過去に不在だった名前を復元しない。旧台帳の削除planned actionだけで再対象化しない。

## 保護対象とHOLDの根拠

研究Cは25本を保持し、従来remote HOLD21本と追加HOLD2本をそのままHOLDとして継承する。整理保全先と独立128統合先の2本はKEEP。元CSVの過去判定を書き換えていない。

Track Dはq64関連5本、統合保全先、O2独立contextをKEEP、q32をHOLD。q32の未追跡runと収集script2本の存在を確認した。q64にも未追跡run/診断/未公開closeout資料があり、過去STOP・固定SOURCE・grant等の条件を変更しない。別の管理枝 `codex/track-d-delete-execution-20261109` は固有commit1件が統合先に含まれず、用途も未確認のためHOLDとした。

`codex/h46-long-time-b3-20261026` は、この環境のlocal/remoteが各1 commit分岐している。以前のSERVER2未push差分のHOLDを解消しない。未公開側の内容を新たに科学評価・公開していない。

研究Cの複数worktreeで未追跡回路・演算子・NPZ/run原本を観測した。Git履歴が保全されてもこれらの全保全を証明したことにはならない。未公開SOURCE・raw/QPY、grant・ledger・marker・mandatory STOP、将来のpush先は、不在や利用終了を推測しない。

## 確認した環境と限界

直接照合したホストは `GPUA100-gleap02`。主repositoryの登録200件と、既知worktreesディレクトリ直下の別repository6件、合計206 checkoutのHEAD・branch・通常statusを読み取り確認した。22 checkoutに変更があり、21 checkoutに未追跡statusがあった。最新remote297本のうち150本は実在checkoutに結び付いている。未追跡statusのdirectory集約数をファイル数とは扱わない。

可視の本人UID processから、当初local-onlyだった `codex/df-full-h-pure-rte-light-20261109` を使うcontroller1・worker8のPython processを観測した。H8 cwdを持つtmux process2件も確認したが、tmuxの存在を科学job実行の証明にはしていない。process、実行SOURCE、出力、worktreeを変更・停止していない。その後のremote公開とHEAD更新は別作業の進行として記録し、使用中の同枝をKEEPとした。

Windows LOCAL、以前の `/home/abe/Project/Partially Randomized Trotter` 環境、その他SERVER1/SERVER2への現在の接続情報はなく、使用状況は未確認。ホスト名や枝名だけでSERVER1/2への対応を推測していない。本人processの一部にも読取制限があり、scheduler・他UID・別ホスト全体を確認済みとはしない。現在のbranch protectionフラグ、ruleset一覧、open PRは観測上0件だが、mainの保持は継続する。

ignored/private payloadの再帰走査、rawの全hash、配列decodeは行っていない。完全な未追跡ファイル名一覧・privateなlocal-only commit identityは公開しない。詳細な現地観測はこの監査の非公開作業領域に残す。

## 履歴保全と未統合変更

最新297本のうち255本のtipは他の現remote branchの祖先、10本は現archive tagから到達可能（重複あり）。他の現remote branch/tagに含まれないcommitを持つtipは38本。枝ごとに保全refの現SHAと `merge-base --is-ancestor` を記録した。保全先が自身の枝の場合は、枝を消す前の保全が未完了である。

`unmerged_commits_vs_preserving_ref=0` は、その保全先に履歴上含まれる意味であり、保全先の最新treeが元treeと同一という意味ではない。元の科学SOURCE・結果・STOPは元full SHAと元pathで取り出す。`tip_tree_identical_to_keeper` も別欄に記録した。科学SOURCEを保全commitや最新mainへ付け替えない。

研究C・Track D・Track Cの元115削除tip、および旧全体整理94 tipの祖先到達を今回照合した。旧96 archive tagも既存 `verify_archive_refs` で実remote SHAと照合し、現103 tagを保持した。これは既存Git objectの保全照合であり、外部LFS・未追跡raw・非公開SOURCE・科学的再現の新保証ではない。研究Cの387 commit/1,171 blobやTrack Cのbundle/rawに関する詳細照合は元記録を引用し、同じ重い監査を再実装・再実行していない。

HOLD166本のうち122本は、他の現remote refへの履歴包含が確認でき、既存の明示HOLD・再出現枝等を除く「用途・依存確認後に再判定し得る集合」として `followup` 欄で抽出した。**122本を追加削除可能とは判定していない。** 保全先自身がHOLDの場合、そのrefを保持する方針も必要。

## 追加統合を検討する系列案と順序

以下は固定SHA・commit履歴・公開pathに基づく整理案であり、科学的な採択や統合実行ではない。名称の似ている全枝を一括mergeしない。すでに履歴が含まれる枝では追加mergeより用途確認と索引整理を優先する。

| 順序 | 系列 | 観測と有用な整理案 |
|---:|---|---|
| 1 | Track D管理記録 | `track-d-delete-execution` の固有管理記録1 commitを、既存consolidatedの完了記録へ追補する案。旧HOLDの経緯を保ち、q32/q64/O2には触れない |
| 2 | PR Reference（51本） | 50 tipは他枝の履歴に包含。V6-Qの現contextを保持し、CT/N/精度予算のSOURCE・版を固定SHA索引で整理。全branchの一括mergeは不要 |
| 3 | RPE・measurement・CS（51本） | 49 tipは他枝へ包含。input-kernelとmeasurement-backend-contractの独立tipを維持し、enclosure/decoder/controlled条件を区別した索引を検討 |
| 4 | O2 A1（22本）・SCV（11本）・その他（25本） | A1 provenance-summary、SCV finite-confidence、vacuum-H4の独立文脈を保持。共通実装を再利用できても実行契約・STOPを混同しない |
| 5 | Track A B2（29本） | 6 tipに固有履歴。元入力回収、memory、HP/strong-PF分岐・shotの公開成果を固定SHAで並べ、異cohortを混ぜない |
| 6 | Track A O1 M1（18本）・M2（13本） | 科学条件別に小計画・停止・価格pilot・core-only等の出典をまとめる案。M1の独立2 tip、M2の独立1 tipを保持 |
| 7 | H8/H46（9本）・三環境evidence | H46のlocal分岐と各環境の非公開原本を先に照合。H8完了証拠、LOCAL入力回収、SERVER差分を同一成果と推測しない |
| 8 | Track Bと独立総括 | 旧削除名の再出現とbranch名依存を先に確認。Track C/O3は今回完了した4本を保持し、研究C HOLD23本とともに追加削除から外す |

## 削除・統合へ進む前の未確認事項

1. ユーザー／各環境担当：LOCAL・SERVER1・SERVER2で対象ごとのworktree/HEAD、本人job・scheduler、未push/未追跡・ignored原本、今後のpush先を確認する。現在の206 checkoutの使用終了は推測しない。
2. ユーザー：HOLDを解く枝と継続枝、特に研究C23本・Track D q32/q64・H46分岐・再出現Track B枝の扱いを指定する。今回の監査はこれらの保持条件を解除しない。
3. Codex：利用終了と原本保全が確認された集合だけ、最新full SHA・保全先SHA・祖先到達性・未統合commit・branch名依存・protection/PRを再照合する。既存台帳と関数を再利用し、全研究の再監査や科学再実行へ拡大しない。

branch名参照検索は公開main、研究C整理、Track D統合、Track C統合、V6-Qの5固定snapshotのcode/文書に限定した。93枝でliteral名参照を観測したが、歴史的文言と実行依存は区別が必要。外部の全依存を確認済みとはしない。

## 次の実行に必要な承認範囲と停止点

- **ユーザー**：まず環境情報・継続用途を提供する。読取照合は今回の依頼範囲であり、同じ読取作業への重複承認は不要。
- **実際の統合を依頼する場合のユーザー承認**：対象full SHA、既存保全先branch、収録する公開文書/履歴、科学SOURCE・raw・STOPを変えない範囲を指定する。private履歴を無条件で公開しない。
- **実際のremote削除を依頼する場合のユーザー承認**：確認を終えたbranch名と現full SHAの明示allowlist、保持ref、remote refだけを対象とする範囲を指定する。承認証明書やgrant発行の管理機構は新設しない。既存固定実行契約の条件は保持する。
- **Codex**：承認された具体的範囲の次作業だけを行う。今回の監査・文書公開の指示は、統合・削除・科学実行・詳細レビューの許可ではない。

公開先は研究C整理branchの直系子commitとし、既存科学treeと過去台帳を上書きせず、本directoryと親REPORTへの案内だけを追加する。公開後、実branch SHA・commit親/tree差分・公開byteと最新refを照合する。

**MANDATORY STOP：棚卸し・候補分類・文書公開と照合まで。追加統合・削除は開始しない。**


## 追補：1週間以上前の枝の実整理（2026-11-09 18:34 JST）

[年齢条件に基づく削除結果](aged_cleanup_20261109/REPORT.md)：古いtip10本を確認し1本削除・9本HOLD。結果のmain同一blob収録と既存archive履歴保全を確認し、全remoteは297→296本。上の分類表は削除前snapshotとして保持する。科学実行・元SOURCE/原本変更0、公開後照合でmandatory STOP。


## 追補：3日以上前の枝の追加整理（2026-11-09）

[3日条件の削除結果と残存一覧](three_day_cleanup_20261109/REPORT.md)：対象137本のうち73本追加削除・64本保持。結果/報告の同一blobと7保全先への元tip祖先到達を確認し、全remoteは296→223本。残存はKEEP131・HOLD92。上記の旧分類表・STOPは過去snapshotとして保持し、科学実行・元SOURCE/原本変更0、公開後照合でmandatory STOP。


公開後の別作業でV6-Q2枝が1本新設されたため、[最新224本の一覧と照合追補](three_day_cleanup_20261109/REPORT.md)へ反映した。今回の実削除73本は変わらず、最新分類はKEEP132・HOLD92。
