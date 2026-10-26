# ローカル75件：一括保全・整理準備報告

PUBLIC_REDACTED_BATCH_75。調査日2026-10-26 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。公開入力・公開開始HEAD：`2caa236fa7332872b5a346f0744318d48d59f6ad`。2026-10-25版共通運用指示の実在ファイルを取得し、§8.4・§10を適用した。

## 確認結果

| 群 | 件数 | 今回の判断 |
| --- | ---: | --- |
| 図表・スライド | 7 | ARCHIVED_VERIFIED_PENDING_DEPENDENCY |
| QC | 18 | VERSIONED_ONLY_PENDING_OWNER |
| N1/N3 | 16 | VERSIONED_ONLY_PENDING_OWNER |
| Research B B2D Outer | 34 | VERSIONED_ONLY_PENDING_OWNER |

75件すべてHEAD／branchは前回と一致、tracked clean、untrackedなし、ignored metadata差分なし。74 HEADは現在GitHubのtipと一致、0207の1 HEADは公開tipの祖先。新しい未push HEADは検出しなかった。local／cached remote refと実GitHub snapshotを区別し、fetchせず確認した。

全75件の他環境path依存は未確認。追加の実ファイル保全が必要な具体的資産は今回の既知scopeに残らなかったが、外部raw／独立backupの不存在や復元性は保証しない。削除可能確定0、削除・登録解除・原本移動・branch整理実行0。

## 7件の実ファイル保全

安全性を確認した既存の非公開資料領域に、新規batch namespaceとして**2,052ファイル／234,267,578 bytes**を保存した。全2,052件のsourceコピー前後・destination SHA-256とGit blob一致を確認し、失敗0・上書き0。画像1,357／作成script26を含む。版ごとの件数であり、異なるGit blobは517。数字はファイル保全の集計であり、保存数値の科学再解析はしていない。

| ID | 保存file instances | bytes | SHA-256・Git blob |
| --- | ---: | ---: | --- |
| LOCAL-WT-0203 | 230 | 32,398,399 | 全件一致 |
| LOCAL-WT-0204 | 509 | 52,835,051 | 全件一致 |
| LOCAL-WT-0205 | 49 | 10,182,944 | 全件一致 |
| LOCAL-WT-0206 | 109 | 15,430,487 | 全件一致 |
| LOCAL-WT-0207 | 133 | 17,526,167 | 全件一致 |
| LOCAL-WT-0208 | 509 | 52,871,318 | 全件一致 |
| LOCAL-WT-0209 | 513 | 53,023,212 | 全件一致 |

旧資料の構成と命名を再利用して図表、スライド構成案、Q&A、参照メモ、必要な保存入力・補助資料・作成script・設定・SOURCE/STOP来歴を保持した。参照用の文脈資料5種を各版へ追加コピーした。元worktreeに依存せず画像・原稿を開ける。生成runtime、旧一時pathに依存するverifier、元SOURCE proseの相対リンクの可搬性まで保証しない。詳細は[整理計画](BATCH_75_CLEANUP_PLAN.md)。

保管前に本人所有、owner-only ACL、容量、既存データとの非衝突を確認。独立regular fileの新規作成だけを行い、sourceへのhardlink、移動、編集はない。Git・venv・cache・科学raw全体を一括コピーしていない。既存の科学grant／ledger／marker／STOPは消費・更新していない。

## 68件の整理可能性と残条件

QC18＋N1/N3 16の34件とResearch B34件は、Git管理された成果・中間履歴が保持refから追跡可能である。ただし利用終了は未確定のため、VERSIONED_ONLY_PENDING_OWNERとした。既存control／source／STOP索引443 locator instancesが現在存在することを限定確認し、履歴保持metadataと実ファイルarchiveを区別した。

QC17祖先＋1同一、N1/N3 14祖先＋1同一＋1非祖先、Research B34祖先という代表成果とのGit関係を引き継いだ。代表handoffだけで別SOURCE／独立cohort／診断／失敗／STOPを包含したと認定しない。各保持branch・完全HEADを残す。Research B0283/0286は対象外のUNRESOLVEDを維持する。

現在維持すべきcheckoutは今回の75件すべて。他環境の実パス依存が未確認で、7件のコピー完了も削除許可を意味しない。所有者の利用終了判断が必要なのは残る68件。既存234件の維持7／未解決13と、主Git・現行研究環境は別枠で保護する。A/Bへ接続せず、DF-native B3に干渉していない。

## 原本・公開範囲・検証

元の公開資料17ファイル、非公開棚卸し／234件原本9ファイルのhash不変を確認した。新たな非公開詳細・完全path対応manifestは旧原本と別ファイルとして保存する。75件のHEAD・branch・status・ignored・index、主checkoutのHEAD・status、共有local/cached remote refs・worktree登録は開始時と一致した。保全対象2,052ファイルもsource/destination hashを再照合して全件一致。2026-10-26T20:46:11+09:00時点の確認で、並行差分は検出しなかった。将来の整理直前確認は未実施のままである。

公開は次の新規4資料だけ。保存した図表・原稿・script・入力等のファイル本体は公開しない。公開manifestは既にGitHubで保持されたtracked相対path、branch、commit、hash、匿名保存item IDを用いる。絶対path、非公開ファイル名、private rawの具体的位置、内部host/IP、ユーザー情報・credentialは非公開原本だけに保持または出力対象外とした。

- [BATCH_75_INVENTORY.csv](BATCH_75_INVENTORY.csv)：75件、元分類・直前判断・今回判断・保持ref・所有者／他環境条件。
- [BATCH_75_ARCHIVE_MANIFEST.csv](BATCH_75_ARCHIVE_MANIFEST.csv)：2,052実コピー行＋68件の「新規コピーなし・versioned参照のみ」行。後者にコピー検証済みとの表示はない。
- [BATCH_75_CLEANUP_PLAN.md](BATCH_75_CLEANUP_PLAN.md)：保全scope、研究単位の対応、担当別の次操作と必要な指示。
- [BATCH_75_REPORT.md](BATCH_75_REPORT.md)：本報告。

## 次の担当とSTOP

**所有者**が68件の利用終了・再使用を判断し、**サーバーA/B担当**が全75件の実参照と固定SOURCE／結果の依存を各自確認する。**ローカルCodex**は次の指示で未解決条件を突き合わせ、具体的対象と保持方針を提示する。**ユーザー**が実際の登録解除・directory削除を対象別に指示してから、該当Codexが直前差分を確認する。今回その削除・統合は行わない。

独立した公開用cloneから専用branchへ通常fast-forward commit/pushし、local/remote SHA一致を確認後、**mandatory STOP**。force push・main merge・他環境操作・科学計算は0。
