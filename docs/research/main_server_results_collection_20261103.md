# サーバー1・2の成果照合とmain収録

2026-11-03。サーバー1の4,002行、サーバー2の7,517行、計11,519行の照合報告を統合した。
両報告の比較基準は `fe2938191f7821b28e565db2880b2d66fbec5f46`。
ローカルでは後続のP1・研究Cの配置を含む `54b5c67e279aa20a5d684a2dd315ded666429f0a` に再照合した。
報告の行にはsnapshot総括・データ群・個別ファイルが混在するため、研究件数や枝数として足し合わせない。
非公開の報告TSV自体は公開しない。

## 今回の収録

| 区分 | ファイル数 | 扱い |
|---|---:|---|
| mainにない追加path | 6,289 | 原内容・modeで追加。既存pathは上書きしない |
| mainと異なる公開版・研究test関連の原版 | 1,042 | `.original` として保存し、元の実行contextへ戻れるようにする |
| 合計 | 7,331 | 447,036,738 bytes |
| 旧総合索引・全体manifest等の別版 | 2,076 | バイト列の重複追加は行わず、固定公開commit参照で保持 |

[全収録ファイルと原snapshotの対応（TSV）](main_server_results_collection_20261103_files.tsv) ／ [旧総合索引等の固定参照（TSV）](main_server_results_collection_20261103_index_references.tsv) ／ [保存原版の入口](../archive/server_collection_20261103/README.md)

両環境のtree/blob/modeを照合し、同じ内容は重複排除した。
追加pathは報告間の同名別版と公開先端の履歴も確認し、公開子孫での削除や異なる版があれば保存原版へ分ける。
同名で内容が違う既存mainファイルは置き換えない。
収録する各ファイルは、対象GitHub repoの既存公開参照から到達する元commitでpath・blobを確認した。
保存先を探すためのviewing snapshotを、各研究の科学SOURCEへ付け替えない。

元の研究test関連ファイルは保存原版に置いた。今回、それらのtest・runnerは実行していない。
過去の研究条件をmain全体の統一仕様へ混ぜたり、同名別版を科学的な優劣で選んだりしていない。
共有ライブラリのAPI整合、数値再現、main全体の回帰検査の完了を意味しない。

## 大きな原データと非公開原本

大きな原データは既存保存先を維持する。binary payload、raw/cache/checkpoint、5 MiBを超えるデータ記録、
privateと指定された領域、ルートの実行claim/marker、特殊entryは今回の追加対象に含めない。
5 MiBはデータ配置の基準であり、科学実行の入力・精度・資源上限を変更するものではない。
外部データ群24行の報告は既存の原本・backupの所在を扱うもので、今回の収録によって
別媒体保存や完全復元・全raw検証が済んだとは認定しない。

原文の相対リンクがmain未収録のrawや元contextを指す場合は、対応表の元snapshotを使う。
同じblobがmainの別pathや保存原版にあることと、現在の実行pathへ採用されていることを区別する。

## 確認範囲

原ファイルのGit blob・mode一致、Python ASTとJSON構文、案内リンク、既存validation manifest checkerを確認する。
Python構文の対象は977件、JSON構文は4075件である。
これは技術的な配置・同一性確認であり、科学的性能、native correctness、原結果の再解析や新しいGOの認定ではない。
既存のstatus、quarantine/missing、科学SOURCE・入力・runtime・契約・grant・ledger・marker・STOPを維持する。
新しい科学計算、数値再解析、回路build/compile、元の研究runner/test実行は0である。

## 残る受渡し

この環境で原blobや固定SOURCE contextを参照できない項目、未コミット原本、非公開保存先の対応は、
ローカル専用の受渡し一覧に残す。GitHubへの原commit照会でも見つからない20件を、公開済みとして扱わない。
B3のローカル版と公開版も、異なるSOURCE・結果として保存先を維持する。
不足分は既存bundleと選択ファイルで受け取り、公開範囲の制約を保ったまま追加照合する。
元repoやworktree、未コミット変更、jobを変更・停止していない。
枝全体のGit merge・ブランチ削除、新しい保存tagの作成は行っていない。

## サーバー1原本からの追補

[追加収録と保存原版の出典](server1_onsite_collection_20261103.md)を参照する。ローカル専用報告・非公開SOURCEや環境記録を除き、選択した過去の資料・SOURCEを保存原版として収録した。旧DRAFT・科学SOURCE・STOPを保持し、枝全体の履歴mergeや現在のAPIへの採用とは区別する。
