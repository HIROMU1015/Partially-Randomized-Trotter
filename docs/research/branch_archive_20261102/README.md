# 旧リモートブランチの保存台帳

2026-11-02 JST、改訂2（元計画：`7ddf2bdf93dadb8c9868676636e06d6b9ca13143`）。
研究結果の入口は引き続き[統一索引](../RESEARCH_RESULTS_INDEX.md)。
本台帳は旧branch名を固定commit・資料・成果物・履歴の保存tagへ結ぶ。
科学的採否、元status、STOP、科学SOURCEと結果commitの区分は変更していない。

## 旧名から研究結果へ

- [ブランチ対応表](branches.csv)：旧名、完全SHA、当時の資料pathと成果物root、保存tag、復元コマンド。
- [保存tag一覧](archives.csv)：tagはbranch全体の履歴を保存する先端commitを指す。
- [機械可読manifest](manifest.json)：調査境界、バックアップ、保存計画と技術確認。
- [別環境Codexへの実行引き継ぎ](HANDOFF.md)：prepare → tag公開 → SHA照合付き削除の手順。

固定SHAの資料は `https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/<fixed_commit>/<path>`、
ローカルでは `git show <fixed_commit>:<path>` で読める。
同じbranch名の複数の観測SHAはknown_commit_versionsに記録した。
必要なら `git fetch origin tag <archive_tag>` の後、対応表のrestore_commandでローカルbranchを復元する。
保存tagより前のcommitも履歴として保持される。

## 2026-11-02の整理境界

実リモートの開始時点は358本。過去に取得済みのbranch名は906本。
そのうち548本は開始時点で既にリモートに存在せず、今回削除した本数へ加えない。
保存tagは96点。保持する枝は`codex/h6-pauli-price-tail-audit-20261102`, `codex/research-c-pr-integrated-20261102`, `codex/research-c-pr-resource-contribution-20261102`, `main`。
改訂2では文書の名前参照がある`track-b-g6-return-generator-audit-20261010`と、
サーバー2の未公開local履歴の確認が残る`codex/h46-long-time-b3-20261026`も保持する。
候補は元の354本から352本となる。保持理由はmanifestの`revision.keep_reasons`を参照。
新規作成またはSHA更新が検出された枝は削除対象から外す。

状態は**改訂案・削除前計画**。元の保存tagはサーバー1でローカルに用意済み。
この改訂を作成したローカル環境ではprepare・tag公開・branch削除は未実行。
tagと台帳の公開を確認後、SHA一致を条件に削除する。
完了・未実行・更新検出等の実績はexecution.jsonと本書の後続記録に残す。

元計画のバックアップはサーバー1でGit bundleとして完全性を検証した報告に基づく。
SHA256とbytesはmanifestに残しており、この改訂作業で実物を再検証したものではない。
ローカル専用branchや未公開の履歴を含むバックアップ自体は公開しない。
未追跡rawの完全保存、Git LFS等の外部ストレージ、科学的再現性を新たに保証するものではない。
既存ローカルbranch・worktree・未コミット作業を削除する操作は含めない。

改訂driverはlocal/public版のSHA256を検査し、push後の実照会で結果を記録する。
エラー後に不在を確認した枝は成功push・保持と区別し、照会できなければ未確認として停止する。
再開時に未確認batchを自動再pushしない。詳細と模擬検査コマンドは[引き継ぎ](HANDOFF.md)を参照。

## 読み分け

actionはこの整理での操作区分であり、科学結果のPASS/FAIL・採否ではない。
ABSENT_BEFORE_THIS_CLEANUPは既に存在しなかったbranch名の索引である。
navigation_pathsとartifact_rootsは資料を探すための手掛かりで、元報告の条件・SOURCE・raw・資格を必ず併読する。
母平均、全law、最終総cost等への新しい科学的結論は出していない。
