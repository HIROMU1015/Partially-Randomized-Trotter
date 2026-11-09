# Track D remote branch削除段階の実施記録

**実削除0/38本、HOLD 38本。削除段階は未完了。認証障害ではなく、関係サーバーでの非使用・今後のpush用途が確認できないため停止した。**

確認開始：`2026-11-09T17:05:04.154913+09:00`。記録：`2026-11-09T17:11:30.842620+09:00`（JST）。通常ユーザー権限のローカルGit環境と接続済みGitHub pluginを使用。

Repository：`HIROMU1015/Partially-Randomized-Trotter`。固定正本commit：`9d12d7c6cf3da8f739e0238f7abbbc7018bb6d53`。候補とlease付きコマンドは同commitの既存資料をそのまま使用した。

| 項目 | 実績 |
|---|---:|
| 固定候補 | 38 |
| DELETE_READY | 0 |
| 削除実行／成功／失敗 | 0 / 0 / 0 |
| HOLD／DELETE_BLOCKED | 38 / 0 |
| 開始時remote総branch数 | 408 |
| 削除可否確認後・管理記録公開前 | 409 |
| 管理記録branch作成後の実取得総数 | 410 |
| 開始時の既存remote refで変更・削除された数 | 0 |

開始後、対象外の並行作業で `codex/pr-reference-v6p-results-20261109` が1本追加された。本作業が追加した管理記録branchは1本であり、408 + 他作業1 + 管理記録1 = 410本。過去の408本という総数を固定条件にしていない。

## 安全条件の確認

| 条件 | 確認結果 |
|---|---|
| A：remote HEAD | 38/38本でexpected_shaと一致 |
| B：履歴到達 | 38/38本の元tipと固定preserving_tipが現在の保全先の祖先 |
| C：保全先 | 統合先とO2保持先がremoteに存在し、削除対象外 |
| D：worktree・job・今後push | **未確認。SERVER1・SERVER2等をlive確認できず、使用終了も確認できないため全38本HOLD** |
| E：他研究依存 | 対象外362 refの独立40 tipについてsrc/scripts/.githubを静的検索し、候補名の参照は0。実行時依存・今後用途の不存在は認定しない |
| F：default・protected | defaultはmain。GitHub公開APIのprotected branchとrulesetはいずれも0件。候補は全て対象外 |
| G：固定SOURCE・結果・STOP | 元45 tipは保全先から到達可能。元commit上のSOURCE/STOP参照108件をGit blobとして確認 |

現在環境の登録worktreeについて、候補に対応する現存pathは見つからず、古い登録5件と追跡設定5件は変更していない。この観測やprocess不在を、他環境・今後の非使用の証拠にはしない。

使用終了についてユーザーへ確認したが確定できなかったため、条件を緩和せずHOLDとした。削除承認を再要求していない。`git push --force-with-lease ... :refs/heads/...` は1本も実行していない。

## 認証と保護

originは `git@github.com:HIROMU1015/Partially-Randomized-Trotter.git`。既存SSH鍵で `HIROMU1015` の認証成功応答を確認した。`git ls-remote` と既存統合commitへの変更なしのpush dry-runは成功した。接続済みGitHubアカウントも同名で、repositoryのadmin/push/pull権限を確認した。

SSH host key検証を無効化せず、新しい認証情報・テストbranch/tagを作成していない。共有credential helper、SSH/Git設定、known_hosts、既存branchは変更していない。削除APIは送信していないため、個別削除のserver側成功は主張しない。

## 保全と公開

- Track Dの保持先：`codex/track-d-consolidated-20261109`、remote SHA `9d12d7c6cf3da8f739e0238f7abbbc7018bb6d53`。元43 tipと固定公開commitの到達を確認。
- O2の保持先：`codex/o2-vacuum-h4-preparation-20261108`、remote SHA `781e155d0907a82c962e41ab8e89f985735bc49b`。元CSWAPを含む2 tipの到達を確認。
- q64の5本とq32保留branch、mainを含む開始時408本のremote refは全て同じSHAで残存。
- 既存の統合、科学コード、SOURCE/result/STOP、local branch/worktree、非公開raw/QPY/NPZは変更していない。他サーバーの非公開原本のbytesは直接再照合していない。
- 管理記録は専用branch `codex/track-d-delete-execution-20261109` にのみ追加する。既存統合branch・元研究branchは更新しない。
- 公開追加物はこの報告、[候補別結果](DELETE_EXECUTION_RESULTS.json)、[保全再確認](PRESERVATION_RECHECK.md)の3文書だけ。SHA/tree/公開bytesは公開後に照合する。

## HOLD一覧

次の38本はすべて同じ未解決条件D（関係環境のjob・依存worktree・今後pushの非使用未確認）によりHOLD。期待SHA、保全先、各検査結果は候補別JSONに記録した。

| Branch | 判定 |
|---|---|
| `codex/o2-vacuum-cswap-20261108` | HOLD |
| `codex/track-d-b1-c1-validation-20261106` | HOLD |
| `codex/track-d-b6b-bc-integration-20261108` | HOLD |
| `codex/track-d-bc-static-preparation-20261106` | HOLD |
| `codex/track-d-bc-total-20261108` | HOLD |
| `codex/track-d-c2-q2-compile-20261106` | HOLD |
| `codex/track-d-c3-q4-q8-validation-20261106` | HOLD |
| `codex/track-d-c4-price-model-preparation-20261106` | HOLD |
| `codex/track-d-c5-whole-path-q2-20261106` | HOLD |
| `codex/track-d-c6a-paired-price-pilot-20261106` | HOLD |
| `codex/track-d-c6b-accelerated-completion-20261107` | HOLD |
| `codex/track-d-c6b-q8-exploratory-20261107` | HOLD |
| `codex/track-d-c6b-state-action-acceleration-20261107` | HOLD |
| `codex/track-d-c6b0-paired-model-preparation-20261107` | HOLD |
| `codex/track-d-c7-0-structural-preparation-20261107` | HOLD |
| `codex/track-d-c7-1-q16-exploratory-20261107` | HOLD |
| `codex/track-d-c7-2-cx-structure-20261107` | HOLD |
| `codex/track-d-c7-3-q32-accelerated-20261107` | HOLD |
| `codex/track-d-c7-4-model-stability-20261107` | HOLD |
| `codex/track-d-cut-artificial-20261108` | HOLD |
| `codex/track-d-cut-h4-q4-20261108` | HOLD |
| `codex/track-d-cut-h4-q4-completion-20261108` | HOLD |
| `codex/track-d-h4-paired32-reference-20261106` | HOLD |
| `codex/track-d-h4-reference-validation-20261106` | HOLD |
| `codex/track-d-minimal-reference-validation-20261106` | HOLD |
| `codex/track-d-o5-c1-h4-q4-20261108` | HOLD |
| `codex/track-d-o5-c1-head-tail-artificial-20261108` | HOLD |
| `codex/track-d-o5-c1-head-tail-static-20261108` | HOLD |
| `codex/track-d-o5-design-20261108` | HOLD |
| `codex/track-d-o5-q4-cohort-metadata-20261108` | HOLD |
| `codex/track-d-pr-rpe-static-survey-20261106` | HOLD |
| `codex/track-d-real-pr-transfer-static-20261106` | HOLD |
| `codex/track-d-reference-control-feasibility-20261106` | HOLD |
| `codex/track-d-vacuum-path-static-check-20261106` | HOLD |
| `track-d-b3-rpe-statistics-preparation-20261106` | HOLD |
| `track-d-b4-1-rpe-statistics-implementation-20261106` | HOLD |
| `track-d-b4-2-rpe-shots-science-20261106` | HOLD |
| `track-d-b5-actual-rpe-static-survey-20261106` | HOLD |

## 次の担当とmandatory STOP

ユーザーまたは関係サーバー・worktreeの担当者が、固定38候補の使用終了、今後のpush不要、実行時依存を確認する。確認できた候補について、Codexが最新SHAと他の必須条件を再確認してから既存lease付きコマンドを使用する。現時点では削除を続行しない。

今後Track Dの研究を開始するbranchは `codex/track-d-consolidated-20261109`。新規科学計算・詳細レビューの開始許可はこの管理作業に含まれない。公開照合後に **mandatory STOP**。
