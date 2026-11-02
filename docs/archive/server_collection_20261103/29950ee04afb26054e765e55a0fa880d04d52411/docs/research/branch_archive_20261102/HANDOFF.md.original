# 別環境Codexへの引き継ぎ：保存tagと旧branch整理

2026-11-02 JST。利用者は、研究結果の固定SHA・資料・成果物へのマップを保ったうえで、
保存tagへ履歴を退避し過去のリモートbranchを減らす方針を承認した。
この環境のGit push認証が使えなかったため、**tag公開と削除の実行担当を別環境へ変更した**。
接続済みGitHub連携によるこの台帳の公開と、Git transportの認証は別である。

## 実行前の状態

- 研究結果の入口は[統一索引](../RESEARCH_RESULTS_INDEX.md)。本台帳はその補助資料。
- 実リモートの計画snapshotは358本。既に存在しなかった548本も含め、旧名906件を固定SHAへ対応付けた。
- 保存tagは96個。初期計画の削除候補は354本。
- `main`、`codex/research-c-pr-integrated-20261102`、`codex/research-c-pr-resource-contribution-20261102`、
  `codex/h6-pauli-price-tail-audit-20261102`を維持する。計画後の新規branch・SHA更新も削除しない。
- この台帳を作成した環境では、ローカル保存tagと完全bundle＋増分bundleを作成・検証した。
  リモートtag公開・branch削除は一件も実行していない。別環境でもprepareでバックアップを作る。
- 元status、STOP、科学SOURCE、数値結果、未公開rawは変更しない。科学計算は不要。

## 別環境で行う順序

既存の未コミット作業を持つworktreeは保持する。必要ならorigin/mainから独立worktreeを作る。
このmainにある `scripts/retire_archived_remote_branches.py` を使用する。

```bash
python3 scripts/retire_archived_remote_branches.py prepare
python3 scripts/retire_archived_remote_branches.py tags
python3 scripts/retire_archived_remote_branches.py delete
```

各段階が成功したことを確認して次へ進む。状態・バックアップは既定でリポジトリrootの
`.branch-cleanup-20261102/`へ保存される。このディレクトリをstage・公開しない。
必要なら各コマンドに同じ `--state-dir /保存先` を指定する。

`prepare`は固定commitの不足分をGitHubから取得し、ローカル保存tagを上書き禁止で作成する。
全既知履歴の被覆を検査してからGit bundleを作り、SHA256と完全性を記録する。
既存worktreeのファイル・local branchをcheckout/reset/deleteしない。

GitHubから取得できない保存commitがあれば停止する。元環境のbundleが必要な場合は、
manifest.jsonのbackup.parts_in_restore_orderにある3部を同じ順番で用意し、prepareへ
`--bundle /完全bundle --bundle /増分bundle --bundle /参照外snapshotのbundle` を追加する。
これらはローカル専用履歴を含み得るためGitHubへアップロードしない。

`tags`はplanの96個だけを、上書きなし・atomicな20個単位でpushし、全tagのSHAを検証する。
他のローカルtagをまとめて送信する操作はしない。

`delete`は公開済みmainの対応表がローカルplanと完全一致すること、全保存tagとバックアップが
保全されていることを再確認してから進む。各branchの初期SHAを明示した
`--force-with-lease`とatomic pushを使用する。一般的なforce pushは使用しない。

実行直前にopen PRや追加の継続作業が分かった場合、そのhead枝は
`delete --keep ブランチ名`で追加保護できる。worktree記録があることだけを理由に全件を保留しない。
計画からSHAが変わった枝・新規枝・失敗したatomic batchは保持し、結果へ明記する。
退避が確認できない状態で削除を進めない。

## 完了記録

`.branch-cleanup-20261102/execution.json`で、当該実行で削除した件数、既に不在だった件数、
更新検出等で保持した枝、最終リモート枝、保存tag数を確認する。
新規branchや再作成されたbranchがあれば、対象外として残す。
タグ・削除の実績と台帳の公開状態をREADMEへ追記し、execution.jsonの公開用部分だけを
この台帳ディレクトリへ保存してcommit・pushする。バックアップ・全stateは公開しない。
mainの他者更新があれば通常の統合で反映し、mainをforce pushしない。

ローカルの一時bare repoによる検査では、公開driverのSHA不一致時のatomic拒否、
一致時の対象限定削除、tag衝突拒否、main削除拒否を確認した。
これは削除手順の技術検査であり、科学結果の再検証ではない。
