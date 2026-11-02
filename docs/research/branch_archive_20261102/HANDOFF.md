# 別環境Codexへの引き継ぎ：保存tagと旧branch整理

2026-11-02 JST、改訂2。元計画はcommit `7ddf2bdf93dadb8c9868676636e06d6b9ca13143`。
サーバー1で作成・公開した保存計画をローカル環境で点検し、driverの結果記録と版照合を修正した。
元計画の354候補からG6・B3の2本を保留し、候補は352本とした。
**この改訂の作成・模擬検査ではprepare・tag公開・branch削除を実行していない。**
利用者のローカル環境での削除指示を受け、改訂を公開してから以下の手順を使う。
その後のローカル実行は完了した。保存tag96個公開、branch94本削除、元354候補中260本保持。
実績は[README](README.md)と[実行結果](execution.json)を参照。以下は実行した手順の記録である。

## 実行前の状態

- 研究結果の入口は[統一索引](../RESEARCH_RESULTS_INDEX.md)。本台帳はその補助資料。
- 実リモートの計画snapshotは358本。既に存在しなかった548本も含め、旧名906件を固定SHAへ対応付けた。
- 保存tagは96個。初期計画の削除候補は354本。
- `main`、`codex/research-c-pr-integrated-20261102`、`codex/research-c-pr-resource-contribution-20261102`、
  `codex/h6-pauli-price-tail-audit-20261102`を維持する。追加保持は次の2本。
  - `track-b-g6-return-generator-audit-20261010`：既存レビューにbranch名を使ったGitHub API URLがある。
  - `codex/h46-long-time-b3-20261026`：サーバー2の棚卸しでlocal/remoteの分岐が報告された。
    local先端`f794a2aa4c76bc024540c808bf9400d9f523bb76`は元の既知908版に含まれず、こちらにobjectもない。
    差分の内容は未確認。remote枝を保持しても未公開local履歴やrawのバックアップにはならない。
- 計画後の新規branch・SHA更新も削除しない。
- この台帳を作成した環境では、ローカル保存tagと完全bundle＋増分bundleを作成・検証した。
  リモートtag公開・branch削除は一件も実行していない。別環境でもprepareでバックアップを作る。
- 元status、STOP、科学SOURCE、数値結果、未公開rawは変更しない。科学計算は不要。

## 別環境で行う順序

既存の未コミット作業を持つworktreeは保持する。必要ならorigin/mainから独立worktreeを作る。
改訂を公開したmainの `scripts/retire_archived_remote_branches.py` を使用する。
`manifest.json`のdriver SHA256とlocal/public driverの両方を照合する。
originのfetch先・全push先が対象リポジトリであることも検査する。
元計画のstateとbackupは保持し、改訂版は新しいstate directoryでprepareする。
元stateは旧manifestに対応しているため改訂版では再利用できない。

```bash
python3 scripts/retire_archived_remote_branches.py prepare --state-dir /保存先/branch-cleanup-20261102-r2
python3 scripts/retire_archived_remote_branches.py tags --state-dir /保存先/branch-cleanup-20261102-r2
python3 scripts/retire_archived_remote_branches.py delete --state-dir /保存先/branch-cleanup-20261102-r2
```

各段階が成功したことを確認して次へ進む。3コマンドには同じ保存先を指定する。
stateとbackupをstage・公開しない。未追跡・ignored rawはGit bundleに入らず、
サーバー1の棚卸しも未受領なので、三環境全体の保存完了という意味にはならない。

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

実行直前にopen PRや追加の継続作業が分かった場合、そのhead/base枝を
`delete --keep ブランチ名`で追加保護する。本実行では最初に指定された除外条件を維持し、
登録worktreeの枝、未push作業の継続先、現在のコード・文書から依存される名前も保持する。
他のworktreeやjobは停止せず、条件を満たす残りの候補を進める。
計画からSHAが変わった枝・新規枝は保持し、結果へ明記する。
push後は成功・失敗にかかわらず実リモートを照会する。通信エラー後に不在なら
`ABSENT_CONFIRMED_AFTER_PUSH_ERROR`として記録し、削除成功・保持とは区別する。
照会できなければ`UNCONFIRMED_AFTER_PUSH`を残して停止する。同じstateでの再実行時は
実状態を照合し、このbatchを再pushしない。拒否されて残った枝も自動再試行しない。
退避が確認できない状態で削除を進めない。

## 完了記録

指定stateの`execution.json`で、成功push後に不在を確認した件数、既に不在だった件数、
通信中断等の後に不在を確認した件数、保持した枝、最終リモート枝、保存tag数を確認する。
途中停止時は`deletion_progress.json`を確認する。最終照会で再出現した枝は保持へ分類する。
新規branchや再作成されたbranchがあれば、対象外として残す。
タグ・削除の実績と台帳の公開状態をREADMEへ追記し、execution.jsonの公開用部分だけを
この台帳ディレクトリへ保存してcommit・pushする。バックアップ・全stateは公開しない。
mainの他者更新があれば通常の統合で反映し、mainをforce pushしない。

改訂の技術検査は`python3 tests/test_retire_archived_remote_branches.py`で実行する。
Git通信を全てmockし、通信失敗後の実状態記録、照会失敗からの再開、driver版不一致、
更新された枝・main・保存tagの保護、別push先の拒否を検査する。
GitHubや一時bare repoへのpush、科学結果の再検証は行わない。
