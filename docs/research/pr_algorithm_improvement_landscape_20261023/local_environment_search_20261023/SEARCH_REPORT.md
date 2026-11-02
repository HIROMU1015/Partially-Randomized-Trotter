# 旧研究資料の現環境探索報告

2026年10月23日 JST。GAP-001とGAP-008の既存資料を、今回の環境 `ENV-LOCAL-01` に限定して探索した。内部絶対パス、host/user識別情報、private cacheの所在・fingerprintは公開版から除外している。

| GAP | 判定 | 確認できた資料 | 未解消の範囲 |
|---|---|---|---|
| GAP-001 | `NOT_FOUND_IN_SEARCH_SCOPE` | 歴史的SOURCEとprose資料 | 指定UWC原JSON、theta sweep、診断raw、元入力とのjoin、run log/manifest |
| GAP-008 | `FOUND_UNVERIFIED` | 旧無効化Cgs 37ファイルと旧screening結果のhistorical identityは`FOUND_MATCHED` | 修正後canonical入力、原anchor raw、実科学runの入力/SOURCE/runtime来歴 |

GAP全体は両方とも未解消。`FOUND_MATCHED`は既存期待identityとのbytes一致を確認した範囲にだけ適用する。旧無効化資料の一致を、科学的妥当性や修正後入力の発見とみなさない。

## 読む資料

- [GAP-001 探索報告](GAP-001_recovery_report.md)と[来歴JSON](GAP-001_provenance.json)
- [GAP-008 探索報告](GAP-008_recovery_report.md)と[来歴JSON](GAP-008_provenance.json)
- [探索範囲と除外条件](SEARCH_SCOPE.json)
- [文書と公開用情報の確認記録](VERIFICATION.json)

## 探索範囲

| 公開用の領域ID | 確認した範囲 |
|---|---|
| `PR_REPOSITORY` | 主repositoryの既存ローカルGitと作業領域 |
| `PR_WORKTREE_STORAGE` | 既知PR worktree保存領域のファイル名、既知相対保存先 |
| `RELATED_PF_REPOSITORY` | 関連する高次PF研究repository内のファイル名 |
| `RELATED_PF_HOLDOUT_WORKTREE` | 関連高次PF holdout領域内のファイル名 |

探索時の登録worktreeは255件。worktree保存領域直下の315ディレクトリも調査した。両件数は重複するので合算しない。主repositoryの作業領域にはGit管理領域だけが存在したため、固定参照資料と削除前原本は既存Git objectから読み取った。

`rg --files -uu --hidden`で1,401,518ファイル名を確認し、`uwc|bliss|df_cgs|df_screening|cgs_cost_table`に一致する4,206候補を抽出した。重複SOURCEや旧結果コピーを含む件数である。内容確認は候補、対応SOURCE、既存manifestと対象pathのGit履歴に限った。SOURCEはimport・実行していない。

隠しファイル・gitignore対象もファイル名検索に含めた。`.git`のfilesystem走査、`.venv`、`venv`、`node_modules`、`__pycache__`、`.pytest_cache`と私的な報告出力領域は除外した。Gitは特定commitと対象pathの履歴を別途確認した。

## 確認できたidentity

削除commit `98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604` の親 `57e49cf27d19bfb35cb71b7e9d90a5a30e371e5b` に、集約Cgs表・index・35splitの計37ファイルが残っていた。全件のGit tree entryとblob ID、bytesから再計算したGit blob IDが一致し、追加commit `c580aec790c46dd9aaa2f2982bc9270210523a0d` のblobとも同一だった。

集約35行とindexが指す35splitの保存entryは全件同一だった。これは既存表の対応確認であり、再fit、screening、ランキング、科学的再解析ではない。各ファイルのbytes/SHA256/Git blob IDと、関連SOURCE snapshotのidentityはGAP別来歴JSONに記録した。

旧screening JSONの現存248コピーを個別にhashし、全件が固定監査commit `403204edf22e5712a71ac9ac7eebdbcd18626421` の旧結果と一致した。読み取り前後のsize/mtime/inodeも各ファイルで不変だった。元入力が無効化された旧`DO NOT USE / stale`は保持する。

## 未調査の範囲

別host、SSH接続先、他ユーザー領域、未知の保存先、symlink先ディレクトリ、Gitのunreachable/dangling object・reflog、無関係な研究領域、アプリcache内部には探索を広げていない。自分のホームと研究保存領域の直下一覧を確認したが、独立した研究backup保存先の手掛かりは得られなかった。旧screeningが指す旧研究rootは現環境に存在しなかった。

対象root内の圧縮物4,235件はファイル名だけを確認した。UWC/Cgs専用と判断できるbackup名は見つからず、member検索・展開は行っていない。圧縮物内部や別環境の不存在は主張しない。探索rootと追加probeに権限エラーはなかった。未調査領域を`ACCESS_UNAVAILABLE`とみなしていない。

## 引継ぎと停止

次の環境へ引き継ぐGAP-IDは **GAP-001、GAP-008**。GAP-008は旧無効化37ファイルの所在が確定したので、重点は修正後canonical入力、元anchor raw、実runと入力の来歴に置く。

- ユーザー：必要なら別環境を選び、その環境のCodexへ上記GAP-IDの探索を別途指示する。
- Codex：今回の探索はmandatory STOP。科学実行・再解析・fit・再ランキング・旧実験再実行・原本の回収/転送を自動開始しない。

本追加は探索報告と公開用メタデータのみ。既存のSOURCE、科学結果、grant、ledger、marker、STOPおよび研究statusは変更しない。
