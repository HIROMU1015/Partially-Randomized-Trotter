# N1/N3旧SOURCE3版：サーバーA限定探索結果

2026-10-26。固定入力 `d908f578a41c8ee235ef5bcbb75af6435f473278` と今回のユーザー指定に基づく。影響する11件はLOCAL-WT-0005、0009〜0016、0018〜0019。ローカル保全scopeの旧SOURCE未解決を、サーバーAの独立した保管物で閉じられるか確認した。

| 公開済み相対path | 宣言された期待SHA-256 | Aの限定探索結果 | bytesの完全一致 | 保全コピー |
| --- | --- | --- | --- | --- |
| `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py` | `6799702fddd0c54784082d0f5648ff1ad6b8b6faa084dee03d41487850db9b28` | 未発見 | 候補bytes未発見のため未実施 | 0 |
| `scripts/run_n3_occupation_transfer_v1.py` | `d43628daa116c12517e95786820885c1cd144988a10f0bca72e2be848e127795` | 未発見 | 候補bytes未発見のため未実施 | 0 |
| `tests/test_n3_occupation_transfer_v1.py` | `6533f3fb82e79f8409a87ef2bbf01e9617e310150603890762f536a60da7b2bc` | 未発見 | 候補bytes未発見のため未実施 | 0 |

**発見0/3版、期待hashとの完全一致確認0/3版、非破壊SOURCE保全コピー0件。** 未発見とhash不一致は同義ではなく、今回候補bytesが得られなかったため実ファイルのSHA-256照合対象自体がなかった。過去の開発版を再構築していない。11件の旧SOURCE未解決は維持する。

## 限定した探索範囲

- Aで既に作成された50,159件の元bytes/hash索引を再利用し、期待3 SHA-256と対象basenameを照合した。記録hash一致0。これは既存索引の照合で、50,159ファイルを現在再読取り・再hashした調査ではない。
- 既存SOURCE/archive/custody索引のうち関連するメタデータ9件を限定確認した。対象3版の保存場所を示す記録は検出しなかった。
- 前回登録済み36 worktreeの対象3相対path、計108組を直接確認した。対象名の現存ファイルは検出されず、対象外SOURCEやrawを再帰走査していない。
- 記録済み保管物5件を確認した。Git bundle1件はheader/headとAに既存のcommitから到達する独自履歴だけを調べ、共有Gitへimportしなかった。tar3件はmember header計2,317件、zip1件はdirectory entry4件を対象basenameで照合した。対象member0。非対象memberのraw本文を抽出せず、tar/zipを研究領域へ展開していない。
- Aにある既知の関連Git common directory2件で、local branchから到達し取得済みorigin参照からは到達しない履歴に対象3pathを限定した。対象変更commit0。既知bundle headのA既存objectも同じ限定条件で0。ローカル担当が探索済みの公開履歴全体を再検索していない。キャッシュされたoriginとの差分であり、全世界での未公開性を証明する分類ではない。
- 本人所有の既知archive領域は上位の保管分類だけを確認した。関連を示す根拠のない旧DF保管領域を丸ごと読まず、他ユーザー領域・home全体・未登録disk・別hostを網羅したとは扱わない。

限定探索の読取りエラー0。既存archive containerはsize・mtime・inodeが探索前後で一致した。詳細な所在・member照合・Git参照の根拠・元メタデータhashは非公開原本で保持する。ファイル本文やprivate archiveの配置は公開しない。

## 由来と次に必要な確認

期待hashの宣言元は、ローカル保全報告が示す `development_before_log_guard` / `development_before_unresolved_schema_guard` の既存synthetic source validationである。この過去の宣言・失敗履歴は保持し、現在SOURCEやSTOP・実行許可の契約を変更しない。ファイル版の回収は科学的検証や再実行の成功を意味しない。

ローカル担当には、各期待hashに対する元宣言のstudy・日時・SOURCE版、開発途中の非公開snapshot/archive/exportの既知索引があれば、その最小の非機密な所在確認結果を返すことを依頼事項として残す。新たな候補が見つかった場合だけ、保存済みbytesを読みSHA-256で完全一致を確認する。本文転送は安全な非公開方法・対象の確認後に行い、Public GitHubへ掲載しない。

現在の旧SOURCE未解決を保全済み・不存在・削除可能へ格上げしない。共有Git・原本・SOURCE・raw・STOPを変更せず、scientific runner実行・編集再構築・他host接続を行っていない。mandatory STOP。
