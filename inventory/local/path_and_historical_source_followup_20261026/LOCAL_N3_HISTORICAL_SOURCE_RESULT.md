# N1/N3旧SOURCE 3版：限定探索結果

PUBLIC_REDACTED_SOURCE_FOLLOWUP。入力commit `d908f578a41c8ee235ef5bcbb75af6435f473278`。2026-10-25正式運用指示の実在ファイルを取得し、§8.1・§8.4・§10を適用した。新規科学計算、SOURCEの再生成・修正・再実行、既存archive上書きは行っていない。

## 結果とhash条件

| 宣言されたSOURCE相対path | 期待SHA-256（ファイル内容） | 結果 |
| --- | --- | --- |
| `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py` | `6799702fddd0c54784082d0f5648ff1ad6b8b6faa084dee03d41487850db9b28` | 今回scopeで未発見／一致0 |
| `scripts/run_n3_occupation_transfer_v1.py` | `d43628daa116c12517e95786820885c1cd144988a10f0bca72e2be848e127795` | 今回scopeで未発見／一致0 |
| `tests/test_n3_occupation_transfer_v1.py` | `6533f3fb82e79f8409a87ef2bbf01e9617e310150603890762f536a60da7b2bc` | 今回scopeで未発見／一致0 |

**3版とも今回の限定範囲で未発見。期待hash一致0、旧SOURCE実コピー0。** 再構築して期待値へ合わせる処理は0。「どこにも存在しない」とは判断しない。表のhashは公開済みSOURCE内容の宣言値であり、内部絶対pathのhashではない。

残る55宣言対応は0件解消／55件未解決。影響11 ID：LOCAL-WT-0005、LOCAL-WT-0009、LOCAL-WT-0010、LOCAL-WT-0011、LOCAL-WT-0012、LOCAL-WT-0013、LOCAL-WT-0014、LOCAL-WT-0015、LOCAL-WT-0016、LOCAL-WT-0018、LOCAL-WT-0019。既に保全済みの2,411対応と、うち10対応に使った旧固定SOURCE2ファイルは再調査していない。既存の記録・ledger・失敗・STOP、元のNOT_AUTHORIZEDや実行条件は保持した。

## 再利用した索引と追加した限定範囲

- 既存SOURCE/input宣言・期待hash・55未解決対応を再利用。保管済みファイルのhash索引42,753 entriesへ3期待hashを照合し、filenameにかかわらず一致0。これは保存数値の科学解析やraw再hashではない。
- 既知の関連exportにある対象SOURCE実ファイル21件をhash照合して一致0。関連保管領域の既知top-level archive／bundle候補は0、既存N1/N3保全索引のarchive unitも0。保存された抽出catalogと95 inventory pagesは期待hashの文字列だけを確認し、一致しなかった。数値fragmentのobject群を再解析・全件hashしていない。
- 既存共通Gitのobject metadata88,443件とtree23,493件を読み、3対象filenameを持つblob版3件をhash照合して一致0。21本のcommit以外のtree参照も対象pathを限定確認した。共有Gitへの書込み・fetch・prune・gcは0。
- tree内path名と結び付かない140 blobについて、実行せず静的Python構文として識別した21候補だけをhash照合し、一致0。その他の科学JSON/raw blobを全件hashしていない。別名で保存された未知のsource等の不存在は保証しない。
- 既知の関連Git store3件はobject metadataだけを比較し、検査済み共通store以外のobjectは0だった。他の無関係なrepoや全filesystemへ探索を広げていない。
- 影響11 checkout内の宣言された3相対path、計33現存ファイルも期待hash一致0。現行版・旧成功版で旧開発版を代用していない。全探索の詳細と実path対応は非公開原本へ保存した。

## SOURCE欠落と物理checkout保持の区別

旧宣言は `development_before_log_guard`／`development_before_unresolved_schema_guard` のsynthetic validationが記録したhashである。見つからなかった旧bytesを現在のSOURCEや科学実行許可として認定しない。既存記録の当時の役割と、現在の実ファイルの回収可能性を区別する。

現存する33個の同名pathには期待旧bytesがなく、過去のvalidation・ledger・失敗／STOP記録は独立コピーに保持されている。したがって、**旧SOURCEの所在未解決だけから、11件の物理checkoutを保持する義務は導けない**。欠けた旧bytesの回収、過去記録の保全、実際にdirectoryを必要とする依存／契約は別々の課題である。

これは11 checkoutの削除可能認定でも、固定path契約の解除でもない。75件全体のローカル索引scopeで必須directory束縛は未検出だが、A/B・他Codex／job・private custodyは未確認。旧SOURCEの未解決状態は独立に追跡し、他環境の既存保管資料で回収できるかは各担当へ残す。

**mandatory STOP。既存SOURCE・raw・grant・ledger・marker・STOP、worktree／branchを変更・削除しない。**
