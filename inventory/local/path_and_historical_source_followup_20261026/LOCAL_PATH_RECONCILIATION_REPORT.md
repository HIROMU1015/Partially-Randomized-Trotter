# ローカル75件：非公開パス照合準備・判断更新

PUBLIC_REDACTED_PATH_FOLLOWUP。固定入力 `d908f578a41c8ee235ef5bcbb75af6435f473278`。サーバーA固定結果 `f8e6cee8d08804095be8ba296726da91b9c8f21b` を読み、75件すべてPATH_VERIFICATION_NOT_POSSIBLEだったことを引き継いだ。これは依存あり／なしのどちらも確定していない状態である。今回もA/BへのSSHや非公開対応表の転送は行っていない。

公開準備中にAの後続commit `3b95bab223427405caa779c01194671f06de1b54` の3資料も取得・確認した。Aでも旧SOURCE3版は限定scopeで未発見、75件の実パス未受領・物理判定未確認を維持していた。Aの追加を保持して本資料を追加する。両環境の準備完了を、非公開情報の受渡し・物理照合の完了とは扱わない。

| 保全群 | 件数 | 今回の保全判断 | 物理path判断 |
| --- | ---: | --- | --- |
| 図表・スライド | 7 | 既存2,052ファイルの保全scope維持 | ローカルscopeで必須束縛未検出、A/B未確認 |
| QC・N1/N3・Research Bの検証済み群 | 57 | 既知の選定scope検証済みを維持 | 同上 |
| N1/N3の旧SOURCE未解決群 | 11 | 現存資料・過去記録は保全、3旧版／55宣言未解決 | 旧SOURCE問題から物理保持義務を自動推論せず、A/B未確認 |

所有者の「全75件を今後の作業用checkoutとして使用しない」方針は確定済み。再承認・利用終了の再質問は不要。元の保持分類と科学契約は変更せず、新しいscope別の状態をCSVへ追加した。**削除可能と確定した件数0、実際の削除・登録解除・原本移動・branch削除0。**

## 非公開パス対応とローカル依存scope

前回75件CSVの正確な集合と既存ID逆引きを使用し、75件の現在branch・完全HEAD一致、本人所有の正規絶対path、Git common directoryとの関係を確認した。既知のリンクと可視mount namespaceを静的照合し、別の実pathとして検証できたaliasは0。これは他host・未記録namespaceのalias不存在を意味しない。案内symlink68記録は候補から共有案内先へ向くもので、候補を必要とする逆向きの科学依存と区別した。

旧control／SOURCEの実パス一致409 contextsを再利用し、過去のworktree一覧・出版manifest・作業場所の記録と、実行時にdirectoryを要求する束縛を分けた。既存索引53,288 metadata entriesのstat差分では51,059件が不変、2,136件が現在利用不可等だった。large control93件と新規／HEAD変化checkoutの最近のcontrol/docを合わせた206ファイルを文字列だけ追加照合し、対象75 pathの新規一致0・読取エラー0。未利用ファイルや未索引の契約の不在まで保証しない。

この確認範囲では、**必須のローカルcheckout物理束縛は未検出75件**。現在の他Codex／別jobが参照していないことを証明したものではない。共有Git本体は全75件の保持前提として残し、通常の登録backlinkを科学的SOURCE供給義務と同一視しない。

保存先については、既存42,753 HEAD版保存pathをfile metadataで確認し、全件regular file、元worktreeへのsymlink／hardlink0だった。過去のSHA-256／Git blob一致を再利用し、科学payloadを再hashしていない。**保全した選定済み実ファイルを読むための元checkout依存は75件で解消済み**と確認できた。科学runtimeの移設、全script再実行、固定SOURCE/path契約の解除、外部raw custody・別障害域backupを確認した意味ではない。

## 安全な照合方法の準備

既存の環境同期方針の実在するローカル写しと、Aの固定報告を確認した。絶対path対応表等は各環境に保管する方針であり、**今回用途・宛先に使える既承認の非公開共有経路は確認できなかった**。共有領域・外部サービス・暗号／認証基盤を新設せず、非公開情報を送信していない。別目的の転送方針を今回へ流用していない。

非公開原本に、正規path／確認済みalias／mount観測／共通Git／SOURCE-control参照の完全表、A向けの必要最小query表、照合仕様と残条件を用意した。A向け最小表はID・branch・checkout HEAD・確認済みdirectory tokenと公開SOURCE/result記録locatorで、private rawの配置や新保存先の具体位置は送付対象から外してある。実際の受渡しは未実施。

Aの後続仕様に合わせ、非公開の補足queryにはpath flavor、環境scope識別子、観測日時も明示した。他hostでのrealpath解決・case fold・Unicode変更・環境変数展開を自動適用せず、親成分を含む表現はsymlinkとの関係を確認してから扱う。

公開CSVはID・branch・HEADと判定状態だけの形式を兼ねる。A側は次を別欄で確認する。

1. branch名／HEAD一致はGit履歴の論理参照。科学SOURCE commitや物理directoryの一致とは別。
2. SOURCE／result記録のcommit・path・hashは各fieldの意味で照合する。文書を含むcheckout HEADを科学SOURCEと推定しない。
3. 合意した非公開経路で受領した正規path／確認済みaliasを、Aの既存control・provider/input契約・既存job／再開設定・共有Git／editable環境参照へ境界付きで照合する。basename／branch名だけで実pathを推測しない。
4. exact pathの文字列一致も、過去snapshot・provenanceだけか、実行時にdirectoryを必要とするのか、役割未解決かを区別する。相対path・変数・mount別名は根拠を確認してから解決する。
5. token未受領・範囲不足ならPATH_VERIFICATION_NOT_POSSIBLEを維持する。不一致は確認scope内だけの結果で、依存不存在を一般化しない。公開へ返すのはID・branch・HEAD・判定状態だけとし、詳細path・文脈・mount情報は各環境に保持する。

現在、Aでは4件の論理参照検出が記録されているが、全75件の物理path状態は未確定。Bは実行中のDF-native B3を保護して後日照合する。他目的のB3や同じbranch名から実行依存を推測しない。

## 次段階と公開範囲

**ユーザー／A担当**は、既存手段の中から宛先・今回75件の最小表に適用できる非公開受渡し方法を確定する。**A担当**は上記仕様で照合し、公開可能な状態だけを返す。**B担当**はB3に干渉しない方法で自身の実参照を確認する。**各研究担当**はprivate原本／archive custodyと、必要なら旧SOURCEの回収を継続する。これらは所有者の将来workspace利用終了の再確認ではない。

**ローカルCodex（照合結果を受ける次の指示後）**は、57件・図表7件・旧SOURCE未解決11件を分け、物理依存、保全記録、SOURCE/path条件を突き合わせる。旧SOURCE未発見を理由に11 checkoutを自動的に維持必須へ格上げせず、逆に記録保全だけで物理依存を解除しない。実際の登録解除・directory削除には**ユーザーの具体的対象別指示**と**該当Codexの直前確認**が必要。

公開するのは本directoryの新規3資料だけ。旧SOURCE本体、非公開対応表・本文、内部絶対path／username／host/IP／mount情報、private raw配置、credential、内部pathの単純SHA-256は公開しない。既存の保全領域・SOURCE／STOP原本・公開資料を変更せず、独立した公開cloneから最新remoteを親にして通常fast-forward pushする。Aの並行成果が先行した場合も保持する。

- [LOCAL_N3_HISTORICAL_SOURCE_RESULT.md](LOCAL_N3_HISTORICAL_SOURCE_RESULT.md)：3旧版の期待hash、限定探索scope、55宣言と物理保持の区別。
- [LOCAL75_PATH_RECONCILIATION_STATUS.csv](LOCAL75_PATH_RECONCILIATION_STATUS.csv)：75件のscope別状態と公開返却様式。
- [LOCAL_PATH_RECONCILIATION_REPORT.md](LOCAL_PATH_RECONCILIATION_REPORT.md)：本報告。非公開完全表・照合仕様は本人所有の既存保管体系の新規directoryへ保存する。

確認時点：2026-10-26T13:25:49.179534+00:00。**mandatory STOP。worktree／branch削除・登録解除・prune、原本移動・既存archive上書き、科学SOURCE／raw／grant／ledger／marker／STOP変更、新規科学計算・compile・sampling・数値再解析、A/BへのSSH・B3干渉は行わない。**
