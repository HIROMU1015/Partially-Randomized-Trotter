# 保全・構造検査・終了監査

2026-10-21 JST。主担当Codex自身の技術監査。別担当の独立科学レビューではない。

## 許可範囲と実施

GitHub connectorで指定7commitと関連SOURCE/結果/監査をread-only取得し、固定Git blob、SHA256、保存JSON field、CSV文字列、gzipの構造を照合した。新しい科学moduleはimportしていない。独立worktree/branchだけを編集し、主checkoutの削除状態や他worktreeのHEAD/statusを変更・固定していない。

- [実取得記録](remote_acquisition.json)と[remote blob照合](remote_blob_validation.json)は、connectorが返したblob IDとローカルGit object IDの一致を示す。取得したremote contentを独立に再hashしたという意味ではない。
- [来歴JSON](provenance.json)は今回読んだ固定資料のpath/blob/SHA256を示す。後続[構造検査](structure_validation.json)で同一blobを再読し一致を検査する。
- [SOURCE receipt照合](source_receipt_matches.json)はH2/H4/H6の公開実行bindingにある科学SOURCE file hashを固定Git bytesへ照合する。historical private runtime全leafの再認証ではない。H8は取得済みSOURCEとpacket/profile補足の対応を区別する。
- [転記JSON](literal_extracts.json)を保存結果のJSON pointerへ一致確認する。平均/SE/分散/確率/費用差/上界/順位の新演算はしない。サンプルmetrics例と旧pilot保存文字列も出典一致を確認する。
- Markdownのローカルリンク、未解決placeholder、stdlib importと禁止された科学関数呼出の不在、scoped内部絶対パス/資格情報patternを検査する。限定pattern検査であり包括的なsecurity保証ではない。
- [保全開始metadata](preservation_start.json)と[保全監査](preservation_audit.json)はこのworktreeに展開した旧tracked fileを検査する。4索引への追記だけを許し、旧bytesがそのままsuffixとして残ることを検査する。原科学SOURCE/result/raw/grant/ledger/marker/STOPを書き換えない。

最初の構造検査は、まだ生成していない監査JSONへのリンクで停止した。[技術検査履歴](technical_audit_attempts.json)に保存し、既知の2出力だけを生成後に存在確認する順序へ修正した。科学実行の失敗・再試行ではない。

## 非実施と終了状態

新sampling、simulation、signal計算、科学array decode、回路生成、compile/transpile（合成回路を含む）、統計解析、数値再解析、再ランキング、新しい希少確率/期待費用/分散/上界の数値生成は全て0。既存最大値をboundへ昇格せず、欠測を0補完しない。GPT/Claude詳細科学レビューと最終evidence freeze未開始。新たな科学grant/ledger/markerの発行・消費なし。

科学worker・monitor・background jobを起動していないため、それらへの停止signalやreapは不要。新規監査scriptは同期実行して終了を確認する。他ユーザー/別4次PF研究のprocess・共有OS/cgroup/scheduler/package/library/権限・既存venvを操作していない。AI subagent使用0、恣意的CPU/RAM/worker制限0。

専用branchは `codex/pr-rare-rte-existing-evidence-20261021`、基点は `80b7a35c4dbf683f952293eeb55944bc10b9efc8`。文書成果物とmetadata検査scriptをローカルcommitする。GitHub pushは0、公開範囲の承認をこの調査承認から推定しない。最終commit IDとclean状態は最終チャットおよびcommit後metadata監査で報告する。

[今回のSTOP記録](STOP.json)により追加stageへ進まない。**mandatory STOP**。次の担当：ユーザーが固定条件と必要な保証範囲を選び、Bの保存値解析・Cの新取得・Dの前提検証のどこまでを承認するかを判断する。Codexは新しい科学実行承認まで停止する。
