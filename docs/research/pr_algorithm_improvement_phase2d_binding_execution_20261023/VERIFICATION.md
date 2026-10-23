# 結果保全・公開照合

保存済みB出力の監査はPASS。11attempt／11completed、39JSON＋log、24行、15payload canonical bytes/SHA256/producer、manifest依存参照、marker/approval/contract binding、SOURCE固定HEAD／10fileを確認した。新しい科学計算・再生成・API probeは0。

公開前の静的照合もPASS。結果文書基点の全1,095fileはbyte同一、新規結果namespaceの15文書・JSONだけを追加した。local文書リンク23件、private path／run ID／approval等の混入を検査した。保存raw40file・packet・marker・STOPと科学SOURCE HEADの不変、環境package922hash／runtime2,713hashの一致も確認した。原workspaceのstatus／worktree一覧／refsは開始時と公開前で同一だった。

公開時にremote HEAD、全公開file bytes／SHA256／Git blob、独立git fetchを照合してpublication receiptを確定する。原本raw／STOP／packetを変更せず、公開用要約だけを追加する。公開receiptを確定後もmandatory STOP。
