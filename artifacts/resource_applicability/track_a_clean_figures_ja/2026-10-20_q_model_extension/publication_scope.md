# 公開範囲の照合と今回の決定

**決定：LOCAL_ONLY_NO_PUSH。追加図・説明・入力写し・scriptをローカル専用branchでcommitし、pushしない。**

利用者の今回の指示は、追加packetを公開する前に固定資料の公開範囲と利用者承認を照合し、公開承認を確認できなければローカルcommitまで完了してSTOPする、という条件である。図・文書作成の承認を公開承認へ自動変換しない。

## 確認した記録

- 現在の6図は公開済み`55b719c8e0881bda848a8ed721b0cd452267b801`。このbranchを基点とし、原図・原notesは変更していない。
- 読取専用`git ls-remote`では450軸branch `codex/pf-q-model-quantitative-comparison-20261020`の先端が`7430a386bcd4dc545107e413cd022ab7cb261968`であることを確認した。公開branchに存在するという観測と、明示的な公開承認が確認できたことは別である。
- 固定450軸資料の[publication_scope](inputs/publication_scope.json)は`NEW_SCOPE_NOT_APPROVED_NOT_PUSHED`、[旧STOP](inputs/STOP.json)は`new_publication_authorized=false`を保持している。旧記録を遡及変更しない。
- 100 scopeの[証拠統合](inputs/B_evidence_integration.md)（`80b7a35c...`）も、公開状態と承認確認状態の区別を求めている。100 scopeの科学的採用は、この追加packetの公開承認ではない。
- 今回の追加図E〜E4・日本語説明・入力写しに対する新しい明示公開承認は確認できなかった。

公開の経緯や主体を推測しない。既存の公開済みcommitを再pushせず、履歴改変・force-push・元資料の削除も行わない。今回の停止は利用者が指定した公開条件によるもので、自動承認レビューの拒否ではない。

## ローカル成果物の範囲

専用branch：`track-a-q-model-figure-extension-20261020`。親commit：`55b719c8...`。新規pathは本ディレクトリと`scripts/extend_q_model_figures_20261020.py`のみ。既存SOURCE/raw/結果/grant/ledger/marker/STOP・正式主表・旧図・スライド原本を変更しない。詳細科学レビュー・最終evidence freeze・新規科学計算は対象外。

公開を希望する場合の次担当は**ユーザー**：追加図・説明・必要な入力写し・scriptのどこまで公開するかを明示する。**Codex**の公開操作はその承認範囲が確認できた場合の別作業。今回の表示・文書更新はローカル保存で完了し、mandatory STOPする。
