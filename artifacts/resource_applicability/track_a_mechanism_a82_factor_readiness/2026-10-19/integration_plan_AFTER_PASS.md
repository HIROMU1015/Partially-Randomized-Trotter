# 将来PASS後の統合計画（今回は未実行）

別途明示承認されたone-shotのnative受入がPASSした場合のみ、今回の利用者指示に従い次を行う。runner自体から次stageへ自動移行しない。

1. 新A82 factor deltaを、既存31比較の保存因子を変更しないstatus overlayとして加える。ID32 unique、2軸ずつ、各7因子/7log項目をmetadata/schemaで確認し、32/32の比較説明因子完備と64/64 side検証を分けて記録する。旧欠測status・過去31/32履歴は保持する。
2. 新論文manifestとClaim–Evidence更新案へ、SOURCE/input/delta SHAと経験的条件を追記する。元core32・SE・費用/shot比・他31比較の因子は再計算しない。
3. 既存Figure 2の数値・描画座標・cohortは維持し、来歴・caption補遺だけを更新する。A82因子を実際に追加描画する必要があれば、その表示変更の範囲を別途判断する。
4. H4 legacy DF12 Pの別cohort、H6直接/model、H8独立n48を維持し、並行H4/H6や共通protocol v1へ接続しない。
5. 保存・独立commit後mandatory STOP。公開承認が明確な資料だけpushしremote SHAを確認。不明ならpushしない。最終evidence freeze・詳細科学レビューは開始しない。

新因子はempirical allowanceと保存DFモデル下の条件付き会計であり、母平均優位・厳密誤差保証・FT資源削減には拡張しない。
