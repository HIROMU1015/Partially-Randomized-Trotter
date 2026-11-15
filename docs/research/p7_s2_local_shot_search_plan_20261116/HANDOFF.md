# 統合担当へ：保存適合点からの少数round編集による有限探索案

提案の正本は[REPORT](REPORT.md)。今回は保存資料の照合と計画作成のみで、新しい配分・T・MSEを計算していない。

推奨はp7/S2・要求幅16h/7、b4296b110の保存B_PRICE適合点
`N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,4]`、条件付き費用435,481,090 RZを出発点とする手順。元configuration a6e55777…・signal carrier06a15ee4…・B2価格1002f1dfea…・total decoder/target/law/loss/τを固定する。

1. 出発点からm12を1/2shotずつ減らす2案、m11を1/2shotずつ減らす2案を順にQ3評価する。1shot減の失敗で2shot減を棄却しない。
2. まだ安い適合点がなければ、上の減算案から別roundへ1/2shot戻す、最大2round変更の振替を検討する。capは元N0、実費用は出発点未満。既評価familyの有効MSE上界と費用を選択優先度に使い、未知MSEの予測やround独立性は仮定しない。
3. 新規Q3は計最大8件（単round4＋振替4）。保存同等結果は再利用。安い適合点1件の発見で早期終了し、残枠で勝手に最小化しない。INCONCLUSIVEを適合にせず、重大gate失敗はUNKNOWNでSTOP、retry/上限拡張なし。

各候補の具体列・実費用・SOURCEをQ3前に固定する。入力と十分性の判定は全13round一組。元価格receipt・physical IDをtotal IDへ付け替えず、p7/S2価格の16直接/10条件付き閉包という区分を保持する。新T・候補順の算術やQ3は、この有限規則を採択した別途明示承認後に実施する。

他候補には、候補固有のconfiguration/bridge・13round packet・価格policy・同契約の保存適合seed・cap・error contractを入力として同じ手順を適用する。PF/PRのlaw・normalization・辞書・scheduleは移送しない。実装時は既存whole-shot/total binding/Q3/exact会計を使い、有限の候補順序部分だけを追加する案で、新controllerは作らない。

ロードマップ本体は統合担当が更新する差分：一律breakpoint2列から、**達成済み点を起点に局所編集を費用・Q3で選ぶ、最大8件の未実行案**へ次の選択肢を具体化した。435,481,090 RZという現在の保存達成額、新しい最良点・十分N/物理RMSE/Gの資格は今回変更していない。

計画公開後mandatory STOP。実装・新数値解析・Q3実行への自動移行なし。
