# B・D1・統合担当へのHANDOFF

対象はFull-H κ2 FULL_H_2a9a68a11d08af234a8c m0-X/Y、採択済み経験的価格規則4a1cb0d7、B6c5569c3の固定1.5倍配分。

D1 b25d7da6dedebc4cd22e8d6f00e6de75d7072292 はrequired coverage gateでSTOP。X/Yのcompile観測は各1062だが単価は両軸null、Cの受入0・保留2。同じ1 normal path、producer7f85f20e3。[D1_MAPPING.json](D1_MAPPING.json) で元STOP・PRICES・OBSERVED・receiptへ戻れる。

F1の保存m1〜m12・24軸を再予測せず接続する会計経路と人工テストを整備した。両軸保留では、T・m0部分額・割合・T−615304502・T−435481090はすべてnull。原F1負値/null、D7凍結係数と資格は変更しない。

統合ロードマップ差分：m0取得はD1で1path/2compile完了したが、参照価格受入は0/2で保留。24保存モデル軸との技術接続は用意済み、全13round比較シナリオは未成立。必要なのは同じ保存pathのcoverage条件と保存されなかった実装gateをD1でどう扱うかの明確化。Cはgateの採択変更・retry・代替pathを実施しない。本体更新は統合担当へ。

集計の承認は1構成・1配分について既存指示どおり維持され、両軸受入時の再承認は不要。ただし今回のD1 STOPを越える取得・固定契約変更の許可は追加しない。成果公開・remote照合後mandatory STOP。


固定C実行SOURCE：`5e619de429ddfc69a3a64da24a766b97db2b546a`。結果入口は [run01/RESULT.json](run01/RESULT.json)、軸別一覧は [run01/AXES.csv](run01/AXES.csv)。`m0_admissions`→元price/観測/compile locator、`rows`→physical ID/axis/N/保存F1 exact引用、`comparisons[*].benchmark`→新PF二基準のCSV labelへ戻れる。結合キーはconfiguration・physical ID・m・axis。X/Yの観測fingerprintは別、共有path数1。

受入0/2・保留2/2、保存F1引用24/24、全額/部分額/割合/比較差はnull。実加算の前で停止した。人工・回帰18 test PASS、Cの新sampling/build/compile/fit/Q3/予測/総額算出0。固定SOURCE・公開commitはGit履歴で区別し、旧STOPの上書きはない。
