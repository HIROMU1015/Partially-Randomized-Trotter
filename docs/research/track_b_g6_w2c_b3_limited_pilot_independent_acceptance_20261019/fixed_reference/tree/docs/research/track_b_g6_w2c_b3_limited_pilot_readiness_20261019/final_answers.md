# Q1–Q10

**Q1.** 独立v3 consumerを固定commitから選択導入した。consumer実装は原本bytesと一致。Python 3.12対応で新規導入testの関数名照合1行だけ修正し、原本/修正後hashを残した。

**Q2.** CLIによる公開raw再検証は正常12件VERIFIED_COMPLETE、障害35件TECHNICAL_FAILURE。46件の費用境界が成立。parent crashの1件はnull/INSUFFICIENT_EVIDENCEのまま。全caseの入力hashは前後一致。

**Q3.** 新fake scope/storeでatomic race、spent、ACK不明、親crash、binding、STOP順序を確認し、保存18ケースをproducer非importのcheckerで再読した。旧47件のP07/P08/P11、完全contract不足30件、P06不足11件、historical runtime/private証拠不足は保持する。

**Q4.** 正式release/input/runtime/adoption、独立approval、単一trusted storeとlogical scope、retry0/resumefalse、cooperative STOP/unknown/no-refund、保存すべき操作履歴を新契約へbindする必要がある。現在はfake authorityだけであり本番への接続権限はない。

**Q5.** P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m5、sigma+1、4方式・A/B・X/Y・両pass、A6 math-only、既存precision/law/bit/event/confidence参照、SOURCE/input hashを候補JSONへ固定した。新seed namespaceは未採択提案。

**Q6.** 推奨検討案はprofile256/cell/pass/条件、S/P1二条件で16,896 calls。64案4,608、1024案66,048も提示。correctness16と複数の固定prefix blockからpath/CPUを記述する根拠を示したが、科学的十分性・coverage・CPU CIを保証していない。旧31,710,848契約は変更・実行していない。

**Q7.** S/P1を同じfused codec・CachedObserver・保存/trace・入力で比較する設計。host parent＋全workerをCPU総額、host outerをwallとし、pair/coarse/fine/IPCを重複加算しない。setup/online、angle/requery、observer/residual、serialization/hash/write/flush/finalize、RSS/HWM/IOを区別。二条件だけではfused単独の増分利益を識別しない。費用は固定portfolioの運用費用である。

**Q8.** 新候補31件PASS。consumer関連540件、別の既存v3統合125件も今回PASS、収集IDの重複なし。途中のfixture/portable assertion/保存形式/harness問題は修正し、履歴を保持した。最終人工受入に未解決failure/error/skipなし。実科学bridge未提供は閉鎖境界であり、科学実行準備完了とは表現しない。

**Q9.** ユーザーとGPTがscope/N、S/P1やreference codec条件、coverage/計測精度、seed/trace/予算、STOPとCPU estimandを決める。次にその採択scopeの正式bundle bridge/独立adapterを別SOURCEとして人工検証・公開し、SOURCE採択と別authorizationを要する。現SOURCEは一つのfake v3 projectionをone-shot実行する準備実装であり、二条件の実科学bundleを実行するbridgeではない。

**Q10.** 今回の実G7/F3 sampling、科学SOURCE import、新科学計算、実authorization/grant/claim発行・消費、量子native build/compile/synthesisはすべて0件。新fake SQLiteだけを試験に使用した。既存科学SOURCE/契約/raw/grant/ledger/marker/STOPは保持。公開検証後mandatory STOPし、次の担当はユーザーとGPT。
