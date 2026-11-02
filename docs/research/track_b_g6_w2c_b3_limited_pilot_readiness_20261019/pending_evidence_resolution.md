# Pendingの境界

|項目|今回確認した範囲|残る状態|
|---|---|---|
|consumer導入・47raw処理|全件再検証、SOURCE/schema/input hash照合|解決済みの技術導入|
|P07/P11 atomic/one-shot|新fake storeの8者race、double claim、spent/retry0/resumefalse、ACK不明/親crash|人工的に確認済み。旧47件のprivate履歴不足は維持。実store/adoptionは正式判断待ち|
|P08 STOP順序|新実schedulerのhash付き操作履歴＋別実装checker|人工的に確認済み。旧47件に履歴を遡及しない|
|完全contract不足|旧障害30件の欠測を認識|保存証拠不足。再生成して原本とはしない|
|P06 worker/通知|旧11件の欠測を保持|保存証拠不足。新runのENDを旧runへ移さない|
|parent crash費用|旧1件と新crashの最終値null|保存証拠不足。0や推計値で補完しない|
|historical Python/private採択bytes|現在のruntimeをpinし、歴史runtimeと分離|歴史的証拠不足。現在のPython 3.12.3は代用ではない|
|障害時の成功receipt欠如|正常receiptがないSTOPとして扱う|期待された状態。成功としない|
|限定pilotのscope/N/seed/trace/CPU estimand|候補文書・JSON・閉じた人工実装|正式契約判断待ち|
|実科学bridge/production release|本SOURCEでは恒常的に拒否|未採択。別SOURCEの実装/審査/採択・別authorizationが必要|

機械可読のpending_evidence_resolution.jsonも参照。技術試験を科学的十分性・CPU母集団の保証・実行承認へ昇格しない。
