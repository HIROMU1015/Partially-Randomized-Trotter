# A1 N3：持続経路の人工検証

2026-11-06 JST、SERVER1 GPUA100-gleap-01。**unit13件＋cron経路6ケースPASS。科学計算0。**

## 固定SOURCEと実行

管理SOURCE v1=`d139cf8dd7d3ff8de6527bc27815cde35a1b4076`、修正版=`5cb542ece4d208878a1dfc7011007b99297a5a55`を、各cron投入前にcommit・公開本文/blob/tree照合した。
実行SPECにcommit、7依存sourceのSHA256、Python実体hash/version、client identityとjob policyを保存した。
実N3、DFモデルの数値評価、RNG word、state作用、Born、Hadamard、実compile、N1/N2/engine認証は一切実行していない。

人工jobは固定bit0とmock価格11をjournalへ書くだけ。量子shot/RZの実科学値ではなく、CVや実campaignへ使用しない。
各caseの1workerは1個の有限な人工taskに対応し、N3のworker数制限ではない。実worker数はcallerが実際の空き資源から選ぶ既存方針を継承する。

## 検証結果

| case | 確認結果 |
|---|---|
| 正常終了 | 親exit0、worker exit0、人工bit/価格を保持 |
| worker異常 | 自分のworkerがexit23。親は失敗exit17、先行STOPとworker exit23を保存、bitを保持し価格は欠測 |
| 親異常 | 自分の親がSIGKILL。managerがraw-9/signal9を保存し、STOP後に残った自分のworkerを回収。workerの終了コードはUNKNOWN |
| client/接続終了を模した状況 | 投入clientはcron job開始前に終了。別の人工clientも終了させ、親/workerは正常完了。実SSHは操作していない |
| 明示STOP/保全 | STOP receiptのmonotonic時刻がPOOL_DRAIN_STARTより早い。保存済bitは維持し、欠測価格を0で埋めない |
| manager異常 | 自分のapplication managerがSIGKILL。外側recorderがraw-9/signal9を保存し、独立STOPとown group回収を実施。未保存の親/worker終了コードはUNKNOWN |

全caseは同じ既存cron経路から実行し、`/system.slice/cron.service`のmembership、親/worker identity、child/ancestor resource記録を保存した。
同じSPECのdispatch再呼出はexit72で拒否し、次のmanager/workerは起動しなかった。一時cron項目はapplication開始前に削除した。
終了後、自分のcrontabが元通り存在せず、両人工runの14件の親/worker identityが不在または非稼働であることを確認した。

## 初回の不具合と修正

qualification_run01では正常caseの実行後、`WORKER_EXIT_CODES.json`をPID identity fileとして読んだため、後処理がKeyErrorで終了した。
cron membershipとcrontab撤去はこのrunでも確認したが、suite成功とは扱わない。v1 source・元の失敗log・取得済人工データを保全した。
数字PIDを持つ`WORKER_<pid>.json`だけを選ぶhelperへ修正し、回収側も同じhelperを再利用する回帰testを追加した。
修正版でunit13件PASS後、独立した**人工qualification_run02**を実行し、6ケースを完了した。これは科学retryではない。

## 再利用と試験の範囲

[unit test](../../../tests/test_a1_n3_managed_runtime.py)は、既存crontab保護、同じreceiptの二重使用拒否、science SPEC拒否、引数/戻り値の保存とpatch復元、STOP順序、worker例外、bit/価格欠測、PID再利用拒否、不要な科学依存の非importを確認する。
旧lifecycle observer/atomic writer/journal parserを再利用した。既存科学codeと旧397件の再取得はない。
managerのraw waitと科学完了は別概念。人工caseでmanager exit0でも、その配下の親は-9/17となることを記録している。

## 証拠と未保証

- [qualification結果](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/qualification_run02/QUALIFICATION_RESULT.json)、[SPEC](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/qualification_run02/SPEC.json)、[cron identity](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/qualification_run02/CRON_IDENTITY.json)。
- [unit結果](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/UNIT_TESTS.json)、[raw log](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/UNIT_TESTS.txt)、[失敗log](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/qualification_run01/qualification.log.txt)。
- [終了後確認](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/QUIESCENCE.json)、[logコピー/hash](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/LOG_COPIES.json)。Gitがignoreする.logも全bytesを.log.txtとして保全した。

実SSH切断、最後のlogout、host/cron daemon/外側recorder全体のkill、実engine長時間負荷を試験していない。
signal送信元の一般的なOS追跡と、wait owner消失後の全worker終了コード回収は保証しない。
unit/mock/短命人工jobのPASSを、実N3完走・数値保証の再認証・CV改善・外部CI/独立再現の証拠にしない。

人工jobは終了し、これ以上の実験を行わず、成果物公開照合後mandatory STOPする。
