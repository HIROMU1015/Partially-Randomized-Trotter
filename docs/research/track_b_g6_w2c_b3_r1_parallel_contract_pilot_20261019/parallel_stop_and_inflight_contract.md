# STOP・in-flight契約案

**未採択。逐次R1と同じ最初の失敗prefixではない。** P1/P2とも、親が最初の失敗通知を処理してglobal STOPを設定した後、新しいRESERVE/ENTER許可を出さない。既存in-flightは強制killせず、完了または失敗prefixを保存してjoinする。ホストの一時的なPSI/CPU競合をkill条件にしていない。

| 状態 | 意味・課金 |
|---|---|
| NOT_STARTED | 予約なし。科学callなし |
| START_RESERVED | 親が一意ID/offset/call枠を確保。再発行・返却なし |
| STARTED | ENTERを親が受理。直ちにIN_FLIGHTへ遷移する開始marker |
| IN_FLIGHT | 親のENTER許可済み、論理完了未確認。call費用対象 |
| COMPLETED | 原F01 engineでemit正常return後のcommitが確認された |
| FAILED | 失敗のterminal worker reportあり。F01のcurrent/last/prefixを保持 |
| CANCELLED_BEFORE_START | reservation後、ENTER前にSTOPを確認。再利用なし |
| TERMINATED_WITH_UNKNOWN_RESULT | worker喪失。開始/書込の不確実性を隠さず枠を保留 |

statefileは各trialの**最新状態**を保持し、全遷移の時系列ログではない。STARTEDはtransientで通常の最終ファイルには残らない。START_RESERVED→STARTED→IN_FLIGHT→COMPLETED、またはFAILED/UNKNOWNが基本。予約後の取消はCANCELLED。一意offsetとterminal再使用拒否により二重計上を防ぐ。

P1は最大workers件の未確定reservationを許す。P2はcap件へ絞り、START_CHECKを追加する。capはcall本体だけでなく結果の監査・angle・保存・次boundaryまでの未確定期間を含む。cap2では8 workerが同時にsamplingできるわけではない。P1でも親の原子的な予約/ENTER管理を使う。

保証の基準時刻は**親のSTOP処理**。worker内で最初に例外が起きた瞬間を全processから先取りできない。ENTER受理から実際のendpoint呼出しまでにも区間があり、親死亡・worker死亡時に実関数本体へ入ったかを完全証明できない。`started_observed` はENTER受理件数であり、実関数本体の厳密な開始回数とは断定しない。START_RESERVED/IN_FLIGHTで喪失した枠は不明として保守的に課金し、呼出し0としない。P2はこの原理的な競合を消さず、最大件数を小さくする。

各FAIL通知にworker観測時刻・CPU・stage・trial ID、親到着に `parent_arrival` と親時刻を記録する。代表failureは親が最初に処理したもの。ほぼ同時の全worker失敗も全terminal reportを保持する。後続の同一原因通知、STOP拒否、cleanup失敗も別観測として残す。親到着順から実際の例外発生全順序を推測しない。worker EOFは結果不明として記録する。

論理完了はStreamの行数と同義ではない。EngineStreamは原emit成功時にpendingを置き、次sample/bits境界・run_pair正常returnで親へCOMPLETEを送る。失敗経路ではF01のlast_completedとpendingのIDが一致するときだけ確認する。flush失敗でOSに完全行があっても、F01未commitならcompleted trialにしない。

親異常終了は人工 `_exit` で検証する。未確定state byte、pair stream、spent SQLite rowが残る。全workerは親pid/pipeを検査し、次境界で新規処理を止める。残存ファイルをread-onlyに調べ、START_RESERVED/STARTED/IN_FLIGHTを不明として扱う。親が死んだrunを正常receiptにせず、再開もしない。unbuffered/pwriteは電源断時の耐久性を保証するfsync契約ではない。

STOP保存が失敗してもprimary failureを保持し、secondary errorを返す。admissionのSTOP保存失敗でもRESERVED/ADMITTEDのspent rowは残る。log上限・disk障害等で全failure情報の永続保存が不可能なら、保存不完全のSTOPでありPASSにしない。障害注入した自分のworker/parent以外にsignalを送っていない。
