# Global identity・state・STOP・保存の独立検証

## 固定版と内部境界

adapter `PUBLIC_P1_ARTIFICIAL_V1` は公開schema `R1_PARALLEL_8PAIR_ARTIFICIAL_RESULT_V1` と、`control.jsonl` の `b3_parallel_artificial_v1` PLANを扱う。
verifierは独自に3 phase×8 taskのarm/layer/key/offset/NとX→Y順序を構成する。correctnessは16/axis、profileは公開settings.profile_nを使い、正式保存Nの実samplingはしない。
run identityは入力manifestのrun_idで外から束縛し、global identityは `(run_id, phase, pass, arm, layer, axis, index)`。PIDやworker keyをtrial IDの代わりにしない。
同じlocal indexでも別pair/axis/runは別identity。固定task/path・offset・連続trial順とterminal prefixを検査し、重複予約/完了/ID、出力path・inode衝突を拒否する。

## Statefileとcall accounting

uint8コードは0 NOT_STARTED、1 START_RESERVED、2 STARTED、3 IN_FLIGHT、4 COMPLETED、5 FAILED、6 CANCELLED_BEFORE_START、7 TERMINATED_WITH_UNKNOWN_RESULT。
fileはplanned bytesちょうど、taskの固定offset順に64KiB以下のblockで走査し、同じ走査でSHAを計算する。24固定counterだけを保持する。
COMPLETED後の未完了位置を越えた再完了や、未予約gapを越えた予約を拒否する。snapshot counter・reportのplanned/reserved/started/state・call capを結合する。

| 座標 | 意味 |
|---|---|
| Planned | 固定task全trial位置 |
| Reserved | NOT_STARTED以外の枠。取消・喪失でも返却しない |
| Started observed | 親のENTER受理counter。実科学関数本体へのentry証明ではない |
| Completed | state COMPLETED。可視JSONL行数とは別 |
| Failed | state FAILED。samplingのrandom ZEROではない |
| Cancelled before start | 予約済み・ENTER前取消。再利用なし |
| Unknown | 結果不明terminal＋親喪失後に残ったSTART_RESERVED/STARTED/IN_FLIGHT。0呼出しへ推定しない |
| Not started | 予約のない位置 |

親crashでstarted counterを保存していなければnull。START_RESERVEDからSTARTEDを推定しない。Unknownは予約を保持し、不明callを補充しない。
stateは最新値であり、STARTEDは普通は上書き済み。最終stateだけで二重ENTER等の全履歴やSTOP時刻の順序を証明できない。
`max_inflight`と最初のfailure snapshotはworker/cap/ID/offset/状態の内部整合を確認するが、保存snapshot申告の範囲である。

## F01・STOP

各pairを旧stream/F01 verifierで検査し、event/canonical/chain/file SHA、ID/N/count、全文prefix、summaryを照合する。
control/reportのsafe_json Fraction wrapperを安全にdecodeし、F01を元APIへ橋渡しする専用temp controlだけを作る。入力directoryへは書かない。
last completed/current/inflight/candidate/emit正常returnの意味は変更しない。完全行があってもstate/F01未commitならglobal Completedへ加算しない。
LOST workerのrawにはengine_committed_trials=nullとし、raw可視prefixを別欄に出す。元pair verifierの可視行countをlogical completionへ昇格しない。
close時flush失敗は全trial commit後でも起こる。一方、emit内flush/file hash失敗は可視2行でも完了1になり得る。双方を別人工検査で確認した。

failureは親arrival連番・親monotonic時刻・全通知と代表failureのfirst-parent-arrival一致を検査する。process間の物理的例外全順序は推定しない。
log上限やdispatch中parent failureではWORKER_START/END等が欠け得る。SOURCEに対応する保存不足はevidence_gaps/PENDINGとして残し、偽の整合PASSへ変えない。

公開controlにはtrialごとのRESERVE/ENTER履歴がないため、**P08はPENDING_EVIDENCE**。
coreの `check_operations` は別のversioned normalized artificial APIであり、seq/run/task/trial、RESERVE→ENTER→terminal、phase barrier、STOP後RESERVE/ENTER禁止、drain、二重完了、未知creditを実行検査できる。
この人工操作logを公開producerが保存した形式だとは扱わない。将来のproducer固有adapterが実際のcontrol証拠へ明示対応を作る必要がある。

## Global保存

固定pair summary順と実completion_orderを区別し、PLAN/phase/WORKER_START/END/PID/report、summary envelopeとreceipt SHA/chainを照合する。
completion receiptのall_workers_joinedはproducerの申告。WORKER_ENDはproducerのjoin前に書かれるため、これだけでOS終了を独立確認したとはしない。
summaryはfinalization前snapshot、返却report.outputはsummary/receipt書込後。前者のresult bytesはpair分、logはstate保持bytes＋receipt前controlを指すことまで照合する。
Result64GiB・log256MiBと、reserved・ACK observed・実bytes・未ACK creditを別欄にする。write成功をACKと推定せず、未ACK creditは返却しない。
全trial完了申告でも未ACKが残ればVERIFIED_COMPLETEにしない。state保持bytesと累積pwrite trafficを同じ値とみなさない。

全文保持区間は独立trace再計算、compact以後はevent hash比較とsummary trace申告の照合まで。hashから全event/連結traceを復元しない。
全ファイルをRAMへ保持せず、最大行・固定24pairのsummary/function/failure metadata・64KiB state blockを使う。巨大単一行自体のRAMは必要。
