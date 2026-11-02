# STOP・保存・費用の統合契約

機械定義は[protocol.json](output_schema/protocol.json)。global identityは
`phase/pass_name/arm/layer/axis/trial_index`、state offsetはtask.offset＋axis_index×N＋index。
親がNOT_STARTED→START_RESERVEDを保存し、別のENTERでSTOPを再確認してSTARTED→IN_FLIGHTへ進む。
statefileは1 byte/callで、全31,710,848 slotの予定でもlog 256 MiBに含める。実full-Nファイルは作っていない。

Coalesced IPCは前trialのCOMPLETEとwrite ACKを次messageへ同梱する。
COMPLETEは正常emit returnと次のengine境界から証明され、axis/bundle末では即時confirmする。
親STOP後に新しいENTER許可は出さず、開始済みcallは協調drainする。
started_observedは親の開始許可の観測数。worker_entry_nsはENTER要求の送信前時刻で、実generator本体への物理entry時刻ではない。
例外の物理発生から親通知までの他workerの進行をゼロとはしない。P1は逐次R1と同じSTOPではない。
FAIL/ENDの複数通知を全件保存し、同一trialの重複通知を別callとして数えない。到着順は物理的な全順序ではない。

F01のcurrent attempt、last completed、candidate trace、committed traceとphysical stream prefixを分離する。
前trialのeventを現在失敗へ流用しない。失敗をZEROへ変換しない。
flush失敗では完全なJSON行があってもemitがreturnせず、論理完了0となる例を公開する。
partial write、worker喪失、parent crash、複数失敗、STOP保存不能を正常完了へしない。
primaryとcleanupの失敗を別記し、outer Admission STOP失敗は例外へ付加する。

Result 64 GiB/log 256 MiBは全worker共有。毎writeの正確なbyte数を親が事前予約する。
未ACK/short-writeの余剰creditは保守的に保持し、返却・再割当・再送しない。
pair JSONLはexclusive xb/unbuffered、append/resumeなし。state pwrite、flush間隔4096、保持規則は元P1を保持。
result/journal/stateにfsync耐久性を新たに主張しない。ledgerのdurable commitとは別。
parent crashのraw START_RESERVEDは後から書換えず、少なくとも一件のunknownという別観測を保存する。
親の最終CPU・started/completed不明はnull、予約はspentのままである。

Summaryはcoordinator snapshotを保存し、journal末尾receiptのsummary_sha/summary_chainで結合する。
formal/source-pins/runtime-manifest digestも照合する。summary保存前のbudget/cost snapshotと、
host reportのfinalization後outputおよび外側Admission STOPを混同しない。
summaryのみで全成功と判定せず、24 pair/statefile/journal/fake ledger STOP・保存エラーも確認する。

主比較表のCPUは`host_outer.parent_cpu_ns + host_outer.worker_cpu_ns`、wallはhost_outer.wall_ns。
host entry・全SOURCE/runtimeチェック・fake approval/SQLite・spawn/import・pair処理・IPC・保存・join・receipt・外側STOPを含む。
fixture/authority生成、readback、tar圧縮は測定外。
Pair CPUはsetup source checkからobserver退出/cleanupまで。spawn前半の費用はchild rusageとpair和の差をstartup/unattributedとして残す。
coordinator snapshot、execution_outer、host_outerは異なる範囲で、加算しない。
coarse/fineも別partitionで、親workerの重なるwallやRPC waitを単純加算しない。

setup、generator/audit、angle/requery、evidence、serialize/hash/write/flushはcoarse/fineのraw recordへ残す。
observer callbackの影響はouter CPUとresidualに含む。独立したobserver費用を分離できたとはせずunknownを0にしない。
unknown_bit_operations、lost pair CPU/wall、親crashの最終費用はnull。
RSS和は共有pageを含むsample値、worker HWMと/proc I/Oも保存する。
S/SFはsequential_pairs、P/PFはconcurrent_pairs。baseline/instrumentedを区別し、observer OFFは用いない。
並列CPUを正式な科学estimandへ採択していない。
