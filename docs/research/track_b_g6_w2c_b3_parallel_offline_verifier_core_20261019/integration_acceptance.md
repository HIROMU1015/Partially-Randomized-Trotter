# P01–P11・fused・CachedObserverの受入

各checkはID、対象SOURCE、固定commit、schema、期待、観測、判定、evidence、未解決を保存する。判定はPASS/FAIL/PENDING_EVIDENCE/REQUIRES_CONTRACT_CHANGE/NOT_APPLICABLE。

| 項目 | 実装・検査 | 限界 |
|---|---|---|
| P01 | run/task/trial ID、固定offset・pair/axis順、重複path/inode | run identityはmanifest外側から束縛 |
| P02 | state/F01/inflight/完了prefix、複数failure、candidate/committed | 未保存eventを復元しない |
| P03 | planned/reserved/started/completed/failed/cancelled/unknown/not-startedとcap | ENTER受理と実科学関数entryは別 |
| P04 | 予約bytes・ACK観測・実bytes・未ACK credit、result/log/state容量 | credit返却なし、writeからACKを推定しない |
| P05 | lost/parent crashの保守的unknown、実rawprefix | 歴史的process終了はproducer申告の範囲 |
| P06 | PLAN、3 phase barrier、成功前提、completion順とcontract順 | log欠落はpending |
| P07 | retry/resume flags、公開metadata、既存atomic admission runtime人工回帰 | prototype保存物だけのatomic claim独立証明はpending |
| P08 | normalized操作coreでSTOP後RESERVE/ENTERを拒否、既開始drain・複数failure | 公開P1にはtrial操作履歴がなくP08自体はpending |
| P09 | host/worker/pair CPU、wall、coarse/fine、IPC/launch/residual | SOURCEの計測境界申告を用いる。estimand未採択 |
| P10 | summary envelope SHA/chain/receipt、全pairとcontrol | joinはreceipt申告、WORKER_ENDはjoin前 |
| P11 | retry0/resume false、scope再利用拒否の固定Admission人工test | 未公開新hostへ一般化しない |

[独立例matrix](independent_artificial_example/integration_acceptance_report.json)。新server SOURCE/schemaへは全て未接続でありPENDING_SERVER_SCHEMA。
未知SOURCE変更を、人工trace一致だけで既存R1互換や正式採択へ変換しない。旧PendingParallelAdapterも変更していない。

## Fused canonical

ad3117d287c4c28c9b861b3b8dffa7bfbba407e1の実helper SOURCEを人工testでloadし、旧harness canonical_checksと直接比較した。
exact int/Fraction・8192bit整数、Unicode/surrogate/DEL、str-key collision（上書き前unsupported object）、event/evidence/angle、F01型failure文書のbytes/SHAがreferenceと一致。
原SOURCE完全SHAも照合。これはhelperの独立人工受入であり、今回新driverに接続・正式採択したものではない。
新serverのfused helperと同一とは仮定しない。後続では実helper/driver/observer pinとF01保存・partial-write全経路を新固定SOURCE上で再検査する。

## CachedObserver

公開77f8667fと一致するoverhead SOURCEをそのままloadし、safe driverを与えたdependency stubでscheduler部分を実行せずobserverだけを試験した。
旧ScopedObserverとcachedの同じnested artificial functionについて、label・call数・fake-clock exclusive CPU/wall rowが一致し、outer/residualに収まることを確認。
helper pathを観測対象に含め、pre-existing profiler拒否も確認。実時刻のns一致、異なるPython版間の全label一致、実科学function全domainは主張しない。
旧339/公開356の件数は参照履歴であり、今回の437件へ重複加算しない。
