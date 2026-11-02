# C1-5A：P5 worker異常終了・技術的STOP

**`C1_5A_TECHNICAL_FAILURE_STOP`。科学attempt 1、retry 0。3方式比較は未完了。** 科学SOURCE、precision、seed、試行数を変更せず、再実行していない。以下は保存済みterminal/summaryの引用と保全情報であり、新しい科学計算ではない。

SOURCE：[900a5278c604b4c58b794ca891720d9dea293919](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/900a5278c604b4c58b794ca891720d9dea293919/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021)。authorization-only直接子：[079ba428257444997d89ee2a975251563d8d16b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/079ba428257444997d89ee2a975251563d8d16b0/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/authorization_receipt.json)。公開bytesを照合してからone-shot markerを消費した。

## 実行範囲と完了状態

Linear H4 1.00 Å、STO-3G、DF12、R1新partition（generation-prefix LD3）、8 system qubit、13 basis/324 label。同じexact p、m=5、T=4/5、q=1、r=4、sigma=+1、tail時間1/5。H2048/K4096、eta=rho=tau=10^-12、angle K256/J64、seed20261021。各方式1024 path、各4 stepを事前固定した。[契約](execution_contract.json)

| 方式 | 状態 | 保存済み完走証拠 |
|---|---|---|
| Ordinary | METHOD_COMPLETE | 1024 path / 4096 microstep、32 batch |
| General Full | METHOD_COMPLETE | 1024 path / 4096 microstep、32 batch |
| Closed P5 | TECHNICAL_FAILURE_STOP | 104,977群のsetupを保存。その後worker異常終了。完走batchは未保存で、実際に途中まで処理したtrial数はUNKNOWN |

P5の欠測をaccepted=0・費用=0として扱わない。3方式合計3072 pathの完走は認定できない。固定SOURCEの`make_report.py`は未完了結果を拒否するため実行していない。正式3方式comparisonやresource winnerを作成していない。

## 確認できた障害と未確定事項

[P5 terminal](result/closed_P5_full/terminal.json)は、`coordinator`がfutureの結果を取得する箇所で`BrokenProcessPool`を記録した。workerが実行中または待機中に突然終了したという範囲まで確認できる。[P5原stderr](result/closed_P5_full.stderr.log)にも同じtracebackがある。rootはP5の非zero exitを検知し、[root terminal](result/terminal.json)をSTOPとして保存した。

P5は群構築後、forkした21 workerによるbatch処理へ移行していた。workerの実exit signal・個別exit codeは原ログに残っておらず、直接の終了原因は未確定。OOM、segfault、外部killなどのどれかを断定する証拠はない。

[失敗後cgroup観測](result/post_failure_cgroup_observation.json)では、自session/user scopeのmemory.eventsにoom/oom_kill=0、memory.max=maxを記録した。これは失敗後の集約観測であり、当該workerの原因証明ではない。memory.peakもsession/user全体の履歴であり、この科学jobのpeakと呼ばない。共有環境の設定変更や他jobへの干渉はしていない。

P5 setupで記録したmax RSSは1,770,984 KiB。大きい群表とfork/COW・worker動作の関係は次の技術調査候補だが、今回原因を確定するstress test、SOURCE修正、H4再実行は行っていない。科学式やprecision不成立が検出された失敗でもないため、P5法の科学的否定にはしない。

## 完走した2方式の保存値（探索診断）

以下は[Ordinary summary](result/ordinary/summary.json)、[Full summary](result/full_return/summary.json)、各setup/terminalの保存値を表示用に丸めたもの。論理回路は**TAIL_ONLY_PLUS_SCALAR**。deterministic halves、native、物理準備/readout、十分NはUNKNOWN。

| 観測項目 | Ordinary | Full |
|---|---:|---:|
| accepted / 全path | 1024 / 1024 | 1024 / 1024 |
| valid zero path | 0 | 0 |
| 経験的weight二乗平均 | 1.1141049662602474 | 1.1141049662602474 |
| 最大観測weight（binary64表示） | 1.0555117082535121 | 1.0555117082535121 |
| raw logical instructions / classical trial | 285.125 | 284.6494140625 |
| raw logical depth / classical trial | 76.673828125 | 76.619140625 |
| sampling等のworker batch CPU秒 | 57.444453536 | 4510.117112964 |
| online wall秒 | 3.9195715226233006 | 282.16968657821417 |
| method hierarchy総CPU秒 | 71.138041 | 4523.42039 |

この標本ではzeroによる回路回避は観測されず、weightの表示値も同じだった。logical instruction差は小さく、method別の固定PRNG系列による標本差を含む。受理率・moment/rangeの母数や認証上界、同精度総資源の優位は推定していない。

Fullの観測CPUは大きいが、既存local全child queryとF3検査を繰り返す実装境界の値である。parent packet cacheや最適selectorを備えた実装の普遍的比較ではない。全方式で同じangle/event cache方針を使い、既存samplerを変更していない。

P5のnormalizer/selector setupはCPU 283.087432224秒、wall 283.0995527729392秒で保存されている。[P5 setup](result/closed_P5_full/setup.json) onlineは欠測であり、このsetup値だけでP5とFullの総費用を比べない。setupの償却範囲も未評価。

## 判断へ返す範囲

今回取得できた2方式の固定小標本から、追加研究投資に対する強い肯定材料は認定できない。一方、強いP5対照が未完了であり、研究の継続・縮小・終了を決める完全な3方式データは揃っていない。技術的停止を研究上のnegative resultへ置換しない。

最小の後続候補は、保存証拠に基づくworker終了・fork/メモリ・監視設計の技術調査である。その必要性・scopeをユーザーが明示承認した後にCodexが担当する。H4再実行には新SOURCE/契約・新authorizationが必要で、今回のone-shotは再利用できない。native証明・全baseline closure等を先に増やす提案はしない。

## 保全と公開形式

SOURCE 549 files、input45、runtime30のSHA256は科学実行後も一致。[保全照合](preservation_verification.json)。旧R1/C1-3-R2/C1-4.5、旧失敗系列、grant/claim/ledger/marker/STOP、非公開原本2件、他worktreeは変更していない。今回の消費済み[marker](attempt_marker.json)をそのまま保存する。

raw全513 filesは内容を変更せず[アーカイブindex](raw_archives/index.json)に各bytes/SHA256を記録した。アーカイブ全memberを元bytesと照合済み。元filesも専用runtimeに保全し、公開用には4 MiB以下のtransport partへ分割した。これは科学STOP後の保存・公開処理であり、科学SOURCEや結果の再計算ではない。[復元説明](STORAGE_AND_PUBLICATION.md)

**公開HEAD・bytes/SHA256・保全・clean確認後mandatory STOP。科学実行は再開しない。**
