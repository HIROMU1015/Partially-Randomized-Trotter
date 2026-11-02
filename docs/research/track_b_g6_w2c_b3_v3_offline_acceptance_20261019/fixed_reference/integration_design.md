# 改善P1・fused・R1の技術統合

2026-10-19 JST。**TECHNICAL_CANDIDATE / NOT_ADOPTED / RUN_READY_FALSE**。
直接基点1cd2a7196bc48a4aa1a904d7bd351e3d8fbf45ff、最終回帰・測定SOURCE `860fef4a3a8ebb5a46628f4e0364e5934b5af416`。
旧R1/科学SOURCE/実結果/contract/authorityは変更せず、追加ファイルだけで構成した。

## 接続

新 `g6_w2c_b3_parallel_host.execute_artificial_v3` は、新Admissionの完全binding検査、独立fake承認照合、
既存SQLiteのatomic RESERVED commit、再照合、ADMITTED遷移を通してからschedulerを呼ぶ。
parentだけがauthority/ledger/claimを保持する。portのcontext退出は例外時もSTOPを保存する。
commit-ACK不確実性や再照合失敗でもreservationを解放しない。retry/resume APIは作らない。

| 新SOURCE | 再利用元と変更 |
|---|---|
| parallel_driver | 固定R1/F01 driverのコピー。exact canonical入口をreference/fusedで選択。旧executeと実science helperを閉じる |
| parallel_scheduler | 固定P1の予約/state/保存/phase処理をコピーし、coalesced client/parentを明示接続。spawnへ変更 |
| parallel_observer | 最終改善P1のcached callbackを適応。code identity/label/row cacheと全call/returnの排他的分割を保持 |
| parallel_admission | 旧Admissionのledger/approval engineを継承。v3 schema・runtime・schema pin・設定全体のbindingを追加 |
| parallel_host | 新entrypoint、phase barrier、global summary/receipt、外側STOP、最小spawn payloadを接続 |
| exact_serialization | ad3117d2のfused helperと**bytes同一**。冪漸化式等は取り込まない |

Production entrypoint `execute_approved_v3` は独立credentialを要求したうえで、非採択SOURCEを拒否する。
このSOURCEのProductionFactory/ProductionBits直接呼出し、実module loader、実ModuleType endpointも閉じた。
契約のboolや旧host v2の編集だけで科学実行へ昇格できない。将来のreleaseには新しいSOURCEと正式採択・独立実行承認が必要。
今回の技術接続が動作した証拠はfake authority/ledgerと人工endpointを用いた実Host経路であり、実G7動作検証ではない。

## Scheduler

3 barrier、各8 pair、pair内X→Y共有、24 fresh process、parent-only atomic call/byte reservationを保持する。
人工逐次対照S/SFも同じscheduler・cached observer・coalesced IPCで1 pairずつ実行する。
旧逐次R1 S0と実装費用が同条件であるとは主張しない。

forkからspawnへの変更で、親のauthority情報をworkerメモリへ継承しない。
渡すのはtask/settings、公開SOURCE/runtime manifest、最小endpoint/bit factory、pipe、STOP flagだけ。
factory payloadの検査と、親にだけ置いた秘密canaryがworkerへ継承されない人工試験を実施した。
独立spawnに必要なimport・interpreter起動・SOURCE検査費用は全worker CPU/host wallに含む。

## 科学条件

P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m5、σ+1、4方式×A/B×X/Y。
A6数学的enclosure τ=10^-6 rad、Kφ32、J32、物理RZ変更なし。
保存N、law/seed/bit順/event/phase/weight/word、誤差/confidenceは旧R1参照との完全比較で保護する。
`full_schedule_static.json` は256＋15,855,296×2＝31,710,848を構造だけ検査する。
実correctness256/full-N、実G7/F3 import/sampling、新科学計算・実保存科学値の数値再解析は行わない。

途中版bdb6/11aeの試験・中断測定はattemptsへ保全し、最終値へ混ぜない。
11aeの自分の人工benchmarkだけをSIGINTで協調停止して直接helper guardを補強し、最終SOURCEで全再検証した。
科学計算や他ユーザーprocessは停止していない。検査はprimary agent自身によるもので、別agentの独立レビューではない。
