# Log/result予算の技術評価

Result64 GiB、Log256 MiBが未採択候補。logはcontrol journal＋statefile保持長＋ordered trace保持bytes。trace自己writeは同じglobal budgetへ一回課金し、自己課金を再帰的にjournal化しない。累積pwrite trafficを保持量へ足さない。

公開正常caseは768 calls、log1,679,509 bytes。独立readerがreserved=ACK=physical、recorded nontrace credit＋trace自己bytesを照合した。receiptはsummary前trace prefixを束ね、後続保存のwrite操作はHost最終trace SHAと予算へ含める。

## 独立synthetic負荷

3案に対応する操作列を新規生成した。64 hex run id、各trial RESERVE/ENTER/COMPLETEに人工result write2回（reserve/ACK各2回）を加えた7操作/trial。二条件を別run identityで生成し、旧core＋独立readerで再読した。科学sample/bitstreamは生成していない。

|n|二条件の静的calls|二条件trace物理bytes|
|---|---:|---:|
|64|4,608|11,407,852|
|256|16,896|41,899,756|
|1,024|66,048|164,161,976|

この明示人工形状ではtrace単独は256 MiB内。journal/state/最終化を加える余地はあるが、**実G7全pathのbytes上界は未確立**。保存event/contextの有理数・bit長、write分割、coalesced ACK、failure/F01、将来二条件bridgeの共通管理費用に依存する。結果fileの総量もこのsynthetic reservation値から認証しない。n256の候補はこの検査で排除されないが、実科学run_readyや必ず予算内とはしない。

## Full-Nのformat下界

1行に64 hexのrun id・previous・shaがあり、これだけで少なくとも192 bytes。正常trialにはRESERVE/ENTER/COMPLETEの3行が必要。従って旧31,710,848 callsでは少なくとも **18,265,448,448 bytes**。JSON keys/escape/seq/trial ID/WRITE等を無視した保守的下界だけで256 MiBを超える。1点の実測を線形外挿した結論ではない。同じ粒度のFull-N traceを仮定できない。

将来Full-Nを選ぶ場合のcheckpoint/transition集約・failure近傍保持は別契約の設計課題であり、今回は実装していない。粒度を黙って落とさない。

## 保存障害

公開log cap100は状態保持768 bytesより小さく早期STOP。新しいRESERVE trace writer故障ではstate/counter不一致が発生した。独立checkerが成功へ通すことはないが、producer会計修正が必要である。未ACK/未記録creditは返却せず、incomplete traceを明示する次版が必要。詳細はproducer_consumer_discrepancies.md。
