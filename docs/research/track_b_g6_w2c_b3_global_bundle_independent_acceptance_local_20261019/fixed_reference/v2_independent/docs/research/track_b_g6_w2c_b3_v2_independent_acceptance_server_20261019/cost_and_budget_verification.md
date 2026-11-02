# 費用・予算の独立照合

final host reportが残る96件は既存独立cost partitionを使い、host parent＋全worker CPU、outer global wall、pair coarse/fine/setup/online/IPC、memory/RSS/HWM/IOの保存境界を検査した。12件のreport欠損ではtotal CPU/global wallをnullとして残し、0へ補完しない。失敗ケースの費用は正常比較へ混ぜない。

pair CPUをworker CPUへ再加算せず、coarseとfineを合算せず、IPCをouter totalへ加算せず、parent/worker wallをglobal wallへ足さない。CachedObserver fine partitionは代替の測定表現であり、observer単独のoverheadは未取得としてnullを維持する。sampled RSS/HWMやshared-page sumをunique memoryの厳密peakと呼ばない。

新RAM shadowはplanned slot当たり1 byteの追加RAM。物理state保持長とは別であり、logへ二度課金しない。正常人工stateは768 retained bytes、累積pwrite trafficは3072 bytesである。serialization/hash/write/flush/close、fault notification、diagnostic等の親処理はhost parent CPU/wallに含まれるが、各処理のexclusive costを独立測定済みとは主張しない。

log/resultのreserved、ACK observed、physical retained bytes、unknown creditを照合し、完全traceでは非trace credit＋trace自身のbytesを一度だけ課金してexact一致を要求する。不完全traceではprefixの下限と物理bytesを照合し、prefix後の会計を独立証明へ格上げしない。未ACK creditはreserved−ACKのまま保持する。diagnosticもresult creditに含め、無料にしない。

固定候補のresult 64GiB/log 256MiB、call/byte cap、retry0/resume falseは無断変更していない。人工fault用の小さいcapは公開fixtureの指定された設定としてそのまま検証した。実行資源は利用可能な64 CPU・十分な空きRAMを確認してraw再読と指定回帰を並列化した。共有OS/kernel/cgroup/quota/packageや他ユーザーのprocessは変更していない。
