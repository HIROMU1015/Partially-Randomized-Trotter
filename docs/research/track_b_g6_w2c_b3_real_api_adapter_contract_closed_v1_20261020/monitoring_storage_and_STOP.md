# Monitoring・Storage・STOP

caller outer parent CPU＋wait済み全worker CPUを一度だけ集計。pair/stage/IPCは内数、coarse/fineは代替partition。global outer wallをpair wall和にしない。shared setup/finalizationをS/P1へ二重配賦しない。

既存CachedObserverが全pinned Python pathを選び、bits/codec/adapt_f3/adapt_g7/angle_binding等のcall-returnをinstrumented passに記録する。setup/online、generator+audit、angle+requery、serialization/hash、IPC、observer、storage/write-flush-close、RSS/HWM、own I/O、STOP/drain/closeは既存coarse/fine/controlから追跡する。

Python inclusive/exclusive境界とFraction C内部のbit演算を区別する。real scientific bit費用、Full local coefficient/child law、real F3角度費用、pure observer overhead、continuous unique global RAM peakは未測定null。人工CPUから科学費用や母平均CIを推定しない。

result64GiB/log256MiBはsingle global candidate budget。raw全file/ACK/unknown creditとfinal/delivery/callerを結合する。fsync/power-loss耐久性やcaller witness自身のcloseの完全証明はNOT_ESTABLISHED。trace/state/receipt/marker/close/caller障害で証拠不足を成功へ補完しない。

利用環境は64 logical CPU・約503GiB RAM。case並列3、P1固定worker8、S→P1逐次を維持。pressureを観測し、共有OS/cgroup/scheduler/venv/cacheや他者processを変更しない。一時pressureのみで正常workerを停止する設計を追加していない。
