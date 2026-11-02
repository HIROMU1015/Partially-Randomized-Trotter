# Monitoring・CPU/wall設計

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

| 項目 | 現時点の取得境界 |
|---|---|
|parent＋全worker CPU、global outer wall、条件wall|固定Bundleに既存instrumentation。pair CPU/IPC/observer/stagesを再加算しない。pair wall和をglobal wallとしない|
|setup/online、generator＋audit、angle＋requery|固定pair coarse stage。fine function observerは別partition|
|Full local coefficient/child law/requery|静的にpacket/parent/word_series/_mass/child_law/event/enclose_atanの関数境界あり。実値は実API接続後のみ|
|Fraction bit演算費用、observer単独費用|現状識別不能。C算術はPython callerへ帰属しbit単位費用へ分解しない|
|serialization/hash/write/flush/close、IPC|固定pair/Bundle記録。親の全個別I/O exclusive費用はnull|
|own RSS/HWM/IO、output bytes|既存記録。RSS samples/HWMはunique continuous global peakではない|
|bit consumption|新人工adapter traceとoffline replay。実G7値はnull|
|inflight/STOP/drain/UNKNOWN credit|固定Bundle trusted trace/RAM/physical/journal join|

新port n256人工観測はparent0.119676087s、全worker39.095133000s、total39.214809087s、outer wall21.936534378s。port exerciser自身の値で、固定Bundleや実科学費用の値ではない。worker preflight/constructorとprocess setupは全worker CPUへ含まれ、pairのtimed region外の分をpair stageへ推定配賦しない。observer込み、coarse/fineを足さない。setup不足・未知費用は0へ置き換えない。

S→P1固定順、共有host負荷、cache等の交絡がある。実G7 path率、accept/reject、有理数bit長、requery頻度、保存量、CPU分布、Full-N費用はpending。CPU母平均CI、fused単独改善率、量子confidenceからCPU CIは導かない。資源確認64CPU・空きRAM十分・CPU PSI0の観測下で合理的並列化。共有OS/cgroup/package/service/他processは変更0。
