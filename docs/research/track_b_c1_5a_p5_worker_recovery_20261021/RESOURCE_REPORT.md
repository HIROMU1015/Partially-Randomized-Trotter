# 費用・monitor補足（保存結果引用）

対象・精度は[handoff](handoff.md)と同じH4 R1固定入力。科学計算の完走と手法の資源優位を区別する。旧Ordinary/Full rawは開かず、旧summary/setup/terminalを引用した。

|保存CPU秒|Ordinary旧|Full旧|P5R新|意味|
|---|---:|---:|---:|---|
|sampler normalizer setup|0.026659721|0.203877525|283.993160344|input復元を別計上。P5の104,977群setupは依然必要|
|sampling/checks/event Evidence|21.573830977|4410.007131644|400.634031349|4096microsteps。算術単独の計測ではない|
|angle query/Evidence|7.104218642|67.692002726|23.539041366|cache miss数は37/44/24、共有範囲が異なる|
|logical build|22.452788567|21.202090197|20.856494351|同じ旧build_tail関数、nativeなし|
|全hierarchy|71.138041|4523.42039|998.71687|新はroot監視・新32process復元も含み、同負荷・同最適化ではない|

P5R prefix初期化は0.493558755 CPU秒。producer wait4は717.120416 CPU秒、logical workers合計wait4は151.310897 CPU秒。新providerのbasis/model復元は合計66.698587068 CPU秒、Evidence decode/validateは40.567487728 CPU秒。全hierarchyはこれらの他にimport・manifest照合・root監視・final summary等を含む。Timerの各stage CPU和とwait4/hierarchyは異なる境界なので、混ぜて共通online値へ見せない。

新P5 producer online wallは430.542883934秒、32provider全phaseは5.506168129秒、全jobは731.998624256秒。単一producerのsampling/checks/event Evidenceが主なonline CPUを占める。今回のproducerは巨大Python stateをfork継承させないための実装上の選択で、samplingが本質的に逐次であるという主張ではない。並列P5の最適実装・最速benchmarkとは扱わない。

|SOURCE由来のquery/cache計数|Ordinary旧|Full旧|P5R新|
|---|---:|---:|---:|
|sampler calls|4096|4096|4096|
|Full local parent packet|対象外|4096|対象外|
|P5 level2 sample+event|対象外|対象外|46|
|event cache hits/misses|3028 / 1068|3038 / 1058|3951 / 145|
|angle cache hits/misses|4059 / 37|4052 / 44|4072 / 24|

内部有理演算・bit演算の全回数ではない。旧各21workerのcache合計と新producer-global cacheの差がある。通常最適化したOrdinary/Fullにも同等のcache/selectorを認める必要がある。

新producer 1processとlogical32processは全てexit0、signalなし。完了33/33task、未完了0、再submit0。観測したsession/user memory.eventsの前後差は全0。これは今回の観測であり、旧workerのOOM/kill原因の確定・否定には使わない。

新producer max RSSは1,772,004 KiB。新logical workerの最大個別wait4 RSSは346,364 KiB。自分のactive childを定期観測した合計の最大はRSS 11,007,236 KiB、PSS 6,993,683 KiB、private 6,862,804 KiB。rootは除外、異なる指標の最大時刻は同一とは限らず、観測間のpeakを保証しない。共有pageを含むRSSの単純和を物理memory使用量と同一視しない。上位cgroup peakは他job込みなので新jobのpeakへ転用しない。

新resultのsummary直前の保存量は38,887,896 bytes。samplingとlogical batchにpathを保存し、fsync・monitor・再読込を加えたため、旧schemaの保存量とは異なる。最終raw transportのbytesは[archive index](raw_archives/index.json)参照。transport gzip/tarのCPU・公開作業は科学job外の保存作業で、科学online CPUに含まれない。

資料：[新summary](result/summary.json)、[terminal](result/terminal.json)、[monitor要約](result/resource_observation_summary.json)、[比較CSV](comparison.csv)。原monitor/exit/receiptsは全てarchive内で保持。追加計算・native取得・統計的判定は実行しない。
