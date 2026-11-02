# CPU・wall費用と比較条件

最上位の比較値は `host_outer.wall_ns` と `host_outer.parent_cpu_ns + worker_cpu_ns`。人工fixtureの構築・fake authority発行・stream再読監査は時間外。実hostのpin確認、admission/SQLite fsync、親scheduler、全workerのfork・setup・work・IPC・終了・join、finalization、外側STOPの費用を時間内に含める。system Python 3.12、追加依存なし。共有OS cacheはuncontrolled、OS cache flushやCPU固定・共有設定変更は行わない。

worker合計CPUは全owned childをjoin後のRUSAGE_CHILDREN差分。各pair CPU/wallはworker内のsetup～close。合計との差にはchild起動/送信/終了等がある。親のProcess.startのCPU/wallは別記するが、child wallや全体wallへ足さない。親process CPUとworker合計CPUだけを足して総CPUとする。

原Meterのexclusive coarse stagesを維持し、generator/audit、angle/requery、serialize/hash/write/flush、source check/constructorのCPU/wallを各pair reportに残す。原FunctionObserverのfine partitionは別に記録し、coarseへ足さない。どちらもouter_totalと残差へreconcileする。人工fine対象には原driver実import pathと人工fixtureを加え、parallel側はscheduler wrapperも含める。これは本番の科学関数call頻度を再現したinstrumentationではない。

`control_rpc` のCPU/wallは予約、ENTER、COMPLETE、byte reservation/ACK等。これらは既にgeneratorやstream等のcoarse範囲とworker総費用に含まれる**重複する診断座標**で、追加で合算しない。P1_1/S対S0の差が新管理のend-to-end費用を表す。parent/childの重なるwall、同時worker wall合計、RPC wall合計を総wallと呼ばない。

各workerのbaseline/final RSS/HWMと、親＋active owned childrenのsampled RSS和を保持する。RSS和には共有pageの重複があり、unique physical RAMでも連続peakでもない。S0はaggregate RSS samplerを追加せず、各worker HWMを記録する。parent/worker `/proc/self/io` 差分とlogical output bytes、state更新pwrite bytesを分ける。page cacheへのunbuffered writeは永続device throughputの測定ではない。

STOP後は親のprocess CPU差分、STOP→全join wall、worker CPUの推定差分を記録する。後者はSTOP時のowned `/proc/pid/stat` ticksとreaped child CPUを合わせた観測で、最大workers/CLK_TCK秒程度の量子化誤差を持つ。消滅pidなどで観測不完全ならnull。未観測費用を0にしない。強制終了によりcleanup費用を消して比較しない。

並列では4方式/A/BがCPU/cache/memory/I/Oを競合する。pair別のCPU/wallやinstrumented/baseline差は逐次R1と同じ測定条件ではない。phase内max/mean CPUを負荷偏りとして記録する。人工workloadではBだけangle surrogateを実行するため、Bがcritical pathになりやすい。正式Nも人工uniform Nと違うので8倍や旧pilot倍率を外挿しない。

科学費用比較の選択肢は、(a)逐次R1を比較の主指標として維持し並列を別実行で補助観測する、または(b)新たなconcurrent bundle内のpair CPUを主指標として採択する、等。前者でも追加実行は別予算/承認が必要で、既存31,710,848 callsへ逐次測定を自動追加しない。今回いずれの科学ルールも採択しない。CPUの公平性を保ったと無条件に主張せず、ユーザーがestimandを選ぶ必要がある。

## P1_8のCPU内訳（秒、3反復中央値）

| 負荷 | setup | generator/audit | angle/requery | evidence | save/hash/write | 親CPU | 親launch CPU | worker RPC CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 0.1834 | 3.7595 | 0.5747 | 2.6839 | 4.0252 | 1.2760 | 0.0182 | 1.0841 |
| mixed | 0.1957 | 2.0606 | 0.5202 | 1.2493 | 2.1754 | 0.7043 | 0.0197 | 0.5671 |
| heavy | 0.1840 | 1.0733 | 2.6678 | 0.5137 | 1.2788 | 0.5230 | 0.0291 | 0.2952 |

setup～saveは原coarse partitionの分類合計。correctness検査/観測/残差/child送受信・終了等が残るためこれだけを総CPUにしない。launchは親CPUに含まれ、RPCはworker coarse/totalに重なる。fineも別partitionであり、この表に追加しない。全ての独立stage名・CPU/wall・fine・残差は圧縮reportに保存。
