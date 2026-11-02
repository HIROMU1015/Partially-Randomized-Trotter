# Process scaling実測

3反復の中央値。min/maxはCSV、worker個別CPU・setup・RSS・I/Oは圧縮JSONLに保存。speedupは同じ人工task集合の1 worker中央値との比。総CPUはworker＋parent。

## pair8_heavy

| Workers | Wall s | Total CPU s | Trials/s | Speedup | Efficiency | Worker HWM MiB | Sampled total RSS MiB | Output MiB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.3812 | 2.3987 | 430 | 1.00× | 100.0% | 20.9 | 47.4 | 1.1 |
| 2 | 1.2847 | 2.3935 | 797 | 1.85× | 92.7% | 20.9 | 68.3 | 2.1 |
| 4 | 0.6825 | 2.3870 | 1500 | 3.49× | 87.2% | 20.9 | 110.1 | 4.0 |
| 8 | 0.6028 | 2.3918 | 1699 | 3.95× | 49.4% | 20.9 | 187.2 | 4.5 |

## pair8_light

| Workers | Wall s | Total CPU s | Trials/s | Speedup | Efficiency | Worker HWM MiB | Sampled total RSS MiB | Output MiB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.7987 | 3.8196 | 4313 | 1.00× | 100.0% | 20.9 | 47.4 | 2.7 |
| 2 | 2.1304 | 3.8038 | 7691 | 1.78× | 89.2% | 20.9 | 68.3 | 4.8 |
| 4 | 1.0792 | 3.7827 | 15182 | 3.52× | 88.0% | 20.9 | 110.1 | 9.5 |
| 8 | 0.5593 | 3.7792 | 29292 | 6.79× | 84.9% | 20.9 | 193.7 | 18.2 |

## pair8_mixed

| Workers | Wall s | Total CPU s | Trials/s | Speedup | Efficiency | Worker HWM MiB | Sampled total RSS MiB | Output MiB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.7765 | 2.7955 | 2951 | 1.00× | 100.0% | 20.9 | 47.4 | 2.0 |
| 2 | 1.5782 | 2.7892 | 5191 | 1.76× | 88.0% | 20.9 | 68.3 | 3.4 |
| 4 | 0.8886 | 2.7812 | 9219 | 3.12× | 78.1% | 20.9 | 110.1 | 6.1 |
| 8 | 0.4611 | 2.7886 | 17765 | 6.02× | 75.3% | 20.9 | 193.7 | 11.8 |

## synthetic64_mixed

| Workers | Wall s | Total CPU s | Trials/s | Speedup | Efficiency | Worker HWM MiB | Sampled total RSS MiB | Output MiB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 11.5132 | 11.6427 | 2846 | 1.00× | 100.0% | 24.2 | 53.2 | 2.0 |
| 2 | 5.9191 | 11.7288 | 5536 | 1.95× | 97.3% | 24.2 | 77.3 | 4.0 |
| 4 | 3.0221 | 11.7274 | 10843 | 3.81× | 95.2% | 24.2 | 125.7 | 7.8 |
| 8 | 1.5867 | 11.7031 | 20652 | 7.26× | 90.7% | 24.2 | 222.4 | 14.8 |
| 16 | 0.8511 | 11.6389 | 38501 | 13.53× | 84.5% | 24.2 | 416.0 | 27.5 |
| 32 | 0.4918 | 11.5560 | 66624 | 23.41× | 73.2% | 24.2 | 802.7 | 47.7 |
| 64 | 0.3130 | 12.6451 | 104690 | 36.78× | 57.5% | 24.2 | 1577.0 | 74.9 |

8-pair構造では8 workerでlight 6.79×、mixed 6.02×、heavy 3.95×。heavyではB側の長いpairがcritical pathとなり、worker追加だけで8倍にはならない。64-task測定は本番topologyの証明ではない。

64-taskの最高throughputは64 worker、並列効率が最も高い並列設定は2 worker。32 workerはCPU費用をほぼ維持して約23倍、64 workerは約37倍にwallを短縮した一方、総CPUは約8.6%増加した。目的がwall最小化かCPU効率かで選択が異なる。RSSは共有pageを含む和で、物理memoryの固有使用量ではない。出力はpage cacheへのlogical throughputで、fsyncを伴うdevice性能ではない。

起動/setup参考値：mixed 8 pair/8 workerのtask内setup wall合計は中央値10.97 ms、worker CPUのtask外部分（IPC/終了等）は30.01 ms、各workerのdispatch→collection wallとtask wallの差は中央値4.21 ms。64 task/64 workerではsetup wall合計77.73 ms、task外worker CPU247.40 ms。これらのwall合計をschedule wallへ足さない。
