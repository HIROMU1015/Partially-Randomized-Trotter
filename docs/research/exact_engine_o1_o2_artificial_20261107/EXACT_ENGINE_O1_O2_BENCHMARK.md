# O1/O2：人工benchmark結果

2026-11-07 JST、SERVER1。**人工入力限定。実H4・SCV/RPEの性能結果ではない。**

## 測定条件

GPUA100-gleap-01、AMD EPYC 7742 64-Core Processor、Python 3.12.3、stdlibのみ。SOURCEとPython/fractions hash、affinity・負荷・memory pressureは [ENVIRONMENT](raw_run01/ENVIRONMENT.json) と [SOURCE](raw_run01/SOURCE.json) に保存。科学依存moduleのimportは0。共有環境へのinstall/変更なし。

[PLAN](raw_run01/PLAN.json) と全振幅を含む [FIXTURES](raw_run01/FIXTURES.json) を測定前に保存した。手書き有理数10 fixtureを固定し、結果後の除外・置換なし。各列は同じ初期stateから12 gateを作用する。qubit数/行列構造/初期bit長はPLANを正本とする。人工Hadamard型は有理数係数で、実gate行列を再生成したものではない。

3方式の全6順列を各fixture/categoryで一回ずつ測り、6標本を保存。unmeasured warmup各1回。入力生成、correctness、gc.collect、記録処理はtimer外。GCはkernel中も通常通り。単独processは比較条件の統制であり科学worker数の上限ではない。wall=perf_counter_ns、CPU=process_time_ns。

denseは固定旧関数そのもの。O1は準備済み疎row＋毎gate index生成、O1+O2は同じ演算＋private index LRU。候補はFraction pair型の検査も毎回行い、旧denseよりvalidationが多い。dense fallbackでもこの追加検査は残る。

coldは各人工gate列開始時に疎rowと空index cacheを新規に作る条件。OS/page/allocator/CPU cacheのcoldではない。warmは疎rowと利用supportのindex planをtimer外で作成する。全方式でstate evolution結果は毎回計算し直し、state/p/bit cacheはない。O3以降の処理は導入していない。

## 前処理なし・warm contraction

時間は12 gate列1回のwall中央値 [最小, 最大]、単位ms。倍率は各方式の中央値の比。1未満は候補が遅い。

| Fixture | Dense ms | O1 ms | O1+O2 ms | Dense/O1 | Dense/(O1+O2) | O1/(O1+O2) |
|---|---:|---:|---:|---:|---:|---:|
| identity_k1 | 4.518 [4.472, 4.533] | 2.069 [2.063, 2.114] | 1.792 [1.787, 1.799] | 2.183 | 2.522 | 1.155 |
| diagonal_k3 | 147.777 [147.404, 148.290] | 19.082 [19.019, 19.164] | 16.588 [16.495, 16.618] | 7.744 | 8.908 | 1.150 |
| xx_block_k2 | 72.800 [72.236, 73.126] | 31.626 [31.403, 31.871] | 29.163 [29.035, 29.277] | 2.302 | 2.496 | 1.084 |
| controlled_k3 | 145.742 [144.902, 146.527] | 27.925 [27.856, 28.132] | 25.313 [25.214, 25.336] | 5.219 | 5.758 | 1.103 |
| sparse_complex_k2 | 84.310 [83.977, 84.631] | 47.989 [47.717, 48.088] | 45.495 [45.391, 45.528] | 1.757 | 1.853 | 1.055 |
| dense_k1 | 6.745 [6.721, 6.948] | 6.870 [6.846, 6.909] | 6.895 [6.876, 6.971] | 0.982 | 0.978 | 0.996 |
| dense_k3 | 44.749 [44.655, 44.935] | 44.922 [44.823, 45.108] | 44.984 [44.879, 45.327] | 0.996 | 0.995 | 0.999 |
| large_bits_diagonal | 90.107 [89.654, 90.434] | 13.670 [13.601, 13.731] | 12.379 [12.326, 12.507] | 6.592 | 7.279 | 1.104 |
| tiny_nonzero_k1 | 4.647 [4.621, 4.747] | 3.793 [3.784, 3.804] | 3.469 [3.459, 3.496] | 1.225 | 1.340 | 1.093 |
| mixed_sequence | 50.519 [50.324, 50.681] | 14.977 [14.940, 15.066] | 13.994 [13.890, 14.033] | 3.373 | 3.610 | 1.070 |

## 前処理込み・cold gate列＋旧Born＋旧threshold

時間は12 gate列1回のwall中央値 [最小, 最大]、単位ms。倍率は各方式の中央値の比。1未満は候補が遅い。

| Fixture | Dense ms | O1 ms | O1+O2 ms | Dense/O1 | Dense/(O1+O2) | O1/(O1+O2) |
|---|---:|---:|---:|---:|---:|---:|
| identity_k1 | 4.676 [4.640, 4.784] | 2.251 [2.242, 2.266] | 1.994 [1.985, 2.036] | 2.077 | 2.345 | 1.129 |
| diagonal_k3 | 149.872 [148.689, 149.989] | 20.850 [20.806, 20.945] | 18.548 [18.479, 18.609] | 7.188 | 8.080 | 1.124 |
| xx_block_k2 | 74.068 [73.838, 74.180] | 33.171 [33.033, 33.340] | 30.835 [30.768, 30.930] | 2.233 | 2.402 | 1.076 |
| controlled_k3 | 147.485 [144.437, 148.414] | 29.558 [29.134, 29.732] | 27.145 [26.684, 27.221] | 4.990 | 5.433 | 1.089 |
| sparse_complex_k2 | 86.958 [86.747, 87.115] | 50.665 [50.462, 50.850] | 48.534 [48.390, 48.664] | 1.716 | 1.792 | 1.044 |
| dense_k1 | 7.124 [7.105, 7.155] | 7.315 [7.296, 7.370] | 7.311 [7.303, 7.363] | 0.974 | 0.975 | 1.001 |
| dense_k3 | 46.158 [45.960, 46.431] | 46.293 [46.130, 46.369] | 46.343 [46.251, 46.556] | 0.997 | 0.996 | 0.999 |
| large_bits_diagonal | 92.557 [92.099, 92.674] | 16.044 [15.950, 16.073] | 14.877 [14.838, 14.895] | 5.769 | 6.222 | 1.078 |
| tiny_nonzero_k1 | 4.785 [4.771, 4.814] | 3.942 [3.922, 3.956] | 3.670 [3.655, 3.678] | 1.214 | 1.304 | 1.074 |
| mixed_sequence | 51.650 [51.491, 52.107] | 16.168 [16.110, 16.242] | 15.506 [15.461, 15.594] | 3.195 | 3.331 | 1.043 |

## warm gate列＋旧Born＋旧threshold

時間は12 gate列1回のwall中央値 [最小, 最大]、単位ms。倍率は各方式の中央値の比。1未満は候補が遅い。

| Fixture | Dense ms | O1 ms | O1+O2 ms | Dense/O1 | Dense/(O1+O2) | O1/(O1+O2) |
|---|---:|---:|---:|---:|---:|---:|
| identity_k1 | 4.674 [4.639, 4.694] | 2.226 [2.220, 2.246] | 1.949 [1.943, 1.958] | 2.099 | 2.398 | 1.142 |
| diagonal_k3 | 149.154 [148.636, 149.933] | 20.778 [20.692, 20.858] | 18.242 [18.182, 18.430] | 7.179 | 8.176 | 1.139 |
| xx_block_k2 | 74.264 [73.878, 74.726] | 33.136 [32.896, 33.423] | 30.638 [30.427, 30.740] | 2.241 | 2.424 | 1.082 |
| controlled_k3 | 147.550 [146.686, 148.630] | 29.570 [29.474, 29.813] | 26.866 [26.779, 26.999] | 4.990 | 5.492 | 1.101 |
| sparse_complex_k2 | 87.316 [86.842, 87.628] | 50.873 [50.554, 51.013] | 48.366 [48.172, 48.481] | 1.716 | 1.805 | 1.052 |
| dense_k1 | 7.127 [7.099, 7.159] | 7.299 [7.269, 7.411] | 7.329 [7.308, 7.447] | 0.976 | 0.972 | 0.996 |
| dense_k3 | 46.077 [45.967, 46.223] | 46.277 [46.107, 46.381] | 46.371 [46.264, 46.481] | 0.996 | 0.994 | 0.998 |
| large_bits_diagonal | 92.003 [91.790, 92.114] | 15.898 [15.865, 16.047] | 14.668 [14.652, 14.681] | 5.787 | 6.272 | 1.084 |
| tiny_nonzero_k1 | 4.833 [4.784, 4.893] | 3.964 [3.931, 4.004] | 3.638 [3.631, 3.740] | 1.219 | 1.329 | 1.090 |
| mixed_sequence | 51.499 [51.345, 207.631] | 15.999 [15.928, 16.097] | 14.960 [14.914, 15.006] | 3.219 | 3.442 | 1.069 |

## CPU time

cold end-to-endのprocess CPU中央値、ms。並列worker時間和ではない。他categoryのCPU rawと中央値も保存している。

| Fixture | Dense CPU ms | O1 CPU ms | O1+O2 CPU ms |
|---|---:|---:|---:|
| identity_k1 | 4.674 | 2.250 | 1.993 |
| diagonal_k3 | 149.769 | 20.849 | 18.539 |
| xx_block_k2 | 74.067 | 33.170 | 30.833 |
| controlled_k3 | 147.450 | 29.557 | 27.143 |
| sparse_complex_k2 | 86.957 | 50.664 | 48.532 |
| dense_k1 | 7.123 | 7.314 | 7.309 |
| dense_k3 | 46.140 | 46.292 | 46.335 |
| large_bits_diagonal | 92.525 | 16.043 | 14.868 |
| tiny_nonzero_k1 | 4.783 | 3.940 | 3.668 |
| mixed_sequence | 51.649 | 16.167 | 15.505 |

## 前処理とindex費用

50反復batchで測った1回当たり時間の6標本中央値、µs。sparse rowはfixtureの全unique matrix、indexは疎経路の全unique supportの合計。mixedはmatrix4種・index3種。dense fallbackではindexを作らない。

| Fixture | sparse row準備 µs | index新規作成 µs | warm lookup µs |
|---|---:|---:|---:|
| identity_k1 | 9.480 | 27.761 | 1.478 |
| diagonal_k3 | 83.872 | 204.529 | 1.660 |
| xx_block_k2 | 24.202 | 194.254 | 1.554 |
| controlled_k3 | 83.013 | 203.345 | 1.661 |
| sparse_complex_k2 | 22.449 | 192.645 | 1.562 |
| dense_k1 | 8.099 | — | — |
| dense_k3 | 61.954 | — | — |
| large_bits_diagonal | 84.322 | 104.445 | 1.671 |
| tiny_nonzero_k1 | 8.569 | 27.923 | 1.469 |
| mixed_sequence | 198.370 | 304.124 | 3.988 |

## Memory・cache

各fixture/方式をfresh subprocessで一度実行。cold end-to-endのtracemalloc peak bytesで、Python object allocationだけを追跡し、fixture生成/importは除外する。前処理・cache・中間state・Bornは含む。native allocationや全worker物理memoryの測定ではない。

| Fixture | Dense peak B | O1 peak B | O1+O2 peak B | O2 entries / indices / miss / hit |
|---|---:|---:|---:|---|
| identity_k1 | 12040 | 13568 | 14680 | 1 / 32 / 1 / 11 |
| diagonal_k3 | 148056 | 150468 | 154292 | 1 / 256 / 1 / 11 |
| xx_block_k2 | 117012 | 118772 | 124164 | 1 / 256 / 1 / 11 |
| controlled_k3 | 143880 | 146520 | 150344 | 1 / 256 / 1 / 11 |
| sparse_complex_k2 | 164580 | 167704 | 173096 | 1 / 256 / 1 / 11 |
| dense_k1 | 22988 | 24468 | 24788 | 0 / 0 / 0 / 0 |
| dense_k3 | 37420 | 43796 | 44116 | 0 / 0 / 0 / 0 |
| large_bits_diagonal | 134360 | 136704 | 138736 | 1 / 128 / 1 / 11 |
| tiny_nonzero_k1 | 12928 | 14536 | 15832 | 1 / 32 / 1 / 11 |
| mixed_sequence | 77636 | 84172 | 90924 | 3 / 384 / 3 / 6 |

RSSはfresh子processの前後high-water（bytes）を [memory raw](raw_run01/memory.jsonl) に保存。前値はfixture/import終了後のhigh-waterで、差をkernelの物理memory増と扱わない。複数processのRSS和も求めていない。retained Python allocationとPIDも同rawにある。
今回の全30sampleの実行後RSS high-water範囲は 27,262,976–27,262,976 bytesで、方式間の物理memory差をこの指標では識別できなかった。

O2 cacheは利用supportだけ遅延生成するLRU（default32 entry）。32はCPU/RAMの実行上限ではなくprototypeのcache容量。今回実使用は0–3 entryで、正しさテストでは容量1/2でeviction・再生成も検査した。index保存量は概ねentryごとO(state dimension)で、workerごとに複製すればその分増える。

## 判断と限界

疎/対角等の8 fixtureでは前処理込みでもO1がdenseより速く、O2の追加後にも中央値の低下を観測した。denseの2 fixtureは元kernelへfallbackし、検査・準備分の小さい悪化が残った。全fixtureを残しており、改善しない条件を除外していない。O2の小さい差は標本範囲と併せて読む。正式な統計的有意差、異なるhost/loadでの安定性は検証していない。

memoryは全fixtureで候補のPython peakが増えた。cold end-to-endでの時間短縮と小さい追加allocationは同時に観測できたが、時間とbytesを単一尺度へ合算していない。多worker、より大きいstate、低再利用率、1 gateだけの用途、実科学のbit長/構造分布での有利性は未測定。

実H4 gate/state/LAW/catalogは一切投入していない。実build・matrix hash・価格・RNG・manager・実worker schedulingは測っていない。従ってここでの倍率をH4全engineやPR/RPE総コストへ外挿できない。旧certificateの認証移転や科学的GOでもない。

## 再現と保存値整合

```bash
python3 -B test_o1_o2.py
python3 -B benchmark.py --run raw_reproduction
python3 -B render_reports.py
```

このdirectoryをcwdにする。benchmarkはfresh出力を要求し、既存raw_run01を上書きしない。render_reportsは保存raw_run01から表を再生成するだけでkernelを再実行しない。SOURCE hash・540 timing・30 memory・方式間threshold・SUMMARY再導出を検査してから出力する。

[timings raw](raw_run01/timings.jsonl)、[preprocessing raw](raw_run01/preprocessing.jsonl)、[SUMMARY](raw_run01/SUMMARY.json)、[correctness](EXACT_ENGINE_O1_O2_CORRECTNESS.md)、[REPORT](REPORT.md)。公開照合後mandatory STOP。
