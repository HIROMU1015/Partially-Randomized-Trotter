# 31,710,848 callsへの条件付き外挿

**実G7の所要時間は未測定・未確定。本表は人工workloadと同じ1 call費用、同じ混合率・pair負荷・出力密度を仮定した機械的な規模換算であり、B3の予報・上下界ではない。** baseline相当の人工callを31,710,848回処理した場合を示す。正式二passのinstrumentation差を省いているため、このまま本番時間として使えない。

| 人工scenario | Workers | 実測trials | 外挿Wall h | 外挿CPU h |
|---|---|---|---|---|
| pair8_light | 1 | 16384 | 2.042 | 2.054 |
| pair8_light | 8 | 16384 | 0.301 | 2.032 |
| pair8_mixed | 1 | 8192 | 2.985 | 3.006 |
| pair8_mixed | 8 | 8192 | 0.496 | 2.998 |
| pair8_heavy | 1 | 1024 | 20.483 | 20.634 |
| pair8_heavy | 8 | 1024 | 5.186 | 20.575 |

軽量/中間/重いscenarioは楽観・中間・保守的な人工仮定の比較としてのみ読める。実G7がこの範囲に入る証拠はない。64 workerの機械scaling倍率は8 pairへ適用しない。

別のmixed人工pass比較では、instrumented/baselineのtask CPU比は中央値 **3.224×**。仮に同じ比が本番で成り立つ場合、二つのfull-N passを両方baseline扱いした表のCPU部分は約(1+3.224)/2倍になる。ただしこの倍率自体が人工endpoint・対象Python関数の呼出頻度に依存し、実G7へ転用できない。

本来は `T = S + 256*c_correctness + 15,855,296*c_baseline + 15,855,296*c_instrumented + Ω` を実入力の各arm/layer別で積算する。並列wallはCPU合計をworker数で割るだけではなく、各pairの長さとpass barrierを考慮する。CPU-boundなら負荷偏り・共有競合、I/O-boundなら出力bytes/実効帯域の床が残る。

不足しているのは実G7のpath発生率、bit draw数、packet/conditional law費用、operand長、F3 source/state guard・angle/requery費用、全Nでのcompact row割合、fine profiler呼出頻度、実際のoutput帯域である。今回のuniform Nと本番のarm別保存Nも異なる。短いfixtureのsetup/full-prefix費用を線形に増やす影響もある。科学保存値の新規再解析でこれを補っていない。
