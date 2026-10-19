# v3人工費用の独立集計

47件中46件は `VERIFIED_PARALLEL_COST_BOUNDARIES`。親crash1件は `INSUFFICIENT_EVIDENCE`、総CPU/wallはnull。これは保存時計値の範囲と算術の照合であり、時計測定そのものの外部認証ではない。

Host total CPU = host_outer.parent_cpu_ns + host_outer.worker_cpu_ns。Global wall = host_outer.wall_ns。coordinator snapshot ≤ execution_outer ≤ host_outerを同じprocess尺度で照合。summaryはcoordinator snapshotで、finalization・outer STOPを後から加えた別snapshotと区別する。

Pair CPUはworker totalに含まれる。worker total − 既知pair CPUをstartup/unattributed/lost-pair-inclusiveとして別表示する。setup/share X/Y/Full prepareはpairの既存coarse subtotalから一度集計し、globalへ足し直さない。coarseとfineは各々outerへ独立reconcileする代替partition。IPC、launch、finalization、observer callbackはouter内なので加算しない。generator/audit、angle/requery、evidence、serialize/hash/write/flush、memory観測、I/O、residualを保存する。

単独observer overhead、receipt self-cost、lost pair CPU、親crash outerは観測不足ならnull。RSS合計をunique physical memoryとしない。並行worker wallをsumしてglobal wallにしない。失敗prefixはラベルを残して正常comparison CSVへ出さない。

P=mode P1/workers8/reference codec、PF=同fused、S=mode S/workers1/reference、SF=同fused。S/SFも同じv3 scheduler・cached observer・coalesced transportを使用し、旧serial S0とは違う。障害時にmeasurement_conditionフィールドがない場合はSOURCE定義をsettingsへ適用したものと明示する。

`artificial_process_costs.csv` はcase/variant/workload/N/flush/mode/worker条件を保持した観測表で、勝敗・改善率・科学estimandを採択しない。正常各caseのcomparison_tableはそのbundle内の記述値。P/PF/S/SF横断を同一実行条件の統計比較としない。実G7の費用や利益率へ外挿しない。
