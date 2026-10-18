# 古典費用の記述的会計

集計対象は今回の**検証済み人工完了bundleだけ**。不完了cell、missing receipt、failure prefix、contract不一致は比較から除外する。入力result schemaと測定SOURCEは基点のまま。

`partition`は各rowのcalls/CPU/wall非負整数を検査し、exclusive_sum＋residual＝own outer totalをCPU/wall独立に確認する。
各trialのgenerator_exclusive_nsをkind別にstreaming合計し、summaryのgenerator stage calls/count/CPU/wallと照合する。

| 座標 | 課金範囲 |
|---|---|
| Setup | source、constructor_and_preflight、share/axis/final identity。pair X/Y共有分を一度 |
| Online | R/C/A別generator_and_audit、evidence、EVENT angle_and_requery |
| Output | serialize、chain/envelope hash、file hash、write、flush、finalize |
| Observation | memory baseline/periodic/final。correctness invariantsは別stage |
| Outer/residual | own child process CPUとmonotonic wall。observer effectsを含む |
| Fine | filename:line:functionの別partition。coarseへ加算しない |
| Parent shared | source/budget/module/dispatchを別座標。dispatch wallはchild wallを内包 |
| Final shared output | summaryのserialize/hash/write/flush/finalize。armへの恣意配賦なし |
| Memory/I/O | baseline/final RSS/HWM、periodic RSS、peak line、実output bytes/rows |

Full F3 prepareはconstructor_and_preflightに内包するため別prepareを加算しない。未知のcoarse stageや重複constructor名は未対応として拒否する。
CPU/acceptedはpair全CPUを観測EVENT数で割った記述値。accepted=0ではnull。CPU/trialはpairのX/Y完了数を分母とする。
4方式×A/Bをbaseline・instrumentedごとに16行で出力し、correctness8pairを別欄へ置く。
両passの差を純粋observer overheadと認定・差引きしない。変更したfine関数を同じlabelの費用だとみなさない。

欠測座標はnull。存在しないstageを測定済み0としない。receipt自己書込時間はproducer仕様のunknown/nullであり0を拒否する。
RSS/HWMを足して固有物理RAMと言わない。periodic RSSは連続peakではない。HWMはそのprocessのbaseline/import等を含む。OS cacheはuncontrolled。
親wall＋child wall、coarse＋fine、X/Y共有setup、prepare＋constructor、両passを混ぜたsingle比較値は作らない。
bit演算数、量子T/CX等の単位変換や任意合成scalar、研究勝敗・ランキング、母平均confidence certificateは作らない。

[JSON](artificial_example/cost_summary.json)／[CSV](artificial_example/comparison_table.csv)／[Markdown](artificial_example/cost_comparison.md)。例はfake clockとmock memoryを使用する人工値。
例のinstrumented fineは空の人工partitionであり、実SOURCEのfine費用の測定ではない。実FunctionObserverの動作は既存人工回帰で別に検証している。
