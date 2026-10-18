# 測定方法・再現・来歴

基点7b6f526b、P1実装はef57714bのbytesと同じもの。共通AGENTS.md、worktree AGENTS.md、固定共通指示§8.4（7eff8a33のGit object）を明示確認。現在の利用者指示を過去runのworker capと区別した。主agent自身で実装・検査し、AI sub-agentや別reviewerは使っていない。

64 CPU affinity、起動時RAM available約469GiB、短時間CPU idle約94.7%、CPU/memory pressure avg10=0。8 pairに対して1/2/4/8 workerを比較する。共有OS・quota・cgroup・venv/cache・他ユーザーprocess/fileを変更しない。fresh worker/host/tempだけを所有し、資源観測の対象はhost情報と自分のPIDに限定する。旧科学条件・承認済み契約・grant/ledger/markerは不変。

CPU-bound人工Endpoint・SHA counter bitsは基点fixtureと同じ。light/mixed/heavyの人工profile Nは128/64/32、correctness16/axis、全3phase。BのFraction級数は元fixtureの冪方式で、F3 enclosureの実計算ではない。law/angle/bitを実G7と同じ性能だとみなさない。科学module importはblockerで拒否。fused canonical・冪漸化式・GMP/native・GPUの実装を取り込んでいない。

各設定3反復、各workload/configでN16 warm-up、固定seedによる設定順shuffle。SOURCEをcommitで固定してからregressionと性能を実行する。SOURCE hashは各suiteの開始/終了で照合する。主表には元S0、同じcached observerを使うS0、旧P1、observerのみ、IPCのみ、両方の1/2/4/8 workerを含める。before_exact8は原親をそのまま使い、追加Counter費用のない厳密な旧P1対照とする。

observer OFFは別の27回の診断であり、研究計測の代用にしない。状態pwriteのCPU/wall/call数はこの診断で追加観測する。instrumentedとOFFの差には実装/同期との相互作用もあるため、独立なobserver費用として単純加算しない。主比較はobserverを保持した実総CPU/wallで行う。

host_outerのwall、parent process CPU、全join後RUSAGE_CHILDREN差分のworker CPUを保存し、総CPUはparent+workersのみ。旧Meter coarse、fine、RPC、起動、memory、I/Oを別座標として残す。coarse/fine、RPC/outerを足さず、concurrent wallの合計を総wallにしない。未知post-STOP費用はnull。RSS合計は共有page重複を含むsample値で、unique物理RAMのpeakとは呼ばない。unbuffered出力はpage cacheへのwriteであり、fsync付きdevice throughputではない。

fake fixture作成・approval作成、SOURCE patch wiring、raw再読/圧縮はhost外側timerの外。実hostの検査、fake SQLite reservation/STOP、全workerの起動/setup・work・保存・IPC・join・summary/receiptは内側。source/policyの追加input pinsと同じ人工harnessを全比較に使う。人工authority/claimのみで、本番のadmission境界を緩めたとは扱わない。

全runの各pair raw chain/file SHAをreportと照合し、時刻費用 `generator_exclusive_ns` だけを除く正規化payloadをS0と比較する。ID、分類、event/evidence/angle、bit消費、trace、retentionを除去しない。raw hashはrun内部で照合し、時刻込みのraw SHAのrun間一致を要求しない。全call数/状態、fine/core call数、cost partitionも検査する。正常代表rawと全障害rawはtar.gz、全run reportはgzip JSONL、全file SHA/正規化SHAはruns.jsonへ保存する。

第1候補SOURCE45c29d98の90性能/27診断/17障害・354 testsと監査はattempts/v1_*へbytes不変で保全。observer_v2_probeは旧runtime45c29d98に対する6回の比較で、SOURCE hashを保存している。追加案を取り込んだ最終SOURCEはsource_inventory.jsonのcommit。旧測定を最終版の性能として流用しない。最終版の全回帰・性能・診断・障害を別のfresh出力先で実行する。

再現には公開baseとmeasurement_source_history.bundleを取得し、bundle verify後、source_inventory.jsonのSOURCE commitを自分の新しい領域へ復元する。

```
PYTHONDONTWRITEBYTECODE=1 python3 docs/research/track_b_g6_w2c_b3_parallel_overhead_optimization_20261019/run_tests.py <fresh-tests-directory>
PYTHONDONTWRITEBYTECODE=1 python3 scripts/tracks/algorithm_codesign/run_b3_overhead_artificial.py diagnostic --output <fresh-output> --repeats 3
PYTHONDONTWRITEBYTECODE=1 python3 scripts/tracks/algorithm_codesign/run_b3_overhead_artificial.py benchmark --output <fresh-output> --repeats 3
PYTHONDONTWRITEBYTECODE=1 python3 scripts/tracks/algorithm_codesign/run_b3_overhead_artificial.py faults --output <fresh-output>
```

性能suite同士は順次実行する。監査のverify_measurements.pyは既知の保存directoryだけを読む。過去の測定namespaceや使用済みfake scopeも上書き・再開しない。実scienceや実authorityへ自動接続するCLIはない。
