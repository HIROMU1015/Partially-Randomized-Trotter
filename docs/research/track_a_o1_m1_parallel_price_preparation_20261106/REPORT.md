# M1価格計測：並列化不足の確認と継続版の準備

2026-11-06、SERVER 2（GPUA100-gleap02）。**現在のC0計算は中断・変更せず継続中。新SOURCEは人工テストまでで、実H4の並列実行は開始していない。** 起点は固定`8a13539e51bb113e82cb86ddb608e2e6e27786ca`。

## 現状の確認

64 logical CPU・affinity64に対して、科学worker PID1848166の3秒区間CPU time増分は3秒、すなわち価格処理はほぼ1コア分だった。MemAvailableは約480GB。自分のcgroupと祖先のCPU/RAM上限はいずれも`max`だった。既存driverはround/armを順次処理し、providerも32 trajectoryとX/Yを順次compileする。M1/M2のagent並行は行ったが、価格の独立タスク分配は不足していた。実行driverの設計上の不足であり、CPU/RAMのquotaやユーザー指定上限が原因ではない。

既存Gaussian/native relative fusion等は再利用済みだが、これらは価格計測自体のprocess並列化とは別である。稼働中SOURCE・旧raw・STOPを保全したまま、専用worktree/branchに以下を準備した。

## 実装と科学条件の維持

- [並列provider](../../../src/trottertracks/o1_m1_parallel_price.py)：固定builder・full wrapper・fingerprint・`transpile_and_measure_cost`・metric accumulator・SE処理を再利用。`spawn` poolで独立タスクを処理する。
- 利用可能CPU数がsample数より多い場合はtrajectoryのX/Yも分ける。64CPU・32 trajectoryなら64 taskになる。CPU数が少なければtrajectory単位で両軸を処理し、重複buildを抑える。MCの先頭1 taskを先に処理してpeak RSSを測り、その結果を再利用する。残り最大63 taskの同時数はCPU・task数・空きRAM／実効cgroup残量と`2×観測peak RSS`から決める。2倍はcompiler/trajectory間のmemory peak変動へのoperational allowanceで、固定の低いworker/RAM上限ではない。baselineは独立な2軸を最大2 workerでcompileする。各worker内のBLAS/OpenMP/Rayon thread poolは1 threadとし、外側process並列と二重にCPU数分のthread poolを起動しない。親や既存科学processの環境は変えない。
- PCG64のevent列は親で同じseedから1回生成する。X/Yは同じevent tupleを受け取り、32 trajectoryという科学sample数を維持する。worker完了順によらず元のsubmission順に既存Welford集計を行い、両軸6指標・SE・cost contextを維持する。
- baselineの一体compile内部を分割する方法やproxyは追加していない。baseline単独pointの並列性は最大2軸であり、全体が64倍速になるという意味ではない。実H4でのspeedupは未測定。

[継続runner](../../../scripts/run_o1_m1_c0_parallel_resume.py)は、元の[serial driver](../../../scripts/run_o1_m1_c0_pilot.py)をSHA256 `0b6da463853bc3b6ba881cae636e6169ef751e3311e5dcc8f92d1cb42c24f8e3`のbyte同一コピーとして再利用する。稼働中の元ファイルは変更していない。

継続時は最新checkpointを読み、C0構成、全7 signal/baseline検査の完了、既存価格のcontext/law/compiler/価格scopeを照合する。完了済み価格行を保持して未完了pointだけを計測し、保存済みshot条件も一致確認する。signal・state-action検証は再実行しない。現在の保存schemaの読取確認では完了16 point・未完了5 pointに対応した（以後の進捗は切替直前に読み直す）。

旧workerのPIDがまだ存在する場合は実行を拒否し、旧rawを書き換えず新しい出力directoryを使う。`C0_NOT_QUALIFIED`やmandatory STOP済みのrunを継続対象にはできない。CLIはworkerを停止する機能を持たない。SCI入力/SOURCE hashは再現性のために保持し、承認証明書やgrant発行は追加していない。

## 人工検証

最終人工suiteは**28 passed in 82.83 s**。対象は[並列価格テスト](../../../tests/test_o1_m1_parallel_price.py)と[継続テスト](../../../tests/test_o1_m1_parallel_resume.py)。発明した2-mode入力および2 spatial/4 spinのstrong spin native回路のみで、Pauli/DF、実Qiskit1.3 compilerのgate指標、平均/SE/metadata、paired/split axes、worker PID、重複draw保持、worker例外、観測memoryからの同時数選択、完了行再利用、未完了pointだけの実行、shot drift拒否、SE不適格STOPを確認した。

Pauli回路の自動生成Gate名がprocess間で違うため、初回の名前fingerprintに依存するmock価格テストは不適切だった。テストをserial referenceの全operator（人工専用atol `1e-12`）との直接照合へ修正し、実compilerの指標一致も追加した。継続テストでは、Python tupleを保存JSONのlistへ変換した後の値を比較するfixtureへ修正した。科学SOURCEや実H4の許容値は変えていない。

runtimeは既存のM1専用Python 3.12 / NumPy1.26.4 / SciPy1.14.1 / OpenFermion1.6.1 / Qiskit1.3.0をread-onlyで再利用した。cacheは`/tmp/o1_m1_parallel_price_tests`配下のworker別directory、bytecode/pytest cacheは無効。実H4 snapshotのdecode・新operator計算・新compile・価格再計算はしていない。

## 切替に必要な操作とSTOP

1. **ユーザー**：正常稼働中の旧jobを計画停止し、この別SOURCEへ同じC0条件で継続することを明示許可する。現在arm内の個別trajectory checkpointはないため、その未完了pointは同じseedで32本を最初から計測し直す。完了済みpointは再compileしない。
2. **Codex（切替許可後）**：旧workerの停止を確認し、最新の原checkpoint・SOURCE・ログを保存する。既存input・C0のenergy/alpha/beta/δ/r/K・32 sample/seed・Qiskit1.3.0/opt1/seed7・no-prep境界を維持し、新directoryで未完了pointだけを並列計測する。最初のC0不適格で停止し、全point通過時だけ既存accountingへ集約する。
3. 許可された検証・公開照合の完了後はmandatory STOPする。並列化を理由に別条件・新fit・追加samplingへ進まない。

共通AGENTSの「正常な計算を自ら中断しない」という指示を守り、現在は旧jobを止めていない。今回の質問を計画停止・未完了point再計測の許可とは扱わない。切替許可がない間は元のC0 jobが従来SOURCEで継続する。
