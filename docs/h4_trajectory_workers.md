# H4 PF/PR：trajectory単位の複数worker

2026-10-23 JST。利用者の「複数ワーカで処理できるようにして」「使えそうな既存スクリプトを流用」に基づく技術実装。専用branch `codex/h4-trajectory-workers-20261023`、基点`0c329b30a81accb35c212be8defa66fee0dd0f12`で実装・人工検証する。並行研究側のH4機能として保持し、本チャット本来の研究概要・発表資料へ数値を統合しない。

## 流用と変更範囲

実行管理は既存`isolated_qiskit_supervisor.py`の`run_supervised`/`worker_ready`を直接呼ぶ。fresh interpreter、ready/release、排他的claim、stdout/stderr、pidfd/subreaperによる子孫回収、失敗時のSTOP・部分結果保全・自動retryなしを流用した。既存supervisor自体は変更していない。

`scripts/pr_cost_candidate_parallel.py`/`scripts/h8_key_compile_parallel.py`も確認した。候補固有のcontrol・128task・fork親・CPU affinity条件に結合した経路なので、H4の一般trajectory実行には、そのまま接続していない。`pr2_v4_s2_parallel_execution.py`のordinal順集約も参照した。歴史的なworker上限は引き継がない。

| ファイル | 役割 |
|---|---|
| `src/trotterlib/h4_trajectory_workers.py` | registry分配、既存supervisor呼出し、認証、index順集約 |
| `src/trotterlib/h4_trajectory_science.py` | 供給済み入力の復元、1trajectoryの実sampler→回路→両軸価格・確率検証 |
| `scripts/run_h4_trajectory_worker.py` | 独立interpreterの入口、process内だけのnested thread設定 |
| `h4_audited_evaluation.py` | opt-in executor追加。既定直列経路の演算・数値会計を保持 |
| `h4_pilot_science.py` / `h4_pilot_execution.py` | PR caseへの接続、固定入力/予算/契約設定の検証 |
| `h4_execution_contract.py` / CLI | worker数をSOURCE・契約・新grantへ結合、CLI `--trajectory-workers` |

旧H2 kernel・Pauli/DF sampler・積公式・既存supervisor・未閾値積分変換は変更しない。実行中の旧H4 worktreeのSOURCE・grant・snapshot・raw・STOPも変更していない。このbranchの新APIは、旧grantを別worktreeから引き継いだり、旧runの途中へ挿入したりする機能ではない。

## 分配・同値性

元の`SeedSequence(config.seed).spawn(count)`全体を先に作り、元のindex/seedを固定する。worker kには`tasks[k::workers]`を割り当てる。重複・欠測・seed改変・未割当taskは拒否する。worker内でも元registryを再構成して照合するが、production samplerの再抽選はしない。

各workerが担当trajectoryのproduction samplerを一度呼び、実戻りevent objectを回路生成前に保存し、同じ順序で回路化する。cosine/sineは同じtrajectoryの両軸であり、別seedへ分割しない。先頭3trajectoryの独立operator照合はglobal index基準で維持し、order0/2/4の代数検査は親caseで一度だけ行う。全sampleのcompile前後確率検査も維持する。

結果はworkerの完了順ではなく元index順に並べて、元のreference/raw/corrected/bias/normalization/Hoeffding会計/統計関数を使う。workerごとのprivate cacheによりcache hit数はworker数で変わり得るが、標本を減らさず、価格・C/N/Gの一致を検証する。

QiskitがPauli controlled gateへ付ける自動名`circuit-N`は、process内の回路生成履歴に依存する。既存fingerprintはその名前も含むため、演算・event・価格が同じでもhashが変わる場合があった。この新経路の直列/並列双方でevolutionの**名前だけ**を`h4_fixed_evolution`へ固定した。gate順・角・global phaseは変更しない。旧版の保存raw hashは書き換えない。serial/parallelで実event・wrapper fingerprint・価格・正のC/N/Gまで一致する人工試験を行う。

## worker数と契約

`--trajectory-workers 8`などの正の整数、または`auto`を指定できる。既定`auto`はaffinityで利用可能なCPU数を同時のPR2caseで分け、trajectory数以下のworkerを選ぶ。小さいCPU値で恒久制限しない。各workerは独立Qiskit processで、process内のBLAS/OpenMP/MKL/Rayon threadを1にし、共有mutable compilerをthread共有しない。OS・共有venv・他processの設定は変更しない。

worker設定は未承認契約candidate v2に含める。SOURCE・契約digest・承認・namespace・新grantを一致させ、input receipt/snapshot、Hamiltonian/state、compiler、固定call数とseed registryを照合する。既存実行済みgrantでの再実行・run ID変更によるcase再投入は行わない。

契約表示（科学実行なし）：

```bash
PYTHONPATH=src python -B scripts/run_h4_non_df_pilot.py --contract --trajectory-workers 8
```

実分子での利用は、v2候補と新SOURCEに結合する新しい一括実行承認・新namespace・新grantで行う。今回の実装要求を、旧runの科学retryや新しい分子runの自動実行に拡張していない。人工実行は`--synthetic --trajectory-workers 2 --pack <未使用の絶対path>`で内部fixtureだけを使う。

## 証拠と限界

[技術証拠](../artifacts/h4_trajectory_workers/2026-10-23/README.md)、[完成候補](../artifacts/h4_trajectory_workers/2026-10-23/H4_execution_contract_DRAFT.json)、[検証集計](../artifacts/h4_trajectory_workers/2026-10-23/verification.json)。人工1/2workerで両表現の同値性、無効grant/seed/設定変更拒否、compile失敗後のpartial保全・回収、既存評価と監視の回帰を検査する。人工8 system/9 wrapper qubitの入力→4case→PR各2worker→20精度行も確認する。

人工8量子ビットの20行は旧allowance条件によりN/G nullのまま。別の小さい人工系で正のC/N/G一致を確認する。人工系の処理時間から実H4のspeedupは保証しない。実分子での並列性能・workerごとのmemory最適値・希少event母平均は未確認。

現在の旧科学runは旧SOURCEのまま継続し、既存の保存検証・要約・ローカルcommit後処理も維持する。新コードを実行中のSOURCEへ上書きしない。今回の次担当はCodexによる技術証拠固定・local commitまで。新科学実行・push・詳細科学レビューへ自動進行しない。
