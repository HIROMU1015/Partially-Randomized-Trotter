# 共有サーバー向け並列検証実行基盤

## 位置づけ

`src/trotterlib/parallel_validation_executor.py` は、既存の検証関数を小さな独立taskとして
bounded subprocessで実行するexecution layerである。科学ロジック、誤差判定、優位性判定は
この層に置かない。生成されるcheckpointとaggregateはraw compute artifactであり、研究上の
検証通過や最終総costの結論を表さない。

Qiskit transpileはCPU taskとして扱う。GPUは既存のstatevector/state-action kernelを呼ぶtaskに
限り、明示されたGPUへ1 jobずつ割り当てる。1 statevectorの複数GPU分散は実装していない。

## ファイル

- `src/trotterlib/parallel_validation_executor.py`: schema、hash、資源制限、worker、resume、集約
- `scripts/run_parallel_validation_batch.py`: manifest生成、dry-run、実行、resume、status
- `tests/test_parallel_validation_executor.py`: unit testとH4 q=1 smoke
- `docs/examples/parallel_validation_h4_q1_manifest.json`: 計算を起動しないdry-run例

## task manifest

schemaは`parallel_validation_task_manifest_v1`である。taskの基本単位は
`(validation_id, ld, delta, r, k, q, trajectory_index)`で、seed、compiler設定、Qiskit version、
入力artifact hash、source hash、adapter parameterもtask IDへ含まれる。manifest内のtask順は
ID順へ正規化される。

```json
{
  "schema_version": "parallel_validation_task_manifest_v1",
  "batch_id": "example-batch",
  "source_paths": [],
  "tasks": [
    {
      "validation_id": "example",
      "adapter": "rpe_hadamard_full_wrapper",
      "resource": "cpu",
      "ld": 3,
      "delta": 0.1,
      "r": 4,
      "k": 2,
      "q": 1,
      "trajectory_index": 0,
      "seed": 20260923,
      "estimated_memory_gib": 1.0,
      "compiler_settings": {
        "basis_gates": ["rz", "sx", "x", "cx"],
        "backend_name": null,
        "coupling_map": null,
        "optimization_level": 1,
        "layout_method": null,
        "routing_method": null,
        "transpiler_seed": 17
      },
      "input_paths": ["artifacts/path/to/snapshot.npz"],
      "source_paths": [],
      "parameters": {
        "snapshot_path": "artifacts/path/to/snapshot.npz",
        "partition": "calibration"
      }
    }
  ]
}
```

`task_id`は通常manifestへ手書きしない。読み込み時に内容から生成される。指定した場合は計算値と
完全一致しなければ拒否される。入力・source pathはリポジトリ内の実在fileに限られる。

同じmaster seedでも`trajectory_index`ごとに独立したseedをSHA-256から導出する。full-wrapper
adapterは、そのseedで既存のMonte Carlo benchmarkを`sample_count=1`として呼び、同じ軌道の
cosine/sine wrapperを同一task内でtranspileする。

## manifestの生成

Hadamard full-wrapper用manifestはworkerを起動せずに生成できる。既存fileは上書きしない。

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --create-hadamard-manifest artifacts/parallel_validation_manifests/m06_ld8.json \
  --batch-id m06-ld8-full-wrapper \
  --validation-id m06-ld8-full-wrapper \
  --snapshot artifacts/path/to/m06_snapshot.npz \
  --ld 8 \
  --delta 0.02 \
  --r 4 \
  --finite-taylor-order 2 \
  --q-values 1,2,4,8 \
  --trajectory-count 64 \
  --seed 20260923 \
  --estimated-task-memory-gib 2.0
```

この例は将来条件の形を示すだけで、M06/L08の妥当な科学条件や実行済み結果を意味しない。
compilerを変更する場合は、compiler設定objectだけを持つリポジトリ内JSONを
`--compiler-settings-json`で指定する。

## dry-run

最初は必ずdry-runし、task数、worker数、memory、affinity、GPU割当、出力先、hashを確認する。
dry-runは出力directoryを作らず、workerも起動しない。

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --task-manifest docs/examples/parallel_validation_h4_q1_manifest.json \
  --dry-run
```

明示値を確認する例:

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --task-manifest artifacts/parallel_validation_manifests/m06_ld8.json \
  --output-dir artifacts/parallel_validation_execution/m06_ld8_run01 \
  --max-workers 4 \
  --memory-budget-gib 32 \
  --cpu-affinity 24-27 \
  --dry-run
```

## CPU実行と資源制限

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --task-manifest artifacts/parallel_validation_manifests/m06_ld8.json \
  --output-dir artifacts/parallel_validation_execution/m06_ld8_run01 \
  --max-workers 4 \
  --memory-budget-gib 32
```

安全側の既定値は次の通り。

- 自動worker数は`min(task数, 8, floor(検出CPU数/4), memory上限)`
- 総memory budgetは32 GiB
- task memoryはmanifestの`estimated_memory_gib`を使い、最大taskを基準にworker数を制限
- 明示なしでは16 workerを超えない
- 明示値でも検出CPUを全占有する設定は拒否
- `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、
  `NUMEXPR_NUM_THREADS`はworker内だけ`1`
- affinity指定時は各workerを選択CPUの一つへpinするだけで、NUMA/system設定は変更しない
- CPU taskではworkerの`CUDA_VISIBLE_DEVICES`を空にする

16 workerを超える指定には`--allow-more-than-16-workers`が必要だが、memory・CPU capacity検査は
引き続き行われる。共有サーバーでの使用可否を保証するoptionではない。

## checkpoint、resume、status

出力は`artifacts/`以下に限定される。省略時は
`artifacts/parallel_validation_execution/<batch_id>/`である。

```text
<output-dir>/
  aggregate.json
  batch_status.json
  checkpoints/<task-id>.json
  logs/<task-id>.attempt-0001.log
  tasks/<task-id>.attempt-0001.json
  worker_results/<task-id>.attempt-0001.json
```

JSONは同じdirectoryの一時fileへ書き、flush、`fsync`、`os.replace`の順で確定する。attempt logと
worker resultはattemptごとに新規作成し、既存fileを置換しない。親processがcheckpoint確定前に
中断した場合も、resumeは古いattemptを保存したまま次のattempt番号を使う。

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --task-manifest artifacts/parallel_validation_manifests/m06_ld8.json \
  --output-dir artifacts/parallel_validation_execution/m06_ld8_run01 \
  --resume
```

完了taskはskipし、failed、interrupted、checkpoint未作成taskだけを実行する。`--resume`なしで
checkpoint、task spec、logなどのruntime stateが存在する場合は上書きせず拒否する。破損、
fingerprint不一致、manifestにないcheckpointも拒否する。

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --task-manifest artifacts/parallel_validation_manifests/m06_ld8.json \
  --output-dir artifacts/parallel_validation_execution/m06_ld8_run01 \
  --status
```

SIGINT/SIGTERMを受けると新規task投入を止め、executorが記録している子PIDだけへterminateを送り、
猶予後も残る同じPIDだけをkillする。process名検索や他processへの操作は行わない。

## deterministic aggregate

`aggregate.json`はcompleted taskをtask ID順に並べ、metricごとに`math.fsum`で平均と標準誤差を
計算する。fingerprintの対象はtask ID、task fingerprint、数値metricであり、PID、完了順、時刻、
attempt番号を含めない。逐次実行と並列実行で同じtask結果ならaggregate fingerprintも一致する。

各checkpointにはsource hash、input hash、compiler設定、Qiskit version、seedを保存する。
`scientific_verdict_included`は常にfalseである。科学的判定は既存の検証moduleで別artifactとして
作成し、このraw aggregateだけから結論を出さない。

## adapter

- `rpe_hadamard_full_wrapper`: 既存snapshot loader、DF partial-S2 preparation、RTE構成、
  Hadamard benchmark generatorを呼ぶCPU専用adapter
- `python_callable`: `trotterlib.*`または`scripts.*`の既存関数をworker内で呼ぶ薄い境界
- `synthetic`: executor回帰テスト専用

`python_callable`の関数は`function(task: dict, *, attempt: int) -> Mapping`の形にする。manifestは
信頼できるリポジトリ管理下の入力だけを使用する。

GPU taskは`resource: "gpu"`とし、実行時にだけGPU IDを明示する。

```bash
.venv/bin/python scripts/run_parallel_validation_batch.py \
  --task-manifest artifacts/parallel_validation_manifests/gpu_jobs.json \
  --gpu-ids 1,3 \
  --max-workers 2 \
  --dry-run
```

GPU IDごとに同時実行は1 jobまでで、workerだけに`CUDA_VISIBLE_DEVICES=<assigned-id>`を設定する。
worker内から見えるdeviceは通常logical ID `0`なので、既存kernelを包むcallableはその対応を使う。
GPU設定、MIG、clock、power limit、driverは変更しない。今回の実装確認ではGPU jobを実行していない。

## 検証範囲

unit testはfingerprint、atomic write、resume、破損・不一致拒否、決定論的集約、資源guard、
failed-only retry、逐次/並列一致、GPU dispatch環境、SIGINTの子PID限定を確認する。integration smokeは
H4 linear chain、1.0 Angstrom、STO-3G、DF rank 12 snapshot、`L_D=3`、`delta=0.1`、`r=4`、
`K=2`、`q=1`、1 trajectoryのfull-wrapper compileだけである。これは実装回帰であり、新しい
科学結果ではない。
