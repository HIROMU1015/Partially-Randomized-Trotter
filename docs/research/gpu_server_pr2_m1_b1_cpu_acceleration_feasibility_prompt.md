# GPUサーバー側Codexへの確認依頼：PR-2 M1-B1 CPU高速化可能性のみ

## 依頼

PR-2 matched-accuracy resource studyのM1-B1は、ローカル環境で認可済みの一回のrunを6 CPU workerで
実行中です。しかしfull-wrapperのQiskit transpileが想定より長く、GPUサーバー側のCPUを使う方が
wall timeを短縮できる可能性があります。

今回は、**GPUサーバーへM1-B1を移す価値があるかを判断するためのzero-science性能確認だけ**を行って
ください。M1-B1科学計算、実candidateのcircuit build/compile、development NPZのload、held-out access、
runtime移送、現在のrunの停止は認可しません。

GPUサーバーという名称ですが、対象処理はQiskit transpilerのCPU処理です。GPUをquery、allocate、使用せず、
`nvidia-smi`、CuPy import、GPU kernelも実行しないでください。

## 固定identity

- repository：`HIROMU1015/Partially-Randomized-Trotter`
- branch：`pr2-v4-s2-parallelization-20260928`
- review済みHEAD：`dea9511f059330b8408c3adc952ec10a009c9fc1`
- M1-A result commit：`3c1831e326c27c5f679b3820997f27916d26ed9f`
- actual execution source commit：`33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62`
- authorization bundle commit：`8fc24000b49b6cdb146c3f084891a2f42898214f`
- execution plan v2 SHA-256：
  `5afc94fac0571b38c74b0b00cfcf68e34e491a5fc65ef579d3ec884d079e5aa5`
- execution plan fingerprint：
  `17c91d41e77d7c085629b60ea470abc87e9590f448ba9bcdfa4342410cd89607`
- execution authorization JSON SHA-256：
  `7d188f782354609fec1ea83e289e9872c6a801c3e0e9493f93753fe7cae9d93a`
- result schema v2 SHA-256：
  `035d9b1e48d8d02718f8c7218ea49d28297b3b3f29c39c4232cdef83c7c1d081`

固定compiler identityは次です。

```json
{
  "backend_name": null,
  "basis_gates": ["rz", "sx", "x", "cx"],
  "coupling_map": null,
  "layout_method": null,
  "optimization_level": 1,
  "qiskit_version": "1.3.0",
  "routing_method": null,
  "transpiler_seed": 17
}
```

source commitに固定された主要source SHA-256は次です。

| path | SHA-256 |
|---|---|
| `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py` | `3523a2fc27c3a39e6db46cf70fb985dbc93ee8b9028bddca2bff2a013adb643e` |
| `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py` | `b7a3a479cbebdd3ff7a93de75c4ba030343769c308401395e208754c3d566987` |
| `scripts/run_pr2_matched_accuracy_m1_b1.py` | `07f34e4f1a22e93ace21d26ce122b7931311f8d94dfebd2d79f1086d9fdc9371` |
| `tests/test_pr2_matched_accuracy_m1_b1_execution.py` | `971caf4c9b5025bfae6779ff15a53656fecebc068152192a72b1cea8555dc712` |

## ローカル実測baseline

ローカルrunは次の固定条件です。

- 32 logical CPU、RAM 62 GiB。
- 6 spawned worker、各BLAS thread 1。
- random 194 cell × 32 trajectory × 2 axes = 12,416 full wrappers。
- baseline 16 cell × 2 axes = 32 full wrappers。
- 合計12,448 full wrappers。
- 1 random cellは64 full-wrapper transpileを行う。重複transpileは確認されていない。
- 2026-09-30の観測時点で、約31分に31/210 cellが完了。
- 6 workerは各約100% CPU、worker RSSは約0.85--1.18 GiB。
- 完了cellの実測例：

| 条件 | 平均compiled circuit size | cell wall time |
|---|---:|---:|
| `q=1, r=8` | 約13,000 | 約2.5分 |
| `q=2, r=1` | 約22,500 | 約4.3分 |
| `q=2, r=32` | 約35,000--36,000 | 約7.3--7.7分 |
| `q=4, r=1` | 約44,400 | 約8.7分 |
| `q=4, r=16` | 約57,100 | 約11.9分 |

固定gridには`q=8`が49 random cell残るため、ローカル6-worker runの総所要は現時点で約5--7時間と
推定しています。この値は正式結果ではなく、サーバー移行価値を判断するための運用baselineです。

## 許可する確認

### 1. clean sourceと環境identityの確認

1. `git fetch origin --prune`後、上記review済みHEADから独立したclean worktreeを作る。同名worktreeやbranchを
   上書きしない。
2. 上記commit系譜、source SHA-256、plan/authorization/schema SHA-256を照合する。
3. 次を読み取り専用で記録する。
   - CPU model、socket、physical core、logical CPU、NUMA構成。
   - RAM、現在利用可能なmemory、swap。
   - OS、kernel、Python、Qiskit、Rust accelerator等のversion。
   - CPU loadと、他jobを妨げず利用できるcore数。
   - repository/outputを置くfilesystemの種類と空き容量。
4. package、Python、Qiskit、CUDA、driver、shell設定を変更しない。互換Pythonがなければそこで
   `ENVIRONMENT_NOT_COMPARABLE`として停止する。

development snapshotがGit treeに存在するかは、次のpathを`git ls-tree`で確認してよいですが、fileをload、
hash、stat、importしてはいけません。

```text
artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
```

held-outを示すpathの探索、resolve、stat、hash、loadは行わないでください。

### 2. zero-science synthetic transpile benchmark

環境identityが一致する場合だけ、repository外の新規一時directoryでsynthetic benchmarkを実行してよいです。

- 実M1-B1 runner/moduleを`run`しない。
- development/held-out NPZ、M1-A candidate record、固定trajectory seed、現在の`.runtime`を読まない。
- 実candidateの回路をbuild、sample、compileしない。
- repositoryのtestに既存の純synthetic circuit fixtureがあればそれを優先して再利用する。
- fixtureがなければ、9 qubit、1 classical bitの完全synthetic回路を固定seedで作り、実測baselineに近い
  おおよそ12k、45k、90k instructionの3規模だけを用いる。研究Hamiltonian、DF factor、candidate、RTE
  distributionを再構成しない。
- transpile条件は上記compiler identityと一致させる。
- 同一の固定synthetic task setを、1 worker、6 worker、12 worker、16 workerで比較する。ただしphysical core、
  memory、共有server負荷から危険なworker数は実行せず、その理由を記録する。
- 全processで次を置換設定する。

```text
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

- synthetic transpileは全条件合計128件以下、総wall time 30分以下とする。
- 各worker条件についてwall time、tasks/min、speedup、parallel efficiency、peak RSS、failure/OOMの有無を記録する。
- benchmark outputは一時directoryだけへ置き、repositoryへcommit/pushしない。

### 3. 移行可能性の見積り

実M1-B1を実行せず、次だけを報告してください。

1. 同一環境・6 workerでもローカルより速いか。
2. 安全な推奨worker数は6、12、16のどれか。必要ならそれ未満を提示する。
3. 12,448 full wrappersの推定wall time。楽観値だけでなく保守値も示す。
4. ローカル継続に対して、server移行準備時間を含めても何時間短縮できるか。
5. scientific source、candidate、seed、compilerを変えずに移行できるか。
6. 現在のruntime/checkpointを移さずfresh runにする場合と、将来別reviewでcheckpoint migrationを認可する場合を
   分けて評価する。今回はどちらも実行しない。

## 禁止事項

- M1-B1 science runnerの`run`または`resume`。
- actual 194 randomまたは16 baseline cellのtrajectory sampling、circuit build、compile。
- development NPZのload、hash、stat。held-outはpath探索も含め全面禁止。
- 現在のローカルprocessの停止、signal送信、priority/affinity変更。
- ローカル`.runtime`、checkpoint、SQLite cacheのcopy、move、rsync、scp、削除、変更。
- worker上限、source、plan、authorization、result schemaの変更。
- GPU query、allocation、kernel、CuPy import、`nvidia-smi`。
- package/environmentのinstallまたはupgrade/downgrade。
- repositoryへのcommit/push、mainへのmerge、force-push。
- M1-B1結果や研究四分岐の推測・決定。

## 判定

次のいずれか一つを先頭に示してください。

- `SERVER_CPU_ACCELERATION_RECOMMENDED`
  - compiler/environment identityが一致し、保守見積りでも移行準備を含めてローカル残時間を40%以上短縮できる。
- `NO_MATERIAL_SERVER_ADVANTAGE`
  - 実行可能だが、保守見積りで40%未満の短縮しか得られない。
- `ENVIRONMENT_NOT_COMPARABLE`
  - Python/Qiskit/compiler identityを変更せず比較できない。
- `BLOCKED_BY_IDENTITY_OR_RESOURCE`
  - source identity、利用可能CPU/RAM、共有server制約のいずれかで安全な比較ができない。

## 最終報告

次を簡潔に示してください。

- 使用したrepository HEADとsource/hash照合結果。
- CPU model、physical/logical core、NUMA、RAM、利用可能memory、既存load。
- Python/Qiskit/compiler identityの一致・不一致。
- synthetic fixtureの生成規則とfingerprint。科学データを0件読んだこと。
- 1/6/12/16 workerのwall time、speedup、parallel efficiency、peak RSS。
- 推奨worker数と12,448 wrapperの楽観・保守wall time。
- 移行準備込みの見込み短縮時間。
- GPU query/use 0、development/held-out access 0、science wrapper 0、repository変更0。
- 上記四分岐statusと根拠。

報告後に停止してください。M1-B1本実行、現run停止、runtime移行、authorization amendment作成には進まないで
ください。
