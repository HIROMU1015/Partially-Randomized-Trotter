# PR-2 M2 transfer 実行authorizationと最終レビュー待ち

作成日：2026-10-04 JST。

本authorizationはH4 linear 1.30 Å、STO-3G、DF rank 12、8 qubits、T=0.8への固定5構成の
one-shot transferだけを結果前に拘束する。今回はauthorizationとレビュー資料のcommit・pushまでで停止する。
最終pre-execution reviewが `APPROVE_M2_EXECUTION` と判断し、利用者から実行指示を受けるまでheld-outを開かない。

machine JSONのstatusは `M2_EXECUTION_AUTHORIZED_ONCE` だが、これは固定条件の定義であって、
最終レビュー済み、または今回launchしてよいことを意味しない。運用statusは
`M2_AUTHORIZATION_FROZEN_AWAITING_FINAL_REVIEW`。独立review待ちは運用上のbarrierであり、
現在のscience runnerがreview承認artifactを機械検査するとは主張しない。

## 結合するidentity

- M1-B1 evidence commit：`8e0814e70c14ecf526444fac8a2142799610dc96`。
- 修正版contract source：`a529e9434d2e62fe752fdab5bd4c9a63fb15e830`。
- 正式contract plan commit：`40888b84b680ca726a603b677296cc97ec3534c4`。
- actual science source：`2978e2fea672b7a1ff20cac74269ec9a610159dc`。
- execution plan commit：`b79c9c1bc392635898254a04a5d5239b55bcce6b`。
- contract plan SHA-256：`d2bb5c5e57002fac5e8045f89a048913f4dadd5177d6f6d1465cc40a8755af7c`。
- execution plan SHA-256：`2aa09a927e5ac58ebe417397802ace0e70c0097e8d0c53c05457075a41e85527`。
- execution plan fingerprint：`ff7ed3d74bf4a0316adecdbac6b633978d71ca87bc1856d03db6fff9a2153276`。
- [machine authorization](../../artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/execution_authorization_v1.json)
  SHA-256：`dbc8b66fad5316004ef404fe8253d3f6cb0f29bf7065502ea995240e5fbd7ff1`。
- result schema v2 SHA-256：`8612a9ba3ebb1a281214ad687b0ca766a51783172bda6cf5a40fca6847c06a36`。

authorizationは128件のsource hashをexecution planからそのまま継承する。actual module/runner/testに加え
全library Pythonをsource commitのblobと照合する。source、plan、schema、compiler、候補、seedを変えない。
authorization commitを文書自身へ自己参照で埋め込まず、後続review依頼とhandoffで固定する。

held-out identityは既存の公開S0 snapshot metadataからの転記である。今回NPZをresolve/stat/hash/loadして
取得した値ではない。file、Hamiltonian、state、state vectorのhashはmachine JSONに固定する。

## 固定する科学計算と資源

| 固定candidate | rank | q | r | K | full wrappers |
|---|---:|---:|---:|---:|---:|
| B2-rank3-q1-r4-K2 | 3 | 1 | 4 | 2 | 64 |
| B2-rank3-q1-r8-K2 | 3 | 1 | 8 | 2 | 64 |
| B0-rank6-q1-r0-K0 | 6 | 1 | 0 | 0 | 2 |
| B1-rank12-q1-r0-K0 | 12 | 1 | 0 | 0 | 2 |
| B3-rank0-q8-r32-K4 | 0 | 8 | 32 | 4 | 64 |

randomは各32 trajectory、Re/Imは同じevolutionとseedを共有し、wrapper keyはaxis別にする。
random192＋baseline4＝196 wrappers、96 random trajectory、snapshot load一回、spawned CPU worker最大5、
各BLAS thread1。実行commandのworkerは5へ固定する。signal/cost candidate fingerprintを一致させ、
step/occurrenceの独立samplingとplanのseedを保持する。

held-outでは固定5構成のsignal/bias/normalization/analytic shotsだけをM1と同じ式で再評価する。
candidate追加・除外・置換やrank/q/r/K/Tの再最適化はしない。accuracy不適格でも構成を置換しない。
Qiskit1.3.0、basis `rz,sx,x,cx`、optimization level1、transpiler seed17、topology/backendなしを維持する。
Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1を既存環境で使い、install/upgradeは行わない。

## 判定とmandatory STOP

[v1契約](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)へ
[usable B2 amendment v2](pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)を適用する。
primaryはshot-weighted compiled RZ、point Paretoは同じ6指標である。

usable B2はaccuracy-eligibleかつprimary重大underestimateなしのB2だけで、
Pareto supportとprimary ratio分子の両方をこの集合に限定する。重大underestimateは
actual primary RZ > 1.10 × (held-out shots × development axis one-shot costs)。
paired Re/Im covarianceを保持した32 trajectoryのdelta-method point±2SEはengineering intervalでありformal CIではない。
状態準備P≥0感度はsecondaryとし、terminal規則を変えない。

terminalは `TRANSFER_SUPPORTED / TRANSFER_NOT_SUPPORTED / TRANSFER_INCONCLUSIVE / IMPLEMENTATION_GATE_FAILED`
のみで、すべて `next_stage_authorized=false` とmandatory STOPを維持する。
SUPPORTEDでもheld-out上のmethod最適性、rank3/q1の一般的最適性、r4/r8の厳密winnerを主張しない。

追加96、別seed救済、H5/H6/H12、別geometry/分子/basis/PF、S3、長RPE、最終総cost、GPU query/allocation/kernel、
CuPy import、nvidia-smi、研究判断の自動実行を禁止する。終了直後に研究方針の全面reviewへ戻す。

## 一回限りの運用と固定command

固定project rootは今回のPR-2 worktreeだけである。

```text
/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928
```

別worktree、別server、別output、別authorizationへの切替で同じrunを追加しない。
固定outputは `artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04`。
今回はこのdirectoryとregistryを作らない。実行時に存在していれば停止する。

実行前review承認と利用者の実行指示を得た後だけ、上記rootをcwdとして次を一回使う。
**この資料準備ではcommandを実行しない。**

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -u \
  scripts/run_pr2_matched_accuracy_m2_transfer.py run \
  --project-root "$PWD" \
  --plan artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/execution_plan_v1.json \
  --authorization artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/execution_authorization_v1.json \
  --output-relative artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04 \
  --workers 5
```

runnerはauthorizationがHEADのcommitted blobと一致することと、source/plan/environmentを科学計算前に検査する。
その後authorization SHAをkeyとするexclusive registryをfixed project root内に作る。
別checkout全体を横断するglobal lockではないため、他場所での再実行禁止は運用契約としても守る。

resume commandはない。失敗・中断・未解決compile reservationがあれば再実行せず停止してreviewへ戻す。
checkpointの完成recordは完全一致したidentityでだけ監査でき、cross-cell流用やruntime移送をしない。
既存資料整理の未commit変更は保持する。無関係なdoc差分を理由にreset/clean/stashしない。

## 実行前後の検査と保存

実行前はsource/plan/authorizationのbyte identity、環境、未使用output/registry、candidate5件、
unique wrapper keys196件、seed96件を照合する。今回もheld-outを触らないzero-science authorization gateだけを使う。
focused84件とhelper regression134件は分子NPZを読まないlocal testsであり、immutable CIではない。

将来の科学実行後は同じ対象testsを再実行し、command/Python identity/時刻/pass/fail/skipを保存する。
runner生成manifestを変更せず、記載resultのbytes/SHAを再照合する。
lightweight result/manifest/completeまたはfailure、test/resource logだけを結果commitに含める。
.runtime、registry、npy、pickle、matrix/vector/state/eigenvectorをcommitしない。
どのstatusでも追加計算へ進まず、次は外部研究方針reviewだけである。
