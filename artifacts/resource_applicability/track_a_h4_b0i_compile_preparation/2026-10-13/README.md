# H4 B0+I：検証用実装と6 compile準備

**PREPARATION ONLY。実H4 compile/transpile/signal/trajectoryは0。B0+Iは未採択の別対照。**

[資料入口](../../../../docs/research/track_a_h4_b0i_compile_preparation_20261013.md)。
SOURCE `8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b`、起点 `f6b0cb5e2ddabd2d7919d0822c58324c7338bac0`。
元科学SOURCE `d0726e946f89f8c30a08d383fb8ff48b363629ec` のB0/science系列を変更しない。
旧prototype SOURCE `c84671a09d13b086c0a38b539295bd13778eba3d` とprivate proposal01は履歴として保全するが、今回の承認対象ではない。

## Bundleの対応

|ファイル|内容|
|---|---|
|`registry.json`|.70 p5/q1、.80 p5/q1、1.40 p3/q1、cos/sin各1、計6 task。新method/template/source/input/phaseのID|
|`plan.json`|固定SOURCE、48 source closure、input/DF/stateのbyte metadata、保存signal/target/bias、compiler/profile、独立output/control/lock、carry、caps|
|`authorization_draft.json`|新schemaの未承認草案。旧grantを受け付けない|
|`review_draft.json`|未承認review、plan/auth digest binding、科学採否・launch未受領|
|`binding.json`|technical_bound=true、runtime_sealed=false、3 digestとactual SOURCE|
|`source_closure.json`|48 Python source/tests/entrypointsのbytes/SHA。元H4 Python38件は不変|
|`frozen_array_identities_v1.json`|検証済みgeneration-freeze metadataからH/DF/state/one/nuclearのdtype・shape・array payload byte SHA。再decodeなし|
|`input_byte_audit.json`|対象NPZ3件、計41160798 bytesのstreaming SHA一致|
|`profile_audit.json`|既存environment6abf53…、compiler fbe36b…に一致。旧host/compiler output bit同一性は未検証|
|`host_readonly.json`|CPU1秒の受動sample、effective memory/PSI/OOM baseline、filesystem/block/inode、quota GETだけ|
|`artificial_tests_v1.json` / `.log`|44限定人工/mock PASS、実入力decode・transpile・process spawn・GPU/molecular import0|
|`preparation_summary_v1.json`|旧B0/source保全、実施/未実施、人工初回のguard拒否と修正履歴|
|`EXECUTION_CONTRACT.md`|承認後のfresh admission、dispatch、own-run STOP・wait/reap・FD/pipe終了、正常終端の定義|
|`BUDGET_AND_SCOPE.md`|6/10/240案、capと見積り/未知の区別|
|`artifact_manifest_v1.json`|新bundleとsource/doc/indexのbytes/SHA、manifest自身を除く|
|`verify_preparation.py`|公開metadataだけのsource/task/profile/flags/digest照合。入力や回路を開かない|

入力DF/stateのSHAには二種類ある。NPZ全fileのbyte SHAは今回streaming確認し、
array payload byte SHAは検証済みfreeze JSONから転記してcanonical H/DF/state/input identityへ照合した。
NPY memberのheader込みSHAとは区別する。H/state科学arrayやNPZ科学arrayのdecodeは0。

## 未承認状態

plan/authは `approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、
`absolute_launch_command=null`、`retry=false`、`automatic_scope_expansion=false`、
`next_stage_authorized=false`、`mandatory_stop=true`。
実回路のnumeric fingerprintはnull／DEFERRED。未取得値を人工回路の値で代用しない。
実回路の承認済みbuild時に旧canonical encoderで計算し、新SOURCE/input/template/phase/control/axisへ結合する。
SOURCE/プロフィール/入力/保存値/出力/carryの技術bindingと、CPU・実行権・最終reviewのsealを区別する。

## 人工検査の再現

1 process、thread1、AS2GiB、CPU120秒、wall180秒。
実/人工transpile、NPZ/NPY読込、subprocess、molecular/CuPy importをguardで禁止する。
限定3-qubitのOperator/Statevectorは人工fixtureだけで、実H4のsignalではない。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 \
  QISKIT_PARALLEL=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR=/home/AbeHiromu/projects/h4-handoff-evidence/20261013/h4-b0i-compile-preparation-20261013/tmp \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python \
  -P -B scripts/resource_applicability/run_h4_b0i_preparation_tests.py \
  --output /home/AbeHiromu/h4-b0i-artificial-reproduction-new
```

metadataだけの照合（private input/venvは不要、SOURCE Git objectが必要）：

```bash
python3 -P -B artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/verify_preparation.py \
  --repo "$PWD" --bundle artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13
```

`prepare_h4_b0i_pilot6.py` はbyte/profile/hostを再観測した草案の生成器。
freeze metadataの転記・技術bindingを含む承認対象はこの固定bundleのplan/registryである。
再生成した草案で固定承認対象を無条件に置き換えない。
future runner `run_h4_b0i_pilot6.py` は**今回は実行しない**。現在のdraftでは最初のauth gateで拒否する。

## 次の科学的判断

B0+I採否、元登録集合に限定するclaim、6件実測の必要性、実測後の10/240案への別判断はGPTへ戻す。
6件だけで397点全体や補助120構成のwinnerを確定しない。
H8 source/worktree/grant/計算、共有環境/private venvへ変更0。
公開可能ならnon-force push・remote byte照合、認証不可ならlocal commitと手動pushを報告してSTOP。
