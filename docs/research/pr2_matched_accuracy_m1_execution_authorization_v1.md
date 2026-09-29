# PR-2 matched-accuracy M1-A execution authorization v1

作成日：2026-09-30
基準commit：`86f524fd0d80650dca8ef4742f25cb056b691a85`
status：`M1_A_EXECUTION_AUTHORIZED_M1_B_REQUIRES_CLEAR_FROZEN_RESULT`

## 1. 許可する一件

保存済みdevelopment H4 linear 1.00 Å、STO-3G、DF rank 12、sector 8 qubitだけを一回読み、
凍結済みM1候補についてM1-Aを一度実行する。M1-Aはsignal、bias、finite-RTE normalization、
解析shot、action proxy、r64 boundary request、16-cell selector、hard precompile barrierまでで停止する。

base候補は208、r64追加は最大4、signal評価は最大212である。物理時間は`T=0.8`、
`q={1,2,4,8}`、`delta=T/q`、random gridは`r={1,2,4,8,16,32}`、`K={2,4}`とする。
M1-AではQiskit circuit object、trajectory列、compile task、transpile、full wrapper、量子shotを一件も
生成しない。small-system dense actionはsignal参照だけに使い、compile costとは呼ばない。

## 2. 入力identity

development入力は次の一件だけである。

- path：`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`
- raw SHA-256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- state hash：`31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd`
- state-vector hash：`c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63`

development raw hashは一回照合し、NPZは一回だけloadする。分子計算、入力再生成、別snapshotへの
fallbackは禁止する。

## 3. held-outと禁止事項

H4 1.30 Å held-outについて、relative/absolute pathの解決、`stat`、存在確認、hash、NPZ load、signal、
cost、rankingを全て0とする。sourceはheld-out path定数をimportせず、M1-A artifactには未開封のboolean
監査だけを保存する。

新geometry、LiF、別分子、別basis、別PF、H12、長RPE、backend/noise、状態準備回路、最終総cost、
S3、量子shotを許可しない。旧S0 STOP、旧S2結果、rank 3/9 control、B2/B3 frontierを変更しない。

## 4. M1-A数値経路とcorrectness gate

M1-A sourceはQiskit circuitを作らず、保存Hamiltonianから8-qubit dense参照を構成する。constant、
one-body、各DF fragmentの再構成、各splitの`H_D+H_R=H`、Rayleigh residual、Hermiticity、
raw/corrected relation、normalizationの直接積/log積、`q*delta=T`を検査する。

deterministic/discardはsymmetric second-order outer action、random候補は同じouter orderとfinite Taylor
momentを用いる。候補fingerprintは凍結済み台帳と同一とし、selector入力の`n_rand`はpaired finite-RTE
分布の期待application数へ`ceil`を一度だけ適用する整数policyで固定する。結果後にproxy、threshold、
candidate、boundary、16-cell capを変更しない。

## 5. hard barrierと停止

selector後の分岐は次だけである。

```text
selection_limited = true
  -> status = SELECTION_LIMITED
  -> compile job/circuit/compile/full wrapper = 0
  -> resultをcommit・pushしてmandatory STOP

selection_limited = false
  -> status = M1_A_COMPLETE_M1_B_ELIGIBLE
  -> M1-A artifactだけをcommitしてbyte identityを固定
  -> selected fingerprintを変更せず、別のM1-B source/authorization freeze後だけcompile可能
```

本authorizationはM1-B direct compileを先取りして許可しない。clear時にM1-Bへ進む場合も、M1-Aの
signal値を使ってselector規則を変更せず、compile結果を見る前にsource、最大6 process、32+96 trajectory、
最大2,048 trajectory、最大4,128 wrapper、output、再開規則を別commitへ固定する。

## 6. process、出力、再開

M1-Aは単一process、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`で実行する。
GPU query/allocation/kernelは0である。出力は次の一件とする。

`artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json`

既存fileを上書きしない。失敗時は同一source/authorization/output identityで、outputが未作成の場合だけ
再実行できる。別outputでthresholdや候補を変更した救済は禁止する。

## 7. test gate

科学実行前に次をpassさせる。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/abe/Project/Partially\ Randomized\ Trotter/.venv311/bin/python -m pytest -q \
  tests/test_pr2_matched_accuracy_m1_execution.py \
  tests/test_pr2_matched_accuracy_m1_contract.py \
  tests/test_pr2_matched_accuracy_m1_precompile_barrier.py \
  -p no:cacheprovider
```

実行後に同じfocused testsを再実行する。result JSONはreserved M1-A schema、source hash、authorization
hash、candidate/record数、全counter、result fingerprintを照合してからcommit・pushする。
