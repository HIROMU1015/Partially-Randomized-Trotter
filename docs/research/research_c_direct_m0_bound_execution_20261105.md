# 研究C：固定15-call・Direct m0検証のexact symmetry STOP

2026-11-05 JST。**固定契約を一度実行し、3 attempts／2 COMPLETE／1 FAILでmandatory STOP。保存sector Hがexact real-symmetricでないことを検出した。Oracle距離とX/Y信号包囲は未取得。Retry/resume=0。**

Branch：`codex/research-c-direct-m0-bound-execution-20261105`。
独立worktree：`/tmp/prt-c-direct-m0-bound-execution-20261105`。
実行起点は公開確認済み `2c0f52632316be8bbf39d9e40b138d0fd669f860`。
Scientific SOURCEは **`098e1bac4a9aa0bd6d1c3c61d19a38a1ac47bc6d`**、tree `ee308151f12b4427dd4ea620c3d0efa8b78e3884`。

## 固定条件と実行前確認

[固定契約](research_c_direct_m0_bound_contract_20261105.json) SHA256：`b727896a81d6404f2ec15a9496b4bad7ed230d9ff07a71d81a2460b5cc3aa286`。
SOURCE manifest SHA256：`592766ffba761c1cb52d011d0a56e5e4bc080b13f90e318ca40387895a8bc8b1`、25 source file・16 pinned input/provenance file。
数値SOURCE、固定契約、入力、gate/phase/register、Arb192、stop条件を変更しなかった。

対象は保存直鎖H4・隣接1.00 Å/STO-3G、number CDF8・元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1。
Nα=Nβ=2／sector36、system256／ancilla-last controlled512、own-DF numerical ground reference・状態準備費用除外。
Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
既存P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、Gaussian/relative fusion、MPFR160 declared phaseを保持する。
今回はm0・1 occurrence、F6:0／F7:6,7交互16回の保存32-event wordだけ。

実行許可を固定15-call範囲へ適用し、承認証明・承認原文保存・新grantを作らなかった。
重要科学レビュー、fresh sampling、新native build/compile、旧Stage P/7-call replay、期待bias校正、RPE shots/energy、Stage C、actual CAは対象外。

[Preflight](../../artifacts/research_c_direct_m0_bound_execution/2026-11-05/execution_preflight.json)で公開SOURCE SHA/treeと契約本文の一致、SOURCE/input hash、fresh不在、資源を確認した。
実行前の人工・回帰は[97 PASS](../../artifacts/research_c_direct_m0_bound_execution/2026-11-05/preexecution_tests.txt)。科学結果やCIを意味しない。
Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、python-flint0.8.0/FLINT3.3.1、FE_TONEARESTという固定runtimeを使用した。

## 実行と停止位置

無変更の`run_research_c_direct_m0_bound.py --execute-science`をforeground childとして一度起動した。
Fresh科学出力：[science-run01](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01)。
終了・標準出力：[launch01](../../artifacts/research_c_direct_m0_bound_execution/2026-11-05/launch01)。

| call | 状態 |
|---|---|
| metadata_load | COMPLETE |
| arrays_load | COMPLETE |
| target_identity | **FAIL** |
| shell_decode〜summaryの残12 units | **NOT_RUN** |

[Journal](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01/science_calls.jsonl)はattempt3、COMPLETE2、FAIL1。
[STOP](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01/STOP.json)は失敗kind=`target_identity`、error=`EXACT_SYMMETRY_REQUIRED_NO_HERMITIANIZATION`、retry/resume0、mandatory STOPを記録する。
[partial_failure](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01/partial_failure.json)と[worker traceback](../../artifacts/research_c_direct_m0_bound_execution/2026-11-05/launch01/worker.log)を保存した。

停止は`c_direct_m0_runner.identity`から`c_direct_m0_oracle.symmetric_matrix`を呼ぶexact equality gate。
保存Hをexact real値へ解釈した後、H[i,j]=H[j,i]が成立せずFAILした。
これは固定契約の前提不成立であり、プロセス消失・compiler失敗・X/Yの数値誤差判定ではない。
Child returncodeは1で、終了signalやsupervisor例外はない。

## 得られたこと・未取得のもの

今回確認した事実は、**同じ保存numerical oracle入力でも、保存H全体をそのままexact real-symmetric Hamiltonianと見なす条件は成立しない**ということ。
Producer/pathの対応だけでは、このexact条件を代替できない。

非対称部分の大きさ、原因となる演算、signalへの影響は評価していない。
小さな丸め差だと仮定せず、旧数値pilotの科学的結論が覆ったとも判断しない。
旧tolerance付きground計算と、今回のexact equality条件は区別する。
Signal評価法一般の不可能性や、人工Arb評価器の不正確さを示す結果でもない。

Shell QPY decode、数値固有値候補生成、exact inertia、ground-distance certificate、sector embedding、state binding、literal block作用、head/forward/inverse state作用、X/Y readout、summaryは未実行。
`identity.json`、oracle certificate、X/Y、`result.json`は存在しない。
未取得のboundを0にせず、X/Yはnull／未資格として検証記録に記載した。
Full512 operator保証、期待bias、全law、RPE accuracy、actual CAは認定しない。

保存Hの対称化、片三角completion、別targetへの置換、誤差許容の追加、再計算、別namespaceでの再実行は行っていない。
未実行12 unitsは転用しない。このrunは停止・消費済みで、元の`execution_ready`を新しい実行許可として再利用しない。

## 資源・保全・技術照合

開始時CPU affinity32、空きRAM約52 GiB、空きdisk約5.8 GB。CPU/memory pressureは低く、GPU driverは利用不能だった。
支配するnative state-actionは既存FLINT局所blockと公開済みinverse共通unitary削減を使う計画だったが、そこへ達しなかった。
小sector固有値計算用のBLAS/OpenMP1 threadは自jobだけに設定し、CPU affinity・共有環境・他jobは変更していない。
CPU/RAM/worker/wall capは新設しなかった。

Runner計測wall 0.070489秒、CPU 0.070367秒、process peak RSS124300 KiB。
外側child wall 0.393065秒、reaped child CPU 0.392707秒。
これらはgate到達までの費用で、Direct信号評価の速度や実行可能性の実測ではない。

[実行照合](../../artifacts/research_c_direct_m0_bound_execution/2026-11-05/execution_verification.json)で25 SOURCE／16 inputのhash、契約/SOURCEコピー、3/2/1 call記録、後続出力不在、retry/resume0を確認した。
科学runの7 raw fileをそのまま保存した。後処理で保存Hを再度decodeしたり、数値原因解析を追加していない。
既存の数学module、元契約、科学SOURCE、旧raw・grant・ledger・marker・STOPは変更していない。
Source/入力をテストや失敗に合わせて修正しなかった。

今回のevidence statusを研究概要、VALIDATION_STATUS、dated note、manifestへ追記した。既存の記録は原条件の履歴として保持する。
公開後はremote SHA/tree、報告本文、主要raw/SOURCE/STOP blobを照合する。

## 次担当とmandatory STOP

次は**ユーザー／研究C**が、own-DF oracleに使うHermitian targetと、保存Hとの差をどう会計するかを判断する。
**Codex**はその具体的依頼後に、必要なtarget/ground perturbation bridgeの技術準備を行う。
新たな数値診断・再評価・再実行と重要科学レビューは、それぞれ適用する明示許可が必要。
今回の失敗を回避するために契約を緩和したり、再実行を開始したりしない。

**実行は登録gateで停止した。部分結果の保全・公開照合後MANDATORY STOP。**
