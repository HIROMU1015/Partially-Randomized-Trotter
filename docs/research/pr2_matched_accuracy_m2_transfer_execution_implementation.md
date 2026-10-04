# PR-2 M2 held-out transfer 実装と実行前の確認

M2の科学実行コードを結果前に実装した。対象はH4 linear 1.30 Å、STO-3G、DF rank 12、8 qubits、
T=0.8への固定5構成のtransferだけである。本資料とコードの追加はheld-out開封や本計算を認可しない。
次はactual execution source commitに結合した別authorizationと、最終pre-execution reviewである。

## 固定した契約とplan

[v1契約](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)は保存し、
[usable B2 amendment v2](pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)を適用した。
修正版契約sourceはcommit `a529e9434d2e62fe752fdab5bd4c9a63fb15e830`、
正式contract planはcommit `40888b8`に保存した。

- 正式plan SHA-256：`d2bb5c5e57002fac5e8045f89a048913f4dadd5177d6f6d1465cc40a8755af7c`。
- 正式plan fingerprint：`7880c8fed57a02f30a07ff7463eb65e423098ad9420e38410f365fad2e24cc4f`。
- 正式plan status：`M2_TRANSFER_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`。
- 旧v1 planと明示的draftは履歴として保持し、実行根拠には使わない。

usable B2はaccuracy-eligibleかつprimary重大underestimateなしのB2である。
Paretoのsupport証人とprimary ratio分子はともにこの集合だけを使う。196-wrapper、各random32 trajectory、
最大5 spawned workers、BLAS各1 thread、全terminal status後STOPは変えない。

## 科学実行コードの役割

- `src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py`はsource/plan/authorization/environmentを照合し、
  認可後だけsnapshotを一度読み、5構成のsignalとfull-wrapper costを評価する。
- `scripts/run_pr2_matched_accuracy_m2_transfer.py`はzero-scienceの`plan`と別認可が必要な`run`を分離する。
- `tests/test_pr2_matched_accuracy_m2_transfer_execution.py`はpure syntheticモデル、保存JSON、mockを使う。
  分子NPZのloadとheld-out pathのopen/stat/resolveをtest fixtureで禁止する。

signal式、finite-RTE normalization、analytic axis shots、DF prefix/discard preparationはM1実装を再利用する。
loaded stateのRayleigh residualを検査し、新しいground-state solveは行わない。5構成の候補fingerprintを
signal/cost経路で一致させる。accuracy不適格でも候補を置換せず、固定構成のcost記録を残す。

wrapperはcontrolled evolutionへcosine/sineの測定を付けたfull wrapperであり、状態準備を含めない。
同じtrajectory request/evolutionを両軸で共有し、axisごとのwrapper keyは分離する。
seedは正式contract planの96件をそのまま使い、outer step内の独立RTE occurrence samplingを既存経路へ渡す。

primaryの標準誤差は、各trajectoryでRe/Imのshot-weighted costを合算してから計算する。
したがってpaired axesのcovarianceを捨てない。candidate間ratioは固定のindependent-candidate delta method
2SE engineering intervalであり、formal confidence intervalではない。10%ちょうどは重大underestimateとしない。

## Sourceと再利用identity

実行planをfreezeする際、actual science module/runner/testとresult schemaに加え、
`src/trotterlib`内の全Python sourceをcommit blobと照合する。transitive helperの変更を見落とさないためである。
Qiskit 1.3.0、basis/optimization/seed等のcompiler identityとPython/NumPy/SciPy identityもplanへ固定する。

execution candidate fingerprintはtransfer configurationと、既存S0公開metadataに記録済みの
held-out file/Hamiltonian/state identityを含む。metadata値は新しいNPZ開封で取得したものではない。
wrapper checkpointはsource commit、candidate、axis、trajectory index/seed、compiler、wrapper semanticsに拘束する。
cross-cell reuse、欠けたcompile結果の推定、別seedへの救済を許可しない。

本実行は一回・snapshot load一回に限定するため、science runnerにresume commandは設けない。
同じauthorizationと固定outputに対するexclusive registryを作り、別outputへの切替や再実行を拒否する。
checkpointは監査および完全一致した完成recordの照合用であり、failed/interrupted runの再実行許可ではない。
compiler呼出前に予約recordを保存し、結果不明の予約を自動再compileしない。中断時は別reviewへ戻す。

## 終了artifactと解釈の範囲

成功した科学評価はresult JSON、manifest、`M2_COMPLETE.json`へ保存する。
statusは`TRANSFER_SUPPORTED`、`TRANSFER_NOT_SUPPORTED`、`TRANSFER_INCONCLUSIVE`のいずれかで、
どれでも`next_stage_authorized=false`とmandatory STOPを保存する。
implementation gate failureは`M2_FAILURE.json`またはgate前stderrの
`IMPLEMENTATION_GATE_FAILED`として区別し、未評価の5構成について架空の数値resultを作らない。

resource集計はcomputed/reused wrapper、trajectory/occurrence、wall/CPU time、parent/worker peak RSSを分離する。
失敗時はin-flight workerを含むpartial countsとper-wrapper ledgerの存在を明示し、0件と誤表示しない。
`.runtime`、registry、matrix/vector、pickle、npyは証拠commitに含めない。

SUPPORTEDでも言えるのはdevelopmentで固定した5構成のtransferだけであり、held-out上のmethod最適性ではない。
追加96、held-out再最適化、H5/H6/H12、別geometry/分子/PF、S3、長RPEへ進まず、研究方針の全面reviewへ戻す。

## ローカル検証と次の停止地点

M2 execution専用35件と、契約・M1-B1関連を合わせたfocused 84件がpassed、fail/skip 0。
syntheticの小さなbaselineでは実Qiskit transpile二軸も確認したが、H4のscientific compileは0である。
これはlocal implementation evidenceであり、immutable CI、外部再現、held-out transfer結果ではない。

source commit後は次のzero-compute commandでexecution planをfreezeできる。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" \
scripts/run_pr2_matched_accuracy_m2_transfer.py plan \
  --project-root "$PWD" --source-commit <actual-science-source-commit> \
  --output artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/execution_plan_v1.json
```

ここで停止する。別commitのresult-prior execution authorizationと最終reviewを経る前に`run`を呼ばない。
