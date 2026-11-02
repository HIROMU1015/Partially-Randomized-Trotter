# Track A PM-1 nearby discard：実行結果と照合

検証日：2026-10-05 JST。terminal status：`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`。

固定8構成を一度だけ実行し、全件がaccuracy適格だった。新しいdiscard集合内のprimary RZ最小は
`PM1-B0-rank5-q1-r0-K0`（229,718,060）。旧B0 rank6・q1より9.34%低いが、
保存済みB2 rank3・q1・r4・K2の32-trajectory点推定（130,774,896.65625）の1.75659倍である。
rank4/5の欠測を埋めても、この比較範囲ではB2の低いprimary点推定は覆らなかった。

これは同じdevelopment条件・second-order DF-prefix implementation class内のlocal evidenceである。
研究継続・scope縮小などの結論はrunnerも本照合も自動決定しない。
`mandatory_stop_reached=true`、`next_stage_authorized=false`、`research_decision=null`。
PM-2、追加trajectory、別geometry、strong synthesis、higher-order PF、energy/RPE接続、Track B統合には進まない。

## 固定条件と認可

H4 linear 1.00 Å、STO-3G、DF rank12 Hamiltonian、8 system qubits、保存済みdevelopment近似状態、
`T=0.8`。B0 discardのDF-prefix `L_D=4,5`、`q=1,2,4,8`、
`delta=T/q=0.8,0.4,0.2,0.1`、`r=K=0`に限定した。
full-H targetは保存済みM1-Aから固定し、新しいground-state solveやexact truncated-H signalを求めていない。

[結果前契約](research/pr2_pm1_nearby_discard_contract_v1.md)、
[authorization](research/pr2_pm1_discard_execution_authorization_v1.md)、
[承認booleanの確定記録](research/pr2_pm1_execution_finalization_20261005.md)を維持した。
利用者提供review `APPROVE_PM1_EXECUTION`と明示指示「PM-1を実行して」を受け、
固定root/output/commandで一度だけ起動した。reviewをrunnerが独立認証したとは主張しない。

CPU単一process、BLAS各1。Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1、
Qiskit 1.3.0、basis `rz,sx,x,cx`、optimization level1、transpiler seed17、
backend/coupling/layout/routing指定なし。環境・science source・候補・thresholdは変更していない。
full measured Hadamard wrapperをcosine/sine二軸でcompileし、状態準備costは除外した。

## 8構成の結果

complex-signal accuracyは0.05。各軸allowanceは `0.05/sqrt(2)-axis_bias`。
primaryは `G_RZ=N_real*C_cosine,RZ+N_imag*C_sine,RZ`。
解析的shot数であり、量子shotは実行していない。discardのnormalizationは1である。

| L_D | q | delta | accuracy適格 | total bias abs | total analytic shots | G_RZ | 保存B2 r4に対するpoint比 |
|---:|---:|---:|---|---:|---:|---:|---:|
| 4 | 1 | 0.8 | yes | 0.026452635 | 109,738 | 784,187,748 | 5.99647 |
| 4 | 2 | 0.4 | yes | 0.032009965 | 597,365 | 8,352,357,430 | 63.86820 |
| 4 | 4 | 0.2 | yes | 0.033323339 | 1,333,877 | 36,887,034,558 | 282.06510 |
| 4 | 8 | 0.1 | yes | 0.033647195 | 1,733,007 | 95,311,918,986 | 728.82427 |
| 5 | 1 | 0.8 | yes | 0.013400419 | 25,910 | 229,718,060 | 1.75659 |
| 5 | 2 | 0.4 | yes | 0.018977875 | 40,086 | 698,378,292 | 5.34031 |
| 5 | 4 | 0.2 | yes | 0.020296255 | 45,772 | 1,580,690,248 | 12.08711 |
| 5 | 8 | 0.1 | yes | 0.020621349 | 47,400 | 3,259,129,200 | 24.92167 |

biasはdiscard＋PFの**総bias**。pure discard / pure PF biasは欠測のため両方nullを維持した。
qを増やしてもこの8件のmatched workは減らなかったが、上の値だけから誤差の成分分解や因果機構を確定しない。
各軸signal、bias/allowance/shots、compiler metrics6指標、fingerprint、wrapper identityは
[result JSON](../artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json)に保存した。

## 保存済み比較集合と主張の範囲

同じH4 1.00 Å、DF rank12、T=0.8の以下5件だけを保存JSONから引用した。
B2/B3を再sample・再compileせず、held-out M2を今回の比較に混ぜていない。

| 保存candidate | primary G_RZ | 新B0 rank5・q1 / 保存candidate |
|---|---:|---:|
| B2-rank3-q1-r4-K2 | 130,774,896.65625 | 1.75659 |
| B2-rank3-q1-r8-K2 | 132,704,255.1875 | 1.73105 |
| B0-rank6-q1-r0-K0 | 253,379,350 | 0.90662 |
| B1-rank12-q1-r0-K0 | 372,523,128 | 0.61665 |
| B3-rank0-q8-r32-K4 | 1,152,790,918.0000002 | 0.19927 |

B2/B3は既存32-trajectory compiled-cost点推定、B0/B1はdeterministic値である。
今回のpoint比をformal CI、厳密winner、一般的method最適性とはしない。新しい10% GO判定も導入しない。
rank5は旧rank6より良いdiscard baselineになったが、これで全prefixやstrong synthesisに対する最適性を示したわけではない。
状態準備を含む最終総cost、energy/QPE/RPEの最終評価も未実施である。
結果を受けた研究方針の全面reviewが次の停止後作業となる。

## 実行資源と結果照合

launch時計観測は2026-10-04 15:28:49 UTC、exit0の確認観測は15:31:43 UTC
（JSTでは10月5日00:28:49–00:31:43）。
これは起動・終了の外側の観測であり、runnerがprecise終了時刻を出したとはしない。
runner内部wallは141.614秒（約2分22秒）、peak RSSは2,145,856 KiB（約2.05 GiB）。

signal attempted/completedは8/8、wrapper reserved/completedは16/16、unique keys16、
未解決予約0、cache reuse0。development hash/loadは各1回、CPU1 process。
random sampling、held-out access、GPU操作、量子shot、retry/resumeは全て0。

結果後照合は保存JSON/sourceだけを読み、NPZ/runtime/cacheを再度開いていない。
134 source blobs、plan/authorization、8 candidate fingerprints、signal式とfingerprints、
軸別compiler/wrapper semantics、6指標work、40 point ratios、16 wrapper keys、
complete marker、STOP/resource gate、runner manifest2件のbytes/SHA-256が全件一致した。
runner生成result/marker/manifestは書き換えず、別のauditとvalidation manifestを追加した。

限定guard付き7-file testsはpre/postとも201 passed（PM-1 49、PM-0 18、helper134）、
fail/skip0、protected access attempt0。preは19.54秒、postは19.44秒。
synthetic小型compileをH4 science wrapper数に含めず、full repository testsは実行しない。
正確なcommand、Python identity、UTC時刻、stdout/stderrは
[launch audit](../artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/launch_audit_v1.json)と
[test audit](../artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/test_audit_v1.json)に保存した。
検証commandと項目別結果は
[result verification](../artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result_verification_v1.json)に保存した。

## Provenanceと停止状態

branch：`pr2-v4-s2-parallelization-20260928`。
launch HEAD：`ab6f0bbe908e28b2ddf90550cd485db7fce89e39`。
actual science source：`fd7552edc0334ccf57ecf501a128c85c8d22822a`。
review bundle：`234369f45e3fea7c825a1b4af85fba6c1edf98c3`。
authorization一項目確定commit：`bf9eaeec868361df0c8e05d06e9570a2bfc5a7a4`。

| identity | SHA-256 |
|---|---|
| sealed plan | `cae692bee2be748ddbf17bace2a5652613537a244fe94238cdf669f5d2ca5624` |
| finalized authorization | `2113978b360ca763071b860c7ea2d14b83eccc68a471510a98da5ea5c2282530` |
| result JSON | `9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b` |
| runner manifest | `3033ab68de04d867b00785446fb6223db0754c8f92218fcc478fc53b101f1317` |

result fingerprint：`87af3ea3132f1ff021e8727afb83948dac7471abf7a9f4634debba2d42dd62fe`。
outputの日付2026-10-04はauthorizationで固定したidentityであり、実行日JSTへrenameしない。
旧draftのfalse authorization hash、準備時の監査違反、finalization時のlaunch待ち記録は履歴として保持する。
利用者の追加指示「コミットプッシュして」を受け、本結果・追加audit・報告をresult commitへ収録する。
正確なcommit hashとpush結果はcommit後のhandoffで示す。保存済み実行監査の未commit labelは照合時点の履歴である。
公開後もsource-bound local execution evidenceであり、immutable CIや外部再現ではない。
registry・runtime・NPZ・matrix/state/vectorは成果物のcommit対象にしない。
成功のままmandatory STOPに入り、以後の科学計算は別review/別認可を要求する。
