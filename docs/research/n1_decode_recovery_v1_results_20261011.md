# N1技術復旧・一回実行：結果とGPT判断事項

技術復旧と認可されたN1のみの一回batchを完了。終端は **N1_RECOVERY_COMPLETE_AWAITING_SCIENTIFIC_REVIEW**。旧 `ungauged_eta` のdecode不備は解消し、全3 contextの科学結果を取得した。一方、費用込みN1 selectorはRZ/CX両方で全contextのexact/ungaugedを選択し、exact対照からのcertified objective改善は **0%**。技術停止と科学的否定結果を混同しない。

実装source `3fe779f88ab6367631355cf5e5152bf32e897a4d`、実行plan固定 `89a40b39e08c6c5d5ea474f7055319bcdced86b2`、branch `n1-decode-recovery-execution-20261011`。準備固定commit `ba4530678df9d665d2e6005fb08179b0f7a7921f` から分離。[元scope](n1_decode_recovery_contract_v1_20261011.md) の44 factor候補・178 pair・356 gauged/ungauged構成を変更しない。科学attempt1、retry0、resume0、条件追加0、N3/N2実行0。中心テーマnull、次段認可false、mandatory STOP。

対象は合成one-body＋signed DF rank2（weights .7,−.4）、JW4 modes・ν2・sector6、h1=diag(.08,−.04,.03,−.02)、scalar .11、geometry/chemical basisなし、DF rank固定2、L_Dなし。決定論S2、T=.6、q1/2/4/8、delta=.6/.3/.15/.075、model budget .02、εcomplex=.04、X/Y total failure .05、normalization1。固定X0X1状態準備を5-qubit full measured wrapper費用に含む。Qiskit1.3.0、optimization1、seed20261011、rz/sx/x/cx、ordinary control、qubits_initially_zero=False。

## selectorと比較結果

| context | q | shots/axis | XY RZ | certified shot×RZ | XY CX | certified shot×CX | N1/exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| planted_local_gauge | 4 | 16448 | 2494 | 41021312 | 1664 | 27369472 | 1.000000 |
| dense_group_frame | 4 | 16933 | 2614 | 44262862 | 1696 | 28718368 | 1.000000 |
| separated_spectrum | 8 | 13631 | 5222 | 71181082 | 3392 | 46236352 | 1.000000 |

全6（3 context×2 metric）のN1選択はexact対照と同じID。finite shifted/zero cutoff対照とall_mixed有限domainの最小も同じexact IDだった。選ばれた同一近似のungauged対照は選択自身と同じ。basis-only選択は最初の2 contextでcluster/exact pair(1,0)、separatedではexact/exact(0,0)だが、費用込みN1は全context exactへ戻った。未知の真誤差は選択に使っていない。

以下は既定q集合内のwhole-Pauli対照。selected targetは閉じたN1 RZ/CX選択に限り、全whole-Pauli候補を最適化していない。

| context | metric | whole-original best q | certified objective | whole selected best q | certified objective |
|---|---|---:|---:|---:|---:|
| planted_local_gauge | rz | 8 | 91170540 | 8 | 89692812 |
| planted_local_gauge | cx | 4 | 119290160 | 4 | 119290160 |
| dense_group_frame | rz | 8 | 111444480 | 8 | 109638144 |
| dense_group_frame | cx | 8 | 146055168 | 8 | 146055168 |
| separated_spectrum | rz | 8 | 174062900 | 8 | 171241620 |
| separated_spectrum | cx | 8 | 228120640 | 8 | 228120640 |

今回のwhole-Pauliはこの固定系・この上界会計ではnative exactより高かった。しかしN1近似の改善がなかったことは変わらない。表はideal-gate certificateに基づくshot×native-wrapper proxy。project全体の最終量子総資源、fault-tolerant費用、一般/scalable優位の評価完了とは呼ばない。

## 技術復旧・保証と監査

元保存pack SHA `70b2631b114983572cfc92527b2d569a4b48b03a6d88790b75dcf89f13614438` を保持。real/imag vector decoder、typed shape・finite・bool拒否・非zero spectrum imag拒否・unknown/oracle whitelist・groups/ID/order検査を追加。全numeric scalarのbinary64 as_integer_ratioを5428件照合。raw gをHermitizeせず、既定micro anti-Hermitian growth会計を保持。元N1 science source `bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46` との48 numerical function/class AST一致。科学kernel変更0。全context preflightをsource固定後にも確認し、PF/compile/truth前に完了した。

新N1-only synthetic18 tests PASS。1424 selector rows、contextごとのselection hashを閉じてから1424 truth診断を実行。wrapper要求2920、basis要求44、実transpile/native IR594、完全な発行回路fingerprint cache hits2370。独立保存checkerはscience module/Qiskit/compile/solverを使わず、全594 IRの全列（basis16列／wrapper32列）を再生し、別symplectic Pauli整数交換子・model/PF/bias会計・XY費用・selector argmin・safe整数shotを照合。26,575 checks PASS、最大絶対native residual 1.30332062708e-12。このbinary64全列検査は厳密native rounding enclosureではないため、claimは `ideal_gate_certificate_with_numerical_compile_validation`。

CPU32、最大2 spawn workers、各BLAS/OMP/Numba/Rayon1 thread、Qiskit parallel FALSE、transpile num_processes1、GPU0。batch wall 99.220915 s、parent CPU 1.206487 s、children CPU 178.120222 s、parent peak RSS 350736384 bytes、child最大peak RSS 399396864 bytes。parent+2×child peakからの保守的aggregate RSS上界 1149530112 bytes（同時測定peakではない）。実行audit時output 24567993 bytes。既存cache/vectorized native pathを再利用したが同条件逐次比較はなくspeedup率は主張しない。worker AS4GiB、aggregate8GiB、shard32MiB、output512MiB、log16MiB、wall/CPU capなし。

[入力・kernel照合](../../artifacts/n1_decode_recovery_v1/2026-10-11/preexecution/kernel_identity.json)、[実保存preflight](../../artifacts/n1_decode_recovery_v1/2026-10-11/preexecution/source_frozen_preflight.json)、[source-bound plan](n1_decode_recovery_v1_execution_plan_20261011.json)、[一次run directory](../../artifacts/n1_decode_recovery_v1/2026-10-11/run1/)、[集約result](../../artifacts/n1_decode_recovery_v1/2026-10-11/run1/result.json)、[比較集約](../../artifacts/n1_decode_recovery_v1/2026-10-11/run1/comparison_summary.json)、[独立監査](../../artifacts/n1_decode_recovery_v1/2026-10-11/run1/independent_audit.json)、[実行監査](../../artifacts/n1_decode_recovery_v1/2026-10-11/run1/run_audit.json)、[STOP](../../artifacts/n1_decode_recovery_v1/2026-10-11/run1/MANDATORY_STOP.json)。各contextのresult/checkpoint/selection/IRはrun下に保存。

## GPTへ返す判断事項

誤差込みselectorで悪化を避ける有限domainのexact fallbackは機能したが、今回の近似構成はexactを上回らなかった。復旧済みの新結果を用いて、N1を中心研究として継続する価値を判断してほしい。今回の科学的結論はこの合成4-mode系・固定candidate/q/precision/compiler条件の範囲に限る。中心テーマを自動採択せず、追加scanで成功を探さない。N3の結果は別branchで管理し、この実行はそれを再実行していない。

local tests/保存監査はimmutable CI・外部科学再現ではない。global manifest checkerの旧 `relevant_commits[7].description` 欠落FAILは修正しない。新entry/旧89 result_sets保持のlocal監査は別に記録。GitHub固定commitと独立再取得はpublication資料へ記録する。

公開前metadata監査PASS：旧manifest89 entriesをraw prefix/suffixのまま保持し、個別entry1件のみ追記。旧source/results220 blobsと準備契約/inputを含む計232 blobsのbyte保持を確認。新entry schemaはPASS、global manifestの既存FAILとは区別する。 保存native最大9,176 operations、serialized integer最大739 bits（全算術中間値の測定とは区別）。 [metadata監査](../../artifacts/n1_decode_recovery_v1/2026-10-11/publication_metadata_audit.json)、[root保持](../../artifacts/n1_decode_recovery_v1/2026-10-11/root_preservation.json)。
