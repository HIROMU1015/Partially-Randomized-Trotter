# RPE m17：作用誤差上界を取得、全supportの測定完了は未保証

2026-11-04 JST。**固定SOURCEで96 bit・1 passを正常完了した。有限性とruntime条件を確認したが、
全supportのguard通過を示せず、Model IIの測定平均に対する十分条件は未判定。**
局所結果は`ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED`。
両軸`U_virtual_IID_upper=null`、固定PCG64の`U_exec_upper=null`を保持する。
計画どおり追加精度・retry・samplingへ進まずmandatory STOPした。

## SOURCE・固定条件

科学実行SOURCE：`c3a5cc387b8962b63782dc71ab175147f2d87e33`。
[固定実行計画](../rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/SCIENTIFIC_EXECUTION_PLAN.md)、
[統合runner](../rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/run_scientific_bridge.py)、
[準備時の人工57 tests](../rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/VALIDATION.txt)を変更せず使用した。
結果branch：`codex/rpe-m17-virtual-bridge-scientific-execution-20261104`。
科学SOURCEと結果公開commitは区別する。

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bの保存raw physical state、
8 qubit・4電子sector70、Partial L_D3、単一δ=.02、m17/q131072/r32/K2。
extract_identity_phase、coefficient_atol=1e-12、N411/axis、保存α=.016957089147912167、
ε=.15244325851620388、β_PF=.014050012886294481、β_RTE=9.899584668737594e-8、β_stat=.38594988811785885を維持した。

確率モデルは独立一様53-bit categorical wordと独立一様53-bit measurement wordを仮定するModel II。
固定seedのPCG64、compiled回路/noise、実state preparation、全18round、energy・総費用は認定対象外。
原RESULTにμ0/N/α/β/εと入力6件のcommit/path/SHA256を保存した。

## 使用した数値contextとcall範囲

latest包含結果commit `34840ce9bf81d985b32764d9bc9d77645ed52e59`のRESULT/export/laws/包含operator、
B結果 `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`のraw ψ、
元m17結果 `cd749066904d3f29b28b03c19ab69370372b157f`のF/R64をbyte/hash照合して使用。
保存local binary64 matrixを数値composeし、basis11件を共有してP_i64を218件構成した。
これは今回固定した新m17 contextであり、run02の未保存payloadを取得したものではない。
state bytes/fingerprint、signed component、product→rotation、F→r events→R、Re/+Imの規約を保持した。

保存θ2個に対してcos/sin64を一度取得し、ideal basisとのcomponent残差、保存F/R包含との差、
既存KernelBoundsによるruntime丸め、raw normとvdot、平均/一様affine growthを96 bitで評価。
q/rの長列はscalar binary powerで扱い、実trajectoryを生成していない。
旧ideal categorical mean、CDF、統計半幅sは再計算せず、保存logical interval/norm²/sを再利用した。

段階callは保存load開始1、numeric context開始1、bound開始1・完了1。
SCF/DF因子化、H生成、state solve、新SOURCE replay、sampling/RNG replay/仮想shot、
controlled回路build/compile、成功率・費用再解析、独立詳細科学レビューは0。
中断・retry/resume・追加精度・他round・B-1b/B-2も0。

## 得られた上界と判定

原記録：[RESULT.json](../../../artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01/RESULT.json)。
下記は原RESULTのexact rational上端を小数表示したもの。正式値は原JSONに従い、
報告作成時にoperator・norm・平均・budgetを追加評価していない。

| 保存項目 | 表示概数 | 意味 |
|---|---:|---|
| conditional_average_error_upper | 8.0111580176e-8 | guard前のraw complex signalとnormalized ideal logical signalの平均絶対誤差majorant |
| conditional_uniform_error_upper | 2.5546158311e-6 | 同差の全support一様majorant |
| uniform_signal_magnitude_upper | 1.00000255461583 | float raw signalの絶対値に対する全support上端 |
| boundary_norm_upper | 1.00000255461580 | state-action境界normの一様上端 |
| matrix_frobenius_upper | 8.3666002653 | 有限性screen等に使った最大行列Frobenius上端 |
| complex_abs_error_upper | 1.1368712815e-13 | このsignal上端に対する既存hypot誤差上端 |

finite_intermediates_proved=true。今回のkernel profile/FP条件下で、no-overflow screenを満たした。
一方、SOURCEはclipの前にraw座標の`[-1-1e-12,1+1e-12]`とcomplex magnitudeの`1+1e-10`を検査する。
上表の一様上端ではその通過を証明できず、all_support_completion_proved=falseとなった。

これは**実際に例外を出すtrajectoryを発見した結果ではない**。
一様majorantが保守的である可能性と、実装が全supportで完了しない可能性を今回分離していない。
有限性の証明、guard通過の証明、平均誤差の小ささは別の条件である。

| 軸 | U_virtual,IID上端 | U_exec上端 | 判定 |
|---|---|---|---|
| cosine | null | null | UNDETERMINED |
| sine | null | null | UNDETERMINED |

全supportで±1測定まで完了すると示せないため、Model IIの無条件測定平均を認定せず、
保存s/εへのstrict比較も成立として報告しない。
上表の平均作用誤差をそのままU_virtualへ代入したり、失敗shotを除外/置換したりしていない。
N/α/β/ε、guard tolerance、clipping方法、拒否確率の扱いを変更しない。

既存のideal categorical局所十分条件確認は有効なまま。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
旧numeric_diagnostic_status=INCONSISTENT、旧u_num_bound=nullを遡及変更しない。

## Runtime・資源・実行記録

実runtimeはRUNTIME_CONDITIONS_CONFIRMED、phase=complete、sequence7、failure=null、child_exit_code0。
NumPy/OpenBLAS/libm hashとPrescott core、CBLAS alpha1/beta0・shape70、
GEMVT4出力partitionの合計70、DOT、AVX2/FMA multiply/subtract、_Py_c_abs/hypotを同じworkerで確認した。
checkpoint前半は4 thread、FLINT開始後は35 thread。観察MXCSRは0x1fa0/0x1fa2で、
nearest-even、FTZ/DAZなし、必要な例外maskを満たした。

gdbのSIGTRAP表示は7 checkpointの観察用signal。例外中断ではなく、実行はexit0で正常完了した。
起動時CPU affinity32、load約1.60、MemAvailable約54,844,596 kB、出力FS空き約103 GiB。
GPU driverは利用不可。支配的処理はCPUの70次元残差包含で、basis共有とscalar binary powerを使った。
FLINT32 thread、BLAS/OpenMP/MKL各4は自jobのphaseに合わせた構成で、共有設定は変更しない。
wall timeは23.5890530907秒。process RSS/占有率/serial比較は測定せず、速度倍率は主張しない。

```bash
PYTHONWARNINGS=ignore MPLCONFIGDIR=<owned-temp-dir> OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/run_scientific_bridge.py --run --output-dir artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01
```

| 保存原本 | Bytes | SHA256 |
|---|---:|---|
| RESULT.json | 62775 | `d2989174012c41e6908d21ccc65d894820e763749e922835f9c5e191eb3c8735` |
| numeric_context.npz | 8067644 | `f8c1f3185423f479d82ba0bc1eb94bfb8fd9d0cafa487c618964f3bd22bddbae` |
| runtime.json | 1164 | `923bb5235429673fc6a5d5a5ac54c35b7ba27334a48d6f8b58c1eaf6127408e6` |
| runtime_request.json | 43 | `ef30a02477cbcd3429668f5c9f02097f12e32ec3106cf7242a87057e68408e46` |
| observer.log | 2627 | `8fe2f23c719df3394f7798d738bb3bc3b7934b1a83164e74fe40e05fd27b7793` |

NPZにP000〜P217、forward/reverse/psiを保存し、成分ID順とcos/sin出力hexはRESULTへ記録した。
原JSON・ログ・NPZは実行後に書換えず、hash/ZIP/schemaと固定SOURCE保全を確認する。
この確認は独立科学レビューや追加数値解析ではない。

## 到達点・次段案・STOP

数値作用誤差のmajorantと有限性は得られたが、仮想±1測定への保証bridgeは未完了。
今回の正常完了を「意図しない中断」と解釈して修正・再実行しない。
SOURCE・raw・旧契約/結果・grant/ledger/marker/STOP、他branch/worktree・SERVER2 H8を保全する。

次段案は**Codex**が、今回の一様majorantの保守性とguard証明の不足をSOURCE・数式から静的に整理し、
既存の科学条件を保ったまま上界を鋭くできるか検討すること。
guard緩和、失敗shotの条件付け、kernel変更等は自動採用しない。
**ユーザー**が次の調査範囲と、必要なら新しい固定SOURCEでの限定数値評価を別途判断・許可する。
今回の結果を公開照合した後にmandatory STOPする。
