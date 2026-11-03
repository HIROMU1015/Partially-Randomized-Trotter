# 現行kernelの実行経路

2026-11-04 JST。現行環境のread-only調査と人工入力だけを使用。実H4の計算・保存数値再解析・samplingなし。
結論：[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。数式：[ROUNDOFF_BOUND_DERIVATION.md](ROUNDOFF_BOUND_DERIVATION.md)。

## 固定参照と対象

bridge準備SOURCEは`728c90a2d4d4335ef5b7237808c15b182e694bb9`の
[adapter](../rpe_m17_virtual_measurement_bridge_preparation_20261104/virtual_bridge_adapter.py)、
[runner](../rpe_m17_virtual_measurement_bridge_preparation_20261104/run_virtual_bridge.py)、test/VALIDATION/2報告。
候補A実行SOURCEは`d0b1da1b04c266a1f7abf8e43966ca6e289a8695`、
carrierは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。
包含科学SOURCEは`18fffce48e33136b91eb6ac5a6f9a5f77a08d36f`、結果公開は
`34840ce9bf81d985b32764d9bc9d77645ed52e59`の[REPORT](../rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md)。
包含平均・state norm・予算は再評価しない。

研究対象はlinear H4/1.00 Å/STO-3G/保存legacy DF12/B raw state/4電子sector70、
L_D3/単一δ=.02/m17 q131072/r32 K2、extract_identity_phase、cutoff1e-12、N411/axisと保存α/β/ε。
今回のprobeはこの物理入力を一切読まず、人工4次元・70次元の配列だけを作る。

## 実環境と観察方法

Python3.11.0rc1、NumPy1.26.4、OpenBLAS0.3.23.dev、ILP64/DYNAMIC_ARCH、Ubuntu glibc2.35。
runtime getterはPrescott。NumPyのbuild表示にあるHASWELLは使用kernelの根拠にしない。

[artificial_dispatch.py](artificial_dispatch.py)を自分の新規子processとしてgdbで起動し、
CBLAS・実kernel・NumPy ufuncのbreakpointを通過した時点の引数とMXCSRを読んだ。
既存jobへのattach、共有library変更、kernel置換、環境のglobal変更は行わない。
probeにはRNG・saved array loader・科学runner呼出しがない。
最初のsandbox内ptraceはOSに拒否されたため、許可された同じ人工子processを通常の承認経路でsandbox外から追跡した。
設定の迂回・system ptrace制限変更はしていない。

[dispatch_observation.txt](dispatch_observation.txt)は最終出力からPROFILE/ARTIFICIAL/TRACE/DONE行だけを抽出したもの。
probe時にBLAS thread数を1へ制限せず、実際のmulti-thread経路も確認した。

```bash
gdb -q -batch -x docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104/trace_dispatch.gdb --args <existing-python> docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104/artificial_dispatch.py
```

## Dispatchの確認結果

| SOURCE処理 | 人工probeで通った実経路 | 保証への採用範囲 |
|---|---|---|
| C連続complex128 A@v | CBLAS layout=102/transpose=112 → zgemv_t_PRESCOTT。70×70ではzgemv_thread_t経由 | C連続A、contiguous正stride v、alpha=1+0i/beta=0、m/n≤70だけ |
| thread wrapper | 70次元ではm=70を保ち、n=4の17呼出し＋n=2の1呼出し | 内積長を分割せず出力列を分担。TRANSAのdriverは出力側range_nをずらし、thread partial-sum reductionを行わない |
| F連続A@v | CBLAS layout=101/transpose=112 → zgemv_n_PRESCOTT | 別経路と明示。今回の係数interfaceではF layoutを受け入れない |
| strided A@v | CBLAS/GEMV breakpointを通らずNumPyの非BLAS経路 | 存在する別経路。今回の実環境profile保証から除外 |
| np.vdot | CDOUBLE_vdotのCBLAS経路 → zdotc_k_PRESCOTT、n=4/70 | conjugationを保持。NumPyのchunk後の初期0への加算もdepth余裕内 |
| complex scalar×vector / subtract | CDOUBLE_multiply_FMA3__AVX2 / CDOUBLE_subtract_FMA3__AVX2 | broadcast scalar、非alias出力、complex128。scalar fallbackも同じ短い演算式で覆う |
| abs(complex signal) | _Py_c_abs → 実libmのhypot（glibc2.35、offset0x26210） | 今回確認した非FMA correction版hypot。有限signal、後述のrange条件が必要 |
| np.linalg.norm(vector) | 実部・虚部のreal DOT各1回、stride2 → sqrt | 旧preparationの診断経路。新bridgeは保存norm²のAcb包含を使うので、これを厳密normの根拠にしない |

元m17 operators.npzのforward/reverseの**NPY headerのみ**を確認し、両方`<c16`、shape(70,70)、fortran_order=Falseだった。
B raw stateのheaderも`<c16`、shape(70,)、fortran_order=False。
行列要素は読込・評価しない。新numeric_componentsもascontiguousarrayを返す。
このlayout対応は実行時にも確認すべき条件で、F/stridedからCへ無言で変更して既存実装を認定しない。

## SOURCEとbinaryの対応

公開SOURCEは処理の役割を確認する参照であり、v0.3.23 releaseとdev binaryのbyte一致は主張しない。
実際の経路と命令種類は、導入済みbinaryのsymbol/objdumpとgdb通過から確認した。

| 確認範囲 | binary側の要点 |
|---|---|
| zgemv_t_PRESCOTT (0x497a00〜) | 入力copy、4出力/2出力/1出力のループ、SSE mulpd/addpd/subpd/haddpd、最後のalpha作用とy加算。dimension≤70では巨大m block反復に入らない |
| zdotc_k_PRESCOTT (0x490000〜0x4915ea) | contiguous/alignment分岐、8/4/2/1要素処理、4部分accumulator、最後の2段reduction/conjugate符号。SSEでFMAなし |
| zscal_k_PRESCOTT (0x494600〜) | beta=0の分岐はzero registerをstoreし、未初期化yの値を乗算しない |
| multiply_FMA3__AVX2 (0x4332e0〜) | vmulpdとvfmaddsub等。各複素要素の2積±和で、reductionなし |
| 実hypot (0x26210〜) | SSE mulsd/addsd/subsd/sqrtsd/divsdとpower-of-two scaling。FMA版へは入っていない |

GEMV/DOTの各実数出力は2n個のsigned real productを一度ずつ足すforestで、
alpha=1/beta=0の追加作用を含めてもdepth≤2n+16、丸めsite≤4n+16で覆える。
特定のreduction順や最小depthは保証に使わない。threadの出力分担はこの個数を増やさない。
これは全BLAS命令のauditではなく、使用する狭い演算graphの確認である。

## FP mode・有効条件

kernel/ufunc/hypotの観察時MXCSRは0x1fa0/0x1fa2。
RC=nearest-even、FTZ=0、DAZ=0、例外はmaskされている。low sticky flagsは異なっていてよい。
GEMVTの各workerでも同条件を観察した。これによりgradual underflowを含む加法誤差モデルを使える。
[IntelのFTZ/DAZ説明](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2023-1/set-the-ftz-and-daz-flags.html)。
将来processの状態まで固定したわけではない。使用時には全参加threadでこのmodeを維持し、
foreign codeによる途中変更がないことが必要。FTZ/DAZ有効、directed rounding、別CPU/kernel/binaryには自動転用しない。

| 今回調べたbinary | SHA256 |
|---|---|
| OpenBLAS .3.23.dev.so | `9254d0854dd7615e11de28d771ae408878ca8123a7ac204f21e4cc7a376cc2e5` |
| NumPy _multiarray_umath.so | `a735e4e8355b75c800112af8a5b1731b891b2016ed57067352eea6cab0aa00bc` |
| libm.so.6 | `1a08a427bbf3790aa434d2310e8c724846ec72ba7811982e85b0002dd296ca92` |

3 hashは対象binaryと調査結果を結ぶ数値provenanceで、全依存環境のidentityや許可の証明ではない。
実データのnorm・guard・全support有限性は未評価。

SOURCE参照：
[NumPy matmul](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/matmul.c.src)、
[vdot](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/vdot.c)、
[ufunc](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/loops_arithm_fp.dispatch.c.src)、
[norm](https://github.com/numpy/numpy/blob/v1.26.4/numpy/linalg/linalg.py)、
[OpenBLAS zgemv](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.23/interface/zgemv.c)、
[thread driver](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.23/driver/level2/gemv_thread.c)、
[GEMVT](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.23/kernel/x86_64/zgemv_t.S)、
[DOT SSE](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.23/kernel/x86_64/zdot_sse.S)、
[CPython complex](https://github.com/python/cpython/blob/v3.11.0rc1/Objects/complexobject.c)、
[glibc hypot](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_hypot.c)。
