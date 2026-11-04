# B追補2：測定入力経路・距離kernelと一様保証の限界

2026-11-05 JST。親commit `4318f0bae689a11256d1e3f638db089a1f6a4a13`、専用branch
`codex/rpe-input-kernel-preparation-20261105`、独立cloneでの技術作業。
旧[追補](FLOAT64_ADDENDUM_20261105.md)、SOURCE、科学原結果は変更しない。

**結論：18round対応の既存測定接続interfaceを追加確認し、wrap用fmodの限定域の正確性を
現在のbinary・束縛先へ接続した。complex exp／atan2の一様誤差契約は未取得で、実H4科学SOURCEは準備不可。**
現行decoderの保証が数学的に不可能という結論ではない。完成には下記の限定域の誤差証明が必要であり、
取得済み資料だけで完成できる見込みは立たない。adapter・科学runnerを追加し続けず、ここでSTOPする。

固定研究条件：linear H4／隣接1.00 Å／STO-3G／保存legacy DF12／Partial L_D=3／δ=.02、
保存physical state normalized ray・4電子sector70、m0..17、q1..131072、保存N/α/β/ε/μ₀/E_ref。
今回はこのH4数値、成功格子、margin、energyを新規評価していない。
sampling、RNG replay、回路build/compile、費用評価も0。A/C・36軸包含の再計算0。

## 必須対応表

| 対象 | 確認済み事項 | 未保証事項 | 次の対応 |
|---|---|---|---|
| 測定入力経路 | 既存18round接続interfaceは整数和/N→complex→**cmath.phase**→scalar decoder。backendは外部callback | 実H4 production runner・backendの採択/接続、callback実dtype契約、IID | 既存interfaceを使うかを明示し、外部callback契約とall-round orchestrationだけを確定 |
| fmod／root wrap | NumPy scalar slot、ufunc、GOT束縛を同定。限定域の整数shift/subtract kernelの正確性を下記で証明 | 将来process/全threadのFP前提、H4全成功入力上のprincipal域・a_m | 同一binary・RN/gradual・principal域の前提を固定すれば既存補題へ接続可能 |
| Complex exp | complex128 loop→nc_exp→npy_cexp→libm cexp。pure imaginaryではsincos主体 | sincosのreduction/table/polynomialと丸めの全域誤差 | 対象binaryの限定域のe_exp証明が必要 |
| atan2 | NumPy通常loopとCPythonの束縛先は同じlibm wrapper、内部IFUNCはFMA variant。cmath特殊値分岐を別確認 | 全引数域の角度誤差。SOURCEの約.524 ulpはrandom sampling由来 | FMA variantの全branch/table/極小比を含む一様証明が必要 |
| 全candidate距離順序 | 既存strict十分条件・全root対応の定理を再利用。遠い候補の誤距離の人工反例も既存testで検出 | d_m未取得のため全候補の順序は未保証 | 全候補に共通するd_mを証明するまで近傍2候補へ縮約しない |
| Float64 decoder全18round | 入力候補SOURCEを特定、条件付きc補題を進めた。人工54件PASS | production a/c/d、実装energy成功保証 | 科学SOURCE未準備。証拠取得の指示があるまでSTOP |

## 入力経路：前回調査の補足・訂正

前回は直接の`reconstruct_rpe_phase`呼出を中心に追跡し、関数を返すloader経由の既存
[measurement_adapter.py](../rpe_18round_measurement_connection_20261103/measurement_adapter.py)を漏らしていた。
**「18roundに対応する測定接続SOURCEがない」とは扱わない。**
このSOURCEは`8722b5a461c617b8846c2754275895f152df78f3`で既に公開されている。
同commitの[接続報告](../rpe_18round_measurement_connection_20261103/IMPLEMENTATION_CONNECTION_REPORT.md)も確認した。
これは人工fixtureで検証したplumbingであり、production backend・科学runnerは未実装と明記されている。

SOURCEを追うと、既存接続は次のとおり：

1. `collect_virtual_round(settings, trajectory_factory, measure_one)`がaxisごとにN回callbackを呼ぶ。
   `total=0; total+=outcome; means.append(total/n)`。**built-in intの±1**というcallback契約ならPython整数和はexact、
   N≤2^52で最後の除算だけbinary64丸めとなる。元`mean_bound_under_rn(...,'integer_sum')`を適用できる。
2. `MeasuredMeans.cosine/sine`はこの契約ではPython float、`.signal`はPython complex scalar。
   型annotationだけは強制ではなく、既存bodyはfloatの±1も受理する。今回SOURCEを変更せず、
   arbitrary callback型まで整数和契約として証明しない。`MeasuredMeans`の外部直接入力も、有限range検査だけでは標本格子を保証しない。
3. `decode_measurements(rounds,means)`は`cmath.phase(z.signal)`のPython float listを作る。
   `original_decoder()`がcarrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`から無変更の3 FunctionDefを読み、scalar decoderを実行する。
   q順序は2^mで検査される。全18roundにも対応するが、この関数自身は長さ18限定のproduction runnerではない。
4. seam/tieのstatusは元選択を変更せずenergy出力を保留する診断。`1e-12`のiscloseは一意性の証明ではない。
   zero signalもdecodeしない。energyは`-final/delta`、E_ref/oracle入力なし。

CPython3.11.0rc1の[cmath SOURCE](https://github.com/python/cpython/blob/v3.11.0rc1/Modules/cmathmodule.c)では、
phaseは`c_atan2`を呼ぶ。有限・虚部非zeroならlibm atan2、虚部±0ではrealの符号によりsigned zero/±πを返す。
そのため、**既存finite_input_envelopeのnp.angle版を無変更でcmath版のa_m保証と呼んではならない**。
有限格子のexact抽出・Arb比較の方式は再利用できるが、実際のphase extractorを明示する必要がある。
今回H4のa_m評価や新しいangle adapterは作成していない。

旧4roundのnp.mean(int8 outcomes)→scalar np.angle、count-grid Monte Carloの`2k/N−1`→vectorized decoder、
fresh-IID診断のfsum/N、V2の保存analytic mean→scalar np.angleは、それぞれ別経路として維持する。
repoのsrc/scripts/tests/docs/researchを直接名・loader名・decode名で追跡した範囲では、
上のplumbingを全18roundの実backendへ接続するproduction entrypointは見つからない。

最小の不足接続は、固定rounds全体に対する`collect_virtual_round`の呼出と、その戻り値を順序通り
`decode_measurements`へ渡すorchestration、および外部factory/measurement callbackの入力・dtype契約である。
整数±1、同じphysical law・fresh-per-shot条件を保てば、数学的なN/α/ε/ideal IIDモデルを変える必要はない。
ただし保証対象の数値経路はcmathになるので、その正式採択とbackend対応を自動的に成立済みとはしない。
新しいpipeline・平均方式・decoderを作成/採択せず、Codex Cの回路研究にも干渉していない。

## 現processのruntime・kernel束縛

追加した[inspect_decoder_kernels.py](inspect_decoder_kernels.py)はread-onlyの技術調査helper。
旧helperには解決済みufunc loop／ELF relocation／FP制御の観測機能がないため、この一点を追加した。
NumPy1.26.4のcapsule docstringと[legacy wrapper SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/legacy_array_method.c)、
installed `ndarraytypes.h`に沿って解決済みloopを読む。unsupported ABIはUNKNOWNで止める。
他processの追跡、ptrace、interposition、sudo、FP rounding設定の変更は使用していない。

[INPUT_KERNEL_VALIDATION.json](INPUT_KERNEL_VALIDATION.json)にbinary hash、解決済みloop、GOT束縛先を記録した。
Python3.11.0rc1／NumPy1.26.4／glibc2.35／x86_64。NumPy拡張・libmは前回と同じhash。
x87 control=0x037f、MXCSRのroundingはnearest、FTZ=false、DAZ=falseを現在threadで観測した。
exception status bitsの値は誤差契約に用いず、将来processの全threadへ一般化しない。

| 段階 | 対象ELF内address／解決先 | 根拠・範囲 |
|---|---|---|
| scalar wrapの% | NumPy `double_remainder` @0xf4570→fmod | CPython stable slot Py_nb_remainder=34を取得。scalar演算は配列ufuncとは別入口 |
| 配列remainder f8 | NumPy `DOUBLE_remainder` @0xe3be0→fmod | 解決済みdtype/stride loop。`npy_divmod`も同じfmodを使う |
| angleのf8 atan2 | NumPy `DOUBLE_arctan2` @0x2cdb20→atan2 | stride8/16の両方で通常loopを解決。AVX512 symbolの存在から推測しただけではない |
| complex128 exp | `PyUFunc_D_D` @0xdb810、userdata `nc_exp` @0x25e510→npy_cexp @0x283b60→cexp | 解決済みlegacy loopとuserdata、disassembly |
| NumPyのGOT | fmod→libm @0x10ea0、atan2→@0x10c10、cexp→@0x363f0 | 実processのrelocation slotをread-onlyで確認 |
| CPython atan2 | 同じlibm @0x10c10 | Python実体の束縛先。cmathの特殊値分岐は別扱い |
| libm内部atan2 | IRELATIVE slot→@0x78060 | resolver @0x25ab0、[FMA選択SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/x86_64/fpu/multiarch/ifunc-avx-fma4.h)・disassemblyに対応 |
| fmod本体 | libm @0x26020、size496、非IFUNC | public wrapper @0x10ea0から有限・除数非zeroで直接到達 |
| cexpの三角関数 | cexp→sincos @0x2fa80 | finite pure-imaginary部分。巨大実部/非有限分岐はdecoderの有限Δ域から除外する条件が必要 |

addressはASLR後の生addressでなく、固定ELFのvirtual address。
helperの解決APIで選択したnumeric loopとSOURCE/disassemblyの対応を確認したが、
全呼出のtraceを取ったとは主張しない。dtype、stride、scalar/array、binary、FP条件が異なる別processへ自動移送しない。
**dispatchの同定と精度の証明は別**であり、helperはprecision_contract=nullを返す。

## fmodの限定域証明とc_mへの接続

p64=π64、D=2p64、u=2^−53。前追補のroot補題を再利用する。
wrap入力x∈[−4,8]ならy=RN(x+p64)∈(−D,2D)、Dのbinary exponentは2。
有限y・正のDに対するpublic fmod wrapperはlibm @0x26020へ進む。
この本体は[glibc2.35の整数shift/subtract SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_fmod.c)と
同じ下記の操作を行う。version名やaccuracyコメントだけでなく、固定binaryの到達branchを確認した。

1. 符号bitを取り、|y|≤Dを整数比較する（@0x26024..0x2607a）。
   |y|<Dは元bit値を返す（@0x261ac）、等号は元符号のzeroを返す（@0x2617b）。
   yがsubnormal/±0でもこのearly returnでexact。負のyはこのcaseだけである。
2. D<y<2Dでは両入力はnormal、exponent差nは0または1。
   significand M_y,M_Dを53-bit整数として取り出す（@0x26080..0x260f8）。
   n0ならM_y−M_Dを取る。n1ならy<2DよりM_y<M_Dなので、一度2M_yへshiftし最後にM_Dを引く。
   対応する命令は@0x2610bのsub、@0x26113の2倍、@0x2611b..0x26125の最後のsub/選択。
3. 得られる整数Rは0<R<M_D<2^53、残差はR·2^(2−52)。最小非zero残差≥2^−50なのでnormal。
   normalize（@0x2612a..0x2614c）とexponent/signのbit再構成（@0x26155..0x26178）はexact。
   64-bit整数のoverflow、丸め、subnormal出力分岐はこのcaseにはない。

従ってこのbinary・引数域のfmodは**exactなtruncation余り**を返す。
IEEEの一般transcendental精度やランダムテスト最大誤差を根拠にしていない。
NumPy scalar `%`の後続符号補正も、配列版と同じく負の余りにDを加え、zeroには正の除数の符号を付ける。
この加算は一般に丸め得るので、exact fmodからexact Python remainderと飛躍しない。

上のkernel/binding、binary64 RN・gradual underflow、x域を前提とすれば、前追補の
`d_T(wrap64(x),x)<64u`が適用できる。さらに全kのwrap前誤差<64uと合わせて
**m≥1、q≤2^17、principal∈[−p64,p64]に対しc_m≤128u**を構成できる。
m0はa₀へ最初のwrapを含めればc₀=0という既存会計でよい。
ここではH4にcを代入せず、production a/c/d配列はnullを維持する。

binaryでの再確認用コマンドは`objdump -d --disassemble=__fmod_finite /usr/lib/x86_64-linux-gnu/libm.so.6`。
machine recordに当該code bytesのhashとsymbol address/sizeを保存した。
前提が変わったkernelや別CPU/binaryへこの証明を移送しない。

## complex exp・atan2の残存証明

[glibc2.35 cexp SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/math/s_cexp_template.c)では、
有限pure-imaginary入力の実部は±0なのでexp(real)=1となり、実質的にsincosの結果を返す。
|Δ|≤DBL_MINでは(sin,cos)=(Δ,1)の小引数分岐がある。
しかし通常域のsincosのrange reduction・lookup table・polynomial・丸め誤差の証明は未取得。
実Δ域[−2p64,2p64]（基本減算丸めも含む）に限定すれば巨大引数reductionを避け得るが、
対象域内全branchの保証はなお必要。超越関数の誤差を0としない。

[atan2 SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/ieee754/dbl-64/e_atan2.c)は
冒頭で約0.524 ulpを**random sampling**由来と明記している。
[FMA版SOURCE](https://github.com/bminor/glibc/blob/glibc-2.35/sysdeps/x86_64/fpu/multiarch/e_atan2-fma.c)も同SOURCEをincludeする。
個別branchのMax ULPコメントやこの値を検証済み一様契約として採用できない。
比の極小域、四象限、signed zero、table index、FMA実演算に対する証明が別途必要。
[GNUの精度説明](https://sourceware.org/glibc/manual/latest/html_node/Errors-in-Math-Functions.html)も
accuracy goalと例外/bugの可能性を区別しており、test thresholdや数ulpという目標を保証定数として転用しない。
最新manualは方針の参考で、2.35 binaryの証明ではない。

得られたe_exp<1とe_angleに対しては、既存Dと同様に

    d_m ≤ 8u + 2^−1074 + asin(e_exp) + e_angle

という接続が可能（principal range・有限性を含む契約が必要）。これらへ未証明の数値を代入しない。
source存在・実束縛・少数のAcb参照照合と、この一様証明は区別する。
現状はd_m=null。原SOURCEを変更してcrlibm等へ差し替えることもしない。

## 全候補・全履歴の縮約と打切り

前追補の全root対応を再利用する。input位相のseamはlabelの巡回置換で扱い、全kに共通するc_m<π/qなら
候補間隔≥2π/q−2c_m>0で欠落/重複なし。全距離に共通のd_mがあれば、
B_m=b_m+a_mに対して

    π−[2B_(m−1)+B_m+q_m(c_(m−1)+c_m+d_m)]>0

が全誤候補を同時に排除する。旧adapterのB_m+q_mc_m<πも保持する。
これは全距離の完全な大小順序の保存を要求するものではなく、正解と全誤候補のstrict順序を保証する。
近傍2候補だけへの縮約はd_m取得前には正当化できない。

有限集合法では、各roundのa_mはexact成功count pair集合S_mを全被覆すればよく、履歴の積は不要。
新たに見つけたcallerでは各点のcmath.phaseを比較する必要がある。
cは上の構造的補題で全root列挙を避けられる。
一方、dを未証明のまま全距離を列挙すると、粗い組合せ上界は
Σ|S_(m−1)||S_m|q_(m−1)q_mで、全履歴列挙は∏|S_m|。
前roundの正解が帰納的に既証明なら到達estimate集合を縮め得るが、証明したいbranchを先に仮定しない。
区間被覆も数学関数の値域だけでなく、**実際のkernelの丸め誤差・制御分岐を全点で覆う証拠**が必要である。
H4のS_mやcall数は今回算出せず、巨大列挙を次段の既定路線にはしない。

取得資料にはこの限定域で使えるsincos/atan2の完全誤差証明がなく、単に再び精度testを増やしても
全域保証にはならない。これを今回の具体的な打切り理由とする。
必要な最小証拠は、(i)対象FMA atan2とsincosの限定域全branch/table/丸めの証明、
(ii)そのkernelと固定runtimeの対応、(iii)正式対象callerの契約である。
新しい検証基盤・別decoderの採択・停止済みvirtual float guard/BLAS系列の再開へ広げない。

## 人工検証・保全・次段

新規[test_input_kernel_connection.py](test_input_kernel_connection.py)10件＋既存44件、計54件PASS。
新規はN3の全count、既存callback→cmath→scalar接続、callback型、zero/±π、resolved loop/GOT、FP制御の非変更、
fmod境界とsubnormal、実distanceのcomplex128 shape、特殊境界、契約不足時のUNKNOWNを確認した。
既存のstrict不足・tie・誤branch/遠い候補の誤距離・全小prefix被覆も再利用。人工Arbは192 bit、旧上界照合384 bit。
これらはH4の数値評価・libm全域精度保証ではない。

```sh
cd docs/research/rpe_three_gaps_preparation_20261104
python -m unittest -v test_input_kernel_connection test_decoder_numeric_preparation test_three_gaps
```

必須固定GitHub成果13ファイルを実際に取得しlocal bytesとGit blob照合した。
A/C結果`61f569c...`、SOURCE`d29a869...`、branch証明`6942e51...`、結果`89cb94f...`、包含`d2e7a7d...`を保全する。
既存A/Cのideal IID成功確率・保存DF sector reference・理想energy結果を再解釈して実装保証へ昇格しない。
旧SOURCE・raw・grant・ledger・marker・STOP、他branch/worktree、共有環境、Codex C/H8には書き込んでいない。
正式共通運用指示2026-10-25の§8.1、§8.4、§10をread-onlyで確認し、不要な承認証明や資源上限は追加していない。

**次の担当・境界**：ユーザーが証拠取得の継続を指示した場合のみ、Codexが上記のkernel誤差証明とcaller契約を確定する。
全a/c/d・全候補対応・実用的計算量が揃った場合に限り、Codexが旧36軸・A/C・ideal marginを再利用する科学SOURCEを準備する。
その時点でinput commit/path/hash、precision、call範囲、strict条件、STOPを固定・公開する。
現時点はその条件が揃わず、H4科学SOURCE・実行計画は未準備。架空のcall数/precisionを置かない。
新科学計算はSOURCE公開後のユーザーの別途明示承認が必要。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`、implementation energy=nullを維持。
**公開照合後mandatory STOP。追加技術系列も科学実行も自動継続しない。**
