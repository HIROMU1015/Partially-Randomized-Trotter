# 研究C：現P0の正常終了・Rust一様誤差の集中検証

2026-11-08 JST。**CASE D：現ENTRY E2/r16について両方とも未成立。ただし、終了性の一般的な十分根拠を反証し、数値誤差の未取得契約を具体化した。**
同じreadinessの再整理で終えず、固定P0の実例・iteration単位の条件付き上限・全branchの誤差分解を得た。
E2の1000回超過反例を見つけたわけではない。実H4計算・保存科学値の数値再解析・実H4 build/compile・
実分子Arb包囲・RPE/資源計算・compiler/target変更・重要GPT科学レビューは0。

## 1. 固定条件と再利用

起点はcommit `95a828cf2ad099e8ff7a6bbe470be1fc94cfb79b`、
[前段CASE C](research_c_entry_e2_r16_p0_uniform_feasibility_20261108.md)。
Linear H4/隣接1.00 Å/STO-3G、ENTRY `H4_100_C0_ENTRY_g003_budget64`、number-CDF8/元順LD2、
F2..F7の線形Zを元κへ移しframe identity/ZZを保持する既存version、partial ordered S2/s1/r16/K0、δ=h=.1、m0..12。
Nα=Nβ=2/sector36、system256/control込み512、own-DF ground oracle、準備費除外。
Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1/qubits_initially_zero=false、
MPFR160 declared phase、phase fitなし、actual raw/compiled inverse、head/X/Y、fenced serialを維持する。

科学SOURCEは`cfc64b70861a6482dddd042da8a110abebd6ad7a`、
code SOURCE SHA256は`6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc`。
候補/J/U/元κ/scalar/物理入力は既存[identity記録](../../artifacts/research_c_entry_e2_all_law/2026-11-08/technical/readiness.json)を再利用し、91保存pathのbyte照合を行う。
上記条件の保存**条件付き総RZ 75,621,395,471**は引用のみで、今回再計算せず精度保証付き価格へ昇格させない。
CA/DF reserve/失敗確率政策も前段のまま。

`c_entry_e2_p0_feasibility`のsame-target連鎖、Gram/norm-growth、exact Euler lemma、grammar ceilingを再利用。
`c_pr_aware_df_c2a`のmonotone/symbolic template/pass graph、`c_pr_aware_df_zsxx_uniform`のproduct rateとcomposer、
既存full512/native/complete-column/actual inverseのscope条件を維持する。
旧Arb384/Stage A/fixed-wordの別candidate・再配置前r32数値は流用も再実行もしない。

## 2. 固定sourceと利用環境

Qiskit基準commitは **`896d1c8b72888a52a46081a6d1f4de8c56db8aea`**。
今回もEuler Rust・inverse cancellation Rustの保存本文、installed builtin_plugins/flow_controllers本文と公式固定commitの一致を確認した。
Optimize1q Python、FixedPoint、BasisTranslator/unroll、phase wrapperは前段のinstalled source hashと照合する。
別versionの挙動で保証を補っていない。

- [builtin_plugins opt1](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L535)
- [DoWhileController](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/passmanager/flow_controllers.py#L89)
- [Euler/ZSXX Rust](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs)
- [inverse cancellation Rust](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/inverse_cancellation.rs)

Read-only ELF確認でinstalled `_accelerate.abi3.so`がlibm.so.6へリンクし、atan2/hypot/exp/sin/cos/sincos/fmod等の
GLIBC symbolを参照することを確認した。Binary/source hashと実libm pathは
[technical.json](../../artifacts/research_c_entry_e2_p0_focused/2026-11-08/technical/technical.json)に保存する。
現在のruntimeはglibc2.39。`GLIBC_2.2.5`はsymbol versionでありruntimeの版ではない。
旧資料のglibc2.35を現在の精度保証へ代用しない。Symbol表全体のacos等を、active ZSXXの処理と混同しない。
再現build・wheelと特定Rust toolchainの同一性・全libm精度契約を証明したという主張ではない。

## 3. 終了性：iteration単位の新しい上限

### 実際のloop

初期Depth/Size測定後、各iterationは

`Optimize1qGatesDecomposition → InverseCancellation → GatesInBasis → 条件付きtranslation → Depth/FixedPoint → Size/FixedPoint`。

全入力・出力がnative basis内でtarget=None/ZSXXのみなら、Rustが受理する1Q置換はstrict shorterで、出力≤5gate。
取消は削除のみで、loop内translationはidle。この条件でsizeが同じならgate列の変形は起きていない。
各関数が決定的なら、CX skeletonも同じままのrunが後から突然変わることはない。
Depth同一だけではsizeが減ってloopを続ける。両方の同一を1回確認して初めて返る。
「固定点がいつか存在する」と「1000回内にそれを確認する」は別。

### CX削除epoch lemma

Cをtranslation後のCX数上限、Kを**CXが一つでも削除されるiterationの数**とする。
CXは新生成されず2個以上ずつ減るので`K<=floor(C/2)`。
各CX削除iterationの間には、CX skeletonが不変な区間がある。

その区間で、各qubitのCX間のmaximal 1Q runは独立に処理される。
最初の1回で全run長≤5になり、その後、各active runは受理される度にstrictに短くなる。
変化しなかったrunが次回wakeしないなら、全runが同時に進むため、追加の長さ低下は最大5回、
最後に変化なしを確認する1回で足りる。したがって**CX-free区間≤7 iteration**。

よって条件付きで

`I <= K + 7(K+1) = 8K+7 <= 8 floor(C/2)+7`。

前段のC≤4,752を代入した構文上限は **I≤19,015**。N+1≤67,747より改善するが1000未満ではない。
E2固有の`K<=124`を全wordで示せれば`I<=999`となる。このK上限は**未取得**。
初回translationやpre-loop取消はこのIに含めず、その後のCX数を保守的Cで覆っている。

成立条件は、有限かつ正常なpass呼出し、fixed native graph、calibration/control flowなし、
全runの処理、同じgate値に対する再合成の決定性、global phaseがrun matrixへ入らないこと。
これは数値精度未指定のAPIだけから得られる無条件証明ではない。
F64/libmの同じ入力への決定性や丸め環境が変われば、no-wakeup部分も別途正当化が必要。
既存sourceのstrict decreaseだけはその条件付き精度モデルから独立して使える。

## 4. 粗いnative grammarだけの終了証明は実P0で成立しない

3qubitでa=CX(0,1)、b=CX(1,2)を交互にn個並べたword W_nを作り、
**人工回路 `W_n reverse(W_n)`** を既存MPFR wrapperを含むunchanged P0へ渡した。
全gateが自己逆なので数学的にはidentity。分子入力・保存E2 word・parameter生成は使わない。

RustはCX run/partitionをsnapshotで作ってから削除するため、最内側の同じqargsの一組を消しても、
次の外側の組は同じpassで再収集されない。交互qargsは他の隣接同一組を作らない。
この構造に対してpre-loop取消が1組、以後各iterationが1組を消す。

| 人工n | 初期CX数 | Optimize1q実行回数 | 結果 |
|---:|---:|---:|---|
| 2 / 7 / 12 | 4 / 14 / 24 | 2 / 7 / 12 | 正常、size0 |
| 1000 | 2000 | 1000 | 正常、size0 |
| 1001 | 2002 | 1000 | **TranspilerError: Maximum iteration reached. max_iteration=1000** |

n=1001でも最後の取消直後sizeは0だった。しかし前回のsizeは2なのでFixedPointはfalseであり、
次の確認iterationが必要になる。Guardは「最後にidentityを得たら返る」条件ではない。
Callbackは観測だけで、guard値・pass graph・compiler政策を変更していない。

2002 CXは4,752以下、総node数も67,746以下なので、**basis閉包＋qubit上限＋node/CX数だけ**のクラスは
正常終了を保証しないことが実装上も確認できた。
ただし、このpalindromeは「16 signed ZZ＋固定Gaussian/head」という**E2の正確なemitter grammarではない**。
E2失敗例や現P0でのE2保証不可能性とは主張しない。
残る終了性課題は、E2 grammar固有の取消依存深さ/epoch Kを抑えることへ縮小する。
固定alphabetが有限というだけではその証明にならない。

## 5. 実Rust誤差：active処理と除外できた課題

`params_zyz_inner`はdet、det.arg/2、`2 atan2(abs(c),abs(a))`、d.arg/c.arg、φ/λ加減算。
ZSXXはU1X phaseへの変換後、`circuit_psx_gen`とpfunを通る。
[num-complex0.4.6](https://github.com/rust-num/num-complex/blob/num-complex-0.4.6/src/lib.rs)ではnormはhypot、
argはatan2、complex expはexp(real)とcos/sinを呼ぶ。SX/Xはdyadic。
Active ZSXXにsqrt(det)やacosを使う別Euler basisを混入しない。

| 処理 | 取得した評価・残る契約 |
|---|---|
| RZ node complex exp | 逐次run積への入力誤差a。Exp(0)、sin/cos、half-angle/underflowを含む一様仕様が必要 |
| Complex matmul/det | 既存IEEE RN/gradual underflow/FMAまたはseparateモデルを再利用。実装順序固定。有限性・実platform適合は条件 |
| Hypot/径方向atan2 | 入力normの正の下限を使い、(0,0)の特異性を除ける。必要な一様abs/angle契約は未取得 |
| c/dのarg | 小振幅を切り分け、角度の小さい絶対誤差を要求しない。大振幅annulus上だけphasor契約を要求 |
| det.arg | det modulusの正の下限を使う。Arg cutはphasor距離で扱う |
| Flip判定 | **精度上界は不要**。どちらを選んでも全角度で厳密恒等式。判定値の有限性/例外排除は必要 |
| 0/1 SX、π近傍collapse、X、RZ drop | 閾値によるoperator誤差を下記で全branch合成。単にatolをζとしない |
| Modulo/phase | Pの意味で周期距離を評価し、RZ単独を2π周期としない。DAG/MPFR final phaseは別 |

Rustの[1.83.0標準source](https://github.com/rust-lang/rust/blob/1.83.0/library/std/src/f64.rs)のexp/hypot/atan2等は
precisionを未指定としている。これはinstalled wheelのtoolchainが1.83.0という主張ではなく、
**Rust言語/API名だけでは一様ULP定数が与えられない**ことの一次資料。
Installed ELF/libmの確認だけでも必要なoperator精度仕様は得られない。
数ULP、有限probe最大値、glibcの精度目標を数学的上界として採用していない。

さらに現在と同じglibc2.39の一次sourceを確認した。
[atan2](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/ieee754/dbl-64/e_atan2.c#L56)の約0.524 ULPという注記は
**“based on random sampling”**と明記されており、その値を全domain契約へ昇格できない。
[hypot](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/ieee754/dbl-64/e_hypot.c#L29)の約0.792/0.948 ULPも期待値の記述で、FMA branchを区別する。
[sin/cos](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/ieee754/dbl-64/s_sin.c#L28)の約0.55 ULPはIEEE round-to-nearestを前提にした実装記述。
これらの数字を実際のwheel/dispatch/全入力の包囲として今回採択していない。
[fmod](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/ieee754/dbl-64/e_fmod.c)はbit/mantissa剰余経路を持つので、
moduloの問題をatan2精度と一括りにせず、別の有限算術義務として扱える。
ただしELFはfmod@GLIBC_2.2.5を参照し、読んだsourceには2.38向けdefault symbol定義もある。
Compat symbol/IFUNC dispatchを含む実経路の同一性は未確認であり、これを理由にμ=0とはしない。

## 6. ωを評価する具体的な条件付き構成

以下は前段exact Euler lemmaを再導出するものではなく、残った
`ω=sup ||V_f64(M)−F(M)||`の評価を二つの有限な契約群へ分割する。
全値はoperator単位でphase-sensitive。実E2についての値はnull。

### 6.1 抽出段：小振幅argを保証対象から外す

Domainは`exists unitary G: ||M−G||<=η<1/4`、det coset4種・全arg cut・zero entryを含む。
r_min=1−η、d_min=(1−η)²、κ=(2η+η²)/(1−η)。0<ρ≤1を小振幅の閾値として選ぶ。

- h：各hypotが|a|/|c|から外れる絶対誤差。`2h<r_min`。
- t：nonnegative radial inputsのatan2 angle誤差。
- d：computed complex detとexact det Mの距離。`d<d_min`。
- g：ρ以上のentry modulusを持つannulusで、argの**単位phasor**距離。
- g_det：computed detの正のmodulus領域でのarg phasor距離。
- e_par：φ/λ/U1X phase等を生成する基本算術の、同じphasor reconstructionへのoperator誤差。Libm・次のgenerator・DAGは含めない。

すると

`Δθ <= 2t + 4h/(r_min−2h)`、
`w_c <= min(2,max(2ρ/r_min,g))`、
`w_d <= min(2,max(2(ρ+η+κ),g))`、
`e_det <= min(2,2d/(d_min−d)+g_det)`、
`ω_extract <= Δθ/2 + e_det + 2w_c + 2w_d + e_par`。

理由：cの重みはC=|c|/rで、小振幅ならphasor差が最大2でも2ρ/r_min以下。
dの重みA=|a|/rは、前段のunitary Qにおけるd0と`|d−d0|<=η+κ`から、
dが小さいときA≤ρ+η+κ。大振幅側は契約gを使える。
Detの正規化phasorは非zero下限による摂動評価を使う。
Fの2個の対角・2個の非対角へのphasor寄与を三角不等式で合わせ、θの変化はunitary rotationの1/2-Lipschitzで抑える。
Arg cutで角度が約2π跳んでも、同じdet.argからφ/phaseを生成する相関を保持したphasor式は破綻しない。
それを無視して独立のphase fitを行うことは認めない。

これでarg(0)や微小c/dに「一様に小さい角度誤差」を要求する根本的な障害を避けられる。
ただしρを小さくすれば良い定数が自動的に得られるわけではなく、gの全annulus仕様が必要。
Actual libm/f64のh,t,d,g,g_det,e_parを取得したとは扱わない。

### 6.2 ZSXX生成段：全branchを有限の誤差項へ落とす

U1X parameter(t,p,l,gamma)の厳密意味は

`U(t,p,l,gamma)=exp(i[gamma+(t+p+l)/2]) RZ(p) RY(t) RZ(l)`。

Sourceのflip更新は`(t,p,l,gamma)→(−t,p+π,l+π,gamma+t)`。
これは**全角度でUを保存する恒等式**であり、mod(l+π)またはmod(p)のguardが正確かどうかは
operator正しさの前提ではない。誤差は実際のπ/基本算術だけへ入る。判定の精度を新たなlibm課題として課金しない。

Sourceが近似する他の箇所は次のとおり。

| Branch | 厳密算術でのoperator摂動 |
|---|---|
| t≈0のzero SX | 固定gammaでt→0とした差≤|t| |
| t≈π/2のone SX | 差≤|t−π/2| |
| t≈πのphi−lam/lam→0 | 差≤|t−π|。Lamの大きさに依存する発散なし |
| SX P(t') SX→X | P(t')とIの距離で評価。t'の真の2π周期距離を使用 |
| 各pfunのRZ drop | pfun全体は`exp(iφ/2)RZ(φ)=P(φ)`。省略差≤|φ| |

τ0はsourceのbinary64 `1e-12`そのもの。Predicate減算誤差b、π中心の差δπ、
全reachable引数に対する`||P(mod_f64(x))−P(x)||<=μ`、
generatorの未包含基本算術/π使用によるphase付きoperator誤差e_genを契約とする。
e_genはmod呼出し、branchの意味論的近似、DAG/final phaseを含めない。

全branchの保守的包絡は

`ω_generate <= 5τ0 + 2b + 2δπ + 4μ + e_gen`。

一般branchでπ-collapse一つ、X選択一つ、pfun省略最大3つなので5τ0。
意味のあるmod呼出しはpfun最大3＋X選択1。Flip用の二つは値の精度を課金しない。
Endpoint clamp、fmod、rem_euclid、binary64周期と数学的周期の差はμに含む。
Pとしての周期対応なので、RZを2πでwrapして符号だけ落とす誤りをしない。
Output arithmeticで使うπの誤差はe_gen、branchを採る中心のずれはδπという別の作用を一度ずつ数える。
Zero/one branchの合計はこの一般branch包絡より小さい。

e_gen/e_parは有限な加減算・half/double・phase加算のshadow replayにより、
全reachable parameter rangeとIEEEモデルから評価できる種類の義務へ縮小した。
ただしその全operation/rangeの誤差証明を終えたという表示はしない。
Finite人工入力でRust出力が一致することは、この全域契約の代わりにはならない。

### 6.3 Composerと二重計上

前段のη_N<1/4、すなわち`Np<1/5`とΣL≤6Nにより、
product＋exact F部分を`E_exact=(1+k(η_N))6Np/(1−Np)`とする既存helperを呼ぶ。

Split経路は

`ε_P0 <= ε_translation + E_exact + N(ω_extract+ω_generate) + ε_DAG/final-phase`。

Whole ζを得た経路では旧式
`ε_translation+6Np/(1−Np)+Nζ+ε_phase`をそのまま使い、splitのexact F/ωを加えない。
両経路の同時入力を新helperが拒否する。
次のrunのRZ matrix評価誤差はそのrunのa/pで会計し、gate列の厳密意味を比較する現在のωに重ねない。
Local generator phaseはω_generate、DAGの加算・normalization・MPFR bind/castは最後のphase項。
Raw→literal、actual inverse、head/X/Y、reference B/U、IID-grid/PCG64、13round精度は別残件。

## 7. 成立範囲・未取得根拠

| 層 | 今回の結論 |
|---|---|
| 数学的/構造的に成立 | Flip恒等式、weighted phasorによる小振幅arg回避、branch意味論の誤差分割、粗いnative count根拠に対する人工反例 |
| 明示条件付き | 決定的で正常なpassの8K+7上限、全domain libm/IEEE契約下のω式、exact-F split composer |
| 未取得・主張不可 | E2のK≤124等のguard内成功、installed pathの全域a/h/t/g/μ・基本算術/phase証明、実H4での上界有用性・全law・CA |

残件は次の少数の証明義務に集約できる。

1. **E2固有の取消依存深さ**：Gaussian/head/16 eventの実grammarをabstractに扱い、CX削除epoch上限または別のguard内成功証明を得る。
   Native gate数だけの議論は上記反例で不足と確定した。
2. **実際に使用する数値関数の全域契約**：限定されたhypot/radial atan2/annulus phasor/RZ exp/phase-modの精度・有限性・必要な決定性。
   ELFやAPI名、精度目標、少数ULP観測だけでは証拠にならない。
3. **有限affine算術とphaseの会計完了**：e_par/e_gen/MPFR・final setterを全branchの実operation/rangeで包む。
   別basis・inactive HLS/acos等を不必要に追加しない。

現P0で不可能とは断定しない。一方、この三義務を閉じる新しい根拠がないまま
同じreadiness作業・H4有限probe・旧Arb/固定word再実行を繰り返しても研究上の進展はない。

## 8. 最小実装・人工回帰・保全

新規は[c_entry_e2_p0_focused.py](../../src/trottertracks/c_entry_e2_p0_focused.py)と
[人工test](../../tests/management/test_research_c_entry_e2_p0_focused.py)。
Epochのscalar評価、ωの契約入力preview、既存split composerへの薄い接続だけを追加した。
新compiler・Hamiltonian/Arb/RPE engine・科学runnerは作らず、全供給値を人工的に満たしても
actual_Rust_omega/actual_H4_bound/error_evidenceはnull、RPEはUNQUALIFIEDのまま。

新29＋前段187＝**216人工・回帰PASS**。
Fixed P0 guard境界、相互CXの取消依存、branch境界、zero/arg cut、P/RZ周期、controlled relative phase、
flip恒等式、欠損契約、誤差の二重計上拒否を確認した。
Arb/flint importを遮断し、SymPyはtest process内だけpure Python backend。
新実H4/保存科学数値の再解析/新Arb/RPE/価格/重要レビューは0。Local人工証拠でありCI/外部再現や独立科学レビューではない。
5 warningsは旧escape SyntaxWarning。詳細source/hash/testと技術STOPを一つのtechnical.jsonへまとめる。

独立branch `codex/research-c-entry-e2-r16-p0-termination-rust-uniform-20261108`、
worktree `/tmp/research-c-entry-e2-p0-focused-20261108`、base95a828cf。
正式指示§8.1/8.2/8.4/10/10.3を確認し、旧SOURCE/input/raw/結果/契約/grant/ledger/marker/STOPと元dirty checkoutを保持する。
共有venv/package/OS/cache/job設定を変更していない。人工回路は3qubit・最大2002 CX、実行は短く、不要な大規模列挙や資源capを設けない。

## 9. 次の研究判断・mandatory STOP

次の担当は**GPT/ユーザーの研究方針判断**。
現P0を継続するなら、上記三義務に限定した証明/実装精度仕様の取得へ投資するかを決める。
新しい一般的なreadinessの再調査や、全word列挙・価格追加標本を次の最小作業として提示しない。

代替compiler/fenceを将来比較するなら、(a)guard内の正常返却根拠、(b)phase込みの局所誤差仕様、
(c)event間融合の範囲、(d)whole-kernelのコスト、(e)same target/inverse/shellの維持を比較軸にする。
今回その政策を実装・正式採択しておらず、現P0とのコスト優位は未測定。
新政策の価格へ756億RZを対応付けることはできない。

現状、新科学実行の方法は完成していないため、science entry/family/call上限はnull。
方法成立後だけ、同じ固定SOURCE/inputと新評価実装を結び、必要なprimitive・数値定数・referenceを
固有fresh outputへone-shotで評価する契約を閉じる。入力差/未包囲branch/phase/guard/数値失敗でSTOP、retry/resumeなし。
旧38/27/86-call契約や承認を新実行へ転用しない。

成果物をcommitしてGitHubへ公開し、remote SHA/tree・本文・主要SOURCE/STOPを照合して終了する。
**Mandatory STOP。科学計算・新compiler・target採択・重要レビューへ自動進行しない。**
