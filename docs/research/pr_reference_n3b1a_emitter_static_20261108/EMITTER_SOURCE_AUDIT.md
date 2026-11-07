# N3-B1a：既存emitterとQiskit 1.3.0のSOURCE監査

2026-11-08 JST。[REPORT](REPORT.md)。基準`9c60c43135666a9ab5497a43a4b1695ab371650f`。読取・静的追跡のみで、circuit／Operator／guardを実行していない。

## 1. 固定SOURCE

Qiskit tag1.3.0はtag object`4812d6e7bf704ca31aea33003e37eb09fc876af1`からcommit **`896d1c8b72888a52a46081a6d1f4de8c56db8aea`**へ対応した。既存venv内のquantumcircuit.py、h/s/t/x.py、qpy binary_io circuits.py/value.py、parameterexpression.pyの**Python8ファイル**はこのcommitのGit blobとbyte一致した。Rustについては同commitのSOURCEを確認したのであって、wheel機械語と全依存の形式的同値性監査ではない。

| 一次SOURCE | 確認箇所 |
|---|---|
| [quantumcircuit.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/quantumcircuit.py) | inverse L1635付近、compose L1811/L2050、global_phase L4133付近 |
| [circuit_data.rs](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/circuit_data.rs) | set_global_phase L848、Floatのrem_euclid L864付近 |
| [operations.rs](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/operations.rs) | ParamとPython→Float／ParameterExpression変換 |
| [h.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/h.py)、[s.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/s.py)、[t.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/t.py)、[x.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/x.py) | named gateの行列意味、inverse、別の定義へdecomposeする経路 |
| [QPY circuits.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/qpy/binary_io/circuits.py)／[value.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/qpy/binary_io/value.py) | numeric phaseのdouble保存と読込constructor、ParameterExpression replay |

定数の由来として[CPython3.12.3 pymath.h](https://github.com/python/cpython/blob/v3.12.3/Include/pymath.h)／[mathmodule.c](https://github.com/python/cpython/blob/v3.12.3/Modules/mathmodule.c)のmath.pi定義も参照した。Rustのfloat定数／remainderの意味は[Rust1.70 core f64](https://github.com/rust-lang/rust/blob/1.70.0/library/core/src/num/f64.rs)／[std f64](https://github.com/rust-lang/rust/blob/1.70.0/library/std/src/f64.rs)を参照した。これはwheelのbuild compilerがRust1.70だったという認定ではなく、標準f64意味論の限定確認である。

## 2. 保存wordからの経路

[lower_native](../../../src/trottertracks/track_c2_b_verify.py)は、native RZのabsolute literal keyとallocationのexponentで保存recordを引く。wordをwritten matrix productとして解釈し、文字列を**逆順**に時系列emitする。T→TGate、t→TdgGate、H/S/X→対応named gate、W→local.global_phaseへのmath.pi/4加算。localは0 phaseから始まる。負角ならlocal全体をinverseし、その後out.composeへ渡す。

元nativeのglobal_phaseはoutへ引き継ぐ。SXはHSH、X／CXはそのnamed gateとして追加する。SX=HSHの位相込み恒等式を保持し、新controlや別のnative compileは後置しない。lower_native自体はmeasure/reset/barrierを受けないので、whole接続時は既存のreset/readout取扱いを別に保つ。

## 3. Written product・inverse

[exact_word_matrix](../../../src/trottertracks/track_c_h4_synthesis.py)はpygridsynthのexact cyclotomic行列を使いWを含める。tをTの7乗へ展開するのは行列評価のためで、emitterのT†を7物理Tへ数える操作ではない。

QiskitのQuantumCircuit.inverseは、global_phaseを否定した新circuitを作り、命令列を逆順にして各operation.inverseを置く。named gateのSOURCEではH→H、X→X、S↔Sdg、T↔Tdg。これらの逆演算をgenericなfloat角の再合成へ置き換えない。

連結word全体を一度にinverseすることで、chunk順序と文字順序の両方が反転する。各chunkだけを逆にして元のchunk順でcomposeする方法にはしない。wordレベルのactual adjointと、float global_phase正規化に伴う小さいscalar差は分けて扱う。

N3-Aの非負整数bit分解では各基本指数を高々1回用いるので、b=36、指数-36…6ならchunk数は43以下になる。近似基本wordは一般には可換でない。指数順を一度固定し、そのwritten productとemitterへ渡す文字列を一致させる必要がある。今回の静的調査では、未実装providerの指数順を新たに採択していない。

## 4. Numeric phaseの意味

Pythonのglobal_phase setterはCircuitDataへ委譲する。Rust側ではParam::Floatを`angle.rem_euclid(2*PI)`へ正規化する。Param::ParameterExpressionは別branchで保持する。現lower_nativeがWに使うのはnumeric float branchである。

composeの該当既定経路は、dest.global_phaseへother.global_phaseを加え、その後gateをwireへ写像して追加する。従って、**localの位相だけでなくcomposeの加算とsetterも**emitter誤差に入る。inverseで位相を否定するとsetterが再度働く。

Rust rem_euclidは負の剰余にdivisorを加える。丸めにより結果がdivisorになる境界があるため、一般のphaseが必ず[0,tau)にあると仮定せず、証明は[0,tau]を覆う。真の2piとbinary periodの差も捨てない。[PHASE_ERROR_BOUND](PHASE_ERROR_BOUND.md)に局所8状態条件と有限回の後続操作を分けた証明を示す。

## 5. Serializationとsymbolic phase

numeric QPYはglobal_phaseをvalue.dumps_value／loads_valueでdoubleとして保存・復元する。このpayloadのbit保存は、exact piへ戻す処理ではない。読込constructorがglobal_phase setterを通るため、phi=tauの境界を含む再正規化を別途評価する。QPY以外の出力や再compileを同じ証明へ含めない。

整数W指数をmetadataへ持たせる設計は可能だが、QPYのfloat phaseだけから元のexact Wを復元したことにはならない。sidecarを使うなら、word/hashとinteger Wを全consumer・inverse・compose・serializationで保つ契約が必要。

ParameterExpressionがsetterに保持されることだけで、exact symbolic piの完全な転送を資格しない。1.3のParameterExpressionは内部replay情報を持ち、QPY v13の経路もそれを使う。直接構築した定数式・変換・serializerの対応は未確認であり、「symbolicにすれば誤差0」という既定案にしない。今回そのadapterを実装／試験していない。

## 6. 有限数値診断と論理回路

named H/S/T等のideal意味論と、numpyの有限精度行列／Operatorで評価した数値を区別する。後者の行列積誤差を、物理gateやemitterの誤差と同じものとはしない。逆に、小さい有限診断誤差を任意wordへの保証にもしない。

[CT-1 word_matrices](../../../src/trottertracks/pr_reference_ct1.py)にはexact word／actual adjoint／±emitterの有限検証があり、数値toleranceはoperator certificateではないと明記されている。CT-4Bはこの資産を再利用し、local correctnessとcompressedなexact-word行列作用／価格会計を行った。q128 whole expanded CT circuitを実装・serializationして全経路を資格した結果ではない。

[CT-4B SOURCE](../../../src/trottertracks/pr_reference_ct4.py)と[CT-1 matrices_for](../../../src/trottertracks/pr_reference_ct1.py)の正負word行列・W台帳は再利用できる。新しい43-word連結やQPY経路への適合は、静的証明の前提と後続の必要最小限の検証へ対応させる。旧結果・correctnessを否定／上書きせず、成立範囲を広げない。
