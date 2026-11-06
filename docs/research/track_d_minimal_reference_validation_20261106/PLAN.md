# Track D：固定人工pathのReference correctness・6 wrapper比較

2026-11-06 JST。起点 `71ac8b96c833c09d17ae6a02118471acedd781df`、[静的照合](../track_d_reference_control_feasibility_20261106/MINIMAL_VALIDATION_STATIC_CHECK.md)の固定scopeを実装・実行する。既存研究のSTOPを解除せず、既存SOURCE/結果/grantを変更しない。本書と新SOURCEを実行前にcommit/pushする。承認証明用の記録・grantは作らない。

## 対象と比較条件

対象は既存 `native-partial-closed:B` のみ。2 system modes、one-body eigenvalues (.7,−.2)、squared DF η=(.4,−.9)、λ=−.6、framesは既存`artificial_inputs()`のu/v。T=.8、q=2、scalar=.17、K=0、τ=.4、同frameのZplus→ZZminusを各stepへ置く。分子・geometry・AO basis・分子DF rank・L_Dは非該当。Hamiltonian/state生成、sampling、shot、T合成、全RPE会計は対象外。この人工τ/outer時間は物理的PR scheduleとして認定されていない。

V=exp(−icT) S²、χ=−cT、B=S、A=exp(−icT)Sを固定。Referenceはstate bits 0/1、vacuum-reference bits 2/3、ancilla 4、CSWAP cascade前後2層、A†/B、P(χ)。A†は実inverse。cat、reset、post-selectionは使わない。

| arm | 構成 |
|---|---|
| ordinary_shared_fused | diagonal-only ordinary control。RTEは既存builderと同じbasis共有B |
| symmetric_shared_fused | D前半uncontrolled、逆順後半directional、中央tail ordinary。RTE共有B |
| reference | 同一Vをcomplete-step境界で分割。各laneは無制御Dと同じ共有RTEを使い、cutを跨ぐ融合は与えない |

対照のD0 step境界は同じblock・basisなので、両armともordinary full-time blockへ融合する。ordinary+ordinaryでは各branchの時間和、directional+uncontrolledではbranch0が−t+t=0、branch1がt+tとなる既知の同block恒等式を使う。中央modeを無視した合算ではない。この限定adapterの融合出力は、元native出力および独立な物理Vと数値照合する。旧native builderを改変・再認定しない。新しい最適化探索は行わない。

## SOURCE・再利用・正しさ

- [新module](../../../src/trotterlib/track_d_minimal_reference.py)、[runner](../../../scripts/run_track_d_minimal_reference.py)、[明示check harness](../../../tests/track_d_minimal_reference_checks.py)。後者は通常pytestで自動実行しない。
- 既存supportのfixture、Fock minor lift、basis/primitive、event builder、counterの`assemble`、native provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`を再利用する。既存科学runner/全test suiteは実行しない。
- 物理Vの独立参照は既存Fock minor liftとoccupation diagonalから構築。full-controlはdiag(I,V)と位相込みで比較する。Referenceは原著の2分枝operatorと、vacuum/ancilla初期0に制限したsystem上の4×4 X/Y測定演算子を比較する。これにより特定input stateだけの偶然一致を避ける。
- absolute maximum entry toleranceは全て `3e-12`、rtol=0、phase fittingなし。P(χ)省略をX/Y両軸の負対照にし、不一致を検出することを要求する。
- PASSは本人工pathのoperator/測定同値性のみ。K>0、tail identity、product、任意Pauli、分子、一般lawの検証ではない。

## compile・性能・成果物

correctness PASS後だけ3 arm×X/Yの **6 wrapperを各1回compile**。H/U/(Sdg)/H/measurementまでを含み、初期state準備は全armから除外する。Reference追加qubits、CSWAP、補正P、cutの融合損失を含む。

全armで既存F2 `compiled_wrapper`/`CompilerProfile`を使う。Qiskit1.3.0、Qiskit Nature0.7.2、basis rz/sx/x/cx、opt1、seed17、approximation_degree=1、qubits_initially_zero=False、backend/couplingなし。Qiskit公式definitionのphase保持loweringを全armに同じく適用する。compile後も全unitary（同registerのrawとの比較）と誘導X/Y演算子を同じ許容差で確認する。

利用可能CPUは32 logical、RAM約49 GiB availableを準備時に確認。GPU driverは利用不能だが、最大32×32の行列と5-qubit回路ではCPUが適する。fixture/小行列を再利用し、6 independent compileを利用可能CPU数とtask数に応じたthread poolで実行する。小行列のBLASと入れ子parallelを各worker内で抑えoversubscriptionを避ける。CPU/RAM/wallのjob全体capは追加しない。既存native fixtureの2000命令guardは保持する。速度向上率は測定比較していないため主張しない。

実行コマンド：既存Qiskit環境のPythonで `python -B scripts/run_track_d_minimal_reference.py --output artifacts/track_d_minimal_reference_validation/2026-11-06`。outputはfresh directory、resume/retryなし。`result.json`に実行SOURCE SHA、入力event、比較operator、各check、環境・実資源・6件のRZ/CX/depth/size/qubitsを保存し、raw/compiled各6回路をQPYで保存する。資源指標は合算せず、Nや総費用へ外挿しない。

## STOP

correctness失敗、phase/SOURCE不明、対応不能な融合、compiler非互換なら即STOPして記録する。別word・別cut・閾値変更・自動retryへ進まない。PASS時も6件の比較、報告、公開取得照合後にmandatory STOP。重要科学レビュー、研究採択、追加計算は別の許可事項である。
