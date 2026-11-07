# CT-2：strict合成とT上界の保証区分

2026-11-07 JST。固定研究参照 `6ac8e1aa3104d8563188612410c1a7533ec47072`。**静的調査のみ**。式の有限precisionへの数値代入、合成、word生成、guard実行、testは行っていない。

## 結論と保証の対象

| 問い | 今回確認できたこと | 資格 |
|---|---|---|
| strict RZの数学的存在 | RZはdet=1で、一般U(2)の任意scalar phaseとは異なる。Clifford+Tの位相/行列規約に原理的な障害を認めない | 数学的存在と固定backendの取得は別 |
| 構成的手法 | 数論法、Solovay–Kitaev（SK）に構成的手順がある | 下記前提付きの理論 |
| 明示T式 | 固定分母の解・正規形を前提に2k、または資格済みSK基底と定数を前提に再帰式 | `THEORETICAL_ONLY` |
| pygridsynth2.0.0で全角度取得 | 固定seed/探索/数値計算を含む完了保証は確認できない | `UNQUALIFIED` |
| 実際の返却wordのstrict誤差 | 本repoのexact-word＋外向き区間guardで個別資格する | CT-1の有限取得域は`PARTIAL_COVERAGE` |
| 固定backendの全support価格 | 理論式を当該backendの全返却価格へ移せない | `UNQUALIFIED` |

`CONSTRUCTIVE_BOUND_AVAILABLE`を、新Aの全support・固定実装に対して付与していない。以下は既知理論の条件整理とCodexによる技術的な適合性推論で、別backendの採択・新研究方式の提案実装ではない。

## RZ・phase・norm

数学的には、RZのdet=1は[SelingerのRemark26](https://arxiv.org/pdf/1212.6253v2)のstrict近似可能なdet条件を満たし、任意の実thetaと正epsilonに対してstrict誤差がepsilon未満のClifford+T wordが存在する。これは全supportにわたる存在の主張であり、固定seed/backendが有限時間・特定T上限でそのwordを取得するという主張ではない。

対象は実binary64 thetaを有理数として解釈した `RZ(theta)=diag(exp(-i theta/2), exp(i theta/2))`。`RZ(theta+2pi)=-RZ(theta)`なので、符号を落とす2pi key正規化は禁止。厳密4pi周期を使う場合さえ、binaryのperiod/remainderと真のpiの差を別途保証する必要がある。今回は縮約法を作らない。

native P→RZで既に許容された**joint全体の共通phase**と、制御前/branch間の相対phaseを分ける。strict wordのWを捨てたり、合成後に新controlを付けたりしない。負角は正角wordのactual adjointを使えば同じspectral誤差上界を持つ。戻りのSOURCE inverseをcompileしたnativeがforward nativeのinverseに一致するとは仮定しない。

本repoの[guard](../../../src/trottertracks/track_c_h4_synthesis.py)はFrobenius上界を使う。文献のspectral保証をetaとすると、2×2行列について `||A||F <= sqrt(2)||A||2`。したがって既存guard許容epsilon_Gへ移すには、`sqrt(2)*eta < epsilon_G`と古典計算誤差の余裕が必要。このnorm変換はCodexの適合性整理で、今回のprecision配分採択ではない。

## 一次文献で確認した境界

### Selinger：固定分母の数論法

[Efficient Clifford+T approximation, arXiv:1212.6253v2](https://arxiv.org/pdf/1212.6253v2)のTheorem22、Algorithm23、Fact27、§8.2–8.3を照合した。候補の存在はnorm方程式の可解性ではない。成功した分母指数kの行列を正規形化すればT≤2k。完了/期待効率には素数分布の仮定と、有限instanceで候補がない場合の扱いが残る。文献のseedingも固定backendには採択しない。一般行列のdeterminant制約はRemark26。

条件を明示するための**式の書換え**は次。`0<eta<=1/2`、

\[
 C=\tfrac52+2\log_2(1+\sqrt2),\qquad
 k_\eta=\lfloor C+2\log_2(1/\eta)\rfloor+1.
\]

Theorem22のstrictなkの不等式を満たすよう端数処理した。**当該kで必要なnorm解を得られること**と、exact synthesis後の正規形を条件に、

\[
 T_{\rm fixed-denominator}(\eta)\le 2k_\eta.
\]

これは、同定数を使えば全binary角で必ずnorm解が得られるという追加証明ではない。hypothesisの漸近確率を、有限precision・全角度の定量的下限へ自動変換しない。固定seedのpygridsynthの実wordにもこの式を適用しない。

### Ross–Selinger：探索とnear-optimality

[Optimal ancilla-free approximation, arXiv:1403.2975v3](https://arxiv.org/pdf/1403.2975v3)の§7–8を照合した。Proposition8.1は終了時のstrict correctness、8.2はfactoring oracle下の最適性。Hypothesis8.3下の8.8/8.11は期待near-optimality/期待効率。典型的な3log則は全角度の有限worst-case価格ではない。Conjecture8.10の4log対3logの角度別議論も、固定backendの一様上界を与える証明には使わない。

exactな候補列挙・完全なfactorization/norm解判定ができる数学的手順と、有限attemptで候補を捨てる実装を区別する。最小Tに関する存在/最適性の式は、そのまま実backendの取得word価格の上界にはならない。

### Dawson–Nielsen：SKのfallback証明枠

[The Solovay–Kitaev algorithm, quant-ph/0505030v2](https://arxiv.org/pdf/quant-ph/0505030v2)のDefinition1、§3、§4.2を照合した。strict SU(2)・inverse閉包・密な生成集合、基底近似表が前提。記載の漸近/近似定数と経験的基底値を、厳密な有限上界定数として採用しない。§4.2は厳密定数の詳細を省略し、有限精度算術にも別の分析が必要と述べる。

証明済み定数c、基底全域誤差eta0、基底wordの最大T数t0を**別途資格した場合**の、再帰式に基づく価格条件は

\[
 \eta_0c^2<1,\quad
 \eta_n=c^{-2}(c^2\eta_0)^{(3/2)^n},\quad
 n=\min\{j\ge0:\eta_j\le\eta\},\quad
 T\le 5^n t_0.
\]

これは隠れたO定数を価格とする式ではなく、欠けているc/eta0/t0を明示した条件式。現資産には、そのstrict位相の生成集合/Clifford+T wordへの展開、全域net certificate、丸め余裕を含む実装が固定されていない。H/Tのprojective普遍性だけでSU(2)の前提を埋めず、追加ancillaや連続scalarも無料に導入しない。新しいSK実装・net生成は行っていない。

## 固定pygridsynth2.0.0の静的照合

公式tag v2.0.0はcommit **`65f6dd06538675d63a239009574dea5c636668b1`**。SERVER 2の既存venv内にある下記7 SOURCEのGit blob hashは同tagと一致した。照合はbyte読取だけで、libraryをimport/実行していない。

| SOURCE | 確認した仕様と資格の限界 |
|---|---|
| [config.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/config.py) | dloop/floop各10、timeout未指定、既定up_to_phase=False。研究wrapperはseed17・up_to_phase=Falseを明示 |
| [loop_controller.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/loop_controller.py) | norm/factor処理の試行回数とtimeoutを制御。全角度取得のiteration上界ではない |
| [diophantine.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/diophantine.py) | 予算切れでもNO_SOLUTIONを返す経路がある。数学的不可解の証明と混同しない。呼出しごとにseedを設定 |
| [gridsynth.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/gridsynth.py#L245) | strict branchはhas_phase=Falseでkを増やす。明示k最大値/全入力完了定理はない。取得価格の最適性からの一様上界もここにはない |
| [mymath.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/mymath.py) | epsilonからmpmath dpsを設定。任意の正epsilonへの形式的な桁計算と、その全入力correctness/完了保証は別。角度域全体のdirected grid判定ではない |
| [synthesis_of_cliffordT.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/synthesis_of_cliffordT.py#L61) | strictではWを明示しNormalFormへ変換。strict searchのn=0経路と、projective/別phaseの経路を混ぜない |
| [quantum_circuit.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/quantum_circuit.py#L38) | simple stringと別phase fieldは同一でない。global Wを含むexact-word guardを最後の証拠とする |

### backendのdiagnostic errorはstrict guardの代用ではない

[gridsynth.error](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/gridsynth.py#L129)は、targetと近似行列の(0,0)成分から得る実内積を二乗する。**静的な代数的確認**では、近似行列が−RZである場合にこの式は0となるが、strict operator距離は2である。関数名や`phase=0`だけを根拠にstrict certificateとしては使えない。

これは今回、新しいwordを生成して反例実験したという意味ではない。実strict searchが誤wordを返したという発見でもなく、CT-1の無効化でもない。本repoはこのdiagnosticを採用せず、[strict_guard](../../../src/trottertracks/track_c_h4_synthesis.py)でexact W付き行列をbinary targetと直接比較している。

## 研究wrapperが保証するもの

`acquire`はbackendへepsilon_G/4を渡し、返却後に区間guardを行い、必要時に同じwordを高精度で再guardする。新wordのguard失敗・nonfinite endpoint・word長超過ならUNQUALIFIEDで例外。backend探索が返ること、あるいは全入力でguardが成立することまで証明してはいない。

SOURCEの`len(sequence)>20000`拒否条件から、新規取得の**受理後**には

`T(s) <= len(s) <= 20000`

と論理的に言える。しかしこれは既存条件の引用・含意であり、新しいT価格評価ではない。**全角度で受理wordが得られるという前件がないため、全support価格上界ではない。** 長さ検査は探索後に行われ、探索時間の上限にもならない。cache hitはこの分岐より先にreturnするので、保存recordにも無条件にこの長さ証明を移さず、必要なら採用recordを個別照合する。

したがって、成功wordのstrict正しさと、全supportに対するtotalなword policy・明示T上界は別に資格しなければならない。**固定backendへの移送はUNQUALIFIED、mandatory STOP。** 数値上界を評価したりfallbackを実装したりせず、判断をGPT／ユーザーへ戻す。
