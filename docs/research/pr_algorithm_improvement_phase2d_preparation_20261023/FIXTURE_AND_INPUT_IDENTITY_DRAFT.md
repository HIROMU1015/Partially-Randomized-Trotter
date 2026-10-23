# Fixture・入力identity候補

全て**静的な数値仕様候補**。今回、以下から行列、Pauli係数表、状態配列、科学inputファイルを生成していない。候補の採用、生成物、hash、件数のbindingは未完了：`INPUT_SPECIFICATION_PENDING`。

## 1. 共通物理入力

synthetic 4 fermionic modes、system qubits 0–3、ancilla 4。Qiskit little endian、ket表記は `|b3 b2 b1 b0>`、system基底indexは `sum 2^p b_p`。ancilla込みindexはsystem indexに `16*b4` を加える。JWは `a_p=(prod_{j<p} Z_j)(X_p+iY_p)/2`、`n_p=a_p†a_p`。Pauli wordの保存表記はqubit 0からの配列とし、Qiskit文字列へ渡す時だけ反転する。

初期状態は `X0 X1 |0000> = |0011>`。prepを含め全arm共通、ancilla初期値0。物理targetは2粒子sectorだが実行空間はfull Fock 16次元、制御込み32次元。中間射影・postselectionなし。geometry、AO basis、分子DF rank／rank policyは非該当。fragment数2を分子DF rank2と呼ばない。固定splitはone-body block一つ `L_D=1` と残差二fragment。SCF／DF再生成、E0／gap／ground-state oracleなし。

以下の有理数は性能結果から選んだものではない。符号、非自明frame、局在と広がりを診断する一候補であり、coverageや優位性を確認済みとはしない。

```text
H = c I + D + R;  R = F0 + F1
c = 1/37
D = sum_p h_p n_p
h = (1/20, -1/30, 1/40, -1/50)
F_l = B_l (sum_p d_lp n_p)^2 B_l†
d_0 = (1/5, -1/7, 1/11, 0)
d_1 = (0, 1/6, -1/8, 1/10)
G_pq(cg,sg): rows/columns (p,q) are [[cg,-sg],[sg,cg]], others identity
V0 = G_01(3/5,4/5) G_12(5/13,12/13)
V1 = G_23(8/17,15/17) G_02(7/25,24/25) G_01(9/41,40/41)
B_l = Gamma(V_l)
```

行列積は右端が先に作用する。`B|vac>=|vac>`、`B a_p† B† = sum_j V[j,p] a_j†` によってsecond quantizationの向きとvacuum phaseを定義する。数値matrixをこの文書へ収録しない。V0は3mode、V1は4modeへ結合する構造候補。frame circuitのSOURCE内 `U.T` conventionを、この定義と照合する義務がある。

scalar policyは `faithful_identity_in_tail`：F_lのI成分をR内に残す。外部scalarは上記cだけ。D内のn_pに含まれるI位相もD recipeに保持する。armごとにIを外へ移動しない。frame側のS atomは元fragmentの区別を維持、Pに替えたfragment間の同一physical Pauli wordは全体加算後にabs/sign/probabilityを決める。S/P間の追加collectやframe同値探索はしない。

係数・V・係数mapは有理数のexact代数を予定する。exact zeroだけ除外し、小係数screeningなし。frame側はdensity-squareの対角Z/ZZ/I展開、P側は同じfragmentの二次／四次fermion式を固定JW代数へ展開する。dense traceからの係数取得をproductionの生成法にしない。独立参照はoccupation CAR行列とGamma(V)のexterior-minor定義による小系経路。これらは将来の計算設計であって今回の導出・生成結果ではない。

## 2. 時間・law・信号

| 段階 | T | q（S2 macrostep反復） | delta=T/q | r（各残差occurrenceのmicrosteps） | h=delta/r | K |
|---|---|---|---|---|---|---|
| F1候補 | 1/5 | 2 | 1/10 | 2 | 1/20 | 2 |
| F2候補 | 1/5 | 1 | 1/5 | 2 | 1/10 | 0 |

一macrostepは時間順 `D(delta/2) → r個のRTE event → D(delta/2)`、D(t)=exp(-itD)。外部scalarは全体 `exp(-icT)`。Kはpaired even cutoffで、有限多項式の通常次数はK+1。F1は四microstep、F2は二microstep。qとrを逆にしない。

arm aのcanonical化後component数をL_a、`lambda_a=sum |a_i|`、`p_i=|a_i|/lambda_a`、`tau_a=lambda_a*h` とする。各armでlaw、角度、Bを再構成する。同じR/h/Kなら有限corrected targetは同じだが、lambda/B/shot負担は同じとは限らない。Tやq/Kが異なるF1の信号・費用をF2へ流用しない。

測定対象は `z_K=<psi| M_K |psi>`、
`M_K=exp(-icT) [D(delta/2) {P_(K+1)(-ihR)}^r D(delta/2)]^q`。
physical signalは別の `z_exact=<psi|exp(-iTH)|psi>`。Hadamard X=Re、Y=+Im。scalarを差し引いたenergy referenceへ無断変更しない。

## 3. Catalog入力

物理fixtureとは別の、価格DPとactual event意味論用の限定catalog。4labelは[MOTIF_CATALOG_DRAFT.md](MOTIF_CATALOG_DRAFT.md)の一意仕様に従う。別input hashを必要とする。catalogを第五のphysical比較armにしない。

## 4. Precision・compiler候補

exact係数からcomplex128／binary64へ一度変換する。符号とzeroは浮動小数点thresholdで決めない。角度・確率・B・係数変換の丸め差をそれぞれ記録。参照はmpmath 1.3.0の90 decimal digits／128 decimal digitsの二経路を最初から登録し、90 Taylor／128 Padeのmatrix expを使う案。二精度一致はinterval certificateではない。selectorのscore／ceil判定にはoutward-rounded scalar interval providerの固定と検査が別に必要で、未binding。

compiler候補：Python >=3.11のexact patchを将来固定、Qiskit 1.3.0、NumPy 1.26.4、SciPy 1.14.1、mpmath 1.3.0。root requirementsのpinは参考で、将来envを検査済みとはしない。transitive dependency／wheel hashes／BLAS／Python patch／SOURCE closureは未確定。

```text
optimization_level=1; seed_transpiler=17
basis_gates=[rz,sx,x,cx]
backend=None; target=None; coupling_map=None; initial_layout=None
layout_method=None; routing_method=None; translation_method=None
scheduling_method=None; approximation_degree=1.0
qubits_initially_zero=False
```

topology-free。追加native T合成・hardware routingなし。full Operator検査の入力を|0>と仮定した消去を防ぐため `qubits_initially_zero=False` を全arm共通とする。layout permutationが発生すれば記録したmappingで比較し、不明ならSTOP。compiler並列度は科学条件ではなく実行時空き資源で決め、seedは全case共通17。sampling RNGは使わない。SOURCEのconfig seedはmetadata用に0を候補とし、確率抽選に利用しない。

## 5. Identityを今後固定する手順

候補文書のhashとscientific input hashを区別する。将来の生成後、exact有理数を既約分数文字列、ordered coefficient records、frame規約、mapping/state/scalar、arm定義、q/r/K/T、precision/compiler、recipe SOURCE closureをcanonical JSON（UTF-8、sort_keys、compact separators、no NaN）へbindingする。full input・physical fixture・catalog・arm coefficient map・law・recipe・environmentを別digestと関連IDで保存。実際のschemaとserializerもSOURCEとして固定する。

現時点の `prototype_source_commit`, `source_closure_sha256`, `fixture_input_sha256`, `catalog_input_sha256`, `environment_lock_sha256`, `L_SS`, `L_SP`, `L_PS`, `L_PP` は**全てnull／未確定**。架空hash、推測L、入力を生成済みと見せるmanifestは作らない。truncation_toleranceは固定Kを選ぶ装置にせず、将来binding時に既存residual-bound metadataを包含する値を全armで明記する必要がある。このAPI整合用値とsignal bias budget .025は別であり、未確定blockerとして残す。
