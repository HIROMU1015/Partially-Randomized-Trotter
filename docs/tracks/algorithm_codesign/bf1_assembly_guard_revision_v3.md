# BF-1 assembly guard amendment v3

2026-10-05 JST。status: `ASSEMBLY_GUARD_REVISION_CONTENT_FROZEN_REVIEW_REQUIRED`。
親source review: [296ec7eの最終レビュー](bf1_final_source_review_296ec7e.md)。
対象commit: `296ec7e4c025f09e4bfda96e56e08d32388b81c5`。
local finding: `REVISE_NUMERICAL_ASSEMBLY_GUARD_BEFORE_AUTHORIZATION`。
**science executionは未認可。commit/pushも本書からは開始しない。**

[preregistration v1](bf1_preregistration_v1.md)と[v2](bf1_execution_gate_revision_v2.md)を維持し、
本書はassemblyからcoherent-signalへ戻す数値guardと、その結果前sealだけを修正する。
H4 development、generation-prefix L_D=3、5-stage family、32 evaluations/arm、O/L/F、
q/R/K policy、allocation、fixed refs、accuracy、primary 5%、資源上限、cross-score、
authorization-only child方式、全case mandatory STOPを変えない。

## 1. Budgetの意味

供給行列を`Ghat_i`、binary64 DF snapshotの係数から数学的に定義したreference generatorを`G_i`とし、
`||G_i - Ghat_i|| <= E_i`、scalar phaseについて`|c - chat| <= E_c`を別々に用いる。
これらは物理DF truncation errorではなく、同じ保存係数からのassembly errorである。

`Task.generator_assembly_bounds`は全generator keyを明示し、`scalar_assembly_bound`はscalarを指定する。
非finite、負値、key不足は拒否する。syntheticでは個別の既知摂動を使う。
旧`matrix_assembly_bound=E`を使う場合、**Eは各generatorおよびscalarそれぞれの上界**と定義する。
全generatorの和のerrorだけをEとして渡すことはできない。

science loaderは保守的なuniform budgetを全generatorとscalarへ明示的に設定する。
input auditへpolicy、各budget、targetのgenerator-budget合計を保存する。
targetの供給行列`hhat = sum Ghat_i + chat I`の加算誤差にも別のgamma allowanceを置き、

\[
E_H=\sum_i E_i+E_c+E_{\rm sum}
\]

を使う。個別errorがfull-Hで相殺するという仮定を置かない。

## 2. Source assembly allowanceの条件付き導出

固定sourceは8 orbitals、12 DF fragments、dimension 256である。
各one-body operatorは高々64個の`a_p^dagger a_q`項からなる。
各項はoccupation basisで符号付きpartial permutationなので、絶対値を取ったmatrixの
row/column sumも係数絶対和で抑えられる。

\[
S=|c_0|+\sum_{pq}|h_{pq}|+
  \sum_{l=0}^{11}|\lambda_l|\bigl(\sum_{pq}|G_{l,pq}|\bigr)^2.
\]

これは相殺を含むoperator normより先に、assemblyの絶対値scaleを与える。
共有sourceのPython経路`_apply_one_body`、`_df_matvec_python`、
`_dense_df_operator_qiskit`、`_dense_block_operators`をそのまま参照する。
共有API・実装は編集しない。

| source operation | 保守的な計上 |
|---|---|
| 64以下のcomplex weighted sum | 各chainで`16*(64+2)=1056` real operations相当 |
| DF fragmentの二回のone-body作用、lambda multiplication、constant/one-body加算、Hermitian化 | local selected/base評価を`gamma(4096) * S`で包む |
| selected-minus-baseによる個別block、one-bodyのconstant除去 | 二つのlocal評価とsubtractionを計上 |
| tailの高々12 block加算、identity除去、scalar加算 | 全体を最大`28*gamma(4096) + gamma(256)`のS倍で包む |

componentwise absolute pathのrow/column sumからoperator normへ移すため、別々のcolumnの
丸めが一致することを仮定しない。Frobenius評価を併用してもよいよう、実装は更にsqrt(256)=16を掛ける。
`gamma(n)=n*eps/(1-n*eps)`の下で、上記の係数は
`gamma(65536)*sqrt(256)`より小さい。したがって既存の65536 allowanceは
**各assembled generatorとscalarを覆うuniform forward budget**として使える。
観測full-H reconstruction discrepancyだけから個別budgetを推定するものではない。

identity抽出で使うDF eigensystemは`diag_hermitian`が実際に返した`eta,U`を
`Spectrum(g, eigensystem=(eta,U))`へ渡し、reconstruction、orthogonality、polar-distanceの
residualを計上する。他のeigensolver出力を代用しない。

identity係数は`lambda * ((Tr G)^2 + Tr(G^2))/4`。
8次元でHermitian摂動のnormがrhoなら、identityの変化は

\[
|\Delta c_l|\le |\lambda_l|
  (36\|G_l\|\rho+18\rho^2)
\]

で抑えられる。実装の
`256 * abs(lambda_l) * rho * max(1, norm(g_l,'fro') + rho)`はこれを覆う。
coefficient算術、identityの加算・引算のroundoffも上のforward-operation allowanceに含める。
uniform Eは従来の式を保持し、このeigensystem allowanceをactual eigensystemに対して評価する。

\[
E=E_{\rm reconstruction}+
\gamma(65536)\sqrt{256}\max(1,S)+E_{\rm identity\ eigensystems}.
\]

これはsourceのoperation数、IEEE binary64、標準forward-error modelと通常のscalar算術に
条件付きの導出であり、directed roundingを伴うinterval certificationではない。
数値guardの導出とsource対応は改訂sourceの最終review対象に残す。

## 3. 各signed factorとfinite積への伝播

`Spectrum.action(..., assembly_error=E_i)`は、eigensystem residualとは別にE_iを受け取る。
unitary actionではHermitian exponentialのLipschitz項`abs(t)*E_i`を加える。
negative timeの符号を消してoperatorを変えることはなく、**errorの上界だけ**に絶対時間を使う。

finite actionはK2=P3、K4=P5とする。microstep半径をmu、Taylor remainderによるnorm上界をMとして、

\[
\mu=\frac{(|t|+u_t)(\|Ghat_i\|+\rho_i+E_i)}{r},\qquad
M=1+\frac{e^{\mu}\mu^{K+2}}{(K+2)!}.
\]

Hermitian generatorなら`||P_(K+1)|| <= M`。r microstepsのnormは`N_i <= M^r`、
matrix perturbationに対するLipschitz係数は

\[
D_i\le(|t|+u_t)e^{\mu}M^{r-1}.
\]

polynomialの各termを微分するとmicrostepで`(|t|+u_t)*exp(mu)/r`、
r個の積のtelescopingでrが打ち消されてこの式になる。
eigensystem perturbationとE_iを`D_i*(rho_i+E_i)`へ戻す。
計算したdiagonalのnormもN_iに含め、time lowering、finite polynomial評価とr乗、matrix-vector
roundoffの既存項を保持する。bound算術には固定gamma余裕を加える。

各factorでreference norm上界N_iとaction errorを使い、

\[
e_i\le N_i e_{i-1}+u_{{\rm action},i}
\]

で伝播する。action errorは計算済みprefix vectorのnormを含むため、prefix/suffixの増幅を
最後のvector normだけで置き換えない。最後のcontrolled relative scalar phaseには
`T*E_c*norm(vector)`を別計上する。
full targetにもE_Hを渡し、bias guardはfinite signalとtargetのguardを合算する。
旧`T*E*max(1,norm(final_vector))`というgenerator assemblyの一括後付け項は除去する。

## 4. 回帰検査とv3 packet

`tests/tracks/algorithm_codesign/test_bf1_assembly_guard.py`は次をsyntheticだけで検査する。

- 保存済みSuzuki5 1×1 counterexample。旧biasが変わらず、新guardが全axis誤差を覆うこと。
- 同じfrozen coefficientのK4、q=2。allocationや係数探索を追加しない。
- uniform/per-generator budgetとfull-targetの合計、非finite・負値・欠落budgetの拒否。
- 非可換Hermitian摂動と負時間のunitary/finite action。
- scalar relative phaseと複数generatorの摂動を同時に持つsignalとtarget。
- identity抽出で実際に使用したeigenvaluesのresidual guard。

v3 packet: `artifacts/track_b_bf1_preparation/2026-10-05/v3/`。
新しいsource plan、test report、guard regression report、false authorization draft、auditを作る。
v1/v2 packet、domain raw SHA、旧reviewとwitnessは上書きしない。
旧reproducerは旧source `296ec7e`と組み合わせた資料であり、改訂sourceで実行して旧結果を再現したとは扱わない。

guard regression reportは旧witnessのpath/hashと新しいguardの観測を対応させる。
科学的なaccuracy適格性、H4でのguard幅、coefficient選択、改善率は未評価である。
guardが厳しくなって実行時に不適格・INCONCLUSIVEになっても、閾値や上限を緩めない。

## 5. 次の境界

今回の修正はuncommitted source-content sealであり、commit-bound sourceではない。
改訂sourceを新しいSへcommit固定した後、Sを対象に最終reviewを行う。
旧`296ec7e`へsource修正を含むauthorization-only childを作らない。
最終reviewと別の明示的利用者実行指示が揃ってから、固定pathだけのauthorization-only Aを作る。
BF-1は一回だけ。その結果直後にmandatory STOPし、Bの方針・RQ・新規性・着地点を全面再評価する。
NPZ操作、Hamiltonian生成、science信号、trajectory、circuit、compile、GPUは今回も0。
