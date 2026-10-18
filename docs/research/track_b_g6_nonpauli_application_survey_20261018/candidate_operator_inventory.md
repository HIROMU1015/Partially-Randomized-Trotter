# 候補演算子の棚卸し

2026-10-18 JST。以下は既存の数式・SOURCEの対応整理であり、新しい分解の提案ではない。判定一覧は[一覧表](application_feasibility_matrix.md)。SOURCE IDは[source_inventory.json](source_inventory.json)、一次文献の版と確認箇所は[既知対照](pauli_and_known_method_baselines.md)に対応する。

## 記法と共役方向

SOURCEのforward basis operationsを時間順に適用したmany-body unitaryを`B_l`とする。既存builderの時間順は`B_l† → central → B_l`であり、演算子は`B_l P B_l†`である。指示文の`G†PG`に対応させると`G=B_l†`。one-particle行列とFock-spaceのBを混同しない。[S09：dense referenceの定義（読むだけ）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py#L907-L930)、[S10：basis順序](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py#L448-L458)。

実際のGaussian生成経路は`build_df_blocks`と`U_to_qiskit_ops_jw`。後者には`BogoliubovTransform(U.T)`という規約がある。記号Bを用いることで、Uの転置・adjointを名前だけから取り違えない。[S13](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_trotter/ops.py#L194-L225)。registry自体は一般のlocal gate列を登録できるので、registryにあるだけでGaussian性まで認定しない。

## C01：乱択tailの符号付き軌道回転Z（1軌道reflection）

\[
T_{lp}=B_l Z_p B_l^\dagger=I-2n_{lp},\quad
n_{lp}=B_l n_p B_l^\dagger,\quad Q_{lp}=s_{lp}T_{lp},\quad s_{lp}=\operatorname{sgn}(c_{lp}).
\]

`T†=T, T²=I`で、`s=±1`を付けても成立する。一般のorbital rotationでは単一の計算基底Pauliとは限らない。特別なBでPauliへ戻る場合もあり、各分子成分の非Pauli性を今回数値認証したわけではない。

SOURCEでは`DFDiagonalPauliComponent`が係数、sign、basis ID/hash、supportを持つ。ランダム側のfragmentにもこの形式が実際に現れる。係数式は[接続文書](pr_rte_connection_and_access.md)。`controlled-Q`はbasisの間のCZと必要なancilla sign位相、回転はcontrolled-RZで実装する既存経路がある。G6固有のpacket→event接続とnative認証は未確認。[S09：抽出](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py#L531-L674)、[S10：central primitive](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py#L473-L502)。

既存reflection監査の記法では、Gaussianの真空位相を固定して`T=Γ(I−2vv†)`と書ける。`sΓ(M)`のscalar signを`Γ(sM)`へ移してはいけない。[S07 §3 M1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)。

## C02：同じDF基底の符号付きZZ（2軌道parity）

\[
T_{lpq}=B_l Z_pZ_q B_l^\dagger
=(I-2n_{lp})(I-2n_{lq}),\quad p\ne q,\quad Q_{lpq}=s_{lpq}T_{lpq}.
\]

同じBの異なるnumber operatorsは可換なので、積はHermitian involutionである。互いに直交するorbital列v,wなら`T=Γ(I−2vv†−2ww†)`という既存監査の対応を使える。異基底の任意二つのreflectionにはこの可換性を移せない。

`controlled-Q`の中央は2個のCZ、回転の中央はcontrolled-RZZ。**ZZそのものの作用とZZに沿った連続回転は別のprimitive**である。特に後者は一般角度でnumber-conserving Gaussianとは限らず、Gaussian専用compilerだけで全eventを覆う根拠にならない。[S10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py#L473-L502)、[S07 §3 M5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)。

## C03：軌道回転number operatorそのもの

`n_lp²=n_lp`であり、full Fock spaceで空状態に0を作用させる。したがって`n_lp²=I`ではなく、**そのままG6の単一QにするのはNOT_APPLICABLE**。`n_lp=(I−T_lp)/2`の既知恒等式を使えばC01とidentityへ戻るが、係数とshiftを一緒に戻す必要がある。`exp(−it a n_lp)=exp(−it a/2) exp(+it a T_lp/2)`であり、前のscalarもcontrol時にはancilla相対位相となる。[S24：既存抽出の規約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/df_rte_tail_extraction.md)。

これは不適合の境界例であり、反射への変換を新しい応用候補の発明として数えない。

## C04：同一基底のdensity-density projectorそのもの

\[
n_{lp}n_{lq}=\tfrac14(I-T_{lp}-T_{lq}+T_{lpq}),\quad p\ne q.
\]

左辺はprojectorで、full Fock spaceの真空では0。単独のinvolutionとしてはNOT_APPLICABLE。右辺は既存DF抽出で扱われるI/Z/ZZへの展開であり、C01/C02を支持する構造である。同じpなら`n_p²=n_p`へ戻り、上式の異なる二軌道扱いを重ねてはいけない。必要な係数・shiftは[接続文書](pr_rte_connection_and_access.md)に示す。[S09 exact_df_diagonal_coefficients](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py#L531-L556)。

## C05：二乗DF fragment全体そのもの

\[
H_l=B_l\,\lambda_l\big(\sum_p\eta_{lp}n_p\big)^2 B_l^\dagger.
\]

Hermitianであっても、unshiftedなこの式は真空に0を作用させるためfull Fock spaceでinvolutionにはならない。scalar倍だけではこの0固有値を解消できない。固定粒子数sectorや別のshift・正規化について、この議論から一般的な不可能性を主張しない。

既存SOURCEは全fragmentを単一Qとせず、I/Z/ZZの**線形結合**へ展開する。C05をそのままQにする提案は不適合、既存展開から得る成分はC01/C02である。新しいshiftや分解policyは考案しない。[S14：DFBlock](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_trotter/model.py)、[S24](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/df_rte_tail_extraction.md)。

## C06：異なる基底のreflection積を一つのgeneratorとする読み方

`Q_a,Q_b`がHermitian involutionでも、`(Q_aQ_b)†=Q_bQ_a`。従って積がHermitianとなるには`[Q_a,Q_b]=0`が必要で、この条件があれば平方もIになる。実際の異基底ペアの可換性は今回計算しておらず、個別pairの適否はNOT_ESTABLISHED。

既存eventのproduct roleでは非Hermitianなunitary積も順序どおり使える。それをG6の**一つの回転generator**へ置換できるという意味ではない。G6も奇数語全体をinvolutionとは仮定していない。[S02 §§1,6,9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)。Gaussian積圧縮の終了済み候補を再開しない。

## C07：現在Dに置かれているone-body／prefix fragmentのreflection

同じZ/ZZ共役は決定論側にも現れ、代数・既存primitiveの成立性はC01/C02と同じである。しかし`prepare_df_partial_s2`はone-body correctionをDへ置き、tail抽出へ渡すのは`partition.randomized_block_indices`である。**Dの存在は現行RでのG6適用の証拠ではない。** これを新たにRへ移すには分割・研究方針の別判断が必要で、総合分類は`REQUIRES_NEW_DECOMPOSITION_OR_RESEARCH_CHOICE`。[S12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_partial_s2.py#L328-L446)。

## 分子固有情報の限界

`eta, lam, U_ops`から係数と実装アクセスを得る既存interfaceは確認できた。一方、今回の調査用に特定分子の全係数表・orbital列を新しく取得・生成したわけではない。既存H2再構成testのSOURCEも存在するが未実行であり、test関数の存在をG6の分子実証と呼ばない。builderの小型testにはRYを使う一般basis fixtureもあり、それだけでGaussian分子検証とは認定しない。[S21](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/tests/test_df_rte_qiskit_builder.py)、[S22](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/tests/test_df_rte_tail.py#L149-L195)。
