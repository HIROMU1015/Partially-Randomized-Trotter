# 現行PR/RTEとの接続・係数・access

2026-10-18 JST。既知の恒等式と固定SOURCEの静的対応。新しい分割・sampler・adapterを設計、実装、実行していない。

## 1. 現行D/Rのどこにあるか

`prepare_df_partial_s2`は採択済みpartitionを検証し、one-body correctionと決定論fragmentをDへ用意する。Rの抽出対象は`partition.randomized_block_indices`。既定の`weight_ranked_prefix`と、事前登録順序を保つ`explicit_ordered_partition`があり、後者を無断で並べ替えない。[S12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_partial_s2.py#L328-L446)。

従ってC01/C02は**既存Rの候補成分そのもの**で、代数的適用可能性を示すために新しいD/R分割を要しない。C07をRへ移すことは別の研究選択であり、今回行わない。`L_D`がendpointでRが空ならG6を走らせるtailはない。新しい`L_D`や分子を選んで非空tailを作ることもしない。

## 2. 二乗fragmentからI/Z/ZZへの既存の対応

記法は[候補棚卸し](candidate_operator_inventory.md)。SOURCEのfragmentは、実数etaと実数lambdaについて

\[
H_l=B_l\lambda_l\big(\sum_p\eta_{lp}n_p\big)^2B_l^\dagger
=c_{lI}I+\sum_pc_{lp}T_{lp}+\sum_{p<q}c_{lpq}T_{lpq},
\]
\[
c_{lI}=\frac{\lambda_l}{2}\left(\sum_p\eta_{lp}^2+\sum_{p<q}\eta_{lp}\eta_{lq}\right),\quad
c_{lp}=-\frac{\lambda_l}{2}\eta_{lp}\sum_q\eta_{lq},\quad
c_{lpq}=\frac{\lambda_l}{2}\eta_{lp}\eta_{lq}.
\]

これは既存`exact_df_diagonal_coefficients`とその文書の式を照合したもの。lambdaには元のDF表現のprefactorが含まれる。別論文の`1/2`やspin-index規約を再び掛けない。同じsupportでも異なるbasisの成分は統合しない。[S09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py#L531-L635)、[S24](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/df_rte_tail_extraction.md)。これは数値係数の新規計算ではない。

identity方針も既存の二通りを保存する。`extract_identity_phase`ではretained identity係数を足し、Rのsamplingから外して決定論的scalarへ置く。`faithful_identity_in_tail`では符号付きIもRへ残す。nonzero thresholdによるdropは別の近似であり、identity抽出をdrop誤差と数えない。[S09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py#L636-L674)。

## 3. G6の入力形式への数式上の対応

保持された乱択成分indexをa、unsigned成分をT_a、係数をc_aとする。零係数を除き、

\[
\Lambda_R=\sum_a|c_a|,\quad p_a=|c_a|/\Lambda_R>0,\quad
Q_a=\operatorname{sgn}(c_a)T_a,\quad H_R=\Lambda_R\sum_ap_aQ_a.
\]

したがって`R_G6=H_R/Λ_R`はG6の`sum p_i Q_i`と対応する。元HamiltonianのRそのものと、G6のnorm-normalized Rを同じ単位で扱わない。SOURCEの`rte_lambda_r`はこの係数L1に対応し、fragment ranking proxyとは異なる。[S09 extraction_to_symbolic_rte_tail](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py#L808-L855)。

既存のmicrostep幅をhとすると対応する無次元量は`x=Λ_R|h|`、signed timeは`σ=sgn(h)`。G6の既存保証は`0<x≤1`、正の有理p、有限奇数mを前提とする。`x=0`はidentityの別経路。旧paired cutoffが最大even Kなら、対応する有限Taylor targetの最高次数は`m=K+1`だが、**同じ多項式でもevent law・角度・weightが同じという意味ではない**。[S01 ReturnKernel](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py#L136-L151)、[S02 §§1,6–8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)。

元分子の実数係数、既存binary64のc/p、G6のexact rational入力を同一視しない。G6に渡す有理入力とその正規化・元係数との差の証拠は今回未取得。保存floatを`Fraction`へ変換するだけで元物理Hamiltonianへの誤差認証が完成したとはしない。`Λ_R=0`のtailでは正規化できず、既存のdeterministic-only処理を維持する。

## 4. 実装interfaceで確認できることと不足

| 層 | 固定SOURCEで確認できること | 今回の不足 |
|---|---|---|
| DF入力 | DFBlockにeta、lam、U_ops。registryに実行gate列とhash | 今回の分子固有入力bundleは未取得。新たにbuildしない |
| R成分 | component ID、c、abs/sign、support、basis/hash、fragment ID | 全ての登録basisのGaussian性を型だけでは保証しない |
| G6局所query | rational p/x、parent係数、symbolic tangent、digital packet | DF入力へのidentity binding、finite-bit誤差との統合が未確認 |
| 既存event | productsを時間順に適用し、最後に一つのrotation | G6 packetのlaw/weight/拒否を接続する既存adapterは確認できない |
| Controlled primitive | 共役Z/ZZと任意float角度の回転をlowerする経路 | G6のword依存角度・strict誤差・native費用の認証なし |
| 推定量 | 旧ordinary samplerはfinite normalizationを保持 | G6 zero-fillの全trial分母・実weightを現行measurementへ統合した証拠なし |

S01の`digital_parent`は回路を作らず、`angle_tangent`と`circuit_word_time_order`を返す。S15の`_make_event`はordinary paired lawと`atan(tau/(order+1))`を作る。RTEEvent自体は有限角度を保持できるので、**G6角度を表現することが数学的・API的に不可能と認定したのではない**。同じevent型に似たfieldがあることを、validatedなG6接続がある証拠にしない。[S01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py#L214-L254)、[S15](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py#L1594-L1659)。

検索範囲は作業基点の`src/trotterlib`、`src/trottertracks/algorithm_codesign`、`scripts`、G6 test。`ReturnKernel/return_aggregation/G6`の静的検索と上記call chainを確認した。別branch、未公開実装、並行作業中のEvidence Packetについて不在を主張しない。SOURCEのimport、既存test、samplerは実行していない。

## 5. 情報・費用の境界

既存full-basisのU_opsアクセスはあるため、thin orbital adapterがないことを理由に単一C01/C02のcontrolled構成まで否定しない。ただしNx r orbital列はregistryのtyped fieldではなく、`basis_unitary`はmany-body referenceである。低rank orbital accessを前提とするactive-span実装を既存機能に数えない。[S07 §4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)。

G6の明示p表read、局所係数query、child選択、有限bit weight、angle取得、全reject trialの古典費用は残る。未知normalizer/acceptanceを無料取得したことにせず、真のZを使うshot削減も認定しない。pの取得、basis生成、cache構築、online costの大小を今回CPU時間や新計算で評価していない。[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)。

**結論：現行Rに代数的対応とprimitive実装はある。現在のG6への直接実装接続はNOT_ESTABLISHEDであり、C01/C02の総合分類は実装未完了。新しいD/Rを必要とするC07とは区別する。**
