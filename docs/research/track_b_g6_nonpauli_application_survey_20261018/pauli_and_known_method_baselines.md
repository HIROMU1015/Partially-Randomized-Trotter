# Pauli・SCU/CTS・既知DF/Gaussian方式との比較条件

2026-10-18 JST。新しい性能比較・ランキング・一般的新規性監査を行わず、今回の演算子クラスに必要な比較条件を確認した。

## 1. 一次文献の固定版・確認箇所

本文PDFを取得して関連箇所を確認した資料はL01/L02/L03/L06/L07。版、bytes/SHA256、確認頁・節は[literature_scope.json](literature_scope.json)。全文を精読したとの意味ではない。論文PDFは/tmpだけに置き、repositoryに転載しない。

| ID・一次資料 | 確認箇所と必要access | 今回の対応・限界 |
|---|---|---|
| L01：Günther et al., [PR arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2) | §IV Eq.(18),(27)–(29)、Appendix A.2 (A18)–(A26)、§VII.B pp.13–14、Appendix C。符号付き係数、orbital basis、回転access | §VII.B末尾は異なる軌道基底の`P̃=U†PU`も乱択法で扱えると明記。G6の一般involution対応だけを元PRとの差にしない |
| L02：Motta et al., [arXiv:1808.02625v2](https://arxiv.org/pdf/1808.02625v2) | 本文Eq.(3),(4)、Supplement “Partial Basis Rotation” Eq.(20)以降、pp.7–8。DF factors/eigenvectorsが入力 | 既知のrotated numberとdensity-density、Givens/partial basisが対照。既存のDF取得誤差・rank truncationを今回の新しい保証へ転用しない |
| L03：Jiang et al., [arXiv:1711.05395v3](https://arxiv.org/pdf/1711.05395v3) | §III Eq.(9),(13)–(15)、Fig.1、§IV、Appendix A。orbital matrixとJW/order規約 | 軌道回転をGivens列へ対応させる既知構成。特定Slater状態だけのgauge freedomや省略可能位相をfull-unitary controlled作用へ移さない |
| L04：Peetz–Smart–Narang, [出版本文 DOI](https://www.nature.com/articles/s41534-025-01168-w) | Theorem 1、Eq.(4)–(6)、Methods “Calculating stochastic operator ensembles”をweb本文/PDF表示で確認 | Pauli収集とEuler pairing、SCUの左右独立samplingを比較する。直接PDF byte取得はHTML応答で失敗し、PDF SHA256は未取得 |
| L05：同[出版補足](https://media.springernature.com/original/springer-static/esm/art:10.1038%2Fs41534-025-01168-w/MediaObjects/41534_2025_1168_MOESM1_ESM.pdf) | 今回直接取得はPDFでない応答、webも取得不能 | **今回の出版補足読了とはしない**。旧S19の確認履歴と、下のarXiv版補足を区別 |
| L07：同著者、[arXiv:2407.21095v2](https://arxiv.org/pdf/2407.21095v2) | Theorem 1、Methods、Supplementary Note 3 p.12、Note 5 pp.14–15 (S13)–(S15) | 収集したPauli係数の回転化と、層ごとの独立samplingを確認。出版補足とのbyte同一性は認定しない |
| L06：Kökcü et al., [arXiv:2303.09538v2](https://arxiv.org/pdf/2303.09538v2) | §IV.B（HTMLではIV.2）Eq.(42),(43)、Appendix C.2 pp.23–24、PDF式(C6),(C7)（HTML表示では(84),(85)）。free-fermion block構造が条件 | controlled Gaussian/free-fermion subcircuitにも既知対照がある。generic ZZ回転やG6全eventがその圧縮条件を満たすとはしない |

L01は応用演算子クラス、L02/L03は取得と基底回転、L04/L07はPauli収集とsampling、L06はcontrolled構成を支える。文献中にG6一式が掲載されているとの意味ではない。一般的なpriority・独立新規性の結論は出さない。

## 2. 演算子の同値性とsampling表現の相違

`T=B Z_p B†`や`B Z_pZ_q B†`は、同じfull Hilbert space上でPauli展開もできる。ただし基底変換native表現と展開後の係数L1、label集合、積の収集、角度辞書は同じとは限らない。Hamiltonianとしての同値性と、有限Taylor target、event law、実weightの一致を別に照合する必要がある。

現行のDF抽出は各basis内でI/Z/ZZ係数を持ち、異なるbasisをまたぐPauli収集はしない。一方、Pauli対照にはその追加関係や正当な前処理を認める。G6のfree-product reductionが追加Pauli関係を使わないことを、対照にも使わせない理由にしない。[既存S19 §§Claim比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)。

## 3. 公平な比較に必要な項目

| 比較軸 | G6側で必要なもの | 対照へ同等に与える条件 |
|---|---|---|
| 物理target | 同じH、D/R、定数shift、時間、入力表現誤差 | Pauli化・DF-nativeのどちらも同じretained Hを再構成 |
| 有限target | 同じ奇数mのfinite polynomial、Taylor bias | 異なるm/rを比較するなら同じ理想exp精度へ戻す。今回選ばない |
| 情報取得 | DF factors、basis、p table、局所query、angle | Pauli展開・収集、layer setup、basis取得も有料。無料oracleにしない |
| controlled回路 | Q、回転、signed angle、parent/identity phase | 同じcontrol規約、routing、workspace、strict精度、compiler機会 |
| sampling | dyadic実law、weight、全trial、moment、range、bias | CTS/ordinaryにも合法なsampling・cache・importance samplingの機会 |
| 再利用 | 隣接する同一basisの共有、厳密inverse簡約 | DF/Gaussian-nativeにも同じboundary共有を認める |
| 費用 | basis、controlled lowering、native T/CX/depth、古典取得 | RZ個数とT数を同一視しない。whole-circuit最適化を片側だけ排除しない |
| 保証 | coefficient/native/Taylor/PF/statistical誤差の分離 | 同じcoherent signalとconfidenceへ比較scopeを合わせる |

CTSは全高次数語の展開だけに限られない。L07 Supplementary Note 3の層別samplingも対照となる。全Pauli表を取得する実装だけへ固定して、G6の非列挙性から取得優位を認定しない。SCUのchannel/observableを、そのままHadamardのoperator meanと同一視することも避け、controlを含むtaskへの対応条件を示す。

## 4. 終了済み系列との境界

Reflection/active-span/Gaussian積圧縮の既存構成は[S07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)と[S08 closure](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md)を継承する。active spanやthin compilerを新手法として再提案しない。元closureは性能negativeを確定したものでもない。

DF sparse correctionのwhole nested commutatorは、Hermitianでも平方Iが保証されない別構造である。既存Pauli/NCC/HNCC等との比較履歴は[S18](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md)を参照するだけとし、そこから新しいG6 generatorを設計しない。その系列の価格や別fixtureの頻度を掛けてG6の分子総費用を作らない。

## 5. 今回の比較判定

構造と必要費用の比較は可能である。しかし同一分子入力・coherent task・finite-bit/native精度で、G6とordinary/Pauli-aware CTS/既知Gaussian-nativeを接続した比較証拠は今回確認できない。性能優位、情報取得優位、総資源改善はすべて`NOT_ESTABLISHED`。新しいT/CX/depth・CPU時間の取得や保存数値の再解析は行っていない。
