# P-A joint synthesis 先行研究・新規性監査

最終更新：2026-09-25 JST

## 結論とstatus

2026-09-25に、P-Aに近い研究を次の6系統へ分けて検索した。

1. low-rank／Double Factorization（DF）量子化学回路
2. partial fermionic basis rotation
3. fermionic Gaussian／Givens network合成
4. 隣接basis networkの融合・Hamiltonian-evolution block圧縮
5. isometry／未使用部分空間のcompletion自由度
6. dynamic programming（DP）・block／sequence-aware量子回路合成

検索範囲では、個々の構成要素は全て既知だった。一方、現在のP-A v1と同じ

> DF/RTE由来の対角event列を、同一source-basis runごとに連続区間へ分け、各区間でfull basisまたは
> support unionを保存する決定論的completionを選び、basis-operation-count proxyをDPで最小化する

という組合せは確認できなかった。従って、P-Aは検索範囲内で独立差分が残る候補として維持する。
監査完了時点のstatusを

`provisional_pending_blind_validation_after_scoped_prior_art_audit`

とした。後続blind transfer validationは両stratumで全gateを通過し、一度は
`advance_pa_v1_to_formal_primary_theme_candidate`となった。その後の
[形式化・機構監査](pa_joint_synthesis_v1_formalization.md)では、blindの54/54 recordで
run内区間分割が0件、48 holdoutの256 eventが全てTaylor order 0と判明した。さらに
[非退化mechanism validation](../research_direction_joint_synthesis_mechanism_validation.md)では、
forced-support order-2のtraining/blind全30 taskでもinterval DPと一区間baselineが同じplanを選んだ。
従って`stop_pa_interval_dp_as_primary_and_return_to_pc`とし、この時点でP-Cへ戻った。
後続の[P-C tracking・breakdown validation](../research_direction_geometry_tracking_breakdown.md)では
current H4 familyも固定停止条件に達したため、現時点でA/B/Cに確認済みの主研究候補はない。

これは新規性の証明、網羅的systematic review、特許調査、査読上の新規性判定ではない。主張できるのは、
明示した検索範囲と確認文献の中に同一問題設定を見つけなかったことだけである。加えて、完成済み証拠が
order 0/2とも一run一segmentに退化しているため、interval DP固有の寄与を主張しない。

## 現在のP-A v1が実際に解いている問題

現行実装は、一般的なcompletion／transition共同最適化ではなく、次の限定問題を解く。

- identityまたはsource basisの変更でevent列を分割し、同一source basisが続くrunを作る。
- 各runを一つ以上の連続区間へ分割する。
- 各区間について次の2候補だけを比較する。
  - 登録済みfull source basis
  - 区間内のdiagonal Pauli supportのunionに属するsource columnsを保存し、残りを決定論的に補完したbasis
- 古典目的関数は各区間のbasis operation数の2倍の和である。同点時は区間数、support-union size、
  full-basis penaltyの順で選ぶ。
- 回路builderが直接共有するのは、隣接する選択basisのIDとhashが完全一致するときだけである。

従ってv1は、異なるsource basisをまたぐ全列最適化、全ての等価completionの探索、任意の隣接basis間の
相対変換の再合成、pairwise compiled transition costの直接最小化を行っていない。

v1の数値証拠も限定される。固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`、
`delta=0.02`、finite Taylor cutoff 2、topology-free Qiskit 1.3.0 optimization level 1で、未使用列長
3、5、8の24 trajectoryにおいて現行`support_run_le_1`比pooled RZを7.194%減らした。operator同値性は
24 trajectory全件ではなく、各holdout長1件の計3 probeで確認した。oracleは比較した4 policy内だけの
有限oracleである。

## 将来のP-A v2との境界

将来の一般化候補では、event $j$の必要部分空間を保存するcompletion familyを
$\mathcal C_j$として、例えば

$$
\min_{C_j\in\mathcal C_j}
\left[
\sum_j C_{\mathrm{local}}(C_j)
+
\sum_j C_{\mathrm{transition}}(C_j,C_{j+1})
\right]
$$

を考えられる。ここでは、異なるcompletion間の相対Gaussian unitaryを再合成し、その実際のgate costを
transition項として扱う。

このv2定式化は研究候補であり、現時点では未実装・未検証である。v1の7.194%改善をv2の証拠として
扱わない。v2を実装する場合は、candidate family、状態、目的関数、計算量、optimality scopeを先に固定し、
v1のblind setとは別の未使用条件で評価する。

## 検索方法

### 検索日と情報源

- 検索日：2026-09-25
- arXiv API：一次論文のtitle、abstract、version、本文確認
- OpenAlex：関連語を含む候補論文の横断検索
- Crossref：DOI、著者、出版年、掲載誌の照合
- 原論文PDFまたはpublisher record：要旨だけでは判断できないpartial rotation、network merging、
  future-work記述の確認

### 主な検索語

- `partial basis rotation quantum chemistry Givens`
- `fermionic Gaussian circuit synthesis Givens`
- `basis rotation double factorization quantum`
- `basis rotation sequence quantum circuit optimization fermionic`
- `Givens network merge basis rotation quantum chemistry`
- `unitary completion fermionic basis rotation quantum circuit`
- `Hamiltonian evolution circuit algebraic compression`
- `dynamic programming quantum circuit synthesis`
- `block synthesis quantum circuit optimization`
- `isometry synthesis unused subspace quantum circuit`

### 判定規則

次を全て含む研究を直接重複候補とした。

1. DFまたはlow-rank factorizationから生じるbasis-rotated diagonal operation列を対象とする。
2. 各operationが必要とする部分空間以外のbasis completion自由度を利用する。
3. operationを独立に合成せず、複数eventからなる列の目的関数を最適化する。
4. full basisとsupport由来completionの区間選択または同等の構造選択を行う。
5. gate count、depthまたはcompiled costで比較する。

タイトルまたは要旨で近い論文は本文の該当箇所まで確認した。surveyは候補探索に用いても、独立差分の
根拠には一次論文を用いた。

## closest prior art比較

| 系統・文献 | 既知の内容 | P-A v1との重なり | 今回残った差分 |
|---|---|---|---|
| Motta et al. (2018/2021) | low-rank電子構造Hamiltonianと、rank $\rho_\ell$に必要なcolumnsだけを処理するpartial basis rotation | full basis全体を回さず必要部分だけを使う | 複数DF eventのrun分割・support-union区間選択・列目的DPではない |
| Kivlichan et al. (2017/2018) | line connectivity上のfermionic swap/Givens network、Slater determinant準備、電子構造Trotter stepの線形depth構成 | Gaussian basis changeを効率的に合成する | 必要部分空間に等価なcompletionをevent列に応じて選ばない |
| Huggins et al. (2019/2021) | low-rank factorizationごとのbasis rotationを使う量子化学測定 | DF/low-rank由来basis rotationを実回路へ接続する | 測定groupingが対象で、時間発展event列のcompletion選択ではない |
| Kökcü et al. (2021/2022) | free-fermionicに対応する限定gate集合でTrotter step列を代数的に一blockへ圧縮する | Hamiltonian-evolution列を局所event単位でなくまとめて圧縮する | supportごとの等価basis completionと区間選択を扱わない |
| Ollitrault et al. (2023/2024) | VQE末尾と測定basisのGivens networkを行列積で一つのnetworkへmergeする | 隣接basis networkの融合・再利用 | DF diagonal-event列のsupport/run構造や複数区間のDP選択を扱わない |
| Iten et al. (2015/2016) | isometryだけが指定されたときの未指定部分を含む回路合成 | 全unitaryを固定しないcompletion自由度という一般概念 | particle-preserving Gaussian completionでもevent列最適化でもない |
| Sridharan et al. (2008) | dynamic programmingを用いたquantum gate complexity評価 | DPによる回路合成・制御の一般概念 | P-A固有の状態、候補family、DF event列目的を与えない |
| Younis et al. (2020/2021) QGo | 大回路を小blockへ分割し、unitary synthesisでblockを置換する | block/peephole的な列最適化 | DF/Gaussian構造やsupport-preserving completionを使わない |
| Langer et al. (2026) | matchgateでpure fermionic Gaussian stateを最適に準備するgate-count/depth条件 | fermionic Gaussian circuitの最適性を扱う最新の近接研究 | state preparationが対象で、basis-rotated diagonal operation列ではない |

既知研究を踏まえると、次を新規性としては主張しない。

- partial basis rotationそのもの
- Givens networkによるfermionic basis changeそのもの
- 隣接Givens networkのmergeそのもの
- DP、block synthesisまたはpeephole optimizationそのもの
- 任意isometryで未使用部分空間を補完できることそのもの

## 固定するnovelty statement

blind validation前に使用してよい最も狭い主張は次である。

> DF/RTE由来のbasis-rotated diagonal event列について、同一source-basis runを連続区間へ分割し、
> 各区間でfull basisまたはsupport unionを保存する決定論的Gaussian completionを選ぶ有限候補問題を
> 定式化する。現行v1はbasis-operation-count proxyをDPで最小化し、固定H4 pilotでは強いproject内
> baselineより低いcompiled RZを示した。

英語では次の範囲に留める。

> We formulate a run-local interval-selection problem for sequences of basis-rotated diagonal events arising
> from DF/RTE circuits. Each interval chooses either the full source basis or a deterministic Gaussian completion
> preserving the union of required support columns, and the current algorithm minimizes a basis-operation-count
> proxy within this finite candidate family.

現段階では次を主張しない。

- 文献全体に対する世界初の方法
- 全量子回路または全Gaussian circuitに対するglobal optimum
- 全ての等価completionを探索済み
- 一般のpairwise basis-transition costを最適化済み
- coupling、noise、backend、full partial-$S_2$、RPE総costまで有利

## 監査判断と次工程

検索範囲内ではP-A v1を既知手法へ完全には還元できなかったため、P-Aを暫定主題として残した。独立差分を
単一H4 pilotだけで判断しないため、次の計算を
[P-A v1 blind transfer validation事前登録](pa_joint_synthesis_blind_validation_preregistration.md)
に固定した条件だけへ限定した。

- physical transfer：未使用H5 snapshot
- compiler transfer：H4の同一event streamを未使用optimization level 2へ移すpaired comparison
- policy、DP目的、4 baseline、gateは結果を見る前に固定
- いずれかの必須gateが不通過ならP-Aを正式主題化せず、P-Cへ戻る

後続の[blind transfer validation結果](../research_direction_joint_synthesis_blind_validation.md)では、
両stratumが全6 gateを通過した。さらに[形式化・機構監査](pa_joint_synthesis_v1_formalization.md)で
最適性範囲、計算量、operator同値性条件を明文化したが、実行済み54 recordではrun内区間分割が
一度も選ばれていないことも判明した。続く事前登録比較でもforced-support order-2全30 taskで
run内分割とcompiled差は0だった。従ってP-A interval claimを主研究候補から停止し、P-Cへ戻る。
scoped監査が新規性の証明ではないという制限は変わらない。

## 参考文献

1. M. Motta et al., “Low rank representations for quantum simulation of electronic structure,”
   [arXiv:1808.02625](https://arxiv.org/abs/1808.02625),
   [DOI:10.1038/s41534-021-00416-z](https://doi.org/10.1038/s41534-021-00416-z).
2. I. D. Kivlichan et al., “Quantum Simulation of Electronic Structure with Linear Depth and Connectivity,”
   [arXiv:1711.04789](https://arxiv.org/abs/1711.04789),
   [DOI:10.1103/PhysRevLett.120.110501](https://doi.org/10.1103/PhysRevLett.120.110501).
3. W. J. Huggins et al., “Efficient and Noise Resilient Measurements for Quantum Chemistry on Near-Term
   Quantum Computers,” [arXiv:1907.13117](https://arxiv.org/abs/1907.13117),
   [DOI:10.1038/s41534-020-00341-7](https://doi.org/10.1038/s41534-020-00341-7).
4. E. Kökcü et al., “Algebraic Compression of Quantum Circuits for Hamiltonian Evolution,”
   [arXiv:2108.03282](https://arxiv.org/abs/2108.03282),
   [DOI:10.1103/PhysRevA.105.032420](https://doi.org/10.1103/PhysRevA.105.032420).
5. P. J. Ollitrault et al., “Estimation of electrostatic interaction energies on a trapped-ion quantum computer,”
   [arXiv:2312.14739](https://arxiv.org/abs/2312.14739),
   [DOI:10.1021/acscentsci.4c00058](https://doi.org/10.1021/acscentsci.4c00058).
6. R. Iten et al., “Quantum Circuits for Isometries,”
   [arXiv:1501.06911](https://arxiv.org/abs/1501.06911),
   [DOI:10.1103/PhysRevA.93.032318](https://doi.org/10.1103/PhysRevA.93.032318).
7. S. Sridharan, M. Gu, and M. R. James, “Gate complexity using dynamic programming,”
   [DOI:10.1103/PhysRevA.78.052327](https://doi.org/10.1103/PhysRevA.78.052327).
8. E. Younis et al., “QGo: Scalable Quantum Circuit Optimization Using Automated Synthesis,”
   [arXiv:2012.09835](https://arxiv.org/abs/2012.09835).
9. M. Langer et al., “Matchgate circuit representation of fermionic Gaussian states: optimal preparation,
   approximation, and classical simulation,” [arXiv:2603.05675](https://arxiv.org/abs/2603.05675).
