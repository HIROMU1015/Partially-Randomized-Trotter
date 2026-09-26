# finite-RTE位相・信号半径分離：scoped先行研究監査

最終更新：2026-09-26 JST
状態：`SCOPED_GAP_SURVIVES_NO_PRIORITY_CLAIM`

本書は[FR-0契約](finite_rte_phase_amplitude_contract.md)の主RQに最も近い公開一次文献を、
式・対象・出力量の単位で対応付ける。網羅的systematic reviewでも、新規性や優先権の証明でもない。
検索とPDF確認は2026-09-26までに公開されていた版を対象とした。

## 1. 限定novelty候補

現段階で検証対象として残る差分は次の限定された問いである。

> 固定された有限paired-Taylor RTE cutoffを、非可換なunitary参照列へ挿入したとき、
> 入力状態で観測するHadamard複素信号について、相対誤差のHermitian/anti-Hermitian成分から
> 位相誤差と信号半径を別々に、かつ実行時に利用可能な信号半径下界だけで認証できるか。

この文は研究結果ではなくFR-1で反証可能にする候補差分である。

## 2. 一次文献との対応

| 文献 | 確認箇所 | 既知である内容 | 本検証との境界 |
|---|---|---|---|
| Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647 / PRX Quantum 7, 020332 (2026) | 公開PDF Appendix A、確認版のA18--A31、A37--A41 | infinite RTE/qDRIFT、normalization、平均信号、absolute-tail-time負担、partial randomization | RTEで位相と振幅が別問題であることもnormalizationも新規ではない。有限cutoffの非可換interleavingに対するstate-conditioned phase/radius境界を同じ形では確認できなかった |
| Wan, Berta, Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, arXiv:2110.12071 / PRL 129, 030503 (2022) | Appendix CのLCU式、Appendix E.2 Theorem 4 | truncated Taylor由来のrandomized LCU、有限打切りによるcomplex estimator biasのoperator-norm/triangle bound、Hoeffding補正 | 有限打切り、確率正規化、bias controlは既知。本案は同じ対象のphase/radiusを非可換列と入力状態に応じて分離できるかだけを問う |
| Yi and Crosson, *Spectral analysis of product formulas for quantum simulation*, npj Quantum Information 8, 37 (2022) | effective Hamiltonian、eigenvalue/eigenvector/gap解析 | unitary PFの固有値誤差と固有ベクトル誤差の分離、state-specific条件 | finite-RTE平均は一般に非unitary。本案はPF spectral theoryの一般化を主張せず、観測複素信号へ限定する |
| Li, *Some Error Analysis for the Quantum Phase Estimation Algorithms*, arXiv:2111.10430 / J. Phys. A 55, 325303 (2022) | Theorems 2.2、3.1、4.1 | 不完全入力、近似unitary、random unitaryをresidual、gap、concentrationから評価 | 状態情報を使うこと自体は既知。有限RTEの平均Hadamard信号をradial/tangentialへ分ける同じ境界ではない |
| Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741 | 公開PDF v1 Appendix F、確認版F8--F14 | random orderingによるspectral shift、dephasing/damping、broadeningの平均信号解析 | 位相と減衰の区別は既知。本案はfinite paired-Taylor cutoffの局所相対誤差と利用可能情報による認証に限定する |
| Van der Houwen--SommeijerおよびPapakostas--Tsitourasのphase-lag/dissipation解析 | DOI:10.1137/0726012、DOI:10.1137/S1064827597315509 | scalar数値積分でphase-lagとamplitude/dissipationの次数が異なり得る | scalar Taylor展開の次数差は新規性でない。FR-1では可換対照と非可換列を明示的に分ける |
| Hu and Jin, *Quantum Simulation of Non-Unitary Dynamics via Amplitude-Phase Separation*, arXiv:2602.09575 v2 (2026) | Cartesian decompositionとalgorithmic framework | 一般nonunitary生成子のcoherent/dissipative分解を使うsimulation framework | 本案はfinite-RTE推定量の局所誤差境界でありgeneric nonunitary simulationではない。`APS`を名称・略称として使わない |
| de Montbrun--GerchinovitzおよびPoiani et al. | arXiv:2308.00978、arXiv:2406.03033 | multi-fidelity optimization/best-arm identificationとcost-aware認証 | 本案はoptimizerを新規化しない。後に設計へ使う場合も既知の選択法との差分を別途監査する |

## 3. 新規性として扱わない要素

- Taylor seriesの有限打切りとremainder bound。
- LCU/RTE sampling normalization $B_K$とattenuation。
- scalarまたは共通固有basisで位相誤差が振幅誤差より高次になること。
- 行列をHermitian部とanti-Hermitian部へ分けること。
- nonunitary dynamics一般のcoherent/dissipative分解。
- 状態依存bound、imperfect eigenstate、random unitary QPEを扱うこと自体。
- 誤差proxyをcost-aware optimizationやbest-arm selectionへ入れること。

## 4. FR-1で差分が消える条件

次のいずれかなら、限定gapが文献上残っていてもこの研究案を主題化しない。

1. 非可換toyで提案境界がSTRONG-NORMより有用にならない。
2. 利益が真の$\rho$を密行列から読む`PROPOSED-REF`だけに依存し、利用可能な
   $\underline\rho$を使う`PROPOSED-AVAILABLE`では消える。
3. 負時間、K=4、反復のいずれかで命題と数値の不整合が説明できない。
4. 既存文献の定理へ単純に代入するだけで、仮定も出力も同じ結果になる。
5. 改善が丸め誤差程度、または信号半径下界を正に保てない。

## 5. FR-0判断

公開文献の限定監査では、上記の「finite cutoff・非可換interleaving・入力状態付き
Hadamard複素信号・phase/radius別認証」を同一の仮定と出力で与える結果は確認できなかった。
ただし不在を証明してはいない。

従って判断は`SCOPED_GAP_SURVIVES_NO_PRIORITY_CLAIM`とし、
[FR-1事前登録](finite_rte_phase_amplitude_fr1_preregistration.md)の小規模機構試験へだけ進む。
結果が有望でも、論文主張を固定する前にfinite LCU/RTE、nonunitary perturbation、
state-dependent phase estimationの追加検索を行う。

## 6. 参照URL

- https://arxiv.org/abs/2503.05647
- https://arxiv.org/abs/2110.12071
- https://www.nature.com/articles/s41534-022-00548-w
- https://arxiv.org/abs/2111.10430
- https://arxiv.org/abs/2606.30741
- https://doi.org/10.1137/0726012
- https://doi.org/10.1137/S1064827597315509
- https://arxiv.org/abs/2602.09575
- https://arxiv.org/abs/2308.00978
- https://arxiv.org/abs/2406.03033
