# R0.5 primary-source identity, locator and algorithm table v1

2026-10-06 JST. 本表の機械版は
[primary_sources_v1.json](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/primary_sources_v1.json)。
P0/P1はordinary端点の確認、P2/P3は指定されたequivalence監査。
全論文/citation networkの不存在証明ではなく、明記した箇所とR0の同version本文確認に限定する。

## Identityと実際に確認した箇所

| ID | Title / authors | Version / journal | Sections/equations read |
|---|---|---|---|
| P0 | A randomized quantum algorithm for statistical phase estimation — K. Wan, M. Berta, E. T. Campbell | [2110.12071v2](https://arxiv.org/pdf/2110.12071v2); PRL 129, 030503 (2022) | Appendix C C4–C5、直後のnorm/phase式。finite cutoffはR0 Appendix F.2確認へ参照 |
| P1 | Phase estimation with partially randomized time evolution — Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl, Aram W. Harrow | [2503.05647v2](https://arxiv.org/pdf/2503.05647v2); journal版は今回主張しない | Appendix A.2 A18–A27、sourceとのsigned angle/order対応。R0の同version確認を継承 |
| P2 | Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations — Pei Zeng, Jinzhao Sun, Liang Jiang, Qi Zhao | [2212.04566v2](https://arxiv.org/pdf/2212.04566v2); PRX Quantum 6, 010359 (2025) | IV.A 42–50（R0と本stageの確認）、IV.B 66–79、IV.C sampling説明、Appendix A A1–A8 / Proposition 10と隣接proof |
| P3 | Quantum Simulation via Stochastic Combination of Unitaries — Joseph Peetz, Scott E. Smart, Prineha Narang | [2407.21095v2](https://arxiv.org/pdf/2407.21095v2); [npj Quantum Information 12, 52 (2026)](https://www.nature.com/articles/s41534-025-01168-w.pdf) | Theorem 1 / 5–8、Methods IV.B、Supplementary Note 3全体、Note 5 S13–S15とnormの説明 |

R0ではP3のexplicit-v2 fetchが失敗したが、今回は取得成功。R0の当時の記録は変更しない。
P3 v2 PDFは16頁、July 8, 2026表示。P2 v2は44頁、March 31, 2025表示。
journal identityは対応する書誌情報として記載し、journal/arXivの全byte一致とはしない。
publisher article pageの取得エラーと、publisher PDFによる書誌確認も分ける。
PDFをrepositoryへ転載せず、byte-level archived-paper fingerprintは今回取得していない。

## Target / information / decomposition / cost / sampling

| ID / scope | Target | Information access | Decomposition / norm | Atom class | Classical preprocessing / generation | Sampling semantics |
|---|---|---|---|---|---|---|
| P0 finite ordinary | P_m、元はinfinite exponential | 原典Pauli。single-Q Euler algebraはI0へ直接拡張可能 | even adjacent pairs; Σeven sqrt(t_n²+t_(n+1)²) | phase × word × elementary rotation。adjointでA左順へ対応 | degree table O(m)、p sampler、word生成O(m)。precision別 | IID word indices、positive coefficient/B。operator first mean |
| P1 finite RTE | P_(K+1)、元はexponential | 原典Pauli / repo DF elementary-involution contract | same even pairing / finite B_K | even phase、rotation last in circuit time | finite order table / IID indices。既存実装はodd非対応 | coherent Hadamard first moment、normalization correction。independent occurrences |
| P2 K=0 | s_c=mで同じP_m | elementary EulerのみならI0 | first two degrees paired, others pure; sqrt(1+x²)+Σn≥2 t_n | elementary rotation or phased unitary word | O(m) table / O(m) word生成、Pauli集約不要 | canonical coefficient/μ。原典observableはindependent forward/backward pair |
| P2 K>0 | V_K=U S_K†、またはcompensation times S_K | order/factorization、sampled Pauli multiplication/classification | leading groupsをIへpair; sqrt(1+etaSigma²)+higher tail | common-angle Pauli rotations / phased Pauli | multinomial/order/indices生成＋per-word Pauli phase/classification。全collection不要 | same remainder mean。AのP_mとの直接norm優劣は対象外 |
| P3 literal collected | C+I+iS=P_mへの有限specialization | I1、Hermitian real/imag Pauli coefficient collection | Lc+sqrt(1+Ls²) | signed Pauli or signed-Pauli common-angle rotation | literal word aggregationはO(L^m)規模の上界。sparse DP/構造で変わり、全方法に必須とはしない | coefficients/μでunitaryをsample。paperのchannelはleft/rightを独立sample |
| P3 Note 3 | exact finite lower blocksのproduct / selected correction | partial Pauli expansionsとその係数/norm | layersのnorm積。full collectionのμより増え得る | sampled lower-block atomsのproduct | lower-block expansionsとsampling tables。full expansionを省ける | independent layer meanはproduct。same collected Lc/Ls/angleは自動保存しない |
| 本監査で明示したCTS padded specialization | 同じP_m | I1のper-word closure/rephasing。I0としない | E-1+sqrt(1+O²) | signed Pauli / common-angle rotation、zero-sum paddingを残す | degree table O(m)、一word Pauli phase計算O(m N)。全係数collectionなし | raw degree / IID indices、anti-Hermitian subensembleを同じ-iでrephase。既知部品の解析specialization |

最後のrowはP3 Note3の記述をそのまま写したalgorithmではない。
P2で使われるzero-sum rephasingとP3のEuler pairingを同規約へ戻して構成した、
result-prior登録済みの比較用specialization。異なるdistributionを既知本文と同一だとはしない。
同様に、Φの角度やgroup weightsを変えただけでAのcoupled all-adjacent familyが
PTSCから直接得られるとはしない。

O(m)、O(L^m)等は指定されたalgebra/word処理のoperation count。
確率・角度の有限precision、native circuits、workspace、synthesis costを0と扱う根拠ではない。
DF異basis multiplicationはper-word global Pauli multiplicationとは異なる。
