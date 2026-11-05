# 一次資料・既存証拠・確認範囲

調査日・cutoff：2026-11-06 JST。文献の発表年、arXiv version日、出版社の公開・受理日は別々に記す。全文取得と全文精読は区別する。本調査は関連箇所の読解であり、定理の独立再証明・詳細科学レビューではない。

入口は [REPORT.md](REPORT.md)、論文の方法・保証・費用の説明は [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md)。以下の `L01` 等はこの6文書内の参照名であり、新しい管理台帳ではない。

## 1. 既存調査の再利用と今回の読み方

固定基点は [3571477093441ac4852166fe829f8ccc5db5ec4f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3571477093441ac4852166fe829f8ccc5db5ec4f)。以下の相対リンクはその内容を継承したcheckoutで確認した。

| 参照 | 確認した資料・範囲 | 今回の役割・限界 |
|---|---|---|
| I-ENTRY | [PROJECT_MAP](../../../PROJECT_MAP.md)、[AGENTS](../../../AGENTS.md)、[研究概要・現状](../研究概要・現状.md)全文、[研究目的](../研究目的・研究課題.md)の目的・状態・signal条件 | 旧時点の本文と後続固定結果を区別。利用可能な正式共通指示2026-10-25の§8.1/8.2/8.4/10/10.3も確認し、指示原文は転載しない |
| I-STATUS | [VALIDATION_STATUS](../../../VALIDATION_STATUS.md)、[manifest](../../../artifacts/validation_manifest.json)の構造・関連status、[統一索引](../RESEARCH_RESULTS_INDEX.md) | 旧local/dirty証拠を外部再現・CI保証へ変えない。manifest全resultの再監査ではない |
| I-INTEGRATED | [固定統合directory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105)の指定5成果物 | REPORT・Gap・残検証を読解。CSV全129行のID/schemaと関連行の条件・status・bindingを静的読取。再利用地図の機能profileと固定資産表を参照。143群/179版を今回再実行・全件精読した意味ではない。REPORTはGitHub connectorでも固定refから取得 |
| I-LANDSCAPE | [70研究カード](../pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md)、[coverage](../pr_algorithm_improvement_landscape_20261023/COVERAGE.md)、[Phase 2A一次資料案内](../pr_algorithm_improvement_phase2a_20261023/PRIMARY_SOURCE_GUIDE.md)、同SOURCE_EVIDENCE_INDEXの外部文献項目、[強対照表](../pr_algorithm_improvement_phase2a_20261023/STRONG_BASELINE_AND_OVERLAP_MATRIX.md)関連節 | 70研究/7課題を再構築しない。Phase 2AのPR/SPRINT/RC-DF等はabstract・目次または部分節確認が中心だった。過去の確認深度を今回の本文読解と分ける |
| I-METHOD | [Phase 2C方法差](../pr_algorithm_improvement_phase2c_design_20261023/METHOD_DELTA_AND_PRIOR_ART.md)、[追加pairの強対照](../pr_algorithm_improvement_phase2c_method_delta_20261023/STRONG_BASELINE_COMPARISON.md)、[限定探索の終了判断](../pr_algorithm_improvement_phase2c_method_delta_20261023/NEXT_STAGE_DECISION.md) | mixed辞書・反可換pair・通常のbasis共有の既知対応を再利用。終了済みpair案を新候補にしない |
| I-SYNTHESIS | [11/1科学landscape](../pr_repository_scientific_landscape_20261101.md)、[claim-gap表](../pr_repository_claim_evidence_gap_matrix_20261101.md)の目的・系列・RPE関連節 | 固定TのGと全RPEの区別、後続資料への入口。11/5統合の更新を優先 |
| I-TRACKB | [Track B入口](../../tracks/algorithm_codesign/README.md)、[相関法の既存一次資料照合](../../tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md)、[合成配置の既存比較](../../tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md)関連節 | PRHS/qSHIFT、TE-PAI、IS、native配分の既知対応・留保を帰属付きで継承。旧レビューを今回再開催していない |
| I-CODE | [rte.py](../../../src/trotterlib/rte.py)のfinite distribution/normalization/event接口、[df_rte_qiskit.py](../../../src/trotterlib/df_rte_qiskit.py)のbuilder/control接口、[full_opt2](../../../src/trotterlib/research_direction_full_opt2.py)の保存証拠loader/再最適化入口、[RPE accounting](../../../src/trotterlib/rpe_resource_accounting.py)の半径・shot接口 | 静的読取のみ。科学moduleのimport・runner・test・保存値の再最適化は実行していない。後続decoder等は統合再利用地図の固定SOURCE参照を継承 |

### 過去の文献確認深度

| 過去資料での区分 | 代表例 | 今回の処置 |
|---|---|---|
| 一次本文の関連箇所まで確認済みという過去記録 | ISのTheorem 1/2、PRHS/qSHIFTの式・補足、unitary partitioningのoperator対応 | 対応箇所を再利用。今回直接読んだ節は下表に限定 |
| 一部節・abstract/目次のみ | PR v2、SPRINT、RC-DF、BLISS、partitioning、connectivity | 中心論点に必要な本文を選択追加。connectivityは要約・過去範囲のまま |
| 要約・二次資料だけ／文献名だけ | 引用網に現れる未展開の論文 | 読了・保証の根拠へ昇格させていない。本成果物の中心claimに使わない |
| 現版・後続が未確認 | TE-PAIの旧v2参照、連続時間PAIの旧abstract参照、UWCの出版状況等 | 現metadataを再取得。新しい版の全文差分を全節比較したとは主張しない |

## 2. 外部一次資料

`本文部分確認` は下記locatorの読解、`要約確認` はmetadata/abstractまで、`継承` は既存固定調査の引用である。arXivページのjournal-refが未確認/未記載であっても「未出版」と断定しない。関連節以外は未精読である。

| ID・タイトル・著者・年 | 確認版・出版情報・URL | 今回確認した箇所／重要claimへの入口 | アクセス・追加更新区分 |
|---|---|---|---|
| L01 **Phase estimation with partially randomized time evolution** — Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl, Aram W. Harrow（2025/2026） | [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2), 2026-07-10。[PRX Quantum 7, 020332](https://doi.org/10.1103/ynxb-p2xq), 2026-05-19 | PDF §II/IV/V/VII、App.Aの平均振幅、App.B.1のshot/attenuation・B30/B31、App.E冒頭のRMSE条件とE19–E24。E21がround合計、E22が費用model、E24以降がrounding-to-residual | PDF取得・部分確認。HTML v2は404でPDFに切替。既存PR比較を再利用し、v2の費用前提を本文で追加確認。出版社版とarXiv v2の全差分は未照合 |
| L02 **Theory and practice of Trotter product formulas for quantum chemistry** — Pablo A. M. Casares, William Maxwell, Danial Motlagh, Hitarth Choubisa, Zy Niu, Ignacio Loaiza, Jonathan E. Mueller, Arne-Christian Voigt, Juan Miguel Arrazola, Stepan Fomichev（2026） | [arXiv:2606.30741v1](https://arxiv.org/abs/2606.30741v1), 2026-06-29 | §IIのGRADE/漏洩、§III.2–6、§IVの選択手順、§VのXAS資源scope、App.E.3、F.1 Eqs.149–165、F.2、F.3 Eqs.177–178 | HTML取得・部分確認。Phase 2Aのabstract/目次から本文へ確認を拡張。処理・漏洩抑制定理の独立検証はしていない |
| L03 **Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime** — Shota Kanasugi, Riki Toshio, Kazunori Maruyama, Hirotaka Oshima（2026） | [arXiv:2603.22778v2](https://arxiv.org/abs/2603.22778v2), 2026-03-30。[出版社受理ページ](https://journals.aps.org/prxquantum/accepted/10.1103/863w-jxzp)：2026-10-28受理、題名に **via unitary weight concentration** を追加 | §II.2.4 Eqs.17–18、§III.2–3 Eqs.28–34とround依存最適化の段落、§IV/Vのモデル資源scope | HTML本文部分確認＋受理metadata更新。受理原稿とarXiv v2の全本文同一性は未確認。外部論文のUWCはrepository旧UWC結果の来歴回復ではない |
| L04 **Resource-Optimal Importance Sampling for Randomized Quantum Algorithms** — Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı（2026） | [arXiv:2603.13495v1](https://arxiv.org/abs/2603.13495v1), 2026-03-13 | §II Eqs.7–13/Theorem 1、§II.1 Eqs.18–20、§III Theorem 2/Eqs.21–26、§IV ZeroFill/Discardの定義 | HTML取得・部分確認。既知最適lawを再利用し、加法・独立層での利益縮小と全trial条件を追加確認。finite-confidence RPE最適性の定理ではない |
| L05 **Importance sampling for stochastic quantum simulations** — Oriel Kiss, Michele Grossi, Alessandro Roggero（2023） | [arXiv:2212.05952v2](https://arxiv.org/abs/2212.05952v2), 2023-04-06。[Quantum 7, 977](https://quantum-journal.org/papers/q-2023-04-13-977/) | PDF §3.2/3.3のbias・concentration条件、§4.1 Eqs.50–55/Theorems 4–5、核格子modelの数値scope | PDF・出版社要約取得、本文部分確認。per-simulation savingと独立experiment数の増加を区別して再利用 |
| L06 **Composite Quantum Simulations** — Matthew Hagan, Nathan Wiebe（2023） | [arXiv:2206.06409v3](https://arxiv.org/abs/2206.06409v3), 2023-07-20。[Quantum 7, 1181](https://quantum-journal.org/papers/q-2023-11-14-1181/) | 出版社abstractのpartition・diamond距離保証。PDF取得済みだが本調査では全定理読解を追加していない | 要約確認＋既存比較継承。coherent PR保証を新たに移さない |
| L07 **Synthesizing efficient circuits for Hamiltonian simulation** — Priyanka Mukhopadhyay, Nathan Wiebe, Hong Tao Zhang（2023） | [arXiv:2209.03478v3](https://arxiv.org/abs/2209.03478v3), 2023-03-08。[npj Quantum Information 9, 31](https://doi.org/10.1038/s41534-023-00697-6) | PDF pp.5–9、Algorithm 1のgreedy割当、可換lexicon/部分割当・qDRIFT比較。出版社のAlgorithm 1/Discussionも検索取得 | 本文部分確認。出版社本文直開きの認証redirect・PDF直downloadのHTML応答は有効本文と数えず、arXiv PDFへ切替。既存cost-aware対応を補強 |
| L08 **TE-PAI: Exact Time Evolution by Sampling Random Circuits** — Chusei Kiumi, Bálint Koczor（2025） | [arXiv:2410.16850v3](https://arxiv.org/abs/2410.16850v3), 2025-09-24。[Quantum Sci. Technol. 10, 045071](https://doi.org/10.1088/2058-9565/ae1160) | §IIのTheorems 1–2、measurement overhead/Eq.12、Loschmidt/statistical PEへの言及、App.Aのoperator平均法との区別 | HTML部分確認。旧v2を使う既存調査から現v3を確認。channel/ancillaをそろえる必要は既知 |
| L09 **Continuous-time evolution via probabilistic angle interpolation and its applications** — Tomoya Hayata, Yuta Kikuchi（2026） | [arXiv:2604.02854v3](https://arxiv.org/abs/2604.02854v3), 2026-10-23。metadataにPRA受理版・Algorithm 1訂正と記載。巻頁・DOIは本調査で未確認 | §II.1 Eqs.7–9、§III.1/III.3、App.A Eqs.24–25、App.B Algorithm 1、App.C Algorithm 2。Hadamard部分はTETRISを用いる | HTML部分確認。既存のabstractのみの参照から本文・改訂情報を追加。旧Algorithm 1の訂正差分全行は未照合 |
| L10 **Continuous Hamiltonian dynamics on digital quantum computers without discretization error** — Etienne Granet, Henrik Dreyer（2023/2024） | [arXiv:2308.03694v2](https://arxiv.org/abs/2308.03694v2), 2023-11-17。出版題名は **Hamiltonian dynamics…**、[npj Quantum Information 10, 82](https://doi.org/10.1038/s41534-024-00877-y)（2024） | 今回は公式abstract/metadata。coherent TETRIS recipeの直接読解はL09 App.C、既存Track B比較を明示して利用 | 原著要約＋L09の具体的引用構成を確認。原著全本文の新読了とはしない |
| L11 **Split-Evolution Quantum Phase Estimation for Particle-Conserving Hamiltonians** — Megan Cerys Rowe, Carlo A. Gaggioli, Ludmila Szulakowska, David Muñoz Ramo, David Zsolt Manrique（2026） | [arXiv:2604.14921v2](https://arxiv.org/abs/2604.14921v2), 2026-06-10 | PDF §II Eqs.1–9の共通固有基底・真空reference、§III Eqs.18–19の費用範囲、§IV/FeMocoと§Vのethylene実験scope | PDF部分確認。指定旧調査集合の語検索で明示対応を確認できず、今回の比較へ追加。repository全体で初発見とは主張しない |
| L12 **Programmable Signal Design for Quantum Phase Estimation via Quantum Signal Processing** — Zikang Jia, Suying Liu, Yulong Dong（2026） | [arXiv:2604.01205v1](https://arxiv.org/abs/2604.01205v1), metadataの提出日2026-04-01（本文見出し日付は2026-08-24） | §II Eqs.1–3、§III Algorithm 1/Theorem 1、§V/VI、App.S4 Theorems 12–13、S6のcontrolled oracle/shift-rescale条件 | HTML部分確認。信号設計とquery-cost対照を今回追加。原oracleでの定理をfinite-RTEへ移す証明はない |
| L13 **phase2: Full-State Vector Simulation of Quantum Time Evolution at Scale** — Marek Miller, Jakob Günther, Freek Witteveen, Matthew S. Teynor, Mihael Erakovic, Markus Reiher, Gemma C. Solomon, Matthias Christandl（2026） | [arXiv:2504.17881v2](https://arxiv.org/abs/2504.17881v2), 2026-07-16。[Communications AI & Computing 1, 5](https://doi.org/10.1038/s44488-026-00002-2) | MethodsのHamiltonian simulation/preparation/Data analysis Eqs.7–10、SupplementのCAS_08手順・Table S2と非収束注記 | arXiv HTML部分確認。出版社本文のredirectはarXiv著者版で補完。classical simulator高速化と量子資源削減を分離 |
| L14 **Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization** — Oumarou Oumarou, Maximilian Scheurer, Robert M. Parrish, Edward G. Hohenstein, Christian Gogolin（2024） | [arXiv:2212.07957v3](https://arxiv.org/abs/2212.07957v3), 2024-06-04。[Quantum 8, 1371](https://doi.org/10.22331/q-2024-06-13-1371) | HTMLのfactorization、RC-DF最適化、`C.4 RC-DF for error corrected quantum computing`、CCSD(T) heuristic/古典最適化の段落 | 部分本文追加。HTMLの主節にAppendix表示が付くため節名で指定。CCSD(T)誤差を厳密な元H基底energy証明にしない |
| L15 **Block-Invariant Symmetry Shift: Preprocessing technique for second-quantized Hamiltonians to improve their decompositions to Linear Combination of Unitaries** — Ignacio Loaiza, Artur F. Izmaylov（2023） | [arXiv:2304.13772v3](https://arxiv.org/abs/2304.13772v3), 2023-10-24 | §II Eq.4の対象symmetry block保存、§III・App.AのLCU比較scope | HTML部分本文追加。今回確認したarXiv metadataにはjournal-refの記載なし。全空間の基底順位保存は主張しない |
| L16 **Assessment of various Hamiltonian partitionings for the electronic structure problem on a quantum computer using the Trotter approximation** — Luis A. Martínez-Martínez, Tzu-Ching Yen, Artur F. Izmaylov（2023） | [arXiv:2210.10189v2](https://arxiv.org/abs/2210.10189v2), 2023-08-07。[Quantum 7, 1086](https://doi.org/10.22331/q-2023-08-16-1086) | §2.1–2.2、2.5、§3の回転/誤差交換、App.Aのadaptive QPE費用scope | HTML部分本文追加。近似上界・rotation proxyと本研究のcompiled full RPEを区別 |
| L17 **Unitary partitioning approach to the measurement problem in the Variational Quantum Eigensolver method** — Artur F. Izmaylov, Tzu-Ching Yen, Robert A. Lang, Vladyslav Verteletskyi（2019） | [arXiv:1907.09040v2](https://arxiv.org/abs/1907.09040v2), 2019-10-18 | §II.1 Eqs.5–7の反可換和のunitary化。coherent pairの比較はI-METHODを継承 | HTML部分確認。VQE grouping/測定削減の保証をPR shot削減へ移さない。出版情報の追加確認なし |
| L18 **Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation** — Shelby Kimmel, Guang Hao Low, Theodore J. Yoder（2015、訂正2021） | [arXiv:1502.02677v3](https://arxiv.org/abs/1502.02677v3), 2021-10-21。[PRA 92, 062315](https://doi.org/10.1103/PhysRevA.92.062315)、[Erratum PRA 104, 069901](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.069901) | PDF冒頭Errata全文。旧§V/App.Cの分枝失敗解析と訂正後の適用範囲を区別。出版社訂正文も確認 | 訂正本文を今回明示。π/(2k_j)からπ/(3k_j)への条件修正を引用するだけで、既存decoderの合否を再判定していない |
| L19 **Quantum Simulation of Electronic Structure with Linear Depth and Connectivity** — Ian D. Kivlichan, Jarrod McClean, Nathan Wiebe, Craig Gidney, Alán Aspuru-Guzik, Garnet Kin-Lic Chan, Ryan Babbush（2018） | [arXiv:1711.04789v2](https://arxiv.org/abs/1711.04789v2), 2018-02-03。[PRL 120, 110501](https://doi.org/10.1103/PhysRevLett.120.110501) | 公式abstract/metadata、旧I-LANDSCAPEに記録された表現・swap-network/Givensの限定読解 | 要約再確認＋既存部分読解継承。任意DF-PR/routingの一般定理として使わない |
| L20 **Phase Estimation with Compressed Controlled Time Evolution** — Erenay Karacan（2026） | [arXiv:2511.21225v2](https://arxiv.org/abs/2511.21225v2), 2026-04-08。[PRA 113, 042420](https://doi.org/10.1103/j26w-w48j) | 公式abstractのtranslation-invariant local条件、control/depth、spin-model実験scope | 要約確認のみ。周辺対照として追加し、DF化学への適用可能性や定理詳細は確定しない |
| L21 **A random compiler for fast Hamiltonian simulation** — Earl Campbell（2019） | [arXiv:1811.08017v2](https://arxiv.org/abs/1811.08017v2), 2019-06-25。[PRL 123, 070503](https://doi.org/10.1103/PhysRevLett.123.070503) | 公式abstract/metadata、PR・IS23との引用関係 | 要約確認＋既存対応継承。qDRIFT channelと平均振幅の違いをL01/L05と合わせて読む |
| L22 **Pathwise Random Hamiltonian Simulation** — Davide Cugini（2026） | [arXiv:2608.29756v1](https://arxiv.org/abs/2608.29756v1), 2026-08-30 | 今回はmetadata/abstract。main Eqs.2–8/Supplement A–Cの詳細はI-TRACKBの固定照合を継承 | 既読範囲の再利用・現版確認。符号・normalization・sampling取得費・RPE接続の旧留保を維持 |
| L23 **qSHIFT: An Adaptive Sampling Protocol for Higher-Order Quantum Simulation** — Sangjin Lee, Sangkook Choi（2026） | [arXiv:2604.26263v3](https://arxiv.org/abs/2604.26263v3), 2026-09-02。既存取得PDFの題名は **Adaptive sampling based quantum simulation** | 今回はmetadata/abstract。III.1/Algorithm 1、Eq.17、B.2等の旧照合はI-TRACKBへ帰属 | 現版確認＋継承。旧v1/v3表示差・残留不整合を今回訂正済み扱いにしない。classical L^r workの留保 |

## 3. 重要claimとlocator

| 本調査が使う事項 | 直接の根拠 | 本調査で確定しないこと |
|---|---|---|
| PR原著にも全round費用とsynthesis/tail交換がある | L01 E21–E24、App.B.1 | RMSE用の経験的定数を任意失敗確率の十分Nへ流用すること |
| 表現・split・RPE費用の結合、round別表現への言及は既知 | L03 Eqs.17–18、33–34と直後の段落 | round別表現が常に有利、全wrapperで最適という主張 |
| randomized PF/合成は位相shift以外に減衰も変える | L02 App.F.1/F.3 | spectroscopy解析をそのままRPE成功定理にすること |
| cost-aware ISは単回価格と二次momentを結ぶ | L04 Eq.7/Thm.1。L05 Eqs.50–55 | 有限confidenceで同じlawが最適、誤差biasが消えるという主張 |
| 連続時間PAIのenergy例のHadamard部分はTETRIS | L09 §III.1/App.C Algorithm 2、L08 App.A | TE-PAI system-channelを現PRのcoherent oracleに直接置換できること |
| SE-QPEは共通固有基底・reference位相を条件に分布を保存 | L11 §II Eqs.3–9、§III Eqs.18–19 | 異なるrandom pathや異なるPF/splitでも条件が自動成立すること |
| QSP信号のquery保証はcontrolled unitaryを前提とする | L12 Thm.1、S4、S6 | 非unitaryなfinite-RTE平均が同じ入力oracleになること |
| RPEの旧失敗条件には訂正がある | L18 v3 Errata、出版社訂正 | repository既存結果が誤りという新判定。互換性確認は未実施 |

## 4. 探索範囲と未確認範囲

旧台帳と引用網を先に読み、一般検索・arXiv・Quantum/APS/Natureの一次ページを使用した。検索軸は部分ランダム化の後続、RPE resource/shot、cost-aware IS、representation/synthesis、controlled evolution、2026年のphase-estimation signalである。日付制限付き検索も用い、11月6日後の資料は根拠に使っていない。Google Scholarの完全な被引用集合は取得しておらず、世界の全関連文献を網羅した新規性調査ではない。

SE-QPE/QSP-PE/phase2/圧縮controlは指定旧調査directoryとTrack B等で語検索した範囲では明示対応を見つけず、本比較表への追加とした。名称変更・別branch・非公開資料に既存参照がないという不存在証明ではない。TE-PAI/TETRIS/PRHS/qSHIFTは既存参照があり、「新発見の方法」としない。

論文PDF/HTML・抽出textは一時領域でのみ使用し、public成果物に転載しない。全論文の全節精読、publisher最終版の全行差分、著者コードの実行、数値の再現、旧private rawの現存監査は行っていない。取得できなかった形式と代替一次資料は表に記した。
