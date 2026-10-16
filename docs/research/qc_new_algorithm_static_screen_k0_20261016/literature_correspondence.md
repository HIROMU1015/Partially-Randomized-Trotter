# 静的K0：一次文献と固定研究資料の対応

2026-10-16 JST。対象はH1–H4の既報性・access条件だけ。外部文献8件の関連節をbrowser本文/PDF textで読んだ。全論文の再レビュー、数値表の再集計、出版版との全差分照合ではない。以下の版を参照点とし、未取得版を代用品から復元しない。論文の原byte/PDF SHAを今回取得したとは主張せず、GitHub原byteのidentityとは区別する。

| ID・取得した一次原文 | 今回確認した箇所・仮定 | 対象／今回の照合結果と限界 |
| --- | --- | --- |
| L1 Hejazi–Soni–Shokrian Zini–Arrazola, *Better product formulas for quantum phase estimation*, [2412.16811v1 HTML](https://arxiv.org/html/2412.16811v1), 2024-12-22 | §I Eq.9、§II Eqs.10–13、§IV、Supplement VIIの所在。有限次元effective-Hamiltonianと対象branchのgap仮定、二blockのfast-forwardability | 固有値誤差のH交換子消去をPF生成に利用済み。全DFの安い二block化や本taskの有限h/copy費用は供給しない。v2 HTMLは404で、読めたv1を明示使用 |
| L2 Bagherimehrabほか, *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, [2409.08265v4 HTML](https://arxiv.org/html/2409.08265v4), 2026-07-28版 | Eqs.4–5,26–28、§5の関連箇所。共役processor／同符号correctorと分解費用を区別 | 純相似変換の固有値不変とkernel変更を混同しない。Eq.28の表示上の留保は既存H-E1にも記録済み。出版版全文との同一性や正式訂正文の不存在は今回認定しない |
| L3 Moreno Casaresほか, *Theory and practice of Trotter product formulas for quantum chemistry*, [2606.30741v1 HTML](https://arxiv.org/html/2606.30741v1), 2026-06-29 | §IIのGRADE/DF回路、§III、§V.3、Appendix E.3 Props.3–4 | 分割・回路単価・leading誤差・processingの強い既知対照。XAS評価を絶対E₀の総費用保証へ転用しない。QSVT等の節はD1外の構成として採用しない |
| L4 Bosseほか, *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, [Nature Communications 16,2673 (2025)出版HTML](https://www.nature.com/articles/s41467-025-57580-5) | Results: THRIFT algorithms、Eqs.2–3。安いexact H₀発展と弱い相互作用の前提 | interaction pictureによる時間依存相互作用は既報。Gaussian共役のみを独立生成としない。browser本文は読めたがpublisher redirect/cookie表示があり、原byteの完全保存・全supplement検査はしていない |
| L5 Moreno Casares–Shokrian Zini–Arrazola, *Quantum simulation of time-dependent Hamiltonians via commutator-free quasi-Magnus operators*, [2403.13889v3 HTML](https://arxiv.org/html/2403.13889v3)／[Quantum 8,1567 (2024)](https://quantum-journal.org/papers/q-2024-12-17-1567/) | §2.2 Eqs.15–18、§3.4、本文のbounded/time-regularity条件。§3の誤差分解の所在確認 | 時刻別Hamiltonianの和の指数と、そのPF実装までを既に扱う。各和がDF一平方のまま安価になるとは限らない。全定数の再導出・出版PDF同一性照合は未実施 |
| L6 Clintonほか, *Quantum Phase Estimation without Controlled Unitaries*, [2410.21517v2 PDF](https://arxiv.org/pdf/2410.21517v2), 2024-10-30,35 pages | §2.1、Appendix A pp.16–17、Appendix B Eq.29–31。多信号/干渉、support、準備/逆準備の条件 | Appendix Aで復元の定数shift不定性を確認。コピーだけで絶対原点まで得る定理ではない。v2 HTMLは取得不能、PDF textを読んだ。2026 PRX Quantum版との数学的diffは未確認 |
| L7 Kujiほか, *Robust phase estimation of the ground-state energy without controlled time evolution on a quantum device*, [2412.19590v2 HTML](https://arxiv.org/html/2412.19590v2)／[PRA 111,042618 (2025)概要](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.042618) | §II Eqs.1–10、Appendix B。共通保存量、古典的に扱えるreference、初期superposition、ASP/逆ASP | 既知reference energyで絶対値を校正する既知法。exact groundコピーは準備・逆準備unitaryを含意しない。出版ページは概要を確認、本文の読解はarXiv版。一般化学の全費用保証は未確認 |
| L8 Cohn–Motta–Parrish, *Quantum Filter Diagonalization with Double-Factorized Hamiltonians*, [2104.08957v1 PDF](https://arxiv.org/pdf/2104.08957v1),15 pages／[PRX Quantum 2,040352出版概要](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040352) | QFD節 Eqs.32–38・Fig.3、近似time propagationとHamiltonian/overlap matrix elementの回路 | H2の近接する既知枠組みとして限定追加。exact groundに対する単一projector比と、QFDの一般化固有値問題の完全同値は主張しない。copy・測定・condition number込みのH2優位を供給しない |

本文の式(1)–(11)は今回の紙上整理で、文献の式番号とは別。文献の表記を本報告の符号規約へ黙って移植していない。特に外側H交換子の消去、processorの相似変換、HadamardによるPauli項相関の取得と、低費用な新算法の成立を区別する。

## GitHubから取得した固定原本

以下13件はContents APIからbase64原byteを取得し、git blob SHAとSHA-256を記録した。[provenance.json](provenance.json)に省略のないrepository/commit/path/URL/bytes/SHAとlocal blob照合結果がある。本文は今回の公開packetへ重複保存しない。

| ID | commit | path・今回の参照範囲 |
| --- | --- | --- |
| G1 | `dbe51f1870cd069b952cc92593833d355da10e28` | `artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json`：正式状態、A2/K1-Fの境界 |
| G2 | `78b50c515d9df9556f261e74ca7c57af461b7199` | `docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md`：既存統合・他系列のscope |
| G3 | `58d8672702284d73321cc67a26c4640adaddde5f` | `docs/research/qc_sos_pf_k0_20261015/technical_audit.md`：§4–8のBCH、二分割、保証、価格 |
| G4 | `01bdc56a2970dcb749390b0540846143f775d6f9` | `docs/research/qc_sos_pf_k05_20261015/technical_report.md`：既知構成、有限stepの入力・剰余・gap |
| G5 | `78b50c515d9df9556f261e74ca7c57af461b7199` | `docs/research/qc_he1_stage0_technical_prevalidation_20261015.md`：H-E1生成差分、processing、既存source |
| G6 | 同上 | `docs/research/qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md`：§2–3の二分割・局所1-RDM還元 |
| G7 | 同上 | `artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json`：正式CLOSED/終了/STOP/PAUSED/HOLD |
| G8 | 同上 | `docs/research/qc_m_fixed_circuit_k0_technical_audit_20261015.md`：counterterm、native grammar、既知射影への還元 |
| G9 | `acfa8674d51ede1d5cd50f1b49873e67de5b71b9` | `docs/research/qc_existing_technical_foundation_map_20261016.md`：両repoのsource/実装と検証scope |
| G10 | 高次PF repo `cd81018932b9557b7a55e2e1cd8c5e3988245409` | `PF_first_study_final_synthesis_20260925.md`：既存高次PF研究との境界。数値は再集計しない |
| G11 | 高次PF repo `e0e82f6c7bee73d2360b77456bd22f7356572e01` | `src/trotterlib/product_formula.py`：既存係数生成sourceの静的読取り。import/実行なし |
| G12 | `58d8672702284d73321cc67a26c4640adaddde5f` | `docs/research/qc_sos_pf_k0_20261015/method_status.json`：既存技術判定の保持 |
| G13 | `01bdc56a2970dcb749390b0540846143f775d6f9` | `docs/research/qc_sos_pf_k05_20261015/literature_correspondence.md`：過去の版/ACM403/取得境界 |

## 未取得と未検証の明示

- Claude最新版`レビューoutter(3).md`原本は今回未取得。ユーザー提供GPT完成版を独立の検証対象として使い、その内部のClaude引用と原本を逐語照合したとは記さない。
- Kang–SuのACM版はK0.5当時403で未取得、arXivとの数学的diff未確認という履歴をそのまま保持。今回その文献を再調査・取得したとは記さない。
- arXiv HTMLがない版は、取得できた明示版/PDFを使用。Clinton出版版diff等の未確認部分は、他版の同一性を仮定して埋めない。今回の限定判定の根拠は表に示した取得済み版に限る。
- 数学的恒等式の成立、任意の多fragmentへの効率的還元、有用なfinite-step保証、全資源での独立差分は別事項。後三者を文献の題名や要旨だけから認定していない。

文献調査はここで完了。新しい候補や関連論文をさらに探索せず、**Mandatory STOP**。
