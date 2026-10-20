# 文献探索ログ

基準日2026-10-21 JST。web search/open/findと一次配布元PDF/HTML取得、Crossref DOI確認を使用。検索結果の自動要約は索引として用い、本文確認の代替にしない。検索件数・検索語数は科学的な網羅性指標にしない。

## 実際に用いた主要検索語

- `"Phase estimation with partially randomized time evolution"`
- `"Composite quantum simulations" qDRIFT`
- `Hamiltonian simulation randomized product formula 2025 2026 partial randomization`
- `"When is randomization advantageous in quantum simulation" arxiv`
- `"Theory and practice of Trotter product formulas for quantum chemistry"`
- `"Optimal Lower Bounds for Hamiltonian Simulation"`
- `"random compiler" Campbell 2019 1811.08017 DOI`
- `"Composite Qdrift-product formulas" 2024`
- `"Hamiltonian simulation with" "Wan" "Berta" "Campbell"`
- `"phase estimation" "randomized Hamiltonians" Kivlichan`
- `"Randomized quantum algorithm for statistical phase estimation" arxiv`
- `"Compilation by stochastic Hamiltonian sparsification" arxiv`
- `"Low rank representations for quantum simulation of electronic structure" arxiv`
- `"A partially random Trotter algorithm" arxiv`
- `"Ordering Matters: Structure, Accuracy and Gate Cost" 2505.04240`
- `"randomized" "Hamiltonian simulation" "multilevel" 2026`
- `"Hamiltonian" "simulation" "average-error" 2026`
- `"Quantum simulation via stochastic combination of unitaries" arxiv`
- `"Greatly improved higher-order product formulae" arxiv`
- `"Randomizing multi-product formulas" arxiv`
- `Peetz Smart Narang "stochastic combination" arxiv`
- `"Efficient and practical Hamiltonian simulation from time-dependent product formulas" arxiv authors`
- `"Qiskit" "analytical" "gate count" "Hamiltonian simulation"`
- `"Hamiltonian simulation" "compiled" "resource estimation" 2025`
- `"Phase estimation" "partially randomized" -site:reddit.com -site:researchgate.net -site:emergentmind.com`
- `"Estimating Trotter approximation errors to optimize Hamiltonian partitioning"`
- `"How to Use Quantum Computers for Biomolecular Free Energies" arxiv`
- `"Faster Quantum Chemistry Simulations" "Active Volume" arxiv`

## 引用関係の追跡

P01の§I B/参考文献からP02/P03/P04/P06/P07/P08/P10/P12/P14/P21/P23/P24/P39へ辿った。P03導入からcoalescing・SparSto・Jin–Liを追跡した。2026年PRの応用P26/P37と、独立したregime/variance/構造化PF系列P25/P27/P29/P30/P31/P32を追加した。sourceのタイトル・著者・日付はarXiv/出版社/Crossrefへ戻って確認した。

検索に現れたブログ、ResearchGate、AI要約、掲示板、二次レビューの論評を科学的結論の根拠にしない。特に2026年文献の新規性・lower boundを二次要約から直接採用していない。web上のQDKドキュメントのsee-also表記は正式書誌の代替にしない。

## 取得範囲

[論文catalog](literature_catalog.json)の`reading_scope`が精読/abstract評価の境界。arXiv 39件、P01のv1/誌版、Morales v2、FreeQuantum誌版、既知HTMLとDOI書誌を取得。HTTP失敗を含む個別URL・SHAは[literature_acquisition](literature_acquisition.json)と[追加取得](supplementary_acquisition.json)。

`arxiv.org/html/2503.05647v3`は存在しない版への試行で404、v2 HTMLも404だったため、landingのversion historyを確認してv1/v2 PDFと誌版を使用した。存在しないv3の内容を引用していない。

## 文献ライブラリの保全

39論文を先行研究の不存在の証明に使わない。背景文献の全本文・全引用先・全Supplementを網羅していない。本報告の新規性は「同じものが見つからない」ではなく、取得した一次文献の具体的な構成・資源指標・比較条件と、固定研究証拠の差に基づく。
