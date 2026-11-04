# 研究C：公平比較監査と近接目的設計の統合

2026-11-03 JST。技術統合であり、新しい科学結果・正式科学レビューではない。

[固定A：保存XDF/CDF監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8375433869999c2e10501b1983c11b2b22c5d423/docs/research/research_c_strong_xdf_cdf_comparison_audit_20261103.md)と
[固定B：UWC/RC-DF/SCDFと近接目的](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2459e018ea2276c359d6fe96253fbcad0bfd65bb/docs/research/research_c_uwc_rcdf_near_objective_comparison_design_20261103.md)を基礎とする。
Bのoracle SOURCEは`acc32878daa9591597a4677b459e5454dd96aeda`、stable optimizerは`025d495166311a58ec9c24c43f07ed6e40f65f42`。

| 判断に必要な点 | 現在の証拠と、今回への含意 |
|---|---|
| 中心的な肯定結果 | 保存H4直鎖1Å/STO-3G、number CDF8/sector36、update064、元順LD2、PR S2/s1/r32/K0、δ=h=.1/m0..12、own-DF ground oracle/状態準備除外、P0/fenced serialでは、同じJ/Uの強決定論S2に対する条件付き総RZ比 .95518、nominal95%近似区間 [.94659,.96377]。独立価格256件の限定結果。 |
| 保存XDF8との関係 | 元Hamiltonian/共通CA/RPE/P0/会計範囲は対応するが、J/U、達成DF誤差、κ基底、ground/PF、合成誤差資格、PRのideal/assumed/evidence laneが異なる。上記結果から保存XDF8への優位・同等性は判定できない。 |
| 既知の構成要素 | 一般CDF、signed J/軌道U、tensor loss、係数正則化、inner/outer、PR向けtail集中はUWC/RC-DF等と重なる。これらを独自原理とは数えない。Bの一次資料整理を再利用し、新しい網羅的レビューは行わない。 |
| 具体的な方法差 | 固定LD2 tailのnumber I/Z/ZZ写像後の非identity smooth massを二乗する。Jの行和の符号相殺を含む。同じtailのweighted entrywise L1二乗はそのtriangle envelopeで、相殺を取り込まない。 |
| 最も直接的な問い | 同じJ/U自由度、rank、tensor loss、tail、物理予算で、この写像後normを使う有限手順がentrywise対照より有用な候補を得るか。λ低下だけでPR量子資源の改善とはしない。 |
| 今回の比較II | 共通の元CDF8初期値から、同じ有限trial/solve/受理更新の機会を与える。物理DF誤差・exact λ・到達度・計算負担を記録する。未収束もその有限手順の結果として残す。 |
| 今回では判断できないこと | 目的関数の最適能力（比較I）、最適強度、全初期値/他分子への汎化、UWC全体への優位、保存XDF8への優位、厳密CA、最終総資源、新規性/論文化GO。 |

数値欄は[既存価格結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md)の引用である。保存配列やcohortの新しい再解析ではない。
「比較I/II」はBと今回の依頼の用語。Aの監査内の比較番号とは別の分類である。

Bの定義を用いると、exact massは
`λ = Σ_tail,i |Σ_j J_ij| + 1/4 Σ_tail,i |J_ii| + Σ_tail,i<j |J_ij|`、
entry envelopeは`B0 = 5/4 Σ_tail,i |J_ii| + 3 Σ_tail,i<j |J_ij|`。
その差は行和の相殺gapであり、smoothingだけの差ではない。
符号相殺がない領域でのexact同値性と、一般signed Jでの非同値性を分ける。

次の成果は[比較IIの技術準備](research_c_near_objective_practical_comparison_preparation_20261103.md)と
[未承認Stage O契約](research_c_near_objective_practical_comparison_contract_unapproved_20261103.json)。
今回の技術作業では分子の科学callは0。A/B・旧Claim・SOURCE/raw/STOPは保全する。
