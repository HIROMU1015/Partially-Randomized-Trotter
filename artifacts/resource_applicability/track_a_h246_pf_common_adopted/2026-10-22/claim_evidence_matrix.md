# Claim–Evidence Matrix: common588 supplement

正式採用は有限共通域の保存点結果に限定する。新しいCEMは[CSV](claim_evidence_matrix_v2.csv)を正とし、旧14行のbyte prefixと追加7行を保持する。
旧行のパス・相対audit linkは[固定旧manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1923fce8f7461513d1bc47b5061e87cd70917e87/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/paper_manifest.json)の文脈で読む。旧主張のcohort・DF・精度・状態を変更しない。
旧CEM overlayも[新paper manifest](paper_manifest.json)で固定参照し、履歴を追跡できる。

| 追加ID | 正式採用する保存結果 | 証拠 | 限定 |
|---|---|---|---|
| COMMON588_DOMAIN | H2 84/H4 228/H6 276、全588評価・未解決0 | [dataset](dataset.json.gz)、[catalog](source_catalog.json) | q≤8・登録prefix・3公式、全候補compileではない |
| COMMON588_H2 | 全5精度で全公式PF最良がPR探索最良より低費用 | [比較B15行](comparisons_all_formulas_15.json) | rank3、p0..2/Full3、広域q16とは別 |
| COMMON588_H4 | .001/.0005でPR低費用、緩い3精度はPF低費用 | [比較B15行](comparisons_all_formulas_15.json) | rank10、p1..9/Full10、旧DF12と非混合 |
| COMMON588_H6 | .001/.0005でPR低費用、緩い3精度はPF低費用 | [比較B15行](comparisons_all_formulas_15.json) | rank19、p5..15/Full19、旧COST終了不明を保持 |
| COMMON588_WITHIN | 同一公式45比較を別に保存 | [比較A45行](comparisons_within_formula_45.json) | m5内の優位を全公式比較へ移送しない |
| COMMON588_H6_DENOMINATOR | H6 m5 prefix拡張比1、公式拡張が主な分母変更 | [保存分解10行](denominator_decomposition_10.json) | 入れ子の有限保存値、単独因果・母平均保証なし |
| COMMON588_SCOPE | SOURCE・取得区分・中断/回復履歴を含め採用 | [台帳v4](evidence_ledger_v4.csv)、[catalog](source_catalog.json) | 旧exit/reap・rare bound null、独立確認と非混合 |

linear1 Å/STO-3G/tol-only1e-8、rank3/10/19、T=.8/P=0、S2/Yoshida4/m5_best、q1/2/4/8、δ=.8/.4/.2/.1、signed bias・経験的allowance、指定full wrapper/compilerに限る。
PR平均・分散・SE・nは保存EXPLORATION outer32/K4/prefix1/3/5から転記し、新規解析・順位選択を行わない。
選択効果未補正、未観測希少event、厳密bias certificate・母平均順位保証・大域最適性・全RPE費用保証の欠如を保持する。
既存主表・Figure2不変、AD03保留、最終evidence freeze未実施。
