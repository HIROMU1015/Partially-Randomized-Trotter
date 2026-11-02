# Track A H8：新PR2候補・独立48軌道の確認結果

2026-10-19 JST。事前に固定した2候補は、各48軌道で固定PF対照より低い**標本の点費用を再現**した。
合計96軌道・192軸の測定が完了した。追加のsampling・compileは開始しない。

## 条件と通常評価

linear H8・隣接距離1.0 Å・STO-3G・tol-only1e-8 DF rank28、16 spin modes・sector4900、T=.8・P=0。
元のDF fragment順序・二次partial PF・r4/K4の無条件finite-RTE lawを維持した。
p8/q2はδ=.4、p7/q16はδ=.05。full-Fock scalar一回適用・reference不回転・ordinary ancillaを用いる。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed_transpiler17。旧compiler outputの完全同一性は未検証。

通常signed bias・allowance倍率1。費用Gは保存shot数を固定した
`G_i = N_cosine C_i,cosine + N_sine C_i,sine`（compiled RZ×shot）。同じtrajectoryの両軸を組にして評価する。

| ε | PR候補 | n | PR平均G | paired SE | 固定PF対照 | PF費用G | PR削減率 |
|---|---|---:|---:|---:|---|---:|---:|
| .05 | B2_p8_q2_2nd_r4_K4 | 48 | 1,230,113,850.125 | 3,175,066.35 | B0I_p11_q1_2nd | 1,267,321,444 | +2.935924% |
| .0005 | B2_p7_q16_2nd_r4_K4 | 48 | 91,082,200,598,761.833… | 144,211,830,223.72 | B0I_p15_q4_4th | 94,193,016,652,860 | +3.302597% |

固定PF単価は直接実測済みの42,268 / 668,636 RZ（各軸）を再利用した。
探索n8・旧確認n32・条件付き希少pilotと標本を合算せず、確認結果に応じた候補差替えは行っていない。

## 副評価と限界

経験的非相殺・同じε/倍率1/P=0の副評価では、p8の固定PFは不適格のため比較null。
p7の固定PF費用は110,353,805,016,120、PR点削減率は17.463471%。厳密bias保証ではない。

観測eventはp8でTaylor0が384、p7で3072、Taylor2/4はいずれも0だった。
未観測の高次event費用を0とは扱わない。点費用の再現を支持する結果であり、
希少eventを含む母平均の優位性は未確定。CI・p値はnull、全184候補の実測最良も主張しない。
SEはpaired trajectory費用の標準誤差であり、量子測定shotの標準誤差ではない。
量子shot実行・実backend/noise・energy/RPEの最終総費用は対象外。

## 保存資料

[比較表：exact fraction・SD/SE・固定shot](../../artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv)、
[192軸の測定値](../../artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/paired_metrics.csv)。
保存済み集計を転記し、今回の公開作業で科学的な再計算・順位付けは行っていない。
元の同じ確認標本の完了済み2軸を再利用し、表では再利用を明示している。
[使用した公開SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/745a674546f2cb6258526fbfde1d9a00601689f8)。

両scopeの正常終了・回収・最終byte照合はローカルで確認済み。詳細監査・hash一覧・運用来歴と、
共通入口の追記は自動承認審査の公開拒否に従いローカルに保全し、この結果公開へ含めない。
独立科学レビュー・immutable CI・外部再現済みとは扱わない。
次はユーザーが科学的解釈と追加の母平均検証の要否を判断する。Codexは公開後mandatory STOPする。
