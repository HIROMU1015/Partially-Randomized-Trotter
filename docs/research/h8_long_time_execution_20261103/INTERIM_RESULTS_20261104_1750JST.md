# H8長時間・高次PR：中間集計

2026-11-04T17:50:35.041683+09:00 の保存済み結果。

Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF28、元state・fragment順序0..27。
T=1.6、ε=0.0005、P=0。PRはR_outer32/K4。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/topologyなし。
費用G=N_cos C_cos+N_sin C_sin。measurement付きfull Hadamard wrapperのRZ数を固定Nで重み付けする。全RPE roundの最終総費用ではない。

Stage A：69候補（適格40、不適格29）。Stage B：保存PASS 49/51標本、代表検証完了 17/19候補。

| 方式 | prefix | q | 保存標本/予定 | 平均G（RZ） | 最良の測定済みPF比 | 状態 |
|---|---:|---:|---:|---:|---:|---|
| B0I 4th(m5_best) | 15 | 1 | 1/1 | 8.9291220e+13 | 1.00000 | 探索群完了 |
| PR 2nd | 7 | 32 | 9/9 | 3.9845944e+14 | 4.46247 | 探索群完了 |
| PR 2nd | 8 | 32 | 8/9 | 5.1741416e+14 | 5.79468 | 暫定・代表検証待ち |
| PR 4th(m5_best) | 7 | 1 | 9/9 | 1.0867626e+14 | 1.21710 | 探索群完了 |
| PR 4th(m5_best) | 8 | 1 | 8/9 | 6.7370816e+13 | 0.75451 | 暫定・代表検証待ち |

PF15候補は全て代表検証済み。測定した範囲で最小はB0+I / m5_best / prefix15 / q1。
PR prefix8の2群は元index0がHOLDとなり、同じ標本を位相補助SOURCEで再検証中。8標本だけの値は利用可能標本の記述値であり、9標本の平均・確定した比較・母集団での優位を意味しない。
PR prefix7は各9標本と代表7検査が完了。m5はq1、S2はq32だが、qだけでなく固定shotと一体compiled価格を含むGで比較する。
sample SEはpaired Gの軸間共分散を含む記述統計で、希少次数の母集団保証や確認cohortの精度を示さない。PFの1回価格には標本SEを作らない。

Signal SOURCE `d1272016592ff06c0567862f770ed796fd61b366`、元native SOURCE `df93fbb88761b61c2691bf01f761ed20a7987d7f`、位相補助SOURCE `f04ef60c960fd2a68bcf0506f4c505f0bc872b78`。
元SOURCE・raw/QPY・失敗/HOLD・seed/event列は保全し、稼働中の科学コードは変更していない。新しい標本は作成していない。

[全19候補CSV](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/interim_20261104_1750JST.csv)、
[固定N・paired統計・49標本の匿名価格metadata](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/interim_20261104_1750JST.json)、
[集計コード](../../../scripts/summarize_h8_long_time_progress.py)。

次の担当：Codexが稼働中2代表の検査結果を確認し、全51標本が検査を満たした場合に完全な探索集計を行う。Stage Cは未実行で、別途条件と実行許可が必要。
