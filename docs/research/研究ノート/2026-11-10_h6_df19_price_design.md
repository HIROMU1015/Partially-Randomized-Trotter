# 2026-11-10：H6 DF19 compiled価格の再利用設計

[REPORT](../h6_df19_price_design_20261110/REPORT.md)。同じH6 D2/m6 worktreeへの競合を避け、固定95d01b6aから独立文書branchを作成した。m3/4の直接価格とPR探索/確認を優先し、m0…2/m5以後の未知Cを0にしない。

H6 m5は11 signed piece、PR5/R32とPR7/R16の別族であり、全13roundの単一q-AffineやH4/Dの係数移植は不適切。最初の提案は保存価格だけのsource構造V1、PR2族それぞれ探索9と独立確認33を分離する。なお不足する場合だけ、δ1.6/q2,q4の新価格pilotを別途検討する。新予測/残差/fit/sampling/build/compile/signal/N/Gは0。V1も未実行。公開照合後STOP、次担当ユーザーが具体的科学scopeを判断する。
