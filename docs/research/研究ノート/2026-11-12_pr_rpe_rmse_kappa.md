# PR–RPE RMSE/κ technical preparation（2026-11-12 JST）

原論文v2 B.2/E.3の経験的Nmとpaper回転数型κを確認。旧高確率型と別policyを追加し、共通κ→r_target→R_outer→既存ABS_TIME配分を生成。finite-K/signed/保存stateへのRMSE保証は未資格のまま。

Bの公開草案d562bc38dを取込み、共有mapper式と必須field、全schedule ID・PF κ=null/PR共通κを17人工mockで確認。初期H6 m0..3の旧prefix5/R32・δ=Tを保持。実Hamiltonianの新signal/N、保存値の新再解析、compile/C/G/最適化は0。

[REPORT](../pr_rpe_rmse_kappa_20261112/REPORT.md)・[HANDOFF](../pr_rpe_rmse_kappa_20261112/HANDOFF.md)。B生成controllerとの最終identity結合、finite-K→全RMSE校正と科学条件の選択は次段階。公開照合後mandatory STOP。
