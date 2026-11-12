# 2026-11-12 PR–RPE 実RMSE評価前のA技術準備

A510/B82bcを固定し、metadata preflight・保存field inventory・B形式の未評価N packetを追加した。新規14＋旧29人工/回帰PASS。実H/stateでの新signal/N、保存数値RMSE再解析、C/G、sampling、P2/P3は実行していない。

B finalはH4 full-H PF baselineと固定L_Dを要求するため、H4旧B0/p7およびH6初期p5/R32を通常p7へ変換せず不一致を明示する。旧d562用packetは保全し、A/B configuration IDを分ける。保存signalが存在してもRMSEの全schedule誤差契約・state/finite-K/decoder資格は未知で、N/boundはnullを維持する。

次はユーザーが1系・固定(M,κ)/RMSE policy・二時点pilotの保存再解析/欠測科学call範囲を承認した後にAが実行する。Bはearly例外とPF baselineの統合仕様を整理する。今回の成果は[REPORT](../pr_rpe_rmse_science_preparation_20261112/REPORT.md)参照。公開照合後mandatory STOP。
