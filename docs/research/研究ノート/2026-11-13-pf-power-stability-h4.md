# 2026-11-13：H4 PF長時間の二経路診断

ユーザーが人工調査dd24dd329・元recipe256caad34の代表F2-COARSE/m12二経路診断を明示承認した。元A branchを継続し、専用runnerを4人工テスト・旧8回帰後にade6438f8へ固定した。[REPORT](../pf_power_stability_h4_20261113/REPORT.md)参照。

保存H4/1Å/STO-3G/tol-onlyDF10/sector(2,2)/36、Full PF S2/prefix10/T409.6/q1024/Δ.4、fragment・scalar・gate1e−9不変。旧spectralとdirect expmを各1回、最大2mean・新reference0で完了した。元経路はstate norm通過、operator unitarity1.0071230339680433e−9で失敗。directは5.899874925253737e−12で通過、mean差4.313865617047318e−11。失敗した元値も診断として保全し、後続の計画済みdirect経路を省略しなかった。

確認できたのは、同じeigensystemのGram defectとhalf-step/macro-step/powerの非unitarity蓄積、および代表1点でのdirect経路の有効性。物理biasも両経路で近く、数値安定性だけで粗いPFの精度が達成されたとはしない。旧run02の他失敗、M5/B0I/RTEの全面資格、十分N/global RMSEは未確定。NumPy1.26.4/SciPy1.14.1、12技術test PASS、6保存整合性確認。

元run01/run02・SOURCE/raw/KEEP/STOPを変更せず、B/C/D非介入。次はユーザーがPF/B0Iをdirect expm＋matrix_powerとする残15セル（Full PF7/B0I2/Full-H RTE6）の再開scopeを判断し、承認後にAが実行する案。Full-H RTEのfinite-K lawは変更しない。今回の最大2meanを超えず、公開・remote照合後mandatory STOP。
