# H6 /proc退出競合修正・run04

SOURCE `2e733d8a683d415792ce92a56c7d149b78f42716`。run03は6compile開始後のVmRSS欠測でSTOP、1 worker PASS、他5 owned連動停止、全6 reap。元proof・実消費を保持し再利用/返却/reset0。親はVmRSS/VmSize欠測時wait4で退出を確認し、未退出なら最終正常観測から最大5秒だけpending、継続欠測はSTOP。他field/error/所有不一致は成功扱いしない。独立observerもmemory field欠落でfresh leaseを発行しない。新8＋既存84の92限定tests、最小2child/全体3processのnative exit0/wait/reap/pipe閉鎖確認PASS。科学/provider/profile/input/compiler/上限・監視5秒不変、subagent0。新run04のfresh gate後に一度実行、次stage未認可。

[技術記録・新起動状態](../../../../docs/research/track_a_h6_proc_exit_fix_20261015.md)。有効grantは`rebinding_v3/`、SOURCE不変。run04-v1/v2はCPU gateで未起動のまま廃止。run04-v3はlauncher1470933/driver1470956で起動、driver CPU62/workers61,60,59,56,49,47。起動は完了の主張ではない。

`running_snapshot_v1.json`は時刻付き正常稼働snapshot（完了ではない）。`publication_status_v1.json`：non-force pushはGitHub認証エラー、remote branch未作成、認証/設定変更0。計算は独立して継続する。
