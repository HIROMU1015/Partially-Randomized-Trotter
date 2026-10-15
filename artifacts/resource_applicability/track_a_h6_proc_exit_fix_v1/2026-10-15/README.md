# H6 /proc退出競合修正・run04

SOURCE `2e733d8a683d415792ce92a56c7d149b78f42716`。run03は6compile開始後のVmRSS欠測でSTOP、1 worker PASS、他5 owned連動停止、全6 reap。元proof・実消費を保持し再利用/返却/reset0。親はVmRSS/VmSize欠測時wait4で退出を確認し、未退出なら最終正常観測から最大5秒だけpending、継続欠測はSTOP。他field/error/所有不一致は成功扱いしない。独立observerもmemory field欠落でfresh leaseを発行しない。新8＋既存84の92限定tests、最小2child/全体3processのnative exit0/wait/reap/pipe閉鎖確認PASS。科学/provider/profile/input/compiler/上限・監視5秒不変、subagent0。新run04のfresh gate後に一度実行、次stage未認可。
