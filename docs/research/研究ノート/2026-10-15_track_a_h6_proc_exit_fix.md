# H6 /proc退出競合修正

H6 run03は全native検証後の退出/proc競合でSTOP、1 worker PASS/受理0・全6 reap。SOURCE2e733d8a…でlive欠測を成功扱いせず5秒内の再観測/reapへ修正、92限定＋最小2childの退出/FD検査PASS。旧消費保持。CPU gateで未起動2grantを廃止、5秒idle選定・fresh gate後にrun04-v3起動。科学/profile/compiler/資源上限不変、subagent0。完了は未主張。

[技術記録](../track_a_h6_proc_exit_fix_20261015.md)。
