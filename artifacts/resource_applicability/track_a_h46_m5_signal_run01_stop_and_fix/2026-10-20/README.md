# H4/H6 m5 signal run01 STOP and empty-tail fix

Run01 started once after approval, then both systems stopped during prefix context initialization.
No candidate was reserved or evaluated. External launcher exit1/reaped; both workers wrote STOP and exited0/reaped.
Original SOURCE480ca450 and run01 are immutable. Fix SOURCEbec018c6 is in a separate worktree.
18 artificial tests pass, including real helper endpoint regression. The new grant is INVALID.
No retry, compile, sampling, accounting, ranking or push. Mandatory STOP.
