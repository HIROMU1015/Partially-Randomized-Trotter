# PR-Pauli continuation after a polling shutdown fault

The parent is search-b1a74e00581b89ea42db at controller SOURCE
95a74e4486131a97bd5235221bf2d8ab9f99f591. Its queue settled all 24
attempts before the polling daemon exceeded its shutdown wait. The
worker recorded connection refusal and a clean, non-pending stop
immediately after the controller exited. The gateway had been using an
intermediate method result as the terminal result of the failed driver.

The controller now closes the worker endpoint before waiting on an idle
poller. If the daemon still does not exit, only the owned poller is
reaped, after the queue and spool agree that all local attempts settled.
A failed invocation is reported as a hold. The continuation contract
checks the previous SOURCE, exact scope, STOP, owner/attempt ledger and
remaining budgets before launching a new version.

The parent consumed 16 signal and 8 compile-stage queue entries.
Remaining budgets are signal 0, compile-stage 0, audit 8 and Q3 32.
There were no real compile or Q3 launches in that parent. The checked
recovery material contains ready rows for L_D=5 at absolute kappa 58 and
L_D=4 at absolute kappa 64, both at h. The normal width entry rechecks
the saved packet, physical schedule and fixed price rows before
installing these routes. The approved requested widths and scientific
SOURCE remain fixed.

The preparation transport validates the legacy calibration view at
13ab1acfaebe5f33958e1db67862f5c45725ab91 and links it beside the
task-owned provider. The published calibration binding check remains in
the existing driver. The eight accepted attempts are retained as
historical records. Their deployment dry-check does not produce a price.

Acceptance covered 36 targeted tests, including idle shutdown,
unresolved attempts, budget resets, scientific STOP preservation and
SOURCE mismatch.
