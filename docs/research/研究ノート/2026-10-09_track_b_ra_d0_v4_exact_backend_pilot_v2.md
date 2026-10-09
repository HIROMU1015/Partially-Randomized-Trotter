# 2026-10-09 Track B exact backend pilot v2

利用者が別pilotとしてresource guard修正とsynthetic-only backend検証を認可。
基点`9e6d37fe6e345b402ba507db75f77dbec0855199`、独立branch
`track-b-ra-d0-v4-exact-backend-pilot-v2-20261009`で新contract／ledgerを固定した。
旧135＋数学監査18＋前pilot28、計181 protected filesは不変。

Phase Aは`GUARD_V2_PASS`、211 checks PASS。
unique PID RSS sum、subreaper/wait4 CPU、output全体scope、wall、kill/reap、retry拒否を確認した。
Phase BでSoPlex 7.0.0 harnessを一回buildし、rational I/OはPASS。
16求解中15件の証明を認証したが、`100_digit_infeasible`でERROR (-15)、証明未取得となった。
残り7 fixtureはNOT_RUN。compile/solver retry0、科学・登録LP・production変更0。

raw controllerはこのERRORにもCERTIFICATE_FAIL labelを付けた。source/outputは変更せず保持し、
指示§14 D／§15 PARTIALに従った監査判定を`V4_EXACT_BACKEND_PARTIAL`として別記する。
不正な証明の観測、元classのinfeasibility、RA-RTEの優位・非優位とは解釈しない。

[技術report](../../tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_v2_20261009.md)と
[GPT handoff](../../tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_v2_gpt_handoff_20261009.md)を公開する。
既存protected正本文書を上書きせず、新証拠は本noteとhandoffから案内する。
backend採用は保留。資料commit・push後mandatory STOPし、判断をGPT側へ戻す。
