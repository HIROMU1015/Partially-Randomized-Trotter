# C2-B strong-baseline saved-data validation

Base: 78004943ed5b718c12841e80f55fddf1d1644364.
User explicitly approved finite numerical optimization and re-accounting in chat.
No approval certificate/grant is introduced. Old SOURCE, raw, results and STOPs remain unchanged.

Linear H4 / 1.00 Angstrom / STO-3G / DF12; supplied exact original-H ground;
no preparation cost; Full Y4 / LD12 / h=.02; 18 RPE rounds;
energy 1.5936e-4 Ha; alpha total .05; phase budget .4 rad; compiler A.
Fixed-matrix representation qualification limits remain inherited.

Only raw_synthesis.tar.xz (7603 strict records), PRECISION_REQUESTS.json.xz,
and verified_final/RESULTS.json.xz and associated checks are used.
All 72 registered cases (18 rounds x 2 caps x 2 error rules) retain their
original two-precision definition e=floor(log10(E/sum(weights))) and e+1.
No cross-round/key union or extra exponent is added. Actual exponents can differ
between error rules because their registered sensitivity weights differ.
The same two-choice rule, cap choices, data, and alpha optimization opportunities
are used for both arms. Costs are explicit integer T counts including actual adjoints.
Nominal requested-epsilon allocation constraints are retained; stored strict
errors are checked, not substituted as a new error-allocation convention.

Reuse existing exponent/greedy/LP/accounting functions and scipy.optimize.milp.
First compare under each original cost-sensitive alpha/cap schedule and under
common stored cost-sensitive alpha vectors. Then allow both arms the same
analytic cost-sensitive alpha update and registered-cap coordinate selection.
Keep integer shot ceilings. Compute fresh cost-sensitive/global-alpha bounds;
the old uniform-alpha LP total is never transferred to these conditions.
Use exact-rational feasibility and fractional-knapsack bounds per case;
distinguish these bounds from solver-reported MIP optimality/gaps.
A continuous-shot Lagrange relaxation may provide a global lower bound over
registered caps, finite precision candidates and alpha allocations. Its lower
bound is not an executable baseline price.

No new synthesis, word acquisition, PF/Hamiltonian, circuit build/compile,
sampling, fusion exploration, C2-A/C2-C/old C-S restart, or long-q prices.
Independent registered MILPs run in computational worker processes with OS
CPU affinity and actual available resources; no AI subagents or shared changes.
Stop after 72 cases and their specified comparisons/bounds are complete, or
on missing input/invalid guarantees. No automatic candidate expansion.
Ordinary technical debugging at identical conditions is allowed with prior
attempt evidence preserved. Publish source/results/report and mandatory STOP.
