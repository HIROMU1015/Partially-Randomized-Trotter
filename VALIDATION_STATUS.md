# Validation status

## 2026-10-25 H6 sector DF/PF run: registered CDF budget not met

[Report](docs/research/research_c_c1_h6_execution_20261025.md), [summary](artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_execution_summary.json), [original inputs/21 fits/STOP](artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/). Frozen science SOURCE33c380d8… unchanged, checkout cc0cdb525…; linear H6 1Angstrom/STO-3G, Nalpha=Nbeta3/sector400. Ordinary-number CDF requested1..21 vs explicit modified-Cholesky XDF prefix; original-g kappa/scalar/frame order, CA0.00159/DF reserve0.00019875 Ha. XDF first-passing R14 norm0.00010247385485275451 + guard1.1384527781887395e-6 Ha; CDF all21 optimizers success but minimum observed norm0.0004679546622353029 + guard1.4487327229419941e-6 Ha remains over budget, requested18/actual18 not adopted. DF_BUDGET_NOT_MET,61/129 calls,rank assessments35,retry/resume0,technical/correctness FAIL0. STOP.failed=false is only an exception flag, not eligibility PASS. Adapter/ground/alias/PF NOT_RUN; planned all-frame deterministic S2 s1/s4 delta0.1 m0..12/no PR split/own-DF oracle/preparation excluded was not reached. Native/shots/cost out of scope. No dense4096/controlled8192, artificial14/SOURCE33/input/export byte checks PASS. Not a universal CDF NO-GO or CA impossibility; strict reference enclosures unresolved. Old H2/H4 evidence preserved; local evidence, not external reproduction; mandatory STOP.

## 2026-10-25 H2 deterministic S2: numerical CA pilot complete

[Report](docs/research/research_c_c1_h2_execution_20261025.md), [raw evidence/52 rounds/calls/STOP](artifacts/research_c_c1_h2_h6_science/2026-10-25/). Frozen SOURCE33c380d8… unchanged; linear H2 1Angstrom/STO-3G, Nalpha=Nbeta1/sector4/full16/controlled32, ordinary-number CDF first-passing R2 vs explicit modified-Cholesky XDF first-passing R3; unlike H4 saved-eigh baseline. DF sector norms CDF1.3176231587153952e-6/XDF4.8652059505012625e-16 Ha, guard separate. Ordered deterministic S2 s1(primary)/s4(control), all frames/no PR split, delta0.1 m0..12, CA0.00159/DF reserve CA/8, own-DF oracle/preparation excluded/ideal IID Born/P0. Four numerical CA PASS/52 round PASS, representation2/full-controlled segment7/unitarity14 PASS, correctness FAIL0. Conditional total RZ savings36.4122%/36.1527%; rigorous reference enclosures/CA remain unresolved, no unconditional resource claim.193/196 units, fits/optimizers2, explicit XDF1, build/compile7 each,QPY14,retry/resume0,quantum shots0. Artificial14 PASS, SOURCE33/input/export postflight PASS. H6 not executed; old evidence preserved; local evidence, not external reproduction. Mandatory STOP.

## 2026-10-25 deterministic fixed-shot CA numerical pilot: all four PASS

[Report](docs/research/research_c_c1_stage3_deterministic_accuracy_execution_20261025.md), [evidence](artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/). Frozen SOURCE `002d0dc2e1dafd8cb71735c43253fa054e675f6d`; saved linear H4 1Angstrom/STO-3G Nalpha=Nbeta2, CDF R8/XDF R10, full-frame deterministic ordered S2 s1(primary)/s4(control), delta0.1 m0..12, own-DF oracle/preparation excluded/ideal IID Born/P0. 110/110 calls, four NUMERICAL_PILOT_CA_PASS,52 round PASS, correctness FAIL0. Energy assessment CDF0.0013549794824010532/XDF0.0012783188045012185 Ha is below CA0.00159 under numerical evidence; fixed saved shots unchanged, both same-s conditional cost comparisons favor CDF. Rigorous CA/reference rounding enclosures remain unresolved; no guaranteed resource/hardware claim, no PR qualification. Preflight12 artificial PASS; SOURCE27/saved49 post-byte identity PASS. New fit/optimizer/retry/resume/build/compile/PF-round or DF-norm recalculation/shot solving0. Old evidence/STOP preserved; local execution, not independent external reproduction. Mandatory STOP.

## 2026-10-25 C1 Stage3 repaired cost-only: completed, conditional estimates

[Report](docs/research/research_c_c1_stage3_cost_compat_execution_20261025.md), [evidence](artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/). Fixed SOURCE `9ce6703c89874c142c64ad934354bdff0db703d1`; saved linear H4 1Angstrom/STO-3G Nalpha=Nbeta2, CDF R8/XDF R10, delta0.1 m0..12, D S2 s1/4 and PR LD2 s4 r16 K0/2. All133 P0 phase/uni checks PASS; 936/936 calls, eight totals/208 axis rows/12 registered comparisons. Deterministic cost support retained; four PR costs remain point estimates with no whole-law interval, synthesis/law/CA or guaranteed resource claim. Old compiler STOP preserved; authorized fresh rerun1, cumulative1042 calls across2 attempts, in-place retry/resume0. Source23/saved20 identities PASS; artificial100 PASS. Local evidence, not external reproduction. Mandatory STOP; earlier entries are history.

## 2026-10-25 C1 Stage3 cost-only execution: technical compiler STOP

[Report](docs/research/research_c_c1_stage3_cost_execution_20261025.md), [failure/partial evidence/QPY/STOP](artifacts/research_c_c1_stage3_cost_science/2026-10-25/). Frozen SOURCE `edb9f6ef99d00e656265900af8447223029fae41`, P-H4 linear1 Angstrom/STO-3G/Nalpha=Nbeta2, CDF R8/XDF R10, delta0.1/m0..12, D symmetric S2 s1/s4, PR LD2/s4/r16/K0/K2, own-DF oracle/preparation excluded, saved optimistic shots/P0. Explicit user permission, fresh output, source unchanged. Qiskit1.3 HighLevelSynthesis equivalence-library access raised `RuntimeError: Already mutably borrowed` during parallel compile; concurrency cause not independently reproduced.106/936 logical units, compile9 attempts/7 COMPLETE/2 FAIL,3 readout-only equivalence and metric checks,34 partial cost rows/16 QPY preserved. No new kernel equivalence checks or candidate totals/comparisons completed. Candidate2 technical FAIL/6 dependency STOP; not a numerical-equivalence failure or scientific NO-GO. Artificial13/preflight PASS, new fit/optimizer/retry/resume0, no further calls after first failure, old evidence preserved. Total costs/CA/superiority remain unavailable. Local execution evidence, not external reproduction; mandatory STOP.

## 2026-10-25 C1 Stage3 optimistic shots screening: saved-data reanalysis complete

[Report](docs/research/research_c_c1_stage3_shots_execution_20261025.md), [20 candidates/260 rounds/520 axes/evidence](artifacts/research_c_c1_stage3_shots_science/2026-10-25/). Frozen SOURCE `f8fd2d373fc81353330e732c46b3b5d471e7c683`, saved feasibility commit `2eef01a4256df4c146324ef571c0074cd0ea5b08`. P-H4 linear1 Angstrom/STO-3G, Nalpha=Nbeta2, CDF R8 number/original-g kappa, XDF R10, CA0.00159/DF reserve CA/8, delta0.1, m0..12; deterministic S2 s1/4, PR LD2/s1/4/r4/16/K0/2; own-DF ground oracle, preparation excluded, ideal finite-RTE IID. 282/282 units COMPLETE, FAIL0, fit/optimizer/retry/resume0, artificial21 PASS.14 optimistic candidate totals;6 prior-infeasible totals null;494 finite axis values/26 ineligible. PR s4/r16/K0 CDF58356/XDF21408; s1/r4/K2 about2.58e46/8.08e25 shots. Counts are counterfactual estimates within the adopted sufficient Hoeffding prescription, not universal lower bounds, actual guaranteed shots or CA achievement. Unknown implementation errors remain null. No PF/RTE/operator recomputation, circuits/compile, actual shots, total costs or resource ranking. Committed local evidence, not external reproduction; mandatory STOP.

## 2026-10-25 C1 Stage3 feasibility-only execution: complete, no CA claim

[Report](docs/research/research_c_c1_stage3_feasibility_execution_20261025.md), [260 cells/20 candidates/calls/STOP](artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/). Frozen SOURCE `6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f`, saved P-H4 linear1 Angstrom/STO-3G, Nalpha=Nbeta2, sector36/full256, CDF R8 number/original-g kappa and saved XDF R10, CA0.00159, DF reserve CA/8, delta0.1, m0..12, D symmetric S2 s1/4; PR LD2/s1/4/r4/16/K0/2; own-DF ground oracle, preparation excluded, ideal finite-RTE IID law. Explicit chat permission, independent worktree and fresh output, unchanged SOURCE. All639 calls COMPLETE; candidate POTENTIALLY_FEASIBLE14/INFEASIBLE6/UNRESOLVED0, round247/13/0. Correctness FAIL0, ground/alias PASS, new fit/optimizer/retry/resume0. Source50/saved inputs17/lock52 versions verified, artificial regression16 PASS. Potential feasibility does not imply CA achievement or practical viability: CDF s1/r4/K2 min attenuation about2.14e-22. INFEASIBLE refers only to the registered sufficient condition. No circuit/compile/shots/cost/ranking, unknown synthesis remains unknown, old evidence preserved. Committed local evidence, not external reproduction; mandatory STOP.

## 2026-10-25 C1 Stage2 limited native execution: all checks PASS

[Report](docs/research/research_c_c1_stage2_native_execution_20261025.md), [summary/evidence](artifacts/research_c_c1_stage2_science/2026-10-25/execution_summary.json). User adopted the recommended conditions and permitted the limited run in chat; no approval-proof mechanism was used. Frozen public SOURCE `a977a4390668bc107360c8ff05f43d851dc81dc9`, P-H4 linear1 Angstrom/STO-3G, Nalpha=Nbeta=2; CDF R8 number/original-g kappa and saved XDF rank10; full256/controlled512, T1/10 one ordered symmetric S2 step, local1/20 and1/10, occupied(0,1,2,3), P0 Qiskit1.3/opt1/seed17/rz,sx,x,cx; L_D/delta window not applicable. All60 primary checks (40 local,4 raw full,4 actual adjoint,4 compiled full,8 XY pairs) and52 nested unitarity checks PASS; FAIL/NOT_RUN0. Compiled max-entry5.917215348217358e-11 <= frozen1e-10; XY max1.3012864077677942e-12 <=2e-10. Calls243/243, new fit/optimizer/retry/resume0, transpile12,68 QPY. Primary full metrics CDF RZ12118/CX7352/depth12724 vs XDF16659/9520/16215. Literal/shared costs match; unequal physical DF accuracy prevents equal-accuracy total-resource claims. No PF error/shot/Stage3; old source/evidence/STOP preserved. Committed local evidence, not external reproduction; mandatory STOP. Earlier unapproved preparation entries are historical.

## 2026-10-25 C1 Stage2 native connection: artificial qualification, science unapproved

[Report](docs/research/research_c_c1_stage2_native_preparation_20261025.md), [unapproved contract](docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json), [SOURCE/environment/evidence](artifacts/research_c_c1_stage2_technical/2026-10-25/). User adopted saved CDF R8 as a limited circuit-validation target; unequal physical DF accuracy precludes equal-accuracy total-resource claims. Implementation `be4f299470fd896c4bfd50b11bb87bf17cd760b2` connects existing general-K/Givens/controlled diagonal phase, actual inverse, measured XY, phase-preserving compiler and metrics without changing old source. New27 tests plus existing regression total80 tests/29 subtests PASS; no real circuit/operator/science execution. Byte preflight34 SOURCE files/25 provenance files and5 core library versions PASS with numerical imports forbidden. Recommended T1/10, local1/20 and1/10, ordered symmetric S2, P0 compiler, operator1e-10/signal2e-10, maximum243 logical units remain UNADOPTED. User must adopt the proposed conditions and explicitly permit science in chat. No approval-proof records or authentication are required or implemented. No new fit/H4 decode/real circuit build/compile/reanalysis, old grants/results/STOP preserved, mandatory STOP, Stage3 unapproved. Local artificial evidence is not molecular qualification or external reproduction.

## 2026-10-25 C1 saved-fit evaluation-only science: all 37 gates PASS

[Report](docs/research/research_c_c1_stage1_eval_only_execution_20261025.md), [summary/evidence](artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/execution_summary.json). Explicit user authorization bound frozen published SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0`, contract, original input and saved export to a fresh approval/output. P-H4 linear1 Angstrom/STO-3G, Nalpha=Nbeta=2 sector36; full256 base/frames; saved number CDF R8/original-g kappa and saved XDF rank10; L_D/delta not applicable. All37 gates PASS, FAIL/NOT_RUN0, including formerly mismatched frame0 (new error7.11493827835666e-15 vs fixed-form allowance4.428842301497364e-10). Sector adapter error4.011807592116179e-15 PASS. Separate report-only physical sector norms: CDF7.666027761392501e-5, XDF3.3751558354542566e-15 Hartree. These two physical errors are not gates; no absolute DF threshold was adopted. Calls141/141, new fit/optimizer/retry/resume0, 24 operators saved; artificial19 tests and preflight PASS. Old failure/STOP preserved; no source/allowance change. Committed local evidence, not immutable CI or external reproduction. Stage2 eligibility requires a separate physical error-budget decision; Stage2/3 unapproved, mandatory STOP. Earlier pending entries remain historical.

## 2026-10-25 C1 saved-fit evaluation-only preparation: science unapproved

[Report](docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md), [contract](docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json), [SOURCE/evidence](artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/). Implementation `88169af81cbc1140c4e51cfc1acf42a66db340b0` adds an evaluation-only dispatcher without fit/optimizer execution, retaining original-fit provenance separately from new evaluation SOURCE. Repaired numerical library is unchanged from `2ab9977994afa31cdfffb8629ce57c6efa83a842`. Artificial82 tests plus29 subtests PASS; byte/metadata/history preflight PASS with numeric imports forbidden. Successful future evaluation budget77+8R=141 at saved R8, new fit0, retry/resume0; base/all full-frame gates precede sector equality/physical DF error norms. Real H4 J/U/input numeric decode, reanalysis and evaluation0. Prior molecular FAIL remains unvalidated, execution requires separate new explicit approval, mandatory STOP. No immutable CI or scientific qualification claimed.

## 2026-10-25 C1 raw frame repair: artificial-only qualification

[Report](docs/research/research_c_c1_frame_repair_20261025.md), [SOURCE/evidence](artifacts/research_c_c1_frame_repair/2026-10-25/), [unapproved handoff](docs/research/research_c_c1_frame_repair_handoff_unapproved_20261025.json). Implementation `2ab9977994afa31cdfffb8629ce57c6efa83a842` replaces optimized einsum in the raw CAR contraction with explicit batched matmul. Artificial63 tests plus29 subtests PASS; 8 fresh processes/64 artificial frames agree with direct CAR loops and the unchanged K route. Legacy cold parallel failures reproduced; preloading the candidate backend also yielded old-path passes, so internal cause and causal attribution to prior H4 FAIL remain unconfirmed. No molecular input/export numeric decode, reanalysis, fit, norm, circuit/compile. Old scientific STOP and unmeasured CDF/XDF errors remain. Separate saved-export evaluation preparation/approval required; current one-fit runner must not be used as an evaluation-only entry. Mandatory STOP, no immutable CI or molecular qualification claimed.

## 2026-10-25 C1 Stage 1 compatibility-source execution: scientific equality STOP

[Report](docs/research/research_c_c1_stage1_compat_execution_20261025.md), [summary/evidence](artifacts/research_c_c1_stage1_compat_science/2026-10-25/execution_summary.json). New user approval bound source `da2edb465d5ae6432439fbd1df8e57512ee59c11`, contract/input and fresh output. One CDF fit converged at R8; kappa/J/U and full-base gates PASS. First full256 frame equality FAIL: error2.8302153250645086 vs frozen allowance4.831545715791788e-10. Registered diagnostics20 PASS/1 FAIL/18 NOT_RUN, physical CDF/XDF sector errors and sector adapter equality not measured. Call107/147, fit attempted1/returned1, science retry/resume0. No route is identified as correct by this comparison alone. Stage1 incomplete; raw evidence/old STOP preserved; no repairs, reanalysis or further science after STOP. Stage2/3 remain unapproved.

## 2026-10-25 C1 Stage 1 compatibility repair: artificial qualification only

[Report](docs/research/research_c_c1_stage1_compatibility_20261025.md), [unapproved contract](docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json), [SOURCE/locks/evidence](artifacts/research_c_c1_stage1_compatibility/2026-10-25/). Implementation `303f73275f133bfed0a0cd93a06f4a46f8ffd7be` retains SciPy1.14.1 and ffsim0.0.83; task-only per-matrix logm plus venv symlink launch fix. Artificial59 tests and29 subtests PASS, including real ffsim initialization on artificial tensors with optimizer blocked. No molecular fit/numeric decode/operator/norm/residual/reanalysis/circuit call. Scientific Stage1 remains incomplete; former spent approval/STOP preserved. New one-fit authorization required, no automatic Stage2/3, mandatory STOP. Local tests archived in Git are not immutable CI or molecular qualification.

## 2026-10-25 C1 Stage 1 authorized attempt: technical STOP before optimizer

[Execution report](docs/research/research_c_c1_stage1_execution_20261025.md), [evidence/summary](artifacts/research_c_c1_stage1_science/2026-10-25/execution_summary.json). Fixed source `de76aee72f35321c00155b0b17d22ef6eb2bcb9a`; preflight PASS, one ffsim call failed during x0 parameterization because SciPy1.14.1 logm rejects batched matrices. Optimizer and all Hamiltonian/norm comparisons were not reached. Physical CDF/XDF errors and adapter equality remain NOT_MEASURED/NOT_RUN. Scientific counter2/maximum147, fit attempted1/returned0, science retry/resume0. Technical launches before science are separately preserved. Prior artificial tests remain valid within their mock/artificial scope; molecular Stage1 qualification is incomplete. No further fit or Stage2/3 authorization is inferred; mandatory STOP.

## 2026-10-25 C1 Stage 1 technical preparation only

[Technical report](docs/research/research_c_c1_stage1_technical_preparation_20261025.md), [unapproved contract](docs/research/research_c_c1_stage1_science_contract_unapproved_20261025.json), [SOURCE/locks/artificial evidence](artifacts/research_c_c1_stage1_technical/2026-10-24/). Implementation commit `394fcf3a445caf0a5f206b7bbcf4a8c88f971504`; 45 artificial top-level tests plus 29 subtests PASS. Adopted P-H4 pilot conditions have not been executed. No real fit/operator/norm/residual/energy/circuit calls. Engineering allowance adoption and new explicit science authorization remain required; no immutable CI or scientific qualification claimed. Earlier evidence/status entries remain historical and unchanged.

## 2026-10-24 disposition update: C4 temporarily closed; C1 static survey complete

The user adopted temporary closure of research-C C4 Rustiq/simple-angle fusion as an independent direction. [Closeout, preserved evidence and reopening conditions](docs/research/research_c_c4_closeout_handoff_20261024.md). All prior scientific statuses, raw packets and STOPs remain unchanged. Earlier statements that closure was pending are historical. This decision is not a general Rustiq or PR NO-GO.

[C1 compressed-DF interface survey](docs/research/research_c_compressed_df_interface_survey_20261024.md) recommends ordinary ffsim C-DF for an initial interface check and RC-DF conditionally on a faithful available fitter. General diagonal blocks require an adapter; SCDF optimizer/input and original-integral provenance gaps remain. No method adoption, new numerical evidence, implementation, science grant or execution. Old invalidated DF screening and prose-only grouped-UWC status are unchanged. User/research C/coordination selects the next scope with research A/B; mandatory STOP after publication verification.

## 2026-10-24 research C4 B2: all correctness gates PASS; simple fusion matches saved R1

One separately approved run at frozen SOURCE `0c9d0a480db95e4bbd57a1cda02f7fb24c2b2162` completed with no retry/resume. Saved-F1 four-mode/two-particle occupied(0,1) density-square synthetic H, L_D1/two residual fragments, T1/5 q2 delta1/10 one point r2 h1/20 K2; geometry/AO basis/molecular DF rank not applicable. Qiskit1.3.0 opt1 seed17 native rz/sx/x/cx, unchanged full measured X/Y scope.
Local10 (against both independent saved targets and pre-fusion literal), controlled4, actual-adjoint4, raw/compiled XY and compiled-operator8 PASS; compiled max-entry1.08720e-14 at absolute tol1e-10/rtol0, no phase fit. Saved references restored14, new references0, Operator26, transpile8/cold8/warm8; old routes and Rustiq/R2 rerun0. All eight compiled metrics match saved R1 on all eight wrappers. Simple adjacent controlled-Pauli fusion is sufficient to reproduce the observed SS gains; no additional R1 compiled metric gain is observed here. This does not identify Rustiq internals or prove general superiority/newness.
[Result, immutable source and new evidence packet](docs/research/research_c_angle_fusion_b2_execution_20261024.md). Committed local evidence, not CI or external independent reproduction. No law-weighted cost/shot/all-round evaluation. Earlier pending-angle-control statements below are historical. User/research C/coordination decides whether to retain the known implementation improvement and close independent C4; no closure is automatically adopted. Mandatory STOP and no additional science.

## 2026-10-24 research C4: limited Q13 integration correctness PASS, local resource differences recorded

Source `b546c4e4f5d6645fd198b9ec3f85e3cd644ac4e9` completed one explicitly approved execution of saved-F1 SS motifs2/3/4 and PP motif0, with B0/B1/R1. Four-mode/two-particle density-square synthetic fixture, L_D1, two residual fragments, T1/5 q2 delta1/10 r2 h1/20 K2; molecular geometry/AO basis/DF rank not applicable. Fixed Qiskit1.3.0 opt1 seed17 native rz/sx/x/cx.
All local20/full-controlled12/actual-adjoint12/raw-and-compiled XY/compiled-operator24 gates PASS; max compiled error1.12226e-14 at tol1e-10, rtol0 and no phase fit. Reference28, transpile24, cold/warm24 each, retry0. B0/B1 match all eight metrics. R1 reduces costs in three selected SS motifs; PP M0 is unchanged. Across eight designed case/axis wrappers the unweighted RZ/CX/depth sums fall2.5%/4.9751%/4.0486%, not a law expectation or total-round advantage. Identical-Pauli angle-merge baseline and general method delta remain unvalidated.
[Results, source, immutable raw packet and mandatory STOP](docs/research/research_c_rustiq_execution_20261024.md). Local committed evidence, not CI or independent reproduction. Old research states/grants/STOP remain preserved; no additional experiment is authorized by completion.


## 2026-10-24 research disposition: Gaussian singleton direction temporarily closed by user adoption

The user adopted temporary closure of Gaussian singleton synthesis/selection optimization as an independent research direction. F1/F2 fixed-fixture correctness and PS48 equivalence PASS remain unchanged; blanket candidate-A use does not support RZ/CX reduction and a general compile-free selector remains unestablished. This is a research-priority decision, not a scientific FAIL or a PR-wide NO-GO.
[Adopted decision, evidence boundaries and next-research handoff](docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md) reuses existing Track B/prior-art records. Gamma/statistical/interval qualification, certified N and N-weighted total cost remain unresolved. No new science, regrouping, implementation or validation path was executed. Earlier preparation and pending-decision entries below are historical. Mandatory STOP; User/GPT chooses any next mechanism and authorizes new work separately.

## 2026-10-24 rectangular Givens PS48: all scientific gates PASS; resource changes mixed

Unchanged source `770e53a` completed one separately approved PS48 execution: six candidate bases and raw/compiled 48 wrappers (24 each support1/support2), 36 transpiles and 12 aliases, zero new high-precision references or retries. Compiled operator max-entry 1.99349e-14, within fixed tolerances with rtol=0 and no phase fit.
Indices21/25 improve RZ/CX/depth in all eight cases each, but the fixed48 equal-case mean increases RZ 7.8571% and CX 1.9802%; total depth decreases 0.7124%. This is not a full-F2 weighted expectation or certified-N comparison. [Results, residuals, costs, provenance and mandatory STOP](docs/research/pr_rectangular_givens_ps48_execution_20261024.md). Committed local science, not CI or independent external reproduction. No blanket rollout or additional experiment.

## 2026-10-24 rectangular Givens candidate A: implementation only, PS48 not executed

Singleton-only Nature0.7.2 Gaussian suffix and canonical central support are connected to the existing builder; shared-frame runs retain the existing synthesis. Eight candidate mocks plus six existing artificial preparation tests PASS, and saved input/source/case hashes pass metadata preflight. No real quantum circuits, numerical matrices, QPY decoding, reference evaluations or new research grants were executed.
PS48 correctness, controlled relative phase and resource reduction remain unvalidated. Saved targets and fixed F2 compiler/tolerances are bound for a separately approved future one-shot. [Implementation, evidence limits and mandatory STOP](docs/research/pr_rectangular_givens_preparation_20261024.md). Local technical evidence only; no CI or scientific PASS claim.

## 2026-10-24 saved F2 attribution: exact grouped reconciliation PASS

Saved-only regrouping of 13,674 physical X/Y records plus two excluded common diagnostics; no new quantum circuit, compile, law, sampling or simulation.
Eight independent partitions produce 490 group rows; all 64 view/arm/axis mass and cost checks and all published expected metrics agree as exact rational numbers. Grouped X/Y costs also agree exactly.
PS has lower frame-containing probability than SP (46.0786% vs 51.6195%) but higher conditional RZ (77.690309 vs 55.713230). Descriptive conditional-cost differences outweigh the probability-mix difference; no causal Gaussian/compiler effect or certified N is inferred.
Five artificial saved-record tests PASS. [Results, definitions, exact artifacts and mandatory STOP](docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md). Committed local analysis, not CI or independent reproduction; prior measurement evidence and statistical qualification unchanged.

## 2026-10-24 corrected limited F2: all registered numerical checks PASS

SOURCE `9468c70` and a fresh grant completed raw/compiled 13,676 cases, 27,356 reference evaluations, all four finite corrected means and exact weighted per-shot resource aggregation.
Unique transpiles: 13,006, aliases: 670. Compiled operator max-entry 2.99943e-14; mean max-entry 5.10708e-15, within unchanged P0 gates, no phase fit.
One corrected execution completed without additional interruption. Old failed run and STOP retained. Actual per-shot metric order: PP < SP < PS < SS for this fixed synthetic context/compiler.
No certified N, N-weighted comparison, physical same-accuracy or general algorithm advantage claim.
[Completed local results and mandatory STOP](docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md). This is committed local evidence, not CI or independent reproduction.

## 2026-10-24 limited F2: compiled operator gate FAIL, partial evidence preserved

SOURCE `9b1e103` and a new one-shot grant completed 13,676 raw wrapper checks and 27,356 reference precision evaluations.
All raw checks passed. Actual transpile calls: 57 unique groups; 54 completed and 3 failed the registered compiled full-operator gate (max-entry up to 1.9999996023, tolerance 5e-10).
Compiled logical cases: 60 PASS, 3 FAIL, 13,613 not evaluated. Finite corrected means and expected arm resource aggregation were not reached.
No complete SS/SP/PS/PP ranking, certified N or N-weighted cost is available. No retry/resume or extra numerical probe.
[Execution report and mandatory STOP](docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md). Old statistical qualification gates and prior evidence remain preserved.

## 2026-10-24 F2 qualification: STOP before scientific execution

F2 scientific execution is explicitly authorized subject to adopted conditions and established qualifications.
The standard conditional Hoeffding argument and five scalar interval/log/ceil technical tests passed.
Candidate log(80)/log(320) enclosures were checked against independent exact rational series bounds.
No applicable combined finite-bit/circuit error guarantee establishes gamma relative to the common finite target.
Statistical parameter/failure-scope adoption remains pending. No F2 grant, build, compile, reference or shot qualification was executed.
All four arms' resource results and N remain null; this is neither a scientific FAIL nor an execution-time technical failure.
[Qualification findings and mandatory STOP](docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md).
The restricted scalar checks do not certify matrix/reference/circuit error or qualify F2 as a whole.

## 2026-10-24 F2 preparation: implementation present, no scientific execution

F2 finite-target comparison SOURCE/input/case/preflight preparation and eight mock/management tests are complete.
Statistical numerical policy, IID/range/finite-bit allowance evidence and interval/ceil qualification remain pending.
No F2 circuit build, compile, law/target evaluation, numerical reanalysis or new grant has occurred.
[Preparation and mandatory STOP](docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md).
This is implementation-only evidence; the F1 registered local PASS below remains unchanged.

## 2026-10-24 mixed Pauli/frame dictionary F1: completed registered local validation

`F1_REGISTERED_CORRECTNESS_PASS_UNQUALIFIED_PRICE_FALLBACK_MANDATORY_STOP`。
修正SOURCE `8192530b17d6bc4af195c734ff43a6e78c506942`、保存B入力、新grantで正常終了。
4 mode・2粒子・L_D=1・残差2fragmentの合成fixture、T=1/5、q2、delta=1/10の一点、r2、h1/20、K2、full16／controlled32。
geometry／AO basis／分子DF rankは非該当。全4armの登録済みH／LAW／mean／operator／signal／compile／referenceとcatalog直接価格-DP照合は固定閾値内。
16familyの未実行caseは0。未資格intervalのfallback処理はPASSだが、認証score・最適選択・shot数・資源優位性は未評価。
[結果とSTOP](docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md)、
[要約JSON](artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/F1_RESULT_SUMMARY.json)、
[raw hash台帳](artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/OUTPUT_HASH_MANIFEST.json)。
ローカル検証でありCI／外部再現ではない。旧停止記録は以下に保持し、F2へは別判断・承認を要する。

## 2026-10-24 mixed Pauli/frame dictionary F1: technical stop, partial local evidence

`F1_TECHNICAL_FAILURE_PARTIAL_RESULTS_MANDATORY_STOP`。固定SOURCE
`31e8391a56fbb1a0e64f477f78e242655329de99` と保存B入力を使ったone-shot。
4 mode・2粒子・L_D=1・残差2fragmentの合成fixture（geometry／AO basis／分子DF rank非該当）、
T=1/5、q2、delta=1/10の一点、r2、h1/20、K2、full16／controlled32。
H-ID4件は全armでPASS、SSのみfinite meanを完了。実行済みcontrolled／raw signal検査は固定閾値内。
最初のcompile requestが `gates.py:81` の `normalized.metadata=None` でTypeErrorとなり、
actual transpiler invocation0のまま停止した。compile request1／completed0を区別する。

LAW metadata47,889、component54、motif3、wrapper1、catalog controlled60、reference109precision evaluationsを
完了したが、全登録範囲の検証ではない。LAW最終mass gate・残るmean／motif／signal・catalog DP比較・
physical期待価格・selector／未資格fallback処理は未完了。価格・shot認証、資源優位性、新規性のGOはない。
rawと消費済みgrant／STOPを保全し、retry／resume・追加科学計算なし。公開は保存診断要約とhash台帳のみで、
ローカル部分検証をCI／独立外部再現と表示しない。
[実行結果](docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/EXECUTION_REPORT.md)、
[要約JSON](artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/F1_RESULT_SUMMARY.json)、
[raw hash台帳](artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/OUTPUT_HASH_MANIFEST.json)。
旧系列のstatusは以下の履歴を保持する。F2は別の判断と実行承認が必要。


## 2026-09-20 delta-schedule central-RTE compiled-cost note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしで、
前項の$\delta=0.01,0.0125,0.02$ scheduleへ境界補正型compiled-cost modelを接続した。

scheduleで現れる短時間幅0.02--0.000390625の12点について、同じ1--4イベント列を
角度だけ変えて個別にコンパイルした。1--3イベントは280 trajectory・3,080 metric比較、
4イベントは160 trajectory・1,760比較で、RZ/CX数・深さ、全体深さ、回路サイズの差は
すべて0だった。したがって、この固定compiler範囲では短時間幅0.02の係数を再利用した。

固定K1--K3を未使用イベント列へ適用すると、$L=8$は全6指標最大2.319%、$L=16$は
3.556%以下で点基準を通過したが、$L=32$は全次数0 RZ誤差7.966%、$z=3.151$で不通過だった。
各4イベントTaylorパターン100標本でK4を較正すると、$L=32$の最大点誤差は4.175%へ下がった。
ただしK4の点ごとの95%診断は5%を超え、厳密な5%保証ではない。次数2が2か所以上のK4窓は
今回のround分布では最大期待数$7.64\times10^{-13}$以下だった。

$r\leq16$にK1--K3、$r=32$にK1--K4を使い、各roundの解析Taylor確率、$q_m$、暫定shot数で
中央RTEブロックを集計した。RZ代理値は$\delta=0.02$で$7.9877\times10^{11}$、0.01で
$9.2021\times10^{11}$、0.0125で$1.2626\times10^{12}$となり、全6指標で0.02が最小だった。
0.01はRZで15.203%増、0.0125は58.073%増である。

これは中央$\widetilde U_{\rm RTE}$ブロックだけのlocal dirty-worktree proxyである。
決定論DF half sweep、決定論/RTE外側境界、制御化、Hadamard wrapper、状態準備、$q>8$一体回路、
最終総cost、immutable CIは未評価。従って$\delta=0.02$は次の優先候補であって最終採用ではない。
次は0.02と比較対照0.01の制御付きpartial-$S_2$反復proxyを検証する。詳細は
[delta schedule中央RTE cost検証](docs/rpe_delta_compiled_cost_validation.md)。
変更後のlocal全テストは`494 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 delta and round-specific finite-RTE schedule note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$で、
既存PF検証で実行済みの10個の$\delta$候補を、暫定`CA/10`、
$\beta_{\rm RPE}=0.4$、経験的$C=0.01342567$でscreeningした。PF位相予算0.02 radを
通過したのは$\delta=0.01,0.0125,0.02$の3候補だった。$\delta=0.0125$は較正と
独立なPF検証gridにおける唯一の通過点である。

各候補の18--19 roundで$r_m\in\{1,2,4,8,16,32,64,128\}$、
$K_m\in\{0,2,4,6,8\}$を走査し、shot数で重み付けしたランダム成分作用数を
暫定proxyとしてround別scheduleを構成した。選択した全56 round点のH4 sector行列検査で、
演算子・信号・適用可能な位相上界、PF/RTE予算、半径下界がすべて通過した。
最小観測半径は0.572701、最大実PF位相誤差は0.0139884 rad、最大有限RTE位相上界は
0.0176405 radだった。

暫定proxyでは$\delta=0.02$が最小だが、$\delta=0.01$との差は約0.87%である。
このproxyはcompiled costではないため、両者をshortlistとし、回路cost modelの
再較正またはholdout後に比較する。$q>8$一体compile、fresh-IID実験、最終総cost、
immutable CIは未評価。詳細は
[delta/round schedule検証](docs/rpe_delta_round_schedule_validation.md)。
変更後のlocal全テストは`490 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 target-precision round-horizon note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1$、$\beta_{\rm RPE}=0.4$について、
$\beta_{\rm RPE}/(2^M\delta)\leq\epsilon_E$を満たす最小round範囲を計算した。
正本文書では$\epsilon_E$は外部入力なので、既存設定`TARGET_ERROR=CA/10`を暫定主条件、
化学精度$CA$を感度比較とした。

補足資料の架空例$\epsilon_E=0.50$は$M=3,q_{\max}=8$、化学精度は
$M=12,q_{\max}=4096$、暫定`CA/10`は$M=15,q_{\max}=32768$となった。
従って、既存4段検証は実目標候補のround範囲を覆わない。

固定$r=4,K=2$をsector行列上で$q=8,4096,32768$へ延ばすと、$q=32768$の
PF位相誤差は0.437285 radで0.02 rad予算を超え、attenuationは
$7.90584\times10^{-13}$だった。finite-RTE演算子・信号上界は通過したが、この固定設定を
長roundへ単純外挿する候補は棄却する。次は$\delta$候補とround別$(r_m,K_m)$を再探索する。

これはlocal dirty-worktreeの小規模行列診断である。$q>8$回路コンパイル、cost proxy、
fresh-IID shot、最終総cost、immutable CIは未評価。詳細は
[round範囲診断](docs/rpe_target_round_horizon_validation.md)。
変更後のlocal全テストは`487 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 physical q=8 and four-round branch-reconstruction note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1,r=4,K=2$、$q=1,2,4,8$で、限定4段集計と同じ
$\beta=(0.02,0.02,0.36)$および重み付き$\alpha$を使い、$q=8$物理信号と4段の
逐次分枝復元を検証した。

$q=8$のfinite-RTE信号半径は0.993219824261、exact信号からの系統位相差は
$1.06819\times10^{-4}$ rad、厳密二項の統計位相失敗率は$2.07129\times10^{-6}$だった。
4段合成の座標失敗率は$1.33625\times10^{-3}$、統計位相失敗率は
$2.14873\times10^{-6}$で、割当予算0.05以内だった。

4段8軸の全1,572 shotへ異なるRTE trajectory seedを割り当てた明示的監査を行い、seed重複なし、
trajectory平均の解析信号からの差は最大1.925標準誤差だった。解析的周辺分布から生成した
10万回の4段測定では分枝失敗・最終位相失敗とも0件で、最終失敗率の片側95%上限は
$2.99569\times10^{-5}$だった。

固定H4の4段end-to-end接続はlocalに通過したが、目標エネルギー精度からのround数決定、$q>8$、
別条件、実backend、noise、状態準備、最終総cost、immutable CIは未評価である。PF係数が経験値の
ため保証statusは引き続き`empirical_screening`である。詳細は
[4段分枝復元検証](docs/rpe_four_round_phase_validation.md)。
変更後のlocal全テストは`484 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-18 limited four-round RPE accounting note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1,r=4,K=2$、$q=1,2,4,8$で、前回選んだ暫定配分
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$と重み付き
$\alpha$を既存の厳格な資源集計APIへ接続した。$q=1,2,4$の固定Hadamard直接costと、
未使用$q=8$ holdout通過済みproxyを出典付き複合providerとして使用した。

4段・8軸の合計は1,572 shot、RZ数32,673,960.607143で、shot×1 shot costの直接再計算と
前回診断値に一致した。保守的$\alpha$ union boundは0.05の予算内だった。
前回の$q=1,2,4$物理信号を固定し、新しいshot数と$\beta_{\rm stat}=0.36$で厳密二項失敗率を
再計算すると、合成座標失敗率$2.2246\times10^{-4}$、統計位相失敗率
$7.7444\times10^{-8}$で、各軸・各段の割当額を満たした。

これはlocal dirty-worktreeの限定4段診断で、PF入力が経験値のため`empirical_screening`である。
$q=8$物理信号・厳密失敗率、4段branch復元、最終全round総コスト、実backend、noise、
immutable CIは未評価。詳細は[限定4段検証](docs/rpe_four_round_accounting_validation.md)。
変更後のlocal全テストは`481 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-01 beta/alpha allocation-sensitivity note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4,8$で、RPE位相誤差・失敗確率配分の感度を評価した。
$q=1,2,4$の状態準備なしHadamard直接compiled costと、未使用holdoutを通過した$q=8$ proxyを
固定入力としたため、sweep中の回路再compileはない。

5種類の$\beta$配分と、一様／cost感度重み$\alpha$配分の10 scenarioは、$\beta$和、$\alpha$和、
PF・RTE実寄与、正半径、shot式、round cost恒等式を全て通過した。単一条件への過適合を避ける
暫定100倍headroom規則では$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$と
cost感度重み$\alpha$を選んだ。各軸shotは$q=1,2,4,8$で229、207、186、164、RZ comparison costは
$3.2674\times10^7$である。現行$(0.08,0.08,0.24)$・一様$\alpha$比では56.35%小さいが、同じ
$\beta$での$\alpha$変更単独は4.35%だった。

これは1 snapshot・1 compiler・固定$(L_D,\delta,r,K)$・RZ指標のlocal dirty-worktree sensitivity
diagnosticである。100倍guardは理論値でなく、比較costは最終総costでもその削減率でもない。
選択後の非一様$\alpha$に対する厳密二項失敗率・仮想測定も再実行していない。今回の失敗確率条件は
Hoeffding shot式とunion boundである。$q=8$物理信号・位相、branch reconstruction、$q>8$、noise、
実backend、immutable CIは未評価である。
変更後のlocal全test suiteは`479 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-01 q=8 Hadamard cost-proxy/resource connection note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしで、状態準備を除くHadamard interrogation全体のcompiled costを
評価した。各$q$ 8 trajectoryの$q=1,2,4$だけでaxis・metric別affine proxyを較正し、
係数固定後に未使用$q=8$の一体compileを予測した。

両軸・全6指標の最大相対点誤差はcircuit sizeの0.726%、RZ countは0.675%で、
事前の5%基準を通過した。較正点とholdoutを含むRZ平均の最大相対標準誤差は
0.732%で、事前2%条件を通過した。holdoutはfitに使用していない。

通過したvalidation fingerprint、Hamiltonian・DF split、$L_D,\delta,r,K$、compilerを要求し、
実際にholdoutした$q$だけを返すproviderをresource accountingへ接続した。
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.08,0.08,0.24)$ rad、
$\alpha_{m,b}=0.05/8$の$q=8$ candidateは各軸414 shot、1 shot RZ count 48135.0804となり、
round RZ cost $3.9855847\times10^7$の再計算が一致した。未検証$q=16$は拒否した。

これは1 snapshot・1 compiler・$q=8$のlocal dirty-worktree evidenceである。$q>8$、別分割、
proxy係数共分散、$q=8$物理信号・位相、複数round合計、状態準備、noise、実backend、
最終総costまたはimmutable CI evidenceではない。
変更後のlocal全test suiteは`478 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-01 virtual-Hadamard statistical failure note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4$で、finite-RTEの物理基底状態信号を用いた
仮想Hadamard測定を検証した。$(\beta_{\mathrm{PF}},\beta_{\mathrm{RTE}},
\beta_{\mathrm{stat}})=(0.08,0.08,0.24)$ rad、$\alpha_{m,b}=0.05/6$から得た各軸shot数は
389、390、391である。

厳密二項計算では、6軸のいずれかの座標誤差が許容量以上となる確率は0.0010363、
3 roundのいずれかの統計位相誤差が0.24 radを超える確率は$1.4775\times10^{-6}$で、
いずれも$\alpha_{\mathrm{tot}}=0.05$以内だった。10万回の周辺Bernoulli反復では座標失敗111回、
位相失敗0回で、片側95%上限はそれぞれ0.001299、$2.996\times10^{-5}$だった。

別に各shotへfresh IIDなRTE trajectoryを割り当て、計2340 trajectoryをsector内で直接作用した。
seed重複はなく、trajectory平均信号と解析信号の差は最大2.063標準誤差、全軸の条件付き測定数は
周辺二項分布の99.9%中央区間内だった。これはshort-roundの測定統計とfresh-IID実装のlocal検証であり、
全roundのRPE branch selection、最終位相復元、実backend、noise、状態準備、最終総costまたは
immutable CI evidenceではない。変更後のlocal全test suiteは`475 passed, 4 warnings`だった。
最初の全suite実行では既存の並列SQLite cache testが一時的な`database is locked`で1件失敗したが、
単独再実行と続く全suite再実行では通過した。4 warningは既存grouped-UWC test由来である。

## 2026-09-01 short-round signal・shot・compiled-cost connection note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4$で、finite-RTE信号検証、RPE shot式、
controlled time-evolution direct provider、状態準備なしHadamard interrogation providerを接続した。
物理full-$H$基底状態のPF信号半径は1から最大$1.11\times10^{-8}$のずれで、単位半径仮定と
実半径から得る各軸shot数は$q=1,2,4$で389、390、391と一致した。

同一compiler条件ではHadamard interrogationのRZ countはtime-evolution部分より各軸+2、
circuit sizeは+5で、CX count/depth、RZ depth、total depthは同じだった。
全roundで`round_cost=N_c g_c+N_s g_s`の再計算、scope識別、古典Monte Carlo標本数8を
量子shot数へ追加乗算していないことを確認し、専用payload validatorを通過した。

これは1 snapshot・1 compiler・$q\leq4$のlocal接続検証である。8 trajectoryのcompiled-cost
点推定を精密な候補順位または最終総costには使わない。仮想Hadamard測定は上記の別検証で
追加した。$q=8$ proxyの1 round接続は上記の別検証で追加したが、$q>8$、
全round集計、状態準備、noise、実backend、immutable CIは未評価である。
変更後のlocal全test suiteは`473 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
保存した接続結果JSONは専用validatorを再通過した。

## 2026-08-26 H4 follow-up・H5 system-size circuit-cost completion note

2026-08-25のH4 follow-upは全jobがreturn code 0で完走し、専用validatorを通過した。同一H4 chain、
距離1.0 Å、STO-3G、DF rank 12、固定snapshot、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしの範囲で、$L_D=0$の全Taylor patternを同一trajectoryで比較した
K1--K3 paired residualは全metric最大1.679%、主RZ 95%上側診断3.201%だった。$L_D=6$、
short-step 0.025、$K=2$の固定K1--K4を未使用$L=8$へ適用した点誤差は最大3.750%だが、
主RZ 95%上側診断は7.045%だった。同一trajectoryのpaired K1--K4 $L=8$構造検証は最大4.008%、
主RZ 95%上側診断4.902%だった。controlled $q=1,2$ affine modelの未使用$q=8$ holdoutは
全metric最大0.0529%だった。

H5 chain、距離1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、short-step 0.025、
$K=2$、同じcompiler条件では、$L=4,6,8$のpaired K1--K3構造残差が全metric最大1.665%、
K1--K4が0.551%だった。5%を満たす最小cluster長としてK1--K3を選んだ。独立calibrationは
最大RZ相対standard error 0.745%で停止し、all-order-0各長さ500、single-order-2各位置125の
独立full holdoutは全metric最大点誤差3.776%、主RZ最大z 2.009、予測側95%半幅1.459%だった。
事前の5%点誤差と2%予測精度を通過した。点wise正規近似95%上側診断7.461%は硬い受理条件ではなく、
rigorousな5%保証とは扱わない。

したがって、exact Hamiltonian snapshot・compiler条件ごとにK1--K3を較正して代表holdoutを行い、
不通過条件だけK4へ進む運用規則を採用する。回路costの広いpilot検証は一旦区切るが、$L>8$、
compiler/coupling/backend変更、新snapshot、controlled scope拡張または5%未満の候補差では再検証する。
これらはlocal dirty-worktree evidenceであり、full RPE、quantum shot、noise、実backend、最終総cost、
immutable CIまたは外部再現結果ではない。

整理後のlocal全test suiteは`472 passed, 4 warnings`だった。4 warningは既存grouped-UWC testの
complex-to-real castであり、今回の回路cost検証由来ではない。H4 paired K4、H5 paired、H5独立
calibration、H5独立holdoutの保存payloadは各専用validatorを再通過した。

## 2026-08-25 circuit-cost follow-up実装note

最初の追加batchはA--Cすべてreturn code 0で完走し、専用validatorを通過した。Aの独立K1--K3は
主誤差12.376%で不通過だが最大z 1.787で原因を分離できず、Bの独立paired K4は最大点誤差
1.281%だがRZ 95%上側診断5.714%、Cのcontrolled $q=8$は全metric最大0.0529%で通過した。

follow-up用に、複数order-2のpaired full/local-window残差、K4 500標本へのincremental resume、
固定K1--K3のL8 holdout、固定K1--K4のL8評価を実装した。変更後のlocal全suiteは
`470 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。follow-upのlive stateは
`artifacts/rte_cost_followup_batch/2026-08-25/status.json`へ保存し、成果物validator通過前は
追加科学結果として扱わない。

## 2026-08-25 circuit-cost追加batch実装note

同一H4 fixed snapshot上で、$K=2$の複数order-2 pattern、$L_D=6$の対応あり独立K4、
controlled $q=8$ holdoutを採取する実装とdetached batch runnerを追加した。
変更後のlocal全suiteは`469 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
数値jobのlive stateは`artifacts/rte_cost_data_batch/2026-08-25/status.json`に保存する。
完走済み成果物を専用validatorで検証するまでは、追加の科学結果や最終cost評価として数えない。

## 2026-08-24 connected-cluster short-step 0.030 transfer completion note

中断していたH4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、short-step 0.030、
finite Taylor cutoff `K=2`、同一Hamiltonian snapshot・compiler条件のtransfer holdoutを再開した。
固定calibrationに対し、未使用`L=4,6` full回路の全order-0を各1500標本、order-2が1回の条件を
各位置500標本とした。

全6 metricの最大絶対相対点誤差は3.541%、主RZ countは3.488%、RZ最大absolute z-scoreは
3.037、予測側95%相対半幅は1.784%だった。事前の点誤差5%と予測精度2%は通過した。
500/150標本の予備runでは全metric 7.138%、RZ 5.754%、予測半幅2.265%で不通過だったため、
当該超過は再開runで維持されなかった。再開時はcalibration相対標準誤差目標も1.0%から0.8%へ
締めており、改善をholdout標本数だけには帰属しない。一方、RZの点wise正規近似95%上側診断は5.738%であり、
残差ゼロまたはrigorousな5%保証とは扱わない。

再開には旧pattern checkpoint、v3固定sample chunk、既存SQLite metric cacheを併用した。8000回路中
1250件がpersistent cache hit、6750件がmissで、3 workerの実経過時間は393.2秒だった。生成artifactは
内部fingerprint・chunk意味検査を通過し、一時checkpointは残っていない。この結果はdirty local
worktree evidenceであり、別compiler、coupling map、系サイズ、full RPEまたは最終総コストへは
一般化しない。

## 2026-08-24 connected-cluster lightweight operation implementation note

既存の主1500/375 holdout結果を変更せず、compiled-cost処理をoffline calibration、transpileを
呼ばないprediction、固定calibrationのtransfer holdoutへ分離した。v2 calibration/transfer schema、
pattern単位task/checkpoint、same/different基底の直接条件付きsampling、cache状態に依存しない
deterministic-work Neyman配分、完全な数値回路・compiler・backendをkeyとするSQLite metric cacheを
実装した。さらにv3では各patternの標本列を固定indexのsample chunkへ分割し、chunkごとの十分統計量を
atomic checkpointへ保存して合成する。標本数を増やす際は完了済みfull chunkを再利用し、末尾partial
chunkの置換と新規chunkだけを計算する。same/different pairはclass別統計量を合成後に解析確率で
再重み付けする。旧v2 pattern checkpointは標本数まで完全一致する場合だけ読み取り再利用する。
production後の実現相対標準誤差が目標未達なら、分散に基づき不足層へ再配分するadaptive roundも追加した。

同じevent identityを複数short-step時間でcompileする角度不変性validatorも追加した。既存H4固定
snapshot、`L_D=3`、short-step 0.020/0.025/0.030、cluster長1--3、各pattern 2標本のsmokeでは
全6 metric差0だった。ただし低標本のimplementation smokeであり、manifestへ科学的artifactとして
登録せず、角度を除外したstructural cache reuseも有効化しない。別`L_D`、境界coverage、
compiler/coupling条件で検証するまで、short-stepごとに数値回路keyを分離する。

H4のpilot 2・production cap 4 smokeはcold約39.8秒、別checkpointから同じSQLite cacheを使う
warm run約37.3秒だった。warm transpile missは0だが、回路構築とcanonical fingerprint計算が残る。
同一checkpoint再開は完了taskを読み飛ばす。checkpoint fingerprintにはtask実装versionも含め、
実装変更後に旧summaryを黙って再利用しない。固定chunk範囲、十分統計量の合成、標本数増加時の
full chunk再利用、旧checkpoint互換、worker数による科学的出力の不変性、chunk seed改変の拒否を
回帰testで確認した。変更後のlocal全suiteは`468 passed, 4 warnings`で、warningは既存grouped-UWC
test由来である。本節は実装能力の記録であり、
新しいcost精度、full RPE、
最終総コストまたは大規模性能の検証ではない。

## 2026-08-24 operational connected-cluster compiled-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`、`K=2`の
1 compiler条件で、Taylor次数条件付きのK1--K3 connected-cluster運用推定器を検証した。
order-0単eventは厳密列挙し、pairはsame/different基底で層別、pilotからRZ countのNeyman配分を
決めた。productionとholdoutのseedは分離した。

正確なDF Hamiltonian配列をNPZ snapshotへ固定し、未使用full回路を`L=4,6,8`で評価した。
全order-0は各長さ1500標本、order-2がちょうど1回は各位置375標本である。RZ countと全6 metricの
最大点誤差は2.936%、RZ最大absolute z-scoreは2.074、予測側95%相対半幅は1.537%だった。
事前の点誤差5%と予測精度2%は通過した。一方、点wise正規近似95%上側診断は5.724%で未達、
order-2 K1は要求1535に対して標本cap 1500へ到達した。したがって代表1条件の
「実用点誤差5%内の暫定候補」であり、rigorousな5%保証とは扱わない。

別processで同じ分子条件から再構築したholdoutは、元holdoutと最大11.68%、z 5.215ずれた。
これは主結果へ結合していない。compiled-cost検証の再現単位は分子条件だけでなく正確なDF snapshotとする。
generatorはpilot K1--K3、production K1--K3、holdout L4/L6/L8の9 taskをfingerprinted checkpointへ
保存し、中断後は未完了taskだけを再開する。

全order-0 RZの差は`L=4,6,8`で+2.59%、-0.35%、+2.94%と単調増加せず、order-2が1回の
RZ点誤差は1.02%以下だった。この条件ではK4を追加せず、別`L_D`、short-step、compiler/coupling
条件への移送holdoutで悪化した場合に再検討する。full RPE、量子shot、noise、実backend、
resource accounting接続、最終総コストは未評価である。

主artifact、snapshot、9 checkpoint、versioned source、2 generator、専用test 4件および
[`docs/rte_connected_cluster_cost_validation.md`](docs/rte_connected_cluster_cost_validation.md)を
追加した。fingerprintとsnapshot SHA-256は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

当該主artifact生成後に軽量運用APIを追加した最新local全test suiteは`468 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のconnected-cluster検証由来ではない。

## 2026-08-24 hierarchical compiled-cost holdout note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`の
同一Hamiltonian表現を3 workerへ渡し、`K=0`の未使用`L=8`、`K=2`の`L=4,6`、
controlled partial-S2の未使用`q=4`を独立seedで評価した。三artifactの`preparation_hash`は一致した。

`K=0`では`C2,C3,C8`を各2000標本とし、pair-onlyはcount/sizeで最大8.851%残った一方、
triple補正は全metric最大1.744%、最大absolute z-score 0.521だった。したがって、この条件では
count/sizeにtriple、depthにpairを候補とし、4-event以上の係数は追加しない。

`K=2` runを監査すると、1 eventのorder-2確率は0.0001063で、旧`C1,C2,C3,C4,C6`
全8000 event位置中order-2は1回だけだった。したがって旧4.113%値は`K=2`内部の根拠から外す。
Taylor次数を強制した独立較正/holdoutでは最大点誤差9.119%、最大z 1.651となり、500標本の
独立係数差引きは精度不足と確認した。一方、同一trajectory上で1--3 event局所窓と全回路の
差を直接取る対応あり検証では、`L=4,6`のall-order-0/order-2が1回の全条件・全metricで
最大点誤差1.373%、RZ countの点ごとの正規近似95%診断1.796%だった。最大z 7.535なので
小さい4-event以上の残差は非ゼロだが、代表条件の暫定5%を通過した。運用時は解析的order重みと
対応ありconnected-cluster係数を使い、独立500標本係数推定は使わない。

controlled `q=1,2`各300標本から別seedの`q=4`を予測すると、全metric最大点誤差0.307%、
最大z 0.925、点ごとの正規近似95%上側診断0.958%だった。この代表条件ではaffine `q` model候補を支持するが、
`q>4`、別`L_D`、compiler/coupling条件または最終resource accountingへ一般化しない。

versioned source、並列generator、専用test、五つのfingerprinted local artifactおよび
[`docs/hierarchical_cost_validation.md`](docs/hierarchical_cost_validation.md)を追加した。
fingerprint、source hash、seed分離は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

最新local全test suiteは`457 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のcost検証由来ではない。

## 2026-08-23 high-statistics and stratified RTE boundary-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`、short-step時間0.025の
同一Hamiltonian表現を3 workerへ渡した。独立2 seedについて`C2,C3,C4,C6`を各1000標本、
別のcalibration/holdout seedでfragment-pair補正を1500/1500標本評価した。三artifactの
`preparation_hash`は一致した。

1000標本runでpair補正の最大絶対相対誤差は8.07%、8.18%、最大absolute z-scoreは
2.96、3.02だった。系統差はcount/sizeで確認し、depthではpair-onlyの最大zは1未満だった。
triple補正は最大2.33%、3.73%、最大z 0.79、0.97だったが、`mu3`自体のabsolute z-scoreは
最大1.40なので、非ゼロを確定したとは扱わない。

same-fragment確率は0.7310604だった。different境界をゼロとするsame-only modelは別seedの
pair holdoutに対して最大誤差3.65%、最大z 2.59で外れた。same/different双方の条件付き補正を
解析確率で重み付けすると最大0.849%、最大z 0.587だった。したがってpair係数には少なくとも
二分類が必要であり、長いevent列ではdepthをpair候補、count/sizeをtriple候補とする。

`rte_boundary_pair_validation_v1` source、並列generator、専用test、三つのfingerprinted local
artifactおよび[`docs/rte_boundary_pair_validation.md`](docs/rte_boundary_pair_validation.md)
を追加した。受理閾値、他の`L_D,K`、compiler、controlled回路、resource accounting接続、
最終総コストは未評価である。artifactはdirty local worktree evidenceでありimmutable CI
evidenceではない。repository全体の`overall_status = not_reproducible_from_repository`は
変更しない。

最新local全test suiteは`451 passed, 4 warnings`だった。

## 2026-08-23 RTE boundary-corrected compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`でshort-step時間を0.025に
固定し、RTE event列のcompiled-cost cluster modelをcalibration/holdout分離して検証した。
`C1`は218 eventを厳密列挙し、`C2,C3`は各300標本で較正した。別seedの未使用`C4,C6`を
各300標本で評価した結果、六指標を通じた最大絶対相対誤差はevent単純和157.51%、pair補正
8.73%、triple補正4.06%だった。最大absolute z-scoreは48.21、1.50、0.52だった。
同一DF fragmentが隣接する確率は0.73106だった。

triple残差は全metricで自身の標準誤差より小さく、depthではpair補正を一様に改善しなかった。
したがってpair補正を次の最小model候補、triple項を高次境界効果の診断量として記録する。
受理閾値、他parameter・compiler条件への一般化、resource accounting接続および最終総コストは
未評価である。

source、generator、専用test 2件、versioned/fingerprinted JSON artifact 1件および
[`docs/rte_boundary_cost_validation.md`](docs/rte_boundary_cost_validation.md)を追加した。
最新local全test suiteは`449 passed, 4 warnings`だった。artifactはdirty local worktree
evidenceでありimmutable CI evidenceではない。したがってrepository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-23 random-circuit compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`K=0`で、
complete circuitのcompiled costと、部分回路を別々にtranspileしたコスト和をpaired比較した。
`r=1`のpartial-S2は218 trajectoryを完全列挙し、3部分加法モデルは最大0.987%過大評価した。
同じDF表現を共有した`r=2`の100標本では、event別加算がRTE occurrence一体compileを
48.30--57.58%過大評価し、paired differenceのabsolute z-scoreは15.86--17.79だった。
別DF表現の300標本replicateでも52.99--61.15%の過大評価を確認した。
したがって、個々のeventを独立加算するモデルは採用せず、RTE occurrence以上をcost proxyの
最小較正単位とするpilot判断を記録した。

source、generator、専用test 3件、versioned/fingerprinted JSON artifact 3件および
[`docs/random_circuit_cost_validation.md`](docs/random_circuit_cost_validation.md)を追加した。
関連する既存cost testを含む最新local実行は`53 passed`だった。artifactはdirty local worktree
evidenceであり、1条件だけのpilotである。controlled回路、実backend、量子shot、ノイズ、
全round RPE、最終compiled総コスト、または他の`L_D,delta,r,K,q`への一般化を検証したものではない。
また、同じ分子条件でもprocess間でDF `preparation_hash`が変わるため、主$r=1,2$比較は
Hamiltonian共有batchへ置き換えた。異なるhash間の絶対compiled costは直接比較しない。

したがって、repository全体の`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-19 current local approximation-validation note

研究内容と現在地の短い統合要約は
[`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)を参照する。

2026-08-19のdirty local worktreeでは、最終コスト評価の入力を検証するため、次の三つの
result setを追加した。

- finite-RTE signal、attenuation、radius、phase-boundのH4 grid検証
- PF誤差surrogate、論文Appendix D Eq. (D6)のCPU Qiskit摂動係数、理想QPE分枝の
  H4全`L_D`検証
- H2--H6の実行可能delta窓と、PF演算子を構築しないEq. (D6) state-action係数検証

対応する文書、source、test、dirty-worktree artifactはmanifestへ登録され、構造検査は
成功している。H4全12分割では、well-conditionedな4点でfitしたEq. (D6)係数と
支配固有位相係数の差が最大0.288%だった。H2--H5の実行可能窓では両者の上包絡差が
最大1.144%で、事前の2%条件を全系が通過した。H6（DF rank 11、$L_D=5$）ではEq. (D6)による
`C_use=0.02086663`を得た。local testは`444 passed, 4 warnings`だった。これらは近似手法と実装経路の
local evidenceであり、最終compiled cost、H12の係数、量子shot、ノイズ、または外部から
再現された科学的結論ではない。artifactはimmutable CI evidenceでもない。

したがって、下記auditの`overall_status = not_reproducible_from_repository`は変更しない。
旧DF screeningとprose-only UWCを使用禁止とする判断も引き続き有効である。

## 2026-08-02 implementation hardening note

The current worktree replaces the DF legacy overlap proxy with a
shift-invariant, state-specific survival-phase-bias estimator (cache schema 8,
definition v3). It records explicit estimator status and is marked
`is_rigorous_bound=false`. Legacy/unmarked Cgs tables are now rejected by the
analytic PR-bound screening entry point, so the stale rankings described below
remain invalid and cannot be silently regenerated from the new surrogate.

Finite RTE distribution validation/serialization, actual-circuit/backend cache
identity, exact-zero circuit pruning, pre/post build workload guards, bounded
metric-only LRU caching, online compiled-cost statistics, and rolling
provenance digests were also hardened. A non-scientific Level-5-R regression
fixture now freezes all 32 combinations of `q=1..4`, raw/boundary-optimized,
controlled/uncontrolled, and exact/Monte Carlo compiled-cost evaluation.

The follow-up memory hardening makes event, partial-S2 request, exact
trajectory, and Monte Carlo trajectory generation single-pass. Level-5-R
provenance retains an explicit bounded prefix (1024 records by default) while
rolling digests cover the full stream. Cache-independent total build,
transpile-request, and instruction-application plans are rejected before any
Qiskit builder is called and checked against actual post-build work. The
Level-5-R fixture was rerun without changing gate/depth/mean/standard-error
values; schema 2 adds runtime/PRNG metadata and a digest over all 32 result
streams. The local regression suite reports `224 passed` in the documented
Python 3.11 environment. This remains implementation evidence only; no H3--H14
scientific baseline was generated by this change, and no long RPE circuit,
quantum shot, GPU statevector, noise simulation, or backend job was run.

## 結論

**監査基準 commit [`cf285c0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cf285c0ac1e3d587df4a8eb6bee2279a12ced462) の内容だけでは、現在の DF screening / UWC 検証結果を外部から再現・追跡できません。**

ここで「外部から再現可能」とは、clean checkout から、commit 済みの入力と手順を使って結果を再生成し、その結果が公開済みの数値と一致することを確認できる状態を指します。本書は既存 artifact の棚卸しであり、新たな科学計算を実行した結果ではありません。

> **DO NOT USE:** 現在 commit されている DF screening JSON を、修正済みの結果または最終結果として引用しないでください。ファイル内の算術は整合していますが、その Cgs 入力は後の commit で基底状態の不整合を理由に削除され、screening は再生成されていません。

## ステータス一覧

| 対象 | commit `cf285c0` にある証拠 | 判定 | 読み方 |
|---|---|---|---|
| 旧来の高次 Trotter 評価 | 出力付き `abe_trotter_project.ipynb` と、`artifacts/trotter_expo_coeff_gr{,_original}/` 内の係数 pickle 計 540 個 | **historical** | README が説明する旧来の高次 Trotter 解析の成果。現在の DF screening / UWC の検証証拠ではない |
| DF reduced screening | `epsilon_total=1e-4` の JSON 1件。635候補、12分子の best を収録 | **DO NOT USE / stale** | 保存値の加算と best 選択は内部整合するが、元の Cgs 表が削除済みで再生成不能。protocol 上も shortlist 前の近似 screening |
| DF 最終評価 | protocol と実装 | **incomplete** | shortlist の explicit-`L_D` Cgs 再 fit、H14 `8th(Morales)`、`4th(new_2)` が未完了 |
| UWC | 実装説明と H2--H6 等の数値表を含む Markdown | **reported only** | 表が参照する machine-readable JSON は commit されておらず、表から元 run を追跡できない |
| テスト | 4ファイルに `test_*` 関数定義が28件。UWC note に過去の `26 passed` の記録 | **current result unknown** | `cf285c0` に対するテスト実行結果ではない。この変更で追加する manifest 構造検査も科学計算・全 test suite は実行しない |

## 証拠と監査結果

### 1. DF screening

対象 artifact:

- [`artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json`](artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json)
- [`Partial Randomized Study Protocol.md`](Partial%20Randomized%20Study%20Protocol.md)
- [`artifacts/partial_randomized_pf/README.md`](artifacts/partial_randomized_pf/README.md)

JSON 自体について確認できる範囲は次のとおりです。

- `candidates` は635件で、1件は `(molecule, PF, L_D)` の組です。
- `best_by_molecule` は H3--H14 の12件です。
- 全635候補で、保存値の `g_total` は `g_det + g_rand` と一致します（最大絶対差 0）。
- 12件の `best_by_molecule` は、それぞれ同じ molecule の候補中で最小の `g_total` と一致します。

これは **JSON 内部の算術と選択処理だけ** の確認です。入力データ、Cgs fit、物理モデル、または結果の科学的妥当性を検証したことにはなりません。

再現性を失っている直接の理由は次のとおりです。

1. JSON の `cgs_table` は `/home/AbeHiromu/Project/.../df_cgs_cost_table.json` という生成環境の絶対パスを指します。
2. commit [`98f960c` (`基底状態ずれてたので削除`)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604) は、集約 Cgs 表、split 表、index の計37ファイルを削除しています。
3. その後も上記 screening JSON は残っていますが、削除理由を反映した正しい Cgs 入力から再生成された artifact はありません。

また protocol は、この計算を候補を絞るための近似と定義しています。screening では anchor の `C_gs,D(p,L_anchor)` を各 `L_D` に使い回し、**最終評価では shortlist の各 `(p, L_D)` で Cgs を再 fit して `G_total` を再計算する必要があります**。同じ protocol には、次も未完了と記録されています。

- H14 `8th(Morales)` の anchor Cgs
- H3--H14 `4th(new_2)` の anchor Cgs 計算、cost table への merge、再 screening
- shortlist に対する explicit-`L_D` Cgs の再 fit

したがって、入力問題がなかったとしても現在の JSON は最終結果ではありません。

### 2. UWC

[`notes/uwc_current_implementation_and_results.md`](notes/uwc_current_implementation_and_results.md) には、H2--H6 grouped UWC、H3 time-grid 診断、theta sweep、simple shift の条件と数値表があります。一方、同文書が参照する次の出力を含む `artifacts/grouped_uwc_pf_qpe/` は commit `cf285c0` に存在せず、`.gitignore` でディレクトリ全体が除外されています。

- `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json`
- `H3_bliss_sector_scaling_diagnostics.json`
- `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json`
- theta sweep 表の元になった run 出力

したがって Markdown の表は「報告された数値」として読めますが、repository 内の canonical raw/summary artifact と照合することはできません。なお文書自身の結論も、現在の simple BLISS quadratic shift では grouped PF+QPE cost がほぼ低下していない、という限定的なものです。

### 3. テストと CI

commit `cf285c0` の `tests/` には、静的に数えた `test_*` 関数定義が28件あります。

- `tests/test_df_hamiltonian.py`: 5件
- `tests/test_df_partial_randomized_pf.py`: 9件
- `tests/test_grouped_uwc_comparison.py`: 7件
- `tests/test_uwc_preprocessor.py`: 7件

UWC note が保存している実行記録は `.venv/bin/python -m pytest -q` の `26 passed` です。これは後から追加されたテストを含む現在の suite に対する結果ではなく、実行 commit、依存環境、完全なログも記録されていません。監査基準 commit `cf285c0` には `.github/workflows/` もありませんでした。この変更では manifest と記載パスの構造検査だけを追加しており、科学計算または全 test suite の CI ではありません。このため、`cf285c0` の28定義が pass するとは本監査から主張できません。

## 再現を妨げているもの

- 修正済みの DF Cgs 集約表・split 表・index がない。
- stale screening JSON に入力 hash、生成元 commit、実行環境、実行 command がない。
- DF screening の修正後再実行と shortlist の explicit-`L_D` 再 fit がない。
- protocol に記載された H14 `8th(Morales)` と `4th(new_2)` が未完了。
- UWC の Markdown 表に対応する machine-readable run artifact がない。
- UWC artifact の保存先が `.gitignore` され、レビュー可能な canonical summary の例外設定がない。
- 現在の HEAD を対象とする自動テスト結果がない。

## 「検証完了」とするための条件

以下をすべて満たした時点で、DF / UWC の結果を repository から外部検証可能と扱います。

1. **DF 入力を修正して固定する。** ground-state のずれを修正した Cgs を再計算し、集約表、全 split 表、index を同時生成する。各表に molecule、PF、`L_D`、入力 Hamiltonian hash、生成元 commit、生成 command を記録し、相互の件数と hash を検査する。
2. **未完了の DF ケースを埋める。** H3--H14 `4th(new_2)` と H14 `8th(Morales)` の必要な anchor Cgs を生成し、同じ canonical table に merge する。失敗または除外する場合は、対象、理由、結果への影響を明記する。
3. **screening を再生成する。** 修正後の canonical table だけを入力として `epsilon_total=1e-4` screening を実行する。結果には相対的な入力 path、全入力の content hash、生成元 commit、command、依存環境、candidate 件数を保存する。`g_total = g_det + g_rand` と molecule ごとの best 選択を自動検査し、旧 JSON を stale として置換または明確に隔離する。
4. **最終 DF 評価を実行する。** screening の shortlist と選定規則を保存し、各 `(PF, L_D)` で anchor ではない explicit-`L_D` Cgs を再 fit して `G_total` を再計算する。最終表から各 fit の machine-readable artifact と入力 hash へ追跡できるようにする。
5. **UWC の根拠データを公開する。** Markdown に載せる全表について canonical JSON/CSV を commit し、条件、baseline、seed（使用時）、backend、入力 hash、生成元 commit、command を保存する。Markdown の値が artifact から自動生成または自動照合されるようにし、必要な summary だけを `.gitignore` の例外にする。
6. **clean checkout で検証する。** 固定した依存環境と文書化した command で、小規模な end-to-end 再生成および全 test suite を CI から実行する。結果 artifact を作った commit に対する成功 check を GitHub 上に残し、比較 tolerance と期待値を test または検証 script に固定する。

上記が完了するまでは、旧来の高次 Trotter artifact、DF screening、UWC 表を互いに独立した進捗資料として扱い、現在の partial-randomized DF/UWC の完成済み検証結果として一括して引用しないでください。
