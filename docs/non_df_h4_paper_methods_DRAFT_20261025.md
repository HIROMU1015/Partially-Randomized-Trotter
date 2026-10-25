# Methods draft: fixed-H4 Pauli/DF empirical and conditional resource comparison

Editorial draft, 2026-10-25. This text reuses saved evidence at base commit `03efd057a7e77c40ae7e90b9c77885d6606e793b`. It does not report a new calculation, accept a certificate or select the paper's final scope. Figure and table numbers refer to the [paper plan](non_df_h4_paper_plan_20261025.md). Repository paths below are relative to the repository root; the [figure/data map](../artifacts/non_df_h4_paper_plan/2026-10-25/figure_table_plan.json) supplies exact pointers and selectors.

## Molecular input and target scope

The case study uses a linear neutral singlet H4 chain with atomic positions (0,0,0), (0,0,1), (0,0,2) and (0,0,3) Å in the STO-3G basis. Both representations use the same saved molecular-orbital integrals and eight system qubits. The Pauli-JW representation contains 184 nonidentity terms, with its scalar contribution retained separately. The DF-native representation uses the saved tol-only factorization at tolerance 10⁻⁸ and actual rank 10, together with a one-body term. The input receipt reports a truncation value of 0 and a full-Hamiltonian agreement diagnostic of `8.462237051574415e-15` at threshold `1e-9`. This is a saved numerical consistency test, not a proof of exact Hamiltonian equality.

The fixed raw input ray is identified by SHA256 `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`. C-reference and local residual calculations use the common exact squared norm of that saved binary64 vector. We do not replace it with an independently normalized vector or divide by an approximate output norm. The retained molecular conventions include real orthonormal spatial MOs, spin-orbital ordering `2*p+alpha(0)/beta(1)`, Qiskit little-endian wires and Pauli labels ordered `q[n-1]...q[0]`. Constants and controlled relative phases are retained.

Sources: `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json`; C models at `artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/{pauli_jw,df_native}_model.json`. Hamiltonian and state hashes are available in those saved inputs. Each representation has its own saved full-H reference signal. Sharing a ray does not establish a new error transfer to a common exact Hamiltonian; the scope is recorded in `reference_scopes.json` under the central conditional-resource results.

## Deterministic splitting and finite PR law

The fixed evolution time is T=0.8 in the saved Hamiltonian's atomic-unit convention, with q=8 symmetric second-order (S2) steps. PR uses r=4 random events per step and finite even Taylor order K=4. PF retains the complete decomposition. Pauli PR retains the first 27 nonidentity terms under descending absolute coefficient weight, with label tie breaking, at the first prefix reaching 50% of the total weight. These saved retained terms contain only Z and ZZ. The constant is always retained separately. DF PR retains fragment 0 and the always-retained one-body term; fragment ranking uses `abs(lambda)*||g||_F²`, with original-index tie breaking. Tail identity contributions enter a deterministic phase. Equal percentage thresholds and different prefix sizes do not define matched physical splits.

The finite paired-Taylor event law allows k=0,2,4, with saved construction weights proportional to

\[
w_k=\frac{\tau^k}{k!}\sqrt{1+\left(\frac{\tau}{k+1}\right)^2},\qquad
\tau=\lambda_R(T/q)/r.
\]

The law normalizes these weights over the allowed finite set. Components are selected according to absolute involution coefficients. Each event applies k signed Pauli or rotated-involution products in their original sequence, followed by one rotation, with the saved order phase `(-1)^(k/2)`. The event generator's rotation parameter is `atan(tau/(k+1))`; its mapping to the circuit RZ angle follows the original primitive, rather than identifying these two angles. The Pauli and DF values of lambda_R, component distributions and rotation operands differ. This method description quotes the original SOURCE; no event probabilities or trajectories were regenerated for the paper plan.

The PR background is [Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2). Its analytical cost model and phase-estimation setting are distinct from this study's compiled-wrapper prices and fixed complex-signal tolerance. The finite law used here must not be presented as the paper's complete infinite even-order law. RTE order k=2/4 is distinct from the S2 product-formula order.

Original SOURCE: `0c329b30a81accb35c212be8defa66fee0dd0f12`, particularly `h4_molecular_inputs.resolve_prefixes`, `non_df_h4_readiness.structural_prefix`, `rte`, `pauli_signal_adapter` and the DF builders. Descriptive SOURCE correspondence is documented in `docs/non_df_h4_cost_reversal_structure_20261024.md`; later D1/D2 results supersede that report's then-unmeasured diagnostic status without changing its historical text.

## Control constructions, wrapper and compilation

Generic Pauli circuits control the saved system body as a whole. Central Pauli circuits instead leave basis/parity conjugations uncontrolled and control the central rotations, preserving signed products, order phases, scalar phases, wire positions and operand order. These are separate implementation series. D2 compares the two constructions on the same saved PF body and PR-index0 body. The resulting cost difference includes the compiler's response to the changed input family; it is not an isolated compiler-pass effect.

DF-native circuits leave Gaussian basis changes uncontrolled and control the central diagonal/product/phase operations. Their controlled-RZZ decomposition can itself use a generic-control path. Adjacent equal-basis reuse occurs within an event occurrence; identity and occurrence boundaries flush the active basis. The saved repeated-step policy is raw concatenation, without extending the builder's reuse across deterministic halves or step boundaries. Possible later compiler cancellations are included in the observed full-circuit costs, not predicted by summing literal component costs.

The full Hadamard wrapper uses ancilla wire 8. Cosine uses the final Hadamard directly; sine uses Sdg before that Hadamard and the saved +Im convention. Ancilla 8 is measured to classical bit 0. Global phases that become controlled relative phases are retained. State preparation is excluded from the reported price.

Compilation uses Qiskit 1.3.0, basis `rz/sx/x/cx`, optimization level 1 and transpiler seed 17, with no backend, coupling map, layout or routing specified. RZ count is the primary price; CX count, depth, RZ depth and CX depth are supplementary metrics. These are logical compiled-gate counts, not fault-tolerant synthesis costs, wall-clock times or end-to-end phase-estimation resources.

Sources: D2 SOURCE `74515d8eb4f073c40fc783681a3775dbca89e9b1`, corresponding saved result/comparison CSV; original DF SOURCE and descriptors; the A_COHORT_COS and A_AXES result directories. No circuit was built or compiled in preparing this draft.

## Saved price cohorts and statistical interpretation

PF has a deterministic singleton price. Each PR representation has a saved cohort of 256 trajectories. The same Pauli trajectory IDs connect generic and central control, and cosine and sine prices are paired by trajectory within a representation. Pauli and DF indices do not form matched pairs because the laws differ. The saved central Pauli cohort has identical cosine/sine prices for all 256 trajectories; this is an observation about that cohort, not an all-law identity.

Means, standard errors, nominal Student-t 95% half-widths and any histograms are quoted from the saved analyses. PF does not receive a Monte Carlo confidence interval. The PR intervals describe nominal finite-sample inference and are not rigorous population-cost bounds, particularly in the presence of unseen event strata. Per-trajectory percentage reductions and reduction of aggregate means are different saved statistics and are labeled separately if used. This draft does not compute new averages, errors, bins, quantiles or rankings.

Sources: `artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json`; `artifacts/non_df_h4_a_cohort_cos/2026-10-24/run01/statistics.json`; `artifacts/non_df_h4_a_axes/2026-10-24/run01/statistics.json`. The previous synthesis supplies pointer-linked quotation CSVs for six circuit-price series and 30 hypothetical-zero-residual resource rows.

## D1 diagnostics and provenance of reproduction

The initial D1 run stopped at the old Pauli-PR recursive-fingerprint mismatch before any of its nine top-level price compilations. Its missing old payload and root cause remain unknown. A later, distinct SOURCE-based criterion-B run verified the saved input/descriptor/phase/control correspondence, reproduced all four old full-circuit RZ prices and completed the remaining five ablations, for nine measured configurations. The later success does not change the initial failure or recover the missing old program.

For each saved generic PR-index0 circuit, retained-only and event-only ablations and an empty wrapper define the saved operational interaction

\[
\Delta=C_F-C_R-C_E+C_0.
\]

This interaction is quoted directly from the saved CSV. It is not a basis-only price or an additive causal allocation of full-circuit cost. The isolated circuits are diagnostics rather than valid alternative simulation algorithms. Depth is reported for each configuration without additive decomposition. The D1 prices cannot be assigned to central-control subcircuits.

Sources: initial D1 SOURCE `48d4d27a05a529e05b11599f5f3c671302826617`; follow-up SOURCE `9843b803878db6c3aee7ab5e7c6bb4818572c944`; `artifacts/non_df_h4_d1_source_execution/2026-10-24/{metrics_table.csv,interaction_table.csv,result_summary.json}`. This is an author-performed evidence correspondence check, not an independent scientific review.

## Conditional shot and total-resource accounting

For each signal tolerance ε_sig in {.05,.01,.005,.001,.0005}, the saved accounting allocates ε_axis using the original binary64 ε_sig/√2 convention and a failure budget α_axis=.025 to each of cosine and sine. One condition therefore has total failure budget .05; this is not a simultaneous 95% statement over all tolerances and methods. The price-cohort size 256 is unrelated to the required quantum measurement count N. No quantum shots were executed in this case study.

The existing calculation uses the saved signed bias b0, saved encoded-error term u_enc, and an unknown nonnegative implementation-residual term ρ:

\[
u_a=u_{\mathrm{enc},a}+\rho_a,\quad
h_a=\epsilon_a-|b_{0,a}|-u_a,\quad
h_a>0,\quad u_a/h_a\le .01.
\]

The normalization sufficient condition uses the unchanged saved policy

\[
B_{\mathrm{up}}=\max\{1,B_{\mathrm{saved}},\exp(\mathrm{saved\ log\_B\_upper})\},\qquad
N_a=\left\lceil\frac{2B_{\mathrm{up}}^2\ln(2/\alpha_a)}{h_a^2}\right\rceil,
\]

with the saved safe-integer condition N_a<2⁵². PF's historical log_B margin is retained even when it makes B_up exceed 1. Saved B values are not replaced by newly certified values. In the adopted accounting u_enc uses the saved C bias-distance upper evidence and the original signed-bias scale; the draft does not reinterpret it as a newly proved full end-to-end error.

Figures quote the previously evaluated hypothetical slice ρ_cos=ρ_sin=0. That slice is a sensitivity condition, not a measured zero residual. For a saved trajectory i, the accounting combines both axes as

\[
G_i=N_{\cos}C_{i,\cos}+N_{\sin}C_{i,\sin}.
\]

The saved PR quantity is the paired-cohort mean of G_i. Its saved nominal uncertainty retains the cosine/sine covariance; the two axis errors must not be treated as independent. This is not a certified population-mean G. The central series reuses the encoded-target bias and normalization of its generic counterpart, hence the saved hypothetical-zero-residual N values agree; this does not prove zero residual for central compiled circuits.

The saved guard comparisons allow method- and axis-specific residuals independently within their fixed boxes. Nonnegative prices and monotonicity of the saved shot sufficient condition give the stored endpoint ranges. The central analysis evaluated three specified pairs at five tolerances, for 15 comparisons; the earlier generic/DF analysis contains 30 comparisons. We quote these results and do not add new pairs. Guard-box ranges are deterministic sensitivity ranges, not confidence intervals and not evidence that actual residuals lie in the boxes.

Sources: generic conditional-accounting SOURCE `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8`; central SOURCE `d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34`; `conditional_functions.json` in the generic run; and `conditional_results`, `axis_results`, `comparison_regions`, `error_scale_provenance` and `reference_scopes` in `artifacts/non_df_h4_central_conditional_cng/2026-10-24/recovery02/run01/`. Formulas are described here, not reevaluated. Legacy actual-N/G null fields remain unchanged.

## Rare events and population costs

Both saved cohorts contain zero order4 trajectories. This absence does not make their cost zero: the finite implemented laws assign them positive saved mass. DF also has a same-data-selected set of unobserved components. That set is descriptive, not an independently fixed coverage stratum. Nominal per-event probabilities, finite-grid implemented trajectory masses and observed counts must be reported as different quantities.

The saved maximum and nominal t interval do not bound unobserved full-wrapper costs or the remainder population mean. A stronger population-cost comparison would require justified cost bounds and suitable statistical treatment for both sides of the comparison. The original PR paper's analytical average-cost formulas cannot be substituted for these missing compiled-cost bounds. Obtaining a gate-count bound is separate from proving numerical operator errors of the compiler. Further sampling, an isolated order4 compile, or a full compiler numerical proof is not a default requirement for the limited empirical/conditional claims in this draft.

Source: `artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/rare_cost_sensitivity.json`; saved order-group summaries in the central cohort statistics; the synthesis report's distinction between nominal and implemented probability evidence.

## Local numerical enclosures and unresolved precision scope

Six saved Arb diagnostics enclose the local implementation residual for both axes of the central Pauli PF singleton and PR trajectories 0 and 1. PR1 contains an order2 event. Each uses the fixed raw ray, exact common norm, saved operands and compiled QPY, preserving relative phases and the +Im sine convention. The local residual is

\[
\rho_{i,a}=B_{\mathrm{saved}}|m_{\mathrm{compiled},i,a}-m_{C,i,a}|,
\]

with B_saved applied once. PF uses the saved complete C-target interval; individual PR C-targets are enclosed from their saved events, rather than replaced by the PR-law mean or empirical binary64 point values. All six diagnostics stopped at 64-bit precision with upper endpoints within their saved local budgets. Table S3 quotes their saved outward endpoints without recomputing them.

The PF result covers the fixed singleton on both axes. PR indices 0 and 1 do not cover the remaining cohort, unseen order4 events or the whole PR law, and no DF coverage follows. R2-C remains `PROOF_INCOMPLETE`; no continuation of that proof was undertaken for this manuscript plan. Local numerical inclusion is distinct from formal certificate acceptance, which remains false. No local bound is inserted into a new N/G calculation here.

Source: `artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.json`, which links each individual result and scientific SOURCE; `docs/non_df_h4_r2c_compiler_phase_proof_20261025.md` for the existing unresolved status.

## Data availability and limits of completion evidence

Saved inputs, JSON/CSV results, QPYs, scientific SOURCE commits and the existing run contracts remain unchanged in the repository. The figure plan links the original datasets and existing PDF/SVG/PNG figures; no new scientific analysis script or dataset was introduced for this draft. The original H4 native-completion evidence remains incomplete/UNKNOWN and the historical Pauli-PR fingerprint failure remains unresolved. Later runs' native exits establish only those later runs' completion; they do not repair the original missing evidence.

The study is a fixed-molecule, fixed-split, fixed-compiler case study. It does not isolate a universal Pauli-versus-DF effect, prove a population-mean PR advantage, establish actual end-to-end signal precision, or provide an energy/RPE/fault-tolerant resource estimate. Original unresolved C/L connections, legacy N/G nulls, full-law residuals and population-cost bounds retain their saved status. Publication of this editorial draft does not accept a formal scientific certificate or a final research conclusion.

Editorial next step: the user chooses the claim scope; Codex can then prepare the main prose and final figure placement within a separately stated editing task. New numerical analysis, circuit compilation, scientific validation or important final review requires its own explicit permitted scope. **Mandatory STOP after saving and remote verification of this preparation.**
