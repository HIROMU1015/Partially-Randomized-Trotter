"""Inactive H4 contract proposals and synthetic dimension preflight.

No H4 molecular generation or science dispatcher is implemented here. Proposed
parameters and structural prefix rules require a later explicit batch approval.
"""
from __future__ import annotations

import math


def structural_prefix(weights):
    """Draft-only rule: first ranked prefix covering half of ranking weight.

    These are representation-specific ranking weights, not common operator
    norms or a cost-optimality criterion. A nonempty random tail is mandatory.
    """
    values=[float(x) for x in weights]
    if len(values)<2 or any(not math.isfinite(x) or x<0 for x in values):
        raise ValueError("DRAFT_PREFIX_NEEDS_TWO_NONNEGATIVE_RANKED_WEIGHTS")
    if any(a<b for a,b in zip(values,values[1:])) or math.fsum(values)<=0:
        raise ValueError("DRAFT_PREFIX_WEIGHT_ORDER")
    target=.5*math.fsum(values)
    for length in range(1,len(values)):
        if math.fsum(values[:length])>=target and math.fsum(values[length:])>0:
            return length
    raise ValueError("DRAFT_PREFIX_NONEMPTY_RANDOM_TAIL_REQUIRED")


def h4_contract_draft():
    from .h2_non_df_pilot import contract as old
    base=old()
    return dict(schema="non_df_h4_pilot_draft_v1",status="DRAFT_NOT_ADOPTED_NOT_AUTHORIZED",
        execution_authorized=False,grant_valid=False,dispatch_enabled=False,
        system=dict(molecule="linear H4",geometry_Angstrom=[["H",[0.,0.,float(i)]] for i in range(4)],
            adjacent_distance_Angstrom=1.,basis="sto-3g",charge=0,spin_2S=0,symmetry=False,
            active_space=None,frozen_core=None,spatial_orbitals=4,spin_orbitals=8,
            SCF=dict(method="canonical real RHF",conv_tol=1e-12,max_cycle=100,init_guess="minao",calls=1),
            integrals="same unthresholded real orthonormal MO h, chemist eri for both representations; interleaved alpha/beta; spin lift and 1/2 exactly as frozen H2",
            qubit_order="qiskit_little_endian",coefficient_atol=0.,identity="retained exactly once including nuclear phase"),
        DF=dict(decomposition_calls=1,df_tol=1e-8,final_rank=None,rank_policy="tol_only; actual rank recorded, never copied from H2 or historical H4",
            generation="existing OpenFermion low_rank_two_body_decomposition; original indices retained",
            rank=None,ranking="descending abs(lambda)*Frobenius(g)^2, original index tie-break",
            ranking_weight_is_involution_lambda=False,one_body="always deterministic",
            prefix="first prefix covering >=50% ranking weight, with positive nonempty tail; structural rule only",
            identity_policy="extract_identity_phase"),
        Pauli=dict(mapping="existing unthresholded supplied-MO JW conversion",nonidentity_term_count=None,
            ordering="descending abs(coefficient), label tie-break",coefficient_cutoff=0.,
            prefix="first prefix covering >=50% nonidentity abs-coefficient sum, with positive nonempty tail; structural rule only"),
        state=dict(method="full non-DF Nalpha=Nbeta=2 sector ground eigenvector; canonical largest amplitude positive real",
            dimension=36,solve_calls=1,same_state_for_four_cases=True,DF_sector_crosscheck_calls=1,
            FCI_crosscheck_calls=1,reference="each representation own full-H at fixed T on the common non-DF state",
            preparation_cost=0.,preparation_excluded=True),
        fixed_candidates=dict(count=4,methods=["PF","PR"],representations=["pauli_jw","df_native"],
            PF="all terms/fragments; deterministic one-body also retained",
            PR="one structural prefix per representation; actual labels/indices/weights bound to input receipt before release",
            same_L_D_is_same_partition=False,search=False,adaptive_candidates=False,
            empty_tail_policy="STOP_SCOPE_NOT_SATISFIED; do not substitute PF or change prefix"),
        evolution=dict(T=.8,q=8,delta=.1,product_formula="S2",r=4,K=4,
            proposal_reason="single smaller step than H2 for a larger system, no tuning or expected accuracy guarantee",
            law="existing finite_rte_even_paired; no new randomization or estimator",
            normalization="each actual involutory tail lambda; q*r log(B_K); identity/event phases preserved"),
        precision=dict(epsilon_sig=[.05,.01,.005,.001,.0005],epsilon_axis="epsilon_sig/sqrt(2)",alpha_axis=.025,
            statistical_model="existing fresh IID corrected +/-1 Hoeffding sufficient N",
            numerical_allowance="same frozen empirical 64*eps operation proxy, safety10, floor1e-12, dense/sparse/Horner closure",
            allowance_is_certified=False,maximum_u_over_headroom=.01,
            ineligible_policy="N/G null with reason; do not lower allowance or change parameters to obtain eligibility",
            large_dimension_caveat="the dense operation proxy may dominate tight precision rows; no guarantee of non-null H4 N/G"),
        MC=dict(trajectories_per_PR=256,paired_cosine_sine=True,adaptive_samples=False,
            proposal_reason="fixed finite pilot cohort larger than H2; 256 is not justified as a rare-order or population-precision guarantee",
            seed_rule="sha256(NON_DF_H4_PILOT_V1/<representation>/<method>)[:8] little-endian; trajectory PCG64 order then components v2 unchanged",
            seed_namespace_fixed_before_execution=True,uncertainty="existing paired covariance SE and nominal Student-t95 only",
            rare_orders="record law probabilities, observed positions, all per-trajectory event records; unobserved costs unknown",
            event_audit="serialize the exact events returned by the production sampler before circuit generation, including position/order/component IDs and scalar/rotation phase, then join compiler price by immutable trajectory_id; no second sampling",
            model_or_estimator_adopted=False),
        compiler=base["compiler"],environment=base["environment"],
        numerical=base["numerical"],
        gates=["SCF convergence and orbital orthonormality; expected 4 spatial/8 spin dimensions",
            "Pauli vs independent JW, DF full operator and existing DF physical-sector operator",
            "common state norm/residual/energy, DF state crosscheck, independent PySCF FCI",
            "dense/sparse reference, independent law mean and raw/normalization closure",
            "forced order0/2/4 algebra checks separate from cost MC; first3 independent trajectories",
            "ordinary controlled phases and all sampled cosine/sine compiled probabilities",
            "per-axis C/N/G, exact rational mean identity, paired covariance SE, event log/price join",
            "SOURCE/input/compiler/version/hash binding, no B0/B0I mixing"],
        execution=dict(new_namespace="must be chosen unused and sealed once before approved execution",
            SOURCE="future clean implementation commit + all scientific code, runner, tests, contract, environment and user approval hashes",
            input_binding="SCF/integral/DF/state receipts and hashes sealed before launching four cases; actual rank and prefix numbers resolved by frozen rules, no search",
            case_claims="fixed grant-specific claim directory and case_id with SOURCE/input/grant binding; changing run ID cannot refund a claim",
            one_shot=True,automatic_retry=False,resume=False,old_grants_reusable=False,mandatory_STOP=True,
            approved_future_call_budget=dict(SCF=1,DF_decomposition=1,FCI=1,nonDF_sector_ground=1,DF_sector_crosscheck=1,fixed_cases=4),
            batch_approval="one explicit approval for the fully sealed H4 execution contract covers all listed stages; no per-stage confirmations",
            partial_failure="stop peer processes, persist pending/failed/complete states; do not retry any case or regenerate inputs",
            watch_policy=dict(ready_timeout=None,run_timeout=None,heartbeat_timeout=None,poll_seconds=.05,terminate_grace=5.,kill_grace=5.),
            watchdog_semantics="no artificial compute deadline; future per-stage contracts may set measured deadlines or independent heartbeat; no thread heartbeat used to declare a busy compiler hung",
            resources="fresh processes per independent case, available resources observed before and during; no arbitrary CPU/RAM/worker caps or global changes",
            deduplication="share immutable hashed input files, per-case single adapter and cache for identical circuits; never drop repeated gates or change random draw order",
            evidence="case SOURCE/input/grant, claim/marker, complete event logs, compiler counts and paired prices, stdout/stderr/exit, numerical gates, ledger, STOP; supervisor own descendants only"),
        readiness=dict(scientific_dispatch_implemented=False,H4_inputs_generated=False,
            missing_before_execution=["explicit H4 batch authorization", "dedicated H4 molecular runner and dimension-specific workload gates",
                "production sampler event-audit hook with no duplicate draw and artificial equivalence test",
                "future clean SOURCE/environment seal and new valid grant", "launch-time resource and filesystem check"],
            permitted_now="artificial process and synthetic eight-qubit circuit tests; saved H2 read-only evidence only"))


def rare_event_acquisition_draft():
    return dict(schema="rare_event_information_acquisition_draft_v1",status="DRAFT_REQUIRES_SCIENTIFIC_DESIGN_AND_EXECUTION_APPROVAL",
        execution_authorized=False,grant_valid=False,dispatch_enabled=False,
        goal="acquire missing full-wrapper costs linked to explicit rare-order event records, not overwrite H2 estimates",
        scope="H2 same snapshots, T=.8 q4 r2 K4 and original Pauli LD7/DF LD1; old grants and cohorts read-only",
        proposed_diagnostic_grid=dict(representations=["pauli_jw_PR","df_native_PR"],forced_orders=[2,4],
            event_positions=list(range(8)),conditional_component_draws_per_cell=8,total_trajectories=256,
            wrappers_per_trajectory=["cosine","sine"],compiler="original H2 Qiskit1.3 opt1 seed17 rz/sx/x/cx ordinary-control full wrapper"),
        proposed_conditional_definition="one preselected position has the declared order; remaining seven positions have order0; component IDs follow each representation's existing normalized involutory law with separately fixed seed namespace",
        scientific_definition_adopted=False,
        seed_rule_proposal="sha256(RARE_EVENT_DIAGNOSTIC_V1/<representation>/<forced_order>/<position>/<index>)[:8] little-endian; fixed before run",
        exact_requirements=["separate new SOURCE/contract/grant/one-shot namespace", "explicit adoption of conditional diagnostic definition",
            "event and phase records before compile; full-wrapper validation and cost join", "no use of forced diagnostic trajectories in original IID MC estimates"],
        limitations=["Conditional cell prices are not a marginal expected population cost",
            "This grid does not evaluate multiple simultaneous rare-order events or full component support",
            "A rigorous population bound needs separately justified cost bounds/support or an approved estimator and statistical design",
            "No unbiasedness, coverage, sample sufficiency, population ranking or rare-event guarantee is adopted here"],
        execution_now=False,new_sampling_now=0,new_compile_now=0,automatic_next_stage=False)
