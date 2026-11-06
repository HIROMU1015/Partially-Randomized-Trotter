"""H6 design metadata and complete-sector circuit checks, dormant on import.

Numerical APIs are for invented circuits in this preparation task. Molecular
use requires a future fixed scientific scope and a separately authorized runner.
The existing optimizer, CAR, compiler, simulator and statistics are not copied.
"""
VERSION = 'h6_complete_sector_raw_compiled_actual_inverse_v1'
R_VALUES = (32, 64, 128)
STRENGTHS = (.03, .1, .3)


def specification():
    """Proposed conditions, not adopted conditions or a generated input."""
    return dict(atoms=6, norb=6, modes=12, Nalpha=3, Nbeta=3,
                distance_angstrom=1., basis='sto-3g', charge=0, spin=0,
                sector_dimension=400, system_dimension=4096,
                controlled_dimension=8192, rank=14, LD=4,
                orbital_frame='canonical_RHF_MO', kappa='original_g_corrected',
                scalar='nuclear_once')


def workload(*, numerical_coverage='registered_subset'):
    """Finite scientific-work ceiling, independent of CPU/thread allocation.

    Only update64 enters downstream; update16 is a saved optimizer diagnostic.
    Count state actions per input column, not per simulator batch/job.
    """
    from .c_near_transfer import optimization_call_model
    per_arm = optimization_call_model(accepted_update_cap=64,
                                      total_trial_cap=512)['per_arm']
    # Preserve the existing prefix16 diagnostic/export as well as terminal64.
    per_arm = dict(per_arm, endpoint_diagnostics=2, arm_export=2)
    trajectories = 2 * len(STRENGTHS)
    candidates = trajectories
    policies = candidates * len(R_VALUES)
    final_pairs = 2 * 2  # two arms, common and retuned, exact overlaps deduplicate
    selection = policies * 32
    final = final_pairs * 128
    kernels = selection + final + final_pairs  # last term: same-J/U D kernels
    if numerical_coverage not in ('registered_subset', 'every_kernel'):
        raise ValueError('FIXED_NUMERICAL_COVERAGE')
    # Subset index0 in each selection policy and each independent final pair,
    # plus every deterministic kernel. No cost-dependent correctness sampling.
    checked = policies + final_pairs + final_pairs if numerical_coverage == 'registered_subset' else kernels
    return dict(trajectories=trajectories, optimizer_checkpoint_exports=12,
                downstream_candidates=candidates, candidate_r_policies=policies,
                selection_native_max=selection, final_pairs_max=final_pairs,
                final_native_max=final, stochastic_native_max=selection+final,
                same_JU_D_max=final_pairs, native_kernels_max=kernels,
                numerical_coverage=numerical_coverage,
                numerically_checked_kernels_max=checked,
                basis_inputs_per_kernel=800,
                state_actions_per_kernel=6*800,
                state_action_columns_max=checked*6*800,
                optimizer_per_trajectory=per_arm,
                optimizer_units_max=trajectories*sum(per_arm.values()),
                # Disjoint stage envelopes; internal optimizer nfev/njev are
                # reported separately, not misrepresented as bounded unit cost.
                stage_units=dict(input_neutral=32,
                    optimization=trajectories*sum(per_arm.values()),
                    endpoint_reference=4096,
                    native_build_compile_and_reports=kernels*64,
                    native_state_actions=checked*6*800,
                    statistics_and_exports=256),
                maximum_logical_units=(32+trajectories*sum(per_arm.values())+
                    4096+kernels*64+checked*6*800+256))


def select_training(rows):
    """Fixed H6 grid; reuse the existing selector and preserve H4 defaults."""
    from .c_integrated_plan import select_training as existing
    return existing(rows, strengths=STRENGTHS, r_values=R_VALUES,
                    common_r=32, tie_order=R_VALUES)


def sector_reference_context(arm, generators, context, h, *, operators=None):
    """Reuse the random-path oracle with sector CAR, avoiding full-Fock CAR.

    Complete physical factors preserve spin sector. Individual native Gaussian
    gates are still simulated on full output space, never projected here.
    """
    from .c_pr_aware_df_stage_b2 import reference_context
    if context.n != arm.J.shape[-1] or len(generators) != len(arm.J)+1:
        raise ValueError('SAME_CANDIDATE_SECTOR_CONTEXT')
    return reference_context(arm, generators, context.bilinears, h, atoms=operators)


def check_kernel(raw, compiled, reference, sector_indices, *,
                 batch_columns=16, threads=1, inverse_factory=None,
                 action_call=None, column_observer=None):
    """All ancilla x sector inputs, full outputs, actual inverses and returns.

    This is NOT full-Fock input coverage. ColumnStream's floating-point level B
    is retained; no rigorous enclosure or implemented-law qualification follows.
    There is no phase fit, output projection, probe selection or relaxed atol.
    """
    import numpy as np
    from . import c1_h6_state_action as action
    from . import c1_subspace_synthesis_bound as bounds
    from .c1_h6_synthesis_bound import circuit_identity
    from .c1_stage2_native import actual_adjoint
    if type(batch_columns) is not int or batch_columns < 1:
        raise ValueError('POSITIVE_BATCH_COLUMNS')
    if raw.num_qubits != compiled.num_qubits or raw.num_qubits < 2:
        raise ValueError('MATCHED_CONTROLLED_DIMENSIONS')
    # P0 structure proves ideal unitarity of the compiled operation, not its
    # equivalence to the Hamiltonian or its numerical simulation accuracy.
    identity = circuit_identity(compiled)
    system = 1 << (raw.num_qubits-1)
    indices = np.asarray(sector_indices)
    w = np.asarray(reference)
    info = bounds.reference_info(w)
    d = len(indices)
    # Validate the complete embedding before any circuit execution.
    bounds.basis_batch(indices, system, w, [0])
    make_inverse = actual_adjoint if inverse_factory is None else inverse_factory
    inverses = {name: make_inverse(qc) for name, qc in
                (('raw', raw), ('compiled', compiled))}
    engine = action.StateActions(threads)
    engine.backend.set_options(zero_threshold=0., enable_truncation=False)
    def apply(qc, states):
        fn = lambda: engine.apply(qc, states)
        return fn() if action_call is None else action_call(len(states), fn)
    labels = ('raw_reference', 'compiled_raw', 'compiled_reference',
              'raw_inverse_reference', 'compiled_inverse_reference',
              'raw_inverse_return', 'compiled_inverse_return')
    streams = {key: bounds.ColumnStream(2*d, 2*system) for key in labels}
    maximum_norm_drift = 0.
    for lo in range(0, 2*d, batch_columns):
        ids = list(range(lo, min(lo+batch_columns, 2*d)))
        initial, expected = bounds.basis_batch(indices, system, w, ids)
        _, inverse_expected = bounds.basis_batch(indices, system, w.conj().T, ids)
        r = apply(raw, initial)
        c = apply(compiled, initial)
        ri = apply(inverses['raw'], initial)
        ci = apply(inverses['compiled'], initial)
        rr = apply(inverses['raw'], r)
        cr = apply(inverses['compiled'], c)
        if column_observer is not None:
            column_observer(ids, r, c)
        pairs = ((r, expected), (c, r), (c, expected),
                 (ri, inverse_expected), (ci, inverse_expected),
                 (rr, initial), (cr, initial))
        for key, (actual, target) in zip(labels, pairs, strict=True):
            streams[key].add(ids, actual, target)
            gate = action.state_gate(actual, target, label=key)
            maximum_norm_drift = max(maximum_norm_drift, gate['norm_drift'])
    reports = {key: stream.finish() for key, stream in streams.items()}
    if any(row['status'] != 'COMPLETE' for row in reports.values()):
        raise ValueError('INCOMPLETE_BASIS')
    return dict(status='COMPLETE_SECTOR_NUMERICAL_CHECK_PASS', version=VERSION,
        scope='all ancilla x supplied sector inputs; all full-space outputs',
        full_Fock_input_coverage=(d == system),
        full_Fock_operator_equality_claim=False,
        input_dimension=2*d, output_dimension=2*system,
        state_action_columns=6*2*d, compiled_structure=identity,
        reference=info, checks=reports, max_norm_drift=maximum_norm_drift,
        component_atol=action.COMPONENT_ATOL, global_phase_fit=False,
        rigorous_enclosure=None, actual_CA_achieved=None,
        implemented_PR_law_error=None)


def memory_model(*, batch_columns=16):
    """Array sizes only; simulator/Python/JAX overhead must be measured."""
    if type(batch_columns) is not int or batch_columns < 1:
        raise ValueError('POSITIVE_BATCH_COLUMNS')
    return dict(full_controlled_matrix_bytes=16*8192**2,
                four_full_controlled_matrices_bytes=4*16*8192**2,
                complete_restricted_columns_bytes=16*800*8192,
                state_batch_bytes=16*batch_columns*8192,
                # 10 live batch-shaped arrays is a conservative algorithmic
                # workspace model, not a process RSS cap or exact peak bound.
                batch_array_workspace_model_bytes=10*16*batch_columns*8192,
                sector_matrix_bytes=16*400**2,
                mapped_tail_atoms_max=10*6*13,
                dense_atom_cache_bytes=780*16*400**2,
                atom_plus_three_r_event_caches_model_bytes=4*780*16*400**2,
                independent_J_parameters=14*6*7//2,
                independent_U_parameters=14*6*5//2)
