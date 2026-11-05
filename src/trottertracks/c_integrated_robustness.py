"""Small adapters for paired finite-recipe robustness; no molecular I/O on import.

Numerical APIs reuse the frozen optimizer, codec and physical callback. Real
molecular use requires the separately authorized integrated scientific scope.
"""
from collections import Counter
from dataclasses import replace
import math
import time

STARTS = ('C0', 'P1', 'P2')
SEEDS = (2026110501, 2026110502)
STRENGTHS = (.03, .1, .3)
CHECKPOINTS = (16, 64)
TRIAL_CAP = 512
PERTURBATION_FROBENIUS = .05
METRICS = ('nfev', 'njev', 'nhev', 'nit', 'value_gradient_calls_in_block',
           'hessian_calls_in_block')


def cells(*, transfer_strengths=None):
    """Metadata only. Transfer strengths must be frozen from development."""
    if transfer_strengths is not None and set(transfer_strengths) != {'PR', 'ENTRY'}:
        raise ValueError('BOTH_TRANSFER_STRENGTHS_REQUIRED')
    strengths = ({arm: STRENGTHS for arm in ('PR', 'ENTRY')}
                 if transfer_strengths is None else
                 {arm: (transfer_strengths[arm],) for arm in ('PR', 'ENTRY')})
    if set(strengths) != {'PR', 'ENTRY'} or any(
            value not in STRENGTHS for values in strengths.values() for value in values):
        raise ValueError('FIXED_RELATIVE_STRENGTH_GRID')
    return [dict(start=start, arm=arm, strength=strength,
                 checkpoints=list(CHECKPOINTS), total_trial_cap=TRIAL_CAP)
            for start in STARTS for arm in ('PR', 'ENTRY') for strength in strengths[arm]]


def call_limits():
    """Per trajectory; logical units, not hardware or elapsed-time limits.

    The accepted-update16 independent assessment is reused at the checkpoint;
    only the final endpoint needs the old extra terminal assessment. Two mass
    diagnostics and two endpoint exports include the checkpoint and terminal.
    """
    points = TRIAL_CAP + 1
    return dict(optimization_driver=1, inner_J_solve=3*points,
                precision_state=2*points, reduced_admission=points,
                independent_terminal_assessment=CHECKPOINTS[-1]+1,
                physical_assessment=CHECKPOINTS[-1]+1,
                endpoint_diagnostics=2, arm_export=2)


def perturb_initial(J, U, seed, *, amplitude=PERTURBATION_FROBENIUS):
    """Fixed PCG64 rotation perturbation; does not fit or decode stored inputs.

    Scale the aggregate skew-generator Frobenius norm, not each frame's norm.
    A later tensor-only restore is shared by both objectives, with no new seed
    or amplitude if this fixed start fails its physical qualification.
    """
    import numpy as np
    from scipy.linalg import expm
    if seed not in SEEDS or amplitude != PERTURBATION_FROBENIUS:
        raise ValueError('FIXED_INITIALIZATION_POLICY')
    if (J.dtype != np.float64 or U.dtype != np.float64 or J.ndim != 3 or
            U.shape != J.shape or J.shape[-1] != J.shape[-2] or
            not np.isfinite(J).all() or not np.isfinite(U).all() or
            not np.array_equal(J, J.transpose(0, 2, 1))):
        raise ValueError('REAL_FINITE_SYMMETRIC_INITIALIZATION')
    n = J.shape[-1]
    if np.max(abs(U.transpose(0, 2, 1) @ U-np.eye(n))) > 1e-10:
        raise ValueError('ORTHOGONAL_INITIAL_U')
    rng = np.random.Generator(np.random.PCG64(seed))
    i, j = np.triu_indices(n, 1)
    a = np.zeros_like(U)
    a[:, i, j] = rng.standard_normal((len(U), len(i)))
    a[:, j, i] = -a[:, i, j]
    norm = float(np.linalg.norm(a))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('INITIAL_DIRECTION_UNRESOLVED')
    a *= amplitude/norm
    rotated = np.stack([u @ expm(direction) for u, direction in zip(U, a, strict=True)])
    return J.copy(), rotated, dict(seed=seed, rng='PCG64',
        aggregate_generator_frobenius=float(np.linalg.norm(a)),
        rotation_side='right', J_unchanged=True, tensor_only_restore_required=True)


def neutral_restore(g, J, U, system, *, loss_scale, physical,
                    before_optimizer, after_optimizer, fitter=None, codec=None,
                    call=lambda kind, label, fn: fn()):
    """One existing gamma0 fitter invocation and existing physical assessment.

    Hooks retain raw solver/evaluation counts and independent terminal checks.
    The initial perturbed tensor may exceed the DF budget; representation
    correctness still applies. Only the restored result can become a start.
    Call in a dedicated process because the existing bridge is process-local.
    """
    from . import c_common_convergence as common
    from . import c_pr_aware_df as pr
    from . import c_near_transfer as transfer
    transfer.validate_arrays(g, J, U, system)
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('GEOMETRY_NEUTRAL_LOSS_SCALE')
    codec = codec or pr.corrected.codec.roundtrip
    fitter = fitter or common.fit
    j, u, x, report, _ = call('initialization_codec', 'faithful tensor-only start',
                               lambda: codec(J, U))
    spec = pr.Objective(system.rank, system.norb, system.ld, loss_scale, 1., 0., 1e-8)
    terminal = {}
    def checked_after(fun, raw):
        value, gradient = fun(raw.x)
        terminal.update(common.terminal(value, gradient, raw, system.rank, system.norb))
        after_optimizer(fun, raw)
    final_j, final_u, raw = call('tensor_only_restore', 'one gamma0 finite fit', lambda:
        fitter(pr, g, j, u, x, spec, before_optimizer=before_optimizer,
               after_optimizer=checked_after, options=dict(common.OPTIONS)))
    if not terminal:
        raise ValueError('INDEPENDENT_TENSOR_RESTORE_TERMINAL_MISSING')
    transfer.validate_arrays(g, final_j, final_u, system)
    row = call('restored_physical_assessment', 'restored common start',
               lambda: physical(final_j, final_u))
    return dict(J=final_j, U=final_u), dict(
        eligible=row['budget_PASS'] is True, physical=row, codec=report,
        independent_terminal=dict(terminal),
        optimizer=dict(success=bool(raw.success), status=int(raw.status),
            message=str(raw.message), nit=int(raw.nit), nfev=int(raw.nfev), njev=int(raw.njev)),
        options=dict(common.OPTIONS), objective='tensor_only_gamma0',
        loss_scale=loss_scale, strict_first_order_required=False,
        replacement_seed=False, retry=0)


def paired_specs(g, J, U, system, *, strength, loss_scale):
    """Use old objectives and smoothing, with one geometry-wide tensor scale.

    Each objective's own initial mass fixes its normalized contribution to the
    selected strength. No endpoint or objective result can reset these scales.
    """
    from . import c_near_transfer as transfer
    if strength not in STRENGTHS or not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('FIXED_OBJECTIVE_SCALE_POLICY')
    specs = transfer.objective_specs(g, J, U, system, strength=strength, epsilon=1e-8)
    return {name: replace(spec, loss_scale=loss_scale) for name, spec in specs.items()}


def recipe_cell(name, g, cold, spec, system, physical, call, record, *,
                recipe_arm=None, trajectory=None):
    """Existing paired arm with task-local64/512 policy and prefix16 capture.

    Both prefixes have the SAME512 total-trial allowance. Therefore the16
    checkpoint is an update-budget anchor, not a replay of the old16/128 run.
    Early finite stopping reuses the last qualified point for both budgets.
    """
    from . import c_near_practical as practical
    from . import c_near_transfer as transfer
    recipe_arm = recipe_arm or transfer.recipe_arm
    metrics, snapshots, last_candidate = Counter(), {}, {}
    tick, cpu = time.monotonic(), time.thread_time()

    def evidence(kind, label, data, arrays):
        if kind == 'inner_raw' and data.get('inner_summary_record'):
            for key in METRICS:
                metrics[key] += data.get(key, 0)
        if kind == 'physical_candidate':
            last_candidate.clear()
            last_candidate.update(data)
        if kind == 'accepted' and last_candidate.get('update') == CHECKPOINTS[0]:
            snapshots[CHECKPOINTS[0]] = dict(
                J=arrays['J'].copy(), U=arrays['U'].copy(),
                evidence=dict(last_candidate), solver_metrics=dict(metrics),
                accepted_U_updates=CHECKPOINTS[0], prefix_reused=True,
                wall_seconds=time.monotonic()-tick, CPU_thread_seconds=time.thread_time()-cpu)
        record(kind, label, data, arrays)

    options = dict(step=replace(practical.STEP, max_updates=CHECKPOINTS[-1]),
        policy=replace(practical.POLICY, outer_maxiter=CHECKPOINTS[-1]),
        maximum_points=TRIAL_CAP+1, arm_limits=call_limits())
    if trajectory is not None:
        options['trajectory'] = trajectory
    summary, terminal, local_call = recipe_arm(name, g, cold, spec, system,
        physical, call, evidence, **options)
    high_eligible = terminal is not None and summary['endpoint_eligible'] is True
    if high_eligible:
        final = dict(J=terminal['J'].copy(), U=terminal['U'].copy(),
            evidence=dict(state=terminal.get('state'), physical=summary['physical'],
                          independent_terminal=summary['independent_terminal']),
            solver_metrics=dict(summary['solver_metrics']),
            accepted_U_updates=summary['trajectory']['accepted_U_updates'],
            stop=summary['status'], prefix_reused=False,
            wall_seconds=summary['wall_seconds'], CPU_thread_seconds=summary['CPU_thread_seconds'],
            mass_diagnostics=summary['mass_diagnostics'])
        snapshots[CHECKPOINTS[-1]] = final
        if CHECKPOINTS[0] not in snapshots:
            snapshots[CHECKPOINTS[0]] = dict(final, prefix_reused=True,
                early_stop_before_checkpoint=True)
    if CHECKPOINTS[0] in snapshots and 'mass_diagnostics' not in snapshots[CHECKPOINTS[0]]:
        near = practical.algorithm.near
        diagnostic_spec = spec if name == 'ENTRY' else near.Objective(
            spec.rank, spec.norb, spec.ld, spec.loss_scale, 1., spec.gamma, spec.epsilon)
        snapshots[CHECKPOINTS[0]]['mass_diagnostics'] = local_call(
            'endpoint_diagnostics', 'prefix16_exact_mass',
            lambda: near.mass_diagnostics(snapshots[CHECKPOINTS[0]]['J'], diagnostic_spec))
    for budget, point in sorted(snapshots.items()):
        metadata = {key: value for key, value in point.items() if key not in ('J', 'U')}
        local_call('arm_export', f'budget{budget}', lambda budget=budget,
            metadata=metadata, point=point: record('checkpoint_export', str(budget), metadata,
                dict(J=point['J'], U=point['U'])))
    return dict(summary=summary, checkpoints=snapshots,
        checkpoint_policy='SHARED512_TRIALS_PREFIX16_TERMINAL64',
        high_terminal_eligible=high_eligible,
        call_counts=dict(local_call.counts), actual_CA_achieved=None)
