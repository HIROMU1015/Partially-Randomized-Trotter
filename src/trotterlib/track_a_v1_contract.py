"""H6 V1 preregistration and metadata arithmetic; standard library only.

No molecular inputs, circuits, probabilities or scientific costs are evaluated
on import. A cancellation-free signal diagnostic is not a certified bias bound.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

BASE = '46c06e5653152eb4ce786d69d927ecbf12f9a2de'
PROVIDER = '87b6b9f46b7e8ba9abe132ee21478c114b6f46f7'
EPSILONS = (.05, .01, .005, .001, .0005)
QS = (1, 2, 4, 8)
FACTORS = (1., 10.)
AXES = ('cosine', 'sine')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def cell(method, prefix, q, formula='2nd', r=None, K=None):
    name = f'{method}_p{prefix}_q{q}_{formula}'
    if r is not None:
        name += f'_r{r}_K{K}'
    return dict(id=name, method=method, prefix=prefix, q=q, formula=formula,
                r=r, K=K, R=q*r if r is not None else None)


def registrations():
    old = [cell(m, p, q) for m in ('B0', 'B0I') for p in (5, 10, 15) for q in QS]
    old += [cell('B1', 19, q, f) for f in ('2nd', '4th') for q in QS]
    old += [cell('B2', p, q, r=r, K=K) for p in (5, 10, 15) for q in QS
            for r in (1, 2, 4) for K in (2, 4)]
    known = {c['id'] for c in old}
    requested = [cell(m, p, q, f) for m in ('B0', 'B0I')
                 for f, ps in (('2nd', range(6, 15)), ('4th', range(11, 16)))
                 for p in ps for q in QS]
    return dict(existing=old, additional=[c for c in requested if c['id'] not in known],
                reused_requested=[c for c in requested if c['id'] in known],
                diagnostics=[dict(prefix=p, q=q, T=.8, kind='EXACT_TAIL_OUTER_S2_ONLY',
                                  formal_candidate=False) for p in (3, 4) for q in QS])


def validation_axes():
    # Two complete axis pairs, four wrapper compiles TOTAL. S2 uses saved anchors.
    return [dict(cell=cell(m, p, q, '4th'), axis=a)
            for m, p, q in (('B0', 11, 1), ('B0I', 15, 8)) for a in AXES]


def plan():
    reg = registrations()
    bases = [c for c in reg['additional'] if c['method'] == 'B0']
    return dict(schema='track_a_v1_h6_readiness_v1', base=BASE, provider=PROVIDER,
        target=dict(system='linear_H6', geometry_angstrom=1., basis='STO-3G',
                    df_rank=19, modes=12, sector_dimension=400, T=.8),
        registrations=reg, epsilons=list(EPSILONS), allowance_factors=list(FACTORS),
        primary=dict(metric='RZ', scope='measured_hadamard_state_preparation_excluded',
                     P=0, control='symmetric_directional', alpha_axis=.025,
                     formula='sum_axis(shots_axis * compiled_full_wrapper_RZ_axis)',
                     PR_scope='H6_CONFIRMATION_UNION8_n32',
                     exploration72_n8_retained_separately=True),
        selection=dict(key=['G_point_exact_binary64_rational', 'cell_id'],
                       tolerance_ties=False, eligibility_monotonicity_assumed=False,
                       scopes=['old_grid_S2', 'fine_grid_S2', 'fine_grid_S2_plus_truncated_S4'],
                       PR_diagnostics_eligible_for_selection=False),
        scientific_report_triggers=dict(intermediate_precision_advantage_change_percentage_points=10,
            endpoint_advantage_sign_change=True,review_is_not_automatic_execution_authorization=True),
        conservative=dict(status='NO_CERTIFIED_BIAS_BOUND_IN_SAVED_CONTRACT',
            auxiliary='sum(abs(axis_signed_discard), abs(axis_signed_PF)); empirical u separately',
            physical_or_operator_error_certificate=False,
            missing_decomposition='UNRESOLVED_NOT_ZERO',
            PR_no_cancellation_envelope='outer_PF + finite_RTE, if saved'),
        numerical=dict(kind='EMPIRICAL', certified=False, safety_factor=10., floor=1e-12,
                       roundoff_multiplier=64., maximum_u_over_headroom=.01,
                       sensitivity_factors=list(FACTORS)),
        cost_model=dict(type='affine per stratum',
            stratum=['method', 'formula', 'axis', 'control', 'compiler_hash', 'profile_hash', 'phase_policy'],
            features=['1', 'q', 'prefix*q'], validation_axes=validation_axes(),
            relative_error='abs(predicted-measured)/measured per axis; no averaging',
            stop_when='any relative_error >= 0.02, missing anchor, identity change, nonpositive estimate',
            S2_fit='saved measurements only; at least three noncollinear anchors per stratum',
            S4_proposal='separate frozen schedule-composition hypothesis; not an S2 fit',
            validation_selection='after signal screen: B0 lexicographic minimum eligible S4 (p,q); B0I maximum; both axes',
            cross_method_or_axis_transfer='requires saved structural/phase and paired-axis receipts',
            refit_after_validation=False, estimated_costs_are_measurements=False),
        demand=dict(new_B0_signals=len(bases), analytic_B0I_overlays=len(bases),
                    exact_truncated_reference_pairs=len({c['prefix'] for c in bases}),
                    outer_S2_diagnostics=8, full_reference_pairs=1,
                    snapshot_load=1, frozen_preparation_decode=1,
                    reference_matvec=400, tail_scalar=9,
                    native_compile_max=4, native_evolution_max=2,
                    measured_wrappers_max=4, control_probes_max=8,
                    SCF=0, DF_decomposition=0, state_solver=0, trajectory_sampling=0),
        resources=dict(processes=1, threads=1, AS_bytes=32*2**30, RSS_bytes=9*2**30,
            CPU_seconds=8*3600, wall_seconds=12*3600, output_bytes=512*2**20,
            primitive_probe_cap=20000, deterministic_actions_per_cell=100000,
            stage_storage_bytes=2*2**30, estimate_kind='UNAPPROVED_STOP_CEILINGS_NOT_PREDICTIONS'),
        approved=False, runtime_authorization=False, allowed_cpus=[],
        absolute_launch_command=None, retry=False, resume=False,
        mandatory_stop=True, next_stage_authorized=False)


def signal_envelopes(record):
    """Triangle inequality for saved signed paths, conditional on empirical u.

    This does not manufacture an error certificate for numerically evaluated
    paths. Return None when the signed decomposition is absent or does not close.
    """
    total = record['error_decomposition']['total_signed']
    usual = {a: abs(total[a]) for a in ('real', 'imag')}
    parts = record['error_decomposition'].get('signed_parts')
    if not parts:
        return usual, None
    if set(parts) not in ({'discard', 'PF'}, {'outer_PF', 'finite_RTE'}, {'PF'}):
        raise ValueError('UNKNOWN_SIGNED_DECOMPOSITION')
    for a in usual:
        if any(not math.isfinite(x[a]) for x in parts.values()):
            raise ValueError('NONFINITE_SIGNED_PART')
        if not math.isclose(sum(x[a] for x in parts.values()), total[a], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('SIGNED_PARTS_DO_NOT_CLOSE')
    return usual, {a: sum(abs(x[a]) for x in parts.values()) for a in usual}


def phase_a_plan():
    """Parallel implementation proposal; no production execution authorized."""
    original=plan()
    caps=dict(original['demand'])
    for key in ('native_compile_max','native_evolution_max','measured_wrappers_max','control_probes_max'):
        caps[key]=0
    caps['primitive']=20000
    return dict(schema='track_a_v1_phase_a_parallel_plan_v1',scope='signal',
        science_contract_sha256=digest(original),registrations=registrations(),
        epsilons=list(EPSILONS),allowance_factors=list(FACTORS),call_caps=caps,
        parallel=dict(max_workers=16,primitive_tasks=16,signal_prefix_tasks=9,PR_diagnostic_tasks=2,
            snapshot_loads=1,archive_decodes=1,shared_reference=True,
            prefix_exact_reference_once=True,blas_threads_per_process=1),
        resources=dict(processes=18,max_workers=16,threads=1,AS_bytes=32*2**30,RSS_bytes=9*2**30,
            observer_AS_bytes=512*2**20,observer_RSS_bytes=512*2**20,
            aggregate_RSS_bytes=160*2**30,CPU_seconds=8*3600,worker_CPU_seconds=8*3600,
            aggregate_CPU_seconds=96*3600,wall_seconds=12*3600,output_bytes=2*2**30,
            stage_storage_bytes=4*2**30,estimate_kind='UNAPPROVED_STOP_CEILINGS_NOT_SPEEDUP_MEASUREMENTS'),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        one_shot=True,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False)


def affine_fit(rows, *, stratum):
    """Exact rational least squares of SAVED compiled RZ metadata only."""
    if len(rows) < 3 or any(r['stratum'] != stratum or r['status'] != 'MEASURED_COMPLETE'
                           or type(r['RZ']) is not int or r['RZ'] <= 0 for r in rows):
        raise ValueError('COMPLETE_MATCHED_ANCHORS_REQUIRED')
    X = [[Fraction(1), Fraction(r['q']), Fraction(r['prefix']*r['q'])] for r in rows]
    a = [[sum(x[i]*x[j] for x in X) for j in range(3)] +
         [sum(x[i]*r['RZ'] for x, r in zip(X, rows))] for i in range(3)]
    for i in range(3):
        pivot = next((j for j in range(i, 3) if a[j][i]), None)
        if pivot is None:
            raise ValueError('UNDERDETERMINED_AFFINE_MODEL')
        a[i], a[pivot] = a[pivot], a[i]
        v = a[i][i]; a[i] = [e/v for e in a[i]]
        for j in range(3):
            if j != i:
                v = a[j][i]; a[j] = [e-v*f for e, f in zip(a[j], a[i])]
    coefficients = [a[i][3] for i in range(3)]
    residuals = [abs(sum(c*x for c, x in zip(coefficients, features))-r['RZ'])/r['RZ']
                 for features, r in zip(X, rows)]
    if max(residuals) >= Fraction(1, 50):
        raise ValueError('SAVED_ANCHOR_RESIDUAL_GE_2_PERCENT')
    return dict(stratum=deepcopy(stratum), coefficients=coefficients,
                maximum_anchor_relative_error=max(residuals), status='FROZEN_UNVALIDATED_MODEL',
                anchor_ids=[r['id'] for r in rows])


def predicted(model, candidate, *, stratum):
    if model['stratum'] != stratum:
        raise ValueError('MODEL_STRATUM_CHANGED')
    result = sum(c*x for c, x in zip(model['coefficients'],
                 (1, candidate['q'], candidate['prefix']*candidate['q'])))
    if result <= 0:
        raise ValueError('NONPOSITIVE_MODEL_ESTIMATE')
    return result


def validate_predictions(rows, *, expected_axes, frozen_model_sha256):
    """Every preregistered axis is required; EXACTLY 2% stops as well."""
    expected = {(r['cell']['id'], r['axis']) for r in expected_axes}
    actual = [(r['cell_id'], r['axis']) for r in rows]
    if len(actual) > 4 or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('INCOMPLETE_OR_EXCESS_VALIDATION_AXES')
    for r in rows:
        if (r['status'] != 'MEASURED_COMPLETE' or r['model_sha256'] != frozen_model_sha256
                or r.get('identity_verified') is not True or r['measured'] <= 0):
            raise ValueError('VALIDATION_BINDING_OR_MEASUREMENT')
        if abs(Fraction(r['predicted'])-Fraction(r['measured']))/Fraction(r['measured']) >= Fraction(1, 50):
            raise ValueError('COMPILE_MODEL_DISAGREEMENT_GE_2_PERCENT')
    return dict(status='VALIDATED_ON_REGISTERED_AXES', population_bound=False,
                extrapolation_guaranteed=False, compiler_or_science_change_authorized=False)


def select_validation_axes(eligible_ids):
    """A fixed signal-only rule, before seeing any new compiled cost.

    A family with no eligible S4 candidate requires no S4 model or compile. The
    default anchors in validation_axes() are the two full-grid endpoints.
    """
    allowed = {x['id'] for x in registrations()['additional']}
    if not set(eligible_ids) <= allowed: raise ValueError('UNREGISTERED_SCREEN_CANDIDATE')
    selected = []
    for method in ('B0', 'B0I'):
        candidates = [x for x in registrations()['additional'] if x['method']==method
                      and x['formula']=='4th' and x['id'] in eligible_ids]
        if candidates:
            key = lambda x: (x['prefix'], x['q'])
            anchor = (min if method=='B0' else max)(candidates, key=key)
            selected.extend(dict(cell=anchor, axis=axis) for axis in AXES)
    return selected


def best(rows):
    """No tolerance-based winner switching; missing eligible costs stop a map."""
    candidates = []
    for row in rows:
        if row['eligible'] is None:
            raise ValueError('UNRESOLVED_ELIGIBILITY')
        if row['eligible']:
            if row['G'] is None or row['cost_status'] not in ('MEASURED_COMPLETE', 'VALIDATED_MODEL_ESTIMATE'):
                raise ValueError('ELIGIBLE_COST_MISSING_OR_UNVALIDATED')
            candidates.append(row)
    return min(candidates, key=lambda x: (Fraction(x['G']), x['cell_id'])) if candidates else None


def complete_reuse(receipt, *, expected):
    if (receipt.get('status') != 'COMPLETE_MANDATORY_STOP' or receipt.get('exit_code') != 0
            or receipt.get('all_reaped') is not True or receipt.get('first_stop') is not None
            or receipt.get('identity') != expected):
        raise ValueError('COMPLETE_ONLY_REUSE_IDENTITY')
    return True
