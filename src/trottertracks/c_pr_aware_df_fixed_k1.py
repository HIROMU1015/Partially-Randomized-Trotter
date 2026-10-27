"""Opt-in exact-Hermitian-kappa/polar-U target. Historical F2 is unchanged.

Only B0 is exponentiated. V1/V2 are defined by exact coefficient identities
and bounded analytically. No molecular work occurs on import.
"""
from fractions import Fraction as F
import math

from . import c_pr_aware_df_fixed_car as car
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_kappa_diagnostic as diagnostic
from . import c_pr_aware_df_zsxx_enclosure as z

VERSION = 'exact_kappa_H_polar_U_CAR_bridge_v1'
TARGET = 'CAR_exact_kappa_H_polar_U_fixed_trajectory_v1'


def exact_kappa(kappa):
    """Exact rational H/A before any Arb rounding, never a float projection."""
    import numpy as np
    a = np.asarray(kappa)
    if (a.ndim != 2 or not len(a) or a.shape != (len(a), len(a))
            or a.dtype not in (np.dtype('float64'), np.dtype('complex128'))
            or not np.isfinite(a).all()):
        raise ValueError('K1_KAPPA_FINITE_SHAPE_DTYPE')
    n = len(a)
    literal = [[diagnostic.q(x) for x in row] for row in a]
    H, A = [], []
    for p in range(n):
        h, anti = [], []
        for q in range(n):
            adj = literal[q][p][0], -literal[q][p][1]
            h.append(diagnostic.mul(diagnostic.add(literal[p][q], adj), F(1, 2)))
            anti.append(diagnostic.mul(diagnostic.sub(literal[p][q], adj), F(1, 2)))
        H.append(h); A.append(anti)
    for p in range(n):
        for q in range(n):
            if (diagnostic.add(H[p][q], A[p][q]) != literal[p][q]
                    or H[p][q] != (H[q][p][0], -H[q][p][1])
                    or A[p][q] != (-A[q][p][0], A[q][p][1])):
                raise ValueError('EXACT_KAPPA_DECOMPOSITION')
    return dict(norb=n, literal=literal, H=H, A=A,
                anti_spectral_upper=sum(abs(x)+abs(y) for row in A for x,y in row))


def ball_coefficients(exact):
    """Outward Arb coefficients; rational subnormal halves stay nonzero."""
    from flint import acb
    return [[acb(z._real(x), z._real(y)) for x, y in row] for row in exact]


def head(ctx, parts, constant=0, *, component='H'):
    """Same SectorCAR.one algebra for literal/H/A; no new CAR implementation."""
    from .c1_molecule_extension import SectorCAR
    arithmetic = car.Arithmetic()
    matrix = SectorCAR.one(ctx, ball_coefficients(parts[component]), arithmetic=arithmetic)
    if constant:
        matrix = matrix + arithmetic.number(constant)*arithmetic.eye(ctx.d)
    return matrix


def input_gate(arm, kappa, constant, events, h, diagnostic_record=None):
    import numpy as np
    parts = exact_kappa(kappa); n = parts['norb']
    if (arm.model.modes != 2*n or arm.plan.ld != 2 or F(h) <= 0
            or not np.isrealobj(constant) or not np.isfinite(constant)
            or not math.isfinite(arm.plan.extracted_identity)
            or arm.J.shape != arm.U.shape or arm.J.shape[1:] != (n,n)
            or len(arm.J) <= arm.plan.ld or not np.isfinite(arm.U).all()):
        raise ValueError('K1_FIXED_GENERATOR_SCHEMA')
    if arm.J.dtype != np.dtype('float64'):
        raise ValueError('K1_REAL_J_REQUIRED')
    for i, J in enumerate(arm.J):
        car._hermitian_input(J)
        V = np.kron(arm.U[i], np.eye(2))
        if not np.array_equal(V, np.asarray(arm.general.frames[i].orbital_rotation)):
            raise ValueError('K1_SPIN_FRAME_SOURCE')
    atoms = {a.atom_id:a for a in arm.plan.atoms}
    for event in events:
        atom = atoms[event.rotation_component_id]
        if (event.taylor_order != 0 or event.phase != 1 or event.product_component_ids
                or not math.isfinite(event.rotation_angle)
                or len(atom.support) not in (1,2) or len(set(atom.support)) != len(atom.support)
                or any(type(i) is not int or not 0 <= i < 2*n for i in atom.support)
                or not atom.coefficient):
            raise ValueError('K1_SIGNED_K0_EVENT')
        frame = next(f.original_index for f in arm.general.frames
            if arm.model.label+':'+f.frame_id == atom.frame_id)
        if not np.array_equal(np.asarray(arm.plan.frames[atom.frame_id]),
                              np.kron(arm.U[frame],np.eye(2))):
            raise ValueError('K1_ATOM_BASIS_SOURCE')
    if diagnostic_record is not None:
        m = diagnostic_record['matrices']
        for key, part in [('kappa_literal','literal'),('kappa_H_exact_diagnostic_only','H'),
                          ('kappa_A_exact_diagnostic_only','A')]:
            if diagnostic.exact_record(parts[part])['entries'] != m[key]['entries']:
                raise ValueError('K1_SAVED_KAPPA_DIAGNOSTIC_IDENTITY')
    return parts


def telescoping(errors, left_norms, right_norms):
    """Application order: later factors multiply on the left. Arb arithmetic."""
    from flint import arb
    if not len(errors) == len(left_norms) == len(right_norms):
        raise ValueError('FACTOR_BOUND_LENGTH')
    total = arb(0)
    for j, error in enumerate(errors):
        term = error
        for bound in left_norms[j+1:]: term *= bound
        for bound in right_norms[:j]: term *= bound
        total += term
    return total


def kappa_bridge(parts, h, U_bridge):
    """V0->V1: two kappa half-heads and an unchanged middle M.

    ||M|| <= L=1+epsilon_U: the old product bound includes every tail-local
    norm excess and nonnegative deterministic-frame error. It is NOT inferred
    from the norm of the complete V1. Real symmetric J makes both literal
    density-frame generators Hermitian even for nonorthogonal U.
    """
    if U_bridge is None or U_bridge.get('upper_exact') is None:
        return dict(status='UNRESOLVED_FACTOR_NORMS', upper_exact=None)
    from flint import arb
    a = parts['anti_spectral_upper']; mu = 2*parts['norb']*a
    L = 1+z._real(F(U_bridge['upper_exact']))
    b = (z._real(abs(F(h))/2)*z._real(mu)).exp()
    local = b-1
    bound = telescoping([local,arb(0),local], [b,L,b], [arb(1),L,arb(1)])
    return dict(status='VALIDATED_ANALYTIC_KAPPA_BRIDGE', from_map='V0=CAR(kappa,U)',
        to_map='V1=CAR(exact_kappa_H,U)', upper_exact=str(z._fraction(bound.upper())),
        spatial_anti_spectral_upper_exact=str(a),full_Fock_particle_upper=2*parts['norb'],
        lifted_anti_norm_upper_exact=str(mu), half_head_time_exact=str(F(h)/2),
        half_head_occurrences=2,half_head_norm_upper_exact=str(z._fraction(b.upper())),
        middle_norm_upper_exact=str(1+F(U_bridge['upper_exact'])),
        literal_trajectory_norm_upper_exact=str(z._fraction((b*L*b).upper())),
        formula='L*(exp(abs(h)*mu)-1), evaluated by two-factor telescoping',
        original_kappa_changed=False, rounded_H_used=False)


def build_reference(arm, kappa, constant, events, h=F(1,40), *, parallel=True,
                    diagnostic_record=None, progress=None):
    import numpy as np
    from flint import ctx, acb_mat
    from .c1_stage1 import car_bilinears
    parts = input_gate(arm,kappa,constant,events,h,diagnostic_record)
    if progress:
        progress('K1_input_gate',dict(status='PASS',H=diagnostic.exact_record(parts['H']),
                                    A=diagnostic.exact_record(parts['A'])))
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        U = car.polar_bridge(arm,h,events)  # premises checked above for H, J, atoms
        U = dict(U, target=TARGET,from_map='V1=CAR(exact_kappa_H,U)',
                 to_map='V2=CAR(exact_kappa_H,right_polar_U)')
        K = kappa_bridge(parts,h,U)
        H_ball = acb_mat(ball_coefficients(parts['H']))
        A_ball = acb_mat(ball_coefficients(parts['A']))
        coefficient_record = dict(H=diagnostic.exact_record(parts['H']),
            A=diagnostic.exact_record(parts['A']), H_ball=car.pack(H_ball), A_ball=car.pack(A_ball))
    if progress: progress('K1_models',dict(kappa_bridge=K,U_bridge=U,kappa=coefficient_record))
    full = car_bilinears(arm.model.modes)
    groups = car.full_partition(arm.model.modes)
    tasks = [(arm,kappa,constant,events,h,idx,
              {(p,q):full[p,q][np.ix_(idx,idx)]
               for p in range(arm.model.modes) for q in range(arm.model.modes) if p%2==q%2})
             for idx in groups]
    plan = fixed.resource_plan(len(groups))
    if parallel and arm.model.modes >= 6 and plan['workers'] > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],
                mp_context=multiprocessing.get_context('spawn')) as pool:
            blocks = list(pool.map(car._block_task,tasks))
    else:
        blocks = [car._block_task(t) for t in tasks]
    return dict(status='VALIDATED_LITERAL_B0_WITH_K1_BRIDGES',target=TARGET,
        system_modes=arm.model.modes,system_dimension=1<<arm.model.modes,blocks=blocks,
        kappa=coefficient_record,kappa_bridge=K,U_bridge=U,resource_plan=plan,
        reference_independent_of_QPY=True,attenuation_applied=False,
        evaluated_map='V0=CAR(original_kappa,original_U)',
        final_map='V2=CAR(exact_kappa_H,right_polar_U)',
        physical_Hamiltonian_bound=None,whole_law_bound=None)


def compare(record, reference, *, raw_identity, parallel=True):
    """Old all-column backend, new target/bounds; never clamp raw/B0 to two."""
    from flint import arb,ctx
    fixed._validate(record)
    d = 2*reference['system_dimension']
    if (reference['target'] != TARGET or reference['status']!='VALIDATED_LITERAL_B0_WITH_K1_BRIDGES'
            or (1<<record['qubits']) != d):
        raise ValueError('K1_CONTROLLED_REFERENCE_TARGET')
    plan = fixed.resource_plan(d)
    if parallel and d>=128 and plan['workers']>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=car._init_compare,initargs=(record,reference)) as pool:
            rows = list(pool.map(car._compare_column,range(d)))
    else:
        car._init_compare(record,reference)
        rows = [car._compare_column(i) for i in range(d)]
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        literal = z._fraction(sum((z._real(F(x['upper_exact']))**2 for x in rows),arb(0)).sqrt().upper())
    raw_lower = max(F(x['lower_exact']) for x in rows)
    K, U = (reference[k].get('upper_exact') for k in ('kappa_bridge','U_bridge'))
    common = dict(status='VALIDATED_FIXED_RAW_IDEAL_CAR',target=TARGET,norm='full_operator_2',
        dimension=d,columns_evaluated=d,phase_fit=False,precision_bits=z.PRECISION,
        columns=rows,literal_CAR_upper_exact=str(literal),literal_CAR_lower_exact=str(raw_lower),
        raw_identity_sha256=raw_identity,resource_plan=plan,
        reference_uncertainty='included once in acb B0 columns;raw radius once per column',
        physical_Hamiltonian_bound=None,whole_law_bound=None,CA_achieved=None,shots=None,total_resources=None)
    if K is None or U is None:
        return dict(common,upper_exact=None,lower_exact=None,qualification='UNRESOLVED_MODEL_BRIDGE')
    if F(K)<0 or F(U)<0:raise ValueError('NEGATIVE_MODEL_BOUND')
    bridge = F(K)+F(U)
    return dict(common,kappa_bridge_upper_exact=K,U_bridge_upper_exact=U,
        upper_exact=str(min(F(2),literal+bridge)),lower_exact=str(max(F(),raw_lower-bridge)),
        counted_terms=['raw/B0','V0/V1_kappa','V1/V2_U'],qualification='VALIDATED_K1_TRIANGLE')


def compose(prior, raw_car, *, raw_identity, expected_raw_identity):
    if raw_identity != expected_raw_identity:
        raise ValueError('TRIANGLE_RAW_IDENTITY')
    if raw_car is None or prior is None or raw_car.get('upper_exact') is None:
        return dict(status='UNRESOLVED',target=TARGET,upper_exact=None)
    if (raw_car.get('target') != TARGET or raw_car.get('raw_identity_sha256')!=raw_identity
            or raw_car.get('columns_evaluated')!=raw_car.get('dimension')
            or raw_car.get('norm')!='full_operator_2' or prior.get('norm')!='full_operator_2'
            or raw_car.get('precision_bits')!=192 or prior.get('precision_bits')!=192):
        raise ValueError('K1_TRIANGLE_TARGET_NORM_IDENTITY')
    result = car.compose(prior,raw_car,raw_identity=raw_identity,expected_raw_identity=expected_raw_identity)
    return dict(result,target=TARGET,counted_terms=['compiled/raw','raw/B0','V0/V1_kappa','V1/V2_U'])
