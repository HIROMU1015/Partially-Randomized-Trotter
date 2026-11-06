"""Fixed LD2 relocation of saved general-J CDF and signed-square XDF.

Imports are passive. Numerical entries accept explicitly supplied arrays only;
the preparation CLI never calls them on saved molecular inputs.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
import math

VERSION = 'saved_CDF_XDF_strong_D_tail_Z_LD2_v1'


@dataclass(frozen=True)
class RelocatedD:
    kind: str
    before: object
    model: object
    one: object
    constant: float
    delta_one: object
    scalar_shift: float
    basis_policy: str
    cdf: object = None
    lambdas: object = None
    matrices: object = None


def _head(one, policy):
    from . import c1_stage2_native as native
    from .c1_h4_strong_baseline import spin_adapted_eigh, SPIN
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives
    u, eps = spin_adapted_eigh(one)[:2] if policy == SPIN else diag_hermitian(one, assume_hermitian=True)
    return native.NativeBlock('one_body', native._rotation(u), one_body_diagonal_primitives(eps, 1.))


def prepare_cdf(kappa, constant, J, U, *, source_identity, input_identity):
    """Reuse number-CDF relocation; retain B2's legacy head basis policy."""
    import numpy as np
    from . import c_h4_relocated_downstream as relocation, c1_stage2_native as native
    from .c1_h4_strong_baseline import LEGACY
    if np.shape(kappa) != (4, 4) or np.shape(J) != (8, 4, 4):
        raise ValueError('FIXED_FOUR_SPATIAL_EIGHT_FRAMES')
    r = relocation.prepare(kappa, constant, J, U, role='baseline',
        source_identity=source_identity, input_identity=input_identity, ld=2)
    before = native.prepare_cdf(r.original.general)
    one, delta = np.kron(r.kappa, np.eye(2)), np.kron(r.delta_kappa, np.eye(2))
    blocks = (_head(one + delta, LEGACY),) + r.arm.model.blocks[1:]
    model = replace(before, blocks=blocks, constant=r.arm.model.constant,
        source_identity=source_identity + ':' + VERSION)
    one.setflags(write=False); delta.setflags(write=False)
    return RelocatedD('CDF', before, model, one, float(constant), delta,
        r.scalar_shift, LEGACY, cdf=r)


def prepare_xdf(one, lambdas, matrices, constant, *, kappa, source_identity, input_identity):
    """H_t=lambda_t*(sum eta_a n_a)^2, NOT half of that square.

    With exact interleaved spin lift, spatial J_ij=2*lambda*eta_i*eta_j.
    Thus delta_kappa_t=2*lambda*tr(G_sp)*G_sp and
    delta_c_t=-2*lambda*tr(G_sp)^2. Only original F2..F7 move.
    The square builder's original I/ZZ primitives and rotations are retained.
    """
    import numpy as np
    from .c1_h4_strong_baseline import prepare_xdf_basis, SPIN
    from .c1_stage3_molecular import component_check
    one, lam, gs = (np.array(x, copy=True) for x in (one, lambdas, matrices))
    if (one.shape != (8, 8) or lam.shape != (8,) or gs.shape != (8, 8, 8)
            or np.shape(kappa) != (4, 4) or not math.isfinite(float(constant))
            or np.iscomplexobj(lam) or not all(np.isfinite(x).all() for x in (one, lam, gs))):
        raise ValueError('FINITE_SAVED_XDF8_SCHEMA')
    component_check(one, np.kron(kappa, np.eye(2)), 1e-10, label='saved XDF original-g kappa')
    for a in (one, *gs):
        if not np.array_equal(a, np.kron(a[::2, ::2], np.eye(2))):
            raise ValueError('EXACT_INTERLEAVED_SPIN_LIFT_NO_PROJECTION')
        component_check(a, a.conj().T, 1e-10, label='saved XDF Hermitian')
    before, _ = prepare_xdf_basis(one, lam, gs, float(constant),
        source_identity=source_identity, input_identity=input_identity, basis_policy=SPIN)
    spatial = gs[:, ::2, ::2]
    traces = np.trace(spatial, axis1=1, axis2=2)
    if np.any(abs(traces.imag) > 1e-12):
        raise ValueError('REAL_HERMITIAN_TRACE_REQUIRED')
    traces = traces.real
    delta_spatial = sum((2*lam[t]*traces[t]*spatial[t] for t in range(2, 8)), np.zeros((4, 4), complex))
    shift = -math.fsum(float(2*lam[t]*traces[t]**2) for t in range(2, 8))
    delta = np.kron(delta_spatial, np.eye(2))
    blocks = [_head(one + delta, SPIN), *before.blocks[1:]]
    for t in range(2, 8):
        block = blocks[t+1]
        blocks[t+1] = replace(block, diagonal=replace(block.diagonal, rz=()))
    model = replace(before, blocks=tuple(blocks), constant=float(constant)+shift,
        source_identity=source_identity + ':' + VERSION)
    for a in (one, lam, gs, delta): a.setflags(write=False)
    return RelocatedD('XDF', before, model, one, float(constant), delta, shift,
        SPIN, lambdas=lam, matrices=gs)


def generators(prepared, car):
    """Independent full-Fock CAR; never derive reference from native gates."""
    import numpy as np
    from . import c_h4_relocated_downstream as relocation
    from .c1_stage3_molecular import component_check
    p = prepared
    if p.model.constant != p.constant+p.scalar_shift:
        raise ValueError('NUCLEAR_AND_RELOCATION_SCALAR_ONCE')
    if p.kind == 'CDF':
        old, new = relocation.raw_generators(p.cdf, car=car)
    elif p.kind == 'XDF':
        def one(a): return np.einsum('pq,pqab->ab', a, car)
        eye = np.eye(car.shape[-1])
        old = [one(p.one) + p.constant*eye]
        for lam, g in zip(p.lambdas, p.matrices, strict=True):
            a = one(g); old.append(float(lam)*(a @ a))
        new = [one(p.one + p.delta_one) + p.model.constant*eye, *old[1:]]
        for t in range(2, 8):
            spatial = p.matrices[t, ::2, ::2]
            delta = 2*float(p.lambdas[t])*float(np.trace(spatial).real)*p.matrices[t]
            shift = -2*float(p.lambdas[t])*float(np.trace(spatial).real)**2
            new[t+1] = old[t+1] - (one(delta) + shift*eye)
    else:
        raise ValueError('UNKNOWN_REPRESENTATION')
    if [b.block_id for b in p.model.blocks] != ['one_body'] + [f'F{i}' for i in range(8)]:
        raise ValueError('ORIGINAL_FRAME_ORDER')
    component_check(sum(old), sum(new), 1e-10, label='relocation full H including scalar')
    for t, (a, b) in enumerate(zip(p.before.blocks[1:], p.model.blocks[1:], strict=True)):
        if (a.rotation != b.rotation or a.diagonal.rzz != b.diagonal.rzz
                or a.diagonal.global_phase != b.diagonal.global_phase
                or (t < 2 and a.diagonal != b.diagonal) or (t >= 2 and b.diagonal.rz)):
            raise ValueError('FRAME_IDENTITY_ZZ_PREFIX_OR_TAIL_Z')
    return old, new


def evaluate(prepared, saved, shells, out, call):
    """Shared CDF/XDF D evaluator. Only future permission or invented tests.

    saved binds the original candidate, own-ground and ascending-Qiskit sector.
    Neither old PF data nor old native metrics are accepted as new results.
    """
    import numpy as np
    from scipy.linalg import eigh
    from concurrent.futures import ThreadPoolExecutor
    from qiskit import qpy
    from . import c1_stage1 as stage, c1_stage1_execution as io
    from . import c1_h4_native_fair as fair, c1_h4_full_synthesis as synthesis
    from . import c1_molecule_extension as molecule, c_pr_aware_df_m2 as m2
    from . import c_pr_aware_df_stage_b2 as b2, c1_stage3_reference as reference
    from . import c_same_ju_strong_d as same, c1_h6_ca_resources as costs
    from .c1_h4_strong_baseline import StrongBuilder
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    from .c_pr_aware_df_stage_b_runner import save_arrays
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    for name in ('circuits', 'operators'): (out/name).mkdir()
    model = prepared.model
    if (saved['candidate_identity'] != prepared.before.source_identity
            or saved['input_identity'] != model.input_identity
            or not saved['DF_budget_PASS'] or not math.isfinite(saved['DF_error_with_guard'])
            or not 0 <= saved['DF_error_with_guard'] <= .00019875):
        raise ValueError('CANDIDATE_REFERENCE_DF_BINDING')
    car = call('car_context', model.label, lambda: stage.car_bilinears(8))
    old, G = call('raw_generators', model.label, lambda: generators(prepared, car))
    ix = stage.sector_indices(4, 2, 2)
    outside = np.setdiff1d(np.arange(256), ix)
    def binding():
        fair.component_check(sum(old)[np.ix_(ix, ix)], saved['H'], 1e-10, label='saved candidate H')
        if saved.get('generators') is not None:
            for a, b in zip(old, saved['generators'], strict=True):
                fair.component_check(a[np.ix_(ix, ix)], b, 1e-10, label='saved original factor')
        for g in G:
            fair.component_check(g[np.ix_(outside, ix)], np.zeros((len(outside), 36)), 1e-10, label='factor leakage')
        return dict(candidate=model.source_identity, original_candidate=saved['candidate_identity'],
            input_identity=model.input_identity, saved_ground_reused=True, DF_recomputed=False, status='PASS')
    io.save(out/'identity.json', call('reference_binding', model.label, binding))
    sector = [g[np.ix_(ix, ix)] for g in G]
    pair = call('sector_eigenpair', 'reference spectral factorization, not a new ground oracle', lambda: eigh(saved['H']))
    def ground_gate():
        psi, energy = saved['sector_state'], float(saved['energy'])
        if (psi.shape != (36,) or not np.isfinite(psi).all() or not math.isfinite(energy)
                or abs(np.vdot(psi, psi)-1) > 1e-12
                or np.linalg.norm(saved['H'] @ psi-energy*psi) > 1e-10
                or abs(float(pair[0][0])-energy) > 1e-10 or not -10 < energy < 10):
            raise ValueError('SAVED_OWN_GROUND_ALIAS')
        return dict(status='PASS', ground_origin=saved['ground_origin'], energy_Ha=energy,
            alias_window_Ha=[-10, 10], reference_factorization_reused_for_13_rounds=True)
    io.save(out/'ground.json', call('ground_gate', model.label, ground_gate))
    kernel = {}
    def pf_call(kind, label, fn):
        value = call('PF_kernel' if kind == 'sector_kernel_reference' else 'PF_round', label, fn)
        if kind == 'sector_kernel_reference': kernel['sector'] = value[0]['pf']
        return value
    pf = molecule.PF_rounds(sector, 1, pf_call, model.source_identity, eigenpair=pair)
    io.save(out/'PF.json', dict(candidate=model.source_identity, rounds=pf))
    packet = dict(id=prepared.kind+'_relocated', arm=prepared.kind, aliases=[prepared.kind+'_relocated'],
        s=1, state_model='own_DF_ground_state_oracle',
        alias_verified=True, df_error_with_guard=saved['DF_error_with_guard'])
    ideal = same.ideal_plans(pf, packet, call)
    if not ideal['energy']['same_accuracy']:
        return dict(status='STOP_PF_RPE_UNQUALIFIED', candidate=model.source_identity,
            ideal_exact_native=ideal, numerical=None, metrics=None, actual_CA_achieved=None)
    ref = call('full_reference', model.label,
        lambda: reference.reference_kernels(G, Fraction(1, 10), ld=8)['pf'])
    io.save(out/'reference_sector.json', call('reference_sector_gate', model.label,
        lambda: fair.component_check(ref[np.ix_(ix, ix)], kernel['sector'], 1e-10, label='full/sector S2')))
    save_arrays(out/'operators/reference.npz', matrix=ref)
    sh_mats, sh_reports, sh_metrics = {}, {}, {}
    for label, qs in shells.items():
        sh_mats[label], structure = call('shell_structure', label,
            lambda: b2.shell_assessment(qs['raw'], qs['compiled'], label))
        h = np.array([[1., 1.], [1., -1.]])/math.sqrt(2)
        target = np.asarray(np.kron(h @ np.diag([1., -1j]) if label == 'Y' else h, np.eye(256)), dtype=complex)
        sh_reports[label] = call('shell_segment', label,
            lambda: synthesis.segment(sh_mats[label]['raw'], sh_mats[label]['compiled'], target, label))
        sh_reports[label]['structure'] = structure
        shell_metrics = circuit_metrics(qs['compiled'])
        sh_metrics[label] = {k: shell_metrics[k] for k in m2.accuracy.METRICS}
    io.save(out/'shells.json', sh_reports)
    builder = StrongBuilder(relative_fusion=True, basis_policy=prepared.basis_policy, call=call)
    for block in model.blocks:
        call('gaussian_build', block.block_id, lambda block=block: fair.basis_request(builder, model, block))
    raw = call('kernel_build', model.label, lambda: builder.ordered(model, Fraction(1, 10), optimized=True))
    io.save(out/'fusion.json', call('fusion_gate', model.label, lambda: m2.fusion_gate(builder, model)))
    compiled = call('compile', model.label, lambda: fair.compile_checked(raw, CompilerProfile()))
    qs = dict(raw=raw, compiled=compiled)
    for kind, q in list(qs.items()):
        qs[kind+'_inverse'] = call('adjoint_build', kind, lambda q=q: b2.b.native.actual_adjoint(q))
    for kind, q in qs.items():
        with (out/'circuits'/(kind+'.qpy')).open('xb') as stream: qpy.dump(q, stream)
    plan = m2.integrated.resource_plan(task_count=4, full_dimension=256)
    io.save(out/'resource_plan.json', plan)
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures = {k: pool.submit(call, 'operator_decode', k, lambda q=q: fair.decode(q)) for k, q in qs.items()}
        mats = {k: f.result() for k, f in futures.items()}
    for k, a in mats.items(): save_arrays(out/'operators'/(k+'.npz'), matrix=a)
    target = fair.controlled(ref)
    segment = call('segment_assessment', model.label,
        lambda: synthesis.segment(mats['raw'], mats['compiled'], target, model.label))
    io.save(out/'segment.json', segment)
    adjoints = {}
    for kind in ('raw', 'compiled'):
        def inverse_gate(kind=kind):
            inv = mats[kind+'_inverse']
            return dict(dagger=fair.component_check(inv, mats[kind].conj().T, 1e-10, label='actual inverse'),
                reference=fair.component_check(inv, target.conj().T, 1e-10, label='inverse/reference'),
                return_gate=fair.component_check(inv @ mats[kind], np.eye(512), 1e-10, label='inverse return'))
        adjoints[kind] = call('adjoint_gate', kind, inverse_gate)
    io.save(out/'adjoint.json', adjoints)
    psi = np.zeros(256, complex); psi[ix] = saved['sector_state']
    io.save(out/'XY.json', call('XY_gate', model.label,
        lambda: b2.XY_gate(mats['raw'], mats['compiled'], ref, psi, sh_mats)))
    metrics = call('metrics', model.label, lambda: circuit_metrics(compiled))
    packet.update(kernel_metrics={k: metrics[k] for k in m2.accuracy.METRICS}, shell_metrics=sh_metrics)
    numerical = m2.evaluate_accuracy(packet, pf, sh_reports, segment, out, call)
    ideal_costs = []
    for row in ideal['rounds']:
        ideal_costs.extend(call('ideal_serial_cost', str(row['m']), lambda row=row: costs.serial_round_cost(packet, row['m'])))
    ideal['cost'] = call('ideal_aggregate', model.label, lambda: m2.accuracy.aggregate_totals(
        ideal['energy'], ideal['rounds'], ideal_costs,
        cost_scope='serial_native_kernel_composition_without_state_preparation'))
    result = dict(status='RELOCATED_STRONG_D_COMPLETE' if numerical['cost']['totals'] is not None
        else 'STOP_NUMERICAL_LANE_UNQUALIFIED', candidate=model.source_identity,
        ideal_exact_native=ideal, numerical=numerical, metrics=metrics,
        PF_rounds=pf, actual_CA_achieved=None, PR_whole_law_qualification=None,
        implemented_PR_law_error=None, guaranteed_resource_superiority=None, mandatory_stop=True)
    io.save(out/'result.json', result)
    return result
