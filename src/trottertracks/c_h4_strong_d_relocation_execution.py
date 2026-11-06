"""Two fixed saved strong-D candidates; default entry is byte-only preflight.

execute() is for a separately permitted future scientific run. No authentication
or approval-proof artifact is created. Tests provide invented data only.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import os

SCOPE = 'H4_SAVED_CDF8_XDF8_STRONG_D_RELOCATION_LD2_V1'
FAMILY = 'artifacts/research_c_h4_strong_d_relocation/2026-11-07'
FRESH = FAMILY + '/science/run01'
TARGET = dict(energy_ha=.00159, df_reserve_fraction=.125, base_time=.1,
    beta_total=math.pi/6, beta_stat=math.pi/12, alpha_axis=.01/26, alpha_total=.01)
SYSTEM = dict(molecule='linear H4', distance_angstrom=1., basis='STO-3G',
    Nalpha=2, Nbeta=2, sector_dimension=36, system_dimension=256, controlled_dimension=512)
CORRECTNESS = dict(operator_component_abs=1e-10, signal_complex_abs=2e-10,
    normalization_abs=1e-12, rtol=0, global_phase_fit=False,
    input_columns=512, output_components=512, actual_raw_inverse=True,
    actual_compiled_inverse=True, inverse_return=True, shell_XY=True,
    qualification='level B full-Fock numerical pilot, no rounding enclosure')
POLICY = dict(candidates=['B2_CDF_relative', 'B3_XDF_spin_relative'], rank=8,
    spatial_orbitals=4, Nalpha=2, Nbeta=2, relocation_ld=2,
    moved_frames=[2, 3, 4, 5, 6, 7], frame_order=list(range(8)),
    PF='all-frame ordered symmetric S2', s=1, h='1/10', rounds=list(range(13)),
    state='own_DF_ground_state_oracle', state_preparation='EXCLUDED',
    lanes=['ideal_exact_native', 'numerical_synthesis'], price_samples_per_D=1,
    new_PR_samples=0, new_DF_fit=0, new_DF_norm=0, retry=0, resume=0)
COMPILER = dict(qiskit='1.3.0', optimization_level=1, seed_transpiler=17,
    basis_gates=['rz', 'sx', 'x', 'cx'], approximation_degree=1., qubits_initially_zero=False)
# Per candidate, including every called legacy numerical/accuracy helper.
LIMITS = dict(candidate_load=1, candidate_prepare=1, shell_load=6, car_context=1,
    raw_generators=1, reference_binding=1, sector_eigenpair=1, ground_gate=1,
    PF_kernel=1, PF_round=13, ideal_round_accuracy=13, ideal_energy=1,
    full_reference=1, reference_sector_gate=1, shell_structure=3,
    shell_segment=3, gaussian_build=9, relative_gaussian=16, kernel_build=1,
    fusion_gate=1, compile=1, adjoint_build=2, operator_decode=4,
    segment_assessment=1, adjoint_gate=2, XY_gate=1, metrics=1,
    round_synthesis=13, round_accuracy=13, energy=1, round_serial_cost=13,
    aggregate=1, ideal_serial_cost=13, ideal_aggregate=1)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''): h.update(data)
    return h.hexdigest()


def preflight(root, path):
    """Stdlib, JSON and file bytes only. Never import numeric libraries."""
    root = Path(root).resolve(); c = read(root/path)
    for key, expected in dict(scope=SCOPE, policy=POLICY, target=TARGET, compiler=COMPILER,
            system=SYSTEM, correctness=CORRECTNESS,
            fresh_output=FRESH, retry=0, resume=0, mandatory_stop=True,
            call_limits_per_candidate=LIMITS, maximum_units_per_candidate=sum(LIMITS.values())).items():
        if c[key] != expected: raise ValueError('FIXED_SCOPE:'+key)
    if (set(c['candidates']) != {'CDF', 'XDF'} or c['technical_blockers']
            or c['readiness'] != 'TECHNICALLY_READY_AWAITING_SCIENCE_PERMISSION'):
        raise ValueError('TECHNICAL_READINESS_OR_CANDIDATES')
    required = {c['original_input'], c['cdf_export'], c['cdf_metadata'], c['E2_summary'],
        c['old_contract'], c['old_STOP'], c['E2_STOP']}
    for p in c['candidates'].values():
        required.update((p['reference'], p['ground'], p['old_packet'], p['old_result']))
    required.update(p for q in c['shells'].values() for p in q.values())
    if not required <= c['saved_inputs'].keys(): raise ValueError('UNPINNED_SCIENTIFIC_INPUT')
    if sha(root/c['SOURCE']) != c['SOURCE_sha256']: raise ValueError('SOURCE_MANIFEST')
    for group in (read(root/c['SOURCE'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or sha(p) != digest: raise ValueError('SOURCE_INPUT:'+name)
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('ENVIRONMENT:'+name)
    old, stop, estop = read(root/c['old_contract']), read(root/c['old_STOP']), read(root/c['E2_STOP'])
    if old['accuracy_target'] != TARGET or old['compiler'] != COMPILER or stop['failed'] or estop['failed']:
        raise ValueError('SAVED_TARGET_COMPILER_STOP')
    e2 = read(root/c['E2_summary'])['statistics']['lanes']['ideal_exact_native']
    if (c['E2_point_RZ'] != 75621395471. or e2['totals']['rz_count'] != c['E2_point_RZ']
            or e2['samples'] != 64 or e2['qualification'] != 'CONDITIONAL_FINITE_BIT_PRICE_PROXY'):
        raise ValueError('E2_SAVED_POINT_NOT_NEW_SAMPLE')
    for arm, candidate in c['candidates'].items():
        expected = 'B2_CDF_relative' if arm == 'CDF' else 'B3_XDF_spin_relative'
        packet, result = read(root/candidate['old_packet']), read(root/candidate['old_result'])
        if (candidate['id'] != expected or candidate['identity'] != arm+':'+c['saved_inputs'][c['cdf_export'] if arm == 'CDF' else c['original_input']]+':prefix8'
                or packet['id'] != expected or packet['s'] != 1
                or packet['df_error_with_guard'] != candidate['DF_error_with_guard']
                or result['status'] != 'NUMERICAL_PILOT_CA_PASS'
                or not math.isfinite(candidate['DF_error_with_guard'])
                or not 0 <= candidate['DF_error_with_guard'] <= .00019875):
            raise ValueError('SAVED_CANDIDATE_DF:'+arm)
    if (root/FRESH).exists(): raise FileExistsError('FRESH_OUTPUT_NO_RETRY_RESUME')
    return c


def load_candidate(root, c, arm):
    """Scientific array whitelist; never called by preflight/preparation."""
    import numpy as np
    from . import c1_stage1_execution as io, c1_stage1 as stage
    root = Path(root); item = c['candidates'][arm]
    with np.load(root/c['cdf_export'], allow_pickle=False) as f:
        export = {k: f[k] for k in ('kappa', 'constant', 'diag_coulomb_mats', 'orbital_rotations')}
    metadata = read(root/c['cdf_metadata'])
    if (metadata['actual_rank'] != 8 or metadata['z_representation'] is not False
            or metadata['frame_ids'] != [f'F{i}' for i in range(8)]
            or metadata['one_body_role'] != 'ffsim_corrected_kappa'):
        raise ValueError('SAVED_CDF_NUMBER_RANK_FRAME_CONVENTION')
    for k, a in export.items():
        if io.array_record(a) != metadata['arrays'][k]: raise ValueError('CDF_EXPORT_MEMBER:'+k)
    p = io.load_input(root/c['original_input'], evaluation=True)
    if (float(export['constant']) != float(p['nuclear_repulsion'])
            or metadata['input_sha256'] != c['saved_inputs'][c['original_input']]):
        raise ValueError('NUCLEAR_SCALAR_OR_ORIGINAL_INPUT')
    from .c1_stage3_molecular import component_check
    component_check(export['kappa'], stage.corrected_kappa(p['one_mo'], p['eri_chemist']),
        1e-10, label='saved original-g kappa, not rederived for use')
    if arm == 'CDF':
        arrays = dict(kappa=export['kappa'], constant=float(export['constant']),
            J=export['diag_coulomb_mats'], U=export['orbital_rotations'])
        with np.load(root/item['reference'], allow_pickle=False) as f: H = f['matrix']
        generators = None
    elif arm == 'XDF':
        arrays = dict(one=p['df_one_body'], lambdas=p['df_lambdas'][:8],
            matrices=p['df_g_matrices'][:8], constant=float(p['nuclear_repulsion']), kappa=export['kappa'])
        with np.load(root/item['reference'], allow_pickle=False) as f:
            generators, ix = f['generators'], f['sector_indices']
        if not np.array_equal(ix, stage.sector_indices(4, 2, 2)): raise ValueError('SAVED_XDF_SECTOR_ORDER')
        H = sum(generators)
    else: raise ValueError('TWO_FIXED_CANDIDATES_ONLY')
    with np.load(root/item['ground'], allow_pickle=False) as f:
        ground = {k: f[k] for k in ('energy', 'sector_state')}
    saved = dict(candidate_identity=item['identity'], input_identity=c['saved_inputs'][c['original_input']],
        DF_budget_PASS=True, DF_error_with_guard=item['DF_error_with_guard'], H=H,
        generators=generators, ground_origin=item['ground'], **ground)
    return arrays, saved


def candidate_worker(root, c, arm, output):
    """A separate process owns its compiler/caches and candidate output."""
    from . import c_h4_strong_d_relocation as adapter
    from . import c1_stage1_execution as io, c1_stage3_integrated as integrated
    from threadpoolctl import threadpool_limits
    from qiskit import qpy
    import resource
    import time
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    call = integrated.Calls(output, LIMITS, sum(LIMITS.values()))
    start = time.monotonic(); failed = None; result = None
    try:
        with threadpool_limits(limits=1):
            arrays, saved = call('candidate_load', arm, lambda: load_candidate(root, c, arm))
            item = c['candidates'][arm]
            prepare = adapter.prepare_cdf if arm == 'CDF' else adapter.prepare_xdf
            prepared = call('candidate_prepare', arm, lambda: prepare(**arrays,
                source_identity=item['identity'], input_identity=saved['input_identity']))
            shells = {}
            for name, paths in c['shells'].items():
                shells[name] = {}
                for kind, path in paths.items():
                    def load(path=path):
                        with (Path(root)/path).open('rb') as stream: values = qpy.load(stream)
                        if len(values) != 1 or values[0].num_qubits != 9: raise ValueError('FULL_SHELL_SCHEMA')
                        return values[0]
                    shells[name][kind] = call('shell_load', name+kind, load)
            result = adapter.evaluate(prepared, saved, shells, output/'evaluation', call)
            if result['status'] != 'RELOCATED_STRONG_D_COMPLETE': failed = result['status']
        return result
    except BaseException as exc:
        failed = repr(exc); call.stop.set(); io.save(output/'failure.json', dict(error=failed)); raise
    finally:
        io.save(output/'STOP.json', dict(failed=failed is not None, reason=failed or 'COMPLETE',
            call_counts=dict(call.counts), retry=0, resume=0, mandatory_stop=True))
        io.save(output/'resources.json', dict(wall_seconds=time.monotonic()-start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def compare_points(results, e2_point):
    if set(results) != {'CDF', 'XDF'} or any(r['status'] != 'RELOCATED_STRONG_D_COMPLETE' for r in results.values()):
        return dict(status='UNQUALIFIED_NO_RANKING', E2_point_RZ=e2_point, comparisons=None)
    points = {k: r['ideal_exact_native']['cost']['totals']['rz_count'] for k, r in results.items()}
    if not all(math.isfinite(x) and x >= 0 for x in points.values()): raise ValueError('INVALID_POINT')
    return dict(status='FIXED_TWO_CANDIDATE_IDEAL_POINT_COMPARISON', E2_point_RZ=e2_point,
        D_points_RZ=points, E2_lower_than={k: e2_point < x for k, x in points.items()},
        decision='E2_POINT_BENEFIT_REMAINS' if all(e2_point < x for x in points.values())
            else 'E2_POINT_SUPERIORITY_NOT_SUPPORTED_FOR_AT_LEAST_ONE_D',
        new_CI=None, new_PR_samples=0, actual_CA_achieved=None, guaranteed_resource_superiority=None)


def execute(root, path):
    """One fresh scientific execution. Requires separate explicit permission."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from . import c1_stage1_execution as io, c1_stage3_integrated as integrated
    root = Path(root).resolve(); c = preflight(root, path)
    out = root/FRESH; out.mkdir(parents=True, exist_ok=False)
    failed = None; results = {}
    try:
        io.save(out/'contract.json', c); io.save(out/'SOURCE.json', read(root/c['SOURCE']))
        plan = integrated.resource_plan(task_count=2, full_dimension=256)
        plan['policy'] = 'two independent candidate processes; private compiler/cache; full512 decodes'
        io.save(out/'resource_plan.json', plan)
        with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(candidate_worker, root, c, arm, out/arm): arm for arm in ('CDF', 'XDF')}
            for future in as_completed(futures):
                arm = futures[future]
                try:
                    results[arm] = future.result(); io.save(out/(arm+'.json'), results[arm])
                    if results[arm]['status'] != 'RELOCATED_STRONG_D_COMPLETE':
                        raise RuntimeError(results[arm]['status'])
                except BaseException:
                    for pending in futures: pending.cancel()
                    raise
        result = compare_points(results, c['E2_point_RZ']); io.save(out/'comparison.json', result)
        return result
    except BaseException as exc:
        failed = repr(exc); io.save(out/'failure.json', dict(error=failed, completed_candidates=list(results))); raise
    finally:
        io.save(out/'STOP.json', dict(failed=failed is not None, reason=failed or 'TWO_CANDIDATES_COMPLETE',
            completed_candidates=list(results), retry=0, resume=0, mandatory_stop=True))
