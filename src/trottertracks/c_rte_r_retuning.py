"""Fixed selected J/U, finite r retuning through existing RTE/RPE/native code.

Import and preflight are passive. run() requires separate scientific permission.
No fitter, PF engine, shot formula, compiler, or approval-proof system is added.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import time

from . import c_pr_integrated_performance as perf
from . import c_pr_aware_df_stage_c as accounting

io, m4, m3 = perf.io, perf.m4, perf.m3
BASE = 'a4d24682e79857f1e97f105cc03448d0e1786ea2'
SCOPE = 'C_SELECTED_H4_FIXED_JU_RTE_R_RETUNING_V1'
VERSION = 'saved_PF_ground_r16_32_64_saved128_native32_v1'
FAMILY = 'artifacts/research_c_rte_r_retuning/2026-11-03'
FRESH = FAMILY + '/run01'
R_VALUES = (16, 32, 64, 128)
NEW_R = R_VALUES[:-1]
CANDIDATE = '938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e'
INPUT = '1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
POLICY = dict(atoms=4, distance_A=1., basis='STO-3G', rank=8, Nalpha=2, Nbeta=2,
    sector_dimension=36, system_modes=8, controlled_dimension=512, ld=2,
    frame_order=list(range(8)), PF='partial ordered symmetric S2', delta='1/10',
    h='1/10', s=1, K=0, rounds=list(range(13)), original_g_kappa=True,
    scalar='nuclear once', oracle='own_DF_ground_state_oracle', state_preparation='EXCLUDED')
COMPILER = dict(qiskit='1.3.0', optimization_level=1, seed_transpiler=17,
    basis_gates=['rz', 'sx', 'x', 'cx'], approximation_degree=1., qubits_initially_zero=False)
COMMON_NATIVE = {k: v for k, v in m3.B_LIMITS.items()
    if k not in {'trajectory_sample', 'kernel_build', 'trajectory_reference', 'raw_operator_gate',
        'compile', 'compiled_operator_gate', 'adjoint_pair_gate', 'XY_gate', 'kernel_metrics',
        'calibration_summary', 'delta_cost_comparison'}}
SAMPLE_NATIVE = {k: v for k, v in m3.B_LIMITS.items()
    if k not in COMMON_NATIVE and k != 'delta_cost_comparison'}
A_LIMITS = dict(evidence_load=1, input_load=1, selected_load=1, reference_load=1,
    candidate_binding=1, sector_context=1, native_model=1, general_factor_gate=9,
    ground_load=1, ground_gate=1, target_validation=1, baseline_attachment=1)
R_LIMITS = dict(law_prepare=1, law_gate=1, normalization_gate=1, corrected_kernel=1,
    round_reference=13, round_classification=13, feasibility_summary=1,
    signal_qualification=3, round_accuracy=39, energy_assessment=3, shots_summary=3,
    native_admission=1, **SAMPLE_NATIVE, cost_calibration=1, serial_cost=26,
    total_aggregation=3, calibration_uncertainty=3, resource_comparison=1)
LIMITS = {**A_LIMITS, **{'native.'+k: v for k, v in COMMON_NATIVE.items()},
    **{f'r{r}.'+k: v for r in NEW_R for k, v in R_LIMITS.items()}, 'result_summary': 1}
MAX_CALLS = sum(LIMITS.values())
RECORDS = {'parent', 'performance', 'parent_STOP', 'performance_STOP', 'parent_SOURCE',
    'parent_contract', 'physical', 'ground', 'physical_ground', 'law', 'D', 'D_STOP',
    'D_SOURCE', 'D_contract', 'shell_reports'}


def sampling(r):
    if r not in NEW_R: raise ValueError('REGISTERED_NEW_R_ONLY')
    return dict(master=2026110301, cohort=f'selected_r{r}_fixed32_v1',
        axis=f'R_RETUNING_r{r}_NATIVE_ONLY', samples=32, events_per_sample=r,
        PRNG='numpy.random.PCG64', paired=False, adaptive=False)


def evidence(root, c):
    """JSON identity/qualifications only; not a new numerical assessment."""
    root = Path(root)
    s = {k: io.read(root/p) for k, p in c['records'].items()}
    parent, old, contract, df = (s[k] for k in ('parent', 'performance', 'parent_contract', 'physical'))
    if (parent['selected']['id'] != 'update005' or parent['selected']['physical'] != df
            or df['candidate_sha256'] != CANDIDATE or df['actual_rank'] != 8
            or not df['budget_PASS'] or df['budget_Ha'] != .00019875
            or df['sector_error'] + df['numerical_guard'] > .00019875
            or df['kappa'] != 'original_g_corrected' or df['scalar'] != 'nuclear_once'):
        raise ValueError('SELECTED_DF_IDENTITY')
    if (s['parent_STOP']['failed'] or s['parent_STOP']['scientific_units'] != 543
            or s['performance_STOP']['failed'] or old['status'] != 'PERFORMANCE_NATIVE_COMPLETE'
            or contract['sampling'] != perf.SAMPLING or contract['target'] != m4.TARGET
            or contract['compiler'] != COMPILER or contract['input_paths']['input'] != c['input']
            or contract['saved_inputs'][c['input']] != INPUT
            or io.sha(root/c['records']['parent_SOURCE']) != contract['source_manifest_sha256']):
        raise ValueError('SAVED_PR_SOURCE_POLICY')
    if any(s['performance_STOP']['call_counts'][k] != 32 for k in
            ('trajectory_sample', 'compile', 'raw_operator_gate', 'compiled_operator_gate', 'XY_gate')):
        raise ValueError('SAVED_R128_COHORT')
    for key in ('ground', 'physical_ground'):
        if s[key]['status'] != 'PASS' or s[key]['alias_margin'] <= 0:
            raise ValueError('SAVED_GROUND_ALIAS')
    if (s['law']['status'] != 'PASS' or s['law']['lambda_R'] != df['exact_tail_lambda_Ha']
            or [x['m'] for x in old['rounds']] != list(range(13))
            or any(x['reference']['pf_operator_error'] is None for x in old['rounds'])):
        raise ValueError('SAVED_PF_LAW')
    for lane in accounting.LANES:
        rows = old['lanes'][lane]['rounds']
        if [x['m'] for x in rows] != list(range(13)):
            raise ValueError('SAVED_R128_ROUNDS')
        if lane != 'evidence' and any(x['target_id'] != 'integrated:'+CANDIDATE for x in rows):
            raise ValueError('SAVED_R128_CANDIDATE')
    if (s['D']['status'] != 'SAME_JU_STRONG_D_COMPLETE' or s['D']['selected_export'] != CANDIDATE
            or s['D']['saved_DF'] != df or s['D_STOP']['failed']
            or s['D_STOP']['scientific_units'] != 156 or s['D_contract']['target'] != m4.TARGET
            or s['D_contract']['compiler'] != COMPILER
            or io.sha(root/c['records']['D_SOURCE']) != s['D_contract']['source_manifest_sha256']):
        raise ValueError('SAME_JU_D_IDENTITY')
    return s


def preflight(root, path):
    """Bytes, JSON metadata and package versions; no NPZ/QPY decode or output."""
    root = Path(root).resolve(); c = io.read(path)
    fixed = dict(base_commit=BASE, scope=SCOPE, implementation=VERSION, policy=POLICY,
        r_values=list(R_VALUES), new_r_values=list(NEW_R), saved_r=128,
        target=m4.TARGET, compiler=COMPILER, allowances=m3.ALLOWANCES,
        scenarios=list(accounting.SCENARIOS), call_limits=LIMITS, maximum_logical_units=MAX_CALLS,
        fresh_output=FRESH, retry=0, resume=0, new_fit=0, optimizer=0,
        new_PF_error=0, new_DF_error=0, new_ground_eigensolve=0,
        sampling={str(r): sampling(r) for r in NEW_R})
    for key, want in fixed.items():
        if c[key] != want: raise ValueError('FIXED_RETUNING:'+key)
    if set(c['records']) != RECORDS: raise ValueError('RECORD_SCHEMA')
    needed = {c[k] for k in ('input', 'selected', 'physical_reference', 'sector_reference', 'basis', 'ground')}
    needed.update(c['records'].values())
    needed.update(p for row in c['shells'].values() for p in row.values())
    if (not needed <= c['saved_inputs'].keys() or c['saved_inputs'][c['selected']] != CANDIDATE
            or c['saved_inputs'][c['input']] != INPUT): raise ValueError('PINNED_CANDIDATE_INPUT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != digest: raise ValueError('SOURCE_INPUT:'+name)
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('ENVIRONMENT:'+name)
    from .c1_h4_native_fair import GAUSSIAN, PHASE
    if c['gaussian'] != GAUSSIAN or c['compiler_phase'] != PHASE: raise ValueError('NATIVE_PRECISION')
    evidence(root, c)
    if (root/FRESH).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def prepare(root, c, out, call):
    import numpy as np
    from . import c1_molecule_extension as molecule
    from . import c_pr_aware_df_m2 as m2
    b, loader = perf.b, perf.loader
    saved = call('evidence_load', 'saved candidate/DF/PF/RPE/native/D provenance', lambda: evidence(root, c))
    p = call('input_load', 'original g corrected kappa/scalar', lambda: loader.load(root/c['input'], ('h', 'g', 'constant')))
    d = call('selected_load', 'selected J/U; no fit', lambda: loader.load(root/c['selected'], ('J', 'U')))
    def load_refs():
        refs = loader.load(root/c['sector_reference'], ('generators', 'H', 'pf'))
        original = loader.load(root/c['physical_reference'], ('J', 'U', 'generators', 'H'))
        basis = loader.load(root/c['basis'], ('sector_basis_indices',))['sector_basis_indices']
        refs.update(original=original, full_input_indices=b.stage1.bit_reverse_indices(8)[basis])
        return refs
    refs = call('reference_load', 'same selected sector/PF/basis references', load_refs)
    def bind():
        m2.validate_arrays(p, d, 'selected')
        if not all(np.array_equal(d[k], refs['original'][k]) for k in ('J', 'U')):
            raise ValueError('SELECTED_ARRAY_IDENTITY')
        m4.sector_schema_gate(refs)
        for k in ('generators', 'H'):
            if not np.array_equal(refs[k], refs['original'][k]): raise ValueError('SAVED_SECTOR_IDENTITY')
        return dict(status='PASS', new_DF_error=0, new_PF_error=0, saved_candidate=CANDIDATE)
    io.save(out/'identity.json', call('candidate_binding', 'saved reference arrays', bind))
    ctx = call('sector_context', 'existing independent sector36 CAR', lambda: molecule.SectorCAR(4))
    arm = call('native_model', 'general K and original kappa/scalar', lambda: b.prepare_arm(**p, **d,
        role='proposed', source_identity=CANDIDATE, input_identity=INPUT, ld=2))
    for i in range(9):
        call('general_factor_gate', str(i), lambda i=i: b.component_check(
            b.candidate_factor(ctx, arm, i), refs['generators'][i], 1e-10, label='same selected generalK/CAR'))
    ground = call('ground_load', 'saved own-DF oracle', lambda: loader.load(root/c['ground'], ('energy', 'sector_state')))
    def ground_check():
        psi = ground['sector_state']; energy = float(ground['energy'])
        if psi.shape != (36,) or not np.isfinite(psi).all(): raise ValueError('SAVED_GROUND_SCHEMA')
        residual = float(np.linalg.norm(refs['H']@psi-energy*psi)); norm = float(abs(np.vdot(psi, psi)-1))
        if (not math.isfinite(residual+norm) or residual > 1e-10 or norm > 1e-12
                or abs(energy-saved['ground']['energy_Ha']) > 1e-12): raise ValueError('GROUND_CORRECTNESS')
        return dict(status='PASS', residual=residual, normalization=norm,
            alias_margin=saved['ground']['alias_margin'], new_eigensolve=0)
    io.save(out/'ground.json', call('ground_gate', 'normalization/residual and saved alias', ground_check))
    target = call('target_validation', 'unchanged CA/phase/failure/13 rounds', lambda: m4.accuracy.AccuracyTarget(**m4.TARGET))
    if target.horizon()['maximum_round_index_M'] != 12: raise ValueError('REGISTERED_HORIZON')
    baseline = call('baseline_attachment', 'saved r128 and same-JU D; no recalculation',
        lambda: dict(r=128, kind='SAVED_BASELINE_NO_NEW_SCIENCE', result=saved['performance'],
            source='980b514c81eb415fe053f558d7d3c0b8cfb5b9b4'))
    io.save(out/'saved_r128.json', baseline)
    return dict(saved=saved, p=p, arm=arm, ctx=ctx, sector=refs, ground=ground['sector_state'], target=target)


def normalization(tail, config, r):
    tau = float(Fraction(tail.lambda_r)*Fraction(1, 10)/r)
    if (config.rte_steps != r or config.finite_taylor_order != 0 or config.evolution_time != .1
            or config.lambda_r != tail.lambda_r
            or not math.isclose(config.dimensionless_step_time, tau, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('REGISTERED_R_TAU')
    rte = perf.b.reference._legacy('rte')
    if not math.isclose(config.distribution_normalization, math.sqrt(1+tau*tau),
            rel_tol=1e-12, abs_tol=1e-12): raise ValueError('K0_STEP_NORMALIZATION')
    rows = []
    for m in range(13):
        got = rte.finite_rte_attenuation(config, tail_evolutions=1 << m)
        want = config.distribution_normalization**(-r*(1 << m))
        if not math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12): raise ValueError('K0_NORMALIZATION')
        rows.append(dict(m=m, attenuation=got))
    return dict(status='PASS', r=r, tau=tau, lambda_R=tail.lambda_r, rows=rows,
        identity_phase_separate=True, definition='A=(1+tau^2)^(-r*2^m/2)')


def assess_lanes(packet, target, call):
    lanes = {}
    for lane in accounting.LANES:
        q = call('signal_qualification', lane, lambda lane=lane: accounting.qualification(packet, lane))
        plans = [call('round_accuracy', lane+str(row['m']),
            lambda row=row: accounting.assess_round(target, packet, q, row)) for row in packet['rounds']]
        energy = call('energy_assessment', lane, lambda: accounting.energy(target, packet, plans))
        shots = call('shots_summary', lane, lambda: None if not energy['same_accuracy'] else sum(2*p['shots'] for p in plans))
        lanes[lane] = dict(qualification=q, rounds=plans, energy=energy, total_shots=shots)
    return lanes


def compare(lanes, saved):
    out = {}
    for lane, value in lanes.items():
        totals = value.get('conditional_totals', {}).get('totals')
        if totals is None:
            out[lane] = dict(status='UNQUALIFIED_OR_INFEASIBLE', ratios=None); continue
        targets = {'saved_r128': saved['performance']['lanes'][lane]['conditional_totals']['totals'],
            'same_JU_D_ideal': saved['D']['ideal_exact_native']['cost']['totals'],
            'same_JU_D_numerical': saved['D']['numerical']['cost']['totals']}
        out[lane] = dict(status='CONDITIONAL_FINITE_COHORT_POINT_COMPARISON',
            ratios={name: {k: totals[k]/old[k] if old[k] else None for k in m4.METRICS}
                for name, old in targets.items()}, primary='rz_count', guaranteed_advantage=None)
    return out


def evaluate_r(root, c, out, call, a, r, prepared):
    from threadpoolctl import threadpool_limits
    from . import c1_stage3_feasibility as feasibility
    from .c_rc2_pr_stage_b import distribution_gate
    b = perf.b; arm = a['arm']; old = a['saved']; sector = a['sector']
    tail, config, dist = call('law_prepare', f'r{r}/K0', lambda: b.configure(arm, Fraction(1, 10), r, 0))
    def law_check():
        row = b.tail_gate(a['ctx'], arm, sector['generators'], tail)
        distribution_gate(tail, config, dist, expected_steps=r)
        if (row['lambda_R'] != old['law']['lambda_R'] or row['identity'] != old['law']['identity']
                or row['atoms'] != old['law']['atoms']): raise ValueError('LAW_CANDIDATE_IDENTITY')
        return row
    io.save(out/'law.json', call('law_gate', 'signed atoms/identity and existing distribution', law_check))
    io.save(out/'normalization.json', call('normalization_gate', 'r-dependent K0', lambda: normalization(tail, config, r)))
    kernels = call('corrected_kernel', 'existing finite RTE only; saved PF untouched',
        lambda: m4.corrected_kernel(sector['generators'], config, ld=2,
            identity_tail=arm.plan.extracted_identity, saved_pf=sector['pf'], expected_steps=r))
    perf.loader.save_arrays(out/'RTE_reference.npz', corrected=kernels['corrected'])
    plan = m4.integrated.resource_plan(task_count=13, full_dimension=36)
    io.save(out/'round_resource_plan.json', plan)
    def task(m):
        row = call('round_reference', str(m), lambda: m4.rte_round(kernels, m,
            old['performance']['rounds'][m]['reference']['pf_operator_error'], state=a['ground']))
        decision = call('round_classification', str(m), lambda: feasibility.classify_round(row, a['target']))
        value = dict(m=m, reference=row, feasibility=decision)
        io.save(out/f'm{m:02d}.json', value); return value
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            rounds = list(pool.map(task, range(13)))
    df = old['physical']['sector_error'] + old['physical']['numerical_guard']
    f = call('feasibility_summary', 'all13 registered sufficient conditions',
        lambda: feasibility.summarize([x['feasibility'] for x in rounds], df, a['target']))
    packet = dict(target_id=f'retuning:{CANDIDATE}:r{r}', rounds=rounds,
        df_bound_Ha=df, error_evidence=None, shells=old['shell_reports'])
    lanes = assess_lanes(packet, a['target'], call)
    admitted = call('native_admission', 'all13 feasibility and ideal energy/shot gate',
        lambda: f['status'] == 'POTENTIALLY_FEASIBLE' and lanes['ideal_exact_native']['energy']['same_accuracy'])
    result = dict(r=r, feasibility=f, rounds=rounds, lanes=lanes, native_admitted=bool(admitted),
        status='REFERENCES_COMPLETE', calibration=None, actual_CA_achieved=None,
        all_law_synthesis=None, implemented_law_error=None, cost_support=None)
    io.save(out/'B.json', result)
    if admitted:
        native_a = dict(arm=arm, p=a['p'], tail=tail, config=config, dist=dist,
            sector=sector, ground=a['ground'])
        samples, cal = perf.native(root, c, out, call, native_a, sampling=sampling(r), prepared=prepared)
        packet.update(kernel_samples=samples, saved_calibration=cal)
        kernel = call('cost_calibration', 'candidate/r-specific fixed32, unknown support', lambda: accounting.cost_calibration(packet))
        costs = [call('serial_cost', f'{m}{axis}', lambda m=m, axis=axis: m3.serial_cost(packet, kernel, m, axis))
            for m in range(13) for axis in ('X', 'Y')]
        for lane, value in lanes.items():
            totals = call('total_aggregation', lane, lambda value=value: m4.accuracy.aggregate_totals(
                value['energy'], value['rounds'], costs, cost_scope=accounting.COST_SCOPE))
            value['conditional_totals'] = totals
            value['weighted_SE'] = call('calibration_uncertainty', lane,
                lambda value=value, totals=totals: m3.calibration_uncertainty(packet, value['rounds'], totals))
        result.update(status='NATIVE_AND_CONDITIONAL_COST_COMPLETE', calibration=cal,
            shared_native_context_origin=prepared['origin'])
    else:
        result['status'] = 'NO_NATIVE_REGISTERED_CONDITIONS_NOT_MET'
    result['comparisons'] = call('resource_comparison', 'saved128 and same-JU D, all six metrics', lambda: compare(lanes, old))
    io.save(out/'result.json', result)
    return result


def execute(root, c, out, call, *, prepare_function=prepare, r_function=evaluate_r):
    a = prepare_function(root, c, out, call); results = {}; prepared = {}
    for r in NEW_R:
        folder = out/f'r{r}'; folder.mkdir()
        def dispatch(kind, label, function):
            key = 'native.'+kind if kind in COMMON_NATIVE else f'r{r}.'+kind
            return call(key, label, function)
        dispatch.stop = call.stop
        results[str(r)] = r_function(root, c, folder, dispatch, a, r, prepared)
        with (out/'progress.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(dict(r=r, status=results[str(r)]['status']))+'\n')
    return call('result_summary', 'all registered r, no new search', lambda: dict(
        status='R_RETUNING_REGISTERED_SCOPE_COMPLETE', r_values=list(R_VALUES), new=results,
        saved_r128='saved_r128.json', candidate=CANDIDATE, retry=0, resume=0,
        actual_CA_achieved=None, guaranteed_resource_superiority=None, mandatory_stop=True))


def run(root, path):
    root = Path(root).resolve(); c = preflight(root, path); out = root/FRESH
    out.mkdir(parents=True); start = time.monotonic()
    call = m4.integrated.Calls(out, LIMITS, MAX_CALLS)
    io.save(out/'contract.json', c); io.save(out/'SOURCE.json', io.read(root/c['source_manifest']))
    try:
        result = execute(root, c, out, call); io.save(out/'result.json', result); return result
    except BaseException as error:
        call.stop.set(); io.save(out/'failure.json', dict(error=repr(error), first_failure=call.first_failure)); raise
    finally:
        io.save(out/'STOP.json', dict(failed=call.stop.is_set(), scientific_units=sum(call.counts.values()),
            maximum_logical_units=MAX_CALLS, call_counts=dict(call.counts), first_failure=call.first_failure,
            retry=0, resume=0, mandatory_stop=True))
        io.save(out/'resources.json', dict(wall_seconds=time.monotonic()-start, CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
