"""Fixed update064 -> existing r32 PR and same-J/U strong S2 evaluators.

Preflight reads bytes/JSON only. Scientific execution is a separate permission
boundary. No fit, PF/RTE engine, compiler, or statistical guarantee is added.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c_rte_r_retuning as retuning
from . import c_same_ju_strong_d as strong

perf, io, m4, m3 = retuning.perf, retuning.io, retuning.m4, retuning.m3
BASE = 'f200e45afecc0a80b9f8a895b68c75ba4dda5dd5'
PEER = 'ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6'
CANDIDATE = '53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3'
INPUT = retuning.INPUT
SCOPE = 'C_UPDATE064_R32_PR_SAME_JU_STRONG_D_V1'
VERSION = 'saved_DF_new_ground_PF_RTE_native32_sameJU_D_v1'
ENTRY = 'scripts/run_research_c_update064_r32_downstream.py'
FAMILY = 'artifacts/research_c_update064_r32_downstream/2026-11-03'
FRESH = FAMILY + '/run01'
POLICY = dict(retuning.POLICY, candidate='update064', r=32,
    deterministic='all9 factors ordered symmetric S2,s1,ld8;r/K not applicable',
    DF_error='same-candidate saved numerical sector norm+engineering guard')
SAMPLING = dict(master=2026110302, cohort='update064_r32_fixed32_v1',
    axis='UPDATE064_R32_NATIVE_ONLY', samples=32, events_per_sample=32,
    PRNG='numpy.random.PCG64', paired=False, adaptive=False)
A_LIMITS = dict(evidence_load=1, input_load=1, selected_load=1, reference_load=1,
    candidate_binding=1, sector_context=1, physical_context=1, native_model=1,
    general_factor_gate=9, ground_reference=2, target_validation=1,
    law_prepare=1, law_gate=1, normalization_gate=1)
B_LIMITS = dict(kernel_reference=1, exact_evolution=13, round_reference=13,
    round_classification=13, feasibility_summary=1, signal_qualification=3,
    round_accuracy=39, energy_assessment=3, shots_summary=3, native_admission=1)
C_LIMITS = dict(**{k: v for k, v in m3.B_LIMITS.items() if k != 'delta_cost_comparison'},
    cost_calibration=1, serial_cost=26, total_aggregation=3, calibration_uncertainty=3)
# The old single-J/U D code's native/PF/accounting operations, with shared fresh
# Stage-A eigenpair instead of loading/re-solving old ground, and no old comparisons.
D_LIMITS = {k: v for k, v in strong.LIMITS.items() if k not in {
    'evidence_load', 'input_load', 'fit_load', 'reference_load', 'candidate_binding',
    'ground_load', 'ground_gate', 'sector_eigenpair', 'comparison', 'result_summary'}}
LIMITS = {**{'A.'+k: v for k, v in A_LIMITS.items()},
    **{'B.'+k: v for k, v in B_LIMITS.items()},
    **{'C.'+k: v for k, v in C_LIMITS.items()},
    **{'D.'+k: v for k, v in D_LIMITS.items()}, 'comparison': 5, 'result_summary': 1}
MAX_CALLS = sum(LIMITS.values())
PEER_ROLES = {'candidate', 'reference', 'basis', 'physical', 'result', 'STOP', 'contract', 'SOURCE'}


def dispatch(call, stage):
    def wrapped(kind, label, function):
        return call(stage+'.'+kind, label, function)
    wrapped.stop = call.stop
    return wrapped


def evidence(root, peer, c):
    """Saved verdicts/provenance only; candidate NPZ remains closed."""
    s = {k: io.read(peer/c['peer_paths'][k]) for k in ('physical', 'result', 'STOP', 'contract', 'SOURCE')}
    df, result, stop, old = (s[k] for k in ('physical', 'result', 'STOP', 'contract'))
    last = result['trajectory']['history'][-1]
    if (last['id'] != 'update064' or last['physical'] != df
            or last['qualification']['status'] != 'NUMERICAL_PILOT_ADMITTED'
            or result['trajectory']['physically_qualified_updates'] != 64
            or stop['failed'] or not stop['mandatory_stop'] or stop['scientific_units'] != 1042
            or stop['stop_reason'] != 'MAXIMUM_ACCEPTED_UPDATES'):
        raise ValueError('UPDATE064_PROVENANCE_STOP')
    if (df['candidate_sha256'] != CANDIDATE or df['actual_rank'] != 8
            or not df['budget_PASS'] or df['budget_Ha'] != .00019875
            or not math.isfinite(df['sector_error']+df['numerical_guard'])
            or min(df['sector_error'], df['numerical_guard']) < 0
            or df['sector_error']+df['numerical_guard'] > .00019875
            or df['kappa'] != 'original_g_corrected' or df['scalar'] != 'nuclear_once'):
        raise ValueError('UPDATE064_DF_IDENTITY')
    if (old['scope'] != 'C_PR_REDUCED_U_BOUNDED_ARMIJO_V1'
            or old['objective']['rank'] != 8 or old['objective']['ld'] != 2
            or old['initial_candidate']['sha256'] != retuning.CANDIDATE
            or old['input_paths']['input'] != c['input']
            or old['saved_inputs'][c['input']] != INPUT
            or io.sha(peer/c['peer_paths']['SOURCE']) != old['source_manifest_sha256']):
        raise ValueError('UPDATE064_SOURCE_INPUT')
    # Old r32 and old D are comparison records only, never numerical inputs.
    old_pr = io.read(root/c['records']['old_PR'])
    old_D = io.read(root/c['records']['old_D'])
    old_contract = io.read(root/c['records']['old_contract'])
    if (old_pr['r'] != 32 or old_D['selected_export'] != retuning.CANDIDATE
            or old_contract['saved_inputs'][old_contract['selected']] != retuning.CANDIDATE
            or old_contract['compiler'] != retuning.COMPILER or old_contract['target'] != m4.TARGET
            or old_contract['saved_inputs'][c['input']] != INPUT
            or io.read(root/c['records']['old_STOP'])['failed']
            or io.read(root/c['records']['old_D_STOP'])['failed']):
        raise ValueError('OLD_COMPARISON_POLICY')
    for lane in ('ideal_exact_native', 'assumed_raw_signal_budget'):
        rows = old_pr['lanes'][lane]['rounds']
        if ([x['m'] for x in rows] != list(range(13))
                or any(x['target_id'] != 'retuning:'+retuning.CANDIDATE+':r32' for x in rows)):
            raise ValueError('OLD_R32_CANDIDATE_BINDING')
    return dict(physical=df, old_PR=old_pr, old_D=old_D,
        shells=io.read(root/c['records']['shells']))


def preflight(root, path, peer):
    """Fixed-commit peer, byte/SOURCE binding and metadata; no array/circuit decode."""
    root, peer = Path(root).resolve(), Path(peer).resolve()
    c = io.read(path)
    fixed = dict(base_commit=BASE, candidate_commit=PEER, scope=SCOPE, implementation=VERSION,
        entry=ENTRY, policy=POLICY, sampling=SAMPLING, target=m4.TARGET,
        compiler=retuning.COMPILER, allowances=m3.ALLOWANCES,
        scenarios=list(perf.accounting.SCENARIOS), call_limits=LIMITS,
        maximum_logical_units=MAX_CALLS, fresh_output=FRESH, retry=0, resume=0,
        new_fit=0, optimizer=0, new_DF_error=0, mandatory_stop=True)
    for key, want in fixed.items():
        if c[key] != want: raise ValueError('FIXED_UPDATE064:'+key)
    if set(c['peer_paths']) != PEER_ROLES: raise ValueError('PEER_ROLE_SCHEMA')
    oid = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=peer, text=True).strip()
    if oid != PEER: raise ValueError('FIXED_PEER_COMMIT')
    needed = {c['input'], *c['records'].values(), c['dependency_lock']}
    needed.update(p for row in c['shells'].values() for p in row.values())
    if (not needed <= c['saved_inputs'].keys()
            or not set(c['peer_paths'].values()) <= c['peer_inputs'].keys()
            or c['saved_inputs'][c['input']] != INPUT
            or c['peer_inputs'][c['peer_paths']['candidate']] != CANDIDATE):
        raise ValueError('PINNED_INPUT_ROLES')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    if ENTRY not in source['files']: raise ValueError('UNPINNED_ENTRY')
    for folder, group in ((root, source['files']), (root, c['saved_inputs']), (peer, c['peer_inputs'])):
        for name, digest in group.items():
            p = (folder/name).resolve()
            if not p.is_relative_to(folder) or io.sha(p) != digest: raise ValueError('SOURCE_INPUT:'+name)
    for package, version in c['packages'].items():
        if importlib.metadata.version(package) != version: raise ValueError('ENVIRONMENT:'+package)
    from .c1_h4_native_fair import GAUSSIAN, PHASE
    if c['gaussian'] != GAUSSIAN or c['compiler_phase'] != PHASE: raise ValueError('NATIVE_PRECISION')
    evidence(root, peer, c)
    if (root/FRESH).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def prepare(root, peer, c, out, call):
    import numpy as np
    from . import c1_molecule_extension as molecule
    from .c_rc2_pr_stage_b import distribution_gate
    b, loader = perf.b, perf.loader
    saved = call('evidence_load', 'update064 DF and comparison-only update005', lambda: evidence(root, peer, c))
    p = call('input_load', 'original h/g/nuclear scalar', lambda: loader.load(root/c['input'], ('h', 'g', 'constant')))
    d = call('selected_load', 'update064 only', lambda: loader.load(peer/c['peer_paths']['candidate'], ('J', 'U')))
    def references():
        sector = loader.load(peer/c['peer_paths']['reference'], ('J', 'U', 'generators', 'H'))
        basis = loader.load(peer/c['peer_paths']['basis'], ('sector_basis_indices',))['sector_basis_indices']
        sector['full_input_indices'] = b.stage1.bit_reverse_indices(8)[basis]
        return sector
    sector = call('reference_load', 'same-candidate physical export and sector basis', references)
    def bind():
        strong.m2.validate_arrays(p, d, 'update064')
        if not all(np.array_equal(d[k], sector[k]) for k in ('J', 'U')):
            raise ValueError('MIXED_CANDIDATE_JU')
        from .c1_stage1 import sector_indices
        G, H, ix = (sector[k] for k in ('generators', 'H', 'full_input_indices'))
        if (G.shape != (9,36,36) or H.shape != (36,36)
                or not all(np.isfinite(x).all() for x in (G,H))
                or not np.array_equal(np.sort(ix), sector_indices(4,2,2))):
            raise ValueError('UPDATE064_SECTOR_SCHEMA')
        b.component_check(sum(sector['generators']), sector['H'], 1e-10, label='same-candidate sector sum')
        return dict(status='PASS', candidate_sha256=CANDIDATE, reused_DF=saved['physical'],
            DF_error_recomputed=False, old_ground_PF_RTE_price_used=False)
    io.save(out/'identity.json', call('candidate_binding', 'exact J/U match for saved DF reuse', bind))
    ctx = call('sector_context', 'independent SectorCAR4', lambda: molecule.SectorCAR(4))
    physical = call('physical_context', 'original physical Hamiltonian;no new DF norm', lambda: ctx.physical(**p))
    arm = call('native_model', 'update064 generalK;original kappa/scalar/order LD2',
        lambda: b.prepare_arm(**p, **d, role='proposed', source_identity=CANDIDATE, input_identity=INPUT, ld=2))
    for i in range(9):
        call('general_factor_gate', str(i), lambda i=i: b.component_check(
            b.candidate_factor(ctx, arm, i), sector['generators'][i], 1e-10, label='new generalK/saved CAR'))
    def ground(label, H):
        pair = loader.ground(H, out, label)
        io.save(out/(label+'_ground.json'), dict(loader.ground_gate(H, pair), candidate_sha256=CANDIDATE if label == 'candidate' else None))
        return pair
    call('ground_reference', 'new physical ground/alias', lambda: ground('physical', physical))
    pair = call('ground_reference', 'new update064 ground/alias shared by PR and D', lambda: ground('candidate', sector['H']))
    target = call('target_validation', 'existing CA/RPE13', lambda: m4.accuracy.AccuracyTarget(**m4.TARGET))
    if target.horizon()['maximum_round_index_M'] != 12: raise ValueError('REGISTERED_HORIZON')
    tail, config, dist = call('law_prepare', 'new signed r32 K0 law', lambda: b.configure(arm, Fraction(1,10), 32, 0))
    def law():
        row = b.tail_gate(ctx, arm, sector['generators'], tail)
        distribution_gate(tail, config, dist, expected_steps=32)
        if not math.isclose(row['lambda_R'], saved['physical']['exact_tail_lambda_Ha'], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('CANDIDATE_LAMBDA_CONVENTION')
        return row
    io.save(out/'law.json', call('law_gate', 'identity separate;new candidate atom weights', law))
    io.save(out/'normalization.json', call('normalization_gate', 'r32 K0', lambda: retuning.normalization(tail, config, 32)))
    return dict(saved=saved, p=p, d=d, arm=arm, ctx=ctx, sector=sector, pair=pair,
        ground=pair[1][:,0], target=target, tail=tail, config=config, dist=dist)


def references(a, out, call):
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from . import c1_stage3_feasibility as feasibility
    b = perf.b; H = a['sector']['H']
    candidate=a.get('candidate_identity',CANDIDATE)
    target_label=a.get('candidate_label','update064')
    kernels = call('kernel_reference', 'fresh '+target_label+' partial PF and corrected r32 RTE',
        lambda: b.kernels(a['arm'], a['sector']['generators'], Fraction(1,10), a['config']))
    perf.loader.save_arrays(out/'sector_reference.npz', pf=kernels['pf'], corrected=kernels['corrected'])
    plan = m4.integrated.resource_plan(task_count=13, full_dimension=36)
    io.save(out/'resource_plan.json', plan)
    def task(m):
        elapsed = Fraction(1,10)*(1<<m)
        exact = call('exact_evolution', str(m), lambda: expm(-1j*float(elapsed)*H))
        row = call('round_reference', str(m), lambda: b.reference.reference_round(kernels,
            (H, elapsed), repetitions=1<<m, state=a['ground'], exact_operator=exact))
        signal = row.pop('raw_signal', None)
        row['raw_signal_parts'] = None if signal is None else [float(signal.real), float(signal.imag)]
        decision = call('round_classification', str(m), lambda: feasibility.classify_round(row, a['target']))
        value = dict(m=m, reference=row, feasibility=decision)
        io.save(out/f'm{m:02d}.json', value)
        return value
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool: rounds = list(pool.map(task, range(13)))
    df = a['saved']['physical']; bound = df['sector_error']+df['numerical_guard']
    f = call('feasibility_summary', 'all13 sufficient conditions', lambda: feasibility.summarize(
        [x['feasibility'] for x in rounds], bound, a['target']))
    packet = dict(target_id=target_label+':r32:'+candidate, rounds=rounds, df_bound_Ha=bound,
        error_evidence=None, shells=a['saved']['shells'])
    lanes = retuning.assess_lanes(packet, a['target'], call)
    admitted = call('native_admission', 'all13 and ideal gate;unknown evidence remains null',
        lambda: f['status'] == 'POTENTIALLY_FEASIBLE' and lanes['ideal_exact_native']['energy']['same_accuracy'])
    result = dict(candidate=candidate, r=32, rounds=rounds, feasibility=f, lanes=lanes,
        native_admitted=bool(admitted), calibration=None, actual_CA_achieved=None,
        all_law_synthesis=None, implemented_law_error=None, cost_support=None)
    io.save(out/'references.json', result)
    return result, packet


def calibrate(root, c, out, call, a, result, packet,*,sampling=None):
    samples, cal = perf.native(root, c, out, call, a, sampling=SAMPLING if sampling is None else sampling)
    packet.update(kernel_samples=samples, saved_calibration=cal)
    kernel = call('cost_calibration', 'update064/r32 fixed32 only', lambda: perf.accounting.cost_calibration(packet))
    costs = [call('serial_cost', f'{m}{axis}', lambda m=m, axis=axis: m3.serial_cost(packet, kernel, m, axis))
        for m in range(13) for axis in ('X', 'Y')]
    for lane, value in result['lanes'].items():
        totals = call('total_aggregation', lane, lambda value=value: m4.accuracy.aggregate_totals(
            value['energy'], value['rounds'], costs, cost_scope=perf.accounting.COST_SCOPE))
        value['conditional_totals'] = totals
        value['weighted_SE'] = call('calibration_uncertainty', lane,
            lambda value=value, totals=totals: m3.calibration_uncertainty(packet, value['rounds'], totals))
    result['calibration'] = cal
    io.save(out/'result.json', result)
    return result


def comparisons(pr, deterministic, saved, call):
    absent = dict(totals=None, cost_scope=perf.accounting.COST_SCOPE)
    pairs = []
    for lane in ('ideal_exact_native', 'assumed_raw_signal_budget'):
        pairs.append(('update064_PR/update005_PR:'+lane,
            pr['lanes'][lane].get('conditional_totals', absent),
            saved['old_PR']['lanes'][lane]['conditional_totals'],
            'same r32/policy;different independent finite cohorts;descriptive,not causal or guaranteed'))
    if deterministic is not None:
        for lane, key in (('ideal_exact_native', 'ideal_exact_native'), ('assumed_raw_signal_budget', 'numerical')):
            pairs.append(('update064_PR/D:'+lane, pr['lanes'][lane].get('conditional_totals', absent),
                deterministic[key]['cost'] if deterministic[key] is not None else absent,
                'same update064 J/U;ideal/ideal or assumed PR/numerical D;qualifications retained'))
        pairs.append(('update064_D/update005_D:ideal', deterministic['ideal_exact_native']['cost'],
            saved['old_D']['ideal_exact_native']['cost'], 'DF endpoint comparison,not isolated objective causality'))
    return {label: call('comparison', label, lambda left=left, right=right, label=label, q=q:
        strong.m2.numerical_compare(left, right, label, qualification=q)) for label, left, right, q in pairs}


def execute(root, peer, c, out, call):
    from .c_update064_strong_d import evaluate as deterministic
    for label in ('A', 'B', 'C', 'D'): (out/label).mkdir()
    a = prepare(root, peer, c, out/'A', dispatch(call, 'A'))
    pr, packet = references(a, out/'B', dispatch(call, 'B'))
    d = None
    if pr['native_admitted']:
        pr = calibrate(root, c, out/'C', dispatch(call, 'C'), a, pr, packet)
        d = deterministic(root, c, out/'D', dispatch(call, 'D'), a)
        status = 'UPDATE064_PR_AND_SAME_JU_D_COMPLETE' if d['native_admitted'] else 'STOP_D_PF_UNQUALIFIED'
    else:
        status = 'STOP_PR_UNQUALIFIED_NO_C_D'
    compared = comparisons(pr, d, a['saved'], call)
    return call('result_summary', 'fixed comparison;mandatory STOP', lambda: dict(status=status,
        candidate=CANDIDATE, PR=pr, D=d, comparisons=compared,
        actual_CA_achieved=None, guaranteed_resource_superiority=None, causal_objective_effect=None,
        update064_r32_optimality=None, strict_DF_enclosure=None, retry=0, resume=0, mandatory_stop=True))


def run(root, path, peer):
    root, peer = Path(root).resolve(), Path(peer).resolve()
    c = preflight(root, path, peer); out = root/FRESH
    out.mkdir(parents=True); call = m4.integrated.Calls(out, LIMITS, MAX_CALLS)
    start, cpu = time.monotonic(), time.process_time()
    result = None
    try:
        io.save(out/'contract.json', c); io.save(out/'SOURCE.json', io.read(root/c['source_manifest']))
        result = execute(root, peer, c, out, call); io.save(out/'result.json', result)
        return result
    except BaseException as error:
        call.stop.set(); io.save(out/'failure.json', dict(error=repr(error), first_failure=call.first_failure)); raise
    finally:
        io.save(out/'STOP.json', dict(failed=call.stop.is_set(), scientific_units=sum(call.counts.values()),
            maximum_logical_units=MAX_CALLS, call_counts=dict(call.counts), first_failure=call.first_failure,
            stop_reason=result['status'] if result else 'CORRECTNESS_OR_EXECUTION_FAILURE',
            retry=0, resume=0, mandatory_stop=True))
        io.save(out/'resources.json', dict(wall_seconds=time.monotonic()-start,
            CPU_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
