"""Saved strong CDF8/XDF8 -> existing conditional ideal RPE/serial lane.

Passive import/preflight. No arrays, operators, fit, circuits, new accuracy
engine or new statistics. Real saved-value joins need separate permission.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from . import c1_h4_strong_ca_resources as saved

io, accuracy, reused, integrated = saved.io, saved.accuracy, saved.reused, saved.integrated
BASE = '8375433869999c2e10501b1983c11b2b22c5d423'
VERSION = 'saved_B2_B3_counterfactual_exact_native_v1'
SCOPE = 'C_XDF8_CDF8_COMMON_IDEAL_S1_V1'
FAMILY = 'artifacts/research_c_xdf_cdf_common_ideal_lane'
FRESH = FAMILY + '/2026-11-03/run01'
TARGET, CONDITIONS = dict(saved.TARGET), dict(saved.CONDITIONS)
ROUTES = dict(saved.ROUTES)
LANE = dict(name='ideal_exact_native', native_synthesis_signal_error=0.,
            synthesis_zero_is='counterfactual_assumption_not_measured',
            PF_error='saved_candidate_specific_retained',
            DF_error='saved_candidate_specific_norm_plus_guard_once',
            attenuation=1., corrected_RTE_error=0., finite_law_error=0.,
            law_model='deterministic', actual_CA_achieved=None)
LIMITS = dict(saved_bundle_load=1, candidate_gate=2, ideal_round_accuracy=26,
              ideal_energy=2, round_serial_cost=26, ideal_aggregate=2,
              comparison=1, result_summary=1)
MAX_CALLS = sum(LIMITS.values())
RECORDS = {'old_result', 'old_SOURCE', 'old_STOP', 'old_contract',
           'CDF_packet', 'XDF_packet'}
FORBIDDEN = ('new_fit', 'U_optimization', 'DF_recomputation', 'PF_recomputation',
             'ground_eigensolve', 'array_decode', 'build_compile',
             'synthesis_recomputation', 'PR_recalculation', 'new_statistics')


def fresh_output(root):
    out = Path(root).resolve() / FRESH
    if out.exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return out


def preflight(root, path):
    """Hash/policy/metadata only; no real-input accuracy, W or cost calls."""
    root = Path(root).resolve()
    c = io.read(path)
    fixed = dict(scope=SCOPE, status='UNAPPROVED_SCIENCE_CONTRACT', base_commit=BASE,
                 implementation=VERSION, conditions=CONDITIONS, target=TARGET,
                 lane=LANE, call_limits=LIMITS, maximum_logical_units=MAX_CALLS,
                 fresh_output=FRESH, retry=0, resume=0, mandatory_stop=True,
                 cost_scope=saved.COST_SCOPE, candidate_order=list(ROUTES.values()))
    for key, want in fixed.items():
        if c[key] != want:
            raise ValueError('COMMON_IDEAL_POLICY:' + key)
    if set(c['records']) != RECORDS or not set(c['records'].values()) <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_SAVED_RECORD')
    source = (root / c['source_manifest']).resolve()
    if not source.is_relative_to(root) or io.sha(source) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(source)['files'], c['saved_inputs']):
        for name, expected in group.items():
            p = (root / name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != expected:
                raise ValueError('SOURCE_INPUT_DRIFT:' + name)
    if any(not name.endswith('.json') for name in c['saved_inputs']):
        raise ValueError('JSON_ONLY_SAVED_INPUTS')
    if any(c[k] for k in FORBIDDEN):
        raise ValueError('FORBIDDEN_SCIENTIFIC_SCOPE')
    for package, expected in c['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('RUNTIME_PACKAGE:' + package)
    fresh_output(root)
    return c


def load_saved(root, c):
    """One future counted saved-evidence load; reuse frozen identity gates."""
    r = {key: io.read(Path(root) / p) for key, p in c['records'].items()}
    old, stop, source, contract = (r[k] for k in ('old_result', 'old_STOP', 'old_SOURCE', 'old_contract'))
    if (old['status'] != 'H4_STRONG_BASELINE_S1_CA_RESOURCES_COMPLETE'
            or old['scope'] != saved.SCOPE or stop['failed'] or not stop['mandatory_stop']
            or stop['retry'] or stop['resume'] or stop['scientific_units'] != saved.MAX_CALLS
            or stop['call_counts'] != saved.LIMITS
            or source['execution_commit'] != '1aeedd96178a3f50a15bc2701c3c332f4f391b64'
            or contract['accuracy_target'] != TARGET or contract['conditions'] != CONDITIONS
            or io.sha(Path(root) / contract['source_manifest']) != contract['source_manifest_sha256']
            or source['files'] != io.read(Path(root) / contract['source_manifest'])['files']
            or contract['compiler'] != c['compiler'] or contract['gaussian'] != c['gaussian']
            or contract['compiler_phase'] != c['compiler_phase']):
        raise ValueError('SAVED_CA_SOURCE_CONTRACT_STATUS')
    r['bundle'] = saved.load_bundle(root, contract)
    return r


def attach_packet(records, arm):
    packet = saved.attach_evidence(records['bundle'], arm)
    if packet != records[arm + '_packet'] or packet['id'] != ROUTES[arm]:
        raise ValueError('SAVED_PACKET_CANDIDATE_DRIFT')
    old = records['old_result']['candidates'][packet['id']]
    if (old['id'] != packet['id'] or old['arm'] != arm or old['s'] != 1
            or old['aliases'] != packet['aliases']):
        raise ValueError('SAVED_RESULT_CANDIDATE_IDENTITY')
    # The packet retains old synthesis evidence, but that evidence is NOT the
    # zero assumed below. Leave the published packet and numerical lane intact.
    return packet


def ideal_round(packet, pf, target):
    """Same conversion as c_same_ju_strong_d.ideal_plans; existing engine."""
    e = accuracy.RoundEvidence(m=pf['m'], attenuation=1.,
        pf_operator_error=pf['pf_operator_error'], rte_corrected_operator_error=0.,
        synthesis_signal_error=0., finite_law_signal_error=0., target_id=packet['id'],
        state_model=packet['state_model'], alias_branch_verified=packet['alias_verified'],
        iid_per_shot=True, law_model='deterministic')
    plan = accuracy.assess_round(target, e)
    if plan['status'] == 'CONDITIONAL_ROUND_PASS':
        # Reuse the saved deterministic fixed-N check; never increase N after
        # observing a failed integer margin or silently certify that round.
        n = plan['shots']
        verified = reused.det.assess_fixed_round(target,
            SimpleNamespace(id=packet['id'], s=1),
            dict(m=pf['m'], attenuation=1., rte_corrected_operator_error=0.,
                 pf_operator_error=pf['pf_operator_error']),
            dict(m=pf['m'], axes={a: dict(optimistic_shots_decimal=str(n)) for a in ('X', 'Y')}),
            dict(occurrences=1 << pf['m'], complex_signal_error_pilot=0.),
            alias_verified=packet['alias_verified'])
        if verified['status'] != 'CONDITIONAL_ROUND_PASS':
            plan.update(status='UNRESOLVED_INTEGER_SHOT_MARGIN', shots=None, shot_check=verified)
        else:
            plan['fixed_statistical_margin'] = verified['fixed_statistical_margin']
    plan.update(lane='ideal_exact_native', native_synthesis_assumed_exact=True,
                synthesis_zero_measured=False, actual_CA_achieved=None)
    return plan


def aggregate(packet, plans, energy, costs):
    total = accuracy.aggregate_totals(energy, plans, costs, cost_scope=saved.COST_SCOPE)
    # Bookkeeping of existing serial records, not an independent cost engine.
    weight = None
    if total.get('totals') is not None:
        shots = {p['m']: p['shots'] for p in plans}
        weight = sum(shots[r['m']] * r['kernel_occurrences'] for r in costs)
    return dict(id=packet['id'], arm=packet['arm'], lane='ideal_exact_native',
        status='CONDITIONAL_IDEAL_CA_PASS' if energy.get('same_accuracy')
            and total.get('totals') is not None else 'INFEASIBLE_OR_UNRESOLVED',
        rounds=plans, energy=energy, per_shot_costs=costs, cost=total,
        kernel_weight_W=weight, saved_DF_error_with_guard=packet['df_error_with_guard'],
        kernel_metrics=packet['kernel_metrics'], shell_metrics=packet['shell_metrics'],
        assumptions=dict(LANE), actual_CA_achieved=None, rigorous_enclosure=None,
        guaranteed_resource_superiority=None, new_statistics=0,
        failure_scope='each predeclared candidate; no joint/post-selection guarantee')


def execute(root, c, out, call):
    records = call('saved_bundle_load', 'saved JSON/source/ground/alias only',
                   lambda: load_saved(root, c))
    packets = {a: call('candidate_gate', ROUTES[a], lambda a=a: attach_packet(records, a)) for a in ROUTES}
    for p in packets.values():
        io.save(out / 'candidates' / (p['id'] + '.packet.json'), p)
    target = accuracy.AccuracyTarget(**TARGET)
    plan = integrated.resource_plan(task_count=2, full_dimension=1)
    plan.update(BLAS_threads_per_worker=None, policy='two independent scalar chains; no dense arrays or shared settings')
    io.save(out / 'resource_plan.json', plan)

    def evaluate(arm):
        packet, plans = packets[arm], []
        key = packet['id']
        for pf in packet['PF_rounds']:
            value = call('ideal_round_accuracy', f'{key}:m{pf["m"]}',
                         lambda pf=pf: ideal_round(packet, pf, target))
            plans.append(value)
            integrated.progress(out, key, 'ideal_round_accuracy', value)
            if value['status'] != 'CONDITIONAL_ROUND_PASS':
                break  # Stop this candidate's dependent evaluation/costs.
        energy = call('ideal_energy', key, lambda: accuracy.assess_energy(target,
            packet['df_error_with_guard'], plans, state_model=packet['state_model'],
            alias_branch_verified=packet['alias_verified']))
        costs = []
        if energy.get('same_accuracy'):
            for row in plans:
                costs.extend(call('round_serial_cost', f'{key}:m{row["m"]}:XY',
                                  lambda row=row: reused.serial_round_cost(packet, row['m'])))
        value = call('ideal_aggregate', key, lambda: aggregate(packet, plans, energy, costs))
        io.save(out / 'candidates' / (key + '.json'), value)
        return arm, value

    results = {}
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        for f in as_completed([pool.submit(evaluate, a) for a in ROUTES]):
            arm, value = f.result()  # Calls stops new dependent work on errors.
            results[arm] = value
    comparison = call('comparison', 'CDF/XDF only; no PR statistics', lambda: dict(
        comparison=accuracy.compare_totals(results['CDF']['cost'], results['XDF']['cost']),
        qualification='counterfactual exact native; fixed gate prices; no statistical test',
        actual_CA_achieved=None, guaranteed_resource_superiority=None))
    result = call('result_summary', 'two saved candidates and ideal lane', lambda: dict(
        status='COMMON_IDEAL_LANE_COMPLETE' if all(v['status'] == 'CONDITIONAL_IDEAL_CA_PASS'
            for v in results.values()) else 'COMMON_IDEAL_LANE_INFEASIBLE_OR_UNRESOLVED',
        scope=SCOPE, candidates={ROUTES[a]: results[a] for a in ROUTES}, comparison=comparison,
        original_scientific_SOURCE='1aeedd96178a3f50a15bc2701c3c332f4f391b64',
        scientific_units=sum(call.counts.values()), call_counts=dict(call.counts),
        actual_CA_achieved=None, PR_XDF_statistical_superiority=None,
        equivalence=None, noninferiority=None, **{k: 0 for k in FORBIDDEN}, mandatory_stop=True))
    io.save(out / 'result.json', result)
    return result


def run(root, path, *, execute_science=False):
    if not execute_science:
        raise ValueError('EXPLICIT_SCIENTIFIC_OPT_IN_REQUIRED')
    root = Path(root).resolve()
    c = preflight(root, path)
    out = fresh_output(root)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'candidates').mkdir()
    call = integrated.Calls(out, LIMITS, MAX_CALLS)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        io.save(out / 'contract.json', c)
        io.save(out / 'SOURCE.json', dict(io.read(root / c['source_manifest']),
                                         execution_commit=saved.prior.source_commit(root)))
        return execute(root, c, out, call)
    except BaseException as error:
        call.stop.set()
        io.save(out / 'failure.json', dict(error=repr(error), first_failure=call.first_failure))
        raise
    finally:
        io.save(out / 'STOP.json', dict(failed=call.stop.is_set(),
            scientific_units=sum(call.counts.values()), call_counts=dict(call.counts),
            first_failure=call.first_failure, retry=0, resume=0, mandatory_stop=True))
        io.save(out / 'resources.json', dict(wall_seconds=time.monotonic()-wall,
            CPU_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
