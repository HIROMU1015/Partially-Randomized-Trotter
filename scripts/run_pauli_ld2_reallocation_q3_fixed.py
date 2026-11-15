"""Adopted PR-Pauli LD2 B_PRICE re-allocations (FPN-B, OWN-B): at most 2 Q3 calls.

Reuses the fixed Pauli Q3 route (scripts/run_pauli_q3_three_quarter.py: same
signal, decoder, target, loss, exact threshold, accelerated backend). Adds only
exact B_PRICE generation from the published model-reference prices and worker-
owned persistence (API return, RESULT, resources) before returning to the
parent. No MC, N search, price fit/prediction, compile, retry or new vectors.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)  # exact rational results can exceed the default digit limit


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Q = load_script('_pauli_three_quarter_route', 'scripts/run_pauli_q3_three_quarter.py')
R, T = Q.R, Q.T
PLAN = '15c08ff29341ba88391b20886369fd65d358967a'
PLAN_DIR = 'docs/research/pauli_ld2_shot_reallocation_plan_20261115/'
Q3_FIXED = 'f54b41a2e311e44124c9e180e26b01fe11d4eabd'
Q3_INPUT = 'docs/research/pauli_q3_three_quarter_20261115/INPUTS.json'
VP_RESULT = ('50be68b803a90fd893a1499ebce74e033225144d', 'docs/research/pauli_q3_three_quarter_20261115/run01/RESULT_LD2.json')
V0_RESULT = ('caf1417ef20d0a5ab2030a2d957ee4f09443be43', 'docs/research/pauli_total_q3_accelerated_results_20261114/run01/RESULT_LD2.json')
PRICES = ('a7eceecba5f1d2172682a4e5cc6f1573568debd0', 'docs/research/pauli_ld2_tref_20261115/AXIS_TABLE.json')
TREF = ('a7eceecba5f1d2172682a4e5cc6f1573568debd0', 'docs/research/pauli_ld2_tref_20261115/RESULT.json')
B_GENERATOR = ('06f4210e1bdfd5b07f42b941fa22d17e228533fc', 'scripts/run_pf6_cost_budget_q3_fixed.py')
PF_MINIMA = ('b4296b110b129d226dbef2743c14f725a27c3bf6', 'docs/research/pf6_cost_budget_q3_results_20261115/SUMMARY.json')
DIRECTORY = 'docs/research/pauli_ld2_reallocation_q3_results_20261115'
INPUT = DIRECTORY+'/INPUTS.json'
CALLER = 'scripts/run_pauli_ld2_reallocation_q3_fixed.py'
VP = [48,45,42,39,36,33,29,26,23,19,16,13,9]
V0 = [64,60,56,52,47,43,38,34,30,25,21,17,12]
CONFIG = 'd7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'
FPN_BUDGET = 615304502
FAILURE = None


def document(commit, path):
    return json.loads(R.blob(commit, path))


def ref(pair, selector=None):
    return dict(commit=pair[0], path=pair[1], **({} if selector is None else dict(selector=selector)))


def frac(value):
    if isinstance(value, dict) and set(value) == {'numerator','denominator'}:
        return Fraction(value['numerator'], value['denominator'])
    if type(value) is int:
        return Fraction(value)
    raise ValueError('Exact rational required')


def exact(value):
    return T.B.a.exact_value(value)


def check_source():
    Q.check_source()
    for path in ('scripts/run_pauli_q3_three_quarter.py', Q3_INPUT):
        if (ROOT/path).read_bytes() != R.blob(Q3_FIXED, path):
            raise ValueError('Fixed Pauli Q3 route/input SOURCE differs: '+path)


def generate(caps, prices, budget, rule='B_PRICE'):
    """B's adopted breakpoint chain (06f4210e generate) with positive rational prices.

    The only difference from B's function is that prices may be positive exact
    rationals (the frozen Pauli model references are not integers).
    """
    if len(caps) != len(prices) or not caps:
        raise ValueError('Price/vector length mismatch')
    if any(type(n) is not int or n <= 0 for n in caps):
        raise ValueError('Positive exact integer caps required')
    prices = [Fraction(c) for c in prices]
    if any(c <= 0 for c in prices):
        raise ValueError('Positive exact prices required')
    budget = Fraction(budget)
    if rule not in ('A_SHAPE','B_PRICE'):
        raise ValueError('Unadopted allocation rule')
    weights = [Fraction(n) if rule == 'A_SHAPE' else Fraction(n)/c for n,c in zip(caps,prices)]
    points = sorted({Fraction(k)/w for n,w in zip(caps,weights) for k in range(1,n+1)})

    def at(lam):
        shots = [max(1,min(n,(lam*w).numerator//(lam*w).denominator)) for n,w in zip(caps,weights)]
        return shots, sum(n*c for n,c in zip(shots,prices))

    if sum(prices) > budget:
        return None
    lo, hi = 0, len(points)
    while lo < hi:
        mid = (lo+hi)//2
        if at(points[mid])[1] <= budget:
            lo = mid+1
        else:
            hi = mid
    index = max(0,lo-1)
    shots, cost = at(points[index])
    if cost > budget:
        raise ValueError('Breakpoint budget inconsistency')
    return dict(rule=rule, lambda_value=points[index], shots=shots, conditional_cost=cost,
        unused_budget=budget-cost, budget=budget,
        next_breakpoint_cost=None if index+1 == len(points) else at(points[index+1])[1],
        finite_breakpoints=len(points))


def count_bounds(shots, frontier=8192):
    """Same count-only support bound as B's PF6 caller (X/Y Cartesian cells)."""
    cells = [(r['N_X']+1)*(r['N_Y']+1) for r in shots]
    states, pairs = min(cells[0],frontier), cells[0]
    for c in cells[1:]:
        pairs += states*c
        states = min(frontier,states*c)
    return dict(maximum_round_outcomes=max(cells),
        verification_pairs_upper=sum(a*b for a,b in zip(cells,cells[1:])),
        transition_pairs_upper=pairs,
        rule='FULL_XY_CARTESIAN_COUNTS_AND_FRONTIER_UPPER_BOUND_NO_PROBABILITIES')


def split_cost(terms):
    early = sum(frac(t['N_times_C_ref']) for t in terms if t['m'] <= 9)
    late = sum(frac(t['N_times_C_ref']) for t in terms if t['m'] >= 10)
    return early, late


def build_inputs():
    check_source()
    fixed = document(Q3_FIXED, Q3_INPUT)
    base = deepcopy(next(c for c in fixed['cases'] if c['label'] == 'LD2'))
    if base['original_configuration_id'] != CONFIG:
        raise ValueError('Fixed LD2 case differs')
    if [r['N_X'] for r in base['allocation']['shots']] != VP or [r['N_Y'] for r in base['allocation']['shots']] != VP:
        raise ValueError('VP cap differs from the fixed Q3 allocation')
    vp = document(*VP_RESULT)
    if vp['status'] != 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD':
        raise ValueError('VP baseline status differs')
    rows = document(*PRICES)['rows']
    rows = sorted(((i,r) for i,r in enumerate(rows)), key=lambda v:(v[1]['m'],v[1]['axis']))
    if [(r['m'],r['axis']) for _,r in rows] != [(m,a) for m in range(13) for a in ('X','Y')]:
        raise ValueError('26 model-reference price axes required')
    if any(r['N'] != VP[r['m']] for _,r in rows):
        raise ValueError('Price table N differs from VP')
    prices = [frac(r['C_ref']) for _,r in rows]
    tref = frac(document(*TREF)['T_ref'])
    if sum(VP[r['m']]*c for (_,r),c in zip(rows,prices)) != tref:
        raise ValueError('T_ref does not reproduce from the price table')
    budgets = {'FPN-B': Fraction(FPN_BUDGET), 'OWN-B': Fraction(3,4)*tref}
    pf = document(*PF_MINIMA)
    proposals, cases, seen = [], [], {}
    known = {tuple(n for n in VP for _ in 'XY'): 'VP_RESULT_50be68b8', tuple(n for n in V0 for _ in 'XY'): 'V0_DIAGNOSTIC_caf1417e'}
    for pid, budget in budgets.items():
        g = generate([VP[r['m']] for _,r in rows], prices, budget, 'B_PRICE')
        allocation_id = 'PAULI_LD2_'+pid.replace('-','_')+'_PRICE_REALLOCATION_01'
        if g is None:
            proposals.append(dict(id=pid, allocation_id=allocation_id, disposition='BUDGET_MENU_EMPTY', scientific_calls=0))
            continue
        shots = [dict(m=m, N_X=g['shots'][2*m], N_Y=g['shots'][2*m+1]) for m in range(13)]
        vector = tuple(g['shots'])
        terms = [dict(m=r['m'], axis=r['axis'], N=n, C_ref=exact(c), N_times_C_ref=exact(n*c),
                      price_source=ref(PRICES,['rows',i]), price_kind='MODEL_REFERENCE_NOT_E_C',
                      extrapolated=r['m'] >= 10)
                 for (i,r),n,c in zip(rows,g['shots'],prices)]
        early, late = split_cost(terms)
        if early+late != g['conditional_cost']:
            raise ValueError('Cost split differs')
        bounds = count_bounds(shots)
        options = fixed['options']
        if (bounds['maximum_round_outcomes'] > options['maximum_round_outcomes']
                or bounds['verification_pairs_upper'] > options['maximum_verification_pairs']
                or bounds['transition_pairs_upper'] > options['total_state_refinement']['maximum_transition_pairs']):
            raise ValueError('Count-only scope cannot contain concrete allocation')
        disposition, reused = 'NEW_EVALUATION', None
        if vector in known:
            disposition, reused = 'SAME_VECTOR_AS_EXISTING_RESULT', known[vector]
        elif vector in seen:
            disposition, reused = 'SAME_VECTOR_AS_OTHER_PROPOSAL', seen[vector]
        label = pid.replace('-','_')+'_LD2'  # final digit selects the LD2 packet in the fixed route
        new = deepcopy(base)
        new.update(label=label, proposal=pid, prior_reference_allocation=deepcopy(base['allocation']),
            allocation=dict(allocation_id=allocation_id, shot_policy='USER_SUPPLIED', shots=shots),
            rule='B_PRICE', budget=exact(budget), conditional_cost=exact(g['conditional_cost']),
            unused_budget=exact(g['unused_budget']), cost_m0_m9=exact(early), cost_m10_m12=exact(late),
            generation=dict(lambda_value=exact(g['lambda_value']), next_breakpoint_cost=exact(g['next_breakpoint_cost'])
                            if g['next_breakpoint_cost'] is not None else None, finite_breakpoints=g['finite_breakpoints'],
                            generator=ref(B_GENERATOR,['generate']), difference='positive rational prices accepted'),
            price_terms=terms, finite_scope_preflight=bounds)
        proposals.append(dict(id=pid, label=label, allocation_id=allocation_id, disposition=disposition,
            reused_result=reused, allocation=deepcopy(new['allocation']), budget=new['budget'],
            conditional_cost=new['conditional_cost'], unused_budget=new['unused_budget'],
            cost_m0_m9=new['cost_m0_m9'], cost_m10_m12=new['cost_m10_m12'], finite_scope_preflight=bounds))
        if disposition == 'NEW_EVALUATION':
            cases.append(new); seen[vector] = label
    if len(cases) > 2:
        raise ValueError('Adopted scientific call limit exceeded')
    contract = deepcopy(fixed['statistical_contract'])
    contract['id'] = 'PAULI_LD2_B_PRICE_REALLOCATION_FPN_B_OWN_B_TOTAL_Q3'
    options = deepcopy(fixed['options'])
    options['model_loss_contract']['scope'] = 'PAULI_LD2_B_PRICE_REALLOCATION_FPN_B_OWN_B'
    concurrency = max(1, min(len(cases), len(os.sched_getaffinity(0))//(fixed['acceleration']['workers_per_configuration']+1)))
    return dict(fixed_plan=dict(commit=PLAN, path=PLAN_DIR+'REPORT.md'), fixed_Q3_route=dict(commit=Q3_FIXED, path=Q3_INPUT),
        B_source=fixed['B_source'], kernel_source=fixed['kernel_source'], C_source=fixed['C_source'], A_handoff=fixed['A_handoff'],
        prices=ref(PRICES), T_ref=exact(tref), T_ref_source=ref(TREF), PF_minima=ref(PF_MINIMA),
        PF_benchmarks=dict(FULL_PF_S2_B_PRICE=615304502, B0I_P7_S2_B_PRICE=435481090),
        baseline=dict(allocation=deepcopy(base['allocation']), result=ref(VP_RESULT), status=vp['status'],
                      MSE_lower=vp['record']['diagnostic']['deterministic_risk']['MSE_lower'],
                      MSE_upper=vp['record']['diagnostic']['deterministic_risk']['MSE_upper'], model_reference_cost=exact(tref)),
        V0_diagnostic=dict(result=ref(V0_RESULT), allocation=V0, adopted_threshold_judgment_saved=False,
                           reusable_for_proposals=False, reason='Different vector; proposals are componentwise <= VP < V0'),
        proposals=proposals, cases=cases, maximum_scientific_calls=2, actual_new_call_limit=len(cases),
        statistical_contract=contract, options=options, acceleration=deepcopy(fixed['acceleration']),
        adopted_evaluation=deepcopy(fixed['adopted_evaluation']), concurrent_configurations=concurrency,
        concurrency_basis='available CPU affinity divided by 8 DP children plus one parent; no OS caps',
        price_policy='MODEL_REFERENCE_C_REF_A7ECEECBA_DIAGNOSTIC_APPLICATION_NOT_E_C',
        signal_recomputed=False, sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, winner=None)


def prepare():
    inputs = build_inputs()
    (ROOT/DIRECTORY).mkdir(exist_ok=True)
    R.write_new(ROOT/INPUT, inputs)
    for p in inputs['proposals']:
        print(p['id'], p['disposition'], [r['N_X'] for r in p['allocation']['shots']], p['conditional_cost'], flush=True)


def init_worker(flag):
    global FAILURE
    FAILURE = flag
    Q.FAILURE = flag


def worker(case, inputs, source, output):
    """Fixed Pauli route numerics (Q.worker); persist before returning to the parent."""
    output = Path(output); start = time.perf_counter(); cpu = time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN); calls = 0; api_saved = False
    original = T.evaluate

    def capture(*args, **kwargs):
        nonlocal calls, api_saved
        if calls:
            raise ValueError('More than one scientific call prohibited')
        calls += 1
        records = original(*args, **kwargs)
        R.write_new(output/'API_RETURN.json', records); api_saved = True
        return records

    try:
        with patch.object(T, 'evaluate', capture):
            result = Q.worker(case, inputs, source, str(output))
        if calls != result['scientific_evaluation_calls']:
            raise ValueError('Scientific call counter mismatch')
        result.update(proposal=case['proposal'], conditional_cost=case['conditional_cost'], budget=case['budget'],
            cost_m0_m9=case['cost_m0_m9'], cost_m10_m12=case['cost_m10_m12'],
            original_api_return_saved=api_saved, persistence_scope='WORKER_OWNED_BEFORE_SUPERVISOR_RETURN')
        R.write_new(output/'RESULT.json', result)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = child.ru_utime+child.ru_stime-child_start.ru_utime-child_start.ru_stime
        R.write_new(output/'RESOURCE_FINAL.json', dict(label=case['label'], source=source,
            wall_seconds=time.perf_counter()-start, worker_CPU_seconds=time.process_time()-cpu,
            child_CPU_seconds=child_cpu, process_tree_CPU_seconds=time.process_time()-cpu+child_cpu,
            worker_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_single_completed_child_peak_RSS_KiB=child.ru_maxrss,
            scope='THROUGH_API_AND_RESULT_PERSISTENCE_CHILD_PEAK_IS_NOT_PROCESS_TREE_SUM'))
        receipt = dict(label=case['label'], proposal=case['proposal'], status=result['status'], source=source,
            scientific_evaluation_calls=calls, api_return_saved=api_saved, result_saved=True, resource_saved=True,
            operational_status='WORKER_ARTIFACTS_SAVED', sufficient_N=None, G_qualified=None)
        R.write_new(output/'PERSISTED.json', receipt)
        return receipt
    except Exception as exc:
        if FAILURE is not None:
            FAILURE.set()
        R.write_new(output/'WORKER_PERSISTENCE_ERROR.json', dict(label=case['label'], type=type(exc).__name__,
            reason=str(exc), scientific_evaluation_calls=calls, api_return_saved=api_saved, mandatory_stop=True))
        raise


def supervise(cases, inputs, source, output, *, monitor=None, poll_seconds=2):
    """Monitor errors stop unstarted calls; started workers save their own artifacts."""
    monitor = monitor or (lambda directory, labels, started, peaks: R.resources(directory, labels, started, peaks))
    output = Path(output); start = time.perf_counter(); context = mp.get_context('fork'); failure = context.Event()
    receipts, errors, monitor_errors, peaks, enabled = [], [], [], {}, True
    if cases:
        with ProcessPoolExecutor(max_workers=inputs['concurrent_configurations'], mp_context=context,
                                 initializer=init_worker, initargs=(failure,)) as pool:
            pending = {pool.submit(worker, c, inputs, source, str(output/c['label'])): c['label'] for c in cases}
            while pending:
                done, _ = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
                for future in done:
                    label = pending.pop(future)
                    try:
                        receipt = future.result(); receipt['sampled_peak_process_tree_RSS_KiB'] = peaks.get(label)
                        receipts.append(receipt); R.write_new(output/label/'SUPERVISOR_RECEIPT.json', receipt)
                        if receipt['status'].startswith('UNKNOWN'):
                            failure.set()
                        print(label+': '+receipt['status'], flush=True)
                    except Exception as exc:
                        failure.set(); errors.append(dict(label=label, type=type(exc).__name__, reason=str(exc)))
                if pending and enabled:
                    try:
                        for label in pending.values():
                            monitor(output/label, [label], start, peaks)
                    except Exception as exc:
                        failure.set(); enabled = False
                        monitor_errors.append(dict(type=type(exc).__name__, reason=str(exc)))
                        R.write_new(output/'MONITOR_ERROR.json', dict(errors=monitor_errors, scientific_retry=False))
    return dict(receipts=receipts, errors=errors, monitor_errors=monitor_errors,
                sampled_process_tree_peak_RSS_KiB=peaks, wall_seconds=time.perf_counter()-start)


def execute():
    check_source(); sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    for path in (INPUT, CALLER):
        if (ROOT/path).read_bytes() != R.blob(sha, path):
            raise ValueError('SOURCE not fixed: '+path)
    inputs = json.loads((ROOT/INPUT).read_text())
    if len(inputs['cases']) > 2 or len({c['label'] for c in inputs['cases']}) != len(inputs['cases']):
        raise ValueError('At most two distinct fixed cases')
    output = ROOT/DIRECTORY/'run01'; output.mkdir()
    for c in inputs['cases']:
        (output/c['label']).mkdir()
    source = dict(commit=sha, path=INPUT, caller=CALLER)
    R.write_new(output/'EXECUTION.json', dict(source=source, started_UTC=R.utc(), maximum_scientific_calls=2,
        cases=[c['label'] for c in inputs['cases']], retry=False))
    result = supervise(inputs['cases'], inputs, source, output)
    status = ('FIXED_SCIENTIFIC_WORKERS_SAVED' if not result['errors'] and not result['monitor_errors']
              and all(not r['status'].startswith('UNKNOWN') for r in result['receipts']) else 'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json', dict(status=status, **result, finished_UTC=R.utc(),
        sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, winner=None, mandatory_stop=True))
    print('Finished:', status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('prepare','execute'))
    (prepare if parser.parse_args().mode == 'prepare' else execute)()
