#!/usr/bin/env python3
"""Approved class comparison: saved N/targets and fresh price cohorts only.

Old output-bound CLI entry points are never launched. H6 price_job is reused
inside an isolated child with OUT redirected to this study's copied inputs and
fresh batch. H4 keeps its original step-seed central_trajectory path.
"""
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from fractions import Fraction as Q

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'artifacts/pauli_hybrid_rte_preparation/2026-10-25'
OUT = ROOT / 'artifacts/pauli_hybrid_rte_comparison/2026-10-25'
COMP = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
AXES = ('cosine', 'sine')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


R = module('hybrid_rte_original_price', 'scripts/pauli_h2_h6_study.py')
C = module('hybrid_rte_class_min', 'src/trotterlib/pauli_hybrid_rte_comparison.py')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    R.save(path, value)


def rows(path):
    with Path(path).open() as f:
        return [{k: None if v in ('', 'null') else v for k, v in r.items()} for r in csv.DictReader(f)]


def csvsave(path, values):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(values)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_check():
    source = read(OUT / 'SOURCE.json')
    for path, expected in source['code_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('SCIENTIFIC_SOURCE_CHANGED ' + path)
    for path, expected in source['input_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('FIXED_INPUT_CHANGED ' + path)
    return source


def stats_function():
    import numpy as np
    return R.A.A.extract('src/trotterlib/fixed_signal_evaluation.py', '_statistics',
                         dict(np=np, math=math, Fraction=Q))


def summary(inventory):
    result = []
    for molecule in ('H2', 'H4', 'H6'):
        for eps in EPS:
            x = dict(molecule=molecule, epsilon_sig=eps)
            for cohort in ('exploration', 'confirmation'):
                for group in ('RTE', 'Hybrid'):
                    x[cohort + '_' + group] = C.saved_class_minimum(
                        inventory, molecule=molecule, epsilon=eps, group=group, cohort=cohort)
                h, r = x[cohort + '_Hybrid'], x[cohort + '_RTE']
                x[cohort + '_Hybrid_over_RTE_exact'] = C.hybrid_over_rte(h, r)
            x['confirmation_classification'] = classify(x['confirmation_Hybrid'], x['confirmation_RTE'])
            result.append(x)
    return result


def classify(hybrid, rte):
    h, r = hybrid['selected'], rte['selected']
    if h is None or r is None:
        return 'CONFIRMATION_MISSING'
    if h['G_SE'] is None or r['G_SE'] is None:
        return 'EMPIRICAL_NEAR_TIE_OR_UNRESOLVED'
    delta = Q(h['G_exact']) - Q(r['G_exact'])
    combined = math.hypot(float(h['G_SE']), float(r['G_SE']))
    if abs(float(delta)) <= 2 * combined:
        return 'EMPIRICAL_NEAR_TIE_OR_UNRESOLVED'
    return 'HYBRID_LOWER_EMPIRICAL_G' if delta < 0 else 'RTE_LOWER_EMPIRICAL_G'


def extension_decision(new_row, other_row):
    """Fixed nominal precision rule; never a population-winner certificate."""
    from scipy.stats import t
    if other_row is None or new_row['G_exact'] is None or other_row['G_exact'] is None:
        return dict(extend=False, reason='NO_COMPLETE_COMPARATOR')
    a, b = float(new_row['G_SE']), float(other_row['G_SE'])
    gap = abs(float(Q(new_row['G_exact']) - Q(other_row['G_exact'])))
    half = float(t.ppf(.975, int(new_row['price_n']) - 1)) * a
    close = gap <= 2 * math.hypot(a, b)
    imprecise = half > .01 * float(Q(new_row['G_exact']))
    floor = close and a <= b
    return dict(extend=close and imprecise and not floor,
                reason='OLD_COMPARATOR_UNCERTAINTY_DOMINATES' if floor else
                       'CLOSE_AND_NOMINAL_HALF_WIDTH_OVER_1PCT' if close and imprecise else
                       'NOT_CLOSE_OR_ALREADY_PRECISE',
                absolute_G_gap=gap, combined_SE=math.hypot(a, b),
                new_nominal_t95_halfwidth=half, nominal_not_simultaneous_or_sequential=True)


def analyze():
    import numpy as np
    import qiskit
    import scipy
    OUT.mkdir(parents=True, exist_ok=False)
    prospective = read(PREP / 'prospective_SOURCE_inputs.json')
    for path, expected in prospective['code_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('PREPARATION_KERNEL_CHANGED ' + path)
    for item in prospective['fixed_inputs']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('PREPARATION_INPUT_CHANGED')
    if (qiskit.__version__, np.__version__, scipy.__version__) != ('1.3.0', '1.26.4', '1.14.1'):
        raise ValueError('FIXED_RUNTIME_VERSION')
    active = read(PREP / 'execution_plan_DRAFT_DISABLED.json')
    active.update(status='ACTIVE_FIXED_SCIENTIFIC_SCOPE', execute_enabled=True,
                  preparation_commit='25d248a1f2ed88de62b0e621c2d1cea8532804c4')
    save(OUT / 'active_scope.json', active)
    code = dict(prospective['code_sha256'])
    code['scripts/run_pauli_hybrid_rte_comparison.py'] = sha(Path(__file__))
    code['src/trotterlib/rte_event_audit.py'] = sha(ROOT / 'src/trotterlib/rte_event_audit.py')
    inp = {x['path']: x['sha256'] for x in prospective['fixed_inputs']}
    for path in [PREP / 'candidate_evidence_inventory.csv', PREP / 'source_manifest.json',
                 PREP / 'execution_plan_DRAFT_DISABLED.json', OUT / 'active_scope.json']:
        inp[str(path.relative_to(ROOT))] = sha(path)
    save(OUT / 'SOURCE.json', dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                  code_sha256=code, input_sha256=inp, state_actions=0))
    save(OUT / 'runtime.json', dict(python=sys.version, qiskit=qiskit.__version__, numpy=np.__version__,
                                   scipy=scipy.__version__, compiler=COMP, affinity=len(os.sched_getaffinity(0))))
    inventory = rows(PREP / 'candidate_evidence_inventory.csv')
    source_check()
    initial = summary(inventory)
    save(OUT / 'initial_comparison.json', initial)
    nominations = {}
    for cell in initial:
        m, e = cell['molecule'], cell['epsilon_sig']
        group = 'RTE' if m == 'H6' else 'Hybrid'
        allowed = m == 'H6' or (m == 'H4' and e in ('0.001', '0.0005'))
        if not allowed or cell['confirmation_' + group]['selected'] is not None:
            continue
        candidate = cell['exploration_' + group]['selected']
        if candidate is None:
            continue
        if candidate['existing_confirmation_n'] is not None:
            raise ValueError('SUCCESSFUL_OLD_COHORT_MUST_NOT_REPEAT')
        key = (m, candidate['candidate_id'])
        if key not in nominations:
            nominations[key] = dict(molecule=m, candidate_id=key[1], group=group,
                                   config={k: int(candidate[k]) for k in ('L_D', 'q', 'r', 'K')},
                                   nominated_eps=[], saved_rows=[r for r in inventory if
                                     (r['molecule'], r['candidate_id'], r['cohort']) == (m, key[1], 'exploration')])
        nominations[key]['nominated_eps'].append(e)
    nominees = list(nominations.values())
    if sum(x['molecule'] == 'H4' for x in nominees) > 2 or sum(x['molecule'] == 'H6' for x in nominees) > 5:
        raise ValueError('CANDIDATE_SCOPE_EXCEEDED')
    save(OUT / 'nominations.json', dict(status='FROZEN_BEFORE_NEW_CONFIRMATION',
             selection='eligible finite exploration minimum per missing class/epsilon; lexical exact ties',
             source_inventory_sha256=sha(PREP / 'candidate_evidence_inventory.csv'), candidates=nominees))
    for m in sorted({x['molecule'] for x in nominees}):
        selected = [x for x in nominees if x['molecule'] == m]
        n = int(m[1:])
        original = 'artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/' if n == 6 else 'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/'
        names = ('model.json', 'reference.json') if n == 6 else ('pauli_jw_model.json', 'pauli_jw_result.json')
        for name, target in zip(names, ('model.json', 'reference.json')):
            dest = OUT / m / 'inputs' / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / original / name, dest)
        batch = OUT / m / 'confirmation01'
        batch.mkdir()
        source = source_check()
        source.update(model_sha256=sha(OUT / m / 'inputs/model.json'),
                      reference_sha256=sha(OUT / m / 'inputs/reference.json'))
        candidates = [dict(candidate_id=x['candidate_id'], molecule=n,
                           config=dict(method='PR', T=.8, **x['config'])) for x in selected]
        save(batch / 'plan.json', dict(molecule=n, candidates=candidates, SOURCE=source,
              nominated_eps={x['candidate_id']: x['nominated_eps'] for x in selected},
              initial_n=48, max_n=384, state_actions=0, compiled_axes_per_trajectory=2))
    print(json.dumps(dict(nominees=[{k:x[k] for k in ('molecule','candidate_id','nominated_eps')} for x in nominees],
                         state_actions=0)), flush=True)


class Capture:
    def observe(self, adapter, sequences, step_seeds):
        self.events = [dict(event_position=i, outer_step_index=i//adapter.r,
                            inner_RTE_position=i%adapter.r, **event.to_dict())
                       for i, event in enumerate(e for sequence in sequences for e in sequence)]
        self.step_seeds = step_seeds


def h4_trajectory(hamiltonian, cfg, seed, capture):
    # The preparation's S.central_trajectory has no audit keyword. Compose the
    # same existing audited adapter/control/wrapper used by the fixed H4 source.
    from hx_study.rte_event_audit import AuditedPauliSignalAdapter
    from hx_study.h4_d2_pauli_control import central_control_body
    from hx_study.fixed_signal_evaluation import hadamard_wrapper
    adapter = AuditedPauliSignalAdapter(hamiltonian, cfg.method, cfg.L_D, cfg.T/cfg.q,
                                       cfg.r, cfg.K, audit=capture)
    body = adapter.trajectory(q=cfg.q, seed=seed)
    controlled = central_control_body(body)
    return body, {a:hadamard_wrapper(controlled.body, a, already_controlled=True) for a in AXES}, controlled.sidecar


def worker(molecule, cid, index):
    import qiskit
    source_check()
    if qiskit.__version__ != '1.3.0':
        raise ValueError('FIXED_QISKIT')
    R.OUT = OUT  # Child-local redirection; fixed input bytes were copied above.
    batch = OUT / molecule / 'confirmation01'
    plan = R.verify(batch)
    if molecule == 'H6':
        R.price_job(batch, cid, 'confirmation', index)
        return
    c = next(x for x in plan['candidates'] if x['candidate_id'] == cid)
    x = c['config']
    from hx_study.pauli_signal_adapter import PauliHamiltonian
    from hx_study.fixed_signal_evaluation import default_compiler
    from hx_study.rte_compiled_cost import transpile_and_measure_cost
    m = read(OUT / molecule / 'inputs/model.json')
    h = PauliHamiltonian(m['n_qubits'], m['hamiltonian']['constant'],
                         tuple(map(tuple, m['hamiltonian']['terms'])), '{}')
    seed = R.S.trajectory_seed(x['L_D'], x['q'], x['r'], x['K'], 'confirmation', index)
    capture = Capture()
    body, wrappers, sidecar = h4_trajectory(h, R.S.config(x['L_D'], x['q'], x['r'], x['K']), seed, capture)
    dest = batch / 'cases' / cid / 'prices/confirmation' / f'{index:04d}'
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(candidate_id=cid, cohort='confirmation', index=index, seed=seed,
                  events=capture.events, orders=[e['taylor_order'] for e in capture.events],
                  event_count=len(capture.events), step_seeds=capture.step_seeds,
                  body_phase=float(body.global_phase), control_sidecar=sidecar,
                  qiskit_version=qiskit.__version__, state_preparation_excluded=True, axes={})
    if result['event_count'] != x['q'] * x['r']:
        raise ValueError('EVENT_COUNT')
    for axis in AXES:
        ins = R.qpy_saved(wrappers[axis], dest / f'{axis}_input.qpy.gz')
        cost = transpile_and_measure_cost(wrappers[axis], default_compiler(),
                                          actual_circuit_fingerprint=ins['raw_sha256'])
        compiled = R.qpy_saved(cost.transpiled_circuit, dest / f'{axis}_compiled.qpy.gz')
        result['axes'][axis] = {k:getattr(cost, k) for k in
             ('rz_count', 'cx_count', 'total_depth', 'rz_depth', 'cx_depth', 'circuit_size', 'global_phase')}
        result['axes'][axis].update(input_qpy=ins, compiled_qpy=compiled)
    save(dest / 'result.json', result)


def aggregate(nominee):
    stats = stats_function()
    m, cid = nominee['molecule'], nominee['candidate_id']
    folder = OUT / m / 'confirmation01/cases' / cid / 'prices/confirmation'
    paths = sorted(folder.glob('*/result.json'))
    raw = [read(p) for p in paths]
    if [r['index'] for r in raw] != list(range(len(raw))):
        raise ValueError('INCOMPLETE_NEW_COHORT')
    prices = [{a:r['axes'][a]['rz_count'] for a in AXES} for r in raw]
    costs = R.A.A.cost_summary(prices, stats)
    costs.update(candidate_id=cid, molecule=m, paths=[str(p.relative_to(ROOT)) for p in paths],
                 observed_orders={str(k):sum(r['orders'].count(k) for r in raw) for k in (0,2,4)})
    result = []
    for old in nominee['saved_rows']:
        row = dict(old)
        row.update(cohort='confirmation', price_n=str(len(prices)), existing_confirmation_n=None,
                   selection_role='NEW_EXPLORATION_NOMINEE_FROZEN_BEFORE_CONFIRMATION',
                   source_commit=read(OUT / 'SOURCE.json')['commit'],
                   source_file=str(folder.relative_to(ROOT)), price_evidence_locator=str(folder.relative_to(ROOT)))
        for axis, short in (('cosine','cos'),('sine','sin')):
            row[f'C_{short}_exact'] = costs['axes'][axis]['mean_exact']
            row[f'C_{short}_SE'] = str(costs['axes'][axis]['standard_error'])
        if old['N_cos'] is not None and old['N_sin'] is not None and old['eligibility'] == 'CONDITIONAL_GUARD_ELIGIBLE':
            weighted = R.A.A.paired(prices, {a:int(old['N_' + s]) for a,s in zip(AXES,('cos','sin'))}, stats)
            row.update(G_exact=weighted['mean_exact'], G_SE=str(weighted['standard_error']), G_availability='SAVED_FINITE_G')
        else:
            row.update(G_exact=None, G_SE=None)
        result.append(row)
    return result, costs


def idle_capacity():
    allowed = os.sched_getaffinity(0)
    def sample():
        values = {}
        for line in Path('/proc/stat').read_text().splitlines():
            x = line.split()
            if x[0].startswith('cpu') and x[0][3:].isdigit() and int(x[0][3:]) in allowed:
                v = list(map(int, x[1:])); values[x[0]] = (sum(v[:8]), v[3] + v[4])
        return values
    a = sample(); time.sleep(.3); b = sample()
    idle = sum((b[k][1]-a[k][1])/max(1,b[k][0]-a[k][0]) for k in a)
    return dict(affinity=len(allowed), idle_core_equivalents=idle,
                workers_from_actual_free_CPU=max(1, min(len(allowed), math.ceil(idle))),
                all_busy_behavior='one own low-priority worker; no global/process changes')


def dispatch(jobs, stage):
    import concurrent.futures
    capacity = idle_capacity()
    workers = min(len(jobs), capacity['workers_from_actual_free_CPU'])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    save(stage / 'dispatch_resources.json', capacity)
    def one(job):
        molecule, cid, index = job
        base = stage / 'native' / f'{molecule}_{cid}_{index:04d}'
        base.parent.mkdir(parents=True, exist_ok=True)
        command = ['nice', '-n', '10', COMP, str(Path(__file__).resolve()), 'worker', molecule, cid, str(index)]
        start = time.time()
        with base.with_suffix('.stdout.log').open('w') as stdout, base.with_suffix('.stderr.log').open('w') as stderr:
            p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
            rc = p.wait()
        return dict(command=command, pid=p.pid, exit_code=rc, reaped=True,
                    elapsed_seconds=time.time()-start, log_base=str(base.relative_to(ROOT)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        native = list(pool.map(one, jobs))
    save(stage / 'native.json', dict(workers=workers, results=native,
                                    all_exit_zero=all(x['exit_code']==0 for x in native)))
    if any(x['exit_code'] for x in native):
        save(stage / 'FAILED_STOP.json', dict(no_automatic_retry=True, all_workers_reaped=True,
             failures=[x for x in native if x['exit_code']]))
        raise RuntimeError('FRESH_BATCH_NATIVE_FAILURE_PRESERVED ' + str(stage))


def run():
    source_check()
    if (OUT / 'STUDY_STOP.json').exists():
        raise ValueError('COMPLETED_STUDY_MUST_NOT_RESUME')
    nominees = read(OUT / 'nominations.json')['candidates']
    saved = rows(PREP / 'candidate_evidence_inventory.csv')
    pending = nominees
    total_calls = 0
    current_rows = []
    final_costs = {}
    for level in (48,96,192,384):
        if not pending:
            break
        stage = OUT / 'stages' / f'n{level:03d}'
        stage.mkdir(parents=True, exist_ok=False)
        jobs = []
        for c in pending:
            folder = OUT / c['molecule'] / 'confirmation01/cases' / c['candidate_id'] / 'prices/confirmation'
            old_n = len(list(folder.glob('*/result.json')))
            jobs.extend((c['molecule'], c['candidate_id'], i) for i in range(old_n, level))
        total_calls += len(jobs)
        if total_calls > 2688:
            raise ValueError('APPROVED_CALL_RANGE_EXCEEDED')
        save(stage / 'plan.json', dict(total_n=level, jobs=jobs, cumulative_new_trajectory_calls=total_calls,
            cumulative_axis_compiles_max=2*total_calls, SOURCE=source_check(), no_automatic_retry=True))
        dispatch(jobs, stage)
        current_rows = []
        for c in nominees:
            new, costs = aggregate(c)
            current_rows.extend(new)
            final_costs[c['molecule'] + ':' + c['candidate_id']] = costs
        save(stage / 'new_confirmation_rows.json', current_rows)
        save(stage / 'price_statistics.json', final_costs)
        combined = saved + current_rows
        comparisons = summary(combined)
        decisions = []
        next_pending = []
        for c in pending:
            should_extend = False
            for eps in c['nominated_eps']:
                own = next(r for r in current_rows if (r['molecule'],r['candidate_id'],r['epsilon_sig'])
                           == (c['molecule'],c['candidate_id'],eps))
                other_group = 'Hybrid' if c['group']=='RTE' else 'RTE'
                other = C.saved_class_minimum(combined, molecule=c['molecule'], epsilon=eps,
                                              group=other_group, cohort='confirmation')['selected']
                decision = extension_decision(own, other)
                decisions.append(dict(molecule=c['molecule'], candidate_id=c['candidate_id'], epsilon_sig=eps, **decision))
                should_extend |= decision['extend']
            if should_extend and level < 384:
                next_pending.append(c)
        save(stage / 'extension_decisions.json', decisions)
        pending = next_pending
        print(json.dumps(dict(completed_total_n=level, new_calls=total_calls,
              next_candidates=[x['candidate_id'] for x in pending])), flush=True)
    finalize(saved, current_rows, final_costs, total_calls)


def finalize(saved, new, costs, calls):
    source_check()
    combined = saved + new
    comparisons = summary(combined)
    controls = rows(ROOT / 'artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/H2_H4_H6_optimized_PR_comparison.csv')
    table = []
    breakdown = []
    for x in comparisons:
        m, e = x['molecule'], x['epsilon_sig']
        control = next(r for r in controls if (r['molecule'],r['epsilon_sig'])==(m,e))
        line = dict(molecule=m, epsilon_sig=e)
        for cohort in ('exploration','confirmation'):
            for group in ('RTE','Hybrid'):
                minimum = x[cohort+'_'+group]
                selected = minimum['selected']
                prefix = cohort+'_'+group+'_'
                for k in ('candidate_id','G_exact','G_SE','price_n','L_D','q','r','K'):
                    line[prefix+k] = selected[k] if selected else None
                line[prefix+'unresolved_ids'] = ';'.join(minimum['unresolved_ids'])
                if selected:
                    breakdown.append(dict(comparison_molecule=m,comparison_epsilon=e,comparison_cohort=cohort,
                        comparison_group=group, **selected))
            ratio = x[cohort+'_Hybrid_over_RTE_exact']
            line[cohort+'_Hybrid_over_RTE_exact'] = ratio
            line[cohort+'_Hybrid_over_RTE_decimal'] = float(Q(ratio)) if ratio is not None else None
        line.update(confirmation_classification=x['confirmation_classification'],
                    deterministic_id=control['deterministic_candidate_id'],
                    deterministic_G_exact=control['deterministic_G_exact'],
                    old_H4_primary_id=control['PR_candidate_id'] if m=='H4' else None,
                    old_H4_primary_G_exact=control['PR_G_exact'] if m=='H4' else None,
                    hypothetical_rho='0', actual_compiled_rho='UNKNOWN', population_tail='UNGUARANTEED',
                    selection='class minima are post-selection; old H4 primary nominee retained separately')
        table.append(line)
    save(OUT / 'comparison_by_precision.json', comparisons)
    csvsave(OUT / 'comparison_by_precision.csv', table)
    csvsave(OUT / 'cost_breakdown.csv', breakdown)
    if new:
        csvsave(OUT / 'new_candidate_accounting.csv', new)
    save(OUT / 'new_price_statistics.json', costs)
    save(OUT / 'historical_baseline_separate.json', [r for r in saved if r['cohort']=='historical_baseline'])
    verify = dict(status='PASS', comparison_rows=len(table), unique_keys=len({(x['molecule'],x['epsilon_sig']) for x in table}),
                  new_trajectory_calls=calls, new_axis_compiles=2*calls, new_state_actions=0,
                  new_Hamiltonian_state_reference=0, old_results_overwritten=False,
                  saved_N_B_bias_allowance_reused=True, all_new_native_exit_zero_and_reaped=True,
                  actual_rho_guaranteed=False, population_tail_guaranteed=False,
                  important_claim_adopted=False, nominal_comparison_not_population_winner=True)
    assert verify['comparison_rows']==verify['unique_keys']==15
    save(OUT / 'verification.json', verify)
    save(OUT / 'STUDY_STOP.json', dict(mandatory_STOP=True, science_complete=True,
         new_science_may_not_resume=True, state_actions=0, trajectory_calls=calls, axis_compiles=2*calls))
    print(json.dumps(verify), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'analyze':
        analyze()
    elif sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'worker':
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise ValueError('MODE')
