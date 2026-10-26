"""Phase B entry, prepared but NOT executed in Phase A.

Use `run` only after the user permits the fixed Phase B scope. There is no
approval certificate/grant system. `check` reads metadata only.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from fractions import Fraction
import fcntl
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import prepare
native, study, report = prepare.native, prepare.study, prepare.report


def read_plan():
    return json.loads((HERE / 'PLAN.json').read_text())


def verify_static(plan):
    assert [x['candidate']['id'] for x in plan['candidates']] == list(prepare.IDS)
    assert plan['cohorts'] == list(prepare.COHORTS)
    assert plan['independent_confirmation'] == 'ALL_SEVEN_UNCONDITIONAL_AFTER_EXPLORATION_SUCCESS'
    assert not plan['confirmation_topup'] and not plan['new_candidate_expansion']
    for name, digest in plan['source_hashes'].items():
        assert prepare.sha(ROOT / name) == digest, 'SOURCE changed: ' + name
    for name, digest in plan['saved_data_hashes'].items():
        assert prepare.sha(ROOT / name) == digest, 'Saved baseline changed: ' + name
    for system in plan['systems'].values():
        for v in system['inputs'].values():
            assert prepare.sha(v['path']) == v['sha256'], 'Saved input changed'
    for v in plan['known_seed_sources']:
        assert prepare.sha(v['path']) == v['sha256'], 'Known seed metadata changed'
    assert prepare.seed_audit(plan['candidates'], [Path(v['path']) for v in plan['known_seed_sources']]) == plan['seed_audit']
    assert Path(plan['python']).is_file()
    # Verify the private provider copy against the original Git tree. No import
    # of its numerical modules, and no circuit construction, occurs here.
    provider = Path(plan['provider_root'])
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', plan['provider_commit'], 'src'], cwd=ROOT, text=True).splitlines()
    for name in files:
        expected = subprocess.check_output(['git', 'show', plan['provider_commit'] + ':' + name], cwd=ROOT)
        assert (provider / name).read_bytes() == expected, 'Provider SOURCE mismatch: ' + name
    return dict(status='STATIC_READY', input_regeneration=0, scientific_calls=0,
                execution_permission='REQUIRES_SEPARATE_USER_INSTRUCTION', provider_files=len(files))


def bind(plan):
    private, public = Path(plan['private_output_root']), Path(plan['public_results_root'])
    assert private.resolve() != study.PRIVATE.resolve()
    assert public.resolve() != study.PUBLIC.resolve()
    native.PRIVATE, native.PUBLIC = private, public
    study.origin_config = lambda: dict(provider_root=plan['provider_root'], systems=plan['systems'])
    study.install()
    return private, public


def accepted_trial(row, cohort, index, root, expected_log_B=None):
    folder = root / 'cost' / cohort / row['id'] / str(index)
    if (folder / 'STOP.json').exists():
        raise RuntimeError('Saved scientific STOP, no automatic retry: ' + str(folder))
    if not folder.exists():
        return None
    if not (folder / 'result.json').exists():
        raise RuntimeError('Partial trial preserved, no automatic replay: ' + str(folder))
    v = study.read(folder / 'result.json')
    assert (v['candidate'], v['cohort'], v['index'], v['status'], v['conditional']) == (row, cohort, index, 'PASS', False)
    assert sum(v['orders'].values()) == row['q'] * row['R_outer']
    assert all(c['error'] < c['tolerance'] and not c.get('phase_alignment', False) for c in v['checks'])
    if index == 0:
        assert {'low_control','high_control_leakage','independent_occupation_vector',
                'Hadamard_cosine','Hadamard_sine'} <= {x['check'] for x in v['checks']}
    assert set(v['prices']) == {'cosine', 'sine'}
    assert all(type(p[m]) is int and p[m] >= 0 for p in v['prices'].values() for m in native.METRICS)
    if expected_log_B is not None:
        assert abs(v['log_B'] - expected_log_B) < 1e-10, 'Saved mean/native normalization differs'
    return v


def process_jobs(jobs, workers, monitor=None, collect=True):
    """Bounded outstanding work; cancel unstarted tasks after first failure."""
    if not jobs:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        queue = iter(jobs);pending = {}
        last_observation = time.monotonic()
        def submit_one():
            job = next(queue, None)
            if job is not None:
                pending[pool.submit(native.cost_trial, job)] = job
        for _ in range(min(workers, len(jobs))):
            submit_one()
        try:
            while pending:
                complete, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if monitor is not None and time.monotonic() - last_observation >= 30:
                    monitor();last_observation = time.monotonic()
                # Check every completed future before refilling, so an already
                # completed failure cannot be hidden behind a successful job.
                values = [future.result() for future in complete]
                for future in complete:
                    pending.pop(future)
                if collect:
                    results.extend(values)
                for _ in complete:
                    submit_one()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results


def resource_workers(pending, representative_peak=None):
    from pf_common_run_contract import resources
    observation = resources()
    if int(observation['idle_cpu']) < 1:
        raise RuntimeError('No idle CPU observed; do not submit scientific workers')
    workers = max(1, min(pending, int(observation['idle_cpu'])))
    # An empirical admission estimate after representatives; no RSS/AS cap,
    # affinity assignment, or inherited fixed 2 GiB/worker admission rule.
    if representative_peak:
        memory_slots = int(observation['available_ram'] // (2 * representative_peak))
        if memory_slots < 1:
            raise RuntimeError('Insufficient memory for observed representative footprint')
        workers = min(workers, memory_slots)
    return workers, observation


def acquire(plan, cohort, private, public):
    count_field = 'n_exploration' if cohort == plan['cohorts'][0] else 'n_confirmation'
    existing, first, remaining = {}, [], []
    for item in plan['candidates']:
        row = item['candidate'];existing[row['id']] = []
        for index in range(item[count_field]):
            trial = accepted_trial(row, cohort, index, private, item['saved_endpoints'][0]['log_B'])
            if trial is not None:
                existing[row['id']].append(trial)
            else:
                (first if index == 0 else remaining).append((row, cohort, index))
    # Context preparation restores saved H/state and constructs existing native
    # preparations only. No signal/reference/shot function is invoked.
    for system in ('H4', 'H6'):
        prefixes = sorted({v['candidate']['prefix'] for v in plan['candidates'] if v['candidate']['system'] == system})
        native.CONTEXT[system] = native.native_context(system, prefixes)
    workers, observation = resource_workers(max(1, len(first)))
    representatives = process_jobs(first, workers)
    for trial in representatives:
        existing[trial['candidate']['id']].append(trial)
    assert all(any(x['index'] == 0 for x in ts) for ts in existing.values())
    # Validate saved signal binding before submitting the remaining draws.
    for item in plan['candidates']:
        row = item['candidate']
        first_trial = accepted_trial(row, cohort, 0, private, item['saved_endpoints'][0]['log_B'])
        prep = native.CONTEXT[row['system']].preparations[row['prefix']]
        scalar = native.CONTEXT[row['system']].ham.constant + prep.extracted_identity_coefficient
        assert first_trial['preparation_hash'] == prep.preparation_hash and abs(first_trial['scalar']-scalar)<1e-12
    peak = max(v['peak_RSS_bytes'] for ts in existing.values() for v in ts)
    workers, observation = resource_workers(max(1, len(remaining)), peak)
    study.save(private / (cohort + '_resources.json'), dict(workers=workers, observation=observation,
        representative_peak_RSS=peak, per_process_hard_memory_cap=None, compiler_processes=1))
    def monitor():
        from pf_common_run_contract import resources
        observed = resources()
        with (private / (cohort + '_resource_observations.jsonl')).open('a') as stream:
            stream.write(json.dumps(dict(monotonic=time.monotonic(), observation=observed)) + '\n')
        if observed['available_ram'] < peak:
            raise RuntimeError('Available memory below observed representative footprint; stop own submissions')
    process_jobs(remaining, workers, monitor, collect=False)
    # Read accepted complete results, including those already present; never
    # count a cached price as a new independent random draw.
    prices = {}
    for item in plan['candidates']:
        row = item['candidate']
        trials = [accepted_trial(row, cohort, i, private, item['saved_endpoints'][0]['log_B']) for i in range(item[count_field])]
        assert all(v is not None for v in trials)
        prep = native.CONTEXT[row['system']].preparations[row['prefix']]
        scalar = native.CONTEXT[row['system']].ham.constant + prep.extracted_identity_coefficient
        assert all(v['preparation_hash'] == prep.preparation_hash and abs(v['scalar']-scalar)<1e-12 for v in trials)
        prices[row['id']] = native.summarize_trials(row, cohort, trials)
    study.save(public / (cohort + '_costs.json.gz'), prices)
    return prices


def account(plan, by_cohort, public):
    rows = [];comparisons = [];new_exploration = {}
    for cohort, costs in by_cohort.items():
        for item in plan['candidates']:
            identity = item['candidate']['id'];price = costs[identity]
            for ep in item['saved_endpoints']:
                if ep['status'] != 'ELIGIBLE':
                    continue
                value = report.cost_for(ep, price)
                rows.append(dict(item['candidate'], epsilon=ep['epsilon'], N_cos=ep['N_cos'], N_sin=ep['N_sin'], **value))
                if cohort == plan['cohorts'][0]:
                    new_exploration[identity, ep['epsilon']] = Fraction(value['G_exact'])
                if ep['epsilon'] != item['primary_epsilon']:
                    continue
                targets = [item['primary_PR']]
                if identity in prepare.IDS[:2]:
                    targets = [item['primary_PF'], *targets]
                for target in targets:
                    exact_difference = Fraction(value['G_exact']) - Fraction(target['G_exact'])
                    difference = float(exact_difference);z = plan['normal_bonferroni_z']
                    se = value['G_SE'];halfwidth = None if se is None else z * se
                    both_se = None if se is None else math.hypot(se, target.get('G_SE') or 0.)
                    comparison = dict(id=identity, cohort=cohort, system=item['candidate']['system'],
                        T=item['candidate']['T'], epsilon=ep['epsilon'], reference_id=target['id'], reference_G=target['G'],
                        G=value['G'], G_SE=se, difference=difference, difference_exact=str(exact_difference), observed_below=exact_difference < 0,
                        fixed_point_interval_lower=None if halfwidth is None else difference-halfwidth,
                        fixed_point_interval_upper=None if halfwidth is None else difference+halfwidth,
                        two_mean_diagnostic_halfwidth=None if both_se is None else z*both_se,
                        normal_approximation_not_population_guarantee=True)
                    if target['id'] == item['primary_PR']['id'] and item['saved_independent_PR_price']:
                        v = report.cost_for(dict(candidate=target, N_cos=target['N_cos'], N_sin=target['N_sin']), item['saved_independent_PR_price'])
                        comparison.update(saved_independent_reference_G=v['G'], saved_independent_reference_G_SE=v['G_SE'],
                            independent_reference_difference=value['G']-v['G'],
                            independent_reference_secondary_diagnostic_halfwidth=None if se is None or v['G_SE'] is None else z*math.hypot(se,v['G_SE']))
                    comparisons.append(comparison)
    prepare.csv_write('results/candidate_costs.csv', rows)
    prepare.csv_write('results/primary_comparisons.csv', comparisons)
    contrasts = []
    for cohort, costs in by_cohort.items():
        for left, right in plan['R_unit_price_contrasts']:
            values = []
            conditions = []
            for identity in (left, right):
                item = next(x for x in plan['candidates'] if x['candidate']['id'] == identity)
                conditions.append(item['candidate'])
                v = report.cost_for(dict(candidate=item['candidate'], N_cos=1, N_sin=1), costs[identity])
                values.append((v['G']/2, v['G_SE']/2))
            difference = values[1][0] - values[0][0]
            se = math.hypot(values[0][1], values[1][1])
            contrasts.append(dict(cohort=cohort, left=left, right=right,
                left_R=conditions[0]['R_outer'], right_R=conditions[1]['R_outer'],
                left_K=conditions[0]['K'], right_K=conditions[1]['K'],
                contrast='R_ONLY' if conditions[0]['K']==conditions[1]['K'] else 'JOINT_R_AND_K; R_effect_not_identified',
                metric='mean_of_cosine_sine_unit_RZ', difference_right_minus_left=difference,
                independent_candidate_SE=se, interval_lower=difference-plan['normal_bonferroni_z']*se,
                interval_upper=difference+plan['normal_bonferroni_z']*se, cross_candidate_pairing=False))
    prepare.csv_write('results/R_unit_price_contrasts.csv', contrasts)
    # Extended-set point count uses exploration prices only. Confirmation is
    # reported separately and never pooled or substituted into this ranking.
    count = 0
    for b in study.read(study.PUBLIC / 'best_configurations.json'):
        if b['family'] != 'PR' or b['formula_group'] != 'ALL':
            continue
        old_pr = Fraction(b['best']['G_exact']);s,t,e = b['system'],b['T'],b['epsilon']
        pf = next(Fraction(v['best']['G_exact']) for v in study.read(study.PUBLIC / 'best_configurations.json')
                  if (v['system'],v['T'],v['epsilon'],v['family'],v['formula_group']) == (s,t,e,'PF','ALL'))
        added = [new_exploration[i['candidate']['id'],e] for i in plan['candidates']
                 if (i['candidate']['system'],i['candidate']['T']) == (s,t)
                 and (i['candidate']['id'],e) in new_exploration]
        count += min([old_pr, *added]) < pf
    study.save(public / 'extended_point_summary.json', dict(original_saved_count=33,
        extended_exploration_point_count=count, total_cells=40, full_domain_optimality=False,
        independent_confirmation_pooled=False))
    return rows, comparisons


def run(plan):
    verify_static(plan)
    subprocess.check_call(['git','diff','--quiet','HEAD','--','scripts','src','references'],cwd=ROOT)
    # Freeze SOURCE before scientific work, without an approval-file system.
    for name in [*plan['source_hashes'], str((HERE / 'PLAN.json').relative_to(ROOT))]:
        committed = subprocess.check_output(['git','show','HEAD:'+name], cwd=ROOT)
        assert committed == (ROOT / name).read_bytes(), 'Uncommitted execution SOURCE'
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    sys.dont_write_bytecode = True
    private_root = Path(plan['private_output_root'])
    for variable, name in [('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba')]:
        os.environ[variable] = str(private_root / 'runtime_cache' / name)
    private, public = bind(plan)
    private.mkdir(parents=True, exist_ok=True);public.mkdir(parents=True, exist_ok=True)
    with (private / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (public / 'STOP.json').exists():
            raise RuntimeError('Prior STOP is preserved; no automatic retry')
        if (public / 'COMPLETION.json').exists():
            return study.read(public / 'COMPLETION.json')
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
            study.save(public / 'execution_provenance.json', dict(SOURCE=commit,
                plan_sha256=prepare.sha(HERE / 'PLAN.json'), provider_SOURCE=plan['provider_commit'],
                started_at=datetime.now(timezone.utc).isoformat(), input_regeneration=0))
            values = {cohort: acquire(plan, cohort, private, public) for cohort in plan['cohorts']}
            account(plan, values, public)
            study.save(public / 'COMPLETION.json', dict(status='FIXED_COUNTS_COMPLETE_MANDATORY_STOP',
                actual_trajectories=sum(v['n'] for c in values.values() for v in c.values()),
                logical_axis_evaluations=sum(2*v['n'] for c in values.values() for v in c.values()),
                recorded_new_compile_calls=sum(2*v['n']-v['cache_hits'] for c in values.values() for v in c.values()),
                recorded_cache_hits=sum(v['cache_hits'] for c in values.values() for v in c.values()),
                automatic_next_stage=False, all_workers_joined=True, population_tail_bound=None))
        except BaseException as exc:
            study.save(public / 'STOP.json', dict(reason=str(exc), retry=False, partials_preserved=True))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run'])
    args = parser.parse_args();plan = read_plan()
    if args.command == 'check':
        print(json.dumps(verify_static(plan)));return
    # The explicit `run` command is an operational entry, not proof of chat
    # authorization. Codex must receive the separate user instruction first.
    run(plan)


if __name__ == '__main__':
    main()
