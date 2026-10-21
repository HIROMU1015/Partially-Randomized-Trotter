"""One-shot saved JSON/CSV analysis. No sampling, circuit or simulator imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import h246_saved_cost_evaluation as science

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ('scripts/h246_saved_cost_evaluation.py',
                'scripts/run_h246_saved_cost_evaluation.py',
                'scripts/prepare_h246_saved_cost_evaluation.py',
                'scripts/d0_frozen_shot_rule_v1.py',
                'scripts/audit_h246_saved_evaluation_preflight.py',
                'tests/test_h246_saved_cost_evaluation.py')
SCOPE = dict(candidate_evaluations=100, PR_groups=36, deterministic_m5_candidates=64,
             exploration_trajectories=1925, candidate_precision_mode_records=1000,
             max_frozen_pair_shot_calls=1000, summary_records=30,
             max_high_order_comparisons=60, new_sampling=0, new_signal=0,
             new_simulation=0, new_circuit=0, new_compile=0,
             new_fit=0, independent_confirmation_sampling=0, AD03='DEFERRED')
require = science.require


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def verify(cfg):
    require(cfg.get('authorized') is True, 'EXPLICIT_NUMERICAL_EXECUTION_REQUIRED')
    grant, plan = cfg.get('grant', {}), cfg['plan']
    require(grant.get('valid') is True and grant.get('retry') is False and
            grant.get('scope') == SCOPE and grant.get('SOURCE') == cfg['SOURCE'] and
            grant.get('plan_sha256') == cfg['plan_sha256'] and
            grant.get('run_root') == cfg['run_root'], 'ONE_SHOT_FIXED_GRANT')
    require(read(cfg['plan_path']) == plan and sha(cfg['plan_path']) == cfg['plan_sha256'],
            'PLAN_SHA')
    require(plan['scope'] == SCOPE and plan['SOURCE'] == cfg['SOURCE'] and
            plan['run_root'] == cfg['run_root'], 'FIXED_SCOPE_SOURCE')
    require(plan['python_executable'] == sys.executable and plan['python_version'] == sys.version,
            'FIXED_ANALYSIS_RUNTIME')
    policy = plan['adoption']
    require(policy['adopted'] is True and policy['proposal_id'] ==
            'H246_OUTER32_SAVED_ACCOUNTING_ADOPTION_PROPOSAL_V1' and
            policy['applicable_series'] == 'H246_HIGHER_PF_OUTER32_EXPLORATION' and
            policy['epsilon_grid'] == list(science.EPS) and
            policy['modes'] == list(science.MODES) and policy['alpha_axis'] == .025 and
            policy['allowance_factor'] == 1 and policy['AD03'] == 'DEFERRED',
            'EXACT_CONDITIONAL_POLICY_ADOPTION')
    require(len(plan['groups']) == 100 and len({g['record']['candidate']['id'] for g in plan['groups']}) == 100,
            'FIXED100_CANDIDATES')
    require(sum(len(g.get('tasks', [])) for g in plan['groups']) == 1925 and
            sum(g['record']['candidate']['method'] == 'PR' for g in plan['groups']) == 36,
            'ALL_EXPLORATION_SAMPLES_NO_POOLING')
    for s in plan['sealed_files']:
        p = Path(s['path'])
        require(p.stat().st_size == s['bytes'] and sha(p) == s['sha256'],
                'SEALED_INPUT_SOURCE_RUNTIME:'+s['role'])
    for rel in SOURCE_PATHS:
        require((ROOT/rel).read_bytes() == subprocess.check_output(
            ['git', 'show', cfg['SOURCE']+':'+rel], cwd=ROOT), 'COMMITTED_SOURCE:'+rel)


def one_group(cfg, group):
    root = Path(cfg['run_root'])
    record = group['record']
    candidate = record['candidate']
    save(root/'ledger'/(candidate['id']+'.json'), dict(candidate_id=candidate['id'],
         state='RESERVED_NONREFUNDABLE', candidate_precision_mode_records=10,
         max_frozen_pair_shot_calls=10, SOURCE=cfg['SOURCE'],
         plan_sha256=cfg['plan_sha256'], new_sampling=0, new_compile=0))
    prices, labels = [], None
    if candidate['method'] == 'PR':
        labels = []
        registered = set()
        for index in group['tasks']:
            require(index['task']['id'] not in registered, 'UNIQUE_SAMPLE')
            registered.add(index['task']['id'])
            require(sha(index['result_path']) == index['sha256'], 'ACCEPTED_TRAJECTORY_SHA')
            r = read(index['result_path'])
            price, label = science.load_price(r, index['task'], group['binding'],
                record['preparation_hash'], record['signal']['log_B'],
                group['acquisition_SOURCE'], group['acquisition_plan_sha256'])
            prices.append(price)
            labels.append(label)
    elif group['direct_prices'] is not None:
        prices = [tuple(group['direct_prices'])]
    value = science.evaluate_candidate(record, prices, labels)
    value['cost_provenance'] = group.get('cost_provenance')
    save(root/'groups'/(candidate['id']+'.json'), value)
    # Per-trajectory cost inputs stay private. No seeds are exported.
    save(root/'trajectory_prices'/(candidate['id']+'.json'),
         dict(paired_RZ=prices, strata=labels, pooling=False))
    return value


def execute(cfg):
    verify(cfg)
    root = Path(cfg['run_root'])
    require(not root.exists(), 'UNUSED_ONE_SHOT_ROOT')
    root.mkdir(parents=True, exist_ok=False)
    save(root/'CONSUMED.json', dict(SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'],
         grant=cfg['grant'], state='CONSUMED_NO_RETRY_NO_RESUME'))
    started = time.monotonic()
    try:
        def stopped(sig, frame):
            raise RuntimeError('OWN_ANALYSIS_SIGNAL_'+str(sig))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stopped)
        # All groups are independent, immutable JSON jobs; no arbitrary low cap.
        workers = min(len(cfg['plan']['groups']), len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda g: one_group(cfg, g), cfg['plan']['groups']))
        require(len(results) == 100 and sum(len(g['rows']) for g in results) == 1000,
                'EXACT_EVALUATION_BUDGET')
        expected = {g['record']['candidate']['id'] for g in cfg['plan']['groups']}
        require({p.stem for p in (root/'ledger').glob('*.json')} == expected, 'EXACT_LEDGER')
        summary = science.summarize(results)
        require(len(summary) == 30, 'EXACT_SUMMARY_BUDGET')
        verify(cfg)
        result = dict(status='SAVED_VALUE_EVALUATION_PASS_MANDATORY_STOP',
            SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'],
            adoption=cfg['plan']['adoption'], scope=SCOPE, groups=results, summary=summary,
            group_threads=workers, wall_seconds=time.monotonic()-started,
            exact_moment_and_paired_covariance_checks='PASS',
            centered_float_and_stdlib_checks='PASS', Decimal_shot_integer_checks='PASS',
            shared_transcendentals_in_independent_shot_check='BINARY64_LIBM_SQRT_EXP_LOG',
            ratio_decomposition_exact_check='PASS', population_guarantee=False,
            selection_bias_corrected=False, independent_confirmation=False,
            cohort_pooling=False, AD03='DEFERRED', retry=False, mandatory_STOP=True)
        save(root/'evaluation.json', result)
        save(root/'completion.json', dict(status=result['status'], SOURCE=cfg['SOURCE'],
            plan_sha256=cfg['plan_sha256'], result_sha256=sha(root/'evaluation.json'),
            candidate_evaluations=100, row_evaluations=1000, summary_records=30,
            retry=False, mandatory_STOP=True))
        return 0
    except BaseException as exc:
        save(root/'STOP.json', dict(reason=str(exc), retry=False, resume=False,
             SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'], mandatory_STOP=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    return execute(read(args.config))


if __name__ == '__main__':
    raise SystemExit(main())
