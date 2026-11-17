"""One R08 source handover after its original controller releases ownership.

No scientific leaf is executed here. The only scientific entry is the unchanged
run_rpe_schedule_search CLI, started once after metadata checks and input publish.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
IDENTITY = ['-c', 'user.name=HIROMU1015', '-c', 'user.email=u411612g@ecs.osaka-u.ac.jp']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def stamp(status, **fields):
    return dict(status=status, UTC=datetime.now(timezone.utc).isoformat(), **fields)


def request_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()[:20]


def reconcile(saved, completed, provenance):
    """Retain original keys, holds and all old state; import terminal metadata."""
    state = deepcopy(saved)
    original_stop = state.pop('stop', None)
    state['resumed_from'] = dict(provenance, previous_stop=original_stop,
                                 scientific_retry=False)
    records = completed['records']
    unresolved = []
    for point in state['points']:
        request = point.get('completion_request')
        if request is None:
            continue
        identifier = request_key(request)
        record = records.get(identifier)
        if record is None:
            exact = json.dumps(request, sort_keys=True, default=str)
            matches = [(k, r) for k, r in records.items()
                       if json.dumps(r.get('request'), sort_keys=True, default=str) == exact]
            if len(matches) > 1:
                point.update(status='HELD', reason='AMBIGUOUS_PRIOR_COMPLETION_NO_DUPLICATE')
                continue
            if matches:
                identifier, record = matches[0]
        if record is None:
            continue
        point['completion'] = dict(status=record['status'], key=identifier,
                                   reason=record.get('reason'), reused_from=provenance)
        if record['status'] == 'READY_FOR_SEARCH':
            if point.get('status') != 'EVALUATED':
                point.update(status='REGISTERED', configuration_id=record['configuration_id'])
                point.pop('reason', None)
        elif record['status'] == 'SELECTED':
            # A dead owner's partial leaf is never a fresh attempt.
            point.update(status='HELD', reason='PRIOR_SELECTED_STAGE_REQUIRES_REVIEW_NO_RETRY')
            unresolved.append(identifier)
        elif record['status'].startswith(('HELD', 'STOP')):
            point.update(status='HELD', reason=record.get('reason', record['status']))
        elif record['status'].startswith('WAITING'):
            point.update(status='WAITING_INPUTS', reason=record.get('reason', record['status']))
    imported = deepcopy(completed)
    imported['previous_source'] = completed['source']
    for record in imported['records'].values():
        record['reused_from'] = dict(provenance, scientific_source=completed['source'])
        if record['status'] == 'SELECTED':
            record.update(status='HELD_WITH_REASON',
                          reason='PRIOR_SELECTED_STAGE_REQUIRES_REVIEW_NO_RETRY')
    return state, imported, unresolved


def validate_ready_points(driver, records):
    """Use the driver's existing per-candidate metadata hold policy unchanged."""
    held = []
    for point in driver.state['points']:
        record = records.get(point.get('completion', {}).get('key'))
        if record is None or record['status'] != 'READY_FOR_SEARCH':
            continue
        try:
            driver.check_completed_configuration(point, record)
        except (ValueError, KeyError, TypeError) as error:
            point.update(status='HELD', reason='COMPLETED_CONFIGURATION_MISMATCH:' + str(error))
            held.append(dict(key=record['key'], family=point['family'], kappa=point['kappa'],
                             reason=point['reason']))
    return held


def q3_preflight(old_run):
    """Receipt/status inspection only. Do not evaluate MSE or re-fit prices."""
    counts = Counter()
    unresolved = []
    from trottertracks import rpe_method_optimizer as O
    for request in sorted((old_run / 'jobs').glob('*/*/REQUEST.json')):
        job = request.parent
        launched = (job / 'LAUNCH.json').exists() or (job / 'STARTING.json').exists()
        outcome = job / 'OUTCOME.json'
        if not outcome.exists():
            if launched:
                unresolved.append(dict(job=str(job), reason='OWNED_Q3_WITHOUT_OUTCOME'))
            continue
        result = read(outcome)
        status = result.get('status')
        counts[status] += 1
        if status not in (O.PASS, O.FAIL, 'INCONCLUSIVE'):
            # Existing normal executor validates and scopes a returned UNKNOWN.
            # Do not start preparation before that review on this handover.
            unresolved.append(dict(job=str(job), reason='PRIOR_Q3_NOT_VALID_TERMINAL', status=status))
        returned = job / 'PROCESS_RETURN.json'
        if launched and (not returned.exists() or read(returned).get('returncode') != 0):
            unresolved.append(dict(job=str(job), reason='PRIOR_Q3_PROCESS_RETURN_NOT_CLEAN'))
    return dict(status_counts=dict(counts), unresolved=unresolved)


def import_inputs(plan, old_run, study):
    from trottertracks import rpe_method_optimizer as O, rpe_kappa_schedule as KS, rpe_schedule_search as W
    import run_rpe_kappa_neighbourhood as K
    from run_rpe_schedule_search import WidthDriver
    from types import SimpleNamespace
    run = study / 'run02'
    if (run / 'SOURCE.json').exists():
        raise RuntimeError('An epoch already exists: use its published resume command, never reimport over it')
    old_root = Path(plan['old_root'])
    old_study = old_run.parent
    for filename, expected in plan.get('old_input_sha256', {}).items():
        if digest(old_study / filename) != expected:
            raise RuntimeError('Old fixed input changed: ' + filename)
    git('diff', '--exit-code', plan['old_code_source'], '--', 'src', 'scripts', cwd=old_root)
    git('diff', '--exit-code', plan['new_code_source'], '--', 'src', 'scripts')
    original = {f: digest(old_run / f) for f in plan['snapshot_files']}
    old_head = git('rev-parse', 'HEAD', cwd=old_root)
    logs = (old_run / 'CONTROLLER.log').read_text()
    (study / 'OLD_CONTROLLER_EXCEPTION.txt').write_text(logs)
    q3 = q3_preflight(old_run)
    # Metadata corruption, ambiguous ownership or failed numerical Q3 is a STOP.
    if q3['unresolved']:
        write(study / 'PRELAUNCH_REVIEW.json', q3)
        raise RuntimeError('Prior Q3 needs review; no scientific launch: PRELAUNCH_REVIEW.json')
    provenance = dict(run_directory=str(old_run), worktree_head=old_head,
                      snapshots_sha256=original, code_source=plan['old_code_source'])
    old_state = read(old_run / 'KAPPA_STATE.json')
    completed = read(old_run / 'completion/STATE.json')
    state, completion, held = reconcile(old_state, completed, provenance)
    remaining = plan['original_cycle_ceiling'] - state['cycle']
    if remaining <= 0:
        raise RuntimeError('Original cycle ceiling reached; not enlarged by technical recovery')
    problem = read(old_study / 'PROBLEM.json')
    spec = read(old_study / 'ROUND.json')
    registry = str(run / 'PRICE_RULES_REGISTRY.json')
    spec['base_commit'] = plan['new_code_source']
    spec['candidate_completion']['source'] = plan['new_code_source']
    spec['candidate_completion']['signal_epoch'] = 'multiwidth_r08_recovery_20261117'
    spec['candidate_completion']['price_registry_file'] = registry
    spec['candidate_completion'].pop('resume_preparation_from', None)
    spec['price_rules_overrides']['registry_file'] = registry
    prior = spec['executor'].setdefault('prior_run_dirs', [])
    if str(old_run) not in prior:
        prior.append(str(old_run))
    # Reuse the existing numerical prefix cache and worker profile unchanged.
    spec['continues'] = dict(prior_run=str(old_run), preserved_stop=old_state.get('stop'),
                             user_directed_continuation=True, technical_source_epoch=True)
    spec['price_batch']['execution_branch'] = git('branch', '--show-current')
    completion['source'] = plan['new_code_source']
    write(study / 'PROBLEM.json', problem)
    write(study / 'ROUND.json', spec)
    write(run / 'KAPPA_STATE.json', state)
    write(run / 'completion/STATE.json', completion)
    for filename in ('PRICE_RULES_REGISTRY.json', 'completion/PRICE_RULES_REGISTRY.json',
                     'completion/STEP_PRICE_RULES.json'):
        write(run / filename, read(old_run / filename))
    effective = K.CLI.price_overrides(K.CLI.scoped(deepcopy(problem), spec.get('method_overrides')), spec)
    widths = W.requested_widths(effective, spec)
    methods = list(widths)
    for method, values in widths.items():
        effective['methods'][method]['requested_delta_pf_values'] = [O.enc(KS._frac(w['raw'])) for w in values]
    fixed = dict(schema='RPE_REQUESTED_WIDTH_FIXED_INPUTS_V1', problem=effective,
                 requested_methods=methods, requested_widths=widths,
                 round_scope={k: spec.get(k) for k in ('method_overrides', 'search_overrides', 'width_families', 'new_families', 'kappa_families')},
                 completion={k: v for k, v in spec['candidate_completion'].items()
                             if k not in ('price_evidence_results', 'plan_only')})
    write(run / 'FIXED_INPUTS.json', fixed)
    args = SimpleNamespace(problem=study / 'PROBLEM.json', round=study / 'ROUND.json',
                           run_dir=run, batch=plan['batch'], max_cycles=remaining, plan_only=True)
    driver = WidthDriver(args)
    driver.verify_metadata_inputs()
    configuration_holds = validate_ready_points(driver, completion['records'])
    state = driver.state
    write(run / 'KAPPA_STATE.json', state)
    if any(digest(old_run / f) != h for f, h in original.items()):
        raise RuntimeError('Old state changed after ownership release; no launch')
    source = dict(scientific_code_source=plan['new_code_source'], previous_epoch=provenance,
                  remaining_cycles=remaining, original_cycle_ceiling=plan['original_cycle_ceiling'],
                  preserved_held_keys=held, configuration_holds=configuration_holds,
                  prior_Q3=q3, old_exception='OLD_CONTROLLER_EXCEPTION.txt',
                  artificial_tests=plan['artificial_tests'], new_scientific_calls_before_launch=0,
                  scientific_retry=False, new_scientific_conditions=False,
                  fixed_input_sha256={f: digest(study / f) for f in ('PROBLEM.json', 'ROUND.json')})
    write(run / 'SOURCE.json', source)
    write(study / 'PRELAUNCH_ACCEPTANCE.json', dict(source,
          imported_completion_counts=dict(Counter(r['status'] for r in completion['records'].values())),
          cached_families=len(state['families']), points=len(state['points']),
          fixed_metadata_gate='PASS_WITH_CONFIGURATION_HOLDS' if configuration_holds else 'PASS',
          old_metadata_unchanged=True))
    return run, remaining


def execute(plan, study):
    old_run = Path(plan['old_run'])
    status = study / 'HANDOVER_STATUS.json'
    write(status, stamp('WAITING_FOR_OLD_OWNER', old_run=str(old_run),
                        new_source=plan['new_code_source'], automatic_scientific_retry=False))
    # The old wrapper holds this lock until its controller and child drain end.
    # No signal or modification of the running old SOURCE is used.
    with (old_run / 'CONTROLLER.lock').open('a') as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX)
        write(status, stamp('OLD_OWNER_RELEASED_PREFLIGHT', old_run=str(old_run)))
        records = [json.loads(l) for l in (old_run / 'CONTROLLER_INVOCATIONS.jsonl').read_text().splitlines()]
        if not records or records[-1].get('event') != 'CONTROLLER_RETURN':
            raise RuntimeError('No recorded old controller return; preserve and review ownership')
        run = study / 'run02'
        if not (run / 'SOURCE.json').exists():
            # Append a fixed terminal metadata locator to our old branch; bytes,
            # old SOURCE, scientific results and STOP remain unchanged.
            old_root = Path(plan['old_root'])
            if git('branch', '--show-current', cwd=old_root) != plan['old_branch']:
                raise RuntimeError('Old branch ownership changed; no automatic edits')
            git('diff', '--exit-code', plan['old_code_source'], '--', 'src', 'scripts', cwd=old_root)
            old_paths = [str((old_run / f).relative_to(old_root)) for f in plan['snapshot_files']]
            old_paths += [str((old_run / f).relative_to(old_root)) for f in
                          ('CONTROLLER.log', 'CONTROLLER_INVOCATIONS.jsonl', 'ROUND_STAGES.jsonl',
                           'DRIVER_PROGRESS.jsonl', 'PROGRESS.jsonl', 'completion/PROGRESS.jsonl')]
            git(*IDENTITY, 'add', '--', *old_paths, cwd=old_root)
            if git('diff', 'HEAD', '--name-only', '--', *old_paths, cwd=old_root):
                git(*IDENTITY, 'commit', '--only', '-m', 'Preserve terminal R08 metadata before technical source handover', '--', *old_paths, cwd=old_root)
                git('push', 'origin', plan['old_branch'], cwd=old_root)
            run, remaining = import_inputs(plan, old_run, study)
            git(*IDENTITY, 'add', '--', str(study.relative_to(ROOT)))
            git(*IDENTITY, 'commit', '--only', '-m', 'Fix R08 recovery inputs after old owner drain; preserve scientific evidence', '--', str(study.relative_to(ROOT)))
            git('push', 'origin', git('branch', '--show-current'))
        else:
            source = read(run / 'SOURCE.json')
            if source['scientific_code_source'] != plan['new_code_source']:
                raise RuntimeError('Published epoch has another SOURCE; preserve and review')
            git('diff', '--exit-code', plan['new_code_source'], '--', 'src', 'scripts')
            for filename, expected in source['fixed_input_sha256'].items():
                if digest(study / filename) != expected:
                    raise RuntimeError('Fixed recovery input changed: ' + filename)
            remaining = plan['original_cycle_ceiling'] - read(run / 'KAPPA_STATE.json')['cycle']
            if remaining <= 0:
                raise RuntimeError('Original cycle ceiling reached; not enlarged on resume')
        commit = git('rev-parse', 'HEAD')
        remote = git('ls-remote', 'origin', 'refs/heads/' + git('branch', '--show-current')).split()[0]
        if remote != commit:
            raise RuntimeError('Fixed input remote mismatch; no launch')
        command = [plan['controller_python'], '-B', 'scripts/run_rpe_schedule_search.py',
                   '--problem', str(study / 'PROBLEM.json'), '--round', str(study / 'ROUND.json'),
                   '--run-dir', str(run), '--batch', str(plan['batch']), '--max-cycles', str(remaining)]
        write(status, stamp('FIXED_SOURCE_RUNNING', input_commit=commit, command=command,
                            old_return=records[-1], run_directory=str(run)))
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))
        with (run / 'CONTROLLER.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(run / 'CONTROLLER_PID.json', stamp('CONTROLLER_START', pid=process.pid,
                  input_commit=commit, command=command, source=plan['new_code_source']))
            code = process.wait()
        write(status, stamp('CONTROLLER_RETURN', returncode=code, automatic_retry=False,
                            run_directory=str(run), stop=read(run / 'KAPPA_STATE.json').get('stop')))
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    study = (ROOT / plan['study']).resolve()
    try:
        return execute(plan, study)
    except Exception as error:
        import traceback
        trace = traceback.format_exc()
        write(study / 'HANDOVER_STATUS.json', stamp('HANDOVER_STOP',
              error_type=type(error).__name__, message=str(error), traceback=trace,
              automatic_retry=False, new_scientific_launch_after_error=False))
        print(trace, flush=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
