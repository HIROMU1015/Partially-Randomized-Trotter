"""Read-only metadata admission for saved-value evaluation; no science execution.

Only JSON metadata, hashes and Git objects are read. No circuit, numerical-array,
sampling, shot, statistical or accounting module is imported or executed. A PASS
of the metadata checks cannot grant admission to an unadopted accounting policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess


class AdmissionError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise AdmissionError(code)


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=None)
def file_identity(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns), 'INPUT_CHANGED_DURING_HASH')
    return after.st_size, digest.hexdigest()


def sha(path):
    return file_identity(str(Path(path).resolve()))[1]


def git_bytes(repo, ref):
    return subprocess.check_output(['git', 'show', ref], cwd=repo)


def source_binding(spec, cfg):
    require(cfg['SOURCE'] == spec['SOURCE'], 'FIXED_SCIENTIFIC_SOURCE')
    actual = subprocess.check_output(
        ['git', 'rev-parse', spec['SOURCE'] + '^{commit}'],
        cwd=spec['repository'], text=True).strip()
    require(actual == spec['SOURCE'], 'SOURCE_GIT_OBJECT')
    subprocess.run(['git', 'merge-base', '--is-ancestor', spec['SOURCE'],
                    spec['public_commit']], cwd=spec['repository'], check=True)
    refs = [row for row in cfg['sealed_files']
            if row['role'].startswith(spec['source_role_prefix'])]
    require(bool(refs), 'SCIENTIFIC_SOURCE_CLOSURE_PRESENT')
    # One Git process for all source objects; no scientific imports.
    expressions = [spec['SOURCE'] + ':' + row['role'].split(':', 1)[1]
                   for row in refs]
    data = subprocess.check_output(['git', 'cat-file', '--batch'],
                                   input=('\n'.join(expressions) + '\n').encode(),
                                   cwd=spec['repository'])
    offset = 0
    for row in refs:
        end = data.index(b'\n', offset)
        header = data[offset:end].split()
        require(len(header) == 3 and header[1] == b'blob', 'SOURCE_BLOB')
        size = int(header[2])
        content = data[end + 1:end + 1 + size]
        require(hashlib.sha256(content).hexdigest() == row['sha256'],
                'GIT_SOURCE_SEAL_BYTES')
        offset = end + 2 + size
    require(offset == len(data), 'SOURCE_BATCH_CLOSED')
    return len(refs)


def cohort_metadata(tasks, system):
    require(len({t['id'] for t in tasks}) == len(tasks), 'UNIQUE_TRAJECTORIES')
    require(all(t['stage'] == system + '_PR_EXPLORATION' and
                t['axes'] == ['cosine', 'sine'] and
                t['candidate']['system'] == system and
                t['candidate']['K'] == 4 and
                t['candidate']['r_policy'] == 'ABS_TIME_PER_OUTER' and
                t['candidate']['r_value'] == 32 for t in tasks),
            'EXPLORATION_ONLY_PAIRED_OUTER32')
    candidates = {t['candidate']['id']: t['candidate'] for t in tasks}
    require(len(candidates) == 12, 'FIXED_TWELVE_GROUPS')
    require({(c['formula'], c['q']) for c in candidates.values()} ==
            {(formula, q) for formula in ('2nd', '4th', '4th(m5_best)')
             for q in (1, 2, 4, 8)}, 'FIXED_FORMULA_Q_DOMAIN')
    return len(candidates)


def verify_system(spec):
    cfg = read(spec['config'])
    root = Path(cfg['run_root'])
    require(read(cfg['plan_path']) == cfg['plan'] and
            sha(cfg['plan_path']) == cfg['plan_sha256'], 'SEALED_PLAN')
    source_objects = source_binding(spec, cfg)
    for item in cfg['sealed_files']:
        require(file_identity(str(Path(item['path']).resolve())) ==
                (item['bytes'], item['sha256']), 'INPUT_RUNTIME_SOURCE_SEAL')
    groups = cohort_metadata(cfg['tasks'], spec['system'])
    audit_path = root / 'end_audit.json'
    audit = read(audit_path)
    completion = read(root / 'completion_witness.json')
    exit_witness = read(cfg['supervisor']['exit'])
    require(audit['status'] == 'ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP'
            and audit['SOURCE'] == cfg['SOURCE'] and
            audit['plan_sha256'] == cfg['plan_sha256'], 'ACQUISITION_AUDIT')
    require(audit['errors'] == completion['errors'] == {} and
            audit['failed_or_incomplete_trajectories'] == 0 and
            audit['completed_trajectories'] == len(cfg['tasks']) ==
            spec['trajectories'] and
            audit['accepted_compile_axes'] == spec['axes'], 'COMPLETE_SCOPE')
    require(audit['completion_sha256'] == sha(root / 'completion_witness.json'),
            'COMPLETION_HASH')
    require(exit_witness['native_exit'] == 0 and exit_witness['reaped'] and
            all(w['exit'] == 0 and w['reaped']
                for w in completion['workers'].values()), 'EXIT_AND_REAP')
    require(audit['byte_SHA_roundtrip_PASS'] is True, 'PRIOR_BYTE_AUDIT')
    expected = {row['id']: row['result_sha256'] for row in audit['results']}
    require(set(expected) == {t['id'] for t in cfg['tasks']}, 'RESULT_SET_SCOPE')
    for task in cfg['tasks']:
        require(sha(Path(cfg['output']) / task['id'] / 'result.json') ==
                expected[task['id']], 'SAVED_RAW_RESULT_HASH')
    public_blob = git_bytes(spec['repository'], spec['public_commit'] + ':' +
                            spec['public_summary'])
    public = json.loads(public_blob)
    public_row = next(row for row in public['stages']
                      if row['stage'] == cfg['stage'])
    require(public['SOURCE'] == cfg['SOURCE'] and
            public_row['end_audit_sha256'] == sha(audit_path),
            'FIXED_PUBLIC_PRIVATE_AUDIT_JOIN')
    if spec['system'] == 'H2':
        signal = read(cfg['signal_path'])
        require(signal['formal_shot_policy_adopted'] is False,
                'H2_SAVED_SIGNAL_ADOPTION_STATUS')
        previous_adoption = signal['formal_shot_policy_adopted']
    else:
        previous_adoption = cfg['decisions']['formal_shot_policy_adopted']
    return dict(system=spec['system'], metadata_status='PASS',
                public_commit=spec['public_commit'], SOURCE=cfg['SOURCE'],
                plan_sha256=cfg['plan_sha256'], groups=groups,
                trajectories=audit['completed_trajectories'],
                axes=audit['accepted_compile_axes'],
                end_audit_sha256=sha(audit_path),
                source_objects_checked=source_objects,
                sealed_files_checked=len(cfg['sealed_files']),
                raw_result_JSON_SHA_checked=len(expected),
                archive_reverification_this_preflight=False,
                prior_archive_roundtrip_audit='PASS',
                cohort='EXPLORATION', prior_formal_shot_policy_adopted=previous_adoption,
                population_guarantee=False)


def require_accounting_policy(policy):
    require(isinstance(policy, dict) and policy.get('adopted') is True,
            'SHOT_POLICY_NOT_ADOPTED_FOR_OUTER32')
    fields = ('adoption_evidence', 'applicable_series', 'epsilon_grid',
              'axis_allocation', 'alpha_axis', 'shot_integer_SOURCE',
              'bias_policy', 'allowance_policy', 'normalization_policy',
              'headroom_guard', 'integer_overflow_policy')
    require(all(policy.get(field) is not None for field in fields),
            'INCOMPLETE_ACCOUNTING_POLICY_BINDING')
    require(policy['applicable_series'] == 'H246_HIGHER_PF_OUTER32_EXPLORATION',
            'ACCOUNTING_POLICY_SERIES_MISMATCH')


def run(context):
    specs = context['systems']
    require({s['system'] for s in specs} == {'H2', 'H4', 'H6'}, 'THREE_SYSTEMS')
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        systems = list(pool.map(verify_system, specs))
    require_accounting_policy(context.get('accounting_policy_adoption'))
    # This tool never executes science, even if a future policy binding exists.
    return systems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    context = read(args.context)
    specs = context['systems']
    require({s['system'] for s in specs} == {'H2', 'H4', 'H6'}, 'THREE_SYSTEMS')
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        systems = list(pool.map(verify_system, specs))
    reason = None
    try:
        require_accounting_policy(context.get('accounting_policy_adoption'))
    except AdmissionError as exc:
        reason = str(exc)
    result = dict(status='STOP_BEFORE_SCIENCE' if reason else 'METADATA_ONLY_PASS',
                  reason=reason, systems=systems,
                  scientific_execution_authorized_by_user=True,
                  admission_passed=reason is None,
                  science_grant_issued=False, science_marker_consumed=False,
                  science_ledger_created=False, new_sampling=0, new_signal=0,
                  new_simulation=0, new_circuit=0, new_compile=0,
                  statistical_evaluations=0, shot_evaluations=0,
                  total_cost_evaluations=0, rankings=0,
                  independent_numerical_crosscheck='NOT_EXECUTED_PRE_SCIENCE_STOP',
                  AD03='DEFERRED', mandatory_STOP=True)
    output = Path(args.output)
    require(not output.exists(), 'NEW_PREFLIGHT_RECEIPT_ONLY')
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return 2 if reason else 0


if __name__ == '__main__':
    raise SystemExit(main())
