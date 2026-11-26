"""Select a host's published execution profile for an exact route.

Operational slots/spool belong to the worker; scientific profile fields are
bound to the controller's assignment and preserved for its whole attempt.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from statistics import median
from pathlib import Path, PurePosixPath
import subprocess


REVIEWED_PR_Q3_ROUTES = {
    ('PR_PAULI', 'pauli_b_common_entry'),
    ('PR_DF', 'prdf_d2_common_entry'),
}
PREFIX_PR_Q3_ROUTES = REVIEWED_PR_Q3_ROUTES | {('PR_DF', 'prdf_d2_split')}

Q3_EXECUTION_ROUTE_KINDS = frozenset((
    'pf_b_pf6', 'pauli_b_ld3', 'pauli_b_ld5', 'pauli_b_common_entry',
    'prdf_d2_split', 'prdf_d2_common_entry', 'hchain_scoped_q3',
))

COMPACT_Q3_RUNTIME_PROBE = '''\
import sqlite3
with sqlite3.connect(':memory:') as db:
    db.execute('CREATE TABLE checkpoints (value INTEGER)')
    db.execute('INSERT INTO checkpoints VALUES (1)')
    if db.execute('SELECT value FROM checkpoints').fetchone() != (1,):
        raise RuntimeError('SQLite checkpoint round trip differs')
'''


def needs_compact_q3_runtime(request):
    return (request['route']['kind'] != 'artificial'
            and (request.get('execution_profile') or {}).get('compact_q3_evidence') is True)


def check_compact_q3_runtime(interpreter, code=COMPACT_Q3_RUNTIME_PROBE):
    result = subprocess.run([interpreter, '-B', '-c', code], capture_output=True, text=True, timeout=15)
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1:] or [f'exit {result.returncode}']
        raise ValueError('Q3 compact evidence requires SQLite in the assigned runtime: ' + detail[-1])
    return result.stdout


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def profile_core(profile):
    keys = ('host_id', 'root', 'interpreter', 'route_views', 'dp_workers',
            'prefix_cache_directory', 'runtime', 'adapter_root', 'adapter_manifest')
    keys += ('runtime_checks', 'reviewed_pr_q3', 'selective_pr_q3')
    return {key: deepcopy(profile[key]) for key in keys if key in profile}


def route_matches(rule, assignment):
    route = assignment['request']['route']
    risk = route.get('horizon_risk_source') or {}
    return (rule.get('route_kind') == route.get('kind')
            and rule.get('view_commit') == route.get('view_commit')
            and rule.get('horizon_risk_commit') == risk.get('commit'))


def validate_adapter_profile(profile):
    root, manifest = profile.get('adapter_root'), profile.get('adapter_manifest')
    if root is None and manifest is None:
        return None
    if not root or not manifest or not manifest.get('files'):
        raise ValueError('Explicit adapter root and fixed manifest are required together')
    root = Path(root).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != manifest['commit']:
        raise ValueError('Adapter HEAD differs from fixed manifest')
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe adapter manifest path')
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Adapter SOURCE hash differs: ' + name)
    return dict(root=str(root), **deepcopy(manifest))


def check_runtime_profile(profile):
    """Run declared source/AST checks in the actual interpreter, without imports or DP."""
    import ast
    jobs = []
    for check in profile.get('runtime_checks', []):
        if check.get('kind') != 'FIXED_FUNCTION_AST':
            raise ValueError('Unknown runtime check')
        view = Path(profile['route_views'][check['view_commit']]).resolve()
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip() != check['view_commit']:
            raise ValueError('Runtime check view HEAD differs')
        hashes = {}
        for name in (check['source_path'], check['guard_path']):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe runtime source path')
            raw = subprocess.check_output(['git', 'show', check['view_commit'] + ':' + name], cwd=view)
            if (view / name).read_bytes() != raw:
                raise ValueError('Fixed runtime source bytes differ: ' + name)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        guard = ast.parse((view / check['guard_path']).read_text())
        expected = [ast.literal_eval(node.value) for node in guard.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == check['guard_constant'] for t in node.targets)]
        if len(expected) != 1:
            raise ValueError('Runtime guard constant is not uniquely defined')
        jobs.append(dict(check, source_file=str(view / check['source_path']),
                         expected_ast_sha256=expected[0], source_files=hashes))
    probe = '''import ast,hashlib,json,platform,sys
rows=[]
for check in json.load(sys.stdin):
    nodes=[n for n in ast.parse(open(check.pop('source_file')).read()).body if isinstance(n,ast.FunctionDef) and n.name==check['function']]
    if len(nodes)!=1: raise ValueError('Runtime function is not uniquely defined')
    actual=hashlib.sha256(ast.dump(nodes[0],include_attributes=False).encode()).hexdigest()
    rows.append(dict(check,actual_ast_sha256=actual,status='PASS' if actual==check['expected_ast_sha256'] else 'FAIL'))
print(json.dumps(dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',runtime=dict(executable=sys.executable,version=sys.version,version_info=list(sys.version_info),platform=platform.platform()),checks=rows)))
'''
    process = subprocess.run([profile['interpreter'], '-B', '-c', probe], input=json.dumps(jobs),
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    report = json.loads(process.stdout)
    runtime = report['runtime']
    if runtime['executable'] != profile['interpreter']:
        actual = Path(runtime['executable']).resolve()
        runtime.update(interpreter_invocation=profile['interpreter'],
                       launcher_sha256=hashlib.sha256(Path(profile['interpreter']).read_bytes()).hexdigest(),
                       actual_executable_realpath=str(actual),
                       actual_binary_sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
    return report


def q3_cpu_class(request, route_kinds=('hchain_scoped_q3',)):
    shots = tuple(request.get('N_X', ())) + tuple(request.get('N_Y', ()))
    kind = request.get('route', {}).get('kind')
    if (kind not in route_kinds or kind not in Q3_EXECUTION_ROUTE_KINDS or not shots
            or any(type(n) is not int or n < 1 for n in shots)):
        return None
    return f"{request['method']}:{max(shots).bit_length()}"


def q3_measurement_host(profile):
    host = profile['host_id']
    return (profile.get('q3_measurement_host_id') or
            (os.environ.get('RPE_Q3_MEASUREMENT_HOST_ID') if host=='local' else None) or host)


def q3_cpu_measurement_key(request, profile):
    """Bind timing to the same fixed execution route and exact shot workload."""
    route = request.get('route') or {}
    if q3_cpu_class(request, Q3_EXECUTION_ROUTE_KINDS) is None:
        return None
    if not all(profile.get(k) for k in ('host_id', 'root', 'interpreter')) or not route.get('view_commit'):
        return None
    execution = {k: v for k, v in (request.get('execution_profile') or {}).items() if k != 'dp_workers'}
    return digest(dict(method=request['method'], route=route['kind'], view=route['view_commit'],
        risk=route.get('horizon_risk_source'), scope=route.get('scope_owner'), binding=route.get('binding'),
        optimizer_source=request.get('optimizer_source'), N_X=request['N_X'], N_Y=request['N_Y'], execution=execution,
        host=q3_measurement_host(profile), root=profile['root'], interpreter=profile['interpreter'],
        adapter_root=profile.get('adapter_root'), adapter=(profile.get('adapter_manifest') or {}).get('commit'),
        route_view=(profile.get('route_views') or {}).get(route['view_commit']), runtime=profile.get('runtime'),
        library_threads=(profile.get('pbs') or {}).get('library_threads', 1)))


def measured_q3_cpu_choice(request, profile, measurements):
    """Choose the observed footprint with the best median reserved CPU-time.

    Compare only completed, independent observations of the same workload.
    No unmeasured count is tried implicitly; the existing contract supplies
    the baseline and any permitted alternatives.
    """
    dp = profile.get('dp_workers')
    baseline = profile.get('task_cpu_count') or dp
    if (type(dp) is not int or dp < 1 or type(baseline) is not int or baseline < dp
            or baseline-dp not in (0,1) or (profile.get('pbs') or {}).get('library_threads', 1) != 1):
        return None
    overhead = baseline-dp
    key = q3_cpu_measurement_key(request, profile)
    if not key:
        return None
    grouped = {}
    for row in measurements.get(key, ()):
        cpus = row.get('task_cpu_count'); seconds = row.get('wall_seconds'); job = row.get('job_id')
        if (row.get('key') != key or not job or row.get('completed') is not True
                or type(cpus) is not int or not 1+overhead <= cpus <= baseline or row.get('dp_workers') != cpus-overhead
                or type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds <= 0):
            continue
        grouped.setdefault(cpus, {})[job] = seconds
    scores = {cpus: (cpus * median(jobs.values()), median(jobs.values()), cpus)
              for cpus,jobs in grouped.items() if len(jobs) >= 2}
    if baseline not in scores:
        return None
    return min(scores, key=lambda cpus: scores[cpus])


def apply_pbs_q3_cpu_policies(distributed, policies):
    """Install the same fixed policy for central accounting and worker selection."""
    pbs = distributed['pbs']
    workers = {row['host_id']: row for row in pbs['workers']}
    if not isinstance(policies, dict) or not set(policies) <= workers.keys():
        raise ValueError('Q3 CPU policy refers to an unconfigured PBS host')
    distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
    for host, policy in policies.items():
        if not isinstance(policy, dict) or set(policy) - {'classes','route_kinds','from_dp_workers','measurements'}:
            raise ValueError('Invalid PBS Q3 CPU policy')
        routes = policy.get('route_kinds', ('hchain_scoped_q3',))
        if not isinstance(routes, (list,tuple)) or any(route not in Q3_EXECUTION_ROUTE_KINDS for route in routes):
            raise ValueError('Q3 CPU policy refers to an unsupported route')
        if not isinstance(policy.get('classes', {}), dict) or not isinstance(policy.get('measurements', {}), dict):
            raise ValueError('Invalid Q3 CPU policy class or observations')
        if any(not isinstance(key, str) or not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
               for key, rows in policy.get('measurements', {}).items()):
            raise ValueError('Invalid Q3 CPU observation')
        profile = workers[host]['profile']
        profile['q3_cpu_policy'] = dict(deepcopy(profile.get('q3_cpu_policy') or {}), **deepcopy(policy))


def apply_q3_cpu_policy(selected, request, policy, *, allow_measurements=True):
    """Only execution parallelism changes. Assigned scientific requests do not."""
    if not policy:
        return selected
    # A profile may explicitly opt in its measured Q3 routes. Existing
    # H-chain class tables never silently alter other evaluator routes.
    key = q3_cpu_class(request, policy.get('route_kinds', ('hchain_scoped_q3',)))
    if key is None:
        return selected
    measured = measured_q3_cpu_choice(request, selected, policy.get('measurements', {})) if allow_measurements else None
    current = max(selected.get('task_cpu_count') or 1, selected.get('dp_workers') or 1)
    if measured is not None:
        # H-chain reserves a coordinator CPU; the existing H4 profile can
        # reserve DP workers directly. Preserve that measured convention.
        if measured < current:
            selected.update(task_cpu_count=measured, dp_workers=measured-(current-selected['dp_workers']))
        return selected
    count = policy.get('classes', {}).get(key)
    if count is None:
        return selected
    if type(count) is not int or not 2 <= count <= current:
        raise ValueError('Q3 CPU policy must select a positive bounded execution footprint')
    selected.update(task_cpu_count=count, dp_workers=count-1)
    return selected


def select_host_profile(base, assignment):
    matches = [rule for rule in base.get('route_profiles', []) if route_matches(rule, assignment)]
    if len(matches) > 1:
        raise ValueError('Ambiguous route-specific host profile')
    source = matches[0]['profile'] if matches else base
    policy = source.get('q3_cpu_policy', base.get('q3_cpu_policy'))
    # The fixed observation index can be large. A durable attempt keeps only
    # its selected profile, not a deep copy of the policy or timing history.
    selected = deepcopy({k: v for k, v in source.items() if k not in ('q3_cpu_policy', 'route_profiles')})
    if selected.get('host_id') != base.get('host_id'):
        raise ValueError('Route-specific profile belongs to another host')
    selected.pop('route_profiles', None)
    request = assignment['request']
    # A saved WORKER_PROFILE already has its chosen resource footprint; the
    # policy is removed so replay of an owned assignment does not reselect it.
    apply_q3_cpu_policy(selected, request, policy, allow_measurements=(base.get('pbs') or {}).get('library_threads', 1) == 1)
    execution = request.get('execution_profile') or {}
    route = request.get('method'), request['route']['kind']
    if route[1] != 'artificial':
        files = selected.get('adapter_manifest', {}).get('files', {})
        selective = execution.get('prefix_selective_checkpoint_rounds') or execution.get('profile_dp_residual') is True
        if selective and (route not in PREFIX_PR_Q3_ROUTES or selected.get('selective_pr_q3') is not True
                or not selected.get('prefix_cache_directory') or not files.get('src/trottertracks/rpe_dp_prefix_reuse.py')):
            raise ValueError('No compatible selective PR Q3 adapter and local cache for this request')
        reviewed = (execution.get('compact_q3_evidence') is True or
                    execution.get('parallel_dp_residual') is True and not selective)
        if reviewed and (route not in REVIEWED_PR_Q3_ROUTES or selected.get('reviewed_pr_q3') is not True
                or not selected.get('prefix_cache_directory') or not files.get('src/trottertracks/rpe_q3_evidence.py')):
            raise ValueError('No compatible reviewed PR Q3 adapter and local cache for this request')
    for key in ('work_dir', 'slots'):
        if key in base:
            selected[key] = deepcopy(base[key])
    bound = assignment.get('host_execution_profile')
    if bound is not None:
        if digest(bound) != assignment.get('host_execution_profile_sha256'):
            raise ValueError('Assignment host execution profile hash differs')
        if profile_core(selected) != bound:
            raise ValueError('Assignment requires a different route-specific host profile; common fallback is forbidden')
    return selected
