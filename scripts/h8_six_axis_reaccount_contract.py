"""Pinned evidence, read-only input joins and one-shot authorization.

No original science runner is invoked. Compiler/profile hashes describe cost
provenance; this saved-scalar stage uses only the separately pinned stdlib runtime.
"""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/trotterlib'))
import h8_six_axis_reaccount as n
import h8_phase_c_accounting as legacy

require, digest = n.require, n.digest
SCOPE = 'H8_SELECTED_SIX_SAVED_ACCOUNTING_BOTH_BIAS_MODES_V1'
INPUTS = 'references/h8_six_axis_reaccount_inputs_v1.json'
SOURCES = ('src/trotterlib/h8_six_axis_reaccount.py',
           'src/trotterlib/h8_phase_c_accounting.py',
           'scripts/h8_six_axis_reaccount_contract.py',
           'scripts/run_h8_six_axis_reaccount.py',
           'scripts/witness_h8_six_axis_reaccount.py',
           'tests/test_h8_six_axis_reaccount.py', INPUTS)
PYTHON = '/usr/bin/python3.12'
GIT = ['git', '--no-optional-locks', '-c', 'index.threads=1',
       '-c', 'core.preloadIndex=false', '-c', 'core.fsmonitor=false',
       '-c', 'maintenance.auto=false', '-c', 'gc.auto=0']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def loads(data):
    def pairs(values):
        result = {}
        for k, v in values:
            require(k not in result, 'DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    def number(text):
        value = float(text)
        require(math.isfinite(value), 'NONFINITE_JSON')
        return value
    def constant(text):
        raise ValueError('NONFINITE_JSON:' + text)
    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def read(path):
    p = Path(path)
    require(not p.is_symlink() and p.is_file() and p.stat().st_size < 32 * 2**20, 'BOUNDED_REGULAR_JSON')
    return loads(p.read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')


def home(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and p.is_relative_to('/home/AbeHiromu'), 'CANONICAL_HOME_PATH')
    return p


def plan():
    return dict(schema='H8_SIX_AXIS_REACCOUNT_PLAN_V1', scope=SCOPE,
        approved=False, runtime_authorization=False, execution_sealed=False,
        science=dict(system='linear H8', spacing_A=1.0, basis='STO-3G', saved_DF_rank=28,
                     spin_modes=16, sector=4900, T=.8, q=[1, 2, 4, 8]),
        epsilons=list(n.EPSILONS), allowance_factors=list(n.FACTORS), alpha_axis=.025, P=0,
        metric='SAVED_SHOT_WEIGHTED_FULL_WRAPPER_RZ_STATE_PREPARATION_EXCLUDED',
        bias_namespaces=list(n.MODES), deterministic=184, old=32, PR_union=6, PR_n=32,
        replacement_RZ=n.REPLACEMENTS,
        cost_priority=['SELECTED_SIX_MEASURED', 'B2_REGISTERED_S4', 'B1_REGISTERED_ANCHORS',
                       'LEGACY_FULL_PF_WITH_ORIGINAL_LIMITS', 'FIXED_B1_MODEL'],
        model=n.MODEL, compiler=n.COMPILER, profile=n.PROFILE,
        input_manifest_sha256=sha(ROOT / INPUTS), source_closure=list(SOURCES),
        selection='EXACT_FRACTION_COST_THEN_LEXICAL_ID',
        old32='MUST_BE_EXACTLY_UNCHANGED', PR='REUSE_SAVED_COST_SELECTION_AND_SE_VERBATIM',
        missing_eligible_cost='STOP', unresolved='NO_FULL_DOMAIN_MINIMUM',
        execution=dict(one_shot=True, scientific_passes=1, modes=2, saved_rows=3800, comparisons=40,
                       retry=False, resume=False, compile=0, sampling=0, signal=0, fit=0,
                       shots=0, bias=0, model_prediction=0, PR_mean_or_SE=0),
        resources=dict(processes='native wait witness, one launcher, one arithmetic child; no workers', stdlib_only=True,
            agent_imposed_CPU_RAM_worker_wall_caps=None, inherited_OS_limits='UNCHANGED',
            estimate_peak_RSS_bytes=256 * 2**20, estimate_output_bytes=64 * 2**20,
            estimate_output_inodes=32, PSI='OBSERVATION_ONLY',
            cleanup='OWN_UNREAPED_DIRECT_CHILD_PIDFD_TERM_KILL_WAIT4_FIRST_REASON_PRESERVED'),
        protection='ONLY_OWN_SOURCE_INPUT_RUNTIME_OUTPUT_GRANT_LOCK; OTHER_WORK_OBSERVATION',
        SOURCE_publication='MUST_BE_CONFIRMED_BEFORE_VALID_GRANT_OR_SCIENTIFIC_DISPATCH',
        rigorous_bias_certificate=None, PR_population_winner_guarantee=False,
        full_domain_direct_measured_optimum=False, mandatory_STOP=True, next_stage_authorized=False)


def environment():
    paths = {PYTHON}
    # Freeze this interpreter's stdlib and extension modules, without importing a
    # chemistry, compiler, numpy or third-party provider environment.
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(Path(path).resolve()).startswith('/usr/lib/python3.12/'):
            if str(path).endswith('.pyc'):
                path = path[:-1]
            if Path(path).is_file():
                paths.add(str(Path(path).resolve()))
    return dict(python=PYTHON, version=sys.version,
                files={p: sha(p) for p in sorted(paths)}, compiler_environment_used=False)


def verify_inputs(config):
    specs = read(ROOT / INPUTS)['files']
    by = n.index(config['inputs'], lambda r: r['role'])
    require(set(by) == {r['role'] for r in specs}, 'EXACT_INPUT_ROLES')
    for spec in specs:
        row = by[spec['role']]
        require(all(row[k] == spec[k] for k in ('bytes', 'sha256')), 'FIXED_INPUT_RECEIPT')
        p = home(row['path'])
        require(not p.is_symlink() and p.is_file() and p.stat().st_size == spec['bytes'] and
                sha(p) == spec['sha256'], 'INPUT_BYTE_SHA:' + row['role'])
    return by


def verify(config):
    require(config['plan_sha256'] == digest(plan()), 'PLAN_BINDING')
    require(subprocess.check_output([*GIT, 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() ==
            config['SOURCE'], 'SOURCE_HEAD')
    require(set(config['source_hashes']) == set(SOURCES), 'EXACT_SOURCE_CLOSURE')
    for name, h in config['source_hashes'].items():
        require(sha(ROOT / name) == h and hashlib.sha256(subprocess.check_output(
            [*GIT, 'show', config['SOURCE'] + ':' + name], cwd=ROOT)).hexdigest() == h, 'SOURCE_BYTE_SHA')
    runtime = config['environment']
    require(runtime['python'] == PYTHON and runtime['version'] == sys.version and
            all(sha(p) == h for p, h in runtime['files'].items()), 'STDLIB_RUNTIME_BINDING')
    return verify_inputs(config)


def authorize(config, grant):
    require(config.get('execution_sealed') is True and grant.get('approved') is True and
        grant.get('runtime_authorization') is True and grant.get('scope') == SCOPE and
        grant.get('SOURCE') == config.get('SOURCE') and len(config.get('SOURCE', '')) == 40 and
        grant.get('plan_sha256') == config.get('plan_sha256') == digest(plan()) and
        grant.get('config_sha256') == digest(config) and grant.get('one_shot') is True and
        grant.get('retry') is False and grant.get('resume') is False and
        grant.get('next_stage_authorized') is False and grant.get('approval_original_sha256') and
        grant.get('source_published_commit') == config.get('SOURCE') and
        type(grant.get('publication_remote_HEAD')) is str and len(grant['publication_remote_HEAD']) == 40 and
        grant.get('SOURCE_is_ancestor_of_publication_HEAD') is True and
        grant.get('source_publication_confirmed') is True and
        grant.get('input_manifest_sha256') == sha(ROOT / INPUTS), 'NEW_PUBLISHED_SEALED_ONE_SHOT_GRANT_REQUIRED')
    require(sha(home(grant['approval_original_path'])) == grant['approval_original_sha256'], 'USER_APPROVAL_BYTES')
    return n.Permit(_token=n._PERMIT)


def paths(config):
    output, control, lock = [home(config[k]) for k in ('output', 'control', 'lock')]
    require(len({output, control, lock}) == 3 and len({p.parent for p in (output, control, lock)}) == 1,
            'EXCLUSIVE_SIBLING_NAMESPACE')
    require(not any(p.exists() for p in (output, control, lock)), 'ONE_SHOT_ALREADY_SPENT')
    protected = [ROOT] + [home(r['path']) for r in config['inputs']]
    require(all(not p.is_relative_to(q) and not q.is_relative_to(p)
                for p in (output, control, lock) for q in protected), 'OUTPUT_SOURCE_INPUT_OVERLAP')
    return output, control, lock


def proof_metadata(config):
    """Saved evidence equality only: no new unit-weighted costs or selection."""
    by = verify_inputs(config)
    def get(role):
        return read(by[role]['path'])
    measured = get('six:six_axis_results.json')
    audit = get('six:result_audit_manifest.json')
    raw = get('six:raw_result')
    for role, key in [('six:raw_result', 'result_sha256'), ('six:raw_terminal', 'terminal_sha256'),
                      ('six:raw_final_gate', 'final_gate_sha256'), ('six:raw_launcher_exit', 'launcher_native_exit_sha256')]:
        require(by[role]['sha256'] == audit[key], 'SIX_ORIGINAL_RECEIPT_SHA')
    require(by['six:six_axis_results.json']['sha256'] == audit['public_projection_sha256'], 'SIX_PUBLISHED_PROJECTION')
    require(measured['SOURCE'] == raw['SOURCE'] == audit['SOURCE'] == n.SIX_SOURCE and
            measured['selected_axes'] == raw['rows'] and len(raw['rows']) == 6, 'SIX_RESULT_SOURCE_ROWS')
    require(measured['status'] == 'PASS_ONLY_SIX_REGISTERED_AXES_MANDATORY_STOP' and
            raw['status'] == measured['status'] and
            get('six:raw_terminal')['status'] == 'COMPLETE_MANDATORY_STOP' and
            measured['full_runtime_final_gate_already_PASS'] is True, 'SIX_COMPLETED_ONLY')
    require(measured['compiler_hash'] == n.COMPILER and measured['profile_sha256'] == n.PROFILE and
            measured['model_sha256'] == n.MODEL and measured['model_refit_performed'] is False,
            'SIX_COMPILER_PROFILE_MODEL')
    expected = {(cid, a) for cid in n.REPLACEMENTS for a in n.AXES}
    require({(r['case_id'], r['axis']) for r in raw['rows']} == expected, 'EXACT_SIX_MEASUREMENTS')
    axes = {}
    for i, row in enumerate(raw['rows'], 1):
        proof = get('six:native_%02d' % i)
        published = measured['axis_native_proofs'][i-1]
        require(by['six:native_%02d' % i]['sha256'] == published['receipt_sha256'] and
                proof['all_reaped'] is True and proof['status'] == 'PASS' and
                proof['cleanup_errors'] == [] and len(proof['workers']) == 1, 'SIX_AXIS_NATIVE_RECEIPT')
        worker = proof['workers'][0]
        task = get('six:task_%02d' % i)
        require(by['six:task_%02d' % i]['sha256'] == row['result_sha256'] and
                all(row[k] == value for k, value in task.items()), 'SIX_RAW_SCALAR_RESULT')
        require(worker['exit_code'] == row['native_exit_code'] == 0 and worker['reaped'] is True and
                row['reaped'] is True and worker['identity'] == row['worker_identity'] and
                worker['result_sha256'] == row['result_sha256'] == published['worker_result_sha256'],
                'SIX_NATIVE_IDENTITY_EXIT_REAP')
        source = n.OLD_SIX_SOURCE if (row['case_id'], row['axis']) == ('B0I_p15_q4_4th', 'cosine') else n.SIX_SOURCE
        require(row['SOURCE'] == source and row['profile_sha256'] == n.PROFILE and
                row['compiler_hash'] == n.COMPILER and row['model_sha256'] == n.MODEL and
                row['measured_RZ'] == n.REPLACEMENTS[row['case_id']] and
                row['model_check']['PASS'] is True and
                0 <= n.F(row['model_check']['exact_relative_error']) < n.F(1,50) and
                0 <= row['max_control_error'] < 1e-8 and
                row['fingerprint_before'] == row['fingerprint_after'], 'SIX_AXIS_SOURCE_PROFILE_COST')
        axes.setdefault(row['case_id'], {})[row['axis']] = row
    gate, native = get('six:raw_final_gate'), get('six:raw_launcher_exit')
    require(gate['SOURCE'] == native['SOURCE'] == n.SIX_SOURCE and gate['status'] == 'PASS_MANDATORY_STOP' and
            gate['native']['all_reaped'] is True and gate['native']['cleanup_errors'] == [] and
            gate['first_stop'] is None and native['native_exit_code'] == 0 and native['reaped'] is True,
            'SIX_FINAL_NATIVE_GATE')
    legacy.b2_gate(dict(binding=get('original:B2_binding'), receipts=get('original:B2_execution_receipts')))
    require(digest(get('original:model')) == n.MODEL, 'B1_FIXED_MODEL')
    require(get('original:B1_terminal')['SOURCE'] == '6fb8a0a71986fda571d82b5658686f5e74a32f8b' and
            get('original:B1_launcher_native_exit')['native_exit_code'] == 0 and
            get('original:B1_launcher_native_exit')['wait_completed'] is True, 'B1_NATIVE_COMPLETION')
    normal_binding = get('usual:execution_binding_v1.json')
    require(normal_binding['B1_result_commit'] == '83a3c4a6688e66576f85471bf41a45e29cbfb705' and
            normal_binding['B2_result_commit'] == 'fef2db7a79886ab682b652da7266b463c98c7d3a', 'B1_B2_RESULT_LINK')
    for receipt in normal_binding['input_receipts']:
        pinned = by['original:' + receipt['role']]
        require((receipt['bytes'], receipt['sha256']) == (pinned['bytes'], pinned['sha256']),
                'SAVED_NORMAL_ACCOUNTING_ORIGINAL_INPUT_JOIN')
    for prefix, source in [('usual', n.USUAL_SOURCE), ('noncancel', n.NONCANCEL_SOURCE)]:
        binding, gate, terminal, native = [get(prefix + ':' + name) for name in
            ('execution_binding_v1.json', 'final_gate_v1.json', 'terminal_v1.json', 'native_completion_v1.json')]
        require(binding['SOURCE'] == gate['SOURCE'] == terminal['SOURCE'] == source and
            binding['model_sha256'] == n.MODEL and binding['profile_sha256'] == n.PROFILE and
            binding['compiler_hash'] == n.COMPILER and binding['scientific_attempts'] == 1 and
            gate['status'] == 'PASS_MANDATORY_STOP' and terminal['status'] == 'COMPLETE_MANDATORY_STOP',
            'COMPLETED_ACCOUNTING_BINDING:' + prefix)
        launcher = native['launcher'].get('native', native['launcher'])
        require(native['driver']['native_exit_code'] == launcher['native_exit_code'] == 0 and
                native['driver']['reaped'] is True and launcher['reaped'] is True, 'ACCOUNTING_NATIVE_REAP')
    return dict(status='PASS_METADATA_ONLY_NO_NEW_SCIENTIFIC_COST', inputs=len(by),
                six_axes=6, B2_axes=4, source=n.SIX_SOURCE), axes


def project_saved(config):
    """Lossless saved-value projection only; no multiplication or ranking."""
    _, axes = proof_metadata(config)
    by = n.index(config['inputs'], lambda r: r['role'])
    def get(role): return read(by[role]['path'])
    def rows(prefix):
        receipt = get(prefix + ':accounting_rows_receipt_v1.json')
        with gzip.open(by[prefix + ':accounting_rows_v1.jsonl.gz']['path'], 'rb') as stream:
            data = stream.read(32 * 2**20 + 1)
        require(len(data) == receipt['original']['bytes'] and
                hashlib.sha256(data).hexdigest() == receipt['original']['sha256'], 'LOSSLESS_SAVED_ROWS')
        result = [loads(line) for line in data.splitlines()]
        require(len(result) == receipt['rows'] == 1900, 'SAVED_1900_ROWS')
        return result
    registry = get('original:registry')
    ids = {x['id'] for x in registry['old'] + registry['additional']}
    d0 = get('original:D0')
    accounting = [dict(cell_id=r['cell_id'], epsilon=r['epsilon'], factor=r['factor'],
        accounting=r['accounting']) for r in d0['H8_EXPLORATION92'] if r['cell_id'] in ids]
    accounting.extend(dict(cell_id=r['cell_id'], epsilon=r['epsilon'], factor=r['factor'], accounting=r['accounting'])
                      for r in get('original:B0I_accounting'))
    with gzip.open(by['original:Phase_A']['path'], 'rb') as stream:
        a = loads(stream.read(32 * 2**20))
    require(a['status'] == 'COMPLETE_MANDATORY_STOP', 'PHASE_A_SAVED_COMPLETION')
    for signal in a['signals']:
        for e in n.EPSILONS:
            for f in n.FACTORS:
                accounting.append(dict(cell_id=signal['cell']['id'], epsilon=e, factor=f,
                                       accounting=signal['precision_rows'][str(e)][str(f)]))
    for r in d0['H8_CONFIRMATION_UNION6']:
        if r['cell_id'] in registry['confirmation_ids']:
            require(r['sample_phase'] == 'confirmation' and r['sample_count'] == 32, 'SAVED_PR_CONFIRMATION')
            accounting.append(dict(cell_id=r['cell_id'], epsilon=r['epsilon'], factor=r['factor'], accounting=r['accounting']))
    normal, noncancel = rows('usual'), rows('noncancel')
    return dict(kind='SAVED_H8', modes={
        n.MODES[0]: dict(rows=normal, accounting=accounting,
                        comparisons=get('usual:comparisons_v1.json')['comparisons']),
        n.MODES[1]: dict(rows=noncancel, accounting=noncancel,
                        comparisons=get('noncancel:comparisons_v1.json')['comparisons'])}), registry, axes


def check_saved_structure(config):
    data, registry, _ = project_saved(config)
    old = {r['id'] for r in registry['old']}
    ids = old | {r['id'] for r in registry['additional']} | set(registry['confirmation_ids'])
    expected = {(cid,e,f) for cid in ids for e in n.EPSILONS for f in n.FACTORS}
    require(len(old) == 32 and len(ids) == 190 and not (old & set(n.REPLACEMENTS)), 'FIXED_REGISTRY_STRUCTURE')
    for mode, bundle in data['modes'].items():
        rows = n.index(bundle['rows'], n.row_key)
        accounting = n.index(bundle['accounting'], n.row_key)
        require(set(rows) == set(accounting) == expected, 'COMPLETE_1900_ROW_SHOT_JOIN:' + mode)
        for key,row in rows.items():
            a = accounting[key]['accounting']
            if row['eligible'] is True:
                require(a is not None and a['cost_acquisition_eligible'] is True and
                        a['eligibility_under_declared_allowance'] == 'ELIGIBLE', 'SAVED_ELIGIBLE_ACCOUNTING')
                require(all(type(a['axes'][axis]['shots']) is int for axis in ('real','imag')), 'SAVED_INTEGER_SHOTS')
    return dict(status='PASS_SAVED_VALUES_ONLY_NO_COST_OR_RANKING', modes=2, rows=3800,
                deterministic_ids=184, PR_confirmation_ids=6, old32_ids=32,
                selected_six_outside_old32=True, new_scientific_calls=0)


def load_saved(config, permit):
    require(isinstance(permit, n.Permit) and not permit.synthetic, 'REAL_DATA_APPROVAL_REQUIRED')
    return project_saved(config)


def observe(base):
    mem = {k.rstrip(':'): int(v.split()[0])*1024 for k,v in
           (line.split(None, 1) for line in Path('/proc/meminfo').read_text().splitlines())
           if k.rstrip(':') in ('MemAvailable', 'MemFree')}
    groups = []
    cg = Path('/sys/fs/cgroup') / next(line.split(':', 2)[2].lstrip('/') for line in
                                      Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    while cg.is_relative_to('/sys/fs/cgroup'):
        row = dict(path=str(cg))
        for name in ('memory.events', 'memory.pressure', 'memory.max', 'memory.current'):
            p = cg / name
            if p.exists(): row[name] = p.read_text()
        groups.append(row)
        if cg == Path('/sys/fs/cgroup'): break
        cg = cg.parent
    stat = os.statvfs(base)
    return dict(monotonic=time.monotonic(), memory=mem, cgroups=groups,
        host_memory_PSI=Path('/proc/pressure/memory').read_text(),
        allowed_cpus=sorted(os.sched_getaffinity(0)), loadavg=os.getloadavg(),
        cpu_stat=Path('/proc/stat').read_text(), filesystem_free_bytes=stat.f_bavail*stat.f_frsize,
        filesystem_free_inodes=stat.f_favail, inherited_AS=resource.getrlimit(resource.RLIMIT_AS),
        inherited_CPU=resource.getrlimit(resource.RLIMIT_CPU), PSI_is_STOP_gate=False)


def fresh(config, grant):
    authorize(config, grant)
    verify(config)
    proof_metadata(config)
    output, control, lock = paths(config)
    # A successful write on this exact filesystem verifies usable capacity even
    # when no per-user quota utility exists. Keep UNKNOWN separate from zero.
    quota = shutil.which('quota')
    q = dict(tool=quota, policy='NO_CAP_ASSUMED_FROM_TOOL_ABSENCE')
    if quota:
        command = subprocess.run([quota, '-s'], capture_output=True, text=True, timeout=15)
        q.update(returncode=command.returncode, stdout=command.stdout, stderr=command.stderr)
        require(command.returncode == 0, 'QUOTA_QUERY_FAILED')
    else:
        q['status'] = 'UTILITY_UNAVAILABLE; SAME_FILESYSTEM_ALLOCATION_PROBE_REQUIRED'
    probe = output.parent / ('allocation-probe-' + str(os.getpid()))
    try:
        with probe.open('xb') as stream:
            os.posix_fallocate(stream.fileno(), 0, 64 * 2**20)
            stream.flush()
            os.fsync(stream.fileno())
        q['allocation_probe_bytes'] = 64 * 2**20
        q['allocation_probe'] = 'PASS; DOES_NOT_PROVE_NUMERICAL_QUOTA_HEADROOM'
    finally:
        if probe.exists(): probe.unlink()
    observation = observe(output.parent)
    effective = observation['memory']['MemAvailable']
    for row in observation['cgroups']:
        limit = row.get('memory.max', 'max').strip()
        if limit != 'max': effective = min(effective, int(limit)-int(row['memory.current']))
    require(effective >= 256 * 2**20 and observation['filesystem_free_bytes'] >= 64 * 2**20 and
            observation['filesystem_free_inodes'] >= 32 and observation['allowed_cpus'], 'ACTUAL_RESOURCE_SHORTAGE')
    return dict(status='PASS', SOURCE=config['SOURCE'], config_sha256=digest(config),
        grant_sha256=digest(grant), observation=observation, effective_available_bytes=effective,
        quota=q, no_other_job_mutated=True, unused_output_control_lock=True,
        source_input_receipt_profile_verified=True)
