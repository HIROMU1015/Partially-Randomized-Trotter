"""Source-bound one-shot diagnostic controller; no production or library changes."""
import argparse
import ctypes
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PREP = 'artifacts/track_b_g1_result_prior_preparation/2026-10-09'
SOURCE_MANIFEST = 'artifacts/track_b_g1_source_preparation/2026-10-09/source_manifest_v1.json'
APPROVAL_SENTENCE = 'source `{source}` のG1固定契約で、decision packetを一回だけ実行し、終了後はmandatory STOPしてください。'
PASS_A = 'G1_STRUCTURE_PASS_WITH_DECLARED_LIMITS'
PASS_B = 'G1_BACKEND_CLOSURE_PASS'
TECH = 'G1_TECHNICAL_INCONCLUSIVE'
ACQUIRE = 'G1_BACKEND_ACQUISITION_INCONCLUSIVE'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def exact_vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError('certificate vector shape')
    for v in value:
        if not isinstance(v, str) or str(F(v)) != v:
            raise ValueError('certificate must use canonical rational strings')


def payload_gate(problem, result, echo=False):
    """No verifier call for unavailable or malformed evidence. Priority is contract-bound."""
    if not isinstance(result, dict) or not isinstance(result.get('status'), str):
        return TECH, 'OUTPUT_FORMAT'
    try:
        actual = result['echo']
        keys = ('c0', 'c', 'U', 'A', 'b', 'H', 'f')
        if not isinstance(actual, dict) or any(actual[k] != problem[k] for k in keys):
            return TECH, 'INPUT_READBACK_MISMATCH'
        if actual['lower'] != ['0']*len(problem['c']) or actual['equal_lhs'] != problem['f']:
            return TECH, 'INPUT_READBACK_MISMATCH'
    except (KeyError, TypeError):
        return TECH, 'INPUT_READBACK_MISSING'
    status = result['status']
    if echo:
        return (None, None) if status == 'ECHO_ONLY' else (TECH, 'ECHO_STATUS')
    if status not in ('OPTIMAL', 'INFEASIBLE'):
        return ACQUIRE, 'BACKEND_STATUS_NO_CERTIFICATE'
    required = ('primal_available', 'dual_available', 'reduced_cost_available') if status == 'OPTIMAL' else ('farkas_available',)
    if any(result.get(flag) is not True for flag in required):
        return ACQUIRE, 'EXACT_PAYLOAD_UNAVAILABLE'
    if status != problem['expected_status']:
        return 'G1_FIXTURE_STATUS_INCONCLUSIVE', 'EXPECTED_STATUS_MISMATCH'
    try:
        n, rows = len(problem['c']), len(problem['A'])+len(problem['H'])
        if status == 'OPTIMAL':
            exact_vector(result['primal'], n)
            exact_vector(result['raw_row_dual'], rows)
            exact_vector(result['raw_reduced_cost'], n)
            exact_vector([result['objective_with_exact_external_offset'], result['backend_objective_without_offset']], 2)
        else:
            exact_vector(result['raw_row_farkas'], rows)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return TECH, 'PAYLOAD_FORMAT'
    return None, None


def verdict_gate(verdict, echo=False):
    if not isinstance(verdict, dict) or type(verdict.get('PASS')) is not bool:
        return TECH, 'VERIFIER_OUTPUT_FORMAT'
    if not verdict['PASS']:
        if echo: return TECH, 'ECHO_VERIFICATION_FAILED'
        return 'G1_BACKEND_INVALID_CERTIFICATE', 'EXACT_CERTIFICATE_FAILED'
    if not echo and verdict.get('status') == 'OPTIMAL':
        try:
            gap = verdict['dual']['exact_gap']
            exact_vector([gap], 1)
        except (KeyError, TypeError, ValueError):
            return TECH, 'VERIFIER_GAP_FORMAT'
        if F(gap) != 0: return ACQUIRE, 'EXACT_OPTIMALITY_NOT_CLOSED'
    return None, None


def verifier_guard_failure(record, verdict):
    # The unchanged verifier uses exit 1 for a well-formed rejected certificate.
    # Keep the raw guard STOP/ledger; interpret that refusal, never retry it.
    if (isinstance(record, dict) and record.get('failure') == 'PROCESS_EXIT_FAILURE'
            and record.get('returncode') == 1 and record.get('residual_processes') == 0
            and isinstance(verdict, dict) and verdict.get('PASS') is False):
        return None, None
    return guard_failure(record)


class Store:
    def __init__(self, private, artifact):
        self.private, self.artifact = Path(private), Path(artifact)
        self.marker, self.stop = self.private/'one_shot_consumed.json', self.private/'STOP.json'

    def start(self, context):
        if self.marker.exists() or self.stop.exists():
            raise PermissionError('consumed marker/STOP; no retry')
        self.private.mkdir(parents=True, exist_ok=True)
        with self.marker.open('x') as stream:
            json.dump(context, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self.artifact.mkdir(parents=True, exist_ok=True)

    def save(self, result):
        for path in (self.private/'result.json', self.artifact/'result.json'):
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
            os.replace(temporary, path)

    def end(self, result):
        if not self.stop.exists():
            with self.stop.open('x') as stream:
                json.dump({'attempt_closed': True, 'retries': 0, 'mandatory_STOP': True}, stream, sort_keys=True)
        self.save(result)


def guard_failure(record):
    if not isinstance(record, dict): return TECH, 'GUARD_RECORD_MISSING'
    failure = record.get('failure')
    if failure:
        if failure in ('WALL_CAP', 'PILOT_WALL_CAP', 'RSS_CAP', 'OUTPUT_CAP'):
            return 'G1_RESOURCE_INCONCLUSIVE', failure
        return TECH, failure
    if record.get('returncode') != 0 or record.get('residual_processes') != 0:
        return TECH, 'GUARD_EXIT_OR_RESIDUAL'
    return None, None


def execute_packet(contract, fixtures, store, transport, context):
    """Core state machine; tests supply isolated fake transport and artificial toy rows."""
    order = contract['LP_order']
    if len(order) != 8 or len(set(order)) != 8 or contract['LP_call_cap'] != 8:
        raise ValueError('exactly eight distinct planned keys required')
    if set(fixtures) != set(order): raise ValueError('fixture key set mismatch')
    store.start(context)  # Outside catch: refusal cannot overwrite old result/STOP.
    result = {'classification': TECH, 'phase': 'A', 'structure_audit_calls': 0, 'echo_only_calls': 0,
              'LP_calls': 0, 'verification_calls': 0, 'rows': [], 'stages': [], 'retries': 0,
              'compile_calls': 0, 'mandatory_STOP': True, 'production_authorized': False,
              'registered_science_synthesis_matrix_circuit_trajectory_DF_molecule_NPZ_GPU': 0,
              'marker_sha256': sha(store.marker), 'source_commit': context.get('source_commit'),
              'contract_sha256': context.get('contract_sha256'), 'instruction_sha256': context.get('instruction_sha256')}
    def fail(classification, reason):
        result.update(classification=classification, reason=reason)
        summarize_resources()
        store.end(result)
        try:
            transport.final_check()
            result['final_identity_check'] = 'PASS'
        except Exception as exc:
            result['initial_stop_classification'] = classification
            result['initial_stop_reason'] = reason
            result['final_identity_check'] = 'FAIL'
            result['classification'] = 'G1_RESOURCE_INCONCLUSIVE' if isinstance(exc, ResourceLimit) else TECH
            result['reason'] = f'POST_STOP_READ_ONLY_CHECK: {type(exc).__name__}: {exc}'
        store.save(result)
        return result
    def summarize_resources():
        records = result['stages']
        result['resources'] = {'child_CPU_seconds': sum(r.get('CPU_seconds', 0) for r in records),
            'peak_observed_RSS_sum_bytes': max((r.get('peak_RSS_sum_bytes', 0) for r in records), default=0),
            'peak_observed_output_bytes': max((r.get('peak_output_bytes', 0) for r in records), default=0),
            'wall_seconds_since_marker': time.time()-context['started_epoch'] if 'started_epoch' in context else None,
            'all_completed_guard_records_residual_zero': all(r.get('residual_processes') == 0 for r in records)}
    def consume(kind):
        transport.check()
        cap = 1 if kind == 'structure_audit_calls' else 8 if kind in ('echo_only_calls', 'LP_calls') else 16
        if result[kind] >= cap: raise RuntimeError('call cap')
        result[kind] += 1
        store.save(result)
    def stage_reply(reply, verifier=False):
        record, data = reply
        result['stages'].append(record)
        code, reason = verifier_guard_failure(record, data) if verifier else guard_failure(record)
        return code, reason, data
    try:
        consume('structure_audit_calls')
        code, reason, data = stage_reply(transport.audit(store))
        result['structure'] = data
        if code: return fail(code, reason)
        if not isinstance(data, dict): return fail(TECH, 'STRUCTURE_OUTPUT_FORMAT')
        classification = data.get('classification')
        if classification != PASS_A:
            if classification not in ('G1_STRUCTURE_COUNTEREXAMPLE', 'G1_STRUCTURE_TECHNICAL_INCONCLUSIVE'):
                classification = TECH
            return fail(classification, 'STRUCTURE_STAGE_DID_NOT_PASS')
        result['phase'] = 'B_ECHO'
        store.save(result)
        for key in order:
            problem = fixtures[key]
            consume('echo_only_calls')
            code, reason, output = stage_reply(transport.acquire(problem, echo=True))
            row = {'id': key, 'mode': 'ECHO_ONLY', 'backend_output': output, 'certificate': 'NOT_RUN'}
            result['rows'].append(row)
            if code: return fail(code, reason)
            code, reason = payload_gate(problem, output, echo=True)
            if code: return fail(code, reason)
            consume('verification_calls')
            code, reason, verdict = stage_reply(transport.verify(problem, output, echo=True), verifier=True)
            row['verification'] = verdict
            if code: return fail(code, reason)
            code, reason = verdict_gate(verdict, echo=True)
            if code: return fail(code, reason)
            row['certificate'] = 'PASS'
            store.save(result)
        result['phase'] = 'B_SOLVE'
        store.save(result)
        for key in order:
            problem = fixtures[key]
            consume('LP_calls')  # Failed launch consumes the key; no suffix on failure.
            code, reason, output = stage_reply(transport.acquire(problem, echo=False))
            row = {'id': key, 'mode': 'SOLVE', 'backend_output': output, 'certificate': 'NOT_RUN'}
            result['rows'].append(row)
            if code: return fail(code, reason)
            code, reason = payload_gate(problem, output)
            if code: return fail(code, reason)
            consume('verification_calls')
            code, reason, verdict = stage_reply(transport.verify(problem, output, echo=False), verifier=True)
            row['verification'] = verdict
            if code: return fail(code, reason)
            code, reason = verdict_gate(verdict)
            if code: return fail(code, reason)
            row['certificate'] = 'PASS'
            store.save(result)
        transport.check()
        result.update(classification=PASS_B, phase='COMPLETE', reason=None)
        summarize_resources()
        store.end(result)
        transport.final_check()
        result['final_identity_check'] = 'PASS'
        store.save(result)
        return result
    except ResourceLimit as exc:
        return fail('G1_RESOURCE_INCONCLUSIVE', str(exc))
    except Exception as exc:
        return fail(TECH, f'{type(exc).__name__}: {exc}')


class ResourceLimit(Exception): pass


def wire(problem):
    lines = [f"{len(problem['c'])} {len(problem['A'])} {len(problem['H'])}", problem['c0'],
             ' '.join(problem['c']), ' '.join(problem['U'])]
    for key, rhs in (('A', 'b'), ('H', 'f')):
        lines.extend(' '.join(row+[value]) for row, value in zip(problem[key], problem[rhs]))
    return '\n'.join(lines)+'\n'


class GuardedTransport:
    def __init__(self, root, contract, runtime, store):
        self.root, self.contract, self.runtime, self.store = root, contract, runtime, store
        self.deadline = time.time()+contract['resources']['total_wall_seconds_from_exclusive_marker']
        self.monotonic_deadline = time.monotonic()+contract['resources']['total_wall_seconds_from_exclusive_marker']
        self.input_paths = {}
        self.roots = [str(store.private), str(store.artifact)]
        manifest = read_json(root/SOURCE_MANIFEST)
        self.source_hashes = dict(manifest['frozen_files'])
        self.source_hashes.update(manifest['protected_unchanged_files'])
        self.source_hashes[SOURCE_MANIFEST] = sha(root/SOURCE_MANIFEST)
        evidence = 'artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json'
        if (root/evidence).is_file(): self.source_hashes[evidence] = sha(root/evidence)
        self.roots += [str(root/p) for p in manifest['output_accounting_extra_roots']]

    def check(self, allow_stop=False):
        if time.monotonic() >= self.monotonic_deadline: raise ResourceLimit('TOTAL_WALL_CAP')
        if self.store.stop.exists() and not allow_stop: raise PermissionError('attempt after STOP')
        seen, total = set(), 0
        for name in self.roots:
            path = Path(name)
            paths = [path] if path.is_file() else path.rglob('*') if path.exists() else []
            for p in paths:
                if p.is_file():
                    s = p.stat()
                    ident = (s.st_dev, s.st_ino)
                    if ident not in seen: seen.add(ident); total += s.st_size
        if total >= self.contract['resources']['new_output_bytes']: raise ResourceLimit('OUTPUT_CAP')

    def final_check(self):
        self.check(allow_stop=True)
        for path, expected in self.source_hashes.items():
            if sha(self.root/path) != expected: raise PermissionError('post-run source/protected identity mismatch: '+path)
        runtime = self.runtime
        identities = {runtime['binary']['path']: runtime['binary']['sha256'], runtime['python']['path']: runtime['python']['sha256']}
        identities.update(runtime['static_libraries'])
        identities.update({item['path']: item['SHA256'] for item in runtime['libraries']})
        for path, expected in identities.items():
            if sha(path) != expected: raise PermissionError('post-run runtime identity mismatch: '+path)
        self.check(allow_stop=True)

    def launch(self, name, command, wall, rss=None):
        self.check()
        private = self.store.private
        paths = {kind: private/kind for kind in ('specs', 'ledger', 'output', 'tmp')}
        for p in paths.values(): p.mkdir(exist_ok=True)
        resources = self.contract['resources']
        effective_deadline = min(self.deadline, time.time()+max(0, self.monotonic_deadline-time.monotonic()))
        spec = {'id': name, 'command': command, 'cwd': str(private), 'ledger': str(paths['ledger']),
                'stop_path': str(self.store.stop), 'stdout': str(paths['output']/f'{name}.stdout.json'),
                'stderr': str(paths['output']/f'{name}.stderr.txt'), 'TMPDIR': str(paths['tmp']),
                'pilot_deadline_epoch': effective_deadline, 'wall_seconds': wall,
                'RSS_bytes': rss or resources['RSS_bytes'], 'address_space_bytes': rss or resources['address_space_bytes'],
                'output_bytes': resources['new_output_bytes'], 'output_roots': self.roots,
                'sample_seconds': float(F(resources['sample_seconds']))}
        spec_path = paths['specs']/f'{name}.json'
        with spec_path.open('x') as stream: json.dump(spec, stream, sort_keys=True)
        guard = self.root/'scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py'
        # The controller is also a subreaper for the exceptional supervisor-timeout cleanup path.
        if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0: raise RuntimeError('controller subreaper unavailable')
        cleanup_guard = readonly_guard_module()
        baseline = cleanup_guard.members(-1, os.getpid())
        proc = subprocess.Popen([self.runtime['python']['path'], '-B', str(guard), str(spec_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        timeout = max(.001, min(wall, effective_deadline-time.time()))+10
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cleanup = cleanup_supervisor(proc, baseline)
            (paths['output']/f'{name}.supervisor.stdout.txt').write_bytes(cleanup['stdout'] or exc.output or b'')
            (paths['output']/f'{name}.supervisor.stderr.txt').write_bytes(cleanup['stderr'] or exc.stderr or b'')
            return ({'id': name, 'failure': 'GUARD_SUPERVISOR_TIMEOUT', 'returncode': proc.returncode,
                     'residual_processes': len(cleanup['survivors']), 'cleanup': {k: v for k, v in cleanup.items() if k not in ('stdout', 'stderr')}}, None)
        (paths['output']/f'{name}.supervisor.stdout.txt').write_bytes(stdout)
        (paths['output']/f'{name}.supervisor.stderr.txt').write_bytes(stderr)
        try: record = read_json(paths['ledger']/f'{name}.result.json')
        except (OSError, ValueError):
            cleanup = cleanup_supervisor(proc, baseline)
            return ({'id': name, 'failure': 'GUARD_RECORD_MISSING', 'returncode': proc.returncode,
                     'residual_processes': len(cleanup['survivors']), 'cleanup': {k: v for k, v in cleanup.items() if k not in ('stdout', 'stderr')}}, None)
        record['id'] = name
        try: data = read_json(Path(spec['stdout']))
        except (OSError, ValueError): data = None
        return record, data

    def audit(self, store):
        contract_path = self.root/self.contract['structure_contract']
        permit = {'stage': 'A_STRUCTURE_AUDIT', 'marker': str(store.marker), 'marker_sha256': sha(store.marker),
                  'structure_contract_sha256': sha(contract_path)}
        path = store.private/'structure_permit.json'
        with path.open('x') as stream: json.dump(permit, stream, sort_keys=True)
        resource = read_json(contract_path)['math_resource_limit']
        command = [self.runtime['python']['path'], '-B', str(self.root/'scripts/tracks/algorithm_codesign/audit_g1_structure.py'),
                   str(self.root), str(contract_path), str(path)]
        return self.launch('A_STRUCTURE_AUDIT', command, resource['wall_seconds'], resource['RSS_bytes'])

    def acquire(self, problem, echo):
        name = problem['id']
        if name not in self.input_paths:
            folder = self.store.private/'inputs'; folder.mkdir(exist_ok=True)
            problem_path, rational_path = folder/f'{name}.json', folder/f'{name}.rational.txt'
            with problem_path.open('x') as stream: json.dump(problem, stream, sort_keys=True)
            with rational_path.open('x') as stream: stream.write(wire(problem))
            self.input_paths[name] = problem_path, rational_path
        command = [self.runtime['binary']['path'], str(self.input_paths[name][1])]+(['--echo-only'] if echo else [])
        return self.launch(('echo_' if echo else 'solve_')+name, command,
                           self.contract['resources']['per_echo_or_solve_wall_seconds'])

    def verify(self, problem, result, echo):
        name = ('echo_' if echo else 'solve_')+problem['id']
        command = [self.runtime['python']['path'], '-B', str(self.root/self.contract['independent_certificate']['verifier']),
                   str(self.input_paths[problem['id']][0]), str(self.store.private/'output'/f'{name}.stdout.json')]
        return self.launch(name+'_verify', command, self.contract['resources']['per_independent_verifier_wall_seconds'])


def readonly_guard_module():
    from importlib.util import module_from_spec, spec_from_file_location
    path = ROOT/'scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py'
    spec = spec_from_file_location('g1_readonly_guard_cleanup', path)
    guard = module_from_spec(spec); spec.loader.exec_module(guard)
    return guard


def cleanup_supervisor(proc, baseline):
    """Exceptional cleanup; preserve pre-existing controller children and PID identities."""
    guard = readonly_guard_module()
    known = guard.members(proc.pid, proc.pid)
    current = guard.members(-1, os.getpid())
    unrelated = {pid for pid, before in baseline.items()
                 if (current.get(pid) or {}).get('start_ticks') == before['start_ticks']}
    while True:
        extra = {pid for pid, row in current.items() if row['ppid'] in unrelated and pid not in unrelated}
        if not extra: break
        unrelated.update(extra)
    known.update({pid: row for pid, row in current.items() if pid not in unrelated})
    own = guard.stat(proc.pid)
    if own: known[proc.pid] = own
    for pid, original in known.items():
        current = guard.stat(pid)
        if current and current['start_ticks'] == original['start_ticks']:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
    stdout, stderr = b'', b''
    try: stdout, stderr = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired: pass
    end = time.monotonic()+3
    reaped = []
    while time.monotonic() < end:
        try:
            pid, status, usage = os.wait4(-1, os.WNOHANG)
        except ChildProcessError: break
        if pid == 0:
            if not any((guard.stat(p) or {}).get('start_ticks') == original['start_ticks'] for p, original in known.items()):
                break
            time.sleep(.01)
        else: reaped.append({'pid': pid, 'returncode': os.waitstatus_to_exitcode(status),
                             'CPU_seconds': usage.ru_utime+usage.ru_stime})
    survivors = [pid for pid, original in known.items() if (guard.stat(pid) or {}).get('start_ticks') == original['start_ticks']]
    return {'survivors': survivors, 'reaped': reaped, 'stdout': stdout, 'stderr': stderr}


def checked_source(root, source, instruction=None):
    if len(source) != 40 or any(c not in '0123456789abcdef' for c in source): raise PermissionError('full lowercase source SHA required')
    git = lambda *a: subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    if git('rev-parse', 'HEAD') != source: raise PermissionError('HEAD mismatch')
    if git('status', '--porcelain'): raise PermissionError('worktree dirty')
    if git('remote', 'get-url', 'origin') != 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git':
        raise PermissionError('wrong repository remote')
    branch = git('branch', '--show-current')
    if not branch or branch in ('main', 'master'): raise PermissionError('independent branch required')
    remote = git('ls-remote', '--heads', 'origin', 'refs/heads/'+branch).split()
    if not remote or remote[0] != source: raise PermissionError('remote source SHA mismatch')
    manifest = read_json(root/SOURCE_MANIFEST)
    for path, expected in manifest['frozen_files'].items():
        if path.lower().endswith('.npz'): raise PermissionError('NPZ path prohibited')
        if sha(root/path) != expected: raise PermissionError('frozen source/input mismatch: '+path)
    for path, expected in manifest['protected_unchanged_files'].items():
        if path.lower().endswith('.npz'): raise PermissionError('NPZ path prohibited')
        if sha(root/path) != expected: raise PermissionError('protected identity mismatch: '+path)
    runtime = read_json(root/(PREP+'/runtime_identity_v1.json'))
    identities = {runtime['binary']['path']: runtime['binary']['sha256'], runtime['python']['path']: runtime['python']['sha256']}
    identities.update(runtime['static_libraries'])
    identities.update({item['path']: item['SHA256'] for item in runtime['libraries']})
    for path, expected in identities.items():
        if sha(path) != expected: raise PermissionError('runtime changed/missing: '+path)
    packet = read_json(root/(PREP+'/decision_packet_contract_v1.json'))
    marker = Path(packet['state']['one_shot_marker'])
    if marker.exists() or Path(packet['state']['stop_file']).exists(): raise PermissionError('consumed marker/STOP')
    if (root/packet['state']['result_artifact_relative']).exists(): raise PermissionError('result directory already exists')
    if instruction is not None and APPROVAL_SENTENCE.format(source=source) not in instruction:
        raise PermissionError('exact explicit one-shot instruction not supplied')
    return manifest, runtime, packet


def main():
    parser = argparse.ArgumentParser(description='G1 limited diagnostic; execution requires separately explicit source-bound instruction')
    parser.add_argument('--source-commit', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-source', action='store_true')
    mode.add_argument('--execute-one-shot', action='store_true')
    parser.add_argument('--instruction-file')
    args = parser.parse_args()
    if args.execute_one_shot and not args.instruction_file: parser.error('--instruction-file required; no execution approval stored in source')
    instruction = Path(args.instruction_file).read_text() if args.instruction_file else None
    manifest, runtime, contract = checked_source(ROOT, args.source_commit, instruction)
    if args.verify_source:
        print(json.dumps({'source': args.source_commit, 'source_identity': 'PASS', 'execution_authorized': False,
                          'LP_calls': 0, 'structure_audit_calls': 0, 'binary_invocations': 0}))
        return 0
    fixtures_manifest = read_json(ROOT/contract['fixture_manifest'])
    fixtures = {}
    for entry in fixtures_manifest['fixtures']:
        p = ROOT/entry['input_path']
        if sha(p) != entry['input_file_sha256']: raise PermissionError('fixture mismatch')
        fixtures[entry['id']] = read_json(p)
    store = Store(contract['state']['private_root'], ROOT/contract['state']['result_artifact_relative'])
    # Deadlines begin before exclusive marker; total allowance is never restarted between stages.
    transport = GuardedTransport(ROOT, contract, runtime, store)
    context = {'source_commit': args.source_commit, 'explicit_one_shot_instruction_bound': True,
               'instruction': instruction, 'instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
               'contract_sha256': sha(ROOT/(PREP+'/decision_packet_contract_v1.json')),
               'source_manifest_sha256': sha(ROOT/SOURCE_MANIFEST), 'started_epoch': time.time(),
               'runs': 1, 'retries': 0, 'mandatory_STOP': True}
    result = execute_packet(contract, fixtures, store, transport, context)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'stages', 'structure')}))
    return 0 if result['classification'] == PASS_B else 1


if __name__ == '__main__': sys.exit(main())
