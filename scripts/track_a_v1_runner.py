"""V1 preparation CLI; execution needs a new sealed, source-bound user grant.

--plan is the only default action. Imports and molecular callbacks follow all
source/input/profile/resource/grant checks, never the other way round.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load('_v1_contract', ROOT/'src/trotterlib/track_a_v1_contract.py')
accounting = load('_v1_legacy_shots', ROOT/'scripts/d0_saved_value_reanalysis_v1.py')


def read_json(path):
    return accounting.strict_json(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


class Writer:
    def __init__(self, directory, guard=lambda: None):
        self.directory, self.guard = Path(directory), guard

    def write(self, name, value):
        self.guard()
        if Path(name).name != name: raise ValueError('OUTPUT_NAME')
        with (self.directory/name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def verify_grant(config, grant, *, config_sha256, source, fresh):
    if (grant.get('approved_by_user') is not True or grant.get('runtime_authorization') is not True
            or grant.get('schema') != 'track_a_v1_signal_grant_v1'
            or grant.get('one_shot') is not True or grant.get('retry') is not False
            or grant.get('resume') is not False or grant.get('scope') not in ('signal', 'compile_validation')
            or grant.get('SOURCE') != source or grant.get('config_sha256') != config_sha256
            or grant.get('plan_sha256') != contract.digest(contract.plan())
            or grant.get('call_caps') != contract.plan()['demand']
            or grant.get('resources') != contract.plan()['resources']):
        raise ValueError('NEW_SEALED_SIGNAL_GRANT_REQUIRED')
    cpus = grant.get('allowed_cpus')
    if (not isinstance(cpus, list) or len(cpus) != 1 or type(cpus[0]) is not int
            or cpus[0] not in os.sched_getaffinity(0)):
        raise ValueError('EXPLICIT_ONE_CPU_ASSIGNMENT_REQUIRED')
    if (fresh.get('resources_idle_verified') is not True or
            fresh.get('profile_verified') is not True or
            fresh.get('protected_jobs_unchanged') is not True or
            fresh.get('source') != source or fresh.get('allowed_cpus') != cpus or
            fresh.get('config_sha256') != config_sha256 or
            fresh.get('external_RSS_monitor_ready') is not True or
            type(fresh.get('monotonic_seconds')) not in (int, float) or
            not 0 <= time.monotonic()-fresh['monotonic_seconds'] <= 30):
        raise ValueError('FRESH_RESOURCE_PROFILE_MONITOR_GATE_REQUIRED')
    if config.get('plan_sha256') != contract.digest(contract.plan()):
        raise ValueError('CONFIG_PLAN_CHANGED')
    if grant.get('scope') == 'compile_validation' and grant.get('frozen_model_sha256') != config.get('frozen_model_sha256'):
        raise ValueError('FROZEN_COST_MODEL_BINDING')
    return True


class Permit:
    def __init__(self, config, grant, output):
        self.config, self.grant, self.output = config, grant, output
        self.started = time.monotonic(); self.used = {}

    def require(self, scope):
        if self.grant['scope'] != scope:
            raise ValueError('EXECUTION_SCOPE')
        self.guard()

    def guard(self):
        ceiling = self.grant['resources']
        if time.monotonic()-self.started > ceiling['wall_seconds']: raise ValueError('WALL_CEILING')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > ceiling['RSS_bytes']:
            raise ValueError('RSS_CEILING')
        size = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
        if size > ceiling['output_bytes']: raise ValueError('OUTPUT_CEILING')

    def take(self, name):
        self.guard()
        if name not in self.grant['call_caps'] or self.used.get(name, 0) >= self.grant['call_caps'][name]:
            raise ValueError('CALL_CAP:'+name)
        self.used[name] = self.used.get(name, 0)+1
        with (self.output/'reservations.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=name, reserved=self.used[name]))+'\n'); f.flush(); os.fsync(f.fileno())

    def json_input(self, name): return read_json(self.config['input_mapping'][name])
    def read_output(self, name): return read_json(self.output/name)
    def phase(self, name): self.guard()
    def update(self, **values): self.guard()


def execute(config_path, grant_path, fresh_path, output):
    config, grant, fresh = map(read_json, (config_path, grant_path, fresh_path))
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    verify_grant(config, grant, config_sha256=sha(config_path), source=source, fresh=fresh)
    for root, hashes in ((ROOT, config['source_hashes']),
                          (Path(config['provider_root']), config['provider_hashes'])):
        fixed = source if root == ROOT else contract.PROVIDER
        for name, expected in hashes.items():
            if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('SOURCE_PATH')
            blob = subprocess.check_output(['git', 'show', fixed+':'+name], cwd=root)
            if hashlib.sha256(blob).hexdigest() != expected or sha(root/name) != expected:
                raise ValueError('SOURCE_BYTES_CHANGED:'+name)
    for receipt in config['files']:
        p = Path(receipt['path'])
        if p.is_symlink() or p.stat().st_size != receipt['bytes'] or sha(p) != receipt['sha256']:
            raise ValueError('INPUT_BYTES_CHANGED:'+receipt['logical_name'])
    for name, expected in config['runtime_file_hashes'].items():
        if sha(name) != expected: raise ValueError('RUNTIME_PROFILE_CHANGED')
    cpus = set(grant['allowed_cpus']); os.sched_setaffinity(0, cpus)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'QISKIT_NUM_PROCS', 'RAYON_NUM_THREADS'):
        if os.environ.get(name) != '1': raise ValueError('THREAD_ONE_ENV_REQUIRED:'+name)
    if Path(sys.executable).resolve() != Path(config['python']).resolve(): raise ValueError('PINNED_PYTHON_REQUIRED')
    output = Path(output)
    if (not output.is_absolute() or output.resolve() != output or output.exists()
            or not output.is_relative_to('/home/AbeHiromu/projects')):
        raise ValueError('NEW_EXCLUSIVE_OUTPUT_REQUIRED')
    if str(output) != config['outputs'][grant['scope']]: raise ValueError('OUTPUT_CONFIG_BINDING')
    lock = Path(config['locks'][grant['scope']])
    if not lock.is_absolute() or lock.resolve() != lock or not lock.is_relative_to('/home/AbeHiromu/projects'):
        raise ValueError('PRIVATE_ONE_SHOT_LOCK_PATH')
    # Lock is outside output, so choosing another output cannot spend twice.
    with lock.open('x') as f: f.write(source+'\n')
    output.mkdir()
    scratch=output/'scratch';scratch.mkdir()
    mpl=scratch/'matplotlib';mpl.mkdir()
    font=config['font_seed']
    if sha(font['path'])!=font['sha256']:raise ValueError('FONT_SEED_CHANGED')
    (mpl/font['name']).write_bytes(Path(font['path']).read_bytes())
    os.environ['MPLCONFIGDIR']=str(mpl);os.environ['NUMBA_CACHE_DIR']=str(scratch/'numba')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    with (output/'ONE_SHOT').open('x') as f: f.write(source+'\n')
    ceiling = grant['resources']
    resource.setrlimit(resource.RLIMIT_AS, (ceiling['AS_bytes'], ceiling['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU, (ceiling['CPU_seconds'], ceiling['CPU_seconds']))
    def timeout(*args): raise ValueError('WALL_CEILING')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(ceiling['wall_seconds'])
    permit = Permit(config, grant, output); writer = Writer(output, permit.guard)
    try:
        # Provider supplies the scientific trotterlib package; this checkout's
        # independent contract was already loaded without package __init__.
        sys.path.insert(0, str(Path(config['provider_root'])/'src'))
        sys.path.insert(0, str(ROOT/'scripts'))
        if grant['scope'] == 'signal':
            adapter = load('_v1_signal_adapter', ROOT/'scripts/track_a_v1_signal.py')
        else:
            adapter = load('_v1_compile_adapter', ROOT/'scripts/track_a_v1_compile.py')
        adapter.evaluate(config, permit, contract, writer)
        for receipt in config['files']:
            if sha(receipt['path']) != receipt['sha256']: raise ValueError('FINAL_INPUT_CHANGED')
        Writer(output).write('terminal.json', dict(status='COMPLETE_MANDATORY_STOP', exit_code=0,
            SOURCE=source, scope=grant['scope'], call_counts=permit.used, next_stage_authorized=False,
            config_sha256=sha(config_path)))
    except BaseException as exc:
        Writer(output).write('terminal.json', dict(status='STOP', error=str(exc),
            SOURCE=source, call_counts=permit.used, retry=False, next_stage_authorized=False))
        raise
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--execute-approved', action='store_true')
    for name in ('config', 'grant', 'fresh-gate', 'output'): p.add_argument('--'+name)
    args = p.parse_args()
    if args.execute_approved:
        if not all((args.config, args.grant, args.fresh_gate, args.output)): p.error('all sealed execution inputs required')
        execute(args.config, args.grant, args.fresh_gate, args.output)
    else:
        if any((args.config, args.grant, args.fresh_gate, args.output)): p.error('real input arguments need --execute-approved')
        print(json.dumps(contract.plan(), indent=2))


if __name__ == '__main__': main()
