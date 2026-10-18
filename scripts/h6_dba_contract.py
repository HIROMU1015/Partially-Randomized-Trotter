"""H6 DBA one-shot contracts and receipts; no scientific imports."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'H6_DBA_20261018'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'DUPLICATE_JSON_KEY'); out[k] = v
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def save(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


class Writer:
    def __init__(self, directory):
        self.directory = Path(directory)
    def write(self, name, value):
        require(Path(name).name == name, 'EXCLUSIVE_OUTPUT_NAME')
        save(self.directory/name, value)


def cell(prefix, q, r, K):
    return dict(id=f'B2_p{prefix}_q{q}_2nd_r{r}_K{K}', method='B2',
                prefix=prefix, q=q, formula='2nd', r=r, K=K, R=q*r)


def plan():
    # Targeted new boundaries, not a Cartesian expansion or global search.
    a = [cell(p, q, 4, K) for p in (3, 4) for q in (2, 4, 8) for K in (2, 4)]
    a += [cell(5, q, 8, 4) for q in (4, 8)]
    a += [cell(5, 16, 2, 2), cell(4, 8, 8, 4)]
    return dict(scope=SCOPE, system='linear_H6_1Angstrom_STO3G_tol_only_DF19', T=.8,
        D=dict(cell_id='B0I_p11_q4_4th', axes=['cosine', 'sine'],
               prediction_source='saved_axis_costs', relative_error_denominator='measured',
               STOP_at_or_above='1/50', native_compile=2, control_probe=4),
        B=dict(groups=[dict(id='K2_s2_multi', cell=cell(5, 8, 2, 2), stratum='s2_multi'),
                       dict(id='K4_s4', cell=cell(5, 8, 2, 4), stratum='s4')],
               n=8, native_compile=32, control_probe=8,
               first_trajectory_control_before_remaining=True,
               conditional_cost_sampling_only=True, K4_s2_multi='UNRESOLVED_NOT_MEASURED'),
        A=dict(cells=a, signal=16, native_compile=0, sampling=0,
               epsilons=[.05,.01,.005,.001,.0005], allowance_factors=[1.],
               bias_rules=['USUAL_SIGNED','EMPIRICAL_NONCANCELLATION'],
               candidate_cost_stage='NOT_AUTOMATIC'),
        retry=False, refit=False, new_SCF_DF_state=False, cost_reaccounting=False,
        mandatory_STOP_per_stage=True, population_mean_certified=False,
        resource_policy='fresh affinity/RAM; transient PSI and unrelated worktree changes are observations')


def axis_check(prediction, measured):
    require(type(measured) is int and measured > 0, 'POSITIVE_INTEGER_MEASURED_RZ')
    prediction = Fraction(prediction)
    require(prediction > 0, 'POSITIVE_PREDICTION')
    error = abs(prediction-measured)/measured
    return dict(exact_relative_error=str(error), relative_error=float(error),
                PASS=error < Fraction(1,50), denominator='measured')


def seed(stage, candidate, stratum, replica, binding):
    require(type(replica) is int and 0 <= replica < 8, 'REPLICA')
    key = [SCOPE, stage, 'H6', binding, '.8', candidate, stratum, replica]
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8], 'big')


def fresh():
    mem = dict((r.split(':')[0],int(r.split()[1])*1024)
               for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r and r.split()[1].isdigit())
    return dict(time=time.time(), affinity=sorted(os.sched_getaffinity(0)),
                available_memory_bytes=mem['MemAvailable'],
                memory_pressure=Path('/proc/pressure/memory').read_text(),
                disk_free_bytes=__import__('shutil').disk_usage(ROOT).free)


def verify(config, grant, stage, *, full_runtime=True):
    require(stage in ('D','B','A') and grant.get('stage') == stage, 'STAGE')
    require(grant.get('approved_by_user') is True and grant.get('one_shot') is True
            and grant.get('retry') is False and grant.get('next_stage_authorized') is False,
            'EXPLICIT_SCOPED_EXECUTION_APPROVAL_REQUIRED')
    require(grant['plan_digest'] == digest(plan()) and grant['config_digest'] == digest(config), 'FIXED_CONTRACT')
    source = grant['SOURCE']
    require(len(source) == 40, 'SEALED_SOURCE')
    for name, h in config['source_hashes'].items():
        require(sha(ROOT/name) == h, 'SOURCE_FILE_CHANGED:'+name)
        raw = subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        require(hashlib.sha256(raw).hexdigest() == h, 'SOURCE_GIT_BINDING:'+name)
    for row in config['inputs']:
        require(Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256'],
                'INPUT_CHANGED:'+row['role'])
    for name, h in config['provider_hashes'].items():
        require(sha(Path(config['provider_root'])/name) == h, 'PROVIDER_CHANGED:'+name)
    if full_runtime:
        for name, h in config['runtime_file_hashes'].items():
            require(sha(name) == h, 'RUNTIME_CHANGED:'+name)
    require(set(grant['worker_cpus']) <= os.sched_getaffinity(0), 'CPU_AFFINITY')
    f = fresh()
    # Admission estimate, not an RSS cap or historical resource ceiling.
    require(f['available_memory_bytes'] > grant['memory_admission_estimate_bytes'], 'INSUFFICIENT_MEMORY_TO_START')
    require(f['disk_free_bytes'] > 2**30, 'INSUFFICIENT_OUTPUT_SPACE')
    return dict(status='PASS', SOURCE=source, inputs=len(config['inputs']),
                provider_files=len(config['provider_hashes']),
                runtime_files=len(config['runtime_file_hashes']) if full_runtime else 0,
                fresh=f)
