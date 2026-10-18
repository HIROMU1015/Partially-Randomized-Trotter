"""Independent H8 A/B scope, grants and immutable receipts; stdlib only."""
from h6_dba_contract import ROOT, require, digest, sha, read, save, Writer, fresh, cell
import hashlib
import os
from pathlib import Path
import subprocess

SCOPE = 'H8_AB_BOUNDARY_RARE_20261018'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'


def plan():
    cells = [cell(p, q, 4, K) for p in (6, 8) for q in (2, 4, 8) for K in (2, 4)]
    cells += [cell(7, q, 8, 4) for q in (4, 8)]
    cells += [cell(7, 16, 4, 4), cell(6, 8, 8, 4)]
    groups = [dict(id=f'q{q}_{s}', cell=cell(7, q, 4, 4), stratum=s)
              for q in (4, 8) for s in ('s2_multi', 's4')]
    return dict(scope=SCOPE, system='linear_H8_1Angstrom_STO3G_tol_only_DF28_sector4900', T=.8, P=0,
        A=dict(cells=cells, signal=16, new_full_reference=0, compile=0, sampling=0,
            epsilons=[.05, .01, .005, .001, .0005], allowance_factors=[1.],
            bias_rules=['USUAL_SIGNED', 'EMPIRICAL_NONCANCELLATION'],
            new_time_blocks=list(range(8)), new_times=[.025, -.025], probes_per_time=3,
            saved_outer=[dict(prefix=7, q=q) for q in (4, 8)],
            new_outer=[dict(prefix=p, q=q) for p in (6, 8) for q in (2, 4, 8)]
                      + [dict(prefix=7, q=16)], new_outer_pair_count=7,
            automatic_cost_stage=False),
        B=dict(groups=groups, n=8, trajectory=32, compile=64, control_probe=16,
            independent_event_oracle=4, native_evolution=32, occurrence=192, RTE_event=768,
            first_trajectory_control_before_remaining=True,
            conditional_cost_sampling_only=True, original_quantum_event_weights_unchanged=True,
            original_n32_mixed=False, population_mean_certified=False),
        immutable_scope=['used_SOURCE', 'inputs', 'provider', 'runtime', 'grant', 'ledger', 'results'],
        unrelated_worktree_changes='observe_only', transient_PSI='observe_only',
        retry=False, refit=False, new_SCF_DF_state=False, regenerate_Gaussian_operations=False,
        cost_reaccounting=False, automatic_next_stage=False, mandatory_STOP_per_stage=True)


def seed(candidate, stratum, replica, input_hash):
    require(type(replica) is int and 0 <= replica < 8, 'REGISTERED_REPLICA')
    return int(digest([SCOPE, 'B', 'H8', input_hash, '.8', candidate, stratum, replica])[:16], 16)


def verify(config, grant, stage):
    require(stage in ('A', 'B') and grant.get('stage') == stage and grant.get('scope') == SCOPE,
            'EXACT_H8_AB_SCOPE')
    require(grant.get('approved_by_user') is True and grant.get('one_shot') is True
            and grant.get('retry') is False and grant.get('next_stage_authorized') is False,
            'EXPLICIT_SCOPED_EXECUTION_APPROVAL_REQUIRED')
    require(config.get('stage') == stage, 'STAGE_SPECIFIC_CONFIG')
    require(grant['plan_digest'] == digest(plan()) and grant['config_digest'] == digest(config),
            'PLAN_CONFIG_BINDING')
    require(config['provider_source'] == PROVIDER and config['compiler_hash'] == COMPILER
            and config['profile_sha256'] == PROFILE, 'FIXED_H8_PROVIDER_COMPILER_PROFILE')
    for name, expected in config['source_hashes'].items():
        require(sha(ROOT / name) == expected, 'SOURCE_CHANGED:' + name)
        raw = subprocess.check_output(['git', 'show', grant['SOURCE'] + ':' + name], cwd=ROOT)
        require(hashlib.sha256(raw).hexdigest() == expected, 'GIT_SOURCE_BINDING:' + name)
    for row in config['inputs']:
        require(Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256'],
                'INPUT_CHANGED:' + row['role'])
    for name, expected in config['provider_hashes'].items():
        require(sha(Path(config['provider_root']) / name) == expected, 'PROVIDER_CHANGED:' + name)
    for path, expected in config['runtime_file_hashes'].items():
        require(sha(path) == expected, 'RUNTIME_CHANGED:' + path)
    cpus = grant['worker_cpus']
    require(cpus and len(cpus) == len(set(cpus)) and all(type(i) is int for i in cpus)
            and set(cpus) <= os.sched_getaffinity(0), 'FRESH_WORKER_AFFINITY')
    observed = fresh()
    require(observed['available_memory_bytes'] > grant['memory_admission_estimate_bytes'], 'MEMORY_ADMISSION')
    require(observed['disk_free_bytes'] > 2**30, 'OUTPUT_SPACE_ADMISSION')
    return dict(status='PASS', SOURCE=grant['SOURCE'], stage=stage, inputs=len(config['inputs']),
        provider_files=len(config['provider_hashes']), runtime_files=len(config['runtime_file_hashes']),
        fresh=observed)
