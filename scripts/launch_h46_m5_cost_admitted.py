"""One detached supervisor per approved DET contract; own-child admission only."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import h46_m5_cost_contract as c
import run_h6_m5_saved_phase as observed


def admission_check(configs, joint, available_ram, disk_free, own_cpus):
    c.require(len(configs) == 2 and {v['stage'] for v in configs} == {'H4_DET', 'H6_DET'}, 'TWO_DET_CONTRACTS_ONLY')
    c.require(joint['status'] == 'JOINT_ADMISSION_ACCEPTED_FOR_APPROVED_DET_SCOPE', 'JOINT_ADMISSION_NOT_ACCEPTED')
    reserved = 0
    used = set()
    for cfg in configs:
        stage = cfg['stage']
        cpus = cfg['resources']['admission_cpu_set']
        c.require(cpus and len(cpus) == len(set(cpus)) and all(type(x) is int for x in cpus), 'EXPLICIT_OWN_CHILD_CPU_SET')
        c.require(set(cpus) <= set(own_cpus) and not (used & set(cpus)), 'CPU_SETS_SUBSET_AND_DISJOINT')
        used.update(cpus)
        c.require(cpus == joint['allocation'][stage], 'EXACT_JOINT_CPU_ALLOCATION')
        c.require(cfg['call_scope']['compile_axes'] == {'H4_DET': 38, 'H6_DET': 54}[stage], 'APPROVED_38_54_AXIS_SCOPE')
        reserved += len(cpus) * cfg['resources']['worker_RSS_proxy_bytes']
    c.require(reserved == joint['reserved_proxy_bytes'] and available_ram >= reserved, 'JOINT_RAM_ADMISSION')
    c.require(disk_free >= joint['scratch_free_proxy_bytes'], 'JOINT_SCRATCH_ADMISSION')
    return dict(observed_at_unix=time.time(), effective_available_RAM_bytes=available_ram,
                reserved_proxy_bytes=reserved, disk_free_bytes=disk_free,
                proxy_is_upper_bound=False, host_settings_changed=False,
                resource_effect='OWN_NEW_CHILD_AFFINITY_ONLY', allocation=joint['allocation'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not args.launch:
        print('SCOPE_ONLY_NO_LAUNCH')
        return 0
    configs = [c.read(path) for path in args.config]
    for cfg in configs:
        c.authorize(cfg)
        c.require(not Path(cfg['run_root']).exists() and not any(Path(p).exists() for p in cfg['supervisor'].values()), 'UNUSED_INDEPENDENT_STAGE')
        resources = cfg['resources']
        path = resources['joint_admission_receipt_path']
        c.require(path in {r['path'] for r in cfg['sealed_files']} and c.sha(path) == resources['joint_admission_receipt_sha256'], 'JOINT_ADMISSION_RECEIPT_SEALED')
    paths = {v['resources']['joint_admission_receipt_path'] for v in configs}
    c.require(len(paths) == 1, 'SAME_RESOURCE_RECEIPT_SEPARATE_SCIENCE_GRANTS')
    path = Path(next(iter(paths)))
    joint = c.read(path)
    admission = admission_check(configs, joint, observed.available_memory(), shutil.disk_usage(path.parent).free, os.sched_getaffinity(0))
    # This marker authorizes no scientific calls; the independent native runners
    # consume their own grants/markers after their own fresh gates.
    c.save(path.parent / 'DET_LAUNCH_DISPATCHED_PRIVATE.json', dict(admission=admission,
           configs=[dict(stage=v['stage'], sha256=c.sha(p)) for v, p in zip(configs, args.config)], retry=False))
    for cfg, config_path in zip(configs, args.config):
        cpus = set(cfg['resources']['admission_cpu_set'])
        log = Path(cfg['run_root']).parent / 'supervisor_stdout_PRIVATE.log'
        with log.open('xb') as stream:
            process = subprocess.Popen([cfg['runtime']['python'], '-B', str(c.ROOT / 'scripts/run_h46_m5_cost.py'),
                                        '--supervise', '--config', str(Path(config_path).resolve())],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        row = dict(stage=cfg['stage'], supervisor_pid=process.pid, SOURCE=cfg['SOURCE'],
                   plan_sha256=cfg['plan_sha256'], cpu_set=sorted(cpus), retry=False,
                   next_phase_authorized=False, completion_not_yet_verified=True)
        c.save(Path(cfg['run_root']).parent / 'detached_supervisor_PRIVATE.json', row)
        print(__import__('json').dumps(row), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
