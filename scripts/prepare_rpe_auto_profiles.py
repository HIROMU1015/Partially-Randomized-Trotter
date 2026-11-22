#!/usr/bin/env python3
"""Write next-run auto worker/timed-adapter profiles; no host contact or Q3.

The original scientific checkout and route views stay fixed. A separately fixed
common checkout supplies transport and the timed adapter. These changed adapter
profiles require ordinary dry-check before they may become eligible.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = ('scripts/run_rpe_optimizer_q3_job.py',
         'src/trottertracks/rpe_dp_prefix_reuse.py', 'src/trottertracks/rpe_q3_timing.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)


def prepare(profiles, layout, source, output, run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Use a simple new run identifier')
    source = subprocess.check_output(['git', 'rev-parse', source+'^{commit}'], cwd=ROOT, text=True).strip()
    optional = ('src/trottertracks/rpe_q3_evidence.py', 'src/trottertracks/rpe_q3_round_tables.py')
    files = FILES + tuple(name for name in optional if subprocess.check_output(
        ['git', 'ls-tree', source, '--', name], cwd=ROOT))
    manifest = dict(commit=source, files={name: hashlib.sha256(subprocess.check_output(
        ['git', 'show', source+':'+name], cwd=ROOT)).hexdigest() for name in files})
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(Path(profiles).glob('*.json')):
        profile = json.loads(path.read_text());host = profile['host_id']
        target = layout['hosts'][host]
        values = dict(source=source, short=source[:9], run_id=run_id)
        transport = target['transport_root'].format(**values)
        remote_profile = target['profile'].format(**values)
        spool = target['work_dir'].format(**values)
        before = deepcopy(profile)
        selected = [profile] + [rule['profile'] for rule in profile.get('route_profiles', [])]
        for value in selected:
            value['adapter_root'] = transport
            value['adapter_manifest'] = deepcopy(manifest)
        profile.update(work_dir=spool, slots=1, slot_policy=dict(mode='auto', initial_slots=1))
        put(output/'worker_profiles'/(host+'.json'), profile)
        worker = dict(host_id=host, root=transport, profile=remote_profile,
                      python=target.get('transport_python', 'python3'))
        for key in ('ssh_alias', 'transport', 'distribution', 'wsl_executable', 'participation_file'):
            if key in target:worker[key]=target[key]
        rows.append(dict(host_id=host, source=source, transport_root=transport,
                         worker_profile=remote_profile, spool=spool, worker=worker,
                         status='TIMED_ADAPTER_PROFILE_REQUIRES_FRESH_DRY_CHECK',
                         original_scientific_root=before['root'], route_views_unchanged=True))
    put(output/'DEPLOYMENT.json', dict(schema='RPE_AUTO_Q3_PROFILE_DEPLOYMENT_V1', source=source,
        adapter_manifest=manifest, hosts=rows, scientific_calls=0, queue_operations=0,
        note='Placement plan only; install fixed transport, copy generated profiles, then run ordinary exact-request dry checks. No old PASS is promoted to this changed adapter profile.'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--layout', type=Path, default=ROOT/'docs/research/rpe_q3_auto_allocation/HOST_LAYOUT.json')
    parser.add_argument('--source', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(args.profiles, json.loads(args.layout.read_text()), args.source, args.output, args.run_id)
    print(json.dumps(dict(output=str(args.output), hosts=len(rows), scientific_calls=0, queue_operations=0)))


if __name__ == '__main__':main()
