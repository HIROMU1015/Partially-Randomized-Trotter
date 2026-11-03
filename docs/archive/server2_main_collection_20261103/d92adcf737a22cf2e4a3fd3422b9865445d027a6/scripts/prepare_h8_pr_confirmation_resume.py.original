"""Bind a quiescent technical-STOP origin to a new parallel continuation.

Only fully completed, exit0/reaped trajectories with original controls are
accepted. Native failures are never accepted. Old STOPs, reservations and seeds
are immutable. This command creates an invalid grant, never a scientific run.
"""
from copy import deepcopy
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_pr_confirmation as h
c = h.c


def row(path, role):
    path = Path(path)
    c.require(path.is_file() and not path.is_symlink(), 'REGULAR_ORIGIN_EVIDENCE')
    return dict(role=role, path=str(path), bytes=path.stat().st_size, sha256=c.sha(path))


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, starttime=int(fields[19]), uid=p.stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def retired(expected):
    observed = identity(expected['pid'])
    c.require(observed != {k: expected[k] for k in ('pid', 'starttime', 'uid')}, 'ORIGIN_STILL_LIVE')


def collect(origin_path, origin_root, parent_wait, parent_identity):
    retired(parent_identity)
    origin = c.read(origin_path)
    c.verify_source(Path(origin_root), origin['SOURCE'], origin['source_hashes'])
    c.verify_byte_rows(origin['inputs'])
    wait = c.read(parent_wait)
    c.require(wait['pid'] == parent_identity['pid'] and wait['reaped'] is True
              and type(wait['exit_code']) is int, 'ORIGIN_PARENT_NATIVE_WAIT')
    output = Path(origin['output'])
    stop = c.read(output / 'STOP.json')
    c.require(stop['reason'] in ('ValueError:NATIVE_START_IDENTITY', 'KeyboardInterrupt:OWN_RUN_SIGNAL_15'),
              'ONLY_KNOWN_TECHNICAL_STOP_OR_EXPLICIT_HANDOFF')
    receipt_path = output / 'workers/parallel_receipt.json'
    receipt = c.read(receipt_path)
    c.require(receipt['all_reaped'] is True and not receipt['cleanup_errors']
              and receipt['first_error'] in ('NATIVE_START_IDENTITY', 'OWN_RUN_SIGNAL_15'),
              'ORIGIN_CLEAN_NATIVE_MAP')
    registry = c.read(origin['registry'])
    by_id = {t['id']: t for t in registry['tasks'] if t['scope'] == origin['scope']}
    accepted, previous_events = {}, {}
    files = [row(origin_path, 'resume:origin_spec'), row(parent_wait, 'resume:parent_native_wait'),
             row(output / 'STOP.json', 'resume:original_STOP'), row(receipt_path, 'resume:map_native_wait')]
    reservations = []
    seen = set()
    for handle in receipt['workers']:
        key = handle['task_id']
        c.require(key in by_id and key not in seen and handle['reaped'] is True, 'REGISTERED_ORIGIN_TASK')
        seen.add(key)
        directory = Path(handle['directory'])
        c.require(directory.resolve().parent == (output / 'workers').resolve()
                  and directory.name == key and not directory.is_symlink(), 'OWN_ORIGIN_TASK_DIRECTORY')
        ident = h.native_identity(handle['identity'])
        c.require(ident[0] == handle['pid'] and ident[2] == parent_identity['pid']
                  and ident[3] == os.getuid(), 'ORIGIN_NATIVE_PARENT_UID')
        c.require(identity(ident[0]) != dict(pid=ident[0], starttime=ident[1], uid=ident[3]), 'ORIGIN_WORKER_STILL_LIVE')
        trace_path = directory / 'trajectory.json'
        if trace_path.exists():
            trace = c.read(trace_path)
            c.require(trace['seed'] == by_id[key]['seed'] and trace['cell'] == by_id[key]['cell']
                      and trace['SOURCE'] == origin['SOURCE'] and trace['cohort'] == h.COHORT
                      and c.digest(trace['events']) == trace['event_digest'], 'ORIGIN_EVENT_STREAM')
            previous_events[key] = trace['event_digest']
            files.append(row(trace_path, 'resume:prior_event:' + key))
        for path in sorted(directory.glob('*_reserved.json')):
            reservations.append(row(path, 'resume:task_charge:' + key + ':' + path.name))
        if handle['exit_code'] != 0 or not handle.get('result_sha256'):
            continue
        h.accept(by_id[key], handle, origin)
        accepted[key] = handle
        for name in ('task_result.json', 'started.json', 'trajectory.json', 'measured_cosine.json', 'measured_sine.json'):
            files.append(row(directory / name, 'resume:accepted:' + key + ':' + name))
        if by_id[key]['replica'] == 0:
            files.append(row(directory / 'control.json', 'resume:accepted:' + key + ':control'))
    for path in sorted(output.glob('*_reserved.json')):
        reservations.append(row(path, 'resume:parent_charge:' + path.name))
    c.require(not accepted or by_id[next(iter(accepted))]['replica'] == 0
              or any(by_id[k]['replica'] == 0 for k in accepted), 'CONTROL_BEFORE_REUSED_DEPENDENTS')
    unique = {r['path']: r for r in files + reservations}
    return dict(origin_spec=str(origin_path), origin_root=str(origin_root),
        origin_SOURCE=origin['SOURCE'], seed_derivation_SOURCE=origin.get('seed_derivation_SOURCE', origin['SOURCE']),
        registry_sha256=c.sha(origin['registry']), scope=origin['scope'], accepted=accepted,
        previous_event_digests=previous_events, old_reservations=reservations,
        failed_or_unfinished_charge_refunded=False, original_STOP_preserved=True,
        parent_identity=parent_identity, parent_wait=str(parent_wait), files=list(unique.values()))


def verify_manifest(spec):
    manifest = c.read(spec['resume_manifest'])
    contents = dict(manifest)
    digest = contents.pop('manifest_digest')
    c.require(c.digest(contents) == digest and manifest['quiescence_observations'] >= 2
              and manifest['scope'] == spec['scope']
              and manifest['registry_sha256'] == c.sha(spec['registry'])
              and manifest['seed_derivation_SOURCE'] == spec['seed_derivation_SOURCE'], 'RESUME_MANIFEST_BINDING')
    c.verify_byte_rows(manifest['files'])
    observed = collect(manifest['origin_spec'], manifest['origin_root'],
                       manifest['parent_wait'], manifest['parent_identity'])
    c.require(all(observed[k] == manifest[k] for k in observed), 'RECHECK_ORIGIN_ACCEPTANCE')
    return manifest


def prepare(origin_path, origin_root, parent_wait, parent_identity, destination, source):
    destination = c.home_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    first = collect(origin_path, origin_root, parent_wait, parent_identity)
    time.sleep(.05)
    second = collect(origin_path, origin_root, parent_wait, parent_identity)
    c.require(first == second, 'STABLE_ORIGIN_PROOF')
    manifest = dict(second, quiescence_observations=2)
    manifest['manifest_digest'] = c.digest(manifest)
    path = destination / 'resume_manifest.json'
    c.save(path, manifest)
    origin = c.read(origin_path)
    tasks = [t for t in c.read(origin['registry'])['tasks']
             if t['scope'] == origin['scope'] and t['id'] not in manifest['accepted']]
    calls = dict(trajectory=len(tasks), native_evolution=len(tasks), wrapper_compile=2*len(tasks),
        occurrence=sum(t['cell']['q'] for t in tasks), RTE_event=sum(t['cell']['R'] for t in tasks),
        control_probe=4*sum(t['replica']==0 for t in tasks),
        independent_event_oracle=sum(t['replica']==0 for t in tasks))
    spec = deepcopy(origin)
    spec.update(SOURCE=source, seed_derivation_SOURCE=manifest['seed_derivation_SOURCE'],
        resume_manifest=str(path), source_hashes={p:c.sha(h.ROOT/p) for p in h.FILES},
        parallel_scheduling='H8_CONFIRMATION_GROUPED_PROBES_AXES_V1',
        output=str(destination/'output'), lock=str(destination/'ONE_SHOT.json'),
        calls={k:v for k,v in calls.items() if v}, approved=False, runtime_authorization=False,
        technical_sealed=True, allowed_cpus=[], absolute_launch_command=None,
        output_estimate_bytes=1024*2**20, inode_estimate=20000)
    spec['inputs'] += manifest['files'] + [row(path, 'resume:manifest')]
    spec['technical_digest'] = h.technical_digest(spec)
    c.save(destination/'technical_seal.json',spec)
    h.verify(spec)
    c.save(destination/'grant_INVALID.json',dict(scope=spec['scope'],SOURCE=source,
        technical_digest=spec['technical_digest'],approved=False,approved_by_user=False,
        runtime_authorization=False,allowed_cpus=[],retry=False,next_stage_authorized=False))
    c.save(destination/'preparation_receipt.json',dict(status='PASS_TECHNICAL_RESUME_ONLY',
        SOURCE=source, accepted_tasks=len(manifest['accepted']), new_calls=spec['calls'],
        new_master_seed_derivation=0, scientific_calls=0, valid_grants_created=0,
        old_reservations_preserved=len(manifest['old_reservations'])))
    return spec


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin-spec',required=True,type=Path)
    parser.add_argument('--origin-root',required=True,type=Path)
    parser.add_argument('--parent-wait',required=True,type=Path)
    parser.add_argument('--parent-identity',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    prepare(args.origin_spec,args.origin_root,args.parent_wait,c.read(args.parent_identity),args.destination,args.source)
