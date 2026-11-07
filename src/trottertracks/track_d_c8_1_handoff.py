"""Stopped C8-1 task inheritance and stage-aware CPU reservation only."""
import hashlib
import json
from pathlib import Path

OLD_SOURCE='6b1d613ded963b7851afa1c07ba8c646124dd44f'
METHODS=('symmetric_directional','reference')
AXES=('X','Y')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def prior_plan(old):
    """Seal already-complete inputs while the original four compiles drain."""
    old=Path(old)
    if read(old/'SOURCE.json')['execution_source']!=OLD_SOURCE:raise ValueError('ORIGINAL_SOURCE')
    d=old/'local/q64_sample00'
    if read(d/'precompile_checks.json')['status']!='PASS':raise ValueError('ORIGINAL_RAW_REQUIRED')
    members=[old/p for p in ('SOURCE.json','seed_plan.json','frozen_models.json','event_bindings.json','local/car_context.npz')]
    for event in read(old/'event_bindings.json'):
        p=old/'local/events'/(event['sample_id']+'.json')
        if sha(p)!=event['record_sha256']:raise ValueError('SEALED_EVENT_CHANGED')
        members.append(p)
    members.extend(d/p for p in ('precompile_checks.json','conditions.json','fusion.json','references.npz','structural_features_raw.json'))
    members.extend((d/'circuits').glob('*_raw.qpy'))
    return dict(original_source=OLD_SOURCE,raw_reused=['q64_sample00'],original_compile_tasks=4,
        sampling_remaining=0,scope=dict(q=64,cut=32,paths=8,occurrences=512,events=16384,total_compile_limit=32),
        members=[dict(path=str(p.relative_to(old)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(members))])


def inheritance_plan(old):
    old=Path(old);result=read(old/'result.json');stop=read(old/'STOP.json')
    if (result['source_commit']!=OLD_SOURCE or not stop['mandatory_stop'] or
        read(old/'failure.json')['error_type']!='KeyboardInterrupt' or
        not result['source_input_preserved'] or result['canonical_occurrences_started']!=512 or
        result['events_sealed']!=8):raise ValueError('DRAINED_C8_SOURCE_INPUT_STOP_REQUIRED')
    events=read(old/'event_bindings.json')
    ids=[f'q64_sample{i:02d}' for i in range(8)]
    if [e['sample_id'] for e in events]!=ids:raise ValueError('FIXED_EVENT_COHORT')
    members=[old/f for f in ('SOURCE.json','result.json','STOP.json','failure.json',
        'seed_plan.json','frozen_models.json','event_bindings.json')]
    members.append(old/'local/car_context.npz')
    for e in events:
        p=old/'local/events'/(e['sample_id']+'.json')
        if sha(p)!=e['record_sha256']:raise ValueError('SEALED_EVENT_CHANGED')
        members.append(p)
    raw_reused=[];compiles=[];posts=[]
    for sid in ids:
        directory=old/'local'/sid
        if not directory.exists():continue
        check=directory/'precompile_checks.json'
        if not check.exists() or read(check)['status']!='PASS':raise ValueError('INCOMPLETE_OR_FAILED_RAW')
        if (directory/'failure.json').exists():raise ValueError('RAW_FAILURE_NOT_RETRYABLE')
        raw_reused.append(sid)
        for method in METHODS:
            for axis in AXES:
                name=method+'_'+axis;marker=directory/(name+'_compile.json');compiled=directory/'circuits'/(name+'_compiled.qpy')
                if marker.exists() or compiled.exists():
                    if not marker.exists() or not compiled.exists() or read(marker)['status']!='PASS':raise ValueError('STARTED_INCOMPLETE_OR_FAILED_COMPILE')
                    if read(marker)['compile_calls']!=1:raise ValueError('ORIGINAL_COMPILE_ACCOUNTING')
                    compiles.append([sid,method,axis])
                post=directory/(name+'_check.json')
                if post.exists():
                    if read(post)['status']!='PASS':raise ValueError('POST_FAILURE_NOT_RETRYABLE')
                    posts.append([sid,method,axis])
                if any(directory.glob(name+'_*failure.json')):raise ValueError('STAGE_FAILURE_NOT_RETRYABLE')
        members.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.qpy','.npz') and 'matplotlib-cache' not in p.parts)
    if raw_reused!=['q64_sample00'] or len(compiles)!=4:
        raise ValueError('AUTHORIZED_HANDOFF_SCOPE_RAW1_COMPILE4')
    if result['compile_calls_started']!=len(compiles):raise ValueError('OLD_COMPILE_COUNT')
    return dict(original_source=OLD_SOURCE,reason='USER_AUTHORIZED_DRAIN_AND_TASK_HANDOFF',
        original_controller_status=result['status'],original_STOP_preserved=True,
        scope=dict(q=64,cut=32,paths=8,occurrences=512,events=16384,compile_limit_total=32),
        sampling_remaining=0,raw_reused=raw_reused,raw_remaining=ids[1:],
        compiles_reused=compiles,postchecks_reused=posts,compile_remaining=32-len(compiles),postcheck_remaining=32-len(posts),
        members=[dict(path=str(p.relative_to(old)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(members))])


def verify(old,packet):
    for row in packet['members']:
        p=Path(old)/row['path']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('INHERITED_EVIDENCE_CHANGED:'+row['path'])


def allocation(allocator,cpu,memory,pending,workers,active):
    # The old allocator considers method=reference heavy even for P0 compile.
    # Normalize only scheduling proxies. Preserve the actual method/task object.
    proxies=[dict(t,kind='raw' if t['stage'] in ('raw','post') else 'compiled',
        method='symmetric_directional',index=i) for i,t in enumerate(pending)]
    return [(pending[p['index']],threads) for p,threads in allocator(cpu,memory,proxies,workers,active)]
