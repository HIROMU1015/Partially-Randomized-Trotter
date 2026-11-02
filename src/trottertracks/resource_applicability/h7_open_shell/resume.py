"""Byte-only admission of completed H7 groups. No scientific recalculation."""
import json
import math
import re
from pathlib import Path
from .contract import digest,target,draft,COMPILER_HASH,tasks,call_limits
from ..h5_open_shell.execution import file_hash

ORIGIN='f93ad35b05a75c5e17e2c53b0f3de2af4fc443f2'
TRANSITION_FILES={'src/trottertracks/resource_applicability/h7_open_shell/execution.py',
 'src/trottertracks/resource_applicability/h7_open_shell/contract.py','tests/test_h7_open_shell.py'}


def read(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>32*2**20:raise ValueError('RESUME_RECORD_LAYOUT')
    return json.loads(path.read_bytes())


def task_key(task):
    return digest({k:v for k,v in task.items() if k!='_cpu'})


def parent_inventory(parent):
    return {str(p.relative_to(parent)):file_hash(p) for p in sorted(Path(parent).rglob('*')) if p.is_file()}


def check_old_processes(identities):
    for identity in identities:
        path=Path(f"/proc/{identity['pid']}/stat")
        try:raw=path.read_text()
        except FileNotFoundError:continue
        fields=raw[raw.rfind(')')+2:].split()
        if int(fields[19])==identity['start_ticks'] and fields[0]!='Z':raise ValueError('OLD_OWN_RUN_STILL_LIVE')


def admit_parent(root,parent):
    root=Path(root);parent=Path(parent).resolve();out=parent/'output01'
    if not parent.is_relative_to('/home/AbeHiromu'):raise ValueError('PARENT_HOME')
    manifest=read(parent/'manifest.json');native=read(parent/'native_wait_v1.json');terminal=read(out/'terminal.json')
    from .contract import verify_grant
    verify_grant(manifest,read(parent/'authorization.json'),ORIGIN)
    if digest(manifest['profile'])!=manifest['profile_digest']:raise ValueError('PARENT_PROFILE_DIGEST')
    if manifest['SOURCE']!=ORIGIN or native['SOURCE']!=ORIGIN or terminal['SOURCE']!=ORIGIN:raise ValueError('PARENT_SOURCE')
    if (not native['wait_completed'] or native['exit_code']!=1 or terminal['status']!='STOP'
        or 'PROCESS_MEMORY_MISSING' not in terminal['first_stop'] or not terminal['all_workers_reaped']
        or terminal['cleanup_errors'] or not all(r['reaped'] for r in terminal['retired'])):
        raise ValueError('PARENT_STOP_OR_CLEANUP')
    check_old_processes([native['identity'],*[r['identity'] for r in terminal['retired']]])
    if manifest['target_digest']!=digest(target()) or manifest['compiler_hash']!=COMPILER_HASH or manifest['contract_digest']!=digest(draft()):
        raise ValueError('PARENT_SCIENTIFIC_CONTRACT')
    for name,sha in manifest['source_hashes'].items():
        if name not in TRANSITION_FILES and file_hash(root/name)!=sha:raise ValueError('NUMERICAL_SOURCE_CHANGED:'+name)
    saved=read(out/'input/group_000/result.json');native_input=read(out/'input/group_000/terminal.json')
    binding=read(out/'realized_input_binding.json');snapshot=saved['snapshot'];rank=saved['DF']['actual_rank']
    if file_hash(snapshot['path'])!=snapshot['sha256'] or binding['snapshot']!=snapshot:raise ValueError('FROZEN_SNAPSHOT')
    if binding['SOURCE']!=ORIGIN or binding['projection_digest']!=digest(saved['DF']):raise ValueError('FROZEN_PROJECTION')
    original_input=digest(dict(snapshot=snapshot['sha256'],projection=saved['DF'],source=ORIGIN))
    if binding['input_digest']!=original_input or binding['profile_digest']!=manifest['profile_digest'] or binding['compiler_hash']!=COMPILER_HASH:
        raise ValueError('ORIGINAL_INPUT_BINDING')
    exits={(r['pid'],r['identity']['start_ticks']):r for r in terminal['retired']}
    def successful_native(record):
        identity=record['identity'];retired=exits.get((identity['pid'],identity['start_ticks']))
        if record['status']!='COMPLETE' or not retired or retired['exit_code']!=0 or not retired['reaped']:
            raise ValueError('GROUP_NATIVE_EXIT')
    successful_native(native_input)
    checks=saved['state_checks']
    if not checks['doublet_engineering_check'] or checks['S_squared_target_residual']>1.1e-9:raise ValueError('PARENT_STATE_CHECKS')
    initial=tasks(rank,ORIGIN,original_input)
    if binding['initial_tasks']!=initial:raise ValueError('ORIGINAL_SEEDS')
    eligible=[r['cell']['id'] for r in saved['signal_records'] if r['cell']['R'] and
        any(v['1.0']['cost_acquisition_eligible'] for v in r['precision_rows'].values())]
    extension=tasks(rank,ORIGIN,original_input,extension=True,eligible=eligible)
    all_tasks=initial+extension;expected={task_key(t):t for t in all_tasks}
    stages={'initial_heavy':[t for t in initial if t['heavy']],
        'initial_normal':[t for t in initial if not t['heavy']],'extension':extension}
    complete={};incomplete=[];refs=[]
    for stage,registered in stages.items():
        directory=out/stage
        if read(directory/'reservations.json')['tasks']!=registered:raise ValueError('PARENT_REGISTRATION')
        for group in sorted(directory.glob('group_*')):
            if not group.is_dir() or not re.fullmatch(r'group_\d{3}',group.name):raise ValueError('GROUP_DIRECTORY')
            index=int(group.name.split('_')[1])
            if index>=len(registered):raise ValueError('UNREGISTERED_GROUP')
            task=registered[index];key=task_key(task)
            if not (group/'result.json').exists() or not (group/'terminal.json').exists():
                incomplete.append(task);continue
            result=read(group/'result.json');done=read(group/'terminal.json')
            if done['status']!='COMPLETE':incomplete.append(task);continue
            successful_native(done)
            if task_key(result['task'])!=key or result['task']!=done['task'] or key in complete:raise ValueError('RESULT_TASK_BINDING')
            if len(result['rows'])!=2 or {r['axis'] for r in result['rows']}!={'cosine','sine'}:raise ValueError('PAIRED_RESULTS')
            if digest(result['events'])!=result['event_digest']:raise ValueError('EVENT_BYTES')
            for row in result['rows']:
                if row['compiler_hash']!=COMPILER_HASH or row['event_digest']!=result['event_digest']:raise ValueError('RESULT_COMPILER_EVENTS')
                if not re.fullmatch('[0-9a-f]{64}',row['fingerprint']):raise ValueError('RESULT_FINGERPRINT')
                if set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'} or any(type(v) is not int or v<0 for v in row['metrics'].values()):raise ValueError('RESULT_METRICS')
            used=dict.fromkeys(call_limits(rank),0);used.update(compile=2,trajectory=1 if task['cell']['R'] else 0,
                sampling_call=1 if task['cell']['R'] else 0,occurrence=task['cell']['R'] or 0,control_probe=4 if task['validate'] else 0)
            if result['calls_attempted']!=used:raise ValueError('RESULT_CALL_COUNTS')
            if task['validate'] and (not result['validation'] or not math.isfinite(result['validation']['max_error']) or result['validation']['max_error']>1.1e-9):raise ValueError('INITIAL_NATIVE_VALIDATION')
            complete[key]=result;refs.append(dict(task_key=key,directory=str(group),result_sha256=file_hash(group/'result.json'),
                terminal_sha256=file_hash(group/'terminal.json'),late_completion=not any(result in read(out/s/'summary.json')['completed'] for s in stages)))
    if not all(task_key(t) in complete for t in initial):raise ValueError('FIRST_SIX_INCOMPLETE')
    remaining=[t for t in all_tasks if task_key(t) not in complete]
    if any(t['validate'] or t['heavy'] or not t['cell']['R'] for t in remaining):raise ValueError('RESUME_SCOPE')
    pending_keys={task_key(t) for t in remaining}
    if any(task_key(t) not in pending_keys for t in incomplete):raise ValueError('PARTIAL_OVERLAPS_COMPLETE')
    caps=dict(compile=40+2*len(incomplete),trajectory=16+len(incomplete),sampling_call=16+len(incomplete),occurrence=32+2*len(incomplete))
    return dict(seed_source=ORIGIN,input_digest=original_input,profile_digest=manifest['profile_digest'],parent=str(parent),
        saved=saved,accepted=list(complete.values()),accepted_references=refs,remaining_tasks=remaining,failed_dispatched=incomplete,
        cumulative_caps=caps,prior_wall_seconds=native['wall_seconds'],old_actual_partial_unknown=True,
        completed_results_are_not_refunds=True,inventory=parent_inventory(parent))


def verify_resume_grant(manifest,grant,source):
    scope=manifest.get('resume')
    if (not scope or not grant or grant.get('kind')!='H7_NEUTRAL_OPEN_SHELL_RESUME_ONE_SHOT_V1'
        or source!=manifest.get('SOURCE') or grant.get('SOURCE')!=source or len(source)!=40):raise ValueError('NEW_RESUME_GRANT_REQUIRED')
    if not all(manifest.get(k) is True and grant.get(k) is True for k in ('approved','runtime_authorization')):raise ValueError('NOT_APPROVED')
    if grant.get('manifest_digest')!=digest(manifest) or grant.get('resume_digest')!=digest(scope):raise ValueError('RESUME_GRANT_DIGEST')
    for k in ('profile_digest','carry_digest','target_digest','output'):
        if grant.get(k)!=manifest.get(k) or not manifest.get(k):raise ValueError('RESUME_GRANT_BINDING:'+k)
    if manifest['target_digest']!=digest(target()) or manifest['compiler_hash']!=COMPILER_HASH or manifest['contract_digest']!=digest(draft()):raise ValueError('RESUME_SCIENTIFIC_SCOPE')
    if grant.get('maximum_compile')!=2*len(scope['remaining_tasks']) or grant.get('cumulative_caps')!=scope['cumulative_caps']:
        raise ValueError('RESUME_BUDGET')
    if not grant.get('one_shot') or grant.get('automatic_retry') is not False:raise ValueError('ONE_SHOT_ONLY')
    output=Path(manifest['output']).resolve()
    if not output.is_relative_to('/home/AbeHiromu') or output==Path(scope['parent'])/'output01':raise ValueError('FRESH_RESUME_OUTPUT')
    return output


def load_resume(root,manifest):
    scope=manifest['resume'];result=admit_parent(root,scope['parent'])
    for k in ('seed_source','input_digest','profile_digest','remaining_tasks','cumulative_caps','inventory','accepted_references','failed_dispatched'):
        if result[k]!=scope[k]:raise ValueError('RESUME_PARENT_CHANGED:'+k)
    if manifest['profile_digest']!=result['profile_digest']:raise ValueError('RESUME_PROFILE')
    return result
