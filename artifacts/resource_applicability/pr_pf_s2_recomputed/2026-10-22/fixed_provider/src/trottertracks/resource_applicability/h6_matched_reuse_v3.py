"""Static admission of immutable v2 evidence; no scientific imports/evaluation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from .ax2a_preparation import digest
from .h6_matched_contract_v1 import plan, file_hash, safe_path
from .h6_matched_accounting_v1 import tasks, seed

OLD_SOURCE = '54986a9f54515c8c2289e38fa16fd81e2b6008c2'
OLD_RESULT = '8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb'
OLD_RAW = 'artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1'
COMPILER_HASH = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'


def read(path):
    if Path(path).stat().st_size > 16*2**20:
        raise ValueError('REUSE_JSON_CAP')
    return json.loads(Path(path).read_text())


def committed_hashes(root, commit, hashes):
    """Batch Git blob verification without importing/recreating any result."""
    names = list(hashes)
    # Limit one batch's captured raw bytes; all science results are small JSON.
    for start in range(0, len(names), 64):
        batch = names[start:start+64]
        raw = subprocess.check_output(['git','cat-file','--batch'], cwd=root,
            input=''.join(commit+':'+p+'\n' for p in batch).encode())
        pos = 0
        for p in batch:
            end = raw.index(b'\n',pos)
            header = raw[pos:end].split()
            if len(header)!=3 or header[1]!=b'blob':
                raise ValueError('REUSE_UNPUBLISHED:'+p)
            size = int(header[2]); data = raw[end+1:end+1+size]; pos = end+2+size
            if hashlib.sha256(data).hexdigest()!=hashes[p] or file_hash(safe_path(root,p))!=hashes[p]:
                raise ValueError('REUSE_COMMITTED_BYTES:'+p)


def admit_group(directory, task, *, source=OLD_SOURCE, compiler_hash=COMPILER_HASH):
    """Reject partial, duplicate, wrong seed/event/axis/compiler/validation groups."""
    directory = Path(directory)
    term=read(directory/'worker_terminal.json'); claim=read(directory/'worker_claim.json')
    binding=read(directory/'task_binding.json'); result=read(directory/'cost_result.json')
    trajectory=read(directory/'trajectory.json')
    if (term.get('status')!='TASK_COMPLETE' or term.get('reason') is not None or
        any(digest(obj.get('task'))!=digest(task) for obj in (term,binding,result,trajectory)) or
        term.get('source_commit')!=source or claim.get('source_commit')!=source or
        claim.get('task_digest')!=digest(task) or
        claim.get('resource_limits',{}).get('rlimit_as')!=[8*2**30]*2 or
        type(task.get('seed')) is not int or task['seed']!=seed(task['cell_id'],task['phase'],task['replica'])):
        raise ValueError('REUSE_TASK_IDENTITY')
    if trajectory.get('event_digest')!=digest(trajectory.get('events')):
        raise ValueError('REUSE_EVENTS')
    events=trajectory['events']
    if task['cell']['method']=='B2':
        if (not isinstance(events,list) or len(events)!=task['cell']['q'] or
            any(len(x)!=task['cell']['r'] for x in events) or
            trajectory['observed_orders']!=[e['taylor_order'] for x in events for e in x] or
            trajectory['finite_distribution'] is None):
            raise ValueError('REUSE_EVENT_SHAPE')
    elif events is not None or trajectory['observed_orders'] or trajectory['finite_distribution'] is not None:
        raise ValueError('REUSE_DETERMINISTIC_EVENTS')
    policies=['symmetric_directional','ordinary'] if task['ordinary'] else ['symmetric_directional']
    expected={(p,a) for p in policies for a in ('cosine','sine')}; keys=[]
    for row in result['rows']:
        key=(row['control'],row['axis']);keys.append(key)
        if (any(type(row.get(k)) is not type(task[k]) or row[k]!=task[k] for k in ('cell_id','phase','replica','seed')) or
            row['event_digest']!=trajectory['event_digest'] or row['compiler_hash']!=compiler_hash or
            len(row.get('fingerprint',''))!=64 or
            set(row['metrics'])!={'RZ','CX','RZ_depth','CX_depth','depth','size'} or
            any(type(v) is not int or v<0 for v in row['metrics'].values()) or
            read(directory/f'wrapper_{key[0]}_{key[1]}.json')!=row):
            raise ValueError('REUSE_WRAPPER_IDENTITY')
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('REUSE_PAIRED_AXES')
    if task['validate']:
        for policy in policies:
            v=read(directory/f'validation_{policy}.json')
            if not isinstance(v.get('max_error'),(int,float)) or not 0<=v['max_error']<=plan()['gates']['absolute_agreement']:
                raise ValueError('REUSE_VALIDATION')
    if result['calls_attempted']['compile']!=len(keys):raise ValueError('REUSE_CALLS')
    return result


def complete_exploration(registered, groups):
    """Mandatory full-set gate before ANY map/selection function is called."""
    if set(groups)!=set(range(len(registered))):raise ValueError('EXPLORATION_NOT_COMPLETE')
    rows=[]
    for i,t in enumerate(registered):
        if digest(groups[i]['task'])!=digest(t):raise ValueError('EXPLORATION_TASK_MISMATCH')
        keys=[(r['control'],r['axis']) for r in groups[i]['rows']]
        expected={(p,a) for p in (['symmetric_directional','ordinary'] if t['ordinary'] else ['symmetric_directional']) for a in ('cosine','sine')}
        if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('EXPLORATION_AXES')
        rows+=groups[i]['rows']
    return rows


def evidence(root, *, verify_git=True):
    root=Path(root);out=root/OLD_RAW;m=read(out/'frozen_manifest.json')
    inv=read(out/'execution_inventory.json');term=read(out/'execution_terminal.json')
    if (m['source_commit']!=OLD_SOURCE or digest(m['plan'])!=digest(plan()) or
        inv['source_commit']!=OLD_SOURCE or term['status']!='H6_MATCHED_RESOURCE_STOP' or
        term['reason']!='RuntimeError:MemoryError:' or term['mandatory_stop'] is not True or
        term['next_stage_authorized'] is not False):raise ValueError('REUSE_ORIGINAL_STOP')
    grant=read(out/'authorization.json')
    if (grant.get('schema')!='h6_matched_authorization_v2' or grant.get('kind')!=m['kind'] or
        grant.get('approved_by_user') is not True or grant.get('one_shot') is not True or
        grant.get('retry') is not False or grant.get('resume') is not False or
        grant.get('source_commit')!=OLD_SOURCE or grant.get('manifest_digest')!=digest(m) or
        read(out/'authorization_source.json')!=grant or read(out/'launch_binding.json')!={
            'manifest_digest':digest(m),'authorization_digest':digest(grant),
            'authorization_source_sha256':file_hash(out/'authorization_source.json')}):
        raise ValueError('REUSE_ORIGINAL_AUTHORIZATION')
    hashes={OLD_RAW+'/'+p:sha for p,sha in inv['file_hashes'].items()}
    for n in ('execution_inventory.json','execution_terminal.json'):
        hashes[OLD_RAW+'/'+n]=file_hash(out/n)
    actual={str(p.relative_to(root)) for p in out.rglob('*') if p.is_file() and '.runtime_cache' not in p.parts}
    if set(hashes)!=actual:raise ValueError('REUSE_RAW_CLOSURE')
    for p,sha in hashes.items():
        if file_hash(safe_path(root,p))!=sha:raise ValueError('REUSE_RAW_CHANGED:'+p)
    if verify_git:
        committed_hashes(root,OLD_RESULT,hashes)
        committed_hashes(root,OLD_SOURCE,m['source_hashes'])
    signals=read(out/'signal/signal_summary.json')['signals']
    signal_terminal=read(out/'signal/worker_terminal.json');signal_claim=read(out/'signal/worker_claim.json')
    if (signal_terminal.get('status')!='TASK_COMPLETE' or signal_terminal.get('source_commit')!=OLD_SOURCE or
        signal_terminal.get('task')!={'phase':'signal'} or signal_claim.get('source_commit')!=OLD_SOURCE or
        signal_claim.get('resource_limits',{}).get('rlimit_as')!=[8*2**30]*2 or
        read(out/'signal/task_binding.json')['manifest_digest']!=digest(m)):
        raise ValueError('REUSE_SIGNAL_TERMINAL')
    if [s['cell'] for s in signals]!=m['plan']['cells'] or len(signals)!=92:raise ValueError('REUSE_SIGNAL_IDS')
    for s in signals:
        if (s!=read(out/'signal'/f'{s["cell"]["id"]}_signal.json') or
            set(s['precision_rows'])!=set(map(str,plan()['epsilons'])) or
            any(set(v)!={'1.0','10.0'} for v in s['precision_rows'].values())):
            raise ValueError('REUSE_SIGNAL_FIELDS')
    b3=read(out/'signal/b3_normalization_diagnostic.json')['rows'];d=plan()['b3_diagnostic']
    if (len(b3)!=84 or {(x['q'],x['r'],x['K']) for x in b3}!={(q,r,k) for q in d['q'] for r in d['r'] for k in d['K']}):
        raise ValueError('REUSE_B3_COVERAGE')
    registered=read(out/'exploration_tasks.json')['tasks'];generated=tasks(signals,plan(),'exploration')
    by_id={s['cell']['id']:s for s in signals}
    for t in generated:
        if t['cell']['method']!='B2':t['expected_signal']=by_id[t['cell_id']]['signals']['corrected']
    if registered!=generated or len(registered)!=592:raise ValueError('REUSE_REGISTRATION')
    accepted=[];missing=[];group_hashes={}
    for i,t in enumerate(registered):
        directory=out/f'cost_exploration_{i:04d}';terminal=directory/'worker_terminal.json'
        if terminal.exists() and read(terminal).get('status')=='TASK_COMPLETE':
            result=admit_group(directory,t)
            if read(directory/'task_binding.json')['manifest_digest']!=digest(m):raise ValueError('REUSE_MANIFEST_BINDING')
            accepted.append(dict(index=i,task=t,directory=str(directory.relative_to(root)),wrappers=len(result['rows'])))
            group_hashes[i]=file_hash(directory/'cost_result.json')
        else:
            missing.append(dict(index=i,task=t,previous_state=('TASK_STOP' if terminal.exists() else
                'DISPATCHED_WITHOUT_COMPLETE_TERMINAL' if directory.exists() else 'NOT_DISPATCHED'),
                partial_event_digest=read(directory/'trajectory.json')['event_digest'] if (directory/'trajectory.json').exists() else None))
    if len(accepted)!=449 or len(missing)!=143 or sum(x['wrappers'] for x in accepted)!=902:
        raise ValueError('REUSE_EXPECTED_COVERAGE')
    return dict(schema='h6_matched_reuse_v3',source_commit=OLD_SOURCE,result_commit=OLD_RESULT,
        original_raw_path=OLD_RAW,original_manifest_digest=digest(m),raw_hashes=hashes,
        legacy_source_hashes=m['source_hashes'],input_identity=m['input_identity'],environment=m['environment'],
        accepted=accepted,missing=missing,original_tasks=registered,signal_count=92,b3_count=84,
        accepted_groups=449,accepted_wrappers=902,missing_groups=143,
        missing_first_index=447,missing_first_cell='B1_p19_q8_4th',
        original_signal_calls=read(out/'signal/signal_summary.json')['calls_attempted'],
        aborted_attempt_counts=None,aborted_attempt_counts_reason='not saved; never filled with zero',
        old_run_status_unchanged=True,old_grant_consumed=True,old_retry_or_resume=False)


def verify_evidence(root, registry):
    if evidence(root)!=registry:raise ValueError('REUSE_REGISTRY_CHANGED')
    return registry
