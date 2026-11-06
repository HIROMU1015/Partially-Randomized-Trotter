"""One explicitly permitted H4 handoff; preserve the original experiment/seeds.

Only completed samples are reused. Incomplete samples keep their original
index/seed and are finished in a fresh output. No import-time numerical work.
"""
from fractions import Fraction
from pathlib import Path
import shutil
from . import c1_stage1_execution as io

VERSION='h4_single_planned_handoff_same_cohorts_v1'


def prior(c,out):
    if 'handoff' not in c:return None
    h=c['handoff'];rel=Path(out).resolve().relative_to(Path(h['new_output']).resolve())
    return Path(h['prior_output'])/rel


def sampling_policy(key,split,r,batch):
    return dict(master=2026110501,cohort=f'integrated:{key}:{split}:r{r}:batch{batch:02d}',
        axis=f'INTEGRATED_{split.upper()}_PRICE_ONLY',samples=32,events_per_sample=r,
        PRNG='numpy.random.PCG64',paired=False,adaptive=False)


def sample_paths(i):
    label=f'sample{i:02d}'
    return ([Path('samples')/f'{label}_{kind}.json' for kind in ('events','raw','compiled','adjoint','XY','metrics')]
        +[Path('circuits')/f'{label}_{kind}.qpy' for kind in ('raw','compiled','raw_adjoint','compiled_adjoint')]
        +[Path('references')/f'{label}.npz'])


def completed_samples(path,policy,candidate,source):
    rows={}
    if path is None or not path.exists():return rows
    for i in range(32):
        files=sample_paths(i)
        if not (path/'samples'/f'sample{i:02d}_metrics.json').exists():continue
        if not all((path/p).is_file() and (path/p).stat().st_size for p in files):
            raise ValueError('INCOMPLETE_VERIFIED_SAMPLE:'+str(path)+':'+str(i))
        event=io.read(path/'samples'/f'sample{i:02d}_events.json')
        if (event['sample']!=i or event['sampling']!=policy or event['JU']!=candidate
                or event['SOURCE']!=source or len(event['events'])!=policy['events_per_sample']
                or not event['coherent_occurrences_independent'] or event['quantum_shots']!=0
                or event['old_cohort_pooled']):raise ValueError('HANDOFF_SAMPLE_DOMAIN')
        if io.read(path/'samples'/f'sample{i:02d}_XY.json')['status']!='PASS':
            raise ValueError('HANDOFF_SAMPLE_CORRECTNESS')
        rows[i]=io.read(path/'samples'/f'sample{i:02d}_metrics.json')
    return rows


def copy_file(source,target):
    if target.exists():raise FileExistsError(target)
    shutil.copyfile(source,target)


def restore_samples(c,out,policy,a):
    old=prior(c,out)
    rows=completed_samples(old,policy,a['arm'].model.source_identity,c['source_manifest_sha256'])
    for i in rows:
        for p in sample_paths(i):copy_file(old/p,Path(out)/p)
    return rows


def restore_cohort(c,out,a,result,packet,*,split,key):
    old=prior(c,out)
    if old is None or not (old/'result.json').exists():return None
    value=io.read(old/'result.json');r=result['r'];rows=[]
    for batch in range(1 if split=='selection' else 2):
        found=completed_samples(old/f'batch{batch:02d}',sampling_policy(key,split,r,batch),
            a['arm'].model.source_identity,c['source_manifest_sha256'])
        if len(found)!=32:raise ValueError('INCOMPLETE_SAVED_COHORT')
        rows.extend(found[i] for i in range(32))
    if (value['candidate']!=result['candidate'] or value['r']!=r or value['split']!=split
            or value['samples']!=len(rows) or not value['native_admitted']):
        raise ValueError('HANDOFF_COHORT_BINDING')
    shutil.copytree(old,out)  # Fresh destination, independent copies of old evidence.
    io.save(Path(out)/'handoff_reuse.json',dict(kind='complete_cohort',source=str(old),samples=len(rows)))
    return value,dict(packet,kernel_samples=rows,saved_calibration=value['calibration'])


def restore_d(c,out,a):
    old=prior(c,out)
    if old is None or not (old/'result.json').exists():return None
    value=io.read(old/'result.json')
    if value['candidate']!=a['arm'].model.source_identity:raise ValueError('HANDOFF_D_BINDING')
    if value['native_admitted']:
        paths=[old/f for f in ('metrics.json','segment.json','adjoint.json','XY.json','PF.json')]
        paths += [old/'circuits'/f'{kind}.qpy' for kind in ('raw','compiled','raw_adjoint','compiled_adjoint')]
        if not all(p.is_file() and p.stat().st_size for p in paths):raise ValueError('INCOMPLETE_SAVED_D')
    if any(Path(out).iterdir()):raise FileExistsError(out)
    shutil.copytree(old,out,dirs_exist_ok=True)
    io.save(Path(out)/'handoff_reuse.json',dict(kind='complete_D',source=str(old)))
    return value


def restore_reference(c,a,r,out,call,cache):
    old=prior(c,out)
    if old is None or not (old.parent/'reference_result.json').exists():return None
    from . import c_pr_aware_df_stage_b as b
    from . import c_h4_relocated_downstream as bridge
    value=io.read(old.parent/'reference_result.json');candidate=a['arm'].model.source_identity
    if value['candidate']!=candidate or value['r']!=r or [x['m'] for x in value['rounds']]!=list(range(13)):
        raise ValueError('HANDOFF_REFERENCE_BINDING')
    for m,row in enumerate(value['rounds']):
        if io.read(old/f'm{m:02d}.json')!=row:raise ValueError('HANDOFF_ROUND_BINDING')
    tail,config,dist=call('law_restore',str(r),lambda:b.configure(a['arm'],Fraction(1,10),r,0))
    call('law_restore_gate',str(r),lambda:bridge.tail_gate(a['ctx'],a['arm'],a['sector']['generators'],tail))
    if cache is None:
        if r!=32:raise ValueError('R32_RESTORE_FIRST')
        # Old files save PF errors, not the PF matrix. Restore that deterministic
        # 36x36 cache once; no old round/price/native samples are re-evaluated.
        kernels=call('kernel_cache_restore',candidate,lambda:b.kernels(a['arm'],a['sector']['generators'],Fraction(1,10),config))
        cache=dict(candidate_sha256=candidate,input_identity=a['arm'].model.input_identity,
            pf=kernels['pf'],pf_errors=[x['reference']['pf_operator_error'] for x in value['rounds']])
    if cache['candidate_sha256']!=candidate or cache['input_identity']!=a['arm'].model.input_identity:
        raise ValueError('HANDOFF_REFERENCE_CACHE_BINDING')
    df=a['saved']['physical'];bound=df['sector_error']+df['numerical_guard']
    packet=dict(target_id=f'integrated:{candidate}:r{r}',rounds=value['rounds'],df_bound_Ha=bound,
        error_evidence=None,shells=a['saved']['shells'])
    shutil.copytree(old,out)
    io.save(Path(out)/'handoff_reuse.json',dict(kind='reference13',source=str(old)))
    return dict(a,tail=tail,config=config,dist=dist),value,packet,cache


def validate_checkpoint(root,c):
    h=c['handoff'];path=Path(root)/h['checkpoint']
    if io.sha(path)!=h['checkpoint_sha256']:raise ValueError('HANDOFF_CHECKPOINT_CHANGED')
    snap=io.read(path);old=Path(h['prior_output'])
    if (snap['version']!=VERSION or snap['prior_output']!=str(old)
            or snap['sampling_SOURCE_sha256']!=c['sampling_SOURCE_sha256']):
        raise ValueError('HANDOFF_CHECKPOINT_BINDING')
    parent_path=Path(h['prior_root'])/h['prior_contract']
    if io.sha(parent_path)!=h['prior_contract_sha256']:raise ValueError('PRIOR_CONTRACT_CHANGED')
    parent=io.read(parent_path)
    metadata={'SOURCE','SOURCE_sha256','base_commit','status','scientific_execution_permission',
        'fresh_output','resources','required_user_decision','date_JST','schema','science_executed'}
    if any(c.get(key)!=value for key,value in parent.items() if key not in metadata):
        raise ValueError('HANDOFF_SCIENTIFIC_CONDITIONS_CHANGED')
    if parent['SOURCE_sha256']!=c['sampling_SOURCE_sha256']:raise ValueError('ORIGINAL_SEED_SOURCE_CHANGED')
    if io.sha(Path(h['prior_root'])/parent['SOURCE'])!=parent['SOURCE_sha256']:
        raise ValueError('PRIOR_SOURCE_MANIFEST_CHANGED')
    for group in (io.read(Path(h['prior_root'])/parent['SOURCE'])['files'],parent['inputs']):
        for name,want in group.items():
            if io.sha(Path(h['prior_root'])/name)!=want:raise ValueError('PRIOR_SOURCE_OR_INPUT_CHANGED')
    termination_path=Path(root)/h['termination_record']
    if io.sha(termination_path)!=h['termination_sha256']:raise ValueError('HANDOFF_TERMINATION_CHANGED')
    termination=io.read(termination_path)
    if termination['prior_output']!=str(old) or not termination['process_gone']:
        raise ValueError('PRIOR_PROCESS_NOT_STOPPED')
    if (old/'STOP.json').exists():
        stop=io.read(old/'STOP.json')
        if not stop['mandatory_stop'] or not stop['failed'] or stop['reason']!='KeyboardInterrupt()':
            raise ValueError('NOT_A_PLANNED_HANDOFF_STOP')
        completion=Path(h['prior_root'])/'artifacts/research_c_h4_relocated_downstream/2026-11-06/execution/completion.json'
        if io.read(completion)['failed']!='KeyboardInterrupt()':raise ValueError('PRIOR_PROCESS_NOT_DRAINED')
        if io.sha(completion)!=snap['completion_sha256']:raise ValueError('PRIOR_COMPLETION_CHANGED')
        batch=old/snap['interrupted_batch']
        for i in range(32):
            if not all((batch/p).is_file() and (batch/p).stat().st_size for p in sample_paths(i)):
                raise ValueError('CURRENT_COHORT_DID_NOT_DRAIN')
    else:
        # The original finally-block cannot run after SIGTERM. Keep its absent
        # STOP/completion absent, and use this explicitly separate observed
        # interruption record; never fabricate a legacy PASS or legacy STOP.
        if (termination['mode']!='SIGTERM_AFTER_SIGINT' or termination['signal']!=15
                or termination['original_STOP_exists'] or termination['original_completion_exists']
                or termination['exit_code']!=143 or snap['completion_sha256'] is not None):
            raise ValueError('NOT_A_PLANNED_HANDOFF_TERMINATION')
        if any((old/snap['interrupted_batch']/'samples').glob('*metrics.json')):
            raise ValueError('FORCED_HANDOFF_REQUIRES_NO_UNCERTAIN_COMPLETED_SAMPLE')
    for name,want in snap['files'].items():
        p=(old/name).resolve()
        if not p.is_relative_to(old) or io.sha(p)!=want:raise ValueError('HANDOFF_PRIOR_CHANGED:'+name)
    return snap
