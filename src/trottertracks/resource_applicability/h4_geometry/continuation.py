"""Verified run10 completions, with unchanged dataset identity and fresh execution.

No signals, input generation, numerical arrays or compiler calls occur here.
The predecessor is read-only. A receipt can only be made after native cleanup.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from .identity import require, fingerprint, sha, hash_id, wrapper_key
from .ledger import wire, validate_complete, reuse_owner
from .prelaunch_audit import private_path, streaming_sha

PARENT_RUN = 'h4-newhost-signal-compile-20261010-run10'
PARENT_SOURCE = '5aa6c3685afb9c3afcfbdc36f80244da0a7fe525'
PARENT_REVIEW = '95c2269263c92688936343b0423e9f1dfa22281c'
PARENT_REVIEW_DIGEST='4e1187e6b8be107f4019a24594de3e1665805c08e7551e3e88b02b0d19952ec3'
STARTED_EXPECTED={'bytes':10580,'sha256':'d17339fa452f69473c015907e4bb4256e4e1e8f5c6d78067b9947e6b6df3e7e2'}
CLOCK_EXPECTED={'bytes':586,'sha256':'c3497f25a9e9ecdcc12576ea08aba6f2d05f570749791af2079cff5e25a61029'}
PARENT_BUNDLE = 'artifacts/resource_applicability/track_a_h4_production_run10/2026-10-10/'
MAX_FILES = 400000
MAX_RECEIPT = 64*2**20
IDENTITY_FIELDS = ('pid', 'start', 'parent', 'uid')


def read_json(path, *, limit=MAX_RECEIPT, expected=None):
    path = private_path(path)
    info=path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid(), 'owned regular continuation JSON')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    require(len(raw) <= limit, 'continuation JSON byte limit')
    if expected is not None:
        require(set(expected)=={'bytes','sha256'} and len(raw)==expected['bytes'] and
                sha(raw)==expected['sha256'], 'continuation JSON byte binding')
    def unique(pairs):
        require(len({k for k,_ in pairs})==len(pairs),'duplicate continuation JSON field')
        return dict(pairs)
    return json.loads(raw,object_pairs_hook=unique,
                      parse_constant=lambda value:require(False,'nonfinite continuation JSON'))


def parent_plan(root):
    from .gates import git_blob
    root = private_path(root)
    path = PARENT_BUNDLE+'plan_authorized_v20.json'
    raw = (root/path).read_bytes()
    require(raw==git_blob(root,PARENT_REVIEW,path), 'pinned run10 plan blob')
    plan = json.loads(raw)
    require(plan['run_id']==PARENT_RUN and plan['source_commit']==PARENT_SOURCE and
            plan['carry']==dict(actual_invocations=0,charged_bytes=0,wall_seconds=0.), 'run10 lineage/carry')
    return plan


def identity_for(plan, geometry, template):
    from .signal import candidate_identity
    require(geometry in plan['inputs'], 'continuation geometry')
    inp = {'geometry':geometry, **{k:plan['inputs'][geometry][k] for k in ('input','H','DF','state')}}
    return candidate_identity(inp, template, PARENT_SOURCE,
                              plan['compiler_fingerprint'], plan['environment_fingerprint'])


def expected_record(plan, record, context=None, *, execution_source=None):
    from .signal import trajectory_seeds
    if context is None:
        context=({fingerprint('h4-template-v1',t):t for t in plan['templates']},{},{})
    templates,identities,seeds=context
    template = templates.get(record['candidate_template'])
    require(template is not None, 'continuation template outside map')
    scope=(record['geometry'],record['candidate_template'])
    if scope not in identities:identities[scope]=identity_for(plan,record['geometry'],template)
    identity=identities[scope]
    index = record['trajectory_index']
    if template['method'] in ('B2','B3'):
        require(type(index) is int and 0 <= index < 32, 'continuation trajectory index')
        if scope not in seeds:seeds[scope]=trajectory_seeds(identity)
        seed = seeds[scope][index]
    else:
        require(index is None, 'continuation baseline index')
        seed = None
    expected = wire(identity,record['axis'],seed,index,record['numerical_circuit_fingerprint'])
    extra={'status','metrics','cache_reuse','cache_owner_wrapper_key','actual_transpile_invocation_id'}
    if execution_source is not None:
        extra.add('execution_source_commit')
        require(record.get('execution_source_commit')==execution_source, 'continuation execution SOURCE')
    require(set(record)==set(expected)|extra, 'continuation completion fields')
    return expected


def native_identities(started):
    require(started['source']==PARENT_SOURCE and started['artifact_commit']==PARENT_REVIEW and
            started['run_id']==PARENT_RUN, 'run10 started receipt lineage')
    rows = [{k:r[k] for k in IDENTITY_FIELDS} for r in started['roles']]
    require(len(rows)==6 and len({r['pid'] for r in rows})==6 and
            {r['pid'] for r in rows}=={started['driver_pid'],started['observer_pid'],*started['worker_pids']} and
            len(started['worker_pids'])==4 and all(r['uid']==os.getuid() for r in rows), 'run10 exact owned native6')
    require(sum(r['role']=='driver' for r in started['roles'])==1 and
            sum(r['role']=='observer' for r in started['roles'])==1 and
            sum(r['role']=='worker' for r in started['roles'])==4 and
            all(r['parent']==started['driver_pid'] for r in rows if r['pid']!=started['driver_pid']),
            'run10 native parent graph')
    return rows


def capture_clock(started):
    from .observer import process_sample
    rows = native_identities(started)
    for expected in rows:
        current = process_sample(expected['pid'])
        require(all(current[k]==expected[k] for k in IDENTITY_FIELDS), 'run10 live clock identity')
    return dict(schema_version='h4-run10-clock-v1', boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                identities=rows, wall_origin_monotonic=wall_origin(started),
                historical_exit_code=None, exact_exit_time=None)


def wall_origin(started):
    obs=started['observer_observation']
    require(type(obs['monotonic']) in (float,int) and type(obs['wall_seconds']) in (float,int) and
            math.isfinite(obs['monotonic']) and math.isfinite(obs['wall_seconds']) and
            0<=obs['wall_seconds']<72*3600, 'run10 wall anchor')
    return obs['monotonic']-obs['wall_seconds']


def audit_absence(identities):
    from .observer import process_sample
    for expected in identities:
        try: current = process_sample(expected['pid'])
        except FileNotFoundError: continue
        require(current['start']!=expected['start'], 'run10 owned process still exists')


def capture_stop_proof(started, clock, *, pause=time.sleep):
    require(clock['schema_version']=='h4-run10-clock-v1' and clock['identities']==native_identities(started) and
            clock['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'run10 clock domain')
    audits=[]
    for index in range(2):
        if index:pause(2)
        audit_absence(clock['identities'])
        audits.append(dict(monotonic=time.monotonic(), identities=clock['identities'], same_owned_remaining=0))
    return dict(schema_version='h4-run10-native-stop-v1', run_id=PARENT_RUN, source_commit=PARENT_SOURCE,
                artifact_commit=PARENT_REVIEW, clock=clock, native_audits=audits,
                historical_exit_code=None, exact_exit_time=None, exit_data_not_inferred=True)


def verify_stop(proof, started, clock):
    require(proof['schema_version']=='h4-run10-native-stop-v1' and proof['run_id']==PARENT_RUN and
            proof['source_commit']==PARENT_SOURCE and proof['artifact_commit']==PARENT_REVIEW and
            proof['clock']==clock and clock['identities']==native_identities(started) and
            clock['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'run10 native stop/clock binding')
    require(clock['wall_origin_monotonic']==wall_origin(started), 'run10 wall origin changed')
    require(proof['historical_exit_code'] is None and proof['exact_exit_time'] is None and
            proof['exit_data_not_inferred'] is True, 'unrecorded exits not inferred')
    audits=proof['native_audits']
    require(len(audits)==2 and all(a['identities']==clock['identities'] and a['same_owned_remaining']==0 for a in audits) and
            all(type(a['monotonic']) in (float,int) and math.isfinite(a['monotonic']) for a in audits) and
            audits[1]['monotonic']-audits[0]['monotonic']>=2 and
            clock['wall_origin_monotonic']<=audits[0]['monotonic']<=audits[1]['monotonic']<=time.monotonic(),
            'run10 two separated native zero audits')
    audit_absence(clock['identities'])
    return float(audits[-1]['monotonic']-clock['wall_origin_monotonic'])


def file_receipt(path):
    """A crash may leave OutputBudget's own final/.pending two-link pair."""
    path = private_path(path)
    info = path.stat()
    if info.st_nlink==1:return streaming_sha(path)
    other = path.with_name(path.name[:-8] if path.name.endswith('.pending') else path.name+'.pending')
    require(info.st_nlink==2 and other.exists() and not other.is_symlink() and
            (other.stat().st_dev,other.stat().st_ino)==(info.st_dev,info.st_ino), 'unexpected receipt hardlink')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(before.st_uid==os.getuid() and stat.S_ISREG(before.st_mode),'owned regular continuation receipt')
        digest=hashlib.sha256();count=0
        while raw:=os.read(fd,65536):digest.update(raw);count+=len(raw)
        after=os.fstat(fd)
        require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')) and
                count==before.st_size, 'continuation receipt changed')
        return dict(bytes=count,sha256=digest.hexdigest())
    finally:os.close(fd)


def inventory(plan):
    files={}
    for label in ('output','control'):
        root=private_path(plan[label+'_root'])
        for path in root.iterdir():
            require(not path.is_symlink(), 'continuation symlink')
            if path.is_dir():
                require(label=='control' and path.name=='private-temp' and not list(path.iterdir()), 'continuation unexpected directory')
                continue
            name=path.name.removesuffix('.pending')
            allowed=(name in ('byte-budget.journal','observer.jsonl','driver-first-stop.json','worker-log-first-stop.txt','ledger.lock','map-complete.json','launch-stop.json') or
                     re.fullmatch(r'ledger-[0-9]{6}\.json|record-[0-9a-f]{64}\.json|signal-.+\.json|worker-log-.+\.json|worker-log-[0-9]{6}\.txt',name)) if label=='output' else name in ('one-shot.json','runner.log')
            require(allowed, 'unexpected continuation file')
            require(len(files)<MAX_FILES, 'continuation file count limit')
            files[label+'/'+path.name]=file_receipt(path)
    return files


def replay(plan, files, *, ancestor=None, execution_source=None, prior_invocations=0, prior_charge=0):
    if ancestor is None:
        require(execution_source is None and prior_invocations==prior_charge==0, 'run10 replay context')
    else:
        require(type(prior_invocations) is int and prior_invocations==ancestor.state.get('total_actual_invocations',len(ancestor.state['reservations'])) and
                type(prior_charge) is int and prior_charge>=0 and execution_source is not None and
                plan['carry']['actual_invocations']==prior_invocations and plan['carry']['charged_bytes']==prior_charge,
                'continuation replay cumulative context')
    root=private_path(plan['output_root']);entries={};reservations={};chain=None
    names=sorted(n for n in files if re.fullmatch(r'output/ledger-[0-9]{6}\.json',n))
    require(names, 'continuation ledger missing')
    context=({fingerprint('h4-template-v1',t):t for t in plan['templates']},{},{})
    for version,name in enumerate(names):
        require(name=='output/ledger-%06d.json'%version, 'continuation ledger gap')
        row=read_json(root/Path(name).name, limit=4096, expected=files[name])
        require(set(row)=={'schema_version','version','previous_digest','entries','reservations','mandatory_stop'} and
                row['schema_version']=='h4-completion-ledger-delta-v1' and row['version']==version and
                row['previous_digest']==chain and row['mandatory_stop'] is True, 'continuation ledger chain')
        require(not (set(entries)&set(row['entries'])), 'duplicate continuation completion')
        require(version!=0 or not row['entries'] and not row['reservations'], 'continuation initial ledger')
        for key,value in row['reservations'].items():
            hash_id(key)
            require(set(value)=={'status','invocation'} and value['status'] in ('RESERVED','COMPLETE') and
                    (key not in reservations and value['status']=='RESERVED' or key in reservations and reservations[key]['status']=='RESERVED' and
                     value=={**reservations[key],'status':'COMPLETE'}), 'continuation reservation transition')
        entries.update(row['entries']);reservations.update(row['reservations']);chain=fingerprint('h4-ledger-delta-v1',row)
    require(prior_invocations+len(reservations)<=plan['caps']['actual_invocations'] and
            {r['invocation'] for r in reservations.values()}=={'science-%06d'%i for i in range(prior_invocations+1,prior_invocations+len(reservations)+1)},
            'continuation contiguous actual invocations')
    records={};registry={};expected={}
    for key,entry in entries.items():
        hash_id(key);name='output/record-'+key+'.json';require(name in files, 'acknowledged completion file missing')
        record=read_json(root/Path(name).name,limit=4096,expected=files[name])
        require(record['wrapper_key']==key and entry['status']=='COMPLETE', 'continuation completion key/state')
        imported='continuation_receipt_sha256' in entry
        fields={'status','digest','cache_owner_wrapper_key'}
        if imported:
            require(ancestor is not None and key in ancestor.records and key not in reservations and
                    entry['continuation_receipt_sha256']==ancestor.digest and
                    files[name]==ancestor.files[name], 'continuation imported lineage/bytes')
            fields.add('continuation_receipt_sha256')
            expect=ancestor.state['expected'][key]
            require(record==ancestor.recover(expect), 'continuation imported record changed')
        else:
            expect=expected_record(plan,record,context,execution_source=execution_source)
            require(ancestor is None or key not in ancestor.records, 'ancestor COMPLETE relabeled as new')
        require(set(entry)==fields, 'continuation completion entry fields')
        registry[key]=expect['numerical_circuit_fingerprint']
        validate_complete(record,expect,registry,entry['digest'])
        require(entry['cache_owner_wrapper_key']==record['cache_owner_wrapper_key'], 'continuation owner binding')
        require(type(record['cache_reuse']) is bool and
                (record['cache_owner_wrapper_key'] is None)==(record['cache_reuse'] is False), 'continuation cache flag/owner')
        if imported:
            pass  # Original reservation remains charged in the ancestor carry.
        elif record['cache_reuse']:
            require(key not in reservations and record['actual_transpile_invocation_id'] is None, 'continuation reused invocation')
        else:
            require(key in reservations and reservations[key]==dict(status='COMPLETE',invocation=record['actual_transpile_invocation_id']),
                    'continuation completed reservation')
        records[key]=record;expected[key]=expect
    require(ancestor is None or set(ancestor.records)<=set(records), 'acknowledged ancestor COMPLETE missing')
    for key,record in records.items():
        if record['cache_reuse']:
            owner=record['cache_owner_wrapper_key'];require(owner in records, 'continuation reuse owner missing')
            reuse_owner(records[owner],expected[owner],expected[key],registry,entries[owner]['digest'])
    require(all(r['status']!='COMPLETE' or k in entries for k,r in reservations.items()), 'completed reservation without record')
    journal=private_path(root/'byte-budget.journal');require('output/byte-budget.journal' in files,'continuation journal missing')
    charged=0;rows=0
    with journal.open('rb') as stream:
        while raw:=stream.read(128):
            require(len(raw)==128 and raw.strip().isdigit(), 'continuation ambiguous journal')
            value=int(raw.strip());require(value>=128,'continuation refunded journal');charged+=value
            if rows==0 and prior_charge:
                require(value==prior_charge+128, 'continuation first journal carry row')
            rows+=1
    require(not prior_charge or rows>0 and charged>=prior_charge+128, 'continuation byte carry missing')
    require(file_receipt(journal)==files['output/byte-budget.journal'] and charged<plan['caps']['output_bytes'], 'continuation byte carry')
    require('output/observer.jsonl' in files, 'continuation observer trace missing')
    last_wall=last_monotonic=0.
    with (root/'observer.jsonl').open('rb') as stream:
        while raw:=stream.readline(8193):
            require(len(raw)<=8192,'continuation observer frame size')
            if not raw.endswith(b'\n'):
                require(not stream.read(1),'continuation partial observer frame before EOF')
                break  # preserve a partial final frame as bytes, never as proof
            row=json.loads(raw)
            if 'wall_seconds' in row:
                value=row['wall_seconds']
                require(type(value) in (int,float) and math.isfinite(value) and value>=last_wall,
                        'continuation observer wall monotonicity')
                last_wall=float(value)
            if row.get('kind')=='observation':
                require(math.isfinite(row['monotonic']) and row['monotonic']>=last_monotonic,
                        'continuation observer clock monotonicity')
                last_monotonic=row['monotonic']
    return dict(records=records,expected=expected,entries=entries,reservations=reservations,
                ledger_head=chain,ledger_versions=len(names),charged_bytes=charged,
                total_actual_invocations=prior_invocations+len(reservations),
                observed_wall_floor=last_wall,observed_monotonic_floor=last_monotonic)


def path_ref(path):
    return dict(path=str(private_path(path)), **streaming_sha(path))


def make_receipt(root, started_path, clock_path, proof_path):
    plan=parent_plan(root)
    started=read_json(started_path,expected=STARTED_EXPECTED)
    clock=read_json(clock_path,expected=CLOCK_EXPECTED)
    proof=read_json(proof_path)
    wall=verify_stop(proof,started,clock)
    files=inventory(plan)
    require('control/one-shot.json' in files, 'run10 one-shot missing')
    marker=read_json(Path(plan['control_root'])/'one-shot.json',limit=4096,expected=files['control/one-shot.json'])
    require(marker['run_id']==PARENT_RUN and marker['source']==PARENT_SOURCE and marker['review']==PARENT_REVIEW_DIGEST and
            marker['driver_pid']==started['driver_pid'] and marker['uid']==os.getuid(), 'run10 marker identity')
    state=replay(plan,files)
    require(wall>=state['observed_wall_floor'] and proof['native_audits'][0]['monotonic']>=state['observed_monotonic_floor'],
            'native audits/wall precede retained observer evidence')
    audit_absence(clock['identities'])
    require(files==inventory(plan), 'predecessor inventory changed during receipt')
    completed=set(state['records'])
    orphan=sorted(Path(n).name[7:-5] for n in files if re.fullmatch(r'output/record-[0-9a-f]{64}\.json',n) and Path(n).name[7:-5] not in completed)
    return dict(schema_version='h4-run10-continuation-receipt-v1',parent_run=PARENT_RUN,parent_source=PARENT_SOURCE,
        parent_review=PARENT_REVIEW,output_root=plan['output_root'],control_root=plan['control_root'],
        started=path_ref(started_path),clock_context=path_ref(clock_path),native_stop_proof=path_ref(proof_path),files=files,
        ledger_head=state['ledger_head'],ledger_versions=state['ledger_versions'],imported_wrappers=len(completed),
        carry=dict(actual_invocations=len(state['reservations']),charged_bytes=state['charged_bytes'],wall_seconds=wall),
        unresolved_reservations=sorted(k for k,r in state['reservations'].items() if r['status']=='RESERVED'),
        orphan_records=orphan,pending_files=sorted(n for n in files if n.endswith('.pending')),
        historical_exit_code=None,exact_exit_time=None,exit_data_not_inferred=True,
        discarded_work_refunded=False,automatic_retry=False)


def load_receipt(root, descriptor):
    require(set(descriptor)=={'path','bytes','sha256'}, 'continuation receipt reference')
    data=read_json(descriptor['path'],expected={k:descriptor[k] for k in ('bytes','sha256')})
    require(data['schema_version']=='h4-run10-continuation-receipt-v1' and data['parent_run']==PARENT_RUN and
            data['parent_source']==PARENT_SOURCE and data['parent_review']==PARENT_REVIEW and
            data['discarded_work_refunded'] is False and data['automatic_retry'] is False and
            data['historical_exit_code'] is None and data['exact_exit_time'] is None and data['exit_data_not_inferred'] is True,
            'continuation receipt lineage/policy')
    plan=parent_plan(root)
    require(data['output_root']==plan['output_root'] and data['control_root']==plan['control_root'], 'continuation parent roots')
    proofs=[]
    for label in ('started','clock_context','native_stop_proof'):
        ref=data[label];require(set(ref)=={'path','bytes','sha256'},'continuation proof reference')
        pinned=STARTED_EXPECTED if label=='started' else (CLOCK_EXPECTED if label=='clock_context' else None)
        if pinned is not None:require({k:ref[k] for k in ('bytes','sha256')}==pinned,'pinned run10 identity/clock receipt')
        proofs.append(read_json(ref['path'],expected={k:ref[k] for k in ('bytes','sha256')}))
    wall=verify_stop(proofs[2],proofs[0],proofs[1])
    require(data['files']==inventory(plan), 'continuation frozen predecessor inventory')
    require('control/one-shot.json' in data['files'], 'run10 one-shot missing')
    marker=read_json(Path(plan['control_root'])/'one-shot.json',limit=4096,expected=data['files']['control/one-shot.json'])
    require(marker['run_id']==PARENT_RUN and marker['source']==PARENT_SOURCE and marker['review']==PARENT_REVIEW_DIGEST and
            marker['driver_pid']==proofs[0]['driver_pid'] and marker['uid']==os.getuid(), 'run10 marker identity on load')
    state=replay(plan,data['files'])
    require(wall>=state['observed_wall_floor'] and proofs[2]['native_audits'][0]['monotonic']>=state['observed_monotonic_floor'],
            'native audits/wall precede retained observer evidence')
    require(data['ledger_head']==state['ledger_head'] and data['ledger_versions']==state['ledger_versions'] and
            data['imported_wrappers']==len(state['records']) and
            data['carry']==dict(actual_invocations=len(state['reservations']),charged_bytes=state['charged_bytes'],wall_seconds=wall),
            'continuation carry/ledger/imported binding')
    store=ResumeStore(plan,data['files'],state,descriptor['sha256'])
    validate_saved_signals(store)
    audit_absence(proofs[1]['identities'])
    return store,data


def validate_saved_signals(store):
    """Validate every saved candidate before it can skip scientific work."""
    plan=store.parent
    from .signal import trajectory_seeds
    for name in store.files:
        if name.startswith('output/signal-') and not name.endswith('.pending'):
            saved=read_json(Path(plan['output_root'])/Path(name).name,limit=524288,expected=store.files[name])
            template=next((t for t in plan['templates'] if t['template_id']==saved['template_id']),None)
            require(template is not None, 'continuation signal template')
            identity=identity_for(plan,saved['geometry'],template)
            seeds=trajectory_seeds(identity) if template['method'] in ('B2','B3') else [None]
            require(Path(name).name=='signal-'+saved['geometry']+'-'+template['template_id']+'.json' and
                    store.complete_signal(identity,template,seeds) is not None, 'unacknowledged continuation signal')


class ResumeStore:
    def __init__(self, plan, files, state, receipt_digest):
        self.parent, self.files, self.state, self.digest = plan, files, state, receipt_digest
        self.records = state['records']

    def recover(self, expected):
        key=expected['wrapper_key']
        if key not in self.records:return None
        record=read_json(Path(self.parent['output_root'])/('record-'+key+'.json'),limit=4096,
                         expected=self.files['output/record-'+key+'.json'])
        validate_complete(record,expected,{key:expected['numerical_circuit_fingerprint']},self.state['entries'][key]['digest'])
        return record

    def expected(self, identity, axis, seed, index):
        key=wrapper_key(identity,axis,seed,index)
        if key not in self.records:return None
        expected=wire(identity,axis,seed,index,self.records[key]['numerical_circuit_fingerprint'])
        self.recover(expected)
        return expected

    def complete_signal(self, identity, template, seeds):
        records=[]
        for index,seed in enumerate(seeds):
            for axis in ('cosine','sine'):
                expected=self.expected(identity,axis,seed,index if seed is not None else None)
                if expected is None:return None
                records.append(self.records[expected['wrapper_key']])
        name='signal-'+identity['geometry']+'-'+template['template_id']+'.json'
        if 'output/'+name not in self.files:return None
        result=read_json(Path(self.parent['output_root'])/name,limit=524288,expected=self.files['output/'+name])
        metrics=[[records[i]['metrics'],records[i+1]['metrics']] for i in range(0,len(records),2)]
        require(result['geometry']==identity['geometry'] and result['template_id']==template['template_id'] and
                result['identity']==identity and result['candidate_fingerprint']==fingerprint('h4-candidate-v1',identity) and
                result['paired_metrics']==metrics and result['saved']['paired_costs']==[[m['rz_count'] for m in pair] for pair in metrics] and
                result['research_decision'] is None and result['next_stage_authorized'] is False, 'continuation signal identity/metrics')
        fingerprint('h4-continued-signal-validation-v1',result)  # finite values
        return result
