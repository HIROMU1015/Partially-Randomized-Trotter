"""Byte/metadata-only binding of nine contracts. Never calls science dispatch.

JSON/CSV projections copy saved values. No preparation decoder, scientific
array reader, shot arithmetic, sample statistics, fit or ranking is invoked.
"""
from copy import deepcopy
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pr_cost_batch as b
c=b.old
PROJECTS=Path('/home/AbeHiromu/projects')
ORIGINS=dict(H4=PROJECTS/'h4-native-validation/20261018/private',
             H6=PROJECTS/'h6-dba-validation/20261018/private',
             H8=PROJECTS/'h8-ab-validation/20261018/private')
CONFIGS=dict(H4='config_SEALED_PRIVATE.json',H6='config_A_run02_sealed.json',
             H8='config_A_recovery01_SEALED_PRIVATE.json')
BASE_SOURCE='33c7231fc7c8f3940106bd8fe473e3db775fbcb0'


def git(*args):
    return subprocess.check_output(['git','--no-optional-locks','-c','maintenance.auto=false',
        '-c','gc.auto=0','-C',str(b.ROOT),*args])


class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.rows=[]
    def bytes(self,role,data,origin):
        digest=hashlib.sha256(data).hexdigest();path=self.root/digest
        if not path.exists():path.write_bytes(data)
        c.require(c.sha(path)==digest,'EXCLUSIVE_SNAPSHOT_BYTES')
        row=dict(role=role,path=str(path),bytes=len(data),sha256=digest,origin=origin)
        self.rows.append(row);return str(path)
    def file(self,role,path,expected=None):
        path=Path(path);data=path.read_bytes()
        if expected:c.require(hashlib.sha256(data).hexdigest()==expected,'ORIGIN_FILE_HASH:'+role)
        return self.bytes(role,data,dict(path=str(path)))
    def blob(self,role,commit,path):
        return self.bytes(role,git('show',commit+':'+path),dict(commit=commit,path=path))
    def json(self,role,value):
        return self.bytes(role,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),
            dict(kind='lossless metadata projection; no numerical analysis'))


def data(path):
    raw=Path(path).read_bytes()
    if raw[:2]==b'\x1f\x8b':raw=gzip.decompress(raw)
    return json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def lines(path):
    raw=Path(path).read_bytes()
    if raw[:2]==b'\x1f\x8b':raw=gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines()]


def python_snapshot(commit,dest):
    """Separate immutable source snapshot, no worktree registration or checkout."""
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    paths=git('ls-tree','-r','--name-only',commit,'scripts','src').decode().splitlines()
    paths=[p for p in paths if p.endswith('.py')]
    proc=subprocess.Popen(['git','-C',str(b.ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    hashes={}
    for name in paths:
        proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush()
        header=proc.stdout.readline().split();raw=proc.stdout.read(int(header[2]));proc.stdout.read(1)
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    proc.stdin.close();c.require(proc.wait()==0,'SOURCE_SNAPSHOT')
    return hashes


def collect_seeds(value):
    out=set()
    if isinstance(value,dict):
        for key,child in value.items():
            if 'seed' in key.lower():
                if type(child) is int:out.add(child)
                elif isinstance(child,str) and child.isdigit():out.add(int(child))
                elif isinstance(child,list):out.update(x for x in child if type(x) is int)
            out.update(collect_seeds(child))
    elif isinstance(value,list):
        for child in value:out.update(collect_seeds(child))
    return out


def csv_pairs(path,binding,ids,phase=None):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    grouped={}
    for row in rows:
        if row['cell_id'] not in ids or (phase and row['phase']!=phase):continue
        if row['control']!='symmetric_directional':continue
        c.require(row['compiler_hash']==binding['compiler'],'SAVED_METRIC_COMPILER')
        pair=grouped.setdefault((row['cell_id'],int(row['replica'])),{})
        c.require(row['axis'] not in pair,'DUPLICATE_METRIC_AXIS');pair[row['axis']]=row
    result={cid:[] for cid in ids}
    for (cid,replica),pair in sorted(grouped.items()):
        c.require(set(pair)=={'cosine','sine'},'EXACT_PAIRED_METRICS')
        first=pair['cosine'];seed=int(first['seed']);event=first['event_digest']
        c.require(all(int(x['seed'])==seed and x['event_digest']==event for x in pair.values()),'PAIRED_SEED_EVENT')
        result[cid].append(dict(cell_id=cid,replica=replica,seed=seed,event_digest=event,
            cohort='CONFIRMATION_N32',origin_phase=first['phase'],binding=binding,
            axes={a:dict(RZ=int(v['RZ']),seed=seed,event_digest=event) for a,v in pair.items()}))
    c.require(all(len(v)==32 for v in result.values()),'EXACT_SAVED_N32_COVERAGE')
    return result,rows


def rewrite_paths(value,mapping):
    if isinstance(value,dict):return {k:rewrite_paths(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [rewrite_paths(x,mapping) for x in value]
    return mapping.get(value,value) if isinstance(value,str) else value


def bind_native(system,entry,store,private):
    original_path=ORIGINS[system]/CONFIGS[system];old=data(original_path)
    for key in ('hamiltonian','state','compiler'):
        c.require(old['binding'][key]==entry['binding'][key],'NATIVE_CONFIG_INPUT_IDENTITY')
    c.require(old['binding'].get('snapshot',old['binding'].get('input'))==entry['binding']['snapshot'],
              'NATIVE_CONFIG_SNAPSHOT_IDENTITY')
    store.file(system+':original_native_config',original_path)
    relocation={};inputs=[]
    for row in old['inputs']:
        new=store.file(system+':native:'+row['role'],row['path'],row['sha256']);relocation[row['path']]=new
        inputs.append(dict(role=row['role'],path=new,bytes=row['bytes'],sha256=row['sha256']))
    stage_root=private/'code'/system/'stage';provider_root=private/'code'/system/'provider'
    stage_hashes=python_snapshot(entry['A_SIGNAL_SOURCE'],stage_root)
    provider_hashes=python_snapshot(old['provider_source'],provider_root)
    for name,h in old['source_hashes'].items():c.require(stage_hashes[name]==h,'ORIGINAL_STAGE_SOURCE')
    for name,h in old['provider_hashes'].items():c.require(provider_hashes[name]==h,'ORIGINAL_PROVIDER_SOURCE')
    relocated=rewrite_paths(old,relocation);relocated['provider_root']=str(provider_root);relocated['inputs']=inputs
    relocated_path=store.json(system+':relocated_native_config',relocated)
    config=dict(native_config=relocated_path,binding=entry['binding'],
        import_paths=[str(provider_root/'src'),str(stage_root/'scripts'),str(stage_root/'src')],
        font_seed=relocated['font_seed'],preparation_identities={})
    policy={}
    if system=='H4':
        config['H4_reconstruct_original']=True
        policy=dict(mode='ORIGINAL_FIXED_BUILDER',prefixes=[2,3,4],maximum_reconstructions=3,
            source_sha256=provider_hashes['src/trotterlib/pr2_matched_accuracy_m1_execution.py'],SCF_DF_state_solver=0)
    elif system=='H6':
        archive=data(relocated['archive']['path']);codec=archive['codec']
        def prep(key):return codec['nodes'][codec['roots'][key]['ref']]['fields']['preparation_hash']
        config['preparation_identities']={'5':prep('faithful:p5')}
        policy=dict(mode='SAVED_P5_PLUS_ORIGINAL_PREFIX_VIEWS',full_p19_hash=prep('faithful:p19'),
            saved_p5_hash=prep('faithful:p5'),view_prefixes=[3,4],
            view_source_sha256=stage_hashes['scripts/h6_dba_signal.py'],new_Gaussian_synthesis=0)
    else:
        path=ORIGINS[system]/'A_recovery01/output/preparation_binding.json'
        store.file(system+':saved_preparation_binding',path)
        receipt=data(path);config['preparation_identities']={k:v['preparation_hash'] for k,v in receipt['prefixes'].items()}
        policy=dict(mode='SAVED_EXPORT_ORIGINAL_PREFIX_VIEW',receipt_sha256=c.sha(path),expected_hashes=config['preparation_identities'])
    return config,policy,{str(stage_root):stage_hashes,str(provider_root):provider_hashes},old


def d_payload(system,entry,store):
    r=b.ROOT;binding=entry['binding'];paired={};measured={}
    if system=='H4':
        root=ORIGINS['H4']/'D_run01/output'
        path=store.file('H4:D_result',root/'result.json');body=data(path)
        rows=[]
        for x in body['rows']:
            rows.append(dict(cell_id=x['cell_id'],method=x['method'],bias_mode=x['bias_mode'],epsilon=x['epsilon'],
                factor=x['allowance_factor'],eligible=x['eligible'],G_exact=x.get('G_RZ_exact'),G_SE=x.get('G_RZ_SE'),
                shots=x.get('shots'),cost_status=x['cost_provenance'],axis_costs=None))
        for group in body['measurements']:
            measured[group['cell_id']]={x['axis']:dict(RZ=x['RZ'],kind='DIRECT_MEASURED',SOURCE=entry['A_SIGNAL_SOURCE'],
                compiler_hash=x['compiler_hash'],result_sha256=c.sha(root/'result.json')) for x in group['rows']}
        for name in ('final_gate.json','terminal.json'):store.file('H4:D_'+name,root/name)
        originals=data(ORIGINS['H4']/'inputs/old_signals.json')['signals']
        old_ids=[x['cell']['id'] for x in originals if x['cell']['method']!='B2']
        pr_ids=sorted({x['cell_id'] for x in rows if x['method']=='B2'})
        pairs_path=store.file('H4:original_wrapper_metrics',ORIGINS['H4']/'inputs/old_metrics.csv')
        paired,seed_rows=csv_pairs(pairs_path,binding,pr_ids)
        det_ids=sorted({x['cell_id'] for x in rows if x['method']!='B2'})
        sets=dict(old_fixed_deterministic=old_ids,old_plus_direct_subset=det_ids)
        return dict(rows=rows,paired_PR=paired,measured=measured,deterministic_sets=sets,PR_ids=pr_ids,
            comparison_limit='partial measured D subset; never all96 optimum'),seed_rows
    if system=='H6':
        base=r/'artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16'
        source=store.file('H6:V1_saved_accounting',base/'candidate_accounting.json.gz');all_rows=data(source)['rows']
        rows=[];exploration=[]
        for x in all_rows:
            if x['method']=='B2' and x.get('sample_scope')!='H6_CONFIRMATION_UNION8':
                exploration.append(x);continue
            ac=x['accounting'];shots=None if not x['eligible'] else {a:ac['axes'][ch]['shots'] for a,ch in c.core.CHANNEL.items()}
            rows.append(dict(cell_id=x['cell_id'],cell=x['cell'],method=x['method'],
                bias_mode='USUAL_SIGNED' if x['bias_rule']=='signed_signal' else 'EMPIRICAL_NONCANCELLATION',
                epsilon=x['epsilon'],factor=x['factor'],eligible=x['eligible'],G_exact=x['G_exact'],
                shots=shots,cost_status=x['cost_status'],axis_costs=x.get('axis_costs')))
        current=ORIGINS['H6']/'D_run01';result=data(store.file('H6:D_result',current/'result.json'))
        measured[result['cell_id']]={x['axis']:dict(RZ=x['RZ'],kind='DIRECT_MEASURED',compiler_hash=x['compiler_hash'],
            SOURCE='579d03c9c8a2b1ce7c22eb751a08642e2d813c3c',result_sha256=c.sha(current/'result.json')) for x in result['rows']}
        for name in ('final_gate.json','terminal.json'):store.file('H6:D_'+name,current/name)
        p7=r/'artifacts/resource_applicability/track_a_v1_p7_compile_run01/2026-10-16'
        p7_source=data(p7/'execution_terminal.json')['SOURCE']
        prices={}
        for axis in ('cosine','sine'):
            raw=data(store.file('H6:p7_'+axis,p7/(axis+'_measurement.json')))
            # Keep its saved acceptance/price record; no model residual arithmetic.
            rz=raw.get('measured_RZ',raw.get('RZ',raw.get('measured',raw.get('rz_count'))))
            if rz is None and 'metrics' in raw:rz=raw['metrics'].get('RZ')
            c.require(type(rz) is int,'P7_SAVED_INTEGER_RZ')
            prices[axis]=dict(RZ=rz,kind='DIRECT_MEASURED',compiler_hash=binding['compiler'],
                SOURCE=raw.get('SOURCE',p7_source),result_sha256=c.sha(p7/(axis+'_measurement.json')))
        measured['B0I_p7_q1_2nd']=prices
        for name in ('execution_terminal.json','launcher_terminal.json','validation_complete.json'):
            store.file('H6:p7_'+name,p7/name)
        config=data(PROJECTS/'h6-handoff-evidence/20261016/track-a-v1-reaccount-run01/config_ready.json')
        pair_path=store.file('H6:original_confirmation_wrappers',config['input_mapping']['confirmation_wrappers'])
        explore_path=store.file('H6:original_exploration_wrappers',config['input_mapping']['exploration_wrappers'])
        pr_ids=sorted({x['cell_id'] for x in rows if x['method']=='B2'})
        paired,seed_rows=csv_pairs(pair_path,binding,pr_ids,'confirmation')
        with Path(explore_path).open() as f:seed_rows+=list(csv.DictReader(f))
        old_signal=data(store.file('H6:original_signals',config['input_mapping']['old_signals']))['signals']
        b0i=data(store.file('H6:original_B0I_signals',config['input_mapping']['b0i_signals']))
        old_ids=sorted({x['cell']['id'] for x in old_signal+b0i if x['cell']['method']!='B2'})
        det_ids=sorted({x['cell_id'] for x in rows if x['method']!='B2'})
        return dict(rows=rows,paired_PR=paired,measured=measured,PR_ids=pr_ids,
            deterministic_sets=dict(old32=old_ids,strengthened_S2_S4=det_ids),
            exploration_reference=exploration,comparison_limit='fixed model remains for unmeasured candidates'),seed_rows
    # H8: exact saved six-axis outputs plus original saved shot records.
    current=r/'artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17'
    saved=lines(store.file('H8:six_axis_saved_rows',current/'accounting_rows.jsonl.gz'))
    d0=data(store.file('H8:D0_saved_shots',r/'artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/candidate_accounting_v1.json'))
    usual={}
    for label,records in d0.items():
        if isinstance(records,list):
            for x in records:
                if isinstance(x,dict) and 'accounting' in x and 'cell_id' in x and (
                        x.get('sample_scope')=='H8_CONFIRMATION_UNION6' or label=='H8_CONFIRMATION_UNION6'):
                    usual[(x['cell_id'],x['epsilon'],x['factor'])]=x['accounting']
    noncancel_path=store.blob('H8:noncancel_saved_shots','23d46e489af054735d9d328aad0342ecdbddf305',
        'artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/accounting_rows_v1.jsonl.gz')
    noncancel={(x['cell_id'],x['epsilon'],x['factor']):x.get('accounting') for x in lines(noncancel_path)}
    registry=data(r/'references/h8_phase_c_registration_v1.json');pr_ids=registry['confirmation_ids'];rows=[]
    for x in saved:
        old=x['updated'];key=(old['cell_id'],old['epsilon'],old['factor']);weights=x.get('saved_shots')
        if old['cell_id'] in pr_ids:
            ac=(usual if x['bias_kind']=='USUAL_SIGNED' else noncancel).get(key)
            if old['eligible']:
                c.require(ac is not None,'H8_SAVED_CONFIRMATION_SHOTS');weights={ch:ac['axes'][ch]['shots'] for ch in ('real','imag')}
        shots=None if weights is None else {a:weights[ch] for a,ch in c.core.CHANNEL.items()}
        rows.append(dict(cell_id=old['cell_id'],method='B2' if old['cell_id'] in pr_ids else 'DETERMINISTIC',
            bias_mode=x['bias_kind'],epsilon=old['epsilon'],factor=old['factor'],eligible=old['eligible'],
            G_exact=old['G_exact'],shots=shots,cost_status=old['cost_status'],axis_costs=old.get('axis_costs')))
    root=PROJECTS/'h8-dba-validation/20261018/private'
    for cid in ('B0_p15_q8_2nd','B0I_p16_q4_4th'):
        path=root/cid/'output/result.json';body=data(store.file('H8:D_'+cid,path))
        measured[cid]={x['axis']:dict(RZ=x['measured_RZ'],kind='DIRECT_MEASURED',SOURCE=x['SOURCE'],
            compiler_hash=x['compiler_hash'],result_sha256=c.sha(path)) for x in body['rows']}
        for name in ('final_gate.json','terminal.json'):store.file('H8:D_'+cid+':'+name,root/cid/'output'/name)
    metrics=store.blob('H8:original_wrapper_metrics','681ec23c93fdd1090d325908f4b155a79115afdf',
        'artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/wrapper_metrics.csv')
    paired,seed_rows=csv_pairs(metrics,binding,pr_ids,'confirmation')
    with Path(metrics).open() as f:seed_rows=list(csv.DictReader(f))
    return dict(rows=rows,paired_PR=paired,measured=measured,PR_ids=pr_ids,
        deterministic_sets=dict(old32=[x['id'] for x in registry['old']],
                               strengthened184=[x['id'] for x in registry['old']+registry['additional']]),
        comparison_limit='fixed model remains for unmeasured candidates'),seed_rows


def conditional_payload(system,store):
    if system=='H4':root=PROJECTS/'h4-followup-validation/20261018/private/B_run01/output';receipt='compile_receipts.json'
    elif system=='H6':root=ORIGINS['H6']/'B_attempt02/output';receipt='worker_receipts.json'
    else:root=ORIGINS['H8']/'B_run01/output';receipt='worker_receipts.json'
    parent=data(store.file(system+':B_worker_receipts',root/receipt));groups={};seeds=set()
    for batch in parent.values():
        if not isinstance(batch,dict) or 'workers' not in batch:continue
        c.require(batch['all_reaped'] is True,'B_PARENT_REAP')
        for worker in batch['workers']:
            c.require(worker['exit_code']==0 and worker['reaped'] is True,'B_NATIVE_CHILD_EXIT')
            directory=Path(worker['directory']);trace=data(store.file(system+':B_trace:'+directory.name,directory/'trajectory.json'))
            result=data(store.file(system+':B_cost:'+directory.name,directory/'task_result.json',worker['result_sha256']))
            c.require(result['event_digest']==trace['event_digest']
                      and c.digest(trace['events'])==trace['event_digest'],'CONDITIONAL_EVENT_BINDING')
            seeds.update(collect_seeds(trace));seeds.update(collect_seeds(result))
            cell=trace['cell'];key=(cell['id'],trace['stratum']);group=groups.setdefault(key,dict(cell=cell,
                stratum=trace['stratum'],original_distribution=trace['original_distribution'],
                recorded_log_mass=trace['log_stratum_mass'],paired_costs=[]))
            c.require(group['original_distribution']==trace['original_distribution']
                      and group['recorded_log_mass']==trace['log_stratum_mass'],'B_SAME_ORIGINAL_LAW')
            axes={}
            for row in result['rows']:
                axes[row['axis']]=dict(RZ=row.get('metrics',row)['RZ'])
            c.require(set(axes)=={'cosine','sine'},'B_PAIRED_AXES')
            group['paired_costs'].append(dict(axes=axes,event_digest=trace['event_digest'],seed=trace['master_seed']))
    for name in ('final_gate.json','terminal.json'):store.file(system+':B_'+name,root/name)
    cases=[]
    for cid in sorted({key[0] for key in groups}):
        selected=[x for (cell_id,stratum),x in groups.items() if cell_id==cid]
        c.require(all(len(g['paired_costs'])==8 for g in selected),'B_COMPLETE_CONDITIONAL_N8')
        cases.append(dict(cell=selected[0]['cell'],conditional_groups=selected,ordinary_samples=[],
            ordinary_receipt_status='saved ordinary n32 wrapper lineage bound separately; per-trajectory event classification not yet joined'))
    return dict(cases=cases,unconditional_comparison_authorized=False,
        missing_exact_exhaustive_mass_receipt=True,unknown_cost_zero_filled=False),seeds


def prepare(private_root,source,runtime_receipt):
    private=Path(private_root);private.mkdir(parents=True,exist_ok=False)
    store=Store(private/'store');registry=data(b.ROOT/c.REGISTRY)
    runtime=Path(runtime_receipt);runtime_data=data(runtime)
    c.require(not runtime_data['missing'] and not runtime_data['mismatch'],'RUNTIME_RECEIPT_PASS')
    runtime_copy=store.file('shared_runtime_receipt',runtime)
    source_hashes={name:c.sha(b.ROOT/name) for name in b.FILES}
    overview=dict(schema='PR_NINE_CONTRACT_BATCH_V1',SOURCE=source,contracts=[],private_specs={},
        batch_lock=str(private/'BATCH_ONE_SHOT'),authorization_directory=str(private/'authorization'),
        independent_scopes=list(b.SCOPES),cross_contract_scientific_dependencies=[],
        old_A_COST_results_not_consumed_by_D=True,automatic_retry=False,next_stage_authorized=False)
    for system,entry in registry['systems'].items():
        start=len(store.rows);adapter,policy,providers,old=bind_native(system,entry,store,private)
        old_runtime=old['runtime_file_hashes']
        c.require(all(runtime_data['expected'].get(name)==h for name,h in old_runtime.items()),'ORIGINAL_RUNTIME_CLOSURE')
        config_rows=deepcopy(store.rows[start:])
        d_start=len(store.rows);d,seed_rows=d_payload(system,entry,store)
        d_path=store.json(system+':D_accounting_payload',d)
        d_rows=deepcopy(store.rows[d_start:])
        b_start=len(store.rows)
        conditional,seeds=conditional_payload(system,store)
        # Preserve ordinary n32 paired-cost/seed/event lineage without classifying
        # real trajectories or recalculating a mean during preparation.
        conditional['ordinary_wrapper_lineage']=d['paired_PR']
        conditional['ordinary_lineage_rows']=len(d['paired_PR'])*32
        b_path=store.json(system+':B_diagnostic_payload',conditional)
        b_rows=deepcopy(store.rows[b_start:])
        seeds.update(old.get('old_seed_values',[]));seeds.update(collect_seeds(seed_rows))
        # Derive only the frozen metadata seed hierarchy; no RTE event sampling.
        import pr_cost_native as n
        for row in seed_rows:
            if not str(row.get('seed','')).isdigit():continue
            cid=row['cell_id'];q=int(cid.split('_q')[1].split('_')[0])
            seeds.update(n.seed_hierarchy(int(row['seed']),q))
        a_result=ORIGINS[system]/dict(H4='A_run01/output/result.json',H6='A_attempt02/output/result.json',H8='A_recovery01/output/result.json')[system]
        a_path=store.file(system+':A_signal',a_result);signals=data(a_path)['signals']
        c.require({x['cell']['id']:c.digest(x) for x in signals}==
            {x['cell']['id']:x['record_sha256'] for x in entry['records']},'PINNED_A_SIGNAL_RECORDS')
        tasks=[];all_seeds=set(seeds)
        for cell in entry['acquisition']['selected']:
            for replica in range(8):
                value=n.seed(system,entry['binding'],cell,replica,source);hierarchy=n.seed_hierarchy(value,cell['q'])
                c.require(len(hierarchy)==len(set(hierarchy)) and set(hierarchy).isdisjoint(all_seeds),'NEW_SEED_COLLISION')
                all_seeds.update(hierarchy);tasks.append(dict(id=cell['id']+'_'+str(replica),cell=cell,replica=replica,seed=value))
        seed_path=store.json(system+':seed_exclusions',dict(values=sorted(seeds),
            derivation='saved masters and derived seeds; original direct_master_to_step_seeds_v1 for saved wrapper masters'))
        a=dict(A_signal=a_path,signal_record_hashes={x['cell']['id']:c.digest(x) for x in signals},
            native_adapter=adapter,preparation_policy=policy,
            new_seed_registry=dict(tasks=tasks,old_seed_exclusions=seed_path,collisions=0,scientific_trajectories_generated=0))
        a_path=store.json(system+':A_cost_payload',a)
        for stage,payload in [('A_COST',a_path),('D_ACCOUNTING',d_path),('B_ANALYSIS',b_path)]:
            scope=system+'_'+stage;folder=private/'contracts'/scope;folder.mkdir(parents=True)
            # Each scalar scope contains only its payload, provenance and actual
            # runtime; A additionally binds restored providers and raw input bytes.
            names={payload,runtime_copy}
            if stage=='A_COST':names|={x['path'] for x in config_rows}|{a['A_signal'],seed_path}
            elif stage=='D_ACCOUNTING':names|={x['path'] for x in d_rows+config_rows if x['role'].endswith('original_native_config') or x in d_rows}
            else:names|={x['path'] for x in b_rows+config_rows if x['role'].endswith('original_native_config') or x in b_rows}
            role_rows=[]
            for i,path in enumerate(sorted(names)):
                role_rows.append(dict(role='input_'+str(i),path=path,bytes=Path(path).stat().st_size,sha256=c.sha(path)))
            spec=dict(schema='PR_INDIVIDUAL_TECHNICAL_SEAL_V1',scope=scope,system=system,stage=stage,SOURCE=source,
                baseline_SOURCE=BASE_SOURCE,technical_sealed=True,approved=False,runtime_authorization=False,
                binding=entry['binding'],source_hashes=source_hashes,inputs=role_rows,
                runtime_receipt=runtime_copy,runtime_receipt_sha256=c.sha(runtime_copy),
                provider_closure=providers if stage=='A_COST' else {},payload=payload,
                python=old['python'],python_sha256=c.sha(old['python']),calls=entry['proposed_calls'] if stage=='A_COST' else
                dict(saved_scalar_pass=1,trajectory=0,wrapper_compile=0,control_probe=0,signal=0,fit=0),
                output=str(folder/'output'),lock=str(folder/'ONE_SHOT'),retry=False,next_stage_authorized=False,
                memory_estimate_per_worker_bytes={'H4':2,'H6':6,'H8':24}[system]*2**30 if stage=='A_COST' else 256*2**20,
                output_estimate_bytes=(8 if system=='H8' else 2)*2**30 if stage=='A_COST' else 128*2**20,
                inode_estimate=8192 if stage=='A_COST' else 128,
                resource_caps=None,allowed_cpus=[],fresh_resource_assignment_required=True,
                scope_limit='recorded conditional-cost diagnostic only; unconditional mean unresolved'
                    if stage=='B_ANALYSIS' else 'saved direct measurements and original confirmed PR; no new A_COST results'
                    if stage=='D_ACCOUNTING' else 'fixed sixteen S2 PR candidates, fresh exploration n8 only')
            spec['technical_digest']=b.metadata_binding(spec);b.sealed(spec)
            path=folder/'technical_seal.json';c.save(path,spec);overview['private_specs'][scope]=str(path)
            overview['contracts'].append(dict(scope=scope,system=system,stage=stage,technical_digest=spec['technical_digest'],
                status='READY_PARTIAL_DIAGNOSTIC' if stage=='B_ANALYSIS' else 'READY',
                calls=spec['calls'],scope_limit=spec['scope_limit'],input_files=len(role_rows),provider_files=sum(len(x) for x in spec['provider_closure'].values())))
    overview['batch_digest']=c.digest(overview)
    (private/'authorization').mkdir();c.save(private/'batch.json',overview)
    draft=dict(schema='INVALID_INTEGRATED_APPROVAL_DRAFT',batch_digest=overview['batch_digest'],
        approved_by_user=False,runtime_authorization=False,selected_scopes=[],individual_grants={},
        retry=False,next_stage_authorized=False)
    c.save(private/'approval_INVALID.json',draft)
    c.save(private/'input_provenance_PRIVATE.json',dict(files=store.rows,scientific_values_reaccounted=False,
        real_preparations_restored=0,real_native_compile=0,new_scientific_statistics=0))
    return overview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private',type=Path,required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--runtime-receipt',type=Path,required=True);args=parser.parse_args()
    result=prepare(args.private,args.source,args.runtime_receipt)
    print(json.dumps(dict(status='NINE_TECHNICAL_SEALS_ONLY',batch_digest=result['batch_digest'],
                         scopes=[x['scope'] for x in result['contracts']],scientific_calls=0)))


if __name__=='__main__':main()
