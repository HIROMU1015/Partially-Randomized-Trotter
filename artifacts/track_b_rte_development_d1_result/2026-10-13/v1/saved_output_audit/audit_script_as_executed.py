"""Post-STOP saved JSON/bytes/rational accounting only; no science imports."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import json,hashlib,csv,subprocess,shutil,sys

ROOT=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-rte-development-d1-one-shot-execution-20261013')
P=ROOT/'artifacts/track_b_rte_development_d1_result/2026-10-13/v1'
PREP=ROOT/'artifacts/track_b_rte_development_source/2026-10-13/v1'
S='874ca12d347f8f35e5d8d64f65f24abc4a2df523'
A='f803291638d453873e4ab6f1ecf7bcc607f48e4b'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_development.evidence import digest,verify_chunks

def load(path):return json.loads(Path(path).read_text())
def write(path,value):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda q:str(q))+'\n')
def approx(q):
    q=F(q)
    with localcontext() as ctx:ctx.prec=12;return str(Decimal(q.numerator)/Decimal(q.denominator))

c=load(PREP/'contract_v1.json');auth=load(PREP/'authorization.json')
terminal=load(P/'terminal.json');outer=load(P/'outer_process_receipt.json');marker=load(P/'one_shot_consumed.json')
model=load(P/'resource_map.json');inventory=load(P/'inventory.json');cache=load(P/'synthesis_cache.json')
manifest=load(PREP/'source_manifest_v1.json');protected=load(PREP/'protected_hashes_v1.json')
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==A
assert subprocess.check_output(['git','-C',str(ROOT),'show','-s','--format=%P',A],text=True).strip()==S
assert terminal['status']==outer['effective_status']==model['classification']=='D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
assert outer['exit_code']==0 and outer['signal'] is None and outer['complete_status_consistent'] is True
assert digest(P/'terminal.json')==outer['terminal_sha256']==load(P/'STOP.json')['terminal_sha256']
ident=dict(source_commit=S,execution_HEAD=A,contract_sha256=digest(PREP/'contract_v1.json'),authorization_sha256=digest(PREP/'authorization.json'))
for obj in (terminal,outer,marker):
    for key,val in ident.items():assert obj[key]==val
    assert obj['runs']==1 and obj['retries']==0 and obj['mandatory_STOP'] is True
assert terminal['marker_sha256']==digest(P/'one_shot_consumed.json')
assert terminal['inventory_sha256']==digest(P/'inventory.json')
assert all(digest(ROOT/f)==h for f,h in manifest['critical_sha256'].items())
assert all(digest(ROOT/f)==h for f,h in protected['sha256'].items())
saved_manifest=load(P/'evidence_manifest.json');chunks=0
for item in saved_manifest['files']:
    path=P/item['path'];assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256']
    if item['chunks']:
        assert verify_chunks(P,item);chunks+=len(item['chunks'])
        assert all(r['bytes']<=524288 for r in item['chunks'])
assert not list(P.rglob('*.partial'))
assert len(model['rows'])==12 and len(model['primary_relations'])==6
assert {(v['steps'],v['arm']) for v in model['rows']}=={(r,a) for r in c['steps'] for a in c['arms']}
invkeys={k['ratio']:k for k in inventory['keys']};assert set(invkeys)==set(cache) and len(invkeys)==36
keytable=[]
for ratio,value in cache.items():
    seq=value['sequence'];assert hashlib.sha256(seq.encode()).hexdigest()==value['sequence_sha256']
    assert value['T_count']==seq.count('T')+seq.count('t')
    assert value['Tdagger_count']==seq.count('t')
    assert value['one_qubit_count']==len(seq)-seq.count('W')
    assert value['global_W_count']==seq.count('W')
    assert value['epsilon']==c['primitive_error'] and value['angle_key']=='atan:'+ratio+':scale:1'
    assert value['error_pass'] is True and 0<=F(value['strict_operator_error_upper'])<=F(c['primitive_error'])
    keytable.append({'ratio':ratio,'inventory_key':invkeys[ratio]['key'],'acquisition':invkeys[ratio]['acquisition'],
        **{k:value[k] for k in ('sequence_sha256','T_count','Tdagger_count','one_qubit_count','global_W_count','strict_operator_error_upper')}})
jobs=list((P/'jobs').glob('key_[0-9][0-9][0-9].json'))
assert len(jobs)==17==model['new_synthesis_calls']==terminal['new_synthesis_calls']
for path in jobs:
    job=load(path);start=load(path.with_suffix('.started.json'));v=load(path.with_suffix('.result.json'))
    assert v['status']=='KEY_COMPLETE' and v['synthesis_calls']==1 and v['retries']==0
    for key,val in ident.items():assert v[key]==job[key]==val
    assert v['inventory_sha256']==job['inventory_sha256']==start['inventory_sha256']==digest(P/'inventory.json')
    assert v['marker_sha256']==job['marker_sha256']==start['marker_sha256']==digest(P/'one_shot_consumed.json')
    assert v['key']==job['key']==start['key']==invkeys[job['ratio']]['key']
    assert v['saved']==cache[job['ratio']]
oldblob=subprocess.check_output(['git','-C',str(ROOT),'show',c['saved_anchor']['commit']+':'+c['saved_anchor']['path']])
assert hashlib.sha256(oldblob).hexdigest()==c['saved_anchor']['sha256']
oldcache=json.loads(oldblob)['synthesis_cache']
assert len(oldcache)==19
assert all(cache[ratio]==v for ratio,v in oldcache.items())

event_count=0;maximum_ideal=maximum_actual=0.;tables=[]
for index,row in enumerate(model['rows']):
    assert load(P/f'rows/row_{index:02d}.json')==row
    es=load(P/row['events']['path']);event_count+=len(es);q=m2=F(0);maximum=F(0);per={k:F(0) for k in ('T','CX','1Q')};worst={k:0 for k in per}
    for num,binding in enumerate(es):
        assert binding['index']==num; e=binding['event'];a,p,w=map(F,(e['coefficient'],e['proposal'],e['weight']))
        assert a>0 and p>0 and w==a/p
        q+=p;m2+=p*w*w;maximum=max(maximum,w)
        price=binding['native_cost']
        # Recount the saved IR with its saved sequence inventory; no lowering/compiling.
        actual={k:0 for k in per}
        for g in binding['native_ir']:
            if g[0]=='R':
                actual['T']+=cache[g[2]]['T_count'];actual['1Q']+=cache[g[2]]['one_qubit_count']
            elif g[0]=='CX':actual['CX']+=1
            else:
                assert g[0]!='CZ'
                actual['1Q']+=g[0] not in ('W','w');actual['T']+=g[0] in ('T','t')
        assert actual=={k:price[k] for k in per}
        for k in per:per[k]+=p*price[k];worst[k]=max(worst[k],price[k])
        assert binding['up_to_phase'] is False
        assert binding['ideal_error_diagnostic']<=c['matrix_tolerance']
        assert binding['actual_error_diagnostic']<=float(F(price['strict_error_upper']))+c['matrix_tolerance']
        maximum_ideal=max(maximum_ideal,binding['ideal_error_diagnostic']);maximum_actual=max(maximum_actual,binding['actual_error_diagnostic'])
    law=row['reference_law_diagnostic_only'];bounds=row['production_step_bounds'];budget=row['budget'];r=row['steps'];n=budget['N_per_axis']
    assert q==F(law['acceptance']) and m2==F(law['M2']) and maximum==F(law['range'])
    assert 0<q<=F(bounds['acceptance_upper'])<=1 and m2<=F(bounds['M2_upper']) and maximum<=F(bounds['range_upper'])
    assert all(per[k]==F(row['per_step_trial_native'][k]) for k in per) and worst==row['worst_single_step_event_native']
    assert F(budget['M2_upper'])==F(bounds['M2_upper'])**r
    assert F(budget['range_upper'])==F(bounds['range_upper'])**r
    assert F(budget['acceptance_upper'])==F(bounds['acceptance_upper'])**r
    ep,e=F(budget['Taylor_step_upper']),F(budget['rounded_native_step_upper'])
    assert F(budget['axis_bias_upper'])==2*((1+ep+e)**r-1)
    assert F(budget['remaining'])==F(c['epsilon_axis'])-F(budget['axis_bias_upper'])>0
    assert F(row['prep_readout_slope'])==2*n*q**r
    for k in per:
        assert F(row['per_whole_trial_native'][k])==r*q**(r-1)*per[k]
        assert F(row['expected_native_intercept'][k])==2*n*r*q**(r-1)*per[k]
        assert F(row['hard_native'][k])==2*n*r*worst[k]
        assert F(row['accepted_tail_native'][k])==budget['accepted_tail_calls']*r*worst[k]
    assert row['ideal_step_mean_residual_diagnostic']<=c['matrix_tolerance']
    assert row['ideal_product_mean_residual_diagnostic']<=c['matrix_tolerance']
    tables.append(dict(steps=r,x=row['x'],arm=row['arm'],events=len(es),N_per_axis=n,
        acceptance=str(q),weight_M2=str(m2),weight_range=str(maximum),M2_budget_upper=budget['M2_upper'],range_budget_upper=budget['range_upper'],
        axis_bias_upper=budget['axis_bias_upper'],statistical_width=budget['remaining'],prep_readout_slope=row['prep_readout_slope'],workspace=row['workspace_beyond_system'],
        **{f'{k}_{field}':row[source][k] for k in per for field,source in [('per_step_trial','per_step_trial_native'),('per_whole_trial','per_whole_trial_native'),('intercept','expected_native_intercept'),('tail_upper','accepted_tail_native'),('hard_upper','hard_native')]}))
assert event_count==1890<=c['caps']['event_bindings']
primary=[]
for relation in model['primary_relations']:
    r,k=relation['steps'],relation['metric'];by={row['arm']:row for row in model['rows'] if row['steps']==r}
    d0=F(by[relation['left']]['expected_native_intercept'][k])-F(by[relation['right']]['expected_native_intercept'][k])
    d1=F(by[relation['left']]['prep_readout_slope'])-F(by[relation['right']]['prep_readout_slope'])
    assert d0==F(relation['difference_intercept']) and d1==F(relation['difference_slope'])
    cross=F(relation['crossover']) if relation['crossover'] is not None else None
    assert cross==(-d0/d1 if d1 and -d0/d1>=0 else None)
    primary.append({**relation,'intercept_difference_decimal_display':approx(d0),'slope_difference_decimal_display':approx(d1),
        'crossover_decimal_display':approx(cross) if cross is not None else None})

audit=P/'saved_output_audit';audit.mkdir(exist_ok=False)
write(audit/'native_resource_rows_exact_v1.json',tables)
write(audit/'synthesis_sequence_index_v1.json',keytable)
write(audit/'primary_affine_relations_v1.json',primary)
with (audit/'native_resource_rows_exact_v1.csv').open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
logs=audit/'outer_logs';logs.mkdir()
for item in outer['external_logs']:
    source=Path(item['path']);assert source.stat().st_size==item['bytes'] and digest(source)==item['sha256']
    shutil.copyfile(source,logs/source.name);assert digest(logs/source.name)==item['sha256']
shutil.copyfile('/tmp/track-b-d1-preflight-metadata.json',audit/'independent_preflight_v1.json')
shutil.copyfile('/tmp/track-b-d1-execution-intake.json',audit/'execution_intake_v1.json')
parent=Path('/home/abe/Project/Partially Randomized Trotter');before=load('/tmp/track-b-d1-execution-intake.json')
assert subprocess.check_output(['git','-C',str(parent),'rev-parse','HEAD'],text=True).strip()==before['parent_HEAD']
assert hashlib.sha256(subprocess.check_output(['git','-C',str(parent),'diff','--binary'])).hexdigest()==before['parent_dirty_diff_SHA256']
assert digest(parent/'.git/index')==before['parent_index_SHA256']
resource=terminal['resource'];assert resource['parent_CPU_seconds']+resource['completed_worker_CPU_seconds']<c['caps']['total_CPU_seconds']
assert resource['peak_observed_aggregate_RSS_bytes']<=c['caps']['aggregate_RSS_MiB']*1048576
assert outer['parent_peak_RSS_KiB']<=c['caps']['parent_RSS_MiB']*1024
assert not outer['external_log_cap_exceeded']
result={'status':'SAVED_OUTPUT_AND_PROVENANCE_AUDIT_PASS','classification':terminal['status'],**ident,
    'one_shot_marker_SHA256':digest(P/'one_shot_consumed.json'),'source_manifest_SHA256':digest(PREP/'source_manifest_v1.json'),
    'completed_rows':12,'axes':24,'event_bindings':event_count,'primary_relations':6,'inventory_keys':36,'reused_fixed_keys':19,'new_synthesis_calls':17,
    'all_key_sequence_counts_error_record_identity':True,'all_event_native_prices_recounted_from_saved_IR':True,
    'all_dyadic_weights_moment_acceptance_accounting':True,'exact_rational_product_cost_equations':True,'strict_phase_diagnostic_coverage':True,
    'max_saved_ideal_native_error_diagnostic':maximum_ideal,'max_saved_actual_native_error_diagnostic':maximum_actual,
    'raw_and_review_chunk_reconstruction':True,'review_chunks_verified':chunks,'source_contract_protected_181_files_unchanged':True,
    'external_exit_terminal_consistent':True,'retry_count':0,'new_synthesis_or_matrix_or_science_in_audit':0,
    'run_quantum_shots_sampling_trajectory_LP_DF_molecule_NPZ_GPU':0,'classical_online_cost':'NOT_ACQUIRED; no full total-cost claim',
    'resource':resource,'outer_resource':{k:outer[k] for k in ('wall_seconds','CPU_seconds_including_children','parent_peak_RSS_KiB')},
    'parent_HEAD_dirty_diff_raw_index_unchanged':True,'future_science_authorized':False,'method_adoption':False,'mandatory_STOP':True}
write(audit/'saved_output_audit_v1.json',result)
files=[]
for f in sorted(P.rglob('*')):
    if f.is_file():files.append({'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':digest(f),'direct_review_fetch_size_ok':f.stat().st_size<=524288})
write(audit/'final_evidence_inventory_v1.json',{'files':files,'original_files_kept':True,'relative_root':str(P.relative_to(ROOT)),
    'oversize_raw_files_use_original_evidence_manifest_chunks':[r['path'] for r in files if not r['direct_review_fetch_size_ok']],
    'original_evidence_manifest_SHA256':digest(P/'evidence_manifest.json'),'source':S,'authorization':A,'mandatory_STOP':True})
print(json.dumps({'audit':'PASS','rows':12,'events':event_count,'review_chunks':chunks,'keys':36,'new_synthesis':17,'marker':result['one_shot_marker_SHA256'],
    'resource':result['outer_resource'],'primary_saved_relations':[(r['steps'],r['metric'],r['classification'],r['crossover_decimal_display']) for r in primary]},ensure_ascii=False))
