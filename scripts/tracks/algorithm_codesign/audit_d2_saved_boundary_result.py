#!/usr/bin/env python3
"""Audit saved D2 scalar/identity records only; no boundary kernel call."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
S='28fc3ca17ec51bb7656a498e3eea9b72966eb3fa'
A='9e753a0d807df683c86f7742ed0c14ddcd4e58ba'
CP=Path('artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json')


def load(path):return json.loads(Path(path).read_text())


def identity(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(65536),b''):h.update(v)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def encode(value):
    if isinstance(value,F):return str(value)
    raise TypeError('unsupported evidence type')


def write(path,value):
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,default=encode)+'\n').encode()
    if len(data)>524288:raise RuntimeError('audit review file size')
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data)


def affine(left,right):
    d=F(left['intercept'])-F(right['intercept'])
    b=F(left['slope'])-F(right['slope'])
    return dict(left_minus_right_intercept=d,left_minus_right_slope=b,
                nonnegative_crossing=(-d/b if b and -d/b>=0 else None),
                relation='left cheaper iff d+b*h<0; h>=0',new_budget_optimization=False)


def main():
    started=time.monotonic();c=load(ROOT/CP);d=ROOT/c['result_directory'];base=ROOT/c['saved_result_directory']
    before={str(p.relative_to(d)):identity(p) for p in d.rglob('*') if p.is_file()}
    audit=d/'saved_output_audit';audit.mkdir(exist_ok=False)
    checks=[]
    def check(label,condition):
        if not condition:raise ArithmeticError(label)
        checks.append(label)
    auth=load(ROOT/c['authorization_path']);terminal=load(d/'terminal.json')
    marker=load(d/'one_shot_consumed.json');outer=load(d/'outer_process_receipt.json');stop=load(d/'STOP.json');rm=load(d/'resource_map.json')
    status='D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
    check('A direct parent S',subprocess.check_output(['git','show','-s','--format=%P',A],text=True).strip()==S)
    changed=set(subprocess.check_output(['git','diff','--name-only',S,A],text=True).splitlines())
    check('authorization-only two paths',changed=={c['authorization_path'],c['optional_receipt_path']})
    check('auth full binding',auth['source_commit']==S and auth['status']=='APPROVED_FOR_ONE_D2_BOUNDARY_RUN' and auth['science_execution_authorized'] is True)
    for name,v in [('terminal',terminal),('outer',outer),('marker',marker)]:
        check(name+' source/HEAD',v['source_commit']==S and v['execution_HEAD']==A)
        check(name+' contract/auth SHA',v['contract_sha256']==identity(ROOT/CP)['sha256'] and v['authorization_sha256']==identity(ROOT/c['authorization_path'])['sha256'])
        check(name+' run1 retry0 STOP',v['runs']==1 and v['retries']==0 and v['mandatory_STOP'] is True)
    check('terminal/outer/resource status',terminal['status']==outer['terminal_status']==outer['effective_status']==rm['classification']==status)
    check('outer exit/signal/complete',outer['exit_code']==0 and outer['signal'] is None and outer['complete_status_consistent'] is True and not outer['external_log_cap_exceeded'])
    check('marker SHA',terminal['marker_sha256']==identity(d/'one_shot_consumed.json')['sha256'])
    check('terminal SHA outer/STOP',outer['terminal_sha256']==stop['terminal_sha256']==identity(d/'terminal.json')['sha256'])
    check('no next science',stop['mandatory_STOP'] is True and terminal['next_science_authorized'] is False and outer['next_science_authorized'] is False)
    for k in ('new_synthesis','new_angles','matrix','native_compile','sampling','GPU'):
        check(k+' zero',terminal[k]==0)
    check('six rows/219303 pairs',terminal['rows']==6 and terminal['registered_pairs']==rm['registered_pairs']==219303 and len(rm['rows'])==6)
    manifest=load(d/'evidence_manifest.json')
    for v in manifest['files']:
        check('output bytes/SHA '+v['path'],identity(d/v['path'])=={k:v[k] for k in ('bytes','sha256')})
        for part in v.get('chunks',[]):
            check('chunk '+part['path'],identity(d/part['path'])=={k:part[k] for k in ('bytes','sha256')})
        if v.get('chunks'):
            raw=b''.join((d/part['path']).read_bytes() for part in v['chunks'])
            check('chunk reconstruction '+v['path'],len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256'])
    rows=[];total=0
    for spec,v in zip(c['registered_rows'],rm['rows']):
        i=spec['row_index'];old=load(base/spec['row_path']);new=load(d/f'rows/row_{i:02d}.json')
        check('resource row match '+str(i),v==new)
        check('row scope/pair '+str(i),new['arm']==old['arm']==spec['arm'] and new['steps']==2 and new['positive_pairs']==spec['positive_step_support']**2==spec['positive_pair_bound'])
        total+=new['positive_pairs']
        Z=F(old['reference_law_diagnostic_only']['acceptance']);N=old['budget']['N_per_axis']
        check('budget/N frozen '+str(i),new['budget_unchanged']==old['budget'] and new['N_per_axis']==N)
        check('slope/workspace frozen '+str(i),new['prep_readout_slope_unchanged']==old['prep_readout_slope'] and new['workspace_unchanged']==old['workspace_beyond_system'])
        check('old tail/hard retained '+str(i),new['old_accepted_tail_native_retained']==old['accepted_tail_native'] and new['old_hard_native_retained']==old['hard_native'])
        check('old bias retained '+str(i),new['old_error_upper_retained'] is True and new['old_D1_classification_changed'] is False)
        masses={int(k):F(v) for k,v in new['depth_probability_unconditional'].items()}
        check('depth masses acceptance '+str(i),sum(masses.values(),F(0))==F(new['acceptance_squared'])==Z*Z and all(k>=0 and v>=0 for k,v in masses.items()))
        r=dict(row_index=i,arm=new['arm'],steps=2,N_per_axis=N,acceptance=Z,
               positive_pairs=new['positive_pairs'],denominator_bits=new['denominator_bits'],
               prep_readout_slope=F(old['prep_readout_slope']),workspace=new['workspace_unchanged'],
               exact_depth_mass=masses,cancellation_probability_unconditional=sum(v for k,v in masses.items() if k>0),
               cancellation_probability_given_nonzero=sum(v for k,v in masses.items() if k>0)/(Z*Z),
               no_pair_re_evaluation=True,metrics={})
        for k in ('T','CX','1Q'):
            original=F(new['additive_per_whole_trial'][k]);saving=F(new['cancellation_saving_per_whole_trial'][k]);after=F(new['after_per_whole_trial'][k])
            check('additive price '+str(i)+k,original==F(old['per_whole_trial_native'][k])==2*Z*F(old['per_step_trial_native'][k]))
            check('exact savings '+str(i)+k,original-saving==after and 0<=saving<=original)
            oldG=F(new['original_expected_native_intercept'][k]);newG=F(new['new_expected_native_intercept'][k])
            check('shots/intercept '+str(i)+k,oldG==F(old['expected_native_intercept'][k])==2*N*original and newG==2*N*after and oldG-newG==2*N*saving)
            worst=F(new['worst_nonzero_pair'][k])
            check('worst conservative '+str(i)+k,0<=worst<=2*old['worst_single_step_event_native'][k] and after<=Z*Z*worst)
            r['metrics'][k]=dict(original_intercept=oldG,after_intercept=newG,
                savings_intercept=oldG-newG,relative_savings=(oldG-newG)/oldG if oldG else None,
                original_per_whole_trial=original,after_per_whole_trial=after,
                saving_per_whole_trial=saving,diagnostic_worst_nonzero_pair=worst)
        rows.append(r)
    check('total registered coverage',total==219303)
    check('CPU/RSS cap',outer['CPU_seconds_including_children']<=c['caps']['total_CPU_seconds'] and outer['parent_peak_RSS_KiB']<=c['caps']['parent_RSS_MiB']*1024 and terminal['resource']['peak_observed_aggregate_RSS_bytes']<=c['caps']['aggregate_RSS_MiB']*1048576)
    check('output bytes cap',sum(v['bytes'] for v in before.values())<=c['caps']['output_bytes'])
    for log in outer['external_logs']:
        src=Path(log['path']);check('process log '+src.name,identity(src)=={k:log[k] for k in ('bytes','sha256')} and src.stat().st_size<=c['caps']['per_key_log_bytes'])
        dst=audit/'outer_logs'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
    comparisons=[]
    p=next(r for r in rows if r['arm']=='partial_return_tail');w=next(r for r in rows if r['arm']=='closed_P5_full')
    for k in ('T','CX','1Q'):
        original=w['metrics'][k]['original_intercept']/p['metrics'][k]['original_intercept']
        after=w['metrics'][k]['after_intercept']/p['metrics'][k]['after_intercept']
        comparisons.append(dict(metric=k,direction='P5 relative to partial',old_ratio=original,new_ratio=after,
                old_relative_difference=original-1,new_relative_difference=after-1,
                new_affine_relation=affine(dict(intercept=w['metrics'][k]['after_intercept'],slope=w['prep_readout_slope']),dict(intercept=p['metrics'][k]['after_intercept'],slope=p['prep_readout_slope'])),
                h_is_metric_specific=True,scientific_adoption=False))
    rankings={k:dict(original=[r['arm'] for r in sorted(rows,key=lambda r:r['metrics'][k]['original_intercept'])],after=[r['arm'] for r in sorted(rows,key=lambda r:r['metrics'][k]['after_intercept'])]) for k in ('T','CX','1Q')}
    source_manifest=load(ROOT/c['source_manifest']);ledger=load(ROOT/c['protected_ledger'])
    for name,table in [('critical',source_manifest['critical_sha256']),('protected',ledger['sha256'])]:
        for path,sha in table.items():
            check(name+' no NPZ '+path,not path.lower().endswith('.npz'))
            check(name+' SHA '+path,identity(ROOT/path)['sha256']==sha)
    check('contract equals S',subprocess.check_output(['git','show',S+':'+str(CP)])==(ROOT/CP).read_bytes())
    check('authorization equals A',subprocess.check_output(['git','show',A+':'+c['authorization_path']])==(ROOT/c['authorization_path']).read_bytes())
    for f,v in before.items():check('raw output immutable '+f,identity(d/f)==v)
    write(audit/'resource_rows_exact_v1.json',{'rows':rows,'scope':'saved scalar accounting only, no rerun or new circuit evaluation'})
    write(audit/'primary_and_rank_relations_v1.json',{'primary':comparisons,'h0_rankings':rankings,'scientific_classification_unchanged':True})
    write(audit/'display_summary_v1.json',{'rows':[dict(arm=r['arm'],positive_pairs=r['positive_pairs'],metrics={k:{field:float(v[field]) for field in ['original_intercept','after_intercept','relative_savings']} for k,v in r['metrics'].items()}) for r in rows],
        'primary':[dict(metric=v['metric'],original_percent=100*float(v['old_relative_difference']),after_percent=100*float(v['new_relative_difference']),crossing=float(v['new_affine_relation']['nonnegative_crossing']) if v['new_affine_relation']['nonnegative_crossing'] is not None else None) for v in comparisons],
        'exact_authority':['resource_rows_exact_v1.json','primary_and_rank_relations_v1.json']})
    write(audit/'raw_output_identity_v1.json',{'files_before_audit':before,'all_unchanged':True,'result_bytes_before_audit':sum(v['bytes'] for v in before.values())})
    write(audit/'saved_output_audit_v1.json',{'status':'PASS','checks':checks,'check_count':len(checks),
        'source_S':S,'authorization_A':A,'marker_sha256':identity(d/'one_shot_consumed.json')['sha256'],
        'original_status':status,'runs':1,'retries':0,'mandatory_STOP':True,
        'audit_new_pair_calls':0,'audit_matrix_compile_synthesis_sampling_LP_DF_NPZ_GPU':0,
        'protected_files':len(ledger['sha256']),'critical_files':len(source_manifest['critical_sha256']),
        'scientific_interpretation_or_algorithm_adoption':False,'wall_seconds':time.monotonic()-started})
    print(json.dumps({'audit':'PASS','checks':len(checks),'primary':[{k:v for k,v in r.items() if k in ('metric','old_relative_difference','new_relative_difference')} for r in comparisons]},default=encode))


if __name__=='__main__':main()
