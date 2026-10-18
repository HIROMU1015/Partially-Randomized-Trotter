"""Three separately granted one-shot stages; no sampling/native build/compile."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
from fractions import Fraction as F
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='7e454b91185ed7b0f92a3790f800ba5d65d5f156'
OLD='artifacts/resource_applicability/track_a_pr_theory_compiled_h46_completed/2026-10-18'
SITES='artifacts/resource_applicability/track_a_h46_s4_primitive_sites/2026-10-18'
CASES={'H4':('B1_p12_q1_4th',12,1),'H6':('B1_p19_q8_4th',19,8)}
MODELS=('A','B','NATIVE_MIXED')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def old(name):return json.loads(subprocess.check_output(['git','-C',str(ROOT),'show',f'{ORIGINAL}:{OLD}/{name}']))


def verify(plan,grant):
    if not (plan['stage'] in (1,2,3) and grant['stage']==plan['stage'] and
            grant['approved'] is True and grant['runtime_authorization'] is True and
            grant['one_shot'] is True and grant['automatic_retry'] is False and
            grant['plan_digest']==digest(plan) and grant['SOURCE']==plan['SOURCE']):
        raise ValueError('NEW_STAGE_GRANT_REQUIRED')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=plan['SOURCE']:
        raise ValueError('SOURCE_CHANGED')
    for path,sha in plan['source_hashes'].items():
        if file_sha(ROOT/path)!=sha:raise ValueError('SOURCE_CLOSURE:'+path)
    for r in plan['inputs'].values():
        p=Path(r['path'])
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or file_sha(p)!=r['sha256']:raise ValueError('INPUT_BINDING')
    if list(sys.version_info[:3])!=plan['environment']['python'] or importlib.metadata.version('numpy')!=plan['environment']['numpy']:
        raise ValueError('ENVIRONMENT_VERSION')
    for path,sha in plan['environment']['files'].items():
        if file_sha(path)!=sha:raise ValueError('ENVIRONMENT_FILES')
    if plan['stage']>1:
        predecessor=read(plan['predecessor_terminal'])
        native=read(plan['predecessor_native_exit'])
        if (predecessor['status']!='COMPLETE_MANDATORY_STOP' or predecessor['stage']!=plan['stage']-1 or
            predecessor['SOURCE']!=plan['predecessor_SOURCE'] or native['SOURCE']!=plan['predecessor_SOURCE'] or
            native['native_exit_code']!=0 or not native['wait_reap_completed']):
            raise ValueError('PREDECESSOR_NOT_ACCEPTED')
    out=Path(plan['output']).resolve()
    if not out.is_relative_to(Path.home()):raise ValueError('HOME_OUTPUT')
    return out


def reserve(plan,out,kind,**fields):
    caps={1:{'artificial_suite':3},2:{'candidate':2,'primitive':66},3:{'reaccount':1}}[plan['stage']]
    if kind not in caps:raise ValueError('UNAPPROVED_CALL_KIND')
    with (out/'calls.jsonl').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);rows=[json.loads(line) for line in f]
        if sum(r['kind']==kind for r in rows)>=caps[kind]:raise ValueError('STAGE_CALL_CAP')
        f.seek(0,2);f.write(json.dumps(dict(kind=kind,**fields))+'\n');f.flush();os.fsync(f.fileno())


def stage1(plan,out):
    suite=unittest.TestSuite();loader=unittest.TestLoader()
    for index,name in enumerate(('test_h46_s4_compatibility.py','test_native_df_gate_model.py','test_pr_theory_compiled_h46.py')):
        reserve(plan,out,'artificial_suite',file=name)
        suite.addTests(loader.loadTestsFromModule(load('_compat_test_'+str(index),ROOT/'tests'/name)))
    with (out/'tests.txt').open('x') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(out/'tests_summary.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)))
    if not result.wasSuccessful():raise ValueError('ARTIFICIAL_REGRESSION_FAILURE')
    a=load('_compat_audit_hex',ROOT/'scripts/audit_h46_s4_count_mismatch.py');f=a.functions()
    m=load('_compat_hex_counter',ROOT/'src/trotterlib/native_df_gate_model.py')
    receipts=[]
    for q in (1,3,8):
        actual=m.pf_schedule(2,.8,q,'4th','symmetric_directional')
        native=f.schedule.deterministic_control_plan(2,.8,q,'4th')
        receipts.append(dict(q=q,stages=[dict(block=s.block,mode=s.mode,model_time=s.time.hex(),native_time=t.time.hex()) for s,t in zip(actual,native,strict=True)]))
    write(out/'stage_time_hex.json',receipts)


def case_worker(plan,grant,out,system):
    verify(plan,grant);own=out/system;own.mkdir()
    x=load('_compat_execution',ROOT/'scripts/h46_cost_execution.py')
    a=load('_compat_prior',ROOT/'scripts/audit_h46_s4_count_mismatch.py');f=a.functions()
    v=load('_compat_sites',ROOT/'scripts/validate_h46_s4_primitive_sites.py')
    archive=v.checked_input(system,plan['inputs'][system]['path'])
    if system=='H4':archive=dict(schema='h6_frozen_df_preparations_v1',binding={},codec=archive,codec_digest=x.archive.digest(archive),export_receipt={})
    graph=x.archive.FrozenMetadataGraph(archive)
    cid,prefix,q=CASES[system];p=x.from_graph(graph,archive['codec']['roots'][f'faithful:p{prefix}'])
    if p['preparation_hash']!=v.CONFIG[system]['preparation'] or p['deterministic_indices']!=list(range(prefix)) or p['randomized_indices']:
        raise ValueError('PREPARATION_IDENTITY')
    prior=read(ROOT/SITES/'primitive_sites.json')[system];cache={};checks=[]
    def factory(j,t):
        key=j,float(t).hex()
        if key not in cache:
            reserve(plan,out,'primitive',system=system,block=j,time_hex=key[1])
            b=p['blocks'][j]
            pp=(f.ops.one_body_diagonal_primitives(f.ops.np.asarray(b['diagonal']),t) if b['one'] else
                f.ops.df_squared_diagonal_primitives(f.ops.np.asarray(b['diagonal']),b['lam'],t))
            if t<0:
                actual=v.primitive_receipt(pp,p['n']);expected=prior['blocks'][j]['native']
                if actual!=expected or key[1]!=prior['native_negative_time_hex']:
                    raise ValueError('SAVED_NATIVE_PRIMITIVE_MISMATCH')
                checks.append(dict(block=j,time_hex=key[1],rz_mask=list(actual['rz']),rzz_mask=actual['nonzero_rzz_pairs'],phase_nonzero=actual['phase_nonzero'],full_primitive_receipt_match=True))
            cache[key]=pp
        return cache[key]
    results=[]
    for model in ('A','B'):
        circuit=x.m.deterministic_circuit(tuple(b['basis'] for b in p['blocks']),factory,.8,q,'4th','symmetric_directional',p['n'],-p['constant']*.8,model)
        evolution_cost=x.m.count(circuit)
        for axis,quad in (('cosine','X'),('sine','Y')):
            cost=evolution_cost+x.m.count(x.m.hadamard_wrapper(x.m.Circuit(),p['n'],quad))
            results.append(dict(system=system,candidate_id=cid,model=model,axis=axis,lower=x.packed_cost(cost),upper=x.packed_cost(cost),theory_kind='DETERMINISTIC',theory_profile=x.m.PROFILE,preparation_hash=p['preparation_hash'],control='symmetric_directional',all_positive_orders_included=False))
        del circuit
    # PF NATIVE_MIXED is exactly A by definition; no third duplicate assembly.
    results += [dict(r,model='NATIVE_MIXED') for r in results if r['model']=='A']
    if len(checks)!=len(p['blocks']) or len(cache)!=2*len(p['blocks']):raise ValueError('FIXED_PRIMITIVE_INVENTORY')
    write(own/'theory.json',results);write(own/'primitive_reuse_checks.json',checks)
    return dict(system=system,primitive_calls=len(cache),negative_stage_checks=len(checks))


def stage2(plan,grant,out):
    children=[];witnesses=[]
    try:
        for system in CASES:
            reserve(plan,out,'candidate',system=system)
            stdout=(out/(system+'_stdout.txt')).open('x');stderr=(out/(system+'_stderr.txt')).open('x')
            child=subprocess.Popen([sys.executable,'-P','-B',str(Path(__file__).resolve()),'--plan',plan['_plan_path'],'--grant',plan['_grant_path'],'--worker',system],stdout=stdout,stderr=stderr)
            children.append((system,child,stdout,stderr))
        for system,child,stdout,stderr in children:
            code=child.wait();stdout.close();stderr.close();witnesses.append(dict(system=system,pid=child.pid,native_exit_code=code,wait_reap_completed=True))
        write(out/'worker_native_exits.json',witnesses)
        if any(r['native_exit_code']!=0 for r in witnesses):raise ValueError('NUMERIC_WORKER_FAILURE')
    finally:
        for _,child,stdout,stderr in children:
            if child.poll() is None:
                try:child.terminate()
                except ProcessLookupError:pass
                child.wait()
            stdout.close();stderr.close()
    rows=[r for system in CASES for r in read(out/system/'theory.json')];accept=[]
    for expected in old('model_native_difference_audit.json')['native_mixed_raw_disagreements']:
        actual=next(r for r in rows if r['system']==expected['system'] and r['model']=='NATIVE_MIXED' and r['axis']==expected['axis'])['lower'][expected['metric'].lower()]
        accept.append(dict(system=expected['system'],candidate=expected['candidate_id'],axis=expected['axis'],metric=expected['metric'],expected=expected['native_explicit_lowered'],actual=actual,exact_integer_match=int(actual)==int(expected['native_explicit_lowered'])))
    write(out/'acceptance.json',accept);write(out/'corrected_theory.json',rows)
    if len(accept)!=8 or not all(r['exact_integer_match'] for r in accept):raise ValueError('EIGHT_INTEGER_ACCEPTANCE_FAILURE')


def direction(lo,hi):
    lo,hi=F(lo),F(hi)
    return 'PR_LOWER' if hi<1 else 'PF_LOWER' if lo>1 else 'TIE' if lo==hi==1 else 'INTERVAL_CONTAINS_ONE'


def stage3(plan,out):
    reserve(plan,out,'reaccount')
    x=load('_compat_summary_execution',ROOT/'scripts/h46_cost_execution.py')
    summary=load('_compat_summary',ROOT/'scripts/h46_cost_summary.py')
    corrected=read(plan['inputs']['corrected_theory']['path'])
    originals=old('stage2/theory.json')+old('stage3/theory.json')+old('stage3/uniform_saved_reuse.json')
    newmap={(r['system'],r['candidate_id'],r['model'],r['axis']):r for r in corrected}
    theory=[newmap.get((r['system'],r['candidate_id'],r['model'],r['axis']),r) for r in originals]
    if len(theory)!=42:raise ValueError('FIXED_THEORY_INVENTORY')
    base=out/'base_tables';base.mkdir();write(out/'theory_rows.json',theory)
    jobs=[]
    for system,ids in x.PANEL.items():
        for cid in ids:
            jobs.append(dict(system=system,candidate=dict(id=cid)))
            own=base/(system+'_'+cid);own.mkdir()
            prefix=f'cohorts/{system}_{cid}/'
            pairs=old(prefix+'paired_theory.json')
            if (system,cid) in {(s,c[0]) for s,c in CASES.items()}:
                pairs=[dict(model=r['model'],axis=r['axis'],cost=r['lower']) for r in corrected if r['system']==system]
            write(own/'paired_theory.json',pairs)
            for axis in ('cosine','sine'):write(own/f'wrapper_{axis}.json',old(prefix+f'wrapper_{axis}.json'))
    empty=out/'empty_rows.json';write(empty,[])
    summary_plan=dict(jobs=jobs,inputs={
        'h4_theory':dict(path=str(out/'theory_rows.json')),'h6_mixed':dict(path=str(empty)),
        'h6_uniform':dict(path=str(empty)),'saved_accounting':plan['inputs']['saved_accounting']})
    counts=summary.summarize(x,summary_plan,base)
    if counts['expectation_comparison_rows']!=84 or counts['same_event_comparison_rows']!=84 or counts['pf_pr_comparison_rows']!=360:
        raise ValueError('FIXED_OUTPUT_ROWS')
    if read(base/'preserved_shots.json')!=old('preserved_shots.json'):raise ValueError('SHOT_RULE_CHANGED')
    write(out/'preserved_shots.json',read(base/'preserved_shots.json'))
    provenance=old('row_provenance_index.json')['candidate_receipts'];differences=[];direction_rows=[]
    def origin(system,cid,model):
        if cid.endswith('_4th'):return plan['stage2_SOURCE']
        receipt=provenance[system+':'+cid]
        return receipt['uniform_H6_theory_SOURCE'] if system=='H6' and model in ('A','B') else receipt['theory_expected_SOURCE']
    for name,fields in (('expectation_vs_saved',('theory_lower','theory_upper')),('same_event_comparisons',('theory_same_event',)),('pf_pr_comparisons',('theory_PR_PF_lower','theory_PR_PF_upper'))):
        before=old(name+'.json');after=read(base/(name+'.json'))
        if len(before)!=len(after):raise ValueError('ROW_INVENTORY_CHANGED')
        for i,(b,r) in enumerate(zip(before,after,strict=True)):
            # The original helper preserves deterministic row order.
            identity=('system','model','axis','metric','candidate_id') if name!='pf_pr_comparisons' else ('system','model','metric','PR','PF','epsilon','factor','status')
            if any(b.get(k)!=r.get(k) for k in identity):raise ValueError('ROW_IDENTITY_CHANGED')
            if name=='expectation_vs_saved':
                for key in ('compiled_saved_mean','sample_size','sample_phase','alignment','estimand'):
                    if b[key]!=r[key]:raise ValueError('SAVED_EXPECTATION_METADATA_CHANGED')
            elif name=='same_event_comparisons':
                for key in ('native_explicit_lowered','compiled_same_event','event_digest','fingerprint','compiler_hash','preparation_hash'):
                    if b[key]!=r[key]:raise ValueError('NATIVE_COMPILED_IDENTITY_CHANGED')
            elif b.get('compiled_PR_PF')!=r.get('compiled_PR_PF'):raise ValueError('COMPILED_RATIO_CHANGED')
            changed=any(b.get(k)!=r.get(k) for k in fields)
            if name!='pf_pr_comparisons' and not r['candidate_id'].endswith('_4th') and changed:
                raise ValueError('S2_PR_REUSE_NOT_INVARIANT')
            record=dict(file=name+'.json',index=i,system=r['system'],model=r['model'],changed=changed,
                        old={k:b.get(k) for k in fields},new={k:r.get(k) for k in fields})
            record.update({k:r[k] for k in ('candidate_id','axis','metric','PR','PF','epsilon','factor') if k in r})
            differences.append(record)
            r['old_theory']={k:b.get(k) for k in fields}
            r['theory_delta']={k:str(F(r[k])-F(b[k])) for k in fields if k in r}
            r['SOURCE_accounting']=plan['SOURCE']
            r['SOURCE_counter_evaluation']=({c:origin(r['system'],c,r['model']) for c in (r['PR'],r['PF'])} if name=='pf_pr_comparisons' else origin(r['system'],r['candidate_id'],r['model']))
            r['compiled_values_reused']=True
            if name!='pf_pr_comparisons':r['input_and_compiled_provenance']=provenance[r['system']+':'+r['candidate_id']]
            else:r['input_and_compiled_provenance']={c:provenance[r['system']+':'+c] for c in (r['PR'],r['PF'])}
            r['new_counter_schedule_profile']='native_binary64_s4_v1'
            if name=='same_event_comparisons':r['counter_native_count_match']=F(r['theory_same_event'])==F(r['native_explicit_lowered'])
            if name=='pf_pr_comparisons' and 'compiled_PR_PF' in r:
                newdir=direction(r['theory_PR_PF_lower'],r['theory_PR_PF_upper']);olddir=direction(b['theory_PR_PF_lower'],b['theory_PR_PF_upper']);compdir=direction(r['compiled_PR_PF'],r['compiled_PR_PF'])
                direction_rows.append(dict(index=i,system=r['system'],PR=r['PR'],PF=r['PF'],epsilon=r['epsilon'],factor=r['factor'],model=r['model'],metric=r['metric'],old_direction=olddir,new_direction=newdir,compiled_direction=compdir,direction_changed=olddir!=newdir,theory_vs_compiled_disagree=newdir!=compdir,old_ratio=b['theory_PR_PF_lower'],new_ratio=r['theory_PR_PF_lower'],compiled_ratio=r['compiled_PR_PF']))
        write(out/(name+'.json'),after)
        with (out/(name+'.csv')).open('x',newline='') as stream:
            columns=list(dict.fromkeys(k for r in after for k in r));writer=csv.DictWriter(stream,columns);writer.writeheader()
            writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in after)
    write(out/'old_new_differences.json',differences);write(out/'direction_audit.json',direction_rows)
    write(out/'shot_weighted_totals.json',read(base/'shot_weighted_totals.json'))
    write(out/'result_summary.json',dict(**counts,changed_rows={n:sum(r['changed'] for r in differences if r['file']==n+'.json') for n in ('expectation_vs_saved','same_event_comparisons','pf_pr_comparisons')},direction_changes=sum(r['direction_changed'] for r in direction_rows),theory_compiled_direction_disagreements=sum(r['theory_vs_compiled_disagree'] for r in direction_rows),new_compile=0,new_trajectory=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--grant',type=Path,required=True);parser.add_argument('--worker',choices=tuple(CASES))
    args=parser.parse_args();plan,grant=read(args.plan),read(args.grant);out=verify(plan,grant)
    if args.worker:
        if plan['stage']!=2 or not (out/'one_shot_lock.json').exists():raise ValueError('WORKER_SCOPE')
        print(json.dumps(case_worker(plan,grant,out,args.worker)),flush=True);return
    out.mkdir();write(out/'one_shot_lock.json',dict(SOURCE=plan['SOURCE'],stage=plan['stage'],grant_sha256=file_sha(args.grant),consumed=True))
    start=time.monotonic();terminal=dict(SOURCE=plan['SOURCE'],stage=plan['stage'],status='MANDATORY_STOP_FAILURE',first_stop=None,mandatory_stop=True,new_compile=0,new_trajectory=0)
    try:
        if plan['stage']==1:stage1(plan,out)
        elif plan['stage']==2:
            # CLI-only paths are not part of the canonical bound plan.
            worker_plan=dict(plan,_plan_path=str(args.plan),_grant_path=str(args.grant));stage2(worker_plan,grant,out)
        else:stage3(plan,out)
        verify(plan,grant)
        if any(k=='qiskit' or k.startswith(('qiskit.','cupy')) for k in sys.modules):raise ValueError('FORBIDDEN_IMPORT')
        terminal.update(status='COMPLETE_MANDATORY_STOP',final_source_input_environment_gate=True)
    except Exception as exc:
        terminal['first_stop']=repr(exc);raise
    finally:
        terminal.update(wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(out/'terminal.json',terminal);print(json.dumps(terminal),flush=True)


if __name__=='__main__':main()
