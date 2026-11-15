"""One adopted coarse-Y4 run: <=6 saved-step means, one packet, <=1 Q3.

Signal and Q3 runtimes run in separate processes. Reuse prior A numerical
helpers and a code-only, pinned B snapshot; no new estimator or price code.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
CALLER='scripts/run_p7_y4_coarse_signal_q3.py'
STUDY='docs/research/d_p7_y4_coarse_signal_q3_20261116'
OUT=ROOT/STUDY
PLAN_SOURCE='d01c780d035eb096186ef1b8aa6d5e580bca2a50'
PLAN_DIR='docs/research/d_pf_first_coarser_width_plan_20261116/'
A='06a15ee45beb177c018f6260ea4a5ac8cf66cfe6'
B='b4296b110b129d226dbef2743c14f725a27c3bf6'
B_DIR='docs/research/pf6_cost_budget_q3_results_20261115/'
CID='dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0'
CURRENT='defc5e88ff4da2e43c99ecdf7157aa41cc47cd32f905f5aa3a1266aa3ecf5416'
LABEL='B0I_P7_Y4_32h_over_5'
VECTOR=[59,55,51,47,43,39,35,31,27,23,19,7,2]
RUNTIME=ROOT/'.q3_runtime'/B


def utc():return datetime.now(timezone.utc).isoformat()
def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def read(ref):
    value=json.loads(blob(ref['commit'],ref['path']))
    for key in ref.get('selector',[]):value=value[key]
    return value
def ref(commit,path,*selector):return dict(commit=commit,path=path,selector=list(selector))
def require(ok,why):
    if not ok:raise ValueError(why)
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False,default=str);f.write('\n')
def source(input_path):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (CALLER,input_path):require((ROOT/path).read_bytes()==blob(sha,path),'Uncommitted SOURCE: '+path)
    return dict(commit=sha,path=input_path,caller=CALLER)
def event(out,name,**values):
    with (Path(out)/'STAGES.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage=name,UTC=utc(),**values),default=str)+'\n')
    print(name,values,flush=True)


def signal_modules():
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    from trottertracks import pf_shortlist_mean_completion as F
    import run_truncated_fullh_signals as TR
    return F,TR


def signal_plan(provenance):
    F,TR=signal_modules();S,I,C=F.S,F.I,F.C
    frozen=read(ref(PLAN_SOURCE,PLAN_DIR+'FIRST_FINITE_PLAN.json'))
    old=read(ref(A,'docs/research/pf_shortlist_mean_completion_20261115/run01/B0I_P7_Y4_16h_over_3/PLAN.json','configuration'))
    require(old['id']==CURRENT,'Current Y4 SOURCE differs')
    spec=deepcopy(old['spec']);spec['delta_pf']=S.enc(32*S.H/5)
    api=I.providers(str(ROOT));cfg=C.configure(ROOT,spec,api=api)
    require(cfg['id']==CID,'Regenerated coarse canonical ID differs')
    manifest=S.git_json(ROOT,S.MANIFEST)
    prepared=I.plan(ROOT,cfg,target=manifest['shared']['target'],signal_source=provenance,
        references=S.references(ROOT,cfg,manifest['shared']['binding']))
    schedule=next(c for c in read(ref(PLAN_SOURCE,PLAN_DIR+'CANDIDATES.json'))['candidates'] if c['configuration_id']==CID)
    cand=dict(label=LABEL,method='B0I',formula='4th',width=32*S.H/5)
    old_packet=read(ref(A,'docs/research/pf_shortlist_mean_completion_20261115/run01/B0I_P7_Y4_16h_over_3/B_SIGNAL_INPUT.json'))
    physics=S.git_json(ROOT,S.PHYSICS_REF)
    saved=[];origins=[];new=[];steps={}
    for req,row in zip(prepared['pending']['requests'],schedule['physical_schedule'],strict=True):
        m=req['m'];c=req['physical_candidate']
        require((c['id'],c['T'],c['q'])==(row['physical_id'],row['T'],row['q']),'Physical schedule differs')
        require(req['structure']['signed_weights']==[1.3512071919596578,-1.7024143839193153,1.3512071919596578],'Signed Y4 differs')
        if m in [0,1,2,3,4,5,12]:
            if m<=4:
                origin=ref(A,'docs/research/pf_shortlist_mean_completion_20261115/run01/B0I_P7_Y4_16h_over_3/round_'+str(m)+'.json')
                record=read(origin)
                require(C.p.same(record['domain'],req['domain']) and record['status']=='NUMERICAL_MEAN_AVAILABLE',
                        'Saved early physical/reference domain mismatch')
                rebound=C.b_mean(req,record)
                for key in ('raw','corrected','reference','normalization'):
                    require(rebound[key]==old_packet['means'][m][key] if key!='normalization' else
                        all(rebound[key][k]==old_packet['means'][m][key][k] for k in ('B','log_B','attenuation','kind')),
                        'Saved early value/normalization changed')
            else:
                origin=frozen['saved_mean_sources'][str(m)]['current'][0]
                hit=dict(source=origin['commit']+':'+origin['path'],note='T2_TWO_TIME_DIRECT_MEAN')
                record=S.stage_record(ROOT,cand,None,req,hit,physics['rounds'][m]['diagnostics']['reference_pair'],provenance)
            saved.append(record);origins.append(dict(m=m,physical_id=c['id'],source=origin))
        else:
            require(m in range(6,12),'Unapproved new round')
            desc=F.step_descriptor(ROOT,cand,req,{})
            steps[desc['path']]=desc
            new.append(dict(m=m,physical_id=c['id'],configuration_id=CID,T=c['T'],q=c['q'],
                original_status='NEW_MEAN_NEEDED',step_source=desc))
    require(len(saved)==7 and len(new)==6 and len(steps)==1,'Fixed 7/6/1 menu differs')
    return prepared,dict(configuration=cfg,new=new,saved_sources=origins,steps=list(steps.values()),shared=manifest['shared']),saved


def signal_prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'SIGNAL_INPUTS.json').exists() and not (OUT/'signal_run01').exists(),'Preserve prior fixed input/run')
    prepared,inputs,saved=signal_plan(dict(caller=CALLER,phase='METADATA_ONLY'))
    inputs.update(approved_plan=ref(PLAN_SOURCE,PLAN_DIR+'FIRST_FINITE_PLAN.json'),
        limits=dict(new_complex_means=6,new_packet=1,new_Q3=1,new_reference=0,new_step_build=0,
                    circuit_build=0,compile=0,price=0,fit=0,new_allocation=0,retry=0),
        local_reused_sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['src/trottertracks/pf_shortlist_mean_completion.py','src/trottertracks/saved_pf_mean_staging.py',
             'src/trottertracks/baseline_signal_pipeline.py','src/trottertracks/baseline_signal_numerics.py',
             'src/trottertracks/rpe_optimization_pipeline.py','scripts/run_truncated_fullh_signals.py']})
    write(OUT/'SIGNAL_INPUTS.json',inputs)
    print('METADATA PASS',CID,'saved=7 new=6 steps=1; no numerical calls',flush=True)


def signal_execute():
    src=source(STUDY+'/SIGNAL_INPUTS.json');frozen=json.loads((OUT/'SIGNAL_INPUTS.json').read_text())
    out=OUT/'signal_run01';out.mkdir()
    started=time.perf_counter();calls=[];result=dict(status='STARTED',source=src,new_means=0,saved_means=7,
        new_reference=0,new_step_build=0,compile=0,price=0,fit=0,Q3_calls=0,retry=0,mandatory_stop=True)
    write(out/'EXECUTION.json',result)
    try:
        for p,sha in frozen['local_reused_sources'].items():
            require(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,'Used SOURCE changed: '+p)
        prepared,current,saved=signal_plan(src)
        require(all(current[k]==frozen[k] for k in current),'Frozen scientific inputs changed')
        F,TR=signal_modules();event(out,'INPUT_LOAD_START')
        inputs=F.load_inputs(ROOT,frozen['shared']['binding']);steps=F.load_steps(ROOT,frozen['steps'])
        F.N.kernels(str(ROOT));event(out,'INPUT_LOAD_END',wall_seconds=time.perf_counter()-started)
        ctx=F.SavedStepContext(ROOT,prepared,inputs,frozen['new'],steps)
        reader=TR.reference_reader(ROOT,frozen['shared'],F.I.providers(str(ROOT)))
        original=F.N.evaluate
        def observed(context,request,reference,*,source):
            m=request['m'];c=request['physical_candidate']
            require(m in range(6,12) and m not in calls and c['id']==frozen['new'][m-6]['physical_id'],'Duplicate/out-of-scope mean')
            calls.append(m);t=time.perf_counter();event(out,'MEAN_START',m=m,q=c['q'])
            value=original(context,request,reference,source=source)
            value['input_source']=dict(saved_step=frozen['new'][m-6]['step_source'],saved_input=inputs['source'],
                scalar_rule=F.S.SCALAR_RULE['B0I'],scalar_values=dict(H_constant=ctx.constant,tail_identity=ctx.tail))
            event(out,'MEAN_END',m=m,status=value['status'],seconds=time.perf_counter()-t)
            return value
        with patch.object(F.N,'evaluate',observed):
            packet=F.I.produce_signals(ROOT,prepared,out/LABEL,context_factory=lambda:ctx,reference_provider=reader,saved=saved)
        producer=packet['producer_result']
        require(producer['status']=='BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED','Signal numerical/input gate failed')
        require(calls==list(range(6,12)) and producer['new_mean_evaluations']==6 and producer['saved_copies']==7,'Mean budget mismatch')
        F.S.require_complete(packet)
        for mean in packet['means']:F.I.provider_binding_check(prepared,mean)
        write(out/'B_SIGNAL_INPUT.json',F.I.signal_input(packet))
        result.update(status='THIRTEEN_ROUND_SIGNAL_PACKET_COMPLETE',new_means=len(calls),scalar_values=dict(H_constant=ctx.constant,tail_identity=ctx.tail))
    except Exception as exc:
        result.update(status='UNKNOWN_STOP_NO_RETRY',reason=str(exc),error_type=type(exc).__name__,new_means=len(calls))
        raise
    finally:
        result.update(wall_seconds=time.perf_counter()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_UTC=utc())
        write(out/'SIGNAL_STATUS.json',result)
        event(out,'SIGNAL_PHASE_END',status=result['status'],new_means=len(calls))


def q3_runtime():
    # A separate process imports only this pinned B code tree. Its Git readers
    # can resolve the new SOURCE in the parent worktree; no B worktree is used.
    spec=importlib.util.spec_from_file_location('_coarse_reused_pf_budget',RUNTIME/'scripts/run_pf6_cost_budget_q3_fixed.py')
    F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
    F.check_source()
    return F


def q3_prepare():
    F=q3_runtime();signal_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    signal_state=read(ref(signal_sha,STUDY+'/signal_run01/SIGNAL_STATUS.json'))
    require(signal_state['status']=='THIRTEEN_ROUND_SIGNAL_PACKET_COMPLETE','No complete fixed packet')
    cfg_source=ref(signal_state['source']['commit'],STUDY+'/SIGNAL_INPUTS.json','configuration')
    packet_source=ref(signal_sha,STUDY+'/signal_run01/B_SIGNAL_INPUT.json')
    cfg=read(cfg_source);packet=read(packet_source)
    binding=F.T.create(F.ROOT,cfg,original_source=cfg_source)
    prior=json.loads(blob(B,B_DIR+'INPUTS.json'))
    old=next(c for c in prior['cases'] if c['label']=='B0I_P7_Y4_16h_over_3__B_PRICE')
    require([r['N_X'] for r in old['allocation']['shots']]==VECTOR and [r['N_Y'] for r in old['allocation']['shots']]==VECTOR,'Fixed vector differs')
    case=dict(label=LABEL,configuration_id=CID,total_view_id=binding['total_configuration']['id'],binding_id=binding['id'],
        configuration=cfg_source,signal_packet=packet_source,
        allocation=dict(allocation_id=LABEL+'_SAME_N_AS_CURRENT_Y4_BPRICE_01',shot_policy='USER_SUPPLIED',shots=deepcopy(old['allocation']['shots'])),
        evaluation_target=deepcopy(old['evaluation_target']))
    F.metadata(case)  # Original canonical binding + deterministic raw intake.
    require(packet['target']['binding']==old['evaluation_target']['binding'],'Evaluation target binding differs')
    inputs={k:deepcopy(prior[k]) for k in ('kernel_source','C_source','statistical_contract','options','adopted_evaluation','acceleration')}
    bounds=F.count_bounds(case['allocation']['shots'])
    require(bounds['maximum_round_outcomes']<=inputs['options']['maximum_round_outcomes'] and
            bounds['verification_pairs_upper']<=inputs['options']['maximum_verification_pairs'] and
            bounds['transition_pairs_upper']<=inputs['options']['total_state_refinement']['maximum_transition_pairs'],'Count-only scope exceeded')
    inputs.update(cases=[case],maximum_scientific_calls=1,actual_new_call_limit=1,concurrent_configurations=1,
        finite_scope_preflight=bounds,conditional_cost=None,G=None,G_qualified=None,sufficient_N=None,winner=None,
        reused_B_runtime=B,saved_current_comparison=ref(B,B_DIR+'SUMMARY.json','rows',15))
    write(OUT/'Q3_INPUTS.json',inputs)
    print('Q3 METADATA PASS; count-only bounds',bounds,flush=True)


def q3_execute():
    src=source(STUDY+'/Q3_INPUTS.json');F=q3_runtime()
    inputs=json.loads((OUT/'Q3_INPUTS.json').read_text());case=inputs['cases'][0]
    require(len(inputs['cases'])==inputs['maximum_scientific_calls']==1 and case['configuration_id']==CID,'Exactly one fixed coarse case')
    require([r['N_X'] for r in case['allocation']['shots']]==VECTOR==[r['N_Y'] for r in case['allocation']['shots']],'Wrong allocation')
    F.metadata(case)
    out=OUT/'q3_run01';out.mkdir();(out/LABEL).mkdir()
    write(out/'EXECUTION.json',dict(source=src,started_UTC=utc(),maximum_scientific_calls=1,price_calls=0,retry=0))
    with F.installed_workers(out):result=F.M.supervise([case],inputs,src,out)
    exits=[c for p in out.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines() for c in json.loads(line)['children']]
    gate=dict(recorded_children=len(exits),all_exit_codes_zero=bool(exits) and all(c['exitcode']==0 for c in exits))
    status='Q3_COMPLETED' if not result['errors'] and not result['monitor_errors'] and gate['all_exit_codes_zero'] and len(result['receipts'])==1 and result['receipts'][0]['status']!='UNKNOWN' else 'UNKNOWN_STOP_NO_RETRY'
    write(out/'COMPLETION.json',dict(status=status,source=src,**result,process_exit_gate=gate,finished_UTC=utc(),mandatory_stop=True,coarse_cost=None))
    print(status,flush=True)


if __name__=='__main__':
    modes={'signal-prepare':signal_prepare,'signal-execute':signal_execute,'q3-prepare':q3_prepare,'q3-execute':q3_execute}
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=modes)
    modes[parser.parse_args().mode]()
