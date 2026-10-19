"""Fresh artificial cases, parallel processes; each bundle keeps S -> P1."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True


def specs():
    cases=[dict(name='normal_n16'),dict(name='normal_n256',manifest={'profile_n':256})]
    for stage in ('generator','codec','evidence','bits','audit','angle','requery','worker_exit'):
        cases.append(dict(name='S_'+stage,manifest=dict(port_faults=[dict(mode='S',pair=1,stage=stage)])))
    cases.append(dict(name='P1_failure_preserves_S',manifest=dict(port_faults=[dict(mode='P1',pair=1,stage='generator')])))
    for name,point in [('trace_close','trace.close'),('trace_partial','trace.write'),('state_close','state.close'),('receipt_close','receipt.json.close'),
                       ('marker_close','marker.json.close'),('delivery_close','delivery.json.close')]:
        cases.append(dict(name=name,fault=(dict(point=point,match={'op':'COMPLETE'},action='partial') if name=='trace_partial' else dict(point=point))))
    cases.extend([dict(name='state_write',fault=dict(point='state.write',match={'state':'IN_FLIGHT'})),
                  dict(name='worker_write',manifest=dict(condition_faults={'S':[[0,0,'write',0]]})),
                  dict(name='worker_flush',manifest=dict(condition_faults={'S':[[0,0,'flush',0]]})),
                  dict(name='call_cap',manifest=dict(call_limit=1)),
                  dict(name='missing_ACK',manifest=dict(condition_faults={'S':[[0,0,'ack_exit',0]]}))])
    return cases


def one(spec,dest):
    name=spec['name'];sp=dest/(name+'_spec.json');sp.write_text(json.dumps(spec))
    record=dest/(name+'_record.json')
    p=subprocess.run([sys.executable,str(HERE/'run_case.py'),'--spec',str(sp),'--record',str(record)],
                     cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    (dest/(name+'.log')).write_text(p.stdout+p.stderr)
    if not record.exists():return dict(case=name,process_exit=p.returncode,error=p.stderr)
    r=json.loads(record.read_text())
    # Consumer in a separate process has no producer modules loaded.
    verify=dest/(name+'_verification.json')
    script="from pathlib import Path;import json,sys;from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.offline import verify;root=Path(sys.argv[1]);trusted=json.loads((root/'candidate_input.json').read_text());result=verify(root,trusted);Path(sys.argv[2]).write_text(json.dumps(result,indent=2,default=str)+'\\n')"
    v=subprocess.run([sys.executable,'-c',script,r['root'],str(verify)],cwd=ROOT,
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    if v.returncode:r['consumer_error']=v.stderr
    else:
        result=json.loads(verify.read_text());r.update(verification_status=result['status'],
            verification_issues=result['issues'],adapter_dispatch=result.get('adapter_dispatch'))
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--new-output',required=True);p.add_argument('--jobs',type=int,required=True)
    p.add_argument('--only',nargs='*');a=p.parse_args();dest=Path(a.new_output);dest.mkdir(exist_ok=False)
    selected=[x for x in specs() if not a.only or x['name'] in a.only]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:rows=list(pool.map(lambda x:one(x,dest),selected))
    (dest/'matrix.json').write_text(json.dumps(dict(cases=rows,separate_reviewer=False),indent=2)+'\n')
    for r in rows:print(r['case'],r.get('verification_status'),r.get('verification_issues'),r.get('fault_hits'),r.get('consumer_error',''))
