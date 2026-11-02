"""Collect new unit IDs and independent-logic raw mutations (same author)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy,importlib,json,os,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
sys.dont_write_bytecode=True

def main():
    label=sys.argv[1]
    dest=HERE/('runs_checks_'+label);dest.mkdir(exist_ok=False)
    from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S,offline as O
    module=importlib.import_module('test_g6_w2c_b3_api_binding_v1')
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    def ids(s):
        for t in s:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    collected=list(ids(suite))
    with (dest/'unit.log').open('w') as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    unit=dict(collected=collected,run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
              FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),
              failures=[dict(id=t.id(),trace=m)for t,m in r.failures+r.errors])
    (dest/'unit.json').write_text(json.dumps(unit,indent=2)+'\n')
    print('New unit tests',unit['PASS'],unit['FAIL'],unit['ERROR'],flush=True)
    assert r.wasSuccessful()
    observations=[]
    for case in ('sealed_small','sealed_n256','S_generator','P1_angle'):
        d=HERE/('runs_'+case);root=Path(json.loads((d/'location.json').read_text())['root'])
        c=json.loads((d/'trusted_candidate.json').read_text())
        v=O.verify(root,c,ROOT);observations.append(dict(case=case,root=str(root),**v))
        print(case,v['status'],v.get('calls'),v['issues'],flush=True)
    assert observations[0]['status']==observations[1]['status']=='VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE'
    assert all(x['status']=='VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE'for x in observations[2:])
    (dest/'port_cases.json').write_text(json.dumps(observations,indent=2)+'\n')
    base=Path(observations[0]['root']);c=json.loads((HERE/'runs_sealed_small/trusted_candidate.json').read_text())
    variants=[]
    def one(name,change):
        target=dest/'mutations'/name/'case';shutil.copytree(base,target)
        trusted=copy.deepcopy(c)
        change(target,trusted)
        result=O.verify(target,trusted,ROOT,execution_root=base)
        assert result['status']!='VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE',(name,result)
        meta=dict(name=name,base_case='sealed_small',original_execution_root=str(base),**result)
        (target.parent/'trusted_candidate.json').write_text(json.dumps(trusted,indent=2)+'\n')
        (target.parent/'observation.json').write_text(json.dumps(meta,indent=2)+'\n')
        return meta
    def mutate_report(path,key,value):
        p=path/'port_report.json';x=O.read(p);x[key]=value;p.write_bytes(S.encoded(x)+b'\n')
        q=path/'port_caller.json';caller=O.read(q);caller['report_sha256']=S.sha(p.read_bytes());q.write_bytes(S.encoded(caller)+b'\n')
    def mutate_port(path,key,value):
        p=path/'ports'/'S__correctness_baseline_ordinary_A.port.json'
        x=O.read(p);x['records'][0][key]=value;p.write_bytes(S.encoded(x)+b'\n')
    def mutate_trace(path,kind):
        p=path/'ports'/'S__correctness_baseline_ordinary_A.port.json';x=O.read(p);t=x['records'][0]['consumption']
        if kind=='value':t['draws'][0]['value_hex']='0x1'
        elif kind=='count':t['draws'].pop()
        elif kind=='domain':t['domain']='REAL/ordinary/X'
        elif kind=='schema':t['schema']='unsupported'
        elif kind=='bool':t['draws'][0]['counter_before']=False
        p.write_bytes(S.encoded(x)+b'\n')
    changes={
        'global_mode_collision':lambda p,c:mutate_port(p,'global_trial_id','collision'),
        'event_identity':lambda p,c:mutate_port(p,'event_digest','0'*64),
        'arm_axis_domain':lambda p,c:mutate_port(p,'domain','ARTIFICIAL_API_BINDING_V1/ordinary/Y'),
        'consumption_value':lambda p,c:mutate_trace(p,'value'),
        'consumption_count':lambda p,c:mutate_trace(p,'count'),
        'consumption_integer_boolean':lambda p,c:mutate_trace(p,'bool'),
        'consumption_domain':lambda p,c:mutate_trace(p,'domain'),
        'trace_schema':lambda p,c:mutate_trace(p,'schema'),
        'candidate_digest':lambda p,c:mutate_report(p,'candidate_digest','0'*64),
        'unsupported_report':lambda p,c:mutate_report(p,'schema','unsupported'),
        'planned_calls':lambda p,c:mutate_report(p,'planned',16897),
        'false_failure_flag':lambda p,c:mutate_report(p,'complete',False),
        'science_GO_flip':lambda p,c:mutate_report(p,'scientific_GO',True),
        'source_pin':lambda p,c:c['source_pins'].__setitem__(next(iter(c['source_pins'])),'0'*64),
        'runtime_pin':lambda p,c:c['python'].__setitem__('executable_sha256','0'*64),
        'missing_caller':lambda p,c:(p/'port_caller.json').unlink(),
        'missing_report':lambda p,c:(p/'port_report.json').unlink(),
        'missing_pair':lambda p,c:(p/'ports'/'S__correctness_baseline_ordinary_A.jsonl').unlink(),
        'partial_pair':lambda p,c:(p/'ports'/'S__correctness_baseline_ordinary_A.jsonl').write_bytes(b'partial'),
        'missing_authority':lambda p,c:(p/'fake_snapshot.json').unlink(),
        'fake_authority_splice':lambda p,c:(p/'authority_binding.json').write_bytes(b'{}'),
        'caller_false_close':lambda p,c:(p/'port_caller.json').write_bytes(S.encoded(dict(O.read(p/'port_caller.json'),report_write_flush_close_observed=False))),
    }
    with ThreadPoolExecutor(max_workers=8) as pool:variants=list(pool.map(lambda item:one(*item),changes.items()))
    (dest/'mutations.json').write_text(json.dumps(variants,indent=2)+'\n')
    print('New offline mutations',len(variants),'rejected',flush=True)
if __name__=='__main__':main()
