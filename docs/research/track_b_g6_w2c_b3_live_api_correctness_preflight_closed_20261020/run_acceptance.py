"""Reproduce new artificial tests and focused old regressions, in own temp roots."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import contract as K, artificial as A, spec as S
import unittest,json,subprocess,tempfile,tarfile,hashlib


def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def tests(dest,category,names):
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    with (dest/(category+'.log')).open('w') as log:
        r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result=dict(run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
        FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),
        failures=[dict(id=t.id(),trace=trace) for t,trace in r.failures+r.errors])
    save(dest/(category+'.json'),result);return result


if __name__=='__main__':
    dest=Path(sys.argv[1]);dest.mkdir(exist_ok=False)
    groups=dict(unit=['test_g6_w2c_b3_live_preflight_v1.Unit'],
        preflight=['test_g6_w2c_b3_live_preflight_v1.PreflightNegative'],
        mutation=['test_g6_w2c_b3_live_preflight_v1.SemanticMutation'],
        producer=['test_g6_w2c_b3_real_adapter_v1'],
        old_regression=['test_g6_w2c_b3_port_bundle_v1.ContractAndAdmission',
            'test_g6_w2c_b3_parallel_codec.Canonical','test_g6_w2c_b3_parallel_f01',
            'test_g6_w2c_b3_storage_v2.Storage'])
    results={k:tests(dest,k,names) for k,names in groups.items()}
    save(dest/'tests.json',results)
    if any(r['FAIL']+r['ERROR']+r['SKIP'] for r in results.values()):
        print(json.dumps(results));sys.exit(1)
    c=K.candidate(ROOT);save(dest/'candidate.json',c)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(dest/'git_source_check.json',K.verify_source_commit(c,commit))
    cases=[]
    for name,faults in [('normal',[]),('S_angle',[dict(mode='S',pair=1,phase=0,trial=0,stage='angle')]),
                         ('P1_angle',[dict(mode='P1',pair=1,phase=0,trial=0,stage='angle')]),('receipt_close',[])]:
        temp=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-live-preflight-'))
        m=K.manifest(c,'ARTIFICIAL_ONLY/CORRECTNESS/'+temp.name,temp/'output',artificial_faults=faults)
        save(temp/'candidate_input.json',c);save(temp/'manifest_input.json',m)
        storage_fault=A.Fault(dict(point='receipt.json.close')) if name=='receipt_close' else None
        r=A.run(c,m,temp,A.FakeStore(temp),storage_fault=storage_fault)
        try:v=A.inspect(temp,c,m)
        except Exception as exc:v=dict(status='CHECKER_ERROR',reason=repr(exc))
        save(temp/'verification.json',v)
        cases.append(dict(case=name,root=str(temp),complete=r['complete'],verification=v))
        print(json.dumps(dict(case=name,complete=r['complete'],status=v['status'])),flush=True)
    save(dest/'cases.json',cases)
    with tarfile.open(dest/'raw_artificial.tar.gz','w:gz') as archive:
        for row in cases:archive.add(row['root'],arcname=row['case'])
    save(dest/'runtime_and_guard.json',dict(python=c['python'],parent=G.observation(),
        measured_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    okay=cases[0]['verification']['status']=='VERIFIED_ARTIFICIAL_CORRECTNESS_ONLY' and all(
        row['verification']['status']=='TECHNICAL_STOP' and not row['complete'] for row in cases[1:3]) and (
        cases[3]['verification']['status']=='INCOMPLETE_EVIDENCE' and not cases[3]['complete'])
    okay=okay and [x['mode'] for x in cases[1]['verification']['conditions']]==['S'] and (
        cases[2]['verification']['conditions'][0]['complete'] is True)
    save(dest/'result.json',dict(artificial_integration_cases=4,PASS=4 if okay else 0,
        measured_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tests=results,guard=G.observation(),mandatory_STOP=True,scientific_operations=0))
    sys.exit(not okay or bool(G.loaded()) or bool(G.ATTEMPTS))
