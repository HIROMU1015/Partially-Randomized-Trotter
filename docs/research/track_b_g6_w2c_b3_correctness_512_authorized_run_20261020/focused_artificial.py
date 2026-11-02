"""Differential artificial tests only: no real approval, resolver, core or seed."""
from pathlib import Path
import sys,os,json,tempfile,unittest,tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];PACKET=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import contract as K,integration as I,offline as O,run_control as A,guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1.mutations import run as mutations

def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
    dest=PACKET/'focused_artificial';dest.mkdir(exist_ok=False)
    os.environ['B3_FINAL_CONTROL_RAW']=str(dest/'control_raw')
    selected=['test_g6_w2c_b3_final_live_v1.Preflight','test_g6_w2c_b3_final_live_v1.Connection',
        'test_g6_w2c_b3_final_live_v1.SetupDocuments','test_g6_w2c_b3_final_live_v1.SimpleState']
    suite=unittest.defaultTestLoader.loadTestsFromNames(selected)
    with (dest/'tests.log').open('w') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    tests=dict(run=r.testsRun,failures=[dict(id=t.id(),trace=v) for t,v in r.failures+r.errors],skips=len(r.skipped))
    save(dest/'tests.json',tests)
    if not r.wasSuccessful() or r.skipped:sys.exit(1)
    c=K.candidate(ROOT);cases=[]
    for scenario,want in [('normal','ARTIFICIAL_TECHNICAL_COMPLETE'),('P1_angle','TECHNICAL_STOP'),('receipt_close','INCOMPLETE_EVIDENCE')]:
        root=Path(tempfile.mkdtemp(prefix='b3-release-test-authorized-diff-',dir=ROOT.parent))
        m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/'+root.name,root/'output',scenario=scenario)
        save(root/'candidate_input.json',c);save(root/'manifest_input.json',m)
        result=I.run(c,m,root,A.SimpleStore(root));check=O.inspect(root,c,m)
        assert check['status']==want,(scenario,check)
        cases.append(dict(scenario=scenario,root=str(root),verification=check))
        if scenario=='normal':mutation=mutations(root,c,m)
        print(json.dumps(dict(scenario=scenario,status=check['status'])),flush=True)
    guard=G.observation();assert not guard['core_loaded'] and not G.ATTEMPTS
    assert mutation['FAIL']==0
    save(dest/'result.json',dict(tests=tests,cases=cases,mutation=mutation,guard=guard,scientific_operations=0,separate_reviewer=False))
    with tarfile.open(dest/'raw_artificial.tar.gz','w:gz') as archive:
        for case in cases:archive.add(case['root'],arcname=case['scenario'])
        if (dest/'control_raw').exists():archive.add(dest/'control_raw',arcname='control_raw')
