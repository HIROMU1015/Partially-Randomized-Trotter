"""Frozen SOURCE artificial acceptance; private scopes only; mandatory STOP."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1 import guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1 import contract as K,integration as I,offline as O,authority as A
import unittest,json,tempfile,subprocess,tarfile,hashlib,os


def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def suite(dest,name,selections):
    selected=unittest.defaultTestLoader.loadTestsFromNames(selections)
    with (dest/(name+'.log')).open('w') as stream:r=unittest.TextTestRunner(stream=stream,verbosity=2).run(selected)
    result=dict(run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
        FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),
        failures=[dict(id=t.id(),trace=v) for t,v in r.failures+r.errors])
    save(dest/(name+'.json'),result);return result


if __name__=='__main__':
    dest=Path(sys.argv[1]);dest.mkdir(exist_ok=False)
    os.environ['B3_RELEASE_AUTHORITY_RAW']=str(dest/'authority_raw')
    categories=dict(unit=['test_g6_w2c_b3_live_release_v1.Unit'],
        preflight=['test_g6_w2c_b3_live_release_v1.Preflight'],
        store=['test_g6_w2c_b3_live_release_v1.Store'],
        regression=['test_g6_w2c_b3_parallel_f01','test_g6_w2c_b3_storage_v2.Storage'])
    if '--smoke' in sys.argv:categories={}
    results={n:suite(dest,n,selected) for n,selected in categories.items()}
    save(dest/'tests.json',results)
    if any(x['FAIL']+x['ERROR']+x['SKIP'] for x in results.values()):print(json.dumps(results));sys.exit(1)
    c=K.candidate(ROOT);save(dest/'candidate.json',c)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if '--development' not in sys.argv:save(dest/'git_source_check.json',K.verify_source_commit(c,source))
    cases=[]
    names=['normal'] if '--smoke' in sys.argv else ['normal','coverage','S_angle','P1_angle','scientific_mismatch','receipt_close','byte_cap','call_cap','worker_exit','unknown_ack']
    for name in names:
        root=Path(tempfile.mkdtemp(prefix='b3-release-test-bundle-'))
        m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/'+root.name,root/'output',scenario=name,
            result_bytes=6<<20 if name=='byte_cap' else 1<<30)
        save(root/'candidate_input.json',c);save(root/'manifest_input.json',m)
        report=I.run(c,m,root,A.TestStore(root))
        try:verified=O.inspect(root,c,m)
        except Exception as exc:
            import traceback
            verified=dict(status='CHECKER_ERROR',reason=repr(exc),trace=traceback.format_exc())
        save(root/'verification.json',verified)
        cases.append(dict(case=name,root=str(root),reported_classification=report['classification'],
                          complete=report['complete'],verification=verified))
        save(dest/'cases.json',cases)
        print(json.dumps(dict(case=name,status=verified['status'],complete=report['complete'])),flush=True)
    save(dest/'cases.json',cases)
    # Mutate saved payloads including original codec seals; no science runtime.
    master=Path(cases[0]['root']);normal_m=json.loads((master/'manifest_input.json').read_text())
    from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1.mutations import run as mutations
    mutations_result=mutations(master,c,normal_m);save(dest/'mutation.json',mutations_result)
    with tarfile.open(dest/'raw_artificial.tar.gz','w:gz') as archive:
        for row in cases:archive.add(row['root'],arcname=row['case'])
        if (dest/'authority_raw').exists():archive.add(dest/'authority_raw',arcname='authority_raw')
    wanted=['ARTIFICIAL_TECHNICAL_COMPLETE','INCONCLUSIVE_COVERAGE','TECHNICAL_STOP','TECHNICAL_STOP',
            'ARTIFICIAL_SCIENTIFIC_MISMATCH_STOP','INCOMPLETE_EVIDENCE','TECHNICAL_STOP','TECHNICAL_STOP',
            'TECHNICAL_STOP','TECHNICAL_STOP']
    okay=all(row['verification']['status']==want for row,want in zip(cases,wanted))
    result=dict(schema='b3_release_preparation_acceptance_v1',measured_SOURCE=source,tests=results,
        mutation=mutations_result,integration_cases=len(cases),integration_PASS=sum(row['verification']['status']==want
            for row,want in zip(cases,wanted)),guard=G.observation(),scientific_operations=0,
        separate_reviewer=False,mandatory_STOP=True,live_compatibility='NOT_ESTABLISHED')
    save(dest/'result.json',result)
    sys.exit(not okay or mutations_result['FAIL']>0 or bool(G.observation()['core_loaded']) or bool(G.ATTEMPTS))
