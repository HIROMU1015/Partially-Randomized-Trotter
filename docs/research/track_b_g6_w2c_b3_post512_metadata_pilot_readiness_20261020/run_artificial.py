"""Focused artificial metadata regressions, old semantic hashes and closed pilot."""
from pathlib import Path
import sys,os,json,unittest,tempfile,tarfile,hashlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import contract as K,integration as I,offline as O,run_control as A,guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1.mutations import run as mutations
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
PRIOR=ROOT/'docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020'

def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def run_case(arg):
    scenario,c=arg;root=Path(tempfile.mkdtemp(prefix='b3-release-test-post512-',dir=ROOT.parent))
    m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/'+root.name,root/'output',scenario=scenario)
    save(root/'candidate_input.json',c);save(root/'manifest_input.json',m)
    report=I.run(c,m,root,A.SimpleStore(root));check=O.inspect(root,c,m)
    save(root/'verification.json',check)
    return dict(scenario=scenario,root=str(root),classification=report['classification'],verification=check,
        scientific_core_loaded=G.observation()['core_loaded'],denied_core_imports=G.ATTEMPTS)

def compare_normal(case,c):
    root=Path(case['root']);new=O.read(root/'output/final.json');count=0;rows=[]
    with tarfile.open(PRIOR/'focused_artificial/raw_artificial.tar.gz','r:gz') as archive:
        old=json.load(archive.extractfile('normal/output/final.json'));oldc=json.load(archive.extractfile('normal/candidate_input.json'))
        oldpairs={x['task']['key']:x for x in old['pairs']}
        for pair in new['pairs']:
            key=pair['task']['key'];other=oldpairs[key];assert pair['outcome']['trace_sha256']==other['outcome']['trace_sha256']
            assert pair['outcome']['axes']==other['outcome']['axes']
            a=archive.extractfile('normal/output/'+key+'.jsonl').read().splitlines();b=(root/'output'/(key+'.jsonl')).read_bytes().splitlines()
            oldbits=[];newbits=[];originals=[]
            for x,y in zip(a,b,strict=True):
                before=V.decode(json.loads(x)['payload']);after=V.decode(json.loads(y)['payload'])
                assert C.driver_canonical(dict(kind=before['kind'],event=before['event']))==C.driver_canonical(dict(kind=after['kind'],event=after['event']))
                oldbits.append(before['evidence']['consumption']);newbits.append(after['evidence']['consumption'])
                for payload,source in [(before,oldc['source_release_candidate']),(after,c['source_release_candidate'])]:
                    if pair['task']['arm']=='full_return' and pair['task']['layer']=='B':source=O.ARTIFICIAL_SOURCE
                    original=C.restore(payload['evidence']['adapted']['codec'],source_identity=source);value=original.read()
                    # New SOURCE identities are mandatory changes in provenance,
                    # never falsely call full file/Evidence digests unchanged.
                    value.pop('source_id',None);originals.append(C.fingerprint(value))
                assert originals[-1]==originals[-2]
                count+=1
            assert C.driver_canonical(oldbits)==C.driver_canonical(newbits)
            rows.append(dict(task=key,event_trace_sha256=pair['outcome']['trace_sha256'],trace_hash_equal=True,
                first_correctness_axis_prefixes_equal=True,bit_consumption_sha256=C.fingerprint(newbits),
                bit_consumption_equal=True,original_evidence_semantics_equal_excluding_new_source_identity=True))
    assert count==512
    return dict(trials=count,pairs=rows,full_file_hashes_expected_to_change_due_to_SOURCE_manifest_pid_timing_and_metadata=True,
        scientific_event_hash_and_bits_unchanged=True,additional_scientific_sampling=0)

if __name__=='__main__':
    dest=HERE/'artificial';dest.mkdir(exist_ok=False);os.environ['B3_FINAL_CONTROL_RAW']=str(dest/'control_raw')
    names=['test_g6_w2c_b3_post512_metadata','test_g6_w2c_b3_final_live_v1.Unit',
        'test_g6_w2c_b3_final_live_v1.Preflight','test_g6_w2c_b3_final_live_v1.SimpleState']
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    with (dest/'tests.log').open('w') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    tests=dict(run=result.testsRun,PASS=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),FAIL=len(result.failures),ERROR=len(result.errors),SKIP=len(result.skipped),failures=[dict(id=t.id(),trace=s) for t,s in result.errors+result.failures])
    save(dest/'tests.json',tests)
    if not result.wasSuccessful() or result.skipped:print(json.dumps(tests));sys.exit(1)
    c=K.candidate(ROOT);cases=[]
    scenarios=[('normal','ARTIFICIAL_TECHNICAL_COMPLETE'),('P1_angle','TECHNICAL_STOP'),('unknown_ack','TECHNICAL_STOP'),('receipt_close','INCOMPLETE_EVIDENCE')]
    with ProcessPoolExecutor(max_workers=len(scenarios),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,(name,c)):want for name,want in scenarios}
        for future in as_completed(futures):
            row=future.result();assert row['verification']['status']==futures[future],row
            assert not row['scientific_core_loaded'] and not row['denied_core_imports'];cases.append(row)
            print(json.dumps(dict(case=row['scenario'],status=row['classification'])),flush=True)
    normal=next(x for x in cases if x['scenario']=='normal');semantic=compare_normal(normal,c);save(dest/'before_after_semantics.json',semantic)
    mutation=mutations(Path(normal['root']),c,O.read(Path(normal['root'])/'manifest_input.json'));assert mutation['FAIL']==0
    save(dest/'results.json',dict(tests=tests,cases=cases,mutations=mutation,semantics=semantic,guard=G.observation(),scientific_operations=0,real_permits_issued=0,real_claims_created=0,separate_reviewer=False))
    with tarfile.open(dest/'raw_artificial.tar.gz','x:gz') as archive:
        for row in cases:archive.add(row['root'],arcname=row['scenario'])
        if (dest/'control_raw').exists():archive.add(dest/'control_raw',arcname='control_raw')
    assert not G.observation()['core_loaded'] and not G.ATTEMPTS
