"""Fresh delivery restore and producer-free re-read of cases and overlays."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util,json,sys,tempfile,collections,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    spec=importlib.util.spec_from_file_location('bundle_restore',HERE/'restore_evidence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    temp=Path(tempfile.mkdtemp(prefix='b3-bundle-restoration-'));restore=m.restore(temp/'restored')
    sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
    from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_offline_v1 as O
    root=temp/'restored';trusted=json.loads((HERE/'technical_contract_candidate.json').read_text());matrix=json.loads((HERE/'artificial_acceptance_matrix.json').read_text());expect={r['case']:r['verification_status']for r in matrix['cases']}
    def one(pair):
        name,path,want=pair;result=O.verify(path,trusted,source_root=ROOT);assert result['status']==want,(name,result['status'],want,result['issues']);return dict(case=name,status=result['status'],issues=result['issues'])
    jobs=[(name,root/'cases'/name,want)for name,want in expect.items()]
    for path in sorted((root/'mutations').glob('*/overlay.json')):
        meta=json.loads(path.read_text());jobs.append((path.parent.name,root/'restored_mutations'/path.parent.name,meta['status']))
    with ThreadPoolExecutor(max_workers=8)as pool:rows=list(pool.map(one,jobs))
    forbidden=[x for x in sys.modules if x.rsplit('.',1)[-1]in ('g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_admission','g7_generator','g6_f3_generator')];assert not forbidden
    result=dict(schema='b3_bundle_restored_delivery_verification_v1',restore=restore,verified_cases=rows,case_count=len(expect),mutation_count=len(jobs)-len(expect),status='PASS',consumer_forbidden_imports=forbidden)
    (HERE/'restore_roundtrip_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(expect),'cases',len(jobs)-len(expect),'mutations',restore['member_files'],'raw files')
if __name__=='__main__':main()
