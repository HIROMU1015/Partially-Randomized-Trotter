"""Fresh-output CLI for the read-only independent bundle consumer."""
from pathlib import Path
import argparse,csv,json
from . import g6_w2c_b3_bundle_independent_adapter_v1 as A
from .g6_w2c_b3_bundle_independent_identity_v1 import Reference,PUBLIC,MEASURED,read

def checks(r,case):
    identity=r.get('identity',{});physical=r.get('state_physically_exact',False);complete=r['status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'
    specs=[('B-I01','Pinned source/schema/contract and declared runtime identity',bool(identity),identity),
           ('B-G01','Single global planned/reserved/started/completed counters',r.get('calls')is not None,r.get('calls')),
           ('B-S01','RAM state has exact physical statefile witness',physical,physical),
           ('B-T01','Trusted operation prefix is validated; no tail reconstruction','trace'in r,r.get('trace')),
           ('B-A01','One fake SPENT claim snapshot binding; no atomic history inference','admission'in r,r.get('admission')),
           ('B-P01','Full two-condition post-close receipt/marker/caller join',complete,r.get('documents')),
           ('B-B01','Global result/log reserved/observed/physical accounting','output_budget'in r,r.get('output_budget')),
           ('B-C01','Parent+worker CPU and caller outer wall; no double addition','cost'in r and r['cost'].get('total_cpu_ns')is not None,r.get('cost',{}).get('status'))]
    out=[]
    for code,expected,ok,observed in specs:
        judgment='PASS'if ok else'FAIL'if r['status']in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA')else'PENDING'
        out.append(dict(check_id=code,fixed_SOURCE_commit=MEASURED,public_HEAD=PUBLIC,input_identity=identity.get('manifest_digest'),
                        expected=expected,observed=observed,judgment=judgment,evidence=str(case),unresolved=None if ok else r['issues']or r['evidence_gaps']or['Required success evidence absent'],formal_adoption_required=True))
    out.append(dict(check_id='B-R01',fixed_SOURCE_commit=MEASURED,input_identity=identity.get('candidate_digest'),expected='Historical runtime actual executable/library bytes independently available',observed=identity.get('historical_runtime_bytes_proven',False),judgment='PENDING',evidence=str(case),unresolved=['Saved hashes are declarations; current runtime is recorded separately'],formal_adoption_required=True))
    return out

def write_results(r,case,dest):
    dest=Path(dest).resolve();case=Path(case).resolve()
    dest.mkdir(parents=True,exist_ok=False)
    def dump(name,value):(dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:{'exact':A.V.canonical(x)})+'\n')
    dump('verification.json',r);dump('acceptance.json',dict(status=r['status'],checks=checks(r,case),scientific_GO=False,mandatory_STOP=True))
    with (dest/'comparison.csv').open('x',newline='')as f:
        w=csv.writer(f);w.writerow(['condition','parent_cpu_ns','worker_cpu_ns','wall_ns'])
        if r.get('comparison_eligible'):
            for x in r['conditions']:w.writerow([x['condition'],x['parent_cpu_ns'],x['worker_cpu_ns'],x['wall_ns']])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--fixed-tree',required=True);p.add_argument('--new-output',required=True);p.add_argument('--trusted-local-candidate');a=p.parse_args(argv)
    case=Path(a.case).resolve();tree=Path(a.fixed_tree).resolve();out=Path(a.new_output).resolve()
    A.V.require(not(out==case or out.is_relative_to(case)or case.is_relative_to(out)or out==tree or out.is_relative_to(tree)or tree.is_relative_to(out)),'OUTPUT_OVERLAPS_INPUT')
    A.V.require(not out.exists(),'FRESH_OUTPUT_REQUIRED')
    candidate=read(a.trusted_local_candidate)if a.trusted_local_candidate else None
    ref=Reference(tree,candidate,allow_runtime_projection=candidate is not None)
    r=A.verify(case,ref);write_results(r,case,out)
    return 0 if r['status']in ('VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE','VERIFIED_ARTIFICIAL_CONDITION_ONLY','TECHNICAL_FAILURE')else 2

if __name__=='__main__':raise SystemExit(main())
