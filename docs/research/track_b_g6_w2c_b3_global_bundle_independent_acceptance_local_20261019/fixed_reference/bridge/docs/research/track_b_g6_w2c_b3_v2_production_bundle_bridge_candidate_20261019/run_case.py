"""One fresh artificial bundle per process; no scientific module imports."""
from pathlib import Path
import argparse,json,sys,tempfile,time,resource,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True

def main():
    from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_contract_v1 as K
    from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_bridge_v1 as B
    from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_store_v1 import FakeStore
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--record',required=True);args=p.parse_args();spec=json.loads(Path(args.spec).read_text())
    root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-'));record=dict(case=spec['name'],root=str(root),spec=spec,measured_git_SOURCE_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    c=K.candidate(ROOT);m=K.manifest(c,'ARTIFICIAL_ONLY/'+root.name,**spec.get('manifest',{}));store=FakeStore(root)
    (root/'candidate_input.json').write_bytes(K.encoded(c));(root/'manifest_input.json').write_bytes(K.encoded(m))
    Path(args.record).write_text(json.dumps(record,indent=2)+'\n')
    start=(time.process_time_ns(),time.monotonic_ns());ch=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        result=B.execute_artificial(c,m,root=root,store=store,storage_fault=B.Fault(spec.get('fault')))
        record.update(returned_complete=result['complete'],fault_hits=result['fault_hits'])
    except BaseException as exc:record.update(exception=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()))
    after=(time.process_time_ns(),time.monotonic_ns());end=resource.getrusage(resource.RUSAGE_CHILDREN)
    record['external_host_outer']=dict(parent_cpu_ns=after[0]-start[0],wall_ns=after[1]-start[1],worker_cpu_ns=round((end.ru_utime+end.ru_stime-ch.ru_utime-ch.ru_stime)*1e9),includes_final_and_delivery_close=True)
    record['forbidden_imports']=[x for x in sys.modules if x.rsplit('.',1)[-1]in B.P.FORBIDDEN]
    obs=dict(schema='b3_bundle_caller_observation_v1',manifest_binding=K.digest(m),run_id=m['run_id'],bridge_returned_normally='exception'not in record,delivery_sha256=__import__('hashlib').sha256((root/'output/delivery.json').read_bytes()).hexdigest()if(root/'output/delivery.json').exists()else None,exception=record.get('exception',{}).get('reason'),host_outer=record['external_host_outer'],forbidden_imports=record['forbidden_imports'])
    data=K.encoded(obs)+b'\n';assert len(data)<=K.OBSERVATION_SLOT
    with (root/'external_observation.json').open('xb',buffering=0)as witness:witness.write(data+b' '*(K.OBSERVATION_SLOT-len(data)));witness.flush()
    Path(args.record).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items()if k not in ('spec','exception')}))
    if 'exception'in record:print(record['exception']['traceback'])
if __name__=='__main__':main()
