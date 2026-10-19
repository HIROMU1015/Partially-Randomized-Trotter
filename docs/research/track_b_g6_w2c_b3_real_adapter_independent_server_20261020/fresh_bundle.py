"""Fresh artificial scopes through the fixed bridge, with preload guards."""
import scope_guard as GUARD  # Also runs before subject imports in spawn children.
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode=True
RUNTIME=Path('/tmp/b3-independent-server-r2-20261020/runtime')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(RUNTIME/'src'))

def gate_observations():
    from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import contract as K,bridge as B,bits as Bits
    from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1.ports import install_guard
    install_guard()
    observations=[]
    for flags in ({},{'run_ready':True,'content_adopted':True,'contract_grants_execution':True},
                  {'science_release':'FAKE','grant':'FAKE','authority':'FAKE'}):
        for label,call in [('contract_entry',K.production_entry),('bridge_entry',B.production_entry),
                           ('factory',B.ProductionFactory()),('production_bits',B.ProductionBits()),('real_bits',Bits.real_bits)]:
            try:call(**flags)
            except RuntimeError as exc:observations.append(dict(name=label,flags=flags,rejected=True,reason=str(exc)))
            else:raise AssertionError('REAL_GATE_OPENED:'+label)
    assert not GUARD.loaded() and not GUARD.ATTEMPTS
    return dict(pid=os.getpid(),observations=observations,core_scientific_modules_loaded=GUARD.loaded(),core_import_attempts=GUARD.ATTEMPTS)

def gate_worker(queue):queue.put(gate_observations())

def run_case(name,port_faults):
    from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import contract as K,bridge as B
    from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_store_v1 import FakeStore
    root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-independent-'))
    candidate=K.candidate(RUNTIME)
    assert candidate['source_release_candidate']=='024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb'
    manifest=K.manifest(candidate,'ARTIFICIAL_ONLY/'+root.name,port_faults=port_faults)
    (root/'candidate_input.json').write_bytes(K.encoded(candidate))
    (root/'manifest_input.json').write_bytes(K.encoded(manifest))
    store=FakeStore(root)
    start=(time.process_time_ns(),time.monotonic_ns(),resource.getrusage(resource.RUSAGE_CHILDREN))
    record=dict(case=name,root=str(root),new_fake_scope=manifest['global_scope'],source='b7ad344401b10395c49e4cfa806909282859bab5')
    try:
        result=B.execute_artificial(candidate,manifest,root=root,store=store,storage_fault=B.Fault(None))
        record['returned_complete']=result['complete']
    except BaseException as exc:
        record['exception']=dict(type=type(exc).__name__,reason=str(exc),trace=traceback.format_exc())
    end=(time.process_time_ns(),time.monotonic_ns(),resource.getrusage(resource.RUSAGE_CHILDREN))
    record['host_outer']=dict(parent_cpu_ns=end[0]-start[0],wall_ns=end[1]-start[1],
        worker_cpu_ns=round((end[2].ru_utime+end[2].ru_stime-start[2].ru_utime-start[2].ru_stime)*1e9),includes_final_and_delivery_close=True)
    assert GUARD.loaded()==[] and GUARD.ATTEMPTS==[]
    obs=dict(schema='b3_bundle_caller_observation_v1',manifest_binding=K.digest(manifest),run_id=manifest['run_id'],
        bridge_returned_normally='exception' not in record,delivery_sha256=__import__('hashlib').sha256((root/'output/delivery.json').read_bytes()).hexdigest(),
        exception=record.get('exception',{}).get('reason'),host_outer=record['host_outer'],forbidden_imports=[])
    raw=K.encoded(obs)+b'\n';assert len(raw)<=4096
    with (root/'external_observation.json').open('xb',buffering=0) as stream:
        stream.write(raw+b' '*(4096-len(raw)));stream.flush()
    (HERE/('fresh_'+name+'_record.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record

def main():
    parent=gate_observations()
    ctx=mp.get_context('spawn');queue=ctx.Queue();process=ctx.Process(target=gate_worker,args=(queue,));process.start()
    child=queue.get(timeout=60);process.join(timeout=60)
    assert process.exitcode==0 and child['core_scientific_modules_loaded']==[] and child['core_import_attempts']==[]
    (HERE/'parent_worker_gate_verification.json').write_text(json.dumps(dict(parent=parent,spawn_worker=child,
        denied_calls=30,actual_core_imports=0,actual_core_import_attempts=0,guard_before_subject_imports=True),indent=2)+'\n')
    rows=[run_case('normal_n16',[]),run_case('S_angle',[dict(mode='S',pair=1,phase=0,trial=0,stage='angle')])]
    (HERE/'fresh_artificial_runs.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
