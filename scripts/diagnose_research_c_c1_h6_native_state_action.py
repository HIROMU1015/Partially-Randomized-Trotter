#!/usr/bin/env python3
"""One saved H6 route, independent state actions; no build/compile/PF/fit."""
import argparse,json,os,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path);args=p.parse_args()
    from trottertracks import c1_h6_native as native
    io=native.io;c=io.read(args.contract)
    if c['scope']!='C1_H6_SAVED_NATIVE_STATE_ACTION_DIAGNOSTIC_V1':raise ValueError('SCOPE')
    if io.sha(ROOT/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in [io.read(ROOT/c['source_manifest'])['files'],c['saved_inputs']]:
        for path,wanted in group.items():
            if io.sha(ROOT/path)!=wanted:raise ValueError('IDENTITY: '+path)
    import importlib.metadata
    for name,version in c['evaluation_packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('ENVIRONMENT: '+name)
    if not args.execute:print(json.dumps(dict(status='PASS',science_calls=0)));return
    if not args.output or args.output.resolve()!=(ROOT/c['fresh_output']).resolve():p.error('Fixed fresh output required')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from qiskit import qpy
    from qiskit.quantum_info import Statevector
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    calls=native.saved.molecule.integrated.Calls(out,c['call_limits'],5)
    start=time.monotonic();cpu=time.process_time()
    def load():
        with (ROOT/c['raw']).open('rb') as f:raw=qpy.load(f)[0]
        with (ROOT/c['compiled']).open('rb') as f:compiled=qpy.load(f)[0]
        with np.load(ROOT/c['ground'],allow_pickle=False) as a:ground=a['state']
        with np.load(ROOT/c['kernel'],allow_pickle=False) as a:kernel=a['pf']
        ctx=native.saved.molecule.SectorCAR(6)
        indices=native.saved.molecule.stage1.bit_reverse_indices(12)[ctx.sector.basis_indices]
        inputs,expected=native.action.probes(indices,ground,4096,kernel)
        if raw.num_qubits!=13 or compiled.num_qubits!=13:raise ValueError('QPY_SCHEMA')
        np.savez_compressed(out/'probes.npz',inputs=inputs,expected=expected)
        return raw,compiled,inputs,expected
    try:
        raw,compiled,inputs,expected=calls('saved_probe_load','CDF_s1_literal fixed5 probes',load)
        cpus=len(os.sched_getaffinity(0));results={};workers=min(cpus,5)
        with threadpool_limits(limits=1):
            for name,circuit in [('raw',raw),('compiled',compiled)]:
                for method in ['Aer','Qiskit']:
                    def evaluate(circuit=circuit,method=method):
                        if method=='Aer':value=native.action.StateActions(cpus).apply(circuit,inputs)
                        else:
                            # Each independent Statevector gets its own input copy.
                            with ThreadPoolExecutor(max_workers=workers) as pool:
                                value=np.asarray(list(pool.map(lambda v:Statevector(v.copy()).evolve(circuit).data,inputs)))
                        if not np.isfinite(value).all():raise ValueError('NONFINITE')
                        np.savez_compressed(out/(name+'_'+method+'.npz'),states=value)
                        return value
                    results[name,method]=calls('state_action',name+'_'+method,evaluate)
        pairs={}
        for name,a,b in [('raw_Aer_reference',results['raw','Aer'],expected),('raw_Qiskit_reference',results['raw','Qiskit'],expected),
                         ('compiled_Qiskit_raw_Qiskit',results['compiled','Qiskit'],results['raw','Qiskit']),
                         ('compiled_Aer_Qiskit',results['compiled','Aer'],results['compiled','Qiskit'])]:
            pairs[name]=dict(max_entry=float(np.max(abs(a-b))),per_probe_max=np.max(abs(a-b),axis=1).tolist(),per_probe_L2=np.linalg.norm(a-b,axis=1).tolist())
        io.save(out/'result.json',dict(status='DIAGNOSTIC_COMPLETE',comparisons=pairs,scientific_units=5,new_build_compile=0,new_DF_PF_fit=0,
            statevector_workers=workers,Aer_threads=cpus,retry=0,resume=0,global_operator_bound=None,mandatory_stop=True))
    except BaseException as error:
        calls.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
