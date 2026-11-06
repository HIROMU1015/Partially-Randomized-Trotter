from pathlib import Path
import json,resource,sys,time
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'/'management')]

def main():
    from threadpoolctl import threadpool_limits
    from test_research_c_native_process import toy,caller,policy
    from trottertracks import c_pr_integrated_performance as perf
    def forbidden(*a,**k):raise AssertionError('NO_SAVED_SCIENTIFIC_ARRAYS')
    perf.loader.load=forbidden
    with threadpool_limits(limits=1):a,prepared=toy(n=4,rank=8,r=32)
    out=ROOT/'.cache/h4_process_production/native32_final';out.mkdir()
    tick=time.monotonic();cpu=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN)
    rows,cal=perf.native(ROOT,dict(source_manifest_sha256='ARTIFICIAL_PROCESS_BENCHMARK_V1'),out,caller(),a,
        sampling=policy(32),prepared=prepared,backend='process')
    elapsed=time.monotonic()-tick;end=resource.getrusage(resource.RUSAGE_CHILDREN)
    total=time.process_time()-cpu+end.ru_utime+end.ru_stime-child.ru_utime-child.ru_stime
    golden=ROOT/'.cache/h4_process_benchmark_r32/threads'
    for i in range(8):
        for kind in ('events','raw','compiled','adjoint','XY','metrics'):
            name=f'sample{i:02d}_{kind}.json'
            assert json.loads((golden/'samples'/name).read_text())==json.loads((out/'samples'/name).read_text()),name
    timing=json.loads((out/'native_timing.json').read_text())
    result=dict(status='ARTIFICIAL_PRODUCTION32_FULL512_PASS',samples=len(rows),events_per_sample=32,
        full_input_output_dimension=512,wall_seconds=elapsed,CPU_seconds=total,average_CPU_cores=total/elapsed,
        same_old_thread_events_metrics_checks_first8=True,all32_full_gates=True,
        molecular_inputs_loaded=0,scientific_performance_claim=False,native_timing=timing)
    (ROOT/'artifacts/research_c_h4_native_process/2026-11-06/technical/production32.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
