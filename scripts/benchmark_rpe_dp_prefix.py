"""Fresh-process artificial DP pair benchmark; no saved physical probabilities."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_prefix_fixture',ROOT/'tests/test_rpe_dp_prefix_reuse.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


def worker(mode,output):
    # Small artificial counts, finite frontier; no simulation of a new H4 condition.
    start=time.perf_counter();_,tt,scope=F.example(shots=4,states=256,pairs=10000000)
    other=F.changed_tail(tt,shots=3);prepared=time.perf_counter();calls=[];answers=[];events=[]
    with tempfile.TemporaryDirectory(prefix='d2-prefix-artificial-')as tmp:
        store=F.C.PrefixStore(None if mode=='memory'else Path(tmp)/'cache')
        for trial in (tt,other):
            if mode=='ordinary':engine=F.P.Accelerator(workers=1,outcome_batch=4)
            else:engine=F.C.accelerator_class(F.P)(store=store,cache_namespace={'fixture':'ARTIFICIAL_BENCHMARK'},workers=1,outcome_batch=4)
            t=time.perf_counter();answer=engine(trial,h=10.,target=0.,scope=scope);duration=time.perf_counter()-t
            answers.append(answer);calls.append(duration)
            events.append(getattr(engine,'cache_events',[]))
            # Simulate a new Q3 worker opening the same owned cache on the next call.
            if mode=='disk':store=F.C.PrefixStore(Path(tmp)/'cache')
        cache_bytes=sum(p.stat().st_size for p in Path(tmp).glob('cache/*.json'))
        if mode=='memory':cache_bytes=len(F.C._bytes(F.C._encode(store.memory)))
    data=dict(mode=mode,scope='ARTIFICIAL_13_ROUNDS_SAME_PREFIX_M12_CHANGED',DP_call_seconds=calls,
        DP_pair_seconds=sum(calls),preparation_seconds=prepared-start,whole_worker_seconds=time.perf_counter()-start,
        peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_serialized_bytes=cache_bytes,events=events,
        output_hashes=[hashlib.sha256(F.C._bytes(F.C._encode(v))).hexdigest()for v in answers],
        MSE=[{k:r[k]for k in ('status','MSE_lower','MSE_upper','transition_pairs','residual_probability')}for r in answers],
        real_Q3_calls=0,python=sys.version)
    output.write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=('ordinary','memory','disk'));p.add_argument('--repetitions',type=int,default=3);a=p.parse_args()
    if a.worker:return worker(a.worker,a.output)
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for repeat in range(a.repetitions):
        baseline=None
        for mode in ('ordinary','memory','disk'):
            dest=a.output/(str(repeat)+'_'+mode+'.json')
            subprocess.run([sys.executable,'-B',__file__,'--output',str(dest),'--worker',mode],check=True,cwd=ROOT)
            r=json.loads(dest.read_text());r['repeat']=repeat
            if mode=='ordinary':baseline=r
            r['same_entire_DP_output']=r['output_hashes']==baseline['output_hashes']
            if not r['same_entire_DP_output']:raise ValueError('Artificial DP outputs differ')
            r['pair_speedup']=baseline['DP_pair_seconds']/r['DP_pair_seconds']
            r['second_call_speedup']=baseline['DP_call_seconds'][1]/r['DP_call_seconds'][1]
            rows.append(r);print(repeat,mode,[round(t,4)for t in r['DP_call_seconds']],flush=True)
    groups=[]
    for mode in ('ordinary','memory','disk'):
        rr=[r for r in rows if r['mode']==mode]
        groups.append(dict(mode=mode,median_DP_call_seconds=[statistics.median(r['DP_call_seconds'][i]for r in rr)for i in (0,1)],
            median_DP_pair_seconds=statistics.median(r['DP_pair_seconds']for r in rr),
            median_pair_speedup=statistics.median(r['pair_speedup']for r in rr),
            median_second_call_speedup=statistics.median(r['second_call_speedup']for r in rr),
            median_peak_process_RSS_KiB=statistics.median(r['peak_process_RSS_KiB']for r in rr),
            median_cache_serialized_bytes=statistics.median(r['cache_serialized_bytes']for r in rr)))
    result=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='ARTIFICIAL_ONLY_NOT_REAL_Q3_TIMING',repetitions=a.repetitions,rows=rows,summary=groups,real_Q3_calls=0,
        caveats=['Each mode in a fresh subprocess; RSS includes interpreter/fixtures.',
          'DP time includes restore/save/hash; does not include full Q3 metadata, closure or table construction.',
          'Serial artificial pair measures opportunity after a completed checkpoint; simultaneous cold jobs can miss.'])
    (a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
