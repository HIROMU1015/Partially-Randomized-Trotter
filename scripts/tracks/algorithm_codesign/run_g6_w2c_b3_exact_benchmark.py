"""Four unadopted artificial exact variants; never runs G7/F3 sessions or claims.

Fresh owned child per pair, sequential variant measurements, no pool/scheduler,
no buffering change. Fixed SOURCE is never modified while measuring.
"""
from fractions import Fraction as F
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from types import ModuleType

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_driver as FAST
from trottertracks.algorithm_codesign import g6_w2c_b3_angle_recurrence as REC
from trottertracks.algorithm_codesign import g6_f3_precision as PREC
BASE='564409487a35e15d73f1d0f0c2611824b400ab02'
raw=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py'])
REF=ModuleType('trottertracks.algorithm_codesign._artificial_benchmark_baseline')
REF.__package__='trottertracks.algorithm_codesign';REF.__file__='FIXED_GIT_BASELINE_DRIVER_NO_SCIENCE'
sys.modules[REF.__name__]=REF;exec(compile(raw,REF.__file__,'exec'),REF.__dict__)
VARIANTS=('baseline','fused','recurrence','combined')


class Bits:
    def __init__(self,axis,arm):self.key='ARTIFICIAL_EXACT_BENCH/'+arm+'/'+axis;self.n=0;self.consumed_bits=0
    def bits(self,n):
        value=int.from_bytes(sha256((self.key+'/'+str(self.n)).encode()).digest(),'big')>>(256-n)
        self.n+=1;self.consumed_bits+=n;return value


class Endpoint:
    def __init__(self,module,variant,width,arm,layer):
        self.module,self.variant,self.width,self.arm,self.layer=module,variant,width,arm,layer
    def shared_identity(self):return 'ARTIFICIAL_FIXED_STATE_NO_MEMO'
    def sample(self,bits,trial_id):
        x=bits.bits(160);kind=('ZERO_R','ZERO_C','EVENT','EVENT')[x%4]
        q=F(2**self.width+(x%129)+1,2**(self.width+2)+57)
        if kind!='EVENT':return self.module.Outcome(kind,evidence={'ARTIFICIAL_ONLY':True,'token':x})
        word=(0,1);weight=q*F(7,11)
        event={'raw_word':word,'word':word,'circuit_time_word':word[::-1],'phase_i_power':2,
               'ratio':q,'weight':weight,'proposal':F(2,3),'coefficient':weight*F(2,3),'child':2,
               'provider_calls':{0:1,1:1,2:2},'ARTIFICIAL_ONLY':True}
        evidence={'ARTIFICIAL_ONLY':True,'rational_tree':[(i,F(i+1,i+19))for i in range(12)],'token':x}
        return self.module.Outcome('EVENT',event,evidence)
    def angle_for(self,result):
        kw=dict(tau=F(1,1000000),root_bits=32,terms=32,sigma=1,
                context={'ARTIFICIAL_ONLY':True,'fixture':'local exact benchmark','event_id':self.module.identity(result.event)})
        f=REC.enclose_atan_artificial if self.variant in ('recurrence','combined')else PREC.enclose_atan
        return f(result.event['ratio'],**kw)


def readback(path):
    previous,normalized_previous='0'*64,'0'*64
    file_hash,normalized_file=sha256(),sha256();rows=0;norm_bytes=0
    with path.open('rb')as f:
        for line in f:
            file_hash.update(line);env=json.loads(line)
            assert env['previous']==previous
            previous=sha256((previous+env['payload']).encode()).hexdigest();assert previous==env['sha256']
            payload=json.loads(env['payload']);payload.pop('generator_exclusive_ns',None)
            text=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True)
            h=sha256((normalized_previous+text).encode()).hexdigest()
            n=(json.dumps({'previous':normalized_previous,'sha256':h,'payload':text},separators=(',',':'))+'\n').encode()
            normalized_previous=h;normalized_file.update(n);norm_bytes+=len(n);rows+=1
    return dict(file_sha256=file_hash.hexdigest(),chain_sha256=previous,rows=rows,
                normalized_file_sha256=normalized_file.hexdigest(),normalized_chain_sha256=normalized_previous,
                normalized_bytes=norm_bytes)


def pair_child(send,path,variant,width,arm,layer,n):
    module=FAST if variant in ('fused','combined')else REF
    memory_before=FAST.memory_snapshot();before=(time.process_time_ns(),time.monotonic_ns())
    meter=module.Meter(module.clock);e=Endpoint(module,variant,width,arm,layer)
    s=module.Stream(path,meter,byte_limit=68719476736,flush_rows=4096)
    summary=module.run_pair(e,lambda axis:Bits(axis,arm),[dict(arm=arm,layer=layer,axis=a,N=n)for a in module.AXES],
        s,meter,phase='profile',pass_name='baseline',correctness_trials=16)
    s.close();after=(time.process_time_ns(),time.monotonic_ns());elapsed=tuple(b-a for a,b in zip(before,after))
    send.send(dict(summary=summary,cpu_ns=elapsed[0],wall_ns=elapsed[1],output_bytes=s.bytes,
        file_sha256=s.file_hash.hexdigest(),chain_sha256=s.previous,coarse=meter.rows,
        reconciliation=meter.reconcile(elapsed),memory_before=memory_before,memory_after=FAST.memory_snapshot()))
    send.close()


def measure(out,variant,width,n):
    out.mkdir();ctx=mp.get_context('fork');rows=[]
    before=(time.process_time_ns(),time.monotonic_ns());child_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    for arm in FAST.ARMS:
        for layer in FAST.LAYERS:
            path=out/(arm+'_'+layer+'.jsonl');receive,send=ctx.Pipe(False)
            p=ctx.Process(target=pair_child,args=(send,path,variant,width,arm,layer,n));p.start();send.close()
            row=receive.recv();receive.close();p.join();assert p.exitcode==0
            row.update(arm=arm,layer=layer,path=path.name);rows.append(row)
    after=(time.process_time_ns(),time.monotonic_ns());child_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    wall=(after[1]-before[1])/1e9;parent_cpu=(after[0]-before[0])/1e9
    child_cpu=child_after.ru_utime+child_after.ru_stime-child_before.ru_utime-child_before.ru_stime
    # Readback/exact normalized comparison excluded from measured interval.
    for row in rows:
        row['readback']=readback(out/row['path'])
        assert row['readback']['file_sha256']==row['file_sha256']and row['readback']['chain_sha256']==row['chain_sha256']
    return dict(variant=variant,width=width,artificial_trials=16*n,wall_s=wall,parent_cpu_s=parent_cpu,
        owned_child_cpu_s=child_cpu,total_cpu_s=parent_cpu+child_cpu,cpu_us_per_trial=(parent_cpu+child_cpu)*1e6/(16*n),
        wall_us_per_trial=wall*1e6/(16*n),output_bytes=sum(r['output_bytes']for r in rows),
        normalized_bytes=sum(r['readback']['normalized_bytes']for r in rows),
        maximum_child_HWM_bytes=max(r['memory_after'].get('VmHWM_bytes')or 0 for r in rows),
        maximum_child_baseline_RSS_bytes=max(r['memory_before'].get('VmRSS_bytes')or 0 for r in rows),
        per_pair_cpu_ns=sum(r['cpu_ns']for r in rows),pairs=rows)


def micro(variant):
    module=FAST if variant=='fused'else REF
    payload={'ARTIFICIAL_ONLY':True,'event':{'phase_i_power':2,'weight':F(5,17),'word':(0,1),'provider_calls':{0:1,1:2}},
             'tree':[(i,F(i+1,i+37),'日本語\x7f')for i in range(32)]}
    function=(lambda:module.serialization.canonical(payload))if module is FAST else(lambda:module.audit.canonical(module.exact(payload)))
    expected=REF.audit.canonical(REF.exact(payload));assert function()==expected
    for _ in range(32):function()
    before=(time.process_time_ns(),time.monotonic_ns());last=None
    for _ in range(4000):last=function()
    after=(time.process_time_ns(),time.monotonic_ns());assert last==expected
    return dict(variant=variant,operations=4000,cpu_ns=after[0]-before[0],wall_ns=after[1]-before[1],
                bytes_per_operation=len(expected.encode()),sha256=sha256(expected.encode()).hexdigest())


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source_paths=['src/trottertracks/algorithm_codesign/'+n for n in ('g6_w2c_b3_driver.py','g6_w2c_b3_exact_serialization.py','g6_w2c_b3_angle_recurrence.py','g6_f3_precision.py','g6_f3_contract.py','return_aggregation.py')]
    pins={x:sha256((ROOT/x).read_bytes()).hexdigest()for x in source_paths}
    rows=[];rng=random.Random(19317)
    for width in (96,256):
        for variant in VARIANTS:measure(a.output/('warm_'+str(width)+'_'+variant),variant,width,4)
        for repeat in range(5):
            order=list(VARIANTS);rng.shuffle(order)
            for variant in order:
                row=measure(a.output/('w'+str(width)+'_r'+str(repeat)+'_'+variant),variant,width,64)
                row['repeat']=repeat;rows.append(row)
                print(json.dumps({k:row[k]for k in ('width','repeat','variant','wall_s','total_cpu_s')}),flush=True)
            current=[r for r in rows if r['width']==width and r['repeat']==repeat]
            ref=next(r for r in current if r['variant']=='baseline')
            for r in current:
                for x,y in zip(ref['pairs'],r['pairs']):
                    assert x['summary']==y['summary']
                    assert all(x['readback'][k]==y['readback'][k]for k in ('normalized_file_sha256','normalized_chain_sha256','normalized_bytes','rows'))
    micros=[]
    for repeat in range(5):
        order=['baseline','fused'];rng.shuffle(order)
        for v in order:micros.append(dict(micro(v),repeat=repeat))
    summary=[]
    for width in (96,256):
        ref={r['repeat']:r for r in rows if r['width']==width and r['variant']=='baseline'}
        for variant in VARIANTS:
            group=[r for r in rows if r['width']==width and r['variant']==variant]
            summary.append(dict(width=width,variant=variant,repeats=5,
                median_cpu_s=statistics.median(r['total_cpu_s']for r in group),median_wall_s=statistics.median(r['wall_s']for r in group),
                median_cpu_us_per_trial=statistics.median(r['cpu_us_per_trial']for r in group),
                median_wall_us_per_trial=statistics.median(r['wall_us_per_trial']for r in group),
                paired_median_CPU_speedup=statistics.median(ref[r['repeat']]['total_cpu_s']/r['total_cpu_s']for r in group),
                paired_median_wall_speedup=statistics.median(ref[r['repeat']]['wall_s']/r['wall_s']for r in group),
                paired_CPU_speedup_range=[min(ref[r['repeat']]['total_cpu_s']/r['total_cpu_s']for r in group),max(ref[r['repeat']]['total_cpu_s']/r['total_cpu_s']for r in group)],
                max_child_HWM_bytes=max(r['maximum_child_HWM_bytes']for r in group),
                output_bytes_range=[min(r['output_bytes']for r in group),max(r['output_bytes']for r in group)],
                normalized_bytes=group[0]['normalized_bytes'],exact_normalized_results=True))
    assert pins=={x:sha256((ROOT/x).read_bytes()).hexdigest()for x in source_paths}
    data=dict(schema='b3_local_artificial_exact_benchmark_v1',baseline_commit=BASE,source_pins=pins,
        environment=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),CPU_count=os.cpu_count(),cpu_model=next((s.split(':',1)[1].strip()for s in Path('/proc/cpuinfo').read_text().splitlines()if s.startswith('model name')),'')),
        measurements=rows,summary=summary,micro=micros,real_G7_science_calls=0,real_authorization_claim_issuance_consumption=0,
        sequential_fresh_child_per_pair=True,scientific_schedule_changed=False,fsync_durability_claim=False,
        precision_module_executed_on_artificial_rationals_only=True,mandatory_STOP=True)
    (a.output/'benchmark_results.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'summary':summary,'micro_CPU_speedup':statistics.median(r['cpu_ns']for r in micros if r['variant']=='baseline')/statistics.median(r['cpu_ns']for r in micros if r['variant']=='fused')},indent=2),flush=True)

if __name__=='__main__':main()
