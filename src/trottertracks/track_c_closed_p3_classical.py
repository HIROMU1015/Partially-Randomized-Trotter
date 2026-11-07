"""Limited classical generation/weight timings, separate from quantum gate costs."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import math
import os
import time
import numpy as np
from .track_c_closed_p3 import *
from .track_c_closed_p3_resources import canonical_tables,setup
from .track_c_closed_p3_weights import compressed_weight,compressed_weight_counts
from trotterlib.rte import _make_event
from .track_c_h4_finite_bits import integer_cdf


def benchmark(cell):
    t,n=cell;p=probabilities();x=F.from_float(common()['preps'][3].exact_rte_lambda_r)*F.from_float(t)/n;count=4096
    start=time.process_time();d,table=canonical_tables(t,n);orders=integer_cdf(d.order_probabilities,BITS);canonical_setup=time.process_time()-start
    def ordinary():
        bits=BitStream(81007);symbols=common()['preps'][3].rte_preparation.symbolic_tail.components
        for _ in range(count):
            oi=bisect_right(orders,bits.bits(BITS));indices=[bisect_right(table,bits.bits(BITS)) for _ in range(1 if oi==0 else 3)]
            _make_event(indices,symbols,d,oi)
    start=time.process_time();ordinary();canonical_online=time.process_time()-start
    start=time.process_time();g=CachedCanonical(p,x,3,'closed_P3_tail',root_bits=BITS,probability_bits=BITS,eta=ETA,rho=RHO);sampler=DigitalSampler(g);generation_setup=time.process_time()-start
    start=time.process_time();cert=certify(g);certification=time.process_time()-start
    rows=[]
    for label in ('cold','warm'):
        bits=BitStream(81007);start=time.process_time();events=[sampler.sample(bits) for _ in range(count)];cpu=time.process_time()-start
        start=time.process_time();weight,error,info=compressed_weight(e['weight'] for e in events);product_cpu=time.process_time()-start
        rows.append(dict(cache=label,microsteps=count,online_cpu_seconds=cpu,weight_product_cpu_seconds=product_cpu,distinct_weight_values=info['distinct_binary_weights'],
            interval_width_guard_pass=True,weighted_product_value=weight,weighted_product_error_upper=str(error),fresh_seeded_bit_map=True))
    one=g.event(0,(),0)
    maximum_registered_events=max(s['signal']['RTE_events'] for s in policy()['selected'] if any(z['time']==t and z['r']==n for z in s['signal']['recipe']))
    start=time.process_time();_,err,info=compressed_weight_counts({float(one['weight']):int(maximum_registered_events)});big_cpu=time.process_time()-start
    return dict(t=t,n=n,canonical_setup_cpu_seconds=canonical_setup,canonical_4096_sampling_and_event_cpu_seconds=canonical_online,
        ClosedP3_group_setup_cpu_seconds=generation_setup,ClosedP3_all_conditional_certificate_cpu_seconds=certification,ClosedP3=rows,
        compressed_registered_count_product_cpu_seconds=big_cpu,compressed_registered_count=maximum_registered_events,compressed_product_distinct=info['distinct_binary_weights'],
        scope='seeded classical diagnostic, not quantum measurements or unbiased CPU performance ranking; no unmeasured whole-RPE wall-time guarantee')


def run():
    target=OUT/'CLASSICAL.json'
    if target.exists():raise RuntimeError('Preserve classical result')
    chosen=sorted({cells()[0],max(cells(),key=lambda q:q[1])})
    with ProcessPoolExecutor(max_workers=len(chosen)) as pool:rows=list(pool.map(benchmark,chosen))
    p=probabilities();chi=sum(q*q for q in p)
    save_json(target,dict(**provenance(),rows=rows,chi=float(chi),chi_exact=str(chi),
        warning='CPU seconds, gate counts, and quantum trial count remain separate. Timings include the stated API boundaries and caches; exact rational full-path products are not naively expanded.',
        path_weight_guarantee='Binary64 event-weight rounding contributes per-step u*B; final rounding and <=1e-24 interval radius are charged in the final qualification. At <=2^23 microsteps, <=49N outward multiplications with conservative u_mp=2^-320 give width far below 1e-24 for positive weights near 1.'))
    print('CLASSICAL_COMPLETE',flush=True)

if __name__=='__main__':run()
