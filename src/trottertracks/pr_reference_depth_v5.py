"""Fixed saved-data V5 native wire-depth bounds; no new compiler or builder."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from fractions import Fraction as F
from pathlib import Path
import hashlib,io,json,lzma,os,platform,subprocess,sys,tarfile
from importlib.metadata import version
from .track_c_h4 import ROOT,common,save_json
from .pr_reference_bprime_accounting import distributions
from . import pr_reference_bprime as bp
from .pr_reference_transfer_qualification import frontier_depth

BASE='224440f740051ed5794e8db87a57908fc02dd02f'
OUT=ROOT/'artifacts/pr_reference_depth_v5/2026-11-07'
PILOT='artifacts/pr_reference_interference_pilot/2026-11-06/'
TRANSFER='artifacts/pr_reference_transfer_qualification/2026-11-06/'
BPRIME='artifacts/pr_reference_bprime/2026-11-07/'

def data(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
def archive(path):
    with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
        return {m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
def attach():
    if (OUT/'STOP.json').exists():raise RuntimeError('Mandatory STOP')
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')
def initialize():
    if OUT.exists():raise RuntimeError('Fresh V5 output required')
    OUT.mkdir(parents=True)
    raw=lzma.decompress(data(PILOT+'common.pkl.xz'));(OUT/'common.pkl').write_bytes(raw)
    attach()
    save_json(OUT/'INPUTS.json',dict(base=BASE,common_sha256=hashlib.sha256(raw).hexdigest(),
        new_SCF_DF_ground_sampling_compile_synthesis=0))
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),platform=platform.platform(),
        python=sys.version,executable=sys.executable,packages={p:version(p) for p in ('qiskit','numpy','scipy','mpmath')},
        affinity=len(os.sched_getaffinity(0)),meminfo=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        shared_settings_changed=False))

@lru_cache(None)
def catalogues():
    old=[json.loads(x) for x in archive(TRANSFER+'local_catalogue_boundary.tar.xz').values()]
    new=[json.loads(x) for x in archive(BPRIME+'Bprime_local_catalogue.tar.xz').values()]
    return old,new

@lru_cache(None)
def records():
    old,new=catalogues()
    return {arm:{r['name']:r for r in row['records']} for arm,row in [
        ('reference',next(x for x in old if x['request']==['reference',True,True,True])),
        ('Bprime',next(x for x in new if x['request']==[True,True]))]}

def apply_ops(instructions,levels):
    levels=list(levels)
    for name,qs,_ in instructions:
        if not qs:continue
        value=max(levels[q] for q in qs)+(name!='barrier')
        for q in qs:levels[q]=value
    return levels

def transfer_matrix(instructions,n):
    """Integer max-plus path lengths. None is absence of a dependency path."""
    rows=[[0 if i==j else None for j in range(n)] for i in range(n)]
    for name,qs,_ in instructions:
        if not qs:continue
        out=[]
        for j in range(n):
            values=[rows[q][j] for q in qs if rows[q][j] is not None]
            out.append(max(values)+(name!='barrier') if values else None)
        for q in qs:rows[q]=out.copy()
    return rows

def matrix_apply(matrix,levels):
    return [max(x+v for x,v in zip(row,levels) if x is not None) for row in matrix]
def stats(record):
    m=transfer_matrix(record['lowering'],record['n']);zero=matrix_apply(m,[0]*record['n'])
    system=range(min(8,record['n']))
    progress=min(max(m[i][j] for j in system if m[i][j] is not None) for i in system)
    return dict(matrix=m,zero_frontier=zero,span=max(zero),system_min_progress=progress,
        size=len(record['lowering']),ops=record['ops'])

def local_analysis():
    attach();selected={a:{k:stats(r) for k,r in recs.items()} for a,recs in records().items()}
    # Existing context inventory: topology, not numerical angle equality, is
    # needed for wire depth. Compare all applicable saved variants.
    checks=[]
    old,new=catalogues()
    for arm,rows in [('reference',[x for x in old if x['request'][0]=='reference']),('Bprime',new)]:
        for row in rows:
            for rec in row['records']:
                if rec['name'] not in selected[arm]:continue
                s=stats(rec);ref=selected[arm][rec['name']]
                assert s['matrix']==ref['matrix'],(arm,rec['name'],'context depth matrix mismatch')
                checks.append([arm,row['request'],rec['name']])
    result=dict(records=selected,context_checks=checks,new_compile=0,passed=True)
    save_json(OUT/'LOCAL_FRONTIERS.json',result)
    return result

@lru_cache(None)
def local():
    return json.loads((OUT/'LOCAL_FRONTIERS.json').read_text())['records']

def d_bound(arm,q,metric):
    a=local()[arm];v=lambda k:a[k][metric]
    if arm=='Bprime':
        return v('left_all')+v('right_all')+(q-1)*(v('left_no0')+v('right_no0')+v('boundary_control'))
    half=[v(f'block:{k}:half') for k in range(4)]
    if q==1:return sum(half)
    n=q//2
    return 2*n*sum(half)+(n-1)*(v('block:0:full')-2*half[0])

def family_record(arm,family):
    return next(r for r in records()[arm].values() if r['meta'].get('family')==family)

@lru_cache(None)
def cascade():
    record=family_record('reference','cswap');ops=[]
    for j in range(8):
        wires=(16,j,8+j)
        ops.extend((name,[wires[k] for k in qs],params) for name,qs,params in record['lowering'])
    return dict(depth=max(apply_ops(ops,[0]*17)),matrix=transfer_matrix(ops,17),
        CSWAPs_per_cascade=8,CX_per_CSWAP=record['ops']['cx'])

def expected_tail(arm,r,bits):
    components,p,p2,pzz,support=distributions(bits,r);a=local()[arm]
    pb=defaultdict(F)
    for c,pi in zip(components,p):
        assert len(c.diagonal_pauli_support) in (1,2),'Identity tail needs another model'
        pb[c.basis_id]+=pi
    num=r*(1+2*p2)
    pair=lambda bid,field:a['basis:'+bid][field]+a['basis_inverse:'+bid][field]
    frames_lo=sum(pair(b,'system_min_progress')*(pi+(num-1)*pi*(1-pi)) for b,pi in pb.items())
    frames_up=sum(pair(b,'span')*(pi+(num-1)*pi*(1-pi)) for b,pi in pb.items())
    if arm=='reference':
        z=local()[arm]['z']['span'];one=1;two=3
    else:
        z=local()[arm]['cz']['span']
        one=next(s['span'] for k,s in a.items() if k.startswith('crz:'))
        two=next(s['span'] for k,s in a.items() if k.startswith('bprime:'))
    rotations=r*((1-pzz)*one+pzz*two);products=2*r*p2*support*z
    # Controlled event-sign phase may remain even for even multiples of pi.
    phase=1-(1-p2)**r if arm=='Bprime' else F()
    maximum=r*(3*max(pair(b,'span') for b in pb)+max(one,two)+4*z)+(arm=='Bprime')
    return dict(lower=frames_lo,upper=frames_up+rotations+products+phase,
        frames_lower=frames_lo,frames_upper=frames_up,rotations_upper=rotations,
        products_upper=products,event_phase_upper=phase,range_upper=maximum,
        p2=p2,pzz=pzz,expected_support=support,basis_probabilities=pb)

def sqrt_upper_fraction(x,bits=128):
    """A rational ceiling for sqrt(Fraction); integer arithmetic only."""
    from math import isqrt
    x=F(x);assert x>=0
    scale=1<<bits
    quotient=(x.numerator*scale*scale+x.denominator-1)//x.denominator
    root=isqrt(quotient)
    if root*root<quotient:root+=1
    return F(root,scale)

def round_bounds(q,r,bits,axis):
    b=expected_tail('Bprime',r,bits);ref=expected_tail('reference',r,bits)
    overhead=2*local()['Bprime']['h']['span']+(axis=='Y')*local()['Bprime']['sdg']['span']+2
    b_lower=d_bound('Bprime',q,'system_min_progress')+q*b['lower']
    b_upper=d_bound('Bprime',q,'span')+q*b['upper']+overhead
    if q==1:
        ref_lower=d_bound('reference',q,'system_min_progress')+ref['lower']
        lane_up=d_bound('reference',q,'span')+ref['upper'];penalty=F()
    else:
        n=q//2
        ref_lower=d_bound('reference',q,'system_min_progress')+n*ref['lower']
        # Independent occurrence costs W in [0,M]. Var(W)<=M^2/4.
        # E max(S_F,S_G)<=n E W + .5 sqrt(2n Var W).
        penalty=sqrt_upper_fraction(F(n*ref['range_upper']**2,8))
        lane_up=d_bound('reference',q,'span')+n*ref['upper']+penalty
    ref_upper=lane_up+2*cascade()['depth']+overhead
    return dict(Bprime={'lower':b_lower,'upper':b_upper},reference={'lower':ref_lower,'upper':ref_upper},
        reference_parallel_penalty=penalty,reference_lane_upper=lane_up,
        tail_Bprime=b,tail_reference=ref,overhead=overhead)

def saved_path_bounds(row,axis):
    """Pathwise structural bounds; no average-event circuit is formed."""
    comps={c.component_id:c for c in common()['preps'][3].rte_preparation.symbolic_tail.components}
    q=row['q'];r=row['r'];events=row.get('events',row.get('event_records'))
    per_occ={a:[] for a in ('Bprime','reference')}
    for index in range(q):
        part={a:[0,0] for a in per_occ};last=None
        for ev in events[index*r:(index+1)*r]:
            rot=ev.get('rotation',ev.get('rotation_component_id'))
            products=ev.get('products',ev.get('product_component_ids'))
            for cid in [*products,rot]:
                c=comps[cid];bid=c.basis_id
                if bid!=last:
                    for arm in part:
                        a=local()[arm]
                        part[arm][0]+=a['basis:'+bid]['system_min_progress']+a['basis_inverse:'+bid]['system_min_progress']
                        part[arm][1]+=a['basis:'+bid]['span']+a['basis_inverse:'+bid]['span']
                    last=bid
            zz=len(comps[rot].diagonal_pauli_support)==2
            part['Bprime'][1]+=6 if zz else 4
            part['reference'][1]+=3 if zz else 1
            support=sum(len(comps[cid].diagonal_pauli_support) for cid in products)
            part['Bprime'][1]+=support*local()['Bprime']['cz']['span']
            part['reference'][1]+=support*local()['reference']['z']['span']
        if any(e.get('order',e.get('taylor_order'))==2 for e in events[index*r:(index+1)*r]):part['Bprime'][1]+=1
        for arm in part:per_occ[arm].append(part[arm])
    overhead=2*local()['Bprime']['h']['span']+(axis=='Y')*local()['Bprime']['sdg']['span']+2
    result={}
    lo=d_bound('Bprime',q,'system_min_progress')+sum(x[0] for x in per_occ['Bprime'])
    hi=d_bound('Bprime',q,'span')+sum(x[1] for x in per_occ['Bprime'])+overhead
    result['Bprime']=[lo,hi]
    n=q//2
    halves=[per_occ['reference'],[]] if q==1 else [per_occ['reference'][:n],per_occ['reference'][n:]]
    lo=d_bound('reference',q,'system_min_progress')+max(sum(x[0] for x in half) for half in halves)
    hi=d_bound('reference',q,'span')+max(sum(x[1] for x in half) for half in halves)+2*cascade()['depth']+overhead
    result['reference']=[lo,hi]
    return result

def validate_saved_cohort():
    attach();raw=[json.loads(x) for x in archive(BPRIME+'paired_Bprime_prices.tar.xz').values()];checks=[]
    for row in bp.saved_samples():
        saved=next(x for x in raw if (x['q'],x['r'],x['rare'],x['j'])==(row['q'],row['r'],row['rare'],row['j']))
        for i,axis in enumerate('XY'):
            bounds=saved_path_bounds(row,axis)
            for arm in bounds:
                value=saved['Bprime_values'][i][2] if arm=='Bprime' else row['values']['reference'][i][2]
                lo,hi=bounds[arm];assert lo<=value<=hi,(row['q'],row['rare'],row['j'],axis,arm,lo,value,hi)
                checks.append(dict(q=row['q'],r=row['r'],rare=row['rare'],j=row['j'],seed=row['seed'],axis=axis,arm=arm,
                    depth_lower=lo,actual_saved_depth=value,depth_upper=hi,passed=True))
    save_json(OUT/'COHORT_DEPTH_CHECKS.json',dict(checks=checks,new_compile=0,passed=True,
        qualification='pathwise structural interval inclusion; not exact reproduction of every cohort native topology'))
    return checks

def validate_fences():
    attach();old,new=catalogues();checks=[]
    for arm,rows in [('Bprime',new),('reference',[x for x in old if x['request'][0]=='reference'])]:
        for row in rows:
            for rec,pre in zip(row['records'],row['pre_last_cancel']):
                selected=rec['meta']['kind'] in ('basis','basis_inverse','baseline_D','D_half','D_full')
                if not selected:continue
                assert not pre['exposed_cancelable_prefix'] and not pre['exposed_cancelable_suffix'],(arm,rec['name'],pre)
                checks.append(dict(arm=arm,context=row['request'],record=rec['name'],pre_last_cancel=pre,passed=True))
    save_json(OUT/'FENCE_CHECKS.json',dict(checks=checks,passed=True,
        SOURCE_condition='basis contains RZ/XXPlusYY, not initial inverse-cancellation families; D directional CX internal only'))
    return checks

def qpy_depth(task):
    path,name,payload=task
    from qiskit import qpy
    q=qpy.load(io.BytesIO(payload))[0];base=Path(name).name
    qn=int(base.split('_')[0][1:]);rare=bool(int(base.split('_')[1][4:]));arm=base.split('_',2)[2][:-4]
    actual=q.depth();depth,levels=frontier_depth(q)
    assert depth==actual
    row=next(x for x in bp.saved_samples() if x['q']==qn and x['rare']==rare and x['j']==0)
    if arm=='Bprime':
        prices=[json.loads(x) for x in archive(BPRIME+'paired_Bprime_prices.tar.xz').values()]
        saved=next(x for x in prices if (x['q'],x['r'],x['rare'],x['j'])==(qn,row['r'],rare,0))['Bprime_values'][0][2]
    else:saved=row['values']['reference'][0][2]
    assert depth==saved
    bounds=saved_path_bounds(row,'X')[arm];assert bounds[0]<=depth<=bounds[1]
    names=defaultdict(int)
    for inst in q.data:names[inst.operation.name]+=1
    assert names['reset']==(8 if arm=='reference' else 0) and names['measure']==1 and not names['barrier']
    return dict(archive=path,file=name,arm=arm,q=qn,r=row['r'],rare=rare,j=0,axis='X',
        Qiskit_native_depth=actual,reused_frontier_depth=depth,wire_frontier=levels,
        saved_whole_price_depth=saved,structural_depth_bounds=bounds,ops=dict(names),passed=True)

def validate_saved_qpy():
    attach()
    from concurrent.futures import ProcessPoolExecutor
    tasks=[]
    for path in (PILOT+'native_circuits.tar.xz',TRANSFER+'native_circuits.tar.xz',BPRIME+'Bprime_native_qpy.tar.xz'):
        tasks.extend((path,name,payload) for name,payload in archive(path).items()
            if Path(name).name.split('_',2)[2][:-4] in ('Bprime','reference'))
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:rows=list(pool.map(qpy_depth,tasks))
    save_json(OUT/'QPY_DEPTH_CHECKS.json',dict(rows=rows,workers=min(len(tasks),len(os.sched_getaffinity(0))),
        new_build_compile_simulation=0,passed=True))
    return rows

def synthetic_tests():
    attach()
    from qiskit import QuantumCircuit
    import random
    rng=random.Random(1729);checks=[]
    # A common barrier joins frontiers at zero layer cost; measure uses cbit.
    for kind in ('parallel','entangled','barrier','reset_measure'):
        q=QuantumCircuit(3,1);q.rz(.123,0);q.rz(.234,1)
        if kind=='entangled':q.cx(0,1);q.cx(1,2)
        if kind=='barrier':q.rz(.3,0);q.barrier();q.cx(1,2)
        if kind=='reset_measure':q.reset(2);q.measure(2,0)
        d,_=frontier_depth(q);assert d==q.depth(),(kind,d,q.depth())
        checks.append(kind+' Qiskit frontier')
    for n in (3,8,9):
        ops=[]
        for _ in range(50):
            wires=rng.sample(range(n),2) if rng.random()<.5 else [rng.randrange(n)]
            ops.append(('cx' if len(wires)==2 else 'rz',wires,[]))
        m=transfer_matrix(ops,n)
        for _ in range(4):
            d=[rng.randrange(100) for _ in range(n)]
            assert matrix_apply(m,d)==apply_ops(ops,d)
        checks.append(str(n)+'wire nonuniform transfer')
        removed=ops[::2];assert max(apply_ops(removed,[0]*n))<=max(apply_ops(ops,[0]*n))
        checks.append(str(n)+'wire deletion monotonicity')
    x=F(2,3);s=sqrt_upper_fraction(x);assert s*s>=x and (s-F(1,1<<128))**2<x
    assert sqrt_upper_fraction(F(9,4))==F(3,2)
    checks.append('rational outward sqrt')
    q=QuantumCircuit(3);q.rz(.1,0);q.rz(.2,1);assert frontier_depth(q)[0]==1 and q.size()==2
    checks.append('parallel depth is not gate count')
    q=QuantumCircuit(17,1);q.reset(range(8,16));q.rz(.1,16);q.sx(16);q.rz(.2,16)
    assert frontier_depth(q)[0]==q.depth()==3
    checks.append('fresh reference resets parallel initial native H')
    save_json(OUT/'SYNTHETIC_TESTS.json',dict(passed=True,checks=checks,scientific_performance_claim=False))
    return checks

def account_all18():
    attach()
    for path in ('FENCE_CHECKS.json','COHORT_DEPTH_CHECKS.json','QPY_DEPTH_CHECKS.json','SYNTHETIC_TESTS.json'):
        assert json.loads((OUT/path).read_text())['passed']
    old=json.loads(data(BPRIME+'TOTALS_18ROUND.json'));prior=json.loads(data(TRANSFER+'STAGE4_RPE.json'))
    totals={a:{k:F() for k in ('lower','upper')} for a in ('Bprime','reference')};rows=[]
    def encode(x):
        if isinstance(x,F):return str(x)
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
        return x
    for saved,prev in zip(old['rows'],prior['rows']):
        q=saved['q'];r=saved['r'];bits=saved['bits'];axes={}
        for axis in 'XY':
            bound=round_bounds(q,r,bits,axis)
            shots={'Bprime':saved['shots_per_axis'],'reference':prev['arms']['reference']['shots_per_axis']}
            for arm in totals:
                for end in totals[arm]:totals[arm][end]+=shots[arm]*bound[arm][end]
            axes[axis]=dict(shots=shots,depth_per_shot=encode(bound))
        rows.append(dict(m=saved['m'],q=q,r=r,bits=bits,axes=axes))
    import math
    def endpoints(x):
        return dict(exact_fraction=str(x),lower=0 if x==0 else math.nextafter(float(x),-math.inf),
            upper=0 if x==0 else math.nextafter(float(x),math.inf))
    b=totals['Bprime']['lower'];ref=totals['reference']['upper'];margin=b-ref
    percentage=100*margin/b
    result=dict(status='CONDITIONAL_ALL18_NATIVE_DEPTH_REDUCTION_VS_BPRIME' if margin>0 else 'UNDETERMINED',
        depth_totals={a:{k:endpoints(v) for k,v in ends.items()} for a,ends in totals.items()},
        comparison=dict(reference_upper_below_Bprime_lower=margin>0,absolute_saving_lower=endpoints(margin),
            relative_saving_percent_lower=endpoints(percentage),exact_expected_depth_not_computed=True),
        rows=rows,cascade=cascade(),inherited_CX_RZ_shots_qubits=old['totals'],
        saved_shots_unchanged=True,qubits={'Bprime':9,'reference':17},reference_extra_qubits=8,
        native_measurement_included=True,reset_in_saved_native={'Bprime':False,'reference':True},
        reset_depth_treatment='8 fresh reference resets at depth 1 run in parallel with initial ancilla H of native depth 3; max(1,3)=3 already included',
        reference_extra_reset_preparations_per_shot=8,source_CSWAPs_per_shot=16,
        new_scientific_build_compile_simulation_sampling_synthesis=0,all18_size_T_Tdepth='NOT_ACQUIRED',
        guarantee_scope=old['guarantee_scope'],
        independent_review_or_external_reproduction=False,
        qualification='structural safe bounds under fixed compiler-A context and canonical independent finite-bit law; not exact expected depths, runtime, or all-resource superiority')
    save_json(OUT/'DEPTH_18ROUND.json',result)
    import csv
    with (OUT/'ROUND_DEPTHS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(['m','q','r','bits','axis','arm','shots','depth_lower_fraction','depth_upper_fraction','weighted_depth_lower_factorized','weighted_depth_upper_factorized'])
        for row in rows:
            for axis,entry in row['axes'].items():
                for arm in totals:
                    d=entry['depth_per_shot'][arm]
                    w.writerow([row['m'],row['q'],row['r'],row['bits'],axis,arm,entry['shots'][arm],d['lower'],d['upper'],str(entry['shots'][arm])+'*('+d['lower']+')',str(entry['shots'][arm])+'*('+d['upper']+')'])
    return result

def finish_stop():
    result=json.loads((OUT/'DEPTH_18ROUND.json').read_text())
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='Authorized V5-A through E complete',
        result=result['status'],old_SOURCE_raw_results_STOPs_preserved=True,
        new_scientific_build_compile_simulation_sampling_synthesis=0,
        next_owner='GPT/user: assess depth versus extra qubits and unchanged RZ/rotation limits; no automatic optimization or T synthesis'))
