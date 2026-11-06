"""Finite C2-B experiment: strict synthesis of conjugate DF blocks.

No SCF/DF/ground solve. No residual law. Source multiplicities, not a long-q
price fit, define the compositional full interrogation circuit.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
import hashlib, json, math, os, shutil, subprocess, time
from pathlib import Path
import mpmath as mp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RZZGate, PhaseGate
from qiskit.quantum_info import Operator
from .track_c_h4 import ROOT, common, save_json, weights, diagonal_primitives
from .track_c_compiler_a import compile_reference
from .track_c_compiler_a_18round import spec, interval_check, I, upper, lower
from . import track_c_h4_synthesis as synth

OUT=ROOT/'artifacts/track_c2_b_pilot/2026-11-06'
synth.OUT=OUT
BASE='720416c5b88b227cf76ff7aa729d91a0a98ed58b'
CAPS=(.01,.02)
OLD=ROOT.parent/'track-c-h4-parallel-pilots-20261106/artifacts/track_c_h4_pilots/2026-11-06/run01/synthesis'

def stored(path):
    return json.loads(subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT))

def f(x):return F.from_float(float(x))

def seq(q):
    """Exact rational fusion of adjacent identical Hermitian fragments."""
    result=[]
    for _ in range(q):
        for w in weights('Y4'):
            t=f(float(w*.02))/2
            for j in [*range(13),*reversed(range(13))]:
                if result and result[-1][0]==j:result[-1]=(j,result[-1][1]+t)
                else:result.append((j,t))
    return result

def mult(q):
    one=Counter(seq(1));two=Counter(seq(2))
    return {k:one[k]+(q-1)*(two[k]-one[k]) for k in one.keys()|two.keys() if one[k]+(q-1)*(two[k]-one[k])}

@lru_cache(None)
def lowered(family,params):
    s=spec(family,params);g={'xx_plus_yy':None}.get(family)
    from .track_c_compiler_a_18round import gate
    if family in ('p','rz'):
        qc=QuantumCircuit(1);getattr(qc,family)(params[0],0)
    else:
        op=gate(s);qc=QuantumCircuit(op.num_qubits);qc.append(op,range(op.num_qubits))
    native=compile_reference(qc)
    lowering=[[x.operation.name,[native.find_bit(q).index for q in x.qubits],list(map(float,x.operation.params))] for x in native.data]
    if family in ('p','rz'):
        # P(a)=exp(i a/2)RZ(a): a common global phase of the already-controlled joint circuit.
        err=0.
    else:
        record=dict(source=s,n=qc.num_qubits,lowering=lowering,lowering_sha256=hashlib.sha256(json.dumps(lowering).encode()).hexdigest())
        err=interval_check(record)['operator_error_upper']
    return native,err

def mapped_compose(target,local,wires):target.compose(local,qubits=list(wires),inplace=True)

def angles(qc):
    return Counter(abs(float(x.operation.params[0])).hex() for x in qc.data if x.operation.name=='rz')

def width(block):
    if block.block_kind=='one_body':v=sum((abs(f(x)) for x in block.diagonal_eigenvalues),F(0))
    else:
        low=sum((min(F(0),f(x)) for x in block.diagonal_eta),F(0));high=sum((max(F(0),f(x)) for x in block.diagonal_eta),F(0))
        sqlo=F(0) if low<=0<=high else min(low*low,high*high)
        v=abs(f(block.lam))*(max(low*low,high*high)-sqlo)
    return math.nextafter(float(v),math.inf)

def diagonal_error(block,t,prim):
    mp.iv.prec=192;errors=[]
    for index in range(256):
        occ=[(index>>q)&1 for q in range(8)]
        if block.block_kind=='one_body':e=sum((I(v)*n for v,n in zip(block.diagonal_eigenvalues,occ)),I(0))
        else:e=I(block.lam)*sum((I(v)*n for v,n in zip(block.diagonal_eta,occ)),I(0))**2
        phi=I(prim.global_phase)
        for q,a in prim.rz:phi-=I(a)*(1-2*occ[q])/2
        for q,r,a in prim.rzz:phi-=I(a)*(1-2*occ[q])*(1-2*occ[r])/2
        errors.append(upper(abs(phi+I(t)*e)))
    return max(errors)

def inventory():
    c=common();prep=c['preps'][12];blocks=prep.deterministic_blocks;basis=[];centres={};checks=[]
    for j,b in enumerate(blocks):
        native=QuantumCircuit(9);source=QuantumCircuit(9);err=0.
        for g,qs in b.runtime_basis_operations:
            source.append(g,list(qs));n,e=lowered(g.name,tuple(map(float,g.params)));mapped_compose(native,n,qs);err+=e
        # Full 256-dimensional basis, including leakage; never project a synthesized basis.
        aa=Operator(source.remove_final_measurements(inplace=False)).data
        bb=Operator(native).data
        z=np.vdot(aa,bb);delta=np.linalg.norm(aa-bb*np.exp(-1j*np.angle(z)),'fro')
        assert delta<max(1e-10,math.sqrt(512)*err+1e-11)
        basis.append(dict(angles=angles(native),cx=int(native.count_ops().get('cx',0)),W=width(b),compiler_error=err))
        checks.append(dict(block=j,joint_dimension=512,basis_compiler_Frobenius=delta,local_operator_error_sum=err))
        print('basis',j,'RZ',sum(basis[-1]['angles'].values()),flush=True)
    for (j,tr) in sorted(mult(2),key=lambda x:(x[0],x[1])):
        b=blocks[j];t=float(tr);prim=diagonal_primitives(b,t);native=QuantumCircuit(9);err=diagonal_error(b,t,prim)
        for a,bq,theta in prim.rzz:
            if theta:
                n,e=lowered('crzz',(float(theta),));mapped_compose(native,n,(8,a,bq));err+=e
        for a,theta in prim.rz:
            n,e=lowered('crz',(float(theta),));mapped_compose(native,n,(8,a));err+=e
        if prim.global_phase:
            n,e=lowered('p',(float(prim.global_phase),));mapped_compose(native,n,(8,));err+=e
        # Conversion of the exact rational merged time is separately bounded.
        err+=abs(float(f(t)-tr))*max(1.,basis[j]['W'])
        name=f'{j}:{tr.numerator}/{tr.denominator}'
        centres[(j,tr)]=dict(name=name,angles=angles(native),cx=int(native.count_ops().get('cx',0)),compiler_error=err,
            structural_weight=upper(min(I(2),abs(I(t))*I(basis[j]['W']))))
    assert Counter(seq(4))==Counter({k:v for k,v in mult(4).items()})
    save_json(OUT/'BLOCKS.json',dict(basis=basis,centres=[dict(block=k[0],rational_time=str(k[1]),**v) for k,v in centres.items()],checks=checks,
        multiplicity_q4_exact=True,formula='counter(q)=counter(1)+(q-1)*(counter(2)-counter(1)); identical fragment merge at period boundary'))
    return basis,centres

def items_for(q,basis,centres,structural):
    rows=defaultdict(lambda:[0,F(0)])
    compile_error=0.;cx=0
    for (j,tr),n in mult(q).items():
        b=basis[j];d=centres[(j,tr)];s=f(d['structural_weight'] if structural else 2.)
        for angle,count in b['angles'].items():
            rows[('B',angle)][0]+=2*n*count;rows[('B',angle)][1]+=n*s*count
        for angle,count in d['angles'].items():
            rows[('D',angle)][0]+=n*count;rows[('D',angle)][1]+=n*count
        compile_error+=n*(d['compiler_error']+2*b['compiler_error']);cx+=n*(2*b['cx']+d['cx'])
    prep=common()['preps'][12];scalar=prep.constant_coefficient+prep.extracted_identity_coefficient
    # Keep the controlled scalar, including the binary64 Y4 stage sum.
    exact=-sum((f(float(w*.02))*f(scalar) for w in weights('Y4')),F(0))*q
    theta=float(exact);compile_error+=abs(float(f(theta)-exact))
    rows[('D',abs(theta).hex())][0]+=1;rows[('D',abs(theta).hex())][1]+=1
    return [dict(role=k[0],angle=k[1],count=v[0],weight=str(v[1])) for k,v in sorted(rows.items())],math.nextafter(compile_error,math.inf),cx

def exponent(budget,total):
    e=math.floor(math.log10(float(budget/total)))
    while F(10)**e*total>budget:e-=1
    return e

def acquire_key(task):
    angle,e=task;theta=float.fromhex(angle);key=hashlib.sha256(f'{theta.hex()}:10^{e}:pygridsynth2.0.0:strict:seed17'.encode()).hexdigest()
    path=OUT/'synthesis'/f'{key}.json';old=OLD/f'{key}.json'
    if old.exists() and not path.exists():
        record=json.loads(old.read_text());assert record['status']=='PASS' and record['spec']==f'{theta.hex()}:10^{e}:pygridsynth2.0.0:strict:seed17'
        shutil.copyfile(old,path)
        reused=True
    else:reused=False
    record=synth.acquire(theta,e)
    return angle,e,record,reused

def allocate(items,budget,records,nonuniform):
    total=sum((F(x['weight']) for x in items),F(0));e=exponent(budget,total)
    chosen={i:e for i in range(len(items))};used=total*F(10)**e
    if nonuniform:
        offers=[]
        for i,x in enumerate(items):
            a=records[(x['angle'],e)];b=records[(x['angle'],e+1)]
            saving=x['count']*(a['T']-b['T']);extra=F(x['weight'])*9*F(10)**e
            if saving>0 and extra>0:offers.append((float(saving/extra),i,saving,extra))
        for _,i,saving,extra in sorted(offers,reverse=True):
            if used+extra<=budget:chosen[i]=e+1;used+=extra
    T=sum(x['count']*records[(x['angle'],chosen[i])]['T'] for i,x in enumerate(items))
    actual=sum((F(x['weight'])*F(records[(x['angle'],chosen[i])]['strict_error_upper']) for i,x in enumerate(items)),F(0))
    assert actual<=used<=budget
    return dict(T=T,nominal_operator_error_upper=float(used),actual_strict_operator_error_upper=math.nextafter(float(actual),math.inf),
        allocation=[dict(**x,exponent=chosen[i],key=records[(x['angle'],chosen[i])]['key']) for i,x in enumerate(items)])

def phase_disk_upper(disk,rho):
    mp.iv.prec=192;x=I(disk)/I(rho)
    return upper(mp.iv.atan2(x,mp.iv.sqrt(1-x*x)))

def shots(rho,beta,alpha):
    mp.iv.prec=192
    if alpha==1/720:alpha=math.nextafter(alpha,0.)
    n=I(4)*mp.iv.log(I(2)/I(alpha))/(I(rho)**2*mp.iv.sin(I(beta))**2)
    return math.ceil(upper(n))

def finish(cases,records,start):
    result={};table=[]
    for structural in (False,True):
        for nonuniform in (False,True):
            name=('structural' if structural else 'gate_sum')+('_nonuniform' if nonuniform else '_uniform');rounds=[]
            for case in cases:
                if case['structural']!=structural:continue
                allocation=allocate(case['items'],F(case['budget']),records,nonuniform)
                # Common cap ablation: same sufficient rho/beta/N for every allocation arm.
                raw=case['synthesis_raw_disk_cap'];rho=lower(I(case['rho_base'])-I(raw)-I(case['additional_raw_compiler_disk']));beta=lower(I(case['beta_base'])-I(phase_disk_upper(upper(I(raw)+I(case['additional_raw_compiler_disk'])),case['rho_base'])))
                n=shots(rho,beta,1/720)
                row=dict(m=case['m'],q=case['q'],cap=case['cap'],rho_lower=rho,beta_stat=beta,shots_per_axis=n,
                    raw_synthesis_disk_cap=raw,actual_synthesis_raw_disk_upper=2*math.sqrt(2)*allocation['actual_strict_operator_error_upper'],
                    normalizer=1.,CX=case['cx'],**allocation)
                assert rho>0 and beta>0 and 3*.4<math.pi
                rounds.append(row)
            # Each method may choose either registered cap per round; same opportunity.
            selected=[min((x for x in rounds if x['m']==m),key=lambda x:x['shots_per_axis']*x['T']) for m in range(18)]
            costs=[2*x['T']*4/(x['rho_lower']**2*math.sin(x['beta_stat'])**2) for x in selected]
            alpha=[.05*x/sum(costs) for x in costs] # round probability, split equally over X/Y
            scale=math.nextafter(.05/sum(alpha),0.)
            alpha=[x*scale for x in alpha]
            total=0;cx=0
            for row,a in zip(selected,alpha):
                row['alpha_per_axis']=a/2;row['cost_sensitive_shots_per_axis']=shots(row['rho_lower'],row['beta_stat'],a/2)
                total+=2*row['cost_sensitive_shots_per_axis']*row['T'];cx+=2*row['cost_sensitive_shots_per_axis']*row['CX']
            result[name]=dict(total_T_uniform_alpha=sum(2*x['shots_per_axis']*x['T'] for x in selected),
                total_T_cost_sensitive_alpha=total,total_CX_cost_sensitive_alpha=cx,total_shots_cost_sensitive=2*sum(x['cost_sensitive_shots_per_axis'] for x in selected),
                rounds=selected,all_cap_cases=rounds)
            table.extend(dict(arm=name,m=x['m'],cap=x['cap'],T_per_shot=x['T'],CX_per_shot=x['CX'],rho=x['rho_lower'],beta=x['beta_stat'],N_axis=x['cost_sensitive_shots_per_axis']) for x in selected)
    save_json(OUT/'RESULTS.json',result)
    base=result['gate_sum_nonuniform']['total_T_cost_sensitive_alpha'];new=result['structural_nonuniform']['total_T_cost_sensitive_alpha']
    save_json(OUT/'SUMMARY.json',dict(status='CONDITIONAL_COMPOSITIONAL_FIXED_MATRIX_PILOT',baseline_T=base,structural_T=new,
        relative_T_reduction=1-new/base,new_synthesis_keys=sum(not x.get('reused',False) for x in records.values()),synthesis_keys=len(records),
        round_count=18,wall_seconds=time.monotonic()-start,normalizer_added=False,old_STOPs_preserved=True,
        unqualified=['unconditional original-integral representation guarantee','globally optimal discrete allocation','unrestricted compiler/fusion optimum','hardware execution and wall time']))
    keys=list(table[0]);(OUT/'ROUND_TABLE.tsv').write_text('\t'.join(keys)+'\n'+'\n'.join('\t'.join(str(x[k]) for k in keys) for x in table)+'\n')
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',automatic_resume=False,workers_remaining=0))
    print('B STOP T reduction',1-new/base,flush=True)

def run():
    if (OUT/'STOP.json').exists():raise RuntimeError('Stopped pilot; no restart')
    start=time.monotonic();(OUT/'synthesis').mkdir(exist_ok=True)
    basis,centres=inventory();prior=stored('artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json')[1]['rows'];cases=[];tasks=set()
    for old in prior:
        m=old['m'];q=old['q']
        for structural in (False,True):
            items,err,cx=items_for(q,basis,centres,structural)
            # Old SOURCE/compiler disks remain; newly fused angles get their own checked allowance.
            extra=2*math.sqrt(2)*err
            rho0=old['raw_radius_lower'];additional_phase=math.asin(extra/rho0)
            for cap in CAPS:
                disk=rho0*math.sin(cap);budget=F.from_float(math.nextafter(disk/(2*math.sqrt(2)),0.))
                e=exponent(budget,sum((F(x['weight']) for x in items),F(0)))
                tasks.update((x['angle'],k) for x in items for k in (e,e+1))
                cases.append(dict(m=m,q=q,structural=structural,items=items,budget=str(budget),cap=cap,rho_base=rho0,
                    beta_base=old['statistical_phase_lower'],synthesis_raw_disk_cap=disk,additional_raw_compiler_disk=extra,additional_phase=additional_phase,cx=cx))
    save_json(OUT/'PRECISION_REQUESTS.json',dict(cases=cases,keys=[list(t) for t in sorted(tasks)],workers=min(len(tasks),len(os.sched_getaffinity(0)))))
    print('B synthesis keys',len(tasks),flush=True);records={};reused=0
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        futures=[pool.submit(acquire_key,t) for t in sorted(tasks)]
        for future in as_completed(futures):
            angle,e,record,reuse=future.result();record['reused']=reuse;records[(angle,e)]=record;reused+=reuse
            if len(records)%100==0:print('B keys',len(records),'/',len(tasks),'reused',reused,flush=True)
    finish(cases,records,start)

if __name__=='__main__':run()
