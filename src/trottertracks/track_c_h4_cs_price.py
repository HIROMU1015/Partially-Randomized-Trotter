"""Same-position residual traces and hierarchical Clifford+T prices."""
from __future__ import annotations
from functools import partial,lru_cache
import hashlib
import json
import math
import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import RZGate,RZZGate,PhaseGate
from .track_c_h4 import *
from .track_c_h4_price import (conditional_flags,trajectory_parts,wrapper,native_compile,
                                d_part,QiskitDFRTEEventCircuitBuilder)
from .track_c_h4_synthesis import t_metrics


def rotation(circuit,theta,support,placeholder=False):
    if placeholder:circuit.append(Gate('trackc_res',len(support)+1,[float(theta)]),[8,*support])
    else:
        gate=RZGate(theta) if len(support)==1 else RZZGate(theta)
        circuit.append(gate.control(1),[8,*support])


@lru_cache(None)
def cs_d_part(arm,time,reverse):
    c=common();qc=QuantumCircuit(9);prep=c['preps'][3]
    builder=QiskitDFPartialS2CircuitBuilder()
    order=reversed(prep.deterministic_blocks) if reverse else prep.deterministic_blocks
    for block in order:
        builder._append_basis(qc,block,inverse=True)
        prim=diagonal_primitives(block,time)
        for a,b,theta in prim.rzz:
            if abs(theta)>=1e-12:rotation(qc,theta,(a,b),arm!='S0')
        for q,theta in prim.rz:rotation(qc,theta,(q,),arm!='S0')
        if abs(prim.global_phase)>=1e-12:qc.append(PhaseGate(prim.global_phase),[8])
        builder._append_basis(qc,block,inverse=False)
    return qc


class ResidualTailBuilder(QiskitDFRTEEventCircuitBuilder):
    @staticmethod
    def _append_central_application(circuit,event,application,*,controlled,ancilla_qubit):
        if application.role=='rotation':
            assert controlled and ancilla_qubit==8
            rotation(circuit,2*event.unsigned_rotation_angle,application.diagonal_pauli_support,True)
            return 0.
        return QiskitDFRTEEventCircuitBuilder._append_central_application(
            circuit,event,application,controlled=controlled,ancilla_qubit=ancilla_qubit)


def residual_realization(qc,rres,rare,rng):
    probabilities=[];inventory=[];logb=0.
    for inst in qc.data:
        if inst.operation.name=='trackc_res':
            theta=float(inst.operation.params[0]);coarse=round(theta/(math.pi/4))*math.pi/4
            x=(theta-coarse)/(2*rres);dist=finite_rte_distribution(x,2)
            probabilities.extend([dist.order_probabilities[1]]*rres)
            inventory.append((theta,coarse,x));logb+=rres*math.log(dist.exact_finite_distribution)
    mass=-math.expm1(math.fsum(math.log1p(-p) for p in probabilities))
    if rare and mass==0:return None,0.,logb
    flags=conditional_flags(probabilities,rare,rng) if probabilities else []
    out=QuantumCircuit(9,1);cursor=index=0
    for inst in qc.data:
        wires=[qc.find_bit(q).index for q in inst.qubits]
        clbits=[qc.find_bit(b).index for b in inst.clbits]
        if inst.operation.name!='trackc_res':out.append(inst.operation,wires,clbits);continue
        theta,coarse,x=inventory[index];index+=1
        blockflags=flags[cursor:cursor+rres];cursor+=rres
        count=int(np.sum(blockflags))
        # All residual factors have the same involution and circuit location.
        # Fuse them exactly, retaining (-1)^(order/2) as a control phase.
        effective=coarse+2*((rres-count)*math.atan(x)+count*math.atan(x/3))
        support=tuple(wires[1:])
        rotation(out,effective,support)
        if count%2:out.z(8)
    out.global_phase=qc.global_phase
    return out,mass if rare else 1-mass,logb


@lru_cache(None)
def structural_counts():
    c=common();prep=c['preps'][3]
    half=transpile(d_part(3,.01),basis_gates=['rz','sx','x','cx'],optimization_level=0,num_processes=1)
    halfcount=int(half.count_ops().get('rz',0))
    maximum=0
    for definition in prep.rte_preparation.basis_registry._definitions.values():
        qc=QuantumCircuit(8)
        for gate,wires in definition.runtime_operations:qc.append(gate,list(wires))
        t=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=0,num_processes=1)
        maximum=max(maximum,int(t.count_ops().get('rz',0)))
    return halfcount,maximum


def rz_bound(q,r,arm):
    half,basis=structural_counts()
    # K2 has <=3 applications/event, two basis changes/application, and
    # <=8 RZ for a controlled Z/ZZ product; phase/readout overhead is explicit.
    literal=q*(2*half+r*3*(2*basis+8)+4)+8
    return literal*(2 if arm!='S0' else 1)


def cs_precision(data):
    task=data['task'];requirements=[]
    for row in data['rows']:
        requirements.append((math.log(row['synthesis_disk_cap'])-math.log(2*math.sqrt(2))
                             -row['log_range']-math.log(rz_bound(row['q'],row['r'],task['arm'])))/math.log(10))
    return math.floor(min(requirements))


def cs_price_task(request):
    task=request['task'];q=request['q'];r=request['r'];rare=request['rare'];rr=request['r_res']
    exponent=request['exponent'];key=request['key'];path=OUT/'prices_T'/f'{key}.json'
    if path.exists():return key,'saved'
    rows=[];outermass=None
    for j in range(request['n']):
        seed=int.from_bytes(hashlib.sha256(f'CS_shared:{r}:{q}:{rare}:{j}'.encode()).digest()[:8],'big')
        origin=dict(pilot='C-N',formula='S2',h=.02,ld=3,name='CS_original')
        parts,_,outermass=trajectory_parts(origin,q,r,rare,seed,
            d_override=partial(cs_d_part,task['arm']),
            tail_override=ResidualTailBuilder if task['arm']=='S1' else None,price_additive=False)
        sample=[]
        for axis in ('X','Y'):
            qc=wrapper(parts,axis)
            pairs=[]
            for inner in ((False,) if task['arm']=='S0' else (False,True)):
                if task['arm']=='S0':realized=qc;mass=1.;logb=0.
                else:realized,mass,logb=residual_realization(qc,rr,inner,np.random.default_rng(seed+int(inner)+1))
                if realized is None:continue
                native=native_compile(realized)
                count=int(native.count_ops().get('rz',0));bound=rz_bound(q,r,task['arm'])
                if count>bound:raise ArithmeticError('native RZ count exceeds structural precision bound')
                result=t_metrics(native,exponent)
                result.update(probability_given_original=mass,log_residual_normalizer=logb,
                              native_RZ_before_synthesis=count,native_RZ_bound=bound)
                pairs.append(result)
            sample.append(pairs)
        rows.append(sample)
    means=[]
    for axis in range(2):
        means.append({metric:float(np.mean([sum(p['probability_given_original']*p[metric]
                              for p in row[axis]) for row in rows]))
                      for metric in ['T','Tdagger','CX','Clifford','depth']})
    save_json(path,dict(request=request,samples=rows,means=means,
          outer_probability=outermass if rare else 1-outermass,status='COMPLETE',
          estimator='hierarchical original stratum / conditional residual strata',
          profile='full wrapper Q opt2 then explicit strict Clifford+T lowering'))
    return key,'complete'


def cs_price_tasks():
    tasks=[];seen=set()
    for task in cs_tasks():
        data=json.loads((OUT/'signals'/f"{task['name']}.json").read_text())
        if not data['all_rounds_feasible']:continue
        exponent=cs_precision(data)
        specs=sorted(set((row['r'],row['r_res']) for row in data['rows']))
        for r,rr in specs:
            for q,n in ((1,32),(2,32),(8,128)):
                for rare in (False,True):
                    key=f"CS_{task['arm']}_eps{exponent}_r{r}_res{rr}_q{q}_rare{int(rare)}"
                    if key in seen:continue
                    seen.add(key);tasks.append(dict(task=task,exponent=exponent,r=r,r_res=rr,q=q,rare=rare,n=n,key=key))
    return tasks
