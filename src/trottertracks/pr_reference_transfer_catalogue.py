"""Finite local compiler contexts for CX transfer and cancellation interfaces."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib,json,os
from types import SimpleNamespace
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZZGate,RZGate,CRZGate,CXGate,HGate,SdgGate,ZGate,CZGate,CSwapGate,PhaseGate
from . import pr_reference_interference_pilot as old
from .pr_reference_transfer_qualification import OUT,common,save_json
from .track_c_h4 import diagonal_primitives
from .track_c_compiler_a import compile_reference
from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder


def fence(instructions,n):
    first={};last={}
    for j,(name,wires,_) in enumerate(instructions):
        for q in wires:
            first.setdefault(q,(j,name));last[q]=(j,name)
    def exposed(end):
        nodes={v[0] for v in end.values()};out=[]
        for j in sorted(nodes):
            name,qs,_=instructions[j]
            if name in ('cx','x') and all(end[q][0]==j for q in qs):out.append([name,qs])
        return out
    return dict(first_names={q:v[1] for q,v in first.items()},last_names={q:v[1] for q,v in last.items()},
        exposed_cancelable_prefix=exposed(first),exposed_cancelable_suffix=exposed(last))


@lru_cache(None)
def chunks():
    c=common();p=c['preps'][3];builder=QiskitDFPartialS2CircuitBuilder();out=[]
    definitions={('D'+str(k)):b.runtime_basis_operations for k,b in enumerate(p.deterministic_blocks)}
    for bid,d in p.rte_preparation.basis_registry._definitions.items():definitions[bid]=d.runtime_operations
    local={}
    for bid,ops in definitions.items():
        q=QuantumCircuit(8)
        for g,qs in ops:
            q.append(g,list(qs))
            if g.name=='xx_plus_yy':
                for op in (g,g.inverse()):
                    key='xx_plus_yy:'+','.join(float(v).hex() for v in op.params)
                    local[key]=(op,list(range(2)))
        out.extend([(f'basis:{bid}',q,dict(kind='basis',id=bid)),(f'basis_inverse:{bid}',q.inverse(),dict(kind='basis_inverse',id=bid))])
    def block(k,t,ctrl=False):
        q=QuantumCircuit(9 if ctrl else 8);req=SimpleNamespace(preparation=p,controlled=ctrl,ancilla_qubit=8 if ctrl else None,step_time=2*t)
        builder._append_block(q,p.deterministic_blocks[k],req,evolution_time=t);return q
    for k in range(4):
        out.append((f'block:{k}:half',block(k,.01),dict(kind='D_half',block=k)))
    out.append(('block:0:full',block(0,.02),dict(kind='D_full',block=0)))
    for name,indices in [('left_all',range(4)),('left_no0',range(1,4))]:
        q=QuantumCircuit(8)
        for k in indices:q.compose(block(k,.01),inplace=True)
        out.append((name,q,dict(kind='baseline_D')))
    out.extend([('right_all',old.directional_right(.01),dict(kind='baseline_D')),
        ('right_no0',old.directional_right(.01,False),dict(kind='baseline_D')),
        ('boundary_control',block(0,.02,True),dict(kind='baseline_D'))])
    for key,(g,wires) in sorted(local.items()):
        q=QuantumCircuit(2);q.append(g,wires)
        out.append((key,q,dict(kind='local',family='xx_plus_yy',params=list(map(float,g.params)))))
    angles=[]
    for r in (1,2,4,8,16,32):
        for order in (0,2):
            for sign in (-1,1):angles.append(sign*2*np.arctan(p.exact_rte_lambda_r*.02/(r*(order+1))))
    for t in (.01,-.01):
        for b in p.deterministic_blocks:
            prim=diagonal_primitives(b,t)
            for _,_,a in prim.rzz:angles.append(float(a))
    for family in ('rzz','crz','crzz'):
        for a in sorted(set(angles)):
            g=RZZGate(a) if family=='rzz' else CRZGate(a) if family=='crz' else RZZGate(a).control()
            q=QuantumCircuit(g.num_qubits);q.append(g,range(g.num_qubits))
            out.append((family+':'+a.hex(),q,dict(kind='local',family=family,params=[a])))
    for family,g in [('cz',CZGate()),('z',ZGate()),('cswap',CSwapGate()),('h',HGate()),('sdg',SdgGate())]:
        q=QuantumCircuit(g.num_qubits);q.append(g,range(g.num_qubits))
        out.append((family,q,dict(kind='local',family=family,params=[])))
    return out


def catalog_one(request):
    arm,ccx,rare,y=request
    n=9;q=QuantumCircuit(n);tags=[]
    selected=[]
    for name,c,meta in chunks():
        if arm=='reference' and (meta.get('family') in ('crz','crzz','cz') or meta['kind']=='baseline_D'):continue
        if arm=='Simon_Love' and meta.get('family') in ('cswap','z'):continue
        if arm=='Simon_Love' and not ccx and meta.get('family')=='crzz':continue
        if not rare and meta.get('family') in ('cz','z'):continue
        if not y and meta.get('family')=='sdg':continue
        selected.append((name,c,meta))
        q.compose(c,qubits=range(c.num_qubits),inplace=True);q.barrier();tags.append((name,c.num_qubits,meta))
    # Keep exactly the reachable post-unroll context present in the whole arm.
    q.append(CRZGate(.123),[0,1]) if arm=='Simon_Love' else q.cswap(0,1,2)
    q.barrier();tags.append(('context_witness',3,dict(kind='witness')))
    alpha=Parameter('symbolic_correction')
    q.p(alpha,0);q.barrier();tags.append(('p_symbolic',1,dict(kind='symbolic')))
    trace={}
    def cb(**kw):
        if type(kw['pass_']).__name__=='UnrollCustomDefinitions':trace['post_unroll']=sorted(set(kw['dag'].count_ops())-{'barrier'})
        if type(kw['pass_']).__name__=='BasisTranslator':
            slices=[];current=[];dag=kw['dag']
            for node in dag.topological_op_nodes():
                if node.op.name=='barrier':slices.append(current);current=[]
                else:current.append([node.op.name,[dag.find_bit(z).index for z in node.qargs],[]])
            assert len(slices)==len(tags) and not current
            trace['pre_last_cancel']=[fence(s,width) for s,(_,width,_) in zip(slices,tags)]
    native=compile_reference(q,callback=cb)
    slices=[];current=[]
    for i in native.data:
        if i.operation.name=='barrier':slices.append(current);current=[]
        else:
            params=[float(v) if not getattr(v,'parameters',None) else str(v) for v in i.operation.params]
            current.append([i.operation.name,[native.find_bit(z).index for z in i.qubits],params])
    assert len(slices)==len(tags) and not current
    records=[]
    for (name,width,meta),insts in zip(tags,slices):
        count={}
        for f,_,_ in insts:count[f]=count.get(f,0)+1
        if name=='p_symbolic':assert len(insts)==1 and insts[0][0]=='rz' and insts[0][2]==['symbolic_correction']
        records.append(dict(name=name,n=width,meta=meta,cx=count.get('cx',0),ops=count,
            fence=fence(insts,width),lowering=insts))
    key=f'{arm}_ccx{int(ccx)}_rare{int(rare)}_Y{int(y)}'
    result=dict(key=key,request=request,post_unroll=trace['post_unroll'],records=records,
        pre_last_cancel=trace['pre_last_cancel'])
    save_json(OUT/'catalogue_boundary'/f'{key}.json',result)
    return key


def run():
    requests=[('Simon_Love',c,z,y) for c in (False,True) for z in (False,True) for y in (False,True)]
    requests += [('reference',True,z,y) for z in (False,True) for y in (False,True)]
    chunks()
    with ProcessPoolExecutor(max_workers=min(len(requests),len(os.sched_getaffinity(0)))) as pool:
        for x in as_completed([pool.submit(catalog_one,r) for r in requests]):print('catalogued',x.result(),flush=True)
