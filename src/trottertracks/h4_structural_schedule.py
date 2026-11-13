"""Technical full-schedule adapter. Physical K0/K1 is lazy and separate.

Reads fixed B identities and copies m0/m6. No physical input, law probability,
kernel or mean evaluation during preparation. Residual C-S belongs to D.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from . import h4_structural_kernel as k

BASE='83d219cc68c25540507a5539ebd2da94c29d06b3'
B_LATEST='3216d27a69ac8532d2a22e78f5f94491d202d238'
SAVED='docs/research/h4_structural_kernel_20261113/run02/'
EXPECTED_R=(2,4,8,16,32,64,128,255,510,1020,2040,4080,8160)
EXISTING=(0,6)
REMAINING=tuple(m for m in range(13) if m not in EXISTING)
SOURCE_FILES=k.SOURCE_FILES+('src/trottertracks/h4_structural_schedule.py','scripts/run_h4_structural_schedule.py',
    'src/trottertracks/track_d_native_resources.py','src/trottertracks/s0a_saved_structure.py')
ACTIVE=None


def read(root,commit,path):return k.c.read(root,commit,path)
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def prepare(root):
    """Fixed metadata only; original four S values are copied, never reevaluated."""
    plan,_,binding=k.check_inputs(root)
    latest=read(root,B_LATEST,k.B_INTAKE)
    raw=k.c.blob(root,k.B_CONFIGURATION_SOURCE,k.B_CONFIGURATION_PATH)
    key=k.B_CONFIGURATION_SOURCE+':'+k.B_CONFIGURATION_PATH
    if hashlib.sha256(raw).hexdigest()!=latest['source_files'][key]:raise ValueError('B_FIXED_CONFIGURATION_BYTES')
    config=json.loads(raw)
    if config['id']!=plan['B_V2_configuration_id'] or config['id']!=latest['configuration_id']:raise ValueError('B_CANONICAL_CONFIGURATION')
    rows=config['rounds']
    if [r['m'] for r in rows]!=list(range(13)):raise ValueError('THIRTEEN_ROUNDS_REQUIRED')
    old=read(root,BASE,SAVED+'K1.json');old_k0=read(root,BASE,SAVED+'K0.json');old_stop=read(root,BASE,SAVED+'STOP.json')
    if old_k0['status']!='PASS' or old_stop['status']!='K0_PASS_K1_FOUR_VALUES_COMPLETE_MANDATORY_STOP':raise ValueError('SAVED_EVIDENCE_STATUS')
    requests=[]
    for r in rows:
        c=r['physical_candidate'];m=r['m'];s=r['structure']
        if (c['system'],c['method'],c['formula'],c['prefix'],c['delta'],c['K'],c['q'],c['R_outer'])!=('H4','PR','2nd',1,.1,2,2**m,EXPECTED_R[m]):raise ValueError('FIXED_PHYSICAL_MENU')
        if s['allocation']!=[EXPECTED_R[m]] or s['signed_weights']!=[1.] or s['event_count']!=c['q']*c['R_outer']:raise ValueError('S2_FIXED_ALLOCATION')
        saved=[deepcopy(v) for v in old['values'] if v['m']==m]
        if m in EXISTING:
            if {v['axis'] for v in saved}!={'cosine','sine'}:raise ValueError('SAVED_TWO_AXIS_RECORDS')
            for v in saved:
                if v['identity']!=r['identity'] or v['law_binding']!=r['law_binding'] or v['candidate_id']!=c['id']:raise ValueError('SAVED_PRICE_IDENTITY_LAW')
                if v['quantity']!='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT' or v['mean_compiled_C'] is not None:raise ValueError('STRUCTURAL_QUANTITY_ONLY')
        elif saved:raise ValueError('UNEXPECTED_SAVED_ROUND')
        requests.append(dict(m=m,physical_candidate=deepcopy(c),identity=deepcopy(r['identity']),law_binding=deepcopy(r['law_binding']),
            structure=deepcopy(s),scientific_status='SAVED_K0_K1_PASS' if saved else 'NOT_EVALUATED',
            saved_values=saved,law_probabilities=None,rotation_angles=None,
            mean_S={a:next((v['mean_S'] for v in saved if v['axis']==a),None) for a in ('cosine','sine')},
            numerical_enclosure={a:next((v['numerical_enclosure'] for v in saved if v['axis']==a),None) for a in ('cosine','sine')},
            mean_compiled_C=None,qualified_for_compiled_price=False,
            K0_required=not bool(saved),new_K1_values_required=0 if saved else 2))
    return dict(status='TECHNICAL_PREPARATION_NO_NEW_SCIENCE',configuration_id=config['id'],
        B_source=dict(commit=B_LATEST,path=k.B_INTAKE),configuration_source=binding['configuration_source'],
        exact_kappa=plan['kappa_paper_exact'],M=12,conditions_source=plan['A_source'],
        preparation_hash=old_k0['preparation_hash'],support=dict(nonidentity_components=324,bases=9),
        existing_source=dict(commit=BASE,path=SAVED),existing_rounds=list(EXISTING),remaining_rounds=list(REMAINING),
        requests=requests,new_scientific_computations=0,mean_compiled_C=None,full_rpe_resources=None,mandatory_stop=True)


def multiply_interval(a,b):
    # Nonnegative binary64 products. nextafter makes every operation outward.
    return max(0.,float(np.nextafter(a[0]*b[0],-np.inf))),float(np.nextafter(a[1]*b[1],np.inf))


def positive_power_interval(base,count):
    if type(count) is not int or count<0:raise ValueError('NONNEGATIVE_INTEGER_EXPONENT')
    result=(1.,1.);steps=0
    while count:
        if count&1:result=multiply_interval(result,base);steps+=1
        count>>=1
        if count:base=multiply_interval(base,base);steps+=1
    return result,steps


def probability_enclosure(lo,hi,events):
    """Same gamma16 formula as old K1, with O(log(events)) fixed precision.

    Only the evaluation of an existing rounding enclosure changes, never the
    law, Z arithmetic, mass tolerance, gamma or central mean.
    """
    gamma=Fraction(16,2**53-16)
    lower,_=positive_power_interval(k.fraction_bounds(1-gamma),events)
    upper,_=positive_power_interval(k.fraction_bounds(1+gamma),events)
    low=multiply_interval((lo,lo),lower)[0];high=multiply_interval((hi,hi),upper)[1]
    relative=float(np.nextafter(upper[1]-1,np.inf))
    return low,high,relative


def make_laws(context,requests):
    """Scientific leaf, never invoked by prepare or the default CLI."""
    tail=context.ctx.prep.rte_preparation.symbolic_tail;output=[]
    for r in requests:
        if r['m'] in EXISTING:raise ValueError('SAVED_M0_M6_MUST_NOT_REEXECUTE')
        c=r['physical_candidate'];R=c['R_outer'];time=c['T']/c['q']
        tau,config,d=k.c.make_law(context,tail,time,R)
        output.append(dict(m=r['m'],candidate_id=c['id'],T=c['T'],q_PF=c['q'],R_outer=R,
            event_count=r['structure']['event_count'],tau=tau,tau_hex=tau.hex(),time_per_outer=time,
            orders=list(d.orders),order_weights=list(d.unnormalized_order_weights),
            order_probabilities=list(d.order_probabilities),event_normalization=d.exact_finite_distribution,
            rotation_angles={str(o):math.atan(tau/(o+1)) for o in d.orders},
            runtime_sampler='PCG64_numpy_float64_CDF_saved_cohort',sampler_ideal_distance=None))
    return output


def phase_checks(model,max_R):
    # Extend the same pi accumulation/reset test to 3*max_R, constant cache.
    wire=k.Wire(model,8);theta=0.;checks=0
    for _ in range(3*max_R+1):
        for initial in (None,math.pi/2,model.exit[8]):
            word=[dict(name='rz',wires=[8],angle=theta),dict(name='cx',wires=[8,0],angle=None)]
            end,cost=wire.step_word(word,initial)
            if end is not None or cost!=int((theta if initial is None else initial+theta)!=0.):raise ValueError('PHASE_CONTROL_BARRIER_PARITY')
            checks+=1
        wire.cache.clear();theta+=math.pi
        if not math.isfinite(theta):raise ValueError('PHASE_FINITE_SUM')
    return checks


def wire_job(q,law_index):return q,k.Wire(ACTIVE,q).build(ACTIVE.laws[law_index])


class ScientificBackend:
    """Use one restored preparation/model and existing wire K0/K1 code."""
    def __init__(self,root,source,manifest,out):
        global ACTIVE
        if manifest!=prepare(root):raise ValueError('FIXED_SCHEDULE_METADATA_CHANGED')
        for path in SOURCE_FILES:
            if (Path(root)/path).read_bytes()!=k.c.blob(root,source,path):raise ValueError('FIXED_SOURCE_CHANGED:'+path)
        if float(np.nextafter(0.,1.)*2)==0.:raise ValueError('GRADUAL_UNDERFLOW_REQUIRED')
        from .s0a_saved_structure import saved as c1_saved
        from .s0a_structure_context import Context,no_circuits
        with no_circuits():context=Context(root,c1_saved(root,'PLAN.json'),'H4')
        _,saved,_=k.check_inputs(root)
        if float(context.scalar).hex()!=saved['scalar']['hex']:raise ValueError('SAVED_SCALAR_BINDING')
        self.requests=[v for v in manifest['requests'] if v['K0_required']]
        self.laws=make_laws(context,self.requests)
        write(out/'LAWS.json',dict(source=source,laws=self.laws,physical_kernel_status='NOT_YET_VALIDATED'))
        self.model=k.Model(root,context,dict(saved,laws=self.laws))
        law_parity=k.law_metadata_parity(context,self.model)
        self.phase_checks=phase_checks(self.model,max(v['R_outer'] for v in self.laws))
        write(out/'SHARED_METADATA.json',dict(source=source,preparation_hash=manifest['preparation_hash'],
            laws=self.laws,source_native_branch_parities=self.model.source_parity,
            law_metadata_parity=law_parity,phase_barrier_checks=self.phase_checks,
            physical_K0_for_each_R_still_required=True))
        ACTIVE=self.model;self.workers=min(9,len(os.sched_getaffinity(0)))
        self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('fork'))

    def k0(self,row,out):
        index=next(i for i,v in enumerate(self.laws) if v['m']==row['m']);results={}
        futures=[self.pool.submit(wire_job,q,index) for q in range(9)]
        try:
            for f in as_completed(futures):
                q,value=f.result();results[q]=[value]
        except Exception:
            for f in futures:f.cancel()
            if results:
                partial=out/'partial';partial.mkdir();states,sha=k.save_kernels(partial,results)
                write(partial/'KERNEL_STATES.json',states)
                write(partial/'STATUS.json',dict(status='PARTIAL_NOT_K0_PASS',completed_wires=sorted(results),kernel_sha256=sha))
            raise
        return results,dict(status='PASS',m=row['m'],workers=self.workers,
            source_native_branch_parities=self.model.source_parity,phase_barrier_checks=self.phase_checks,
            wires=[results[q][0]['metadata'] for q in range(9)],
            mass_tolerance=k.MASS_TOLERANCE,K1_started=False)

    def k1(self,row,results):
        law=next(v for v in self.laws if v['m']==row['m'])
        return k.run_K1(results,[law],probability_enclosure=probability_enclosure)

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


def execute(root,source,out,*,backend_factory=ScientificBackend,manifest=None):
    """Future single approved batch, per-round durable PASS before K1."""
    if manifest is None:manifest=prepare(root)
    out=Path(out);out.mkdir(parents=True,exist_ok=False);write(out/'INPUT_METADATA.json',manifest)
    values=[deepcopy(v) for r in manifest['requests'] for v in r['saved_values']]
    write(out/'SAVED_VALUES.json',dict(source=manifest['existing_source'],values=values))
    backend=None;current=None;stage='SHARED_SETUP'
    def coverage(status):
        write(out/'COVERAGE.json',dict(status=status,configuration_id=manifest['configuration_id'],
            values=values,missing_rounds=[m for m in range(13) if not any(v['m']==m for v in values)],
            mean_compiled_C=None,full_rpe_resources=None,qualified_for_compiled_price=False))
    try:
        backend=backend_factory(root,source,manifest,out)
        for row in manifest['requests']:
            if row['m'] in EXISTING:continue
            current=out/('m'+str(row['m']));current.mkdir()
            stage='K0'
            results,summary=backend.k0(row,current);summary['source']=source
            if summary['status']!='PASS':
                write(current/'K0.json',summary);coverage('PARTIAL_K0_'+summary['status'])
                write(out/'STOP.json',dict(status='K0_'+summary['status']+'_MANDATORY_STOP',m=row['m'],source=source,K1_for_failed_round=False));return
            states,sha=k.save_kernels(current,results);summary['kernel_sha256']=sha
            write(current/'KERNEL_STATES.json',states);write(current/'K0.json',summary)
            stage='K1'
            new=backend.k1(row,results)
            if {(v['m'],v['axis']) for v in new}!={(row['m'],'cosine'),(row['m'],'sine')}:raise ValueError('TWO_STRUCTURAL_AXES_REQUIRED')
            for v in new:
                v['identity']=row['identity'];v['law_binding']=row['law_binding'];v['kernel_source']=source
                if v['quantity']!='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT' or v['mean_compiled_C'] is not None:raise ValueError('STRUCTURAL_QUANTITY_ONLY')
            write(current/'K1.json',dict(status='K0_PASS_TWO_STRUCTURAL_MEANS',values=new,source=source));values.extend(new)
        coverage('ALL_THIRTEEN_ROUNDS_STRUCTURAL_MEAN_RECORDS_AVAILABLE')
        write(out/'STOP.json',dict(status='REMAINING_ELEVEN_COMPLETE_MANDATORY_STOP',source=source,new_mean_values=22,saved_values=4,automatic_follow_on=False))
    except Exception as e:
        label=str(e);status='FAIL' if any(t in label for t in ('PARITY','ENCLOSURE_FAILED','STATE_BOUND_FAILED','EXIT_DEPENDS','QUEUE')) else 'UNKNOWN'
        if current is not None and not (current/'K0.json').exists():write(current/'K0.json',dict(status=status,error=type(e).__name__+': '+label,K1_started=False,source=source))
        coverage('PARTIAL_'+status)
        write(out/'STOP.json',dict(status=status+'_MANDATORY_STOP',source=source,stage=stage,error=type(e).__name__+': '+label,automatic_follow_on=False))
        raise
    finally:
        if backend is not None:backend.close()
