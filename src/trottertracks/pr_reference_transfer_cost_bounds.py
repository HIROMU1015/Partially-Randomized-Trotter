"""CX-only structural bounds; depth is kept as an actual-DAG measurement."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import json,math
from pathlib import Path
import numpy as np
from .pr_reference_transfer_qualification import OUT,OLD,data,common,read_prior_prices,save_json,METRICS
from . import pr_reference_interference_pilot as old
from .track_c_h4_finite_bits import integer_cdf


def catalog_bounds():
    rows=[json.loads(p.read_text()) for p in (OUT/'catalogue_boundary').glob('*.json')]
    assert len(rows)==12
    values={'Simon_Love':defaultdict(list),'reference':defaultdict(list)}
    unsafe=[]
    for row in rows:
        arm=row['request'][0]
        for rec,pre in zip(row['records'],row['pre_last_cancel']):
            values[arm][rec['name']].append(rec['cx'])
            if arm=='Simon_Love':
                selected=rec['meta']['kind'] in ('basis','basis_inverse','baseline_D') or rec['meta'].get('family') in ('crz','crzz','cz','h','sdg')
                if selected and pre['exposed_cancelable_prefix']:unsafe.append([row['key'],rec['name'],pre])
    bounds={a:{k:[min(v),max(v)] for k,v in d.items()} for a,d in values.items()}
    assert not unsafe,unsafe[:1]
    # Registered numeric variants have these CX costs in every reachable context.
    expected={'xx_plus_yy':2,'rzz':2,'crz':2,'crzz':14,'cz':1,'cswap':8,'h':0,'sdg':0,'z':0}
    for row in rows:
        for rec in row['records']:
            family=rec['meta'].get('family')
            if family in expected:assert rec['cx']==expected[family],(row['key'],rec['name'])
    return rows,bounds


def parameters():
    rows,b=catalog_bounds();comp={c.component_id:c for c in common()['preps'][3].rte_preparation.symbolic_tail.components}
    # Fences isolate the baseline chunks. Native post-translation X/CX removal
    # cannot start a cross-chunk cancellation if no first-exposed X/CX exists.
    fixed={k:b['Simon_Love'][k][0] for k in ('left_all','left_no0','right_all','right_no0','boundary_control')}
    assert all(b['Simon_Love'][k][0]==b['Simon_Love'][k][1] for k in fixed)
    basis={}
    for cid,c in comp.items():
        k=c.basis_id
        pair=sum(b['Simon_Love'][prefix+k][0] for prefix in ('basis:','basis_inverse:'))
        assert pair==sum(b['reference'][prefix+k][1] for prefix in ('basis:','basis_inverse:'))
        basis[k]=pair
    ref={k:b['reference'][f'block:{k}:half'][1] for k in range(4)}
    ref_full=b['reference']['block:0:full'][1]
    return fixed,basis,ref,ref_full,comp


def d_cost(q,fixed,ref,ref_full):
    base=fixed['left_all']+fixed['right_all']+(q-1)*(fixed['left_no0']+fixed['right_no0']+fixed['boundary_control'])
    candidate=(2*sum(ref.values()) if q==1 else 4*ref[0]+(q-2)*ref_full+2*q*sum(ref[k] for k in (1,2,3)))+16*8
    return base,candidate


def path_prices(q,r,events,params):
    fixed,basis,ref,full,comp=params
    base,cand=d_cost(q,fixed,ref,full)
    for i in range(q):
        last=None
        for e in events[i*r:(i+1)*r]:
            rot=e.get('rotation',e.get('rotation_component_id'))
            prod=e.get('products',e.get('product_component_ids'))
            for cid in [*prod,rot]:
                b=comp[cid].basis_id
                if b!=last:base+=basis[b];cand+=basis[b];last=b
            support=len(comp[rot].diagonal_pauli_support)
            base+=2 if support==1 else 14
            cand+=0 if support==1 else 2
            base+=sum(len(comp[c].diagonal_pauli_support) for c in prod)
    return base,cand


def distribution(values,bits=None):
    if bits is not None:
        cdf=integer_cdf(values,bits)
        return [F(b-a,1<<bits) for a,b in zip((0,*cdf[:-1]),cdf)]
    values=list(map(F.from_float,map(float,values)));s=sum(values)
    return [x/s for x in values]


def expected(q,r,params,bits=None):
    fixed,basis,ref,full,comp=params
    components=list(common()['preps'][3].rte_preparation.symbolic_tail.components)
    p=distribution([c.probability for c in components],bits)
    dist=old.finite_rte_distribution(common()['preps'][3].exact_rte_lambda_r*.02/r,2)
    p2=distribution(dist.order_probabilities,bits)[1]
    p_basis=defaultdict(F)
    for c,pi in zip(components,p):p_basis[c.basis_id]+=pi
    applications=F(r)*(1+2*p2)
    bcost=q*sum(F(basis[b])*(pb+(applications-1)*pb*(1-pb)) for b,pb in p_basis.items())
    rot_base=sum(pi*(2 if len(c.diagonal_pauli_support)==1 else 14) for c,pi in zip(components,p))
    rot_ref=sum(pi*(0 if len(c.diagonal_pauli_support)==1 else 2) for c,pi in zip(components,p))
    product=sum(pi*len(c.diagonal_pauli_support) for c,pi in zip(components,p))
    dbase,dref=d_cost(q,fixed,ref,full)
    base=F(dbase)+bcost+q*r*(rot_base+2*p2*product)
    candidate=F(dref)+bcost+q*r*rot_ref
    return base,candidate


def qualify():
    params=parameters();catalog,bounds=catalog_bounds()
    raw_old=read_prior_prices()
    raw_new=[json.loads(p.read_text()) for p in (OUT/'raw').glob('*.json')]
    checks=[]
    for x in raw_old+raw_new:
        q=x['q'];r=x.get('r',1);events=x.get('events',x.get('event_records'))
        base,upper=path_prices(q,r,events,params)
        for axis in range(2):
            actual_b=x['values']['Simon_Love'][axis][1]
            actual_r=x['values']['reference'][axis][1]
            checks.append(dict(q=q,r=r,rare=x['rare'],j=x['j'],axis='XY'[axis],
                baseline_predicted=base,baseline_actual=actual_b,candidate_upper=upper,
                candidate_actual=actual_r,baseline_equal=base==actual_b,candidate_included=actual_r<=upper))
    contexts={tuple(c['post_unroll']) for c in catalog}
    context_checks=[]
    for x in raw_new:
        for arm,axes in x['traces'].items():
            for axis,trace in axes.items():
                context=next(t['operations'] for t in trace if t['pass_name']=='UnrollCustomDefinitions')
                key=tuple(sorted(set(context)-{'measure','barrier'}))
                context_checks.append(dict(q=x['q'],r=x['r'],arm=arm,axis=axis,covered=key in contexts))
    passed=all(x['baseline_equal'] and x['candidate_included'] for x in checks) and all(x['covered'] for x in context_checks)
    result=dict(status='CX_PRICE_TRANSFER_QUALIFIED' if passed else 'UNDETERMINED_CX_MODEL_MISMATCH',
        checks=checks,contexts=context_checks,registered_context_count=12,
        prefix_fences='all baseline Gaussian frame, deterministic chunk and CRZ/CRZZ/CZ prefixes contain no exposed native X/CX before last cancellation',
        baseline_CX='exact under fixed definition-based compiler-A context rules and canonical family',
        reference_CX='upper bound before optional cross-rotation cancellations; not an actual expected price',
        source_structure='same fixed whole-trajectory split and canonical finite events; no placement or law changes',
        basis_run_expectation='sum_b c_b [p_b+(r*(1+2*p_order2)-1)*p_b*(1-p_b)] per RTE occurrence',
        all_round_other_metrics='UNDETERMINED; full native DAG depth is measured only on held-outs',
        T=None,pass_to_stage3=bool(passed))
    save_json(OUT/'STAGE2_CX_CERTIFICATE.json',result)
    return result
