"""Conditional all18 resource bounds and new sufficient integer shot schedules."""
from __future__ import annotations
from fractions import Fraction as F
from collections import defaultdict
import json,math
import mpmath as mp
from .pr_reference_transfer_qualification import OUT,common,save_json,data,OLD
from .pr_reference_transfer_cost_bounds import parameters,expected,distribution,catalog_bounds
from .pr_reference_transfer_error import ROUND_PATH
from . import pr_reference_interference_pilot as old
from . import track_c_compiler_a_18round as iv


def rz_parameters():
    catalog,_=catalog_bounds()
    values={a:defaultdict(list) for a in ('Simon_Love','reference')}
    families={a:defaultdict(list) for a in values}
    for context in catalog:
        arm=context['request'][0]
        for r in context['records']:
            count=r['ops'].get('rz',0);values[arm][r['name']].append(count)
            if r['meta']['kind']=='local':families[arm][r['meta']['family']].append(count)
    bounds={a:{k:[min(x),max(x)] for k,x in records.items()} for a,records in values.items()}
    fbound={a:{k:[min(x),max(x)] for k,x in records.items()} for a,records in families.items()}
    return bounds,fbound


def expected_rz_bounds(q,r,bits,axis):
    bounds,family=rz_parameters()
    components=list(common()['preps'][3].rte_preparation.symbolic_tail.components)
    p=distribution([c.probability for c in components],bits)
    dist=old.finite_rte_distribution(common()['preps'][3].exact_rte_lambda_r*.02/r,2)
    p2=distribution(dist.order_probabilities,bits)[1]
    pb=defaultdict(F)
    for c,pi in zip(components,p):pb[c.basis_id]+=pi
    apps=r*(1+2*p2);results={}
    for arm in bounds:
        ends=[]
        for end in (0,1):
            b=bounds[arm];f=family[arm]
            if arm=='Simon_Love':
                D=b['left_all'][end]+b['right_all'][end]+(q-1)*(b['left_no0'][end]+b['right_no0'][end]+b['boundary_control'][end])
            else:
                C=lambda k:b[f'block:{k}:half'][end]
                D=2*sum(C(k) for k in range(4)) if q==1 else 4*C(0)+(q-2)*b['block:0:full'][end]+2*q*sum(C(k) for k in (1,2,3))
                D+=16*f['cswap'][end]
            bc=q*sum((b['basis:'+name][end]+b['basis_inverse:'+name][end])*(pi+(apps-1)*pi*(1-pi)) for name,pi in pb.items())
            if arm=='Simon_Love':
                rotation=sum(pi*f['crz' if len(c.diagonal_pauli_support)==1 else 'crzz'][end] for c,pi in zip(components,p))
                product=sum(pi*len(c.diagonal_pauli_support)*f['cz'][end] for c,pi in zip(components,p))
                extra_phase=q*(1-(1-p2)**r) # Controlled builder appends P iff >=1 order2.
            else:
                rotation=F(1) # RZ or RZZ, exactly one native RZ.
                product=sum(pi*len(c.diagonal_pauli_support)*f['z'][end] for c,pi in zip(components,p))
                extra_phase=0
            wrapper=2*f['h'][end]+(f['sdg'][end] if axis=='Y' else 0)+1
            ends.append(F(D)+bc+q*r*(rotation+2*p2*product)+extra_phase+wrapper)
        results[arm]=ends
    # Source Z cancellation can only reduce the reference RZ count.
    return results['Simon_Love'],results['reference'][1]


def I(x=0):return iv.I(x)
def fraction_iv(x):return I(x.numerator)/x.denominator
def up(x):return iv.upper(x)
def low(x):return iv.lower(x)


def validate_rz_paths():
    from .pr_reference_transfer_qualification import read_prior_prices
    bounds,family=rz_parameters();comp={c.component_id:c for c in common()['preps'][3].rte_preparation.symbolic_tail.components}
    samples=read_prior_prices()+[json.loads(p.read_text()) for p in (OUT/'raw').glob('*.json')]
    checks=[]
    for sample in samples:
        q=sample['q'];r=sample.get('r',1);ev=sample.get('events',sample.get('event_records'))
        for axis_index,axis in enumerate(('X','Y')):
            ranges={}
            for arm in bounds:
                costs=[]
                for end in (0,1):
                    b=bounds[arm];f=family[arm]
                    if arm=='Simon_Love':
                        price=b['left_all'][end]+b['right_all'][end]+(q-1)*(b['left_no0'][end]+b['right_no0'][end]+b['boundary_control'][end])
                    else:
                        C=lambda k:b[f'block:{k}:half'][end]
                        price=2*sum(C(k) for k in range(4)) if q==1 else 4*C(0)+(q-2)*b['block:0:full'][end]+2*q*sum(C(k) for k in (1,2,3))
                        price+=16*f['cswap'][end]
                    price+=2*f['h'][end]+(f['sdg'][end] if axis=='Y' else 0)+1
                    for i in range(q):
                        last=None;phase=False
                        for e in ev[i*r:(i+1)*r]:
                            rot=e.get('rotation',e.get('rotation_component_id'));prod=e.get('products',e.get('product_component_ids'))
                            phase=phase or e.get('order',e.get('taylor_order'))>0
                            for cid in [*prod,rot]:
                                bid=comp[cid].basis_id
                                if bid!=last:price+=b['basis:'+bid][end]+b['basis_inverse:'+bid][end];last=bid
                            price+=f['crz' if len(comp[rot].diagonal_pauli_support)==1 else 'crzz'][end] if arm=='Simon_Love' else 1
                            price+=sum(len(comp[c].diagonal_pauli_support) for c in prod)*f['cz' if arm=='Simon_Love' else 'z'][end]
                        if arm=='Simon_Love' and phase:price+=1
                    costs.append(price)
                ranges[arm]=costs
            bactual=sample['values']['Simon_Love'][axis_index][0];cactual=sample['values']['reference'][axis_index][0]
            checks.append(dict(q=q,r=r,rare=sample['rare'],j=sample['j'],axis=axis,
                baseline_actual=bactual,baseline_range=ranges['Simon_Love'],reference_actual=cactual,
                reference_upper=ranges['reference'][1],passed=ranges['Simon_Love'][0]<=bactual<=ranges['Simon_Love'][1] and cactual<=ranges['reference'][1]))
    save_json(OUT/'RZ_PATH_BOUND_CHECKS.json',dict(checks=checks,all_passed=all(x['passed'] for x in checks)))
    assert all(x['passed'] for x in checks),next(x for x in checks if not x['passed'])


def account():
    proof=json.loads((OUT/'STAGE3_ALL_SUPPORT.json').read_text())
    assert proof['pass_to_stage4']
    validate_rz_paths()
    oldcase=next(x for x in json.loads(data(ROUND_PATH)) if x['name']=='CN_S2_h0.02_ld3')
    alias=json.loads(data('artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/GROUND_ALIAS.json'))
    assert alias['all_spectrum_endpoint_margin_pass']
    mp.iv.prec=192;params=parameters();rows=[]
    totals={a:dict(shots=0,CX_lower=I(0),CX_upper=I(0),RZ_lower=I(0),RZ_upper=I(0)) for a in ('Simon_Love','reference')}
    for row,oldrow in zip(proof['rows'],oldcase['rows']):
        q=row['q'];r=row['r'];bits=oldrow['finite_bit_proposal']['bits_per_uniform_draw']
        bcx,rcx=expected(q,r,params,bits)
        price={}
        for axis in ('X','Y'):
            brz,rrz=expected_rz_bounds(q,r,bits,axis)
            price[axis]=dict(baseline_CX_exact=str(bcx),reference_CX_upper=str(rcx),
                baseline_RZ_lower=str(brz[0]),baseline_RZ_upper=str(brz[1]),reference_RZ_upper=str(rrz))
        arms={}
        for arm,a in row['arms'].items():
            rho=I(a['raw_radius_lower']);beta=I(a['statistical_phase_lower'])
            epsilon=rho*mp.iv.sin(beta)/mp.iv.sqrt(I(2))
            bound=2*mp.iv.log(I(1440))/epsilon**2
            with mp.workprec(192):n=int(mp.ceil(mp.mpf(bound._mpi_[1])))
            sums=totals[arm];sums['shots']+=2*n
            c=fraction_iv(bcx if arm=='Simon_Love' else rcx)
            if arm=='Simon_Love':sums['CX_lower']+=2*n*c
            sums['CX_upper']+=2*n*c
            for axis in ('X','Y'):
                p=price[axis]
                if arm=='Simon_Love':
                    sums['RZ_lower']+=n*fraction_iv(F(p['baseline_RZ_lower']))
                    sums['RZ_upper']+=n*fraction_iv(F(p['baseline_RZ_upper']))
                else:sums['RZ_upper']+=n*fraction_iv(F(p['reference_RZ_upper']))
            arms[arm]=dict(shots_per_axis=n,raw_radius_lower=a['raw_radius_lower'],
                total_raw_disk_upper=a['total_raw_disk_upper'],statistical_phase_lower=a['statistical_phase_lower'],
                epsilon_coordinate_lower=low(epsilon),success_conditions_closed=True,
                alpha_axis='1/720',CX_price_kind='exact' if arm=='Simon_Love' else 'upper bound')
        rows.append(dict(m=row['m'],q=q,r=r,bits=bits,arms=arms,price_per_axis=price))
    outtotals={}
    for arm,t in totals.items():
        outtotals[arm]=dict(total_shots=t['shots'],
            total_CX_lower=low(t['CX_lower']) if arm=='Simon_Love' else None,
            total_CX_upper=up(t['CX_upper']),
            total_RZ_lower=low(t['RZ_lower']) if arm=='Simon_Love' else None,
            total_RZ_upper=up(t['RZ_upper']),total_depth=None,total_size=None,total_T=None)
    cxpass=outtotals['reference']['total_CX_upper']<outtotals['Simon_Love']['total_CX_lower']
    rzpass=outtotals['reference']['total_RZ_upper']<outtotals['Simon_Love']['total_RZ_lower']
    result=dict(status='CONDITIONAL_ALL18_CX_RZ_REDUCTION' if cxpass and rzpass else 'NO_CERTIFIED_COMMON_RESOURCE_REDUCTION',
        CX_reduction_certified=cxpass,RZ_reduction_certified=rzpass,totals=outtotals,rows=rows,
        old_8598_shots_used_as_new_sufficient=False,
        old_saved_ideal_schedule_total=oldcase['total_shots'],
        full_failure_probability='36*(1/720)=0.05',energy_accuracy_Ha=1.5936e-4,phase_budget='.4 rad',
        alias_inherited='same fixed saved-Hermitian-H prior, q schedule and .4-rad cap; endpoint margins pass',
        quantum_sampling_law='old finite-bit floor CDF, exact rational order/component probabilities; fresh independent bits; no IS',
        classical_price_MC_sampler='new paired NumPy strata are holdouts, not the cost-expectation certificate',
        price_bounds='reference UPPER versus baseline LOWER; bounds are not feasible baseline prices or sample means',
        depth_and_size='all18 native critical-path depth/size not qualified; no affine fill or component-depth sum',
        reference_extra_qubits=8,reference_qubits=17,baseline_qubits=9,
        extra_reference_resets=8*outtotals['reference']['total_shots'],
        source_CSWAPs=16*outtotals['reference']['total_shots'],
        source_and_arithmetic_scope=proof['scope'],arithmetic_assumptions=proof['arithmetic_assumptions'],
        T='NOT_ACQUIRED; no new synthesis',new_PF_law_gadget_or_placement=False)
    save_json(OUT/'STAGE4_RPE.json',result)
    with (OUT/'STAGE4_ROUNDS.tsv').open('w') as f:
        f.write('m\tq\tr\tbaseline_shots_per_axis\treference_shots_per_axis\tbaseline_radius_lower\treference_radius_lower\tbaseline_stat_phase_lower\treference_stat_phase_lower\n')
        for x in rows:
            b=x['arms']['Simon_Love'];c=x['arms']['reference']
            f.write('\t'.join(map(str,[x['m'],x['q'],x['r'],b['shots_per_axis'],c['shots_per_axis'],
                b['raw_radius_lower'],c['raw_radius_lower'],b['statistical_phase_lower'],c['statistical_phase_lower']]))+'\n')
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',stage1='COMPLETE',stage2='CX_TRANSFER_QUALIFIED',
        stage3=proof['status'],stage4=result['status'],reason='Authorized finite scope complete',
        old_STOPs_preserved=True,source_or_science_scope_expanded=False,
        next_owner='GPT/user: judge fixed-model resource bounds, arithmetic assumptions, and whether another metric or stronger control policy merits separate work'))
    return result
