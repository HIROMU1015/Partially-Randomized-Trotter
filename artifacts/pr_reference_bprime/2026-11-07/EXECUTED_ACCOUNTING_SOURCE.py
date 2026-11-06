"""Fixed B-prime safe price bounds, inherited interval reuse and RPE accounting."""
from __future__ import annotations
from collections import defaultdict, Counter
from fractions import Fraction as F
from functools import lru_cache
import io, json, math, tarfile, inspect
from pathlib import Path
import mpmath as mp
from qiskit import QuantumCircuit
from . import pr_reference_bprime as b
from . import pr_reference_transfer_cost_bounds as cost
from . import pr_reference_transfer_accounting as account
from . import track_c_compiler_a_18round as iv
from .track_c_h4 import common,save_json

@lru_cache(None)
def inherited_catalogue():
    # Snapshot only: old outputs and their STOP are never opened for writing.
    dest=b.OUT/'inherited_catalogue'
    dest.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b.data(b.OLD+'local_catalogue_boundary.tar.xz')),mode='r:xz') as t:
        for m in t.getmembers():
            if m.isfile():
                path=dest/'catalogue_boundary'/Path(m.name).name
                path.parent.mkdir(exist_ok=True);path.write_bytes(t.extractfile(m).read())
    cost.OUT=dest
    return dest

def distributions(bits,r):
    components=list(common()['preps'][3].rte_preparation.symbolic_tail.components)
    p=cost.distribution([c.probability for c in components],bits)
    prep=common()['preps'][3]
    dist=b.pilot.finite_rte_distribution(prep.exact_rte_lambda_r*.02/r,2)
    p2=cost.distribution(dist.order_probabilities,bits)[1]
    pzz=sum((pi for c,pi in zip(components,p) if len(c.diagonal_pauli_support)==2),F())
    support=sum((pi*len(c.diagonal_pauli_support) for c,pi in zip(components,p)),F())
    return components,p,p2,pzz,support

def price_bounds(q,r,bits):
    inherited_catalogue();params=cost.parameters()
    old_cx,ref_cx=cost.expected(q,r,params,bits)
    components,p,p2,pzz,support=distributions(bits,r)
    # All parity CX may be removed in this conservative lower bound.
    # Protected CRZ cores and fenced D/frames/products remain.
    upper=old_cx-10*q*r*pzz;lower=upper-2*q*r*pzz
    prices={}
    for axis in ('X','Y'):
        old_rz,ref_rz=account.expected_rz_bounds(q,r,bits,axis)
        new_rz=[v-22*q*r*pzz for v in old_rz]
        bounds,_=account.rz_parameters()
        assert all(v[0]==v[1] for v in bounds['reference'].values())
        # In this fixed context, all RZ other than product Z lowerings survive.
        # Discarding every product-Z lowering supplies a safe reference lower.
        ref_rz_lower=ref_rz-2*q*r*p2*support
        prices[axis]=dict(Bprime_CX_lower=lower,Bprime_CX_upper=upper,
            Bprime_RZ_lower=new_rz[0],Bprime_RZ_upper=new_rz[1],reference_CX_upper=ref_cx,
            reference_RZ_lower=ref_rz_lower,reference_RZ_upper=ref_rz,registered_CX=old_cx,
            registered_RZ_lower=old_rz[0],registered_RZ_upper=old_rz[1])
    fixed,basis,ref,full,_=params
    db,dr=cost.d_cost(q,fixed,ref,full)
    pb=defaultdict(F)
    for c,pi in zip(components,p):pb[c.basis_id]+=pi
    frames=q*sum(F(basis[k])*(pi+(r*(1+2*p2)-1)*pi*(1-pi)) for k,pi in pb.items())
    parts=dict(P_ZZ=pzz,p_order2=p2,expected_support=support,D_baseline_CX=F(db),D_reference_CX_including_CSWAP=F(dr),
        shared_basis_CX=frames,Bprime_rotation_CX_before_cancellation=q*r*(2+2*pzz),
        Bprime_product_CX=q*r*2*p2*support,reference_rotation_CX=q*r*2*pzz,
        removable_parity_CX_allowance=2*q*r*pzz,source_CSWAP_CX=F(128))
    return prices,parts

def validate_saved_prices():
    inherited_catalogue();checks=[];comp=b.params()[-1]
    contexts={tuple(json.loads(p.read_text())['post_unroll']) for p in (b.OUT/'catalogue').glob('*.json')}
    source_positions=[]
    for path in sorted((b.OUT/'raw').glob('*.json')):
        d=json.loads(path.read_text());q=d['q'];r=d['r'];z=d['structural_bounds']['ZZ_rotations']
        sample=next(x for x in b.saved_samples() if (x['q'],x['r'],x['rare'],x['j'])==(q,r,d['rare'],d['j']))
        _,_,events=b.reconstructed(sample)
        rz_path=0;last_order2=[]
        # Registered RZ is exact in these fixed contexts; replacement removes
        # 24-2 RZ per ZZ. Source/final CX cancellations never remove RZ.
        for i,axis in enumerate('XY'):
            old=d['saved_values']['Simon_Love'][i];actual=d['Bprime_values'][i];ref=d['saved_values']['reference'][i]
            predicted_rz=old[0]-22*z
            trace=d['traces'][axis];unroll=next(x for x in trace if x['pass_name']=='UnrollCustomDefinitions')
            context=tuple(sorted(set(unroll['ops'])-{'measure','barrier'}))
            inverses=[x for x in trace if x['pass_name']=='InverseCancellation']
            deficit=d['structural_bounds']['bprime_CX_upper']-actual[1]
            ref_deficit=d['structural_bounds']['reference_CX_upper']-ref[1]
            first_parity=inverses[0]['parity_CX_removed']
            passed=predicted_rz==actual[0] and context in contexts and 0<=deficit<=2*z
            checks.append(dict(path=path.name,axis=axis,q=q,r=r,baseline_CX_range=[d['structural_bounds']['bprime_CX_lower'],d['structural_bounds']['bprime_CX_upper']],
                actual_CX=actual[1],actual_RZ=actual[0],predicted_RZ=predicted_rz,
                boundary_CX_deficit=deficit,reference_saved_CX_deficit=ref_deficit,
                source_parity_CX_removed=first_parity,observed_deficits_equal=deficit==ref_deficit,
                passed=passed))
            source_positions.append(dict(path=path.name,axis=axis,
                initial_removed_parity_CX_nodes=[x for x in inverses[0]['removed_CX_nodes'] if all(a<8 for a in x['wires'])]))
    assert len(checks)==128
    passed=all(x['passed'] for x in checks)
    result=dict(status='SAFE_CX_LOWER_UPPER_AND_RZ_PRICES_QUALIFIED' if passed else 'UNDETERMINED',
        checks=checks,source_parity_positions=source_positions,
        equal_deficits_all_saved=all(x['observed_deficits_equal'] for x in checks),
        use_equal_cancellation_as_all_support_theorem=False,
        all_support_CX_lower='registered additive price - 12*#ZZ; only parity CX discarded',
        all_support_CX_upper='registered additive price - 10*#ZZ',
        all_support_RZ='registered RZ - 22*#ZZ; final pass removes CX/X only; source pass removes no RZ',
        passed=passed)
    save_json(b.OUT/'PRICE_BOUND_CHECKS.json',result)
    if not passed:raise ArithmeticError('Mandatory STOP: B-prime price model failed')
    return result

def v2_builder_witness(qvalue=4):
    # Instrument only this process, restore in finally, build only an existing
    # q4/r1 rare path. No new RPE condition or long-r compile.
    assert qvalue in (1,4)
    row=next(x for x in b.saved_samples() if x['q']==qvalue and x['rare'] and x['j']==0)
    cfg,dist,events=b.reconstructed(row)
    prop=QuantumCircuit.global_phase;calls=[];objects=[]
    def setter(q,value):
        caller=inspect.currentframe().f_back.f_code.co_name
        calls.append((q,caller));objects.append(q)
        return prop.fset(q,value)
    try:
        QuantumCircuit.global_phase=property(prop.fget,setter,prop.fdel,prop.__doc__)
        a=b.transfer.assemble(qvalue,1,cfg,dist,events)
    finally:QuantumCircuit.global_phase=prop
    rows=[];n=max(1,qvalue//2)
    for lane in ('F','G'):
        q=a[lane];updates=Counter(caller for obj,caller in calls if obj is q)
        count=sum(i.operation.name in ('rz','rzz') for i in q.data)
        active=sum(n for k,n in updates.items() if k!='__init__')
        assert count<=(336+49)*n+24 and active<=20*n
        rows.append(dict(lane=lane,steps=n,r=1,RZ_RZZ_subtractions=count,
            bound_RZ_RZZ=(336+49)*n+24,phase_setter_calls=dict(updates),
            non_constructor_updates=active,phase_update_bound=20*n))
    filename='V2_BUILDER_WITNESS.json' if qvalue==4 else 'V2_Q1_BUILDER_WITNESS.json'
    save_json(b.OUT/filename,dict(rows=rows,
        all_support_derivation='D: 336n+24; RTE<=3r basis runs x16 RZ + r rotations; lane gp <=(7n+1)+n+1=8n+2<=20n; RTE internal update<=n separately',
        instrumentation_restored=True,new_long_r_condition=False,passed=True))
    return rows

def error_and_shots():
    inherited_catalogue();mp.iv.prec=192
    saved=json.loads(b.data(b.OLD+'STAGE3_LOCAL_INTERVALS.json'))['records']
    errors={}
    for x in saved:
        key=x['family'],tuple(float(v).hex() for v in x['params']);errors[key]=max(errors.get(key,0),x['upper'])
    I=iv.I
    def basis_error(ops):
        terms=[]
        for g,_ in ops:
            for op in (g,g.inverse()):
                if op.name in ('rz','cx'):continue
                terms.append(I(errors[(op.name,tuple(float(x).hex() for x in op.params))]))
        return sum(terms,I(0))
    p=common()['preps'][3];derr=[basis_error(x.runtime_basis_operations) for x in p.deterministic_blocks]
    # Same outward helper rounding as old bound; CRZZ term is removed only
    # after every actual new kernel lowered to its exact binary half angles.
    derr=[I(iv.upper(x)) for x in derr]
    tail=I(max(iv.upper(basis_error(d.runtime_operations)) for d in p.rte_preparation.basis_registry._definitions.values()))
    family=lambda f:I(max(x['upper'] for x in saved if x['family']==f))
    round_path='artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json'
    old=next(c for c in json.loads(b.data(round_path)) if c['name']=='CN_S2_h0.02_ld3')['rows']
    reference=json.loads(b.data(b.OLD+'STAGE3_ALL_SUPPORT.json'))['rows']
    correctness=json.loads((b.OUT/'CORRECTNESS.json').read_text());assert correctness['passed']
    rows=[]
    for o,ref in zip(old,reference):
        q=o['q'];r=o['r'];error=(q+1)*derr[0]+2*q*sum(derr[1:],I(0))+3*q*r*tail+4*q*r*family('cz')+2*family('h')+family('sdg')
        shared=2*mp.iv.sqrt(I(2))*I(o['source_build_and_finite_RTE_operator_upper'])+I(o['event_phase_disk_inherited'])+I(o['finite_bit_proposal']['directed_signal_error_upper'])
        disk=shared+2*mp.iv.sqrt(I(2))*error
        rho_nom=I(o['raw_radius_lower'])+I(o['raw_extra_disk_upper']);rho=rho_nom-disk
        ratio=disk/rho_nom;beta=I('.4')-I(o['PF_phase_upper'])-mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))
        # Serialize outward endpoints before shot computation, as previous arm.
        rho_lower=iv.lower(rho);beta_lower=iv.lower(beta)
        assert rho_lower>0 and beta_lower>0
        epsilon=I(rho_lower)*mp.iv.sin(I(beta_lower))/mp.iv.sqrt(I(2));required=2*mp.iv.log(I(1440))/epsilon**2
        with mp.workprec(192):n=int(mp.ceil(mp.mpf(required._mpi_[1])))
        rows.append(dict(m=o['m'],q=q,r=r,bits=o['finite_bit_proposal']['bits_per_uniform_draw'],
            native_joint_error_upper=iv.upper(error),native_raw_disk_upper=iv.upper(2*mp.iv.sqrt(I(2))*error),
            inherited_source_phase_proposal_raw_disk_upper=iv.upper(shared),total_raw_disk_upper=iv.upper(disk),
            raw_radius_lower=rho_lower,statistical_phase_lower=beta_lower,shots_per_axis=n,
            reference_error_reused=ref['arms']['reference'],old_compiler_replaced=True,new_kernel_error_exact=0))
    alias=json.loads(b.data('artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/GROUND_ALIAS.json'))
    assert alias['all_spectrum_endpoint_margin_pass']
    result=dict(status='ALL18_BPRIME_SUCCESS_CONDITIONS_CLOSED_IN_FIXED_MODEL',rows=rows,
        total_shots=sum(2*x['shots_per_axis'] for x in rows),interval_bits=192,
        old_shots_blindly_reused=False,alpha_per_axis='1/720',failure_total='36/720=.05',
        energy_accuracy_Ha=1.5936e-4,phase_cap_rad=.4,
        new_kernel='exact parity-CRZ-uncompute at actual SOURCE angle; theta/2 exact binary scaling',
        old_intervals_reused_after_lowering_equality=True,
        reference_and_old_baseline='unchanged saved error/shot schedules reused',
        scope='saved Hermitian H4, exact ground supplied, ideal logical gates/reset/measurement, finite independent bits, historical binary64 remainder model')
    save_json(b.OUT/'ERROR_RPE.json',result)
    return result

def validate_reference_rz_lower():
    inherited_catalogue();bounds,fam=account.rz_parameters();comp=b.params()[-1];checks=[]
    for path in sorted((b.OUT/'raw').glob('*.json')):
        d=json.loads(path.read_text());q=d['q'];r=d['r']
        sample=next(x for x in b.saved_samples() if (x['q'],x['r'],x['rare'],x['j'])==(q,r,d['rare'],d['j']))
        _,_,events=b.reconstructed(sample)
        prod=sum(len(comp[cid].diagonal_pauli_support) for e in events for cid in e.product_component_ids)
        phase=sum(any(e.taylor_order==2 for e in events[j*r:(j+1)*r]) for j in range(q))
        B=bounds['Simon_Love'];R=bounds['reference']
        db=B['left_all'][0]+B['right_all'][0]+(q-1)*(B['left_no0'][0]+B['right_no0'][0]+B['boundary_control'][0])
        C=lambda k:R[f'block:{k}:half'][0]
        dr=2*sum(C(k) for k in range(4)) if q==1 else 4*C(0)+(q-2)*R['block:0:full'][0]+2*q*sum(C(k) for k in (1,2,3))
        dr+=16*fam['reference']['cswap'][0]
        for i,axis in enumerate('XY'):
            ref_upper=d['Bprime_values'][i][0]-(db-dr)-q*r-3*prod-phase
            ref_lower=ref_upper-prod;actual=d['saved_values']['reference'][i][0]
            checks.append(dict(path=path.name,axis=axis,reference_lower=ref_lower,reference_upper=ref_upper,
                actual_reference_RZ=actual,all_product_Z_allowance=prod,passed=ref_lower<=actual<=ref_upper))
    passed=all(x['passed'] for x in checks);save_json(b.OUT/'REFERENCE_RZ_LOWER_CHECKS.json',dict(checks=checks,passed=passed))
    if not passed:raise ArithmeticError('Mandatory STOP: reference RZ lower failed')
    return passed

def totals_and_stop():
    b.attach_cache()
    assert json.loads((b.OUT/'PRICE_BOUND_CHECKS.json').read_text())['passed']
    assert json.loads((b.OUT/'REFERENCE_RZ_LOWER_CHECKS.json').read_text())['passed']
    assert json.loads((b.OUT/'V2_INPUTS_CONSTANTS.json').read_text())['all_angles_equal']
    assert (b.OUT/'V2_BUILDER_WITNESS.json').exists() and (b.OUT/'V2_Q1_BUILDER_WITNESS.json').exists()
    proof=json.loads((b.OUT/'ERROR_RPE.json').read_text());mp.iv.prec=192
    prior=json.loads(b.data(b.OLD+'STAGE4_RPE.json'))
    totals={k:F() for k in ('CX_lower','CX_upper','RZ_lower','RZ_upper')};rows=[];ref_rz_lower=F()
    for row,prev in zip(proof['rows'],prior['rows']):
        q=row['q'];r=row['r'];n=row['shots_per_axis'];prices,parts=price_bounds(q,r,row['bits'])
        for axis,prices_axis in prices.items():
            for k in totals:totals[k]+=n*prices_axis['Bprime_'+k]
            ref_rz_lower+=prev['arms']['reference']['shots_per_axis']*prices_axis['reference_RZ_lower']
        rows.append(dict(**row,price_per_axis={axis:{k:str(v) for k,v in p.items()} for axis,p in prices.items()},
            contributions={k:str(v) for k,v in parts.items()}))
    def fi(v):return iv.I(v.numerator)/v.denominator
    t={k:(iv.lower(fi(v)) if k.endswith('lower') else iv.upper(fi(v))) for k,v in totals.items()}
    compare={}
    for metric in ('CX','RZ'):
        # Saving lower: unknown actual baseline >= lower, reference <= upper.
        base=fi(totals[metric+'_lower']);ref=iv.I(prior['totals']['reference']['total_'+metric+'_upper'])
        saving=base-ref;percent=100*(1-ref/base)
        compare[metric]=dict(reference_upper_below_Bprime_lower=iv.lower(saving)>0,
            guaranteed_absolute_saving_lower=iv.lower(saving),guaranteed_relative_saving_percent_lower=iv.lower(percent),
            interpretation='negative lower margin does not prove absence of actual improvement')
    ref_totals=dict(prior['totals']['reference']);ref_totals['total_RZ_lower']=iv.lower(fi(ref_rz_lower))
    rz_increase=fi(ref_rz_lower)-fi(totals['RZ_upper'])
    compare['RZ'].update(reference_lower_above_Bprime_upper=iv.lower(rz_increase)>0,
        guaranteed_absolute_increase_lower=iv.lower(rz_increase),
        guaranteed_relative_increase_percent_lower=iv.lower(100*rz_increase/fi(totals['RZ_upper'])),
        reference_lower_rule='discard ALL product-Z RZs; other RZ counts identical across fixed reference contexts')
    previous_comparisons={}
    for metric in ('CX','RZ'):
        base=iv.I(prior['totals']['Simon_Love']['total_'+metric+'_lower'])
        ref=iv.I(prior['totals']['reference']['total_'+metric+'_upper'])
        previous_comparisons[metric]=dict(absolute_saving_lower=iv.lower(base-ref),
            relative_saving_percent_lower=iv.lower(100*(1-ref/base)))
    result=dict(status='CONDITIONAL_CX_REDUCTION_VS_BPRIME' if compare['CX']['reference_upper_below_Bprime_lower'] else 'CX_ADVANTAGE_UNQUALIFIED_OVERLAPPING_BOUNDS',
        totals={'registered_14CX':prior['totals']['Simon_Love'],
            'Bprime':dict(total_shots=proof['total_shots'],**{'total_'+k:v for k,v in t.items()},total_depth=None,total_size=None,total_T=None),
            'reference':ref_totals},comparisons=compare,previous_registered_comparisons=previous_comparisons,rows=rows,
        exact_Bprime_total_fractions={k:str(v) for k,v in totals.items()},
        exact_reference_RZ_lower_fraction=str(ref_rz_lower),
        reference_prices='saved UPPER, not actual expected prices',
        Bprime_prices='safe lower discards all parity CX; upper ignores cross-kernel cancellation; RZ exact in fixed contexts',
        guarantee_scope=proof['scope'],reference_qubits=17,baseline_qubits=9,
        reference_extra_resets=prior['extra_reference_resets'],reference_source_CSWAPs=prior['source_CSWAPs'],
        no_value_threshold_introduced=True,V3_to_V6_not_executed=True)
    save_json(b.OUT/'TOTALS_18ROUND.json',result)
    import csv
    with (b.OUT/'ROUNDS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['m','q','r','Bprime_shots_axis','Bprime_radius_lower','Bprime_stat_phase_lower','P_ZZ','p2','expected_support'])
        for x in rows:w.writerow([x['m'],x['q'],x['r'],x['shots_per_axis'],x['raw_radius_lower'],x['statistical_phase_lower'],x['contributions']['P_ZZ'],x['contributions']['p_order2'],x['contributions']['expected_support']])
    save_json(b.OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='Authorized V1 and necessary V2 complete',
        V1=result['status'],V2='constants / actual angles / platform / structural appendix completed',
        old_SOURCE_results_STOPs_preserved=True,V3_to_V6_not_started=True,
        next_owner='GPT/user: scientific interpretation of Bprime resource bounds and choice of any separately authorized next work'))
    return result
