"""Tiny artificial selected words only. Never load/evaluate real H4 states."""
import copy
from fractions import Fraction as Q
import importlib.util
import math
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
P=runpy.run_path(str(ROOT/'scripts/inspect_h4_arb_pr0.py'))['P']
B=P.B;C=B._backend()
spec=importlib.util.spec_from_file_location('reused_native_fixture',ROOT/'tests/test_h4_arb_bridge.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)


def fixture(n=2,orders=(0,2,4,0),q=2,r=2,blocks=True,scalar=(.1,.2,-.3),components=None,angles=(.125,.25,.375)):
    tail=lambda word:'I'*(n-len(word))+word
    components=components or [dict(label=tail('X'),sign=-1,coefficient_abs=.2),dict(label=tail('Y'),sign=1,coefficient_abs=.1),
        dict(label=tail('Z'),sign=-1,coefficient_abs=.4),dict(label=tail('XI'),sign=1,coefficient_abs=.5)]
    case=dict(config=dict(method='PR',T=q*.1,q=q,r=r,K=4),
        blocks=[dict(label=tail('YI'),coefficient=.3),dict(label=tail('XZ'),coefficient=-.1)] if blocks else [],
        components=components,law=dict(angles=list(angles)),native_scalar_phases=list(scalar),saved_normalization=dict(B=1.5))
    model=dict(representation='pauli_jw',qubit_order='qiskit_little_endian',n_qubits=n,step_time=.1,cases=dict(PR=case))
    events=[]
    assert len(orders)==q*r
    for i,k in enumerate(orders):
        products=[str(j%len(components)) for j in range(k)];rotation=str((i+2)%len(components));ids=[*products,rotation]
        apps=[dict(application_index=j,component_id=cid,role='product' if j<k else 'rotation',
            coefficient_abs=components[int(cid)]['coefficient_abs'],coefficient_sign=components[int(cid)]['sign'],
            is_identity=all(x=='I' for x in components[int(cid)]['label']),basis_id=None,basis_hash=None,
            df_fragment_id=None,basis_change_operations=[]) for j,cid in enumerate(ids)]
        events.append(dict(event_position=i,outer_step_index=i//r,inner_RTE_position=i%r,taylor_order=k,
            product_component_ids=products,rotation_component_id=rotation,selected_component_ids=ids,application_sequence=apps,
            rotation_angle=angles[k//2],phase=dict(real=float((-1)**(k//2)),imag=0.)))
    join=dict(method='PR',q=q,r=r,trajectory_id='TOY_SELECTED',events=events)
    return model,join,P.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_PR0')


def raw(n):return [[(i+1)/8,(2-i)/16] for i in range(1<<n)]


def dense_selected(model,join,encoded):
    """256bit dense route using original C primitives and original record operands."""
    with C.ctx.workprec(256):
        case=model['cases']['PR'];cfg=case['config'];dim=len(encoded);I=C.identity(dim)
        psi=C.acb_mat([[C.complex_ball(x)] for x in encoded]);v=psi
        blocks=[]
        for b in case['blocks']:
            angle=C.real(Q(b['coefficient'])*Q(model['step_time'])/2)
            blocks.append(I*angle.cos()+C.pauli(b['label'])*C.acb(0,-angle.sin()))
        matrices={str(i):C.pauli(t['label'])*t['sign'] for i,t in enumerate(case['components'])}
        event_ops=[]
        for e in join['events']:
            product=I
            for cid in e['product_component_ids']:product=matrices[cid]*product
            angle=C.real(Q(e['rotation_angle']));rotation=I*angle.cos()+matrices[e['rotation_component_id']]*C.acb(0,-angle.sin())
            event_ops.append((rotation*product)*int(e['phase']['real']))
        scalar=C.phase(C.real(sum(map(Q,case['native_scalar_phases']),Q(0))))
        for step in range(cfg['q']):
            for op in blocks:v=op*v
            for op in event_ops[step*cfg['r']:(step+1)*cfg['r']]:v=op*v
            for op in reversed(blocks):v=op*v
            v=v*scalar
        nu=sum((Q(x)**2+Q(y)**2 for x,y in encoded),Q(0))
        return C.signal(psi,v).real/C.real(nu)


class SelectedTargetTests(unittest.TestCase):
    def test_two_three_qubit_dense_reference_contained_all_precisions(self):
        for n in (2,3):
            model,join,p=fixture(n);encoded=raw(n);saved=copy.deepcopy(encoded);reference=dense_selected(model,join,encoded)
            widths=[];before=C.ctx.prec
            for bits in (64,96,128):
                signal,r=P.evaluate_target_artificial(p,encoded,bits=bits,fixture_id='TOY_PR0')
                self.assertTrue(signal.contains(reference),(n,bits,signal,reference))
                widths.append(Q(r['interval_width_exact']));self.assertEqual(C.ctx.prec,before)
                self.assertFalse(r['PR_law_mean_used']);self.assertFalse(r['normalization_B_applied_to_target'])
            self.assertTrue(widths[0]>widths[1]>widths[2]);self.assertEqual(encoded,saved)

    def test_y_little_endian_and_signed_pauli_exact_permutations(self):
        for label in ('YI','XY','YZ','YIX','IZY'):
            for sign in (-1,1):
                n=len(label);mapping=P._pauli_map(label,sign);matrix=C.pauli(label)*sign
                for col in range(1<<n):
                    v=[C.acb(int(i==col)) for i in range(1<<n)]
                    actual=P._pauli_action(v,mapping,C)
                    self.assertEqual(actual,[matrix[row,col] for row in range(1<<n)])

    def test_noncommuting_product_order_phase_and_negative_sign_analytic(self):
        for sign in (1,-1):
            comps=[dict(label='IX',sign=sign,coefficient_abs=.2),dict(label='IY',sign=1,coefficient_abs=.3),dict(label='IZ',sign=1,coefficient_abs=.4)]
            _,_,p=fixture(orders=(2,),q=1,r=1,blocks=False,scalar=(.25,),components=comps,angles=(.125,0.,.375))
            signal,_=P.evaluate_target_artificial(p,[[1.,0.],[0.,0.],[0.,0.],[0.,0.]],bits=64,fixture_id='TOY_PR0')
            with C.ctx.workprec(256):expected=-sign*C.real('1/4').sin()
            # phase(-1)*Y*X = +iZ; a negative X changes that sign once.
            self.assertTrue(signal.contains(expected))

    def test_retained_exact_product_not_rounded_binary64_angle(self):
        _,_,p=fixture();angle=Q(.3)*Q(.1)/2
        self.assertEqual(Q(p['retained_blocks'][0]['angle_exact']),angle)
        self.assertNotEqual(angle,Q(float(angle)))
        p['retained_blocks'][0]['angle_exact']=str(Q(float(angle)))
        with self.assertRaisesRegex(ValueError,'RETAINED_EXACT_PRODUCT'):P.validate_plan(p)

    def test_native_scalar_exact_sum_not_point_accumulation(self):
        _,_,p=fixture();expected=sum(map(Q,(.1,.2,-.3)),Q(0))
        self.assertEqual(Q(p['native_scalar_exact_sum']),expected)
        point=0.
        for x in (.1,.2,-.3):point=float(point+x)
        self.assertNotEqual(expected,Q(point))
        p['native_scalar_exact_sum']=str(Q(point))
        with self.assertRaisesRegex(ValueError,'EXACT_NATIVE_SCALAR_SUM'):P.validate_plan(p)

    def test_event_binary64_value_not_intended_decimal(self):
        _,_,p=fixture(angles=(.1,.25,.375));self.assertEqual(Q(p['events'][0]['rotation_angle_exact']),Q(.1))
        p['events'][0]['rotation_angle_exact']='1/10'
        with self.assertRaises(ValueError):P.validate_plan(p)

    def test_event_application_order_sign_and_phase_corruption_rejected(self):
        model,join,_=fixture()
        for kind in ('order','sign','phase','position','rotation'):
            bad=copy.deepcopy(join)
            if kind=='order':bad['events'][1]['application_sequence'].reverse()
            if kind=='sign':bad['events'][1]['application_sequence'][0]['coefficient_sign']*= -1
            if kind=='phase':bad['events'][1]['phase']['real']=1.
            if kind=='position':bad['events'][0]['inner_RTE_position']=1
            if kind=='rotation':bad['events'][0]['rotation_angle']=.126
            with self.assertRaises(ValueError):P.build_plan(model,bad,domain='ARTIFICIAL',fixture_id='TOY_PR0')

    def test_common_raw_norm_and_target_unscaled_by_B(self):
        _,_,p=fixture();encoded=raw(2);saved=copy.deepcopy(encoded)
        first,r=P.evaluate_target_artificial(p,encoded,bits=64,fixture_id='TOY_PR0')
        p['B_saved_binary64_exact']='3'
        second,r2=P.evaluate_target_artificial(p,encoded,bits=64,fixture_id='TOY_PR0')
        self.assertEqual(C.endpoints(first),C.endpoints(second));self.assertEqual(encoded,saved)
        self.assertEqual(Q(r['raw_norm_exact']),sum((Q(x)**2+Q(y)**2 for x,y in encoded),Q(0)))
        self.assertFalse(r2['state_renormalized'])

    def test_mean_point_and_target_scale_cannot_replace_selected_word(self):
        _,_,p=fixture()
        for key in ('PR_law_mean_used','normalization_B_applied_to_target','probability_weights_applied'):
            bad=copy.deepcopy(p);bad[key]=True
            with self.assertRaises(ValueError):P.validate_plan(bad)
        bad=copy.deepcopy(p);bad['kind']='EMPIRICAL_ONLY_NO_RHO_UPPER'
        with self.assertRaises(ValueError):P.validate_plan(bad)

    def test_H4_and_real_domain_rejected_even_in_private_kernel_before_backend(self):
        _,_,p=fixture()
        cases=[dict(p,domain='SAVED_PR0_NO_EXECUTION',n_system=8),dict(p,n_system=8)]
        with patch.object(B,'_backend',side_effect=AssertionError('NO_H4_NUMERICAL_IMPORT')):
            for bad in cases:
                for function in (P.evaluate_target_artificial,P._target_artificial):
                    with self.assertRaisesRegex(ValueError,'NO_REAL_H4'):
                        if function is P._target_artificial:function(bad,raw(2),64,'TOY_PR0')
                        else:function(bad,raw(2),bits=64,fixture_id='TOY_PR0')
        with self.assertRaises(PermissionError):P.execution_request()

    def test_invalid_zero_raw_state_rejected(self):
        _,_,p=fixture()
        with self.assertRaisesRegex(ValueError,'POSITIVE_RAW_NORM'):P.evaluate_target_artificial(p,[[0.,0.]]*4,bits=64,fixture_id='TOY_PR0')

    def test_compiled_control_measurement_and_single_B_scale_full_pair(self):
        model,join,p=fixture(orders=(0,),q=1,r=1,blocks=False,scalar=(.25,),components=[dict(label='IZ',sign=1,coefficient_abs=.2)])
        def h():return [N.gate('rz',2,theta=math.pi/2),N.gate('sx',2),N.gate('rz',2,theta=math.pi/2)]
        gates=[*h(),N.gate('rz',0,theta=.125),N.gate('cx',2,0),N.gate('rz',0,theta=-.125),N.gate('cx',2,0),N.gate('rz',2,theta=.25),*h()]
        program=N.program(3,gates,phase=.125);program['fixture_id']='TOY_PR0'
        encoded=raw(2)
        record=P.compare_artificial(p,program,encoded,budget_exact='1/1000000',bits=64,fixture_id='TOY_PR0')
        target=dense_selected(model,join,encoded);compiled=N.dense_oracle(program,encoded)
        with C.ctx.workprec(256):reference=abs(C.real('3/2')*(compiled-target))
        lo,hi=record['rho']['interval']['lower'],record['rho']['interval']['upper']
        with C.ctx.workprec(256):enclosure=C.real(lo).union(C.real(hi))
        self.assertTrue(enclosure.contains(reference))
        self.assertEqual(record['rho']['B_saved_exact'],'3/2');self.assertEqual(record['artificial_state_actions'],2)
        self.assertEqual(record['rho']['budget_relation'],'UPPER_WITHIN_BUDGET')
        self.assertFalse(record['formal_certificate_accepted']);self.assertFalse(record['LOCAL_BUDGET_MET'])
        bad=copy.deepcopy(program);bad['instructions'][7]['params_exact']=[str(Q(-.25))]
        wrong=P.compare_artificial(p,bad,encoded,budget_exact='1/1000000',bits=64,fixture_id='TOY_PR0')
        self.assertEqual(wrong['rho']['budget_relation'],'LOWER_EXCEEDS_BUDGET')

    def test_pair_rejects_wrong_layout_before_any_target_action(self):
        _,_,p=fixture();program=N.program(2);program['fixture_id']='TOY_PR0'
        with patch.object(B,'_backend',side_effect=AssertionError('MUST_REJECT_BEFORE_MATH')):
            with self.assertRaises(ValueError):P.compare_artificial(p,program,raw(2),budget_exact='1',bits=64,fixture_id='TOY_PR0')


if __name__=='__main__':unittest.main(verbosity=2)
