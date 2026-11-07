"""New binding/strict-angle deltas only; invented data, no molecular calls."""
import copy
from fractions import Fraction as F
import math
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from trottertracks.algorithm_codesign import cs_v0 as v, cs_v0_native as n
from trottertracks.algorithm_codesign.same_position_residual import Angle, Replacement, plan_sequence
from trottertracks.algorithm_codesign.g7_generator import CanonicalGenerator
from trottertracks.algorithm_codesign.parent_allocation_native import operator_target


def fixture():
    basis = [dict(name='rz', qubits=[0], parameters=[.5], matrix_sha256='mock')]
    prep = dict(deterministic_blocks=[dict(block_id='one', block_kind='one_body', order_index=0,
                num_system_qubits=2, basis_hash='basis', basis_change_operations=basis)],
                basis_operations={'basis': basis}, components=[dict(component_id='a', is_identity=False,
                coefficient_sign=-1, diagonal_pauli_support=[1], basis_id='R',basis_hash='Rhash'), dict(component_id='z', is_identity=False,
                coefficient_sign=1, diagonal_pauli_support=[0])])
    law = dict(label_masses=[dict(label='z', exact='0'), dict(label='a', exact='1')])
    uniform = dict(inventory=dict(deterministic_half_step_records=[dict(block_id='one',
        half_step_gates=[dict(key=['rz', [(-.5).hex()]], qubits=[0]),
                        dict(key=['crzz', [(.25).hex()]], qubits=[2, 0, 1]),
                        dict(key=['crz', [(.125).hex()]], qubits=[2, 0]),
                        dict(key=['p', [(.5).hex()]], qubits=[2]),
                        dict(key=['rz', [(.5).hex()]], qubits=[0])])]))
    return prep, law, uniform


def policy():
    return dict(policy_fingerprint='mock-policy', definitions=[dict(source_basis_id='R',
        source_basis_hash='Rhash', support=[1], definition=dict(basis_id='selected',
        basis_hash='selectedhash',num_system_qubits=2,operations=[]))])


def tool():
    return dict(name='pygridsynth', version='2.0.0', source_sha256='1'*64, seed=0,
                tool_identity_sha256='2'*64, runtime_executable_sha256='3'*64,
                package_lock_sha256='4'*64, python='artificial', interval_dps=90,
                synthesizer_options=dict(seed=0, up_to_phase=False, dps=90))


class V0DeltaTests(unittest.TestCase):
    def test_saved_schema_binding_and_stable_lineage(self):
        p, l, u = fixture(); calls = []
        b = v.bind_slots(p, l, u, lambda c: calls.append(c['component_id']) or Angle(rad=F(-3, 10)), n=2,policy=policy())
        self.assertEqual(calls, ['a'])
        d, r = b['sites']; self.assertEqual((d.generator, r.generator), ('IZZ', 'IZI'))
        self.assertEqual(d.site_id, 'D/forward/one/central0/factor0')
        self.assertEqual((d.angle, r.angle), (Angle(rad=F(1, 8)), Angle(rad=F(-3, 20))))
        self.assertEqual([s.generator for s in b['fixed_control_partners']], ['ZZZ', 'ZZI'])
        self.assertTrue(all(s.role=='FIXED' for s in b['fixed_control_partners']))
        self.assertEqual(b['fixed_D_boundaries'][0]['central_records'][2]['key'][0], 'p')
        self.assertEqual(b['fixed_selected_R_basis']['basis_hash'],'selectedhash')
        self.assertEqual(b, v.bind_slots(p, l, u, lambda c: Angle(rad=F(-3, 10)), n=2,policy=policy()))

    def test_bad_saved_role_phase_or_basis_stops_before_resolver(self):
        for mutate in (lambda p,l,u: p['deterministic_blocks'][0].update(block_kind='unknown'),
                       lambda p,l,u: u['inventory']['deterministic_half_step_records'][0].update(block_id='other'),
                       lambda p,l,u: u['inventory']['deterministic_half_step_records'][0]['half_step_gates'][0].update(qubits=[1]),
                       lambda p,l,u: u['inventory']['deterministic_half_step_records'][0]['half_step_gates'][3].update(qubits=[0])):
            p,l,u = fixture(); mutate(p,l,u)
            with self.assertRaises(ValueError):
                v.bind_slots(p,l,u,lambda c: self.fail('invalid binding reached resolver'),n=2,policy=policy())

    def test_first_positive_unsupported_label_is_not_replaced(self):
        p,l,u=fixture(); p['components'][0]['is_identity']=True
        with self.assertRaises(ValueError):v.bind_slots(p,l,u,lambda c: self.fail('substitution'),n=2,policy=policy())
        p,l,u=fixture(); l['label_masses'][1]['label']='missing'
        with self.assertRaises(ValueError):v.bind_slots(p,l,u,lambda c: self.fail('substitution'),n=2,policy=policy())
        p,l,u=fixture(); bad=policy();bad['definitions']=[]
        with self.assertRaises(ValueError):v.bind_slots(p,l,u,lambda c: self.fail('missing selected policy'),n=2,policy=bad)

    def test_binary_parameter_conversion_is_exact_and_closed(self):
        self.assertEqual(v.binary_angle((.1).hex()).rad, F(.1))
        for x in (.1, 'nan', 'inf', '0.1'):
            with self.assertRaises((ValueError,TypeError)):v.binary_angle(x)

    def test_original_R_event_angle_boundary_on_invented_input(self):
        from trottertracks.algorithm_codesign.shared_cpu import module
        from dataclasses import asdict
        rte=module('rte')
        distribution=asdict(rte.finite_rte_distribution(.25,2))
        component=dict(component_id='invented',probability=1.,coefficient_abs=1.,
            coefficient_sign=-1,is_identity=False,df_fragment_id='invented-fragment',
            basis_id='invented-basis',basis_hash=None,diagonal_pauli_support=(0,))
        angle=v.resolve_r_angle(component,distribution)
        self.assertEqual(angle.rad,F(-2*math.atan(.25)))

    def test_coarse_policy_exact_ties_and_common_arm_table(self):
        for a in (Angle(pi=2,rad=F(1,512)), Angle(pi=-3,rad=F(-3,512)), Angle(rad=F(7,1000))):
            c=v.coarse(a); r=a+c.scaled(-1)
            self.assertEqual(c.pi,a.pi); self.assertEqual(r.pi,0)
            self.assertLessEqual(abs(r.rad),F(1,512));self.assertEqual(c+r,a)
        self.assertEqual(v.coarse(Angle(rad=F(-1,512))).rad,F(-1,256))
        p,l,u=fixture(); b=v.bind_slots(p,l,u,lambda _:Angle(rad=F(1,17)),n=2,policy=policy())
        table={s.site_id:v.coarse(s.angle) for s in b['sites']}
        for arm in ('S0','S1','S2'):
            out=plan_sequence(b['sites'],table,arm=arm)
            for q in out:
                if isinstance(q,Replacement):self.assertEqual(q.coarse,table[q.original.site_id])

    def test_rational_residual_table_and_error_enclosure(self):
        with patch.object(CanonicalGenerator,'sample',side_effect=AssertionError('no RNG')):
            table=v.residual_table(Angle(rad=F(-1,512)))
        self.assertEqual(len(table['events']),2)
        self.assertTrue(all(e['rotation_sign']==-1 for e in table['events']))
        self.assertEqual(sum(e['proposal'] for e in table['events']),1)
        with mp.workdps(100):
            I=mp.eye(2); Z=mp.diag([1,-1]); cv=lambda q:mp.mpf(q.numerator)/q.denominator
            def rot(_,q,s):
                t=mp.atan(cv(q));return mp.cos(t)*I-1j*s*mp.sin(t)*Z
            mean=mp.zeros(2)
            for e in table['events']:
                op=operator_target(e,identity=lambda:I,multiply=lambda a,b:a*b,
                    involution=lambda _:Z,rotation=rot,scale=lambda a,b:a*b)
                mean+=cv(e['coefficient'])*op
            exact=mp.diag([mp.exp(1j/ mp.mpf(1024)),mp.exp(-1j/mp.mpf(1024))])
            err=max(abs(mean[i,i]-exact[i,i]) for i in range(2))
            self.assertGreater(err,0);self.assertLess(err,cv(table['finite_error_upper']))
        self.assertLess(table['finite_error_upper'],F(1,10**12))
        for a in (Angle(pi=1),Angle(rad=F(1,511))):
            with self.assertRaises(ValueError):v.residual_table(a)
        self.assertTrue(v.residual_table(Angle())['exact_identity'])

    def test_strict_fast_path_preserves_2pi_and_rejects_bare_T(self):
        for angle in (Angle(),Angle(pi=F(1,2)),Angle(pi=1),Angle(pi=2),Angle(pi=4)):
            ex=n.Expression.linear(angle);seq=n.exact_strict(ex)
            self.assertIsNotNone(seq);self.assertLess(n.strict_guard(seq,ex,dps=90),F(1,10**80))
        self.assertEqual(n.exact_strict(n.Expression.linear(Angle(pi=2))),'WWWW')
        ex=n.Expression.linear(Angle(pi=F(1,4)))
        self.assertIsNone(n.exact_strict(ex));self.assertGreater(n.strict_guard('T',ex,dps=90),F(1,2))

    def test_mixed_angle_mock_provider_strictly_guards_exact_target(self):
        ex=n.Expression.linear(Angle(pi=F(1,2),rad=F(1,10**8)))
        req,sign=n.request(ex,F(1,10**6),tool()); calls=[]
        row=n.acquire(req,synth=lambda *args:calls.append(args) or 'WWWWWWWS')
        self.assertEqual(sign,1);self.assertEqual(row['backend_calls'],1);self.assertEqual(len(calls),1)
        self.assertGreater(F(row['operator_error_upper']),0)
        self.assertEqual(row['global_W_count'],7);n.checked(row,req)
        with self.assertRaises(ValueError):n.acquire(req,synth=lambda *args:'S')

    def test_positive_key_and_actual_adjoint_include_scalar(self):
        plus=n.Expression.linear(Angle(pi=F(1,2),rad=F(1,10**8)))
        minus=n.Expression.linear(Angle(pi=F(-1,2),rad=F(-1,10**8)))
        req,s=n.request(plus,F(1,10**6),tool()); other,t=n.request(minus,F(1,10**6),tool())
        self.assertEqual(req,other);self.assertEqual((s,t),(1,-1))
        row=n.acquire(req,synth=lambda *args:'WWWWWWWS')
        self.assertEqual(n.time_letters(row,req,-1),('w',)*7+('s',))
        self.assertEqual(n.time_letters(row,req,1),('S',)+('W',)*7)
        # Independent 2x2 check including explicit scalar; time order matters.
        w=np.exp(1j*math.pi/4);m={'W':w*np.eye(2),'w':w.conjugate()*np.eye(2),
                                'S':np.diag([1,1j]),'s':np.diag([1,-1j])}
        products=[]
        for sign in (1,-1):
            u=np.eye(2,dtype=complex)
            for g in n.time_letters(row,req,sign):u=m[g]@u
            products.append(u)
        np.testing.assert_allclose(products[1],products[0].conj().T,atol=1e-15,rtol=0)

    def test_scaled_atan_boundary_and_sign(self):
        ex=n.Expression('atan',F(1,2),-2); req,s=n.request(ex,F(1,1000),tool())
        self.assertEqual(s,-1);self.assertEqual(req['spec']['angle'],dict(kind='atan',a='1/2',b='2'))
        with n.precision(90):
            direct=mp.iv.mpf(2)*mp.iv.atan2(mp.iv.mpf(1)/2,mp.iv.mpf(1))
            self.assertEqual(n.enclosure(direct),n.enclosure(n.Expression(**req['spec']['angle']).value(mp.iv)))

    def test_precision_tool_and_cache_mixing_rejected(self):
        ex=n.Expression.linear(Angle(pi=F(1,2)));req,_=n.request(ex,F(1,10**6),tool())
        row=n.acquire(req,synth=lambda *args:self.fail('fast path must not acquire'))
        for field,val in [('interval_dps',100),('runtime_executable_sha256','f'*64),('version','2.0.1')]:
            t=tool();t[field]=val
            if field=='version':
                with self.assertRaises(ValueError):n.request(ex,F(1,10**6),t)
            else:
                other,_=n.request(ex,F(1,10**6),t)
                with self.assertRaises(ValueError):n.checked(row,other)
        other,_=n.request(ex,F(1,10**7),tool())
        with self.assertRaises(ValueError):n.checked(row,other)
        bad=copy.deepcopy(row);bad['operator_error_upper']='0'
        with self.assertRaises(ValueError):n.checked(bad,req)

    def test_unsupported_and_projective_options_rejected(self):
        for fn in (lambda:n.Expression('float',1),lambda:n.Expression('linear',.1),
                   lambda:n.Expression('atan',0,1),lambda:n.request(Angle(),F(1,100),tool())):
            with self.assertRaises((ValueError,TypeError)):fn()
        t=tool();t['synthesizer_options']['up_to_phase']=True
        with self.assertRaises(ValueError):n.request(n.Expression('linear'),F(1,100),t)

    def test_no_repeat_after_exception_and_no_forbidden_call(self):
        events=[];calls=v.Calls(events.append)
        with self.assertRaises(ArithmeticError):calls.invoke('binding','one',lambda:(_ for _ in ()).throw(ArithmeticError('mock')))
        self.assertEqual([r['event'] for r in events],['CALL_START','STOP'])
        with self.assertRaises(RuntimeError):calls.invoke('binding','two',lambda:self.fail('retry'))
        with self.assertRaises(RuntimeError):v.Calls(events.append).invoke('Born','no',lambda:self.fail('forbidden'))

    def test_finite_key_cap_and_duplicate_acquisition(self):
        events=[];calls=v.Calls(events.append)
        for i in range(8):self.assertEqual(calls.invoke('positive_key',str(i),lambda:i),i)
        for i in (0,8):
            with self.assertRaises(RuntimeError):calls.invoke('positive_key',str(i),lambda:self.fail('extra'))
        self.assertEqual([e['event'] for e in events[-2:]],['STOP','STOP'])

    def test_weighted_native_and_finite_errors_not_zero_filled(self):
        table=v.residual_table(Angle(rad=F(1,512)));e=F(1,10**6)
        out=v.local_bounds(table,e,(e,e))
        self.assertEqual(out['operator_mean_error_upper'],
                         table['finite_error_upper']+2*table['statistics']['coefficient_mass']*e)
        for errors in ((e,), (e,-e)):
            with self.assertRaises(ValueError):v.local_bounds(table,e,errors)

    def test_local_joint_Z_lowering_reuses_literal_accounting(self):
        req,_=n.request(n.Expression.linear(Angle(pi=F(1,2))),F(1,10**6),tool())
        row=n.acquire(req,synth=lambda *a:self.fail('exact path'))
        out=n.local_native('IZZ',row,req,-1,phase_i_power=2)
        self.assertEqual(out['counts']['CX'],2);self.assertEqual(out['counts']['T'],0)
        self.assertEqual(out['time_gates'][:4],(('W',1),)*4)
        self.assertEqual(out['primitive_strict_error_upper'],F(row['operator_error_upper']))
        self.assertFalse(out['compiled'])
        with self.assertRaises(ValueError):n.local_native('IXZ',row,req,1)

    def test_static_preflight_cannot_activate_science(self):
        script=Path(__file__).resolve().parents[3]/'scripts/tracks/algorithm_codesign/cs_v0_preflight.py'
        spec=importlib.util.spec_from_file_location('cs_v0_static_only',script)
        pre=importlib.util.module_from_spec(spec);spec.loader.exec_module(pre)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'docs/research/cs_v0_technical_preparation_20261108';d.mkdir(parents=True)
            raw=b'{"invented": true}';(root/'input.json').write_bytes(raw);(root/'code.py').write_text('pass\n')
            runtime=n.hashlib.sha256(Path(pre.sys.executable).resolve().read_bytes()).hexdigest()
            (d/'NATIVE_PROFILE.json').write_text(json.dumps(dict(tool=dict(runtime_executable_sha256=runtime,tool_identity_sha256='0'*64))))
            lock=dict(fixed_inputs=[dict(path='input.json',bytes=len(raw),sha256=n.hashlib.sha256(raw).hexdigest())],
                      execution_source_sha256={'code.py':n.hashlib.sha256(b'pass\n').hexdigest()})
            (d/'SOURCE_AND_CHECKS.json').write_text(json.dumps(lock))
            with patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime',return_value={}),patch.object(v,'bind_slots',side_effect=AssertionError('no real binding')),patch.object(n,'backend',side_effect=AssertionError('no synthesis')):
                self.assertEqual(pre.inspect(root)['science_calls'],0)
                (root/'code.py').write_text('changed\n')
                with self.assertRaises(ValueError):pre.inspect(root)


if __name__ == '__main__':
    unittest.main()
