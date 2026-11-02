"""Synthetic references, toy circuits, fake processes; no molecular input loads."""
from copy import deepcopy
import cmath
import importlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pf_direct_retry106 as r
import pf_direct_compile_contract as c
from pf_direct_control_reference import policy_for,bind_references,OVERLAY
from observe_pf_direct_retry216 import main as observe


def task(method='B0I',prefix=1,formula='2nd',q=2):
    return dict(system='H2_DF3',candidate_id=f'{method}_p{prefix}_q{q}_{formula}',prefix=str(prefix),PF=formula,q=str(q))


def synthetic_row(t,*,C=.23,base=.3+.4j):
    policy=policy_for(t);rotated=base*cmath.exp(-.8j*C) if policy['method']=='B0I' else base
    pack=lambda z:dict(real=z.real,imag=z.imag)
    return dict(cell=dict(id=t['candidate_id'],method=policy['method'],prefix=int(t['prefix']),q=int(t['q']),formula=t['PF']),
        signals=dict(corrected=pack(rotated),exact_truncated=pack(rotated)),
        independent_forward_signals=dict(corrected=pack(base),exact_truncated=pack(base)),
        allowance=dict(policy=OVERLAY),normalized_deterministic=True,
        scalar_receipt=dict(policy='EXACT_STORED_FULL_FOCK_TAIL_IDENTITY',C=C,scalar=1.+C))


class ReferenceTests(unittest.TestCase):
    def test_B0I_forward_rotated_once(self):
        t=task();row=synthetic_row(t);before=deepcopy(row)
        expected,receipt=bind_references(t,row,policy_for(t))
        self.assertAlmostEqual(abs(expected[0]-expected[1]),0.,places=14)
        self.assertGreater(abs(expected[0]-complex(.3,.4)),.01)
        self.assertEqual(receipt['forward_rotation_angle'],-.8*.23)
        self.assertEqual(row,before)

    def test_single_term_S4_uses_already_saved_exact_field(self):
        for method in ('B0','B0I'):
            t=task(method,0,'4th');row=synthetic_row(t);row['signals']['corrected']=dict(real=-.9,imag=.01)
            expected,receipt=bind_references(t,row,policy_for(t))
            self.assertAlmostEqual(abs(expected[0]-expected[1]),0.,places=14)
            self.assertEqual(receipt['field'],'exact_truncated')
            self.assertTrue(receipt['single_term_S4_saved_signal_inconsistent'])
            self.assertEqual(row['signals']['corrected'],dict(real=-.9,imag=.01))

    def test_single_term_S2_stays_on_corrected(self):
        self.assertEqual(policy_for(task(prefix=0))['field'],'corrected')

    def test_bad_identity_and_overlay_rejected(self):
        t=task();row=synthetic_row(t)
        for field,value in [('id','other'),('prefix',2),('q',4),('method','B0')]:
            changed=deepcopy(row);changed['cell'][field]=value
            with self.assertRaisesRegex(ValueError,'OWN_REFERENCE_IDENTITY'):bind_references(t,changed,policy_for(t))
        row['allowance']['policy']='unknown'
        with self.assertRaisesRegex(ValueError,'EXPLICIT_SAVED_B0_OVERLAY'):bind_references(t,row,policy_for(t))

    def test_no_outcome_dependent_reference_choice(self):
        t=task();p=policy_for(t);p['field']='exact_truncated'
        with self.assertRaisesRegex(ValueError,'FIXED_REFERENCE_POLICY'):bind_references(t,synthetic_row(t),p)

    def test_missing_and_nonfinite_reference_rejected(self):
        t=task();row=synthetic_row(t);row['independent_forward_signals'].clear()
        with self.assertRaises(KeyError):bind_references(t,row,policy_for(t))
        row=synthetic_row(t);row['signals']['corrected']['real']=float('nan')
        with self.assertRaisesRegex(ValueError,'FINITE_SAVED_REFERENCE'):bind_references(t,row,policy_for(t))


class ScopeTests(unittest.TestCase):
    def test_exact_106_disjoint_682(self):
        _,rows=r.expected_tasks();old=c.read_json(r.OLD_MEASUREMENTS_PATH)['rows']
        self.assertEqual(len(rows),106);self.assertEqual(len(old),682)
        self.assertFalse({v['task_id'] for v in rows}&{v['task_id'] for v in old})
        policies=r.reference_bindings(rows)
        self.assertEqual(len(policies),53)
        self.assertEqual(sum(v['single_term_S4_saved_signal_inconsistent'] for v in policies.values()),10)
        caps=c.call_caps(rows);self.assertEqual((caps['native_evolution'],caps['control_probe'],caps['wrapper_compile']),(53,212,106))
        for name in c.FORBIDDEN_CALLS:self.assertEqual(caps[name],0)

    def test_missing_grant_rejected_before_scope_access(self):
        with patch.object(r,'expected_tasks',side_effect=AssertionError('scope accessed')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_RETRY106_GRANT_REQUIRED'):r.authorize({}, {})

    def test_metadata_import_does_not_import_science(self):
        cmd="import sys;sys.path.insert(0,'scripts');import pf_direct_retry106;assert all(x not in sys.modules for x in ('numpy','qiskit','trotterlib','pyscf'))"
        subprocess.run([sys.executable,'-B','-c',cmd],cwd=ROOT,check=True)

    def test_saved_audit_rejects_missing_qualification_or_failed_control(self):
        t=task(prefix=0,formula='4th');policy=policy_for(t);key=t['system']+':'+t['candidate_id']
        files={'control_reference.json':dict(policy),'control_diagnostics.json':dict(reference_policy=policy,
            status='PASS',all_errors_finite=True,errors=[1e-14],tolerance=1e-9),
            'accepted_measurements.json':dict(rows=[dict(control_reference_binding=policy,
                requires_saved_classical_signal_reconciliation=True)])}
        spec=dict(control_reference_bindings={key:policy})
        r.validate_saved_worker([t],files,{},spec)
        files['accepted_measurements.json']['rows'][0]['requires_saved_classical_signal_reconciliation']=False
        with self.assertRaisesRegex(ValueError,'ROW_REFERENCE_QUALIFICATION'):r.validate_saved_worker([t],files,{},spec)
        files['control_diagnostics.json']['status']='STOP_CONTROL'
        with self.assertRaisesRegex(ValueError,'SAVED_CONTROL_DIAGNOSTICS_PASS'):r.validate_saved_worker([t],files,{},spec)


class ToyNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read only the saved provider path, never its molecular snapshots.
        spec=c.read_json(r.PREVIOUS_PRIVATE/'sealed01/current_remaining_216_spec.json')
        cls.provider=Path(spec['systems']['H2_DF3']['config']['provider_root'])
        sys.path.insert(0,str(cls.provider/'src'))

    def test_old_one_term_iterator_has_extra_time(self):
        iterator=runpy.run_path(str(self.provider/'src/trotterlib/pf_decomposition.py'))['iter_pf_steps']
        w=1/(2-2**(1/3));middle=1-2*w
        steps=list(iterator(1,[middle,w]))
        self.assertEqual(len(steps),5)
        self.assertAlmostEqual(sum(value for _,value in steps),2-w,places=14)
        self.assertGreater(abs(sum(value for _,value in steps)-1),.3)
        self.assertAlmostEqual(2*w+middle,1.,places=14)

    def test_real_native_one_term_S4_matches_toy_analytic_reference(self):
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare_discard
        from trottertracks.resource_applicability import h6_matched_port_v1 as m
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        from pf_direct_execution import instruction_bound
        diagonal=np.array([.11,-.23,.37,.41]);constant=.13
        ham=DFHamiltonian(constant,np.diag(diagonal),np.array([.2]),(np.eye(4),),{})
        prep=_prepare_discard(ham,0)
        state=np.zeros(16,dtype=complex);state[0]=state[-1]=1/math.sqrt(2)
        low=np.concatenate((state,np.zeros_like(state)));high=np.concatenate((np.zeros_like(state),state))
        for q in (1,2,4,8,16):
            for C in (0.,.23):
                evolution=native.build_deterministic_native(prep.deterministic_blocks,num_system_qubits=4,T=.8,q=q,
                    formula='4th',scalar=constant+C,control_policy='symmetric_directional',
                    max_instructions=instruction_bound(native,prep.deterministic_blocks,q,'4th'))
                left=m.simulate_statevector(evolution.circuit,low);right=m.simulate_statevector(evolution.circuit,high)
                exact=(cmath.exp(-.8j*(constant+C))+cmath.exp(-.8j*(constant+C+sum(diagonal))))/2
                self.assertLess(float(np.linalg.norm(left-low)),1e-11)
                self.assertLess(abs(complex(np.vdot(state,right[16:]))-exact),1e-11)


class ObserverInjectionTests(unittest.TestCase):
    def test_injected_commands_are_owned_and_reaped(self):
        for auditor_code in (0,2):
            with tempfile.TemporaryDirectory(prefix='pf106-fake-observer-') as directory:
                private=Path(directory);seal=private/'seal';seal.mkdir();observer=private/'observer01';observer.mkdir()
                spec=dict(SOURCE='e'*40,observer_python=sys.executable)
                r.save(seal/'fake_spec.json',spec);r.save(seal/'fake_grant.json',{})
                def command(action,*_):
                    if action=='run':code='from pathlib import Path;Path('+repr(str(private/'run01'))+').mkdir();Path('+repr(str(private/'run01/terminal.json'))+").write_text('{}')"
                    else:code='raise SystemExit('+str(auditor_code)+')'
                    return [sys.executable,'-B','-c',code]
                api=SimpleNamespace(GROUPS=('fake',),PRIVATE=private,ROOT=ROOT,save=r.save,sha=r.sha,
                    authorize=lambda *_:None,gate=lambda *_:dict(status='PASS'),command=command)
                self.assertEqual(observe(seal,observer,api=api),0 if auditor_code==0 else 2)
                for name,expected in [('coordinator',0),('metadata_auditor',auditor_code)]:
                    receipt=c.read_json(observer/(name+'_native_exit.json'))
                    self.assertEqual(receipt['exit_code'],expected);self.assertTrue(receipt['reaped'])


if __name__=='__main__':unittest.main()
