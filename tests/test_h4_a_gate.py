"""Twelve artificial body-only/authority checks; no H4 input is built."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_h4_d1_adapter import A,toy
from test_h4_controlled_comparison import record,event_oracle,rotation,lift
from d1_private import h4_a_gate as D
from d1_private import h4_d2_pauli_control as P
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
import numpy as np
C=D.C


def fixture():
    model,join=toy('pauli_jw','PR');join=copy.deepcopy(join)
    labels=None
    for step,k in enumerate((2,2)):
        e,labels=record(k)
        join['events'][step]=dict(e.to_dict(),outer_step_index=step,inner_RTE_position=0)
    # Second event uses two positive, noncommuting products: its order phase
    # remains visible instead of cancelling the negative-product sign.
    second=join['events'][1]
    second['product_component_ids']=['1','3'];second['selected_component_ids']=['1','3','2']
    for app,cid in zip(second['application_sequence'],second['selected_component_ids']):
        app['component_id']=cid;app['coefficient_sign']=-1 if int(cid)%2==0 else 1
    model['cases']['PR']['components']=[dict(label=labels[str(i)],sign=-1 if i%2==0 else 1,coefficient_abs=1.) for i in range(5)]
    return model,join


def active():
    target=C.read(D.DESIGN/'cohort_manifest.json')['selected']
    return dict(kind=D.KIND,execution_authorized=True,dispatch_enabled=True,grant_valid=True,target=target,
        fixed=C.read(D.D2.BASE/'active_contract.json')['fixed'],call_limits=D.LIMITS,
        per_axis_compile_max=dict(cosine=1,sine=1),mandatory_STOP=True,automatic_retry=False,automatic_next_stage=False,
        science_SOURCE='TOY',seal_sha256='TOY',approval_sha256='TOY')


class GateTests(unittest.TestCase):
    def test_a_no_authority_no_quantum_import(self):
        code="""import sys,types,tempfile
from pathlib import Path
p=types.ModuleType('d1_private');p.__path__=[str(Path.cwd()/'src/trotterlib')];sys.modules['d1_private']=p
from d1_private import h4_a_gate as D
with tempfile.TemporaryDirectory() as t:
 D.BASE=Path(t)
 try:D.authority(consume=True)
 except FileNotFoundError:pass
 else:raise AssertionError('accepted')
assert not any(k=='qiskit' or k.startswith('qiskit.') for k in sys.modules)
"""
        self.assertEqual(subprocess.run([sys.executable,'-B','-c',code],cwd=D.ROOT).returncode,0)

    def test_b_body_tokens_and_phase_match_original_D1_schedule(self):
        m,j=fixture();old=A.build(m,'PR',j);new=D.body_only(m,j)
        inspector=D.D2.S.inspector()
        self.assertEqual([inspector.primitive_token(i,old.body) for i in old.body.data],
                         [inspector.primitive_token(i,new.body) for i in new.body.data])
        self.assertEqual(float(old.body.global_phase),float(new.body.global_phase))
        self.assertLess(float(np.max(np.abs(Operator(old.body).data-Operator(new.body).data))),2e-12)

    def test_c_body_only_never_calls_generic_wrapper_or_sampler(self):
        m,j=fixture();a=A.api()
        with ExitStack() as stack:
            for obj,name in [(A,'build'),(a,'hadamard_wrapper'),(QuantumCircuit,'to_gate'),(Gate,'control'),(a,'transpile_and_measure_cost')]:
                stack.enter_context(patch.object(obj,name,side_effect=AssertionError(name)))
            import importlib
            for name in ('sample_rte_events','_make_event'):
                stack.enter_context(patch.object(importlib.import_module('d1_private.rte'),name,side_effect=AssertionError(name)))
            result=D.body_only(m,j)
        self.assertEqual(result.body.num_qubits,2);self.assertEqual(result.body.num_clbits,0)

    def test_d_phase_ledger_is_original_event_then_constant_order(self):
        m,j=fixture();result=D.body_only(m,j)
        self.assertEqual([x['owner'] for x in result.sidecar['phase_updates']],['event','constant','event','constant'])
        self.assertEqual([x['role'] for x in result.sidecar['tags']],['retained','event','retained']*2)
        self.assertEqual(result.sidecar['aggregate_phase'],float(result.body.global_phase))

    def test_e_noncommuting_products_both_control_branches_and_axes(self):
        m,j=fixture();a=A.api();result=D.body_only(m,j);u=np.eye(4,dtype=complex)
        labels={str(i):c['label'] for i,c in enumerate(m['cases']['PR']['components'])}
        for step in range(2):
            u=rotation('ZI',.7*.1)@u;e=A.event_from_record(j['events'][step]);u=event_oracle(e,labels)@u
            u=rotation('ZI',.7*.1)@u;u=np.exp(-1j*.37*.2)*u
        controlled=P.central_control_body(result.body).body
        self.assertLess(float(np.max(np.abs(Operator(controlled).data-lift(u)))),2e-12)
        h=np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2),np.eye(4));sdg=np.kron(np.diag([1,-1j]),np.eye(4))
        for axis in D.AXES:
            q=a.hadamard_wrapper(controlled,axis,measure=False,already_controlled=True)
            expected=h@(sdg if axis=='sine' else np.eye(8))@lift(u)@h
            self.assertLess(float(np.max(np.abs(Operator(q).data-expected))),2e-12)

    def test_f_saved_order_or_q_r_corruption_rejected(self):
        m,j=fixture();j['events'][0]['application_sequence']=list(reversed(j['events'][0]['application_sequence']))
        with self.assertRaises(ValueError):D.body_only(m,j)
        m,j=fixture();j['r']=2
        with self.assertRaises(ValueError):D.body_only(m,j)

    def test_g_scope_rejects_new_target_more_calls_and_auto_stage(self):
        D.scope(active())
        for key,value in [('execution_authorized',False),('automatic_next_stage',True),('call_limits',{}),('target',{})]:
            c=active();c[key]=value
            with self.assertRaises((ValueError,KeyError)):D.scope(c)

    def test_h_grant_consumption_binding_and_STOP(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)),patch.object(D,'verify_seal'):
            base=Path(td);ns=base/'run01';ns.mkdir();s=dict(SOURCE='TOY',approval_sha256='TOY');C.save(base/'seal.json',s)
            c=active();c['seal_sha256']=C.sha(base/'seal.json');C.save(base/'active_contract.json',c)
            g=dict(kind='H4_A_GATE_ORDER2_ONE_SHOT',SOURCE='TOY',namespace=str(ns),contract_sha256=C.sha(base/'active_contract.json'),
                seal_sha256=C.sha(base/'seal.json'),approval_sha256='TOY',one_shot=True,automatic_retry=False,call_limits=D.LIMITS)
            g['grant_id']=C.digest(g);C.save(ns/'grant.json',g);D.authority(consume=True)
            with self.assertRaisesRegex(ValueError,'CONSUMED'):D.authority(consume=True)
            C.save(ns/'ENGINE_STOP.json',dict(status='TOY_STOP'))
            with self.assertRaisesRegex(ValueError,'STOPPED'):D.authority()

    def test_i_body_and_cost_call_limits_nonrefundable(self):
        with tempfile.TemporaryDirectory() as td:
            b=C.CallBudget(td,D.LIMITS);b.take('body_replay')
            with self.assertRaises(ValueError):b.take('body_replay')
            b.take('cost_transpile');b.take('cost_transpile')
            with self.assertRaises(ValueError):b.take('cost_transpile')
            with self.assertRaises(ValueError):b.take('sampling')
            self.assertEqual(C.read(Path(td)/'call_ledger.json')['used']['cost_transpile'],2)

    def test_j_stage_A_or_B_failure_prevents_wrappers_and_compile(self):
        for failed in ('stage_A','stage_B'):
            with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)):
                ns=Path(td)/'run01';ns.mkdir()
                with (patch.object(D,'authority',return_value=(active(),{},{})),patch.object(D,'stage_A',return_value=({}, {}, [])) as a,
                      patch.object(D,'stage_B',return_value=SimpleNamespace()) as b,patch.object(D,'wrappers') as w,patch.object(D,'compile_both') as c):
                    (a if failed=='stage_A' else b).side_effect=ValueError('TOY_FAILURE')
                    with self.assertRaises(ValueError):D.worker()
                    w.assert_not_called();c.assert_not_called()
                self.assertEqual(C.read(ns/'ENGINE_STOP.json')['status'],'A_GATE_FAILED_NO_RETRY')

    def test_k_reused_dispatcher_reserves_two_distinct_axis_calls(self):
        calls=[]
        class Child:
            pid=os.getpid()
            def __init__(self,command,**kw):calls.append(command)
            def poll(self):return 0
            def wait(self):return 0
        with tempfile.TemporaryDirectory() as td:
            ns=Path(td);C.save(ns/'grant.json',dict(grant_id='TOY'))
            with patch.object(D.D2.subprocess,'Popen',Child):D.compile_both(ns)
            self.assertEqual([c[-1] for c in calls],['cosine','sine'])
            self.assertTrue(all(str(D.SCRIPT) in c for c in calls))
            self.assertEqual(C.read(ns/'call_ledger.json')['used']['cost_transpile'],2)
            self.assertEqual(len(list((ns/'native_tasks').glob('*/native_exit.json'))),2)

    def test_l_tampered_input_and_duplicate_axis_block_compile(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)):
            ns=Path(td)/'run01';d=ns/'variants/cosine';d.mkdir(parents=True);r=ns/'native_tasks/compile_cosine';r.mkdir(parents=True)
            C.save(ns/'stage_C_inputs.json',dict(status='PASS_BOTH_AXIS_INPUTS_SAVED'))
            C.save(r/'reservation.json',dict(kind='compile',case='cosine',grant_id='TOY'))
            (d/'input.qpy').write_bytes(b'TAMPERED');C.save(d/'input.json',dict(input_qpy_sha256='0'*64))
            with patch.object(D,'authority',return_value=(active(),{},dict(grant_id='TOY'))),patch.object(A,'compile_saved') as compiler:
                with self.assertRaisesRegex(ValueError,'INPUT_CHANGED'):D.compile_task('cosine')
                compiler.assert_not_called()
                with self.assertRaises(FileExistsError):D.compile_task('cosine')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateTests))
    if len(sys.argv)==2:Path(sys.argv[1]).write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scope='ARTIFICIAL_ONLY_NO_REAL_H4'),indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
