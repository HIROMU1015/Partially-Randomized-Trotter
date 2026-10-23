"""Artificial small circuits and authorization tests only; never read real H4."""
import copy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if 'd1_private' not in sys.modules:
    pkg=types.ModuleType('d1_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
    sys.modules['d1_private']=pkg
from d1_private import h4_d1_adapter as A

def basis():
    a=A.api();reg=a.DFBasisRegistry()
    definition=reg.register([(a.RZGate(.31),(0,)),(a.XXPlusYYGate(.42,.17),(0,1))],num_system_qubits=2)
    return json.loads(json.dumps(dict(basis_hash=definition.basis_hash,operations=[
        dict(qubits=list(op.qubits),description=asdict(op),matrix=[],inverse_matrix=[])
        for op in definition.metadata.operations])))

def event(k=0,df=False,b=None):
    ids=['0']*k+['1'];apps=[]
    for j,cid in enumerate(ids):
        apps.append(dict(application_index=j,role='rotation' if j==k else 'product',
            component_id=cid,coefficient_abs=1.,coefficient_sign=-1 if cid=='0' else 1,
            is_identity=False,df_fragment_id='f' if df else None,
            basis_id='df-basis-'+b['basis_hash'] if df else None,
            basis_hash=b['basis_hash'] if df else None,
            diagonal_pauli_support=([0] if cid=='0' else [0,1]) if df else None,
            basis_change_operations=[o['description'] for o in b['operations']] if df else []))
    return dict(taylor_order=k,rotation_component_id='1',product_component_ids=ids[:-1],
        selected_component_ids=ids,df_fragment_ids=['f' if df else None]*(k+1),
        event_probability=1.,event_coefficient=1.,event_normalization=1.,
        phase=dict(real=(-1)**(k//2),imag=0.),rotation_angle=.19,
        basis_id=apps[0]['basis_id'],basis_hash=apps[0]['basis_hash'],
        basis_reuse_intervals=[],application_sequence=apps)

def toy(rep,method):
    b=basis() if rep=='df_native' else None
    if b:
        blocks=[dict(operator_key='one_body',basis=b,primitives=dict(rz=[[0,.23],[1,-.13]],rzz=[],global_phase=.04)),
                dict(operator_key='0',basis=b,primitives=dict(rz=[[0,-.17]],rzz=[[0,1,.29]],global_phase=-.03))]
        comps=[dict(sign=-1,coefficient_abs=1.,support=[0],basis=b),
               dict(sign=1,coefficient_abs=1.,support=[0,1],basis=b)]
    else:
        blocks=[dict(operator_key='0',label='ZI',coefficient=.7),
                dict(operator_key='1',label='XX',coefficient=-.4)]
        comps=[dict(label='IZ',sign=-1,coefficient_abs=1.),dict(label='XX',sign=1,coefficient_abs=1.)]
    case=dict(config=dict(T=.4,q=2,r=1,K=4,method=method),blocks=blocks if method=='PF' else blocks[:1],
              components=comps if method=='PR' else [],native_scalar_phases=[.03,-.07])
    m=dict(representation=rep,n_qubits=2,step_time=.2,hamiltonian=dict(constant=.37),cases={method:case})
    records=[]
    if method=='PR':
        for step,k in enumerate((2,0)):
            records.append(dict(event(k,b is not None,b),outer_step_index=step,inner_RTE_position=0))
    join=dict(q=2,r=1,method=method,representation=rep,events=records)
    return m,join

class AdapterTests(unittest.TestCase):
    def assertMatrix(self,actual,expected):
        a=A.api()
        self.assertLess(float(a.np.max(a.np.abs(actual-expected))),4e-13)

    def test_event_roundtrip_and_order_rejection(self):
        for k in (0,2,4):
            e=event(k);self.assertEqual(json.loads(json.dumps(A.event_from_record(e).to_dict())),e)
        e=event(2);e['application_sequence'].reverse()
        with self.assertRaises(ValueError):A.event_from_record(e)

    def test_basis_roundtrip_and_hash_rejection(self):
        a=A.api();b=basis();reg=a.DFBasisRegistry();d=A.restore_basis(reg,b,2)
        q=a.QuantumCircuit(2);proxy=types.SimpleNamespace(runtime_basis_operations=d.runtime_operations)
        a.QiskitDFPartialS2CircuitBuilder._append_basis(q,proxy,inverse=True)
        a.QiskitDFPartialS2CircuitBuilder._append_basis(q,proxy,inverse=False)
        from qiskit.quantum_info import Operator
        self.assertMatrix(Operator(q).data,a.np.eye(4))
        b=copy.deepcopy(b);b['operations'][0]['description']['parameters']=['.32']
        with self.assertRaises(ValueError):A.restore_basis(a.DFBasisRegistry(),b,2)

    def test_pauli_full_and_ablation_control_branches(self):
        a=A.api();np=a.np
        from qiskit.quantum_info import Operator,Pauli
        def rotation(label,theta):
            p=Pauli(label).to_matrix();return math.cos(theta)*np.eye(4)-1j*math.sin(theta)*p
        for method in ('PF','PR'):
            model,join=toy('pauli_jw',method);case=model['cases'][method]
            for part in ('full','retained','event'):
                u=np.eye(4,dtype=complex)
                for step in range(2):
                    if part!='event':
                        for b in case['blocks']:u=rotation(b['label'],b['coefficient']*.1)@u
                    if part!='retained' and method=='PR':
                        e=join['events'][step]
                        for app in e['application_sequence']:
                            c=case['components'][int(app['component_id'])]
                            p=Pauli(c['label']).to_matrix()
                            op=c['sign']*p if app['role']=='product' else rotation(c['label'],c['sign']*.19)
                            u=op@u
                        u=e['phase']['real']*u
                    if part!='event':
                        for b in reversed(case['blocks']):u=rotation(b['label'],b['coefficient']*.1)@u
                        u=np.exp(-1j*.37*.2)*u
                built=A.build(model,method,join,part)
                body=Operator(built.body).data
                self.assertMatrix(body,u)
                h=np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2),np.eye(4))
                control=np.zeros((8,8),complex);control[:4,:4]=np.eye(4);control[4:,4:]=u
                self.assertMatrix(Operator(built.circuit.remove_final_measurements(inplace=False)).data,h@control@h)

    def test_df_native_control_vs_artificial_global_control(self):
        a=A.api();np=a.np
        from qiskit.quantum_info import Operator
        for method in ('PF','PR'):
            m,j=toy('df_native',method);case=m['cases'][method]
            for part in ('full','retained','event'):
                built=A.build(m,method,j,part)
                # Independent little-system oracle uses primitive matrices and saved event angles.
                reg=a.DFBasisRegistry();d=A.restore_basis(reg,basis(),2)
                bq=a.QuantumCircuit(2)
                for gate,qs in d.runtime_operations:bq.append(gate,list(qs))
                b=Operator(bq).data;u=np.eye(4,dtype=complex)
                z0=np.diag([1,-1,1,-1]);z1=np.diag([1,1,-1,-1])
                for step in range(2):
                    for reverse in (False,True):
                        if reverse and method=='PR' and part!='retained':
                            e=j['events'][step]
                            for app in e['application_sequence']:
                                z=z0 if app['component_id']=='0' else z0@z1
                                op=app['coefficient_sign']*z if app['role']=='product' else np.diag(np.exp(-.19j*app['coefficient_sign']*np.diag(z)))
                                u=b@op@b.conj().T@u
                            u=e['phase']['real']*u
                        if part!='event':
                            if not reverse:u=np.exp(1j*sum(case['native_scalar_phases']))*u
                            for block in (reversed(case['blocks']) if reverse else case['blocks']):
                                p=block['primitives'];diag=np.eye(4,dtype=complex)*np.exp(1j*p['global_phase'])
                                for left,right,angle in p['rzz']:diag=np.diag(np.exp(-.5j*angle*np.diag(z0@z1)))@diag
                                for qb,angle in p['rz']:diag=np.diag(np.exp(-.5j*angle*np.diag(z0 if qb==0 else z1)))@diag
                                u=b@diag@b.conj().T@u
                native=Operator(built.body).data
                self.assertMatrix(native[:4,:4],np.eye(4));self.assertMatrix(native[4:,4:],u)
                self.assertMatrix(native[:4,4:],np.zeros((4,4)))
                self.assertTrue(all(x['cancelled_basis_change_pairs']>=0 for x in built.sidecar['basis_counters']))

    def test_qpy_roundtrip_preserves_recursive_input(self):
        a=A.api()
        for rep in ('pauli_jw','df_native'):
            m,j=toy(rep,'PR');b=A.build(m,'PR',j)
            with tempfile.TemporaryDirectory() as td:
                saved=A.save_input(b,td)
                with (Path(td)/'input.qpy').open('rb') as f:c=a.qpy.load(f)[0]
                self.assertEqual(a.canonical_qiskit_circuit_fingerprint(c),saved['input_fingerprint'])

    def test_build_never_samples_or_prepares(self):
        a=A.api()
        targets=[('rte','sample_rte_events'),('rte','_make_event'),
                 ('pauli_signal_adapter','sample_rte_events'),
                 ('df_rte_circuit','sample_rte_events'),
                 ('df_partial_s2','prepare_df_partial_s2'),
                 ('rte_compiled_cost','transpile')]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for module,name in targets:
                stack.enter_context(patch.object(importlib.import_module(A.__package__+'.'+module),name,side_effect=AssertionError('FORBIDDEN:'+name)))
            for rep in ('pauli_jw','df_native'):
                m,j=toy(rep,'PR');A.build(m,'PR',j)

    def test_unsigned_identity_and_negative_phase(self):
        # Reuse an existing artificial native phase oracle without H4 data.
        spec=importlib.util.spec_from_file_location('d1_gate_tests',ROOT/'tests/test_native_df_gate_model.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        check=mod.MatrixChecks('test_identity_negative_product_phase_is_not_discarded')
        r=unittest.TestResult();check.run(r);self.assertTrue(r.wasSuccessful())

    def test_interaction_signed_counts(self):
        data={}
        for rep in ('pauli_jw','df_native'):
            for suffix,value in [('full',15),('retained',10),('event',8)]:
                data[rep+'_PR_'+suffix]=dict(RZ=value,CX=value)
        data['empty']=dict(RZ=2,CX=2)
        self.assertEqual(A.interactions(data)['pauli_jw']['RZ']['Delta_interaction'],-1)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        scope='ARTIFICIAL_ONLY_NO_REAL_H4_READ_OR_COST_COMPILE',
        project_modules={str(Path(m.__file__).resolve().relative_to(ROOT)):'' for name,m in sys.modules.items()
            if name.startswith('d1_private.') and getattr(m,'__file__',None) and ROOT in Path(m.__file__).resolve().parents})
    if len(sys.argv)>1:
        Path(sys.argv[1]).write_text(json.dumps(output,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
