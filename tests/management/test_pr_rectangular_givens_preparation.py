"""Artificial tokens/mocks only: no matrices, quantum circuits or science calls."""
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_dictionary_validation import rectangular as candidate
from trottertracks.pr_dictionary_validation import rectangular_preflight as preflight
from trottertracks.pr_dictionary_validation import rectangular_execution as execution
from trottertracks.pr_dictionary_validation import circuits
from trottertracks.pr_dictionary_validation.prototype import MixedAtom


def atom(word,kind='FRAME_Z',coefficient='-1',frame='F1'):
    return MixedAtom(word,kind,Fraction(coefficient),('mock',),tuple(word),
        'mock_frame' if kind=='FRAME_Z' else 'mock_pauli' if kind!='IDENTITY' else None,
        frame if kind=='FRAME_Z' else None)


class FakeCircuit:
    def __init__(self,*args): self.operations=[]
    def copy(self): return self
    def inverse(self): return 'actual_inverse_of_mock_G'
    def append(self,gate,qubits): self.operations.append((gate.name,tuple(qubits)))
    def compose(self,other,**kwargs): self.operations.append(('compose',other))
    def p(self,angle,qubit): self.operations.append(('p',angle,qubit))
    def rz(self,angle,qubit): self.operations.append(('rz',angle,qubit))
    def cx(self,a,b): self.operations.append(('cx',a,b))
    def cz(self,a,b): self.operations.append(('cz',a,b))


class PreparationTests(unittest.TestCase):
    def tearDown(self):
        for name in ('qiskit','numpy','mpmath','qiskit_nature','trotterlib'):
            self.assertNotIn(name,sys.modules)

    def test_helper_prefix_is_exact_not_global_X_filter(self):
        gate=lambda name:SimpleNamespace(name=name)
        register=['q0','q1','q2','q3']
        suffix=(gate('xx_plus_yy'),('q2','q1'))
        self.assertEqual(list(candidate.gaussian_suffix([
            (gate('x'),('q0',)),(gate('x'),('q1',)),suffix],register,2)),[suffix])
        for invalid in ([(gate('x'),('q1',))],[],
                [(gate('x'),('q0',)),(gate('x'),('q2',))],
                [(gate('x'),('q0',)),(gate('xx_plus_yy'),('q3','q0'))]):
            with self.assertRaises(ValueError):list(candidate.gaussian_suffix(invalid,register,1))

    def test_adapter_passes_only_selected_transposed_copy_and_no_initialization(self):
        selected=Mock();working_copy=object();selected.T.copy.return_value=working_copy
        class Frame:
            shape=(4,4)
            def __len__(self):return 4
            def __getitem__(self,key):
                self.key=key;return selected
        frame=Frame();np=SimpleNamespace(asarray=Mock(return_value=frame),
            isfinite=Mock(return_value=SimpleNamespace(all=lambda:True)))
        def helper(register,matrix):
            self.assertIs(matrix,working_copy)
            yield SimpleNamespace(name='x'),(register[0],)
            yield SimpleNamespace(name='x'),(register[1],)
            yield SimpleNamespace(name='xx_plus_yy'),(register[2],register[1])
        module='qiskit_nature.second_q.circuit.library.initial_states.utils.givens_rotations'
        with patch('importlib.metadata.version',return_value='0.7.2'),patch.dict(sys.modules,{
                'numpy':np,'qiskit':SimpleNamespace(QuantumCircuit=FakeCircuit,
                    QuantumRegister=lambda n:list(range(n))),
                module:SimpleNamespace(_prepare_slater_determinant_jw=helper)}):
            circuit=candidate.rectangular_gaussian('mock_frame',(2,3))
        self.assertEqual(frame.key,(slice(None),[2,3]))
        self.assertEqual(circuit.operations,[('xx_plus_yy',(2,1))])

    def test_provider_keeps_original_identity_and_shared_runs_legacy(self):
        provider=candidate.RectangularBasisProvider(4,{'F1':'mock_frame'})
        a=atom('IIIZ');b=atom('IIZI');g=FakeCircuit()
        with (patch.object(candidate,'rectangular_gaussian',return_value=g) as synth,
                patch.object(circuits.BasisProvider,'basis',return_value='legacy') as legacy):
            provider.basis(a,True);provider.basis(a,True);provider.basis(b,True)
            self.assertEqual(synth.call_count,2)
            self.assertEqual(provider.reference_support(a,True),(0,))
            self.assertEqual(provider.reference_support(b,True),(0,))
            self.assertEqual(provider.reference_support(atom('IIZZ'),True),(0,1))
            self.assertEqual(provider.reference_support(a,False),(3,))
            self.assertEqual(provider.basis(a,False),'legacy')
            legacy.assert_called_once_with(a,False)
            self.assertEqual(a.support,(3,));self.assertEqual(a.basis_key,'mock_frame')

    def test_router_uses_canonical_central_support_and_actual_inverse_only_for_singleton(self):
        g=FakeCircuit();provider=candidate.RectangularBasisProvider(4,{'F1':'mock'})
        a=atom('IIIZ');b=atom('IIZZ')
        event=lambda x:execution.BoundK0Rotation(x,0.125,'mock_law_hash')
        with (patch.dict(sys.modules,{'qiskit':SimpleNamespace(QuantumCircuit=FakeCircuit)}),
                patch.object(provider,'basis',return_value=g) as basis):
            circuit=circuits.build_event_sequence((event(a),),(a,),provider)
            self.assertEqual(circuit.operations,[('compose','actual_inverse_of_mock_G'),
                ('rz',-0.125,0),('cx',4,0),('rz',0.125,0),('cx',4,0),('compose',g)])
            basis.assert_called_once_with(a,singleton=True)
            shared=circuits.build_event_sequence((event(a),event(b)),(a,b),provider)
            basis.assert_called_with(a,singleton=False)
            self.assertIn(('rz',-0.125,3),shared.operations)
            self.assertIn(('cx',2,3),shared.operations)

    def test_saved_angle_binding_ignores_basis_RZ_and_rejects_sign_or_phase_drift(self):
        frame=atom('IIIZ');identity=atom('IIII','IDENTITY','1')
        tokens=[('rz',(1,),(999.0,)),('rz',(3,),(-0.125,)),('cx',(4,3),()),
            ('rz',(3,),(0.125,)),('cx',(4,3),()),('p',(4,),(-0.125,)),('p',(4,),(-0.0625,))]
        self.assertEqual(execution.bind_saved_angles(tokens,[frame,identity],-0.0625),0.125)
        with self.assertRaises(ValueError):execution.bind_saved_angles(tokens,[atom('IIIZ',coefficient='1'),identity],-0.0625)
        with self.assertRaises(ValueError):execution.bind_saved_angles(tokens,[frame,identity],-0.125)
        with self.assertRaises(ValueError):execution.bind_saved_angles(tokens+tokens,[frame,identity],-0.0625)

    def test_fixed48_inventory_and_no_old_grant_can_start_science(self):
        rows=preflight.cases()
        self.assertEqual(len(rows),48);self.assertEqual(len({r['id'] for r in rows}),48)
        self.assertEqual({r['partner'] for r in rows},{0,2})
        self.assertEqual({r['frame_atom'] for r in rows},set(range(20,26)))
        self.assertEqual(len({(r['i'],r['j']) for r in rows}),24)
        for scope in ('F1','F2','F2_RESOURCE','RECTANGULAR_PREPARATION'):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);(p/'authorization.json').write_text(json.dumps(dict(
                    stage=scope,authority='USER_EXPLICIT_F2_RESOURCE',status='APPROVED_BOUND_ONE_SHOT')))
                with self.assertRaises(PermissionError):
                    preflight.verify_execution(p,'missing',ROOT,'missing','missing',p/'output')
                self.assertFalse((p/'output').exists())
                self.assertEqual([x.name for x in p.iterdir()],['authorization.json'])

    def test_existing_claim_is_consumed_once_and_failed_output_creation_never_unclaims(self):
        from trottertracks.pr_dictionary_validation.authorization import claim_packet
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);v=dict(output=p/'out',consumption_path=p/'marker',authorization_digest='mock_digest')
            claim_packet(v)
            with self.assertRaises(FileExistsError):claim_packet(v)
            self.assertEqual(v['consumption_path'].read_text(),'mock_digest\n')
            v['consumption_path']=p/'second_marker'
            with self.assertRaises(FileExistsError):claim_packet(v)
            self.assertTrue(v['consumption_path'].exists())

    def test_artificial_fully_bound_future_packet_connects_to_existing_claim(self):
        from trottertracks.pr_dictionary_validation.authorization import claim_packet,file_sha,_canonical
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);approval=b'ARTIFICIAL TEST ONLY';(p/'user_approval.txt').write_bytes(approval)
            out=p/'source/artifacts/pr_rectangular_givens/attempt';out.parent.mkdir(parents=True)
            plan=dict(source_files={'mock.py':'mock_sha'},dependency_versions={},nature_helper_hashes={})
            v=dict(plan=plan,source_root=p/'source',plan_sha256='mock_plan')
            grant=dict(stage=preflight.STAGE,authority='USER_EXPLICIT_RECTANGULAR_PS48',
                status='APPROVED_BOUND_ONE_SHOT',retry_allowed=False,mandatory_stop=True,
                approval_sha256=file_sha(approval),plan_sha256='mock_plan',source_commit='mock_head',
                source_closure_sha256=file_sha(_canonical(plan['source_files'])),
                output=str(out),grant_id='rectangular_mock_ONLY')
            (p/'authorization.json').write_text(json.dumps(grant))
            with (patch.object(preflight,'inspect_preparation',return_value=v),
                    patch.object(preflight.subprocess,'check_output',return_value='mock_head\n'),
                    patch.object(preflight.sys,'version_info',(3,11,9)),
                    patch.object(preflight.sys,'prefix','mock_venv')):
                verified=preflight.verify_execution(p,'unused','unused','unused','unused',out)
            claim_packet(verified)
            self.assertTrue(out.is_dir())
            self.assertEqual(verified['consumption_path'].read_text(),verified['authorization_digest']+'\n')


if __name__=='__main__':unittest.main()
