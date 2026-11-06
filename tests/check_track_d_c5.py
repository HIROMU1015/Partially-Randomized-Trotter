"""Two-mode synthetic tests only: no molecule, sampler, transpile or Aer run."""
from dataclasses import replace
from fractions import Fraction
import importlib.util
import json
import hashlib
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_track_d_c2_q2 as pipeline
import run_track_d_c5_q2 as execution
pipeline.old().bootstrap(os.environ['TRACK_D_ORIGIN'])
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives
from trottertracks import track_d_whole_path_price as api
c3 = api.c3


class SyntheticFixture:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track-d-c5-synthetic-")
        self.root = Path(self.temp.name)
        for name in ['samples', 'circuits', 'references']:
            (self.root/name).mkdir()
        self.npatch = patch.object(c3.base, 'N', 2); self.npatch.start()
        blocks=[]; self.bases={}
        for i, theta in enumerate([.23, -.37, .41]):
            basis=QuantumCircuit(2);basis.append(XXPlusYYGate(theta,.17*i),[0,1])
            basis.rz(.13*(i+1),0);basis.global_phase=.07*(i+1)
            self.bases[str(i)]=basis
            blocks.append(S(block_id=str(i),rotation=((math.cos(theta),-math.sin(theta)),
                (math.sin(theta),math.cos(theta))),
                diagonal=DiagonalEvolutionPrimitives((.21+.1*i-.17)/2,
                    ((0,.21+.1*i),(1,-.17)),())))
        model=S(modes=2,blocks=blocks,constant=.17,label='synthetic',
                source_identity=api.c3.paired.ENDPOINT,input_identity=api.INPUT_PACKET)
        builder=c3.base.StrongBuilder(relative_fusion=True)
        builder._basis=lambda model, block:self.bases[block.block_id]
        builder.transition=lambda model, a,b:self.bases[a.block_id].compose(self.bases[b.block_id].inverse())
        atoms=[S(atom_id='plus',coefficient=Fraction(2,5),basis_key='p',support=(0,)),
               S(atom_id='minus',coefficient=Fraction(-7,10),basis_key='m',support=(0,))]
        self.atoms={a.atom_id:a for a in atoms};self.theta=math.atan(.2)
        self.steps={}
        for i in range(8):
            ids=['plus','plus'] if i%2==0 else ['minus','plus']
            events=[]
            for name in ids:
                a=self.atoms[name];sign=1 if a.coefficient>0 else -1
                app=S(coefficient_sign=sign,coefficient_abs=abs(float(a.coefficient)),
                      basis_id=a.basis_key,diagonal_pauli_support=a.support)
                events.append(S(rotation_component_id=name,application_sequence=(app,),
                    rotation_angle=self.theta,unsigned_rotation_angle=sign*self.theta,
                    event_normalization=1.02,phase=1+0j))
            self.steps[i]=tuple(events)
        half=tuple((i,Fraction(1,20)) for i in range(3))
        f=c3.base.control_one(builder.sequence(model,half,optimized=True,scalar_time=0))
        b=c3.base.control_one(builder.sequence(model,tuple(reversed(half)),optimized=True,scalar_time=0))
        scalar=-(.17+.04)*.1
        self.context=S(builder=builder,arm=S(model=model,plan=S(atoms=atoms,extracted_identity=.04)),
            forward=f,backward=b,scalar_phase=scalar,config=S(dimensionless_step_time=.2),
            provider=None,tail=S(lambda_r=2.))
        def tail(events,*args,**kwargs):
            out=QuantumCircuit(3)
            for event in events:
                basis=self.bases['1' if event.rotation_component_id=='plus' else '2']
                out.compose(basis.inverse(),range(2),inplace=True)
                out.crz(2*event.unsigned_rotation_angle,2,0)
                out.compose(basis,range(2),inplace=True)
            return out
        self.tail=tail
        self.patches=[patch.object(c3.paired,'saved_record',side_effect=lambda record,i:self.steps[i]),
                      patch.object(c3.base,'build_event_sequence',side_effect=tail)]
        for p in self.patches:p.start()
        for i in range(8):
            physical=f.compose(c3.base.control_one(tail(self.steps[i]))).compose(b)
            physical.global_phase+=scalar
            matrix=Operator(physical).data
            np.savez(self.root/f'references/sample{i:02d}.npz',system_reference=matrix)
            direct=QuantumCircuit(3);direct.append(UnitaryGate(np.block([[np.eye(4),np.zeros((4,4))],
                [np.zeros((4,4)),matrix]])),range(3))
            c3.base.save_qpy(self.root/f'circuits/sample{i:02d}_raw.qpy',direct)
            (self.root/f'samples/sample{i:02d}_events.json').write_text('{}')
        return self

    def __exit__(self,*args):
        for p in reversed(self.patches):p.stop()
        self.npatch.stop();self.temp.cleanup()


class C5Tests(unittest.TestCase):
    def test_public_projection_removes_operational_observations_without_mutation(self):
        original=dict(status='PASS',peak_RSS_KiB=123,wall_seconds=1.2,
            checks=[dict(max_entry=1e-14)],nested=dict(state_action_threads=4,phase=.3))
        value=execution.public_view(original)
        self.assertEqual(value,dict(status='PASS',checks=[dict(max_entry=1e-14)],nested=dict(phase=.3)))
        self.assertEqual(original['peak_RSS_KiB'],123)

    def test_reuse_requires_exact_raw_bytes_and_keeps_exact_compiled_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';local=root/'new'
            old.mkdir();(local/'circuits').mkdir(parents=True)
            members=[]
            with tarfile.open(old/'payload.tar.gz','w:gz') as tar:
                for method in api.METHODS:
                    for axis in api.AXES:
                        name=method+'_'+axis
                        for kind in ['raw','compiled']:
                            path='circuits/'+name+'_'+kind+'.qpy';data=(name+kind).encode()
                            info=tarfile.TarInfo(path);info.size=len(data);tar.addfile(info,io.BytesIO(data))
                            members.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                            if kind=='raw':(local/path).write_bytes(data if name=='reference_X' else b'different')
                        (old/(name+'_compile.json')).write_text(json.dumps(dict(status='PASS',compiler={'synthetic':True})))
            (old/'payload_archive.json').write_text(json.dumps(dict(archive='payload.tar.gz',members=members)))
            with patch.object(execution,'ROOT',root),patch.object(execution.c3,'C2_OUTPUT','old'):
                records=execution.reuse_compiles(local)
            self.assertTrue(records['reference_X']['reused'])
            self.assertEqual(sum(r['reused'] for r in records.values()),1)
            self.assertEqual((local/'circuits/reference_X_compiled.qpy').read_bytes(),b'reference_Xcompiled')
            self.assertEqual(json.loads((local/'reference_X_compile.json').read_text())['compile_calls'],0)

    def test_request_order_q_cut_and_source_rejections(self):
        r=api.OrderedRequest(2,(0,1),1)
        self.assertNotEqual(r.fingerprint,api.OrderedRequest(2,(1,0),1).fingerprint)
        for args in [(2,(0,),1),(2,(0,1),0),(2,(0,1),2),(True,(0,1),1),(2,(0,32),1)]:
            with self.subTest(args=args),self.assertRaises(ValueError):api.OrderedRequest(*args)
        for field in ['science_source','carrier','endpoint','cohort','cohort_source','law_sha256','input_packet_sha256']:
            with self.subTest(field=field),self.assertRaises(ValueError):replace(r,**{field:'wrong'})

    def test_common_path_AB_adjoint_vacuum_scalar_gaussian_signed_tail_and_XY(self):
        with SyntheticFixture() as f:
            r=api.OrderedRequest(2,(0,1),1);bundle=api.prepare(f.context,f.root,r)
            a,b,adj=[Operator(x).data for x in [bundle.later,bundle.earlier,bundle.later_adjoint]]
            np.testing.assert_allclose(a@b,bundle.reference,atol=1e-13,rtol=0)
            np.testing.assert_allclose(adj,a.conj().T,atol=1e-13,rtol=0)
            np.testing.assert_allclose(adj@a,np.eye(4),atol=1e-13,rtol=0)
            vacuum=np.zeros(4,complex);vacuum[0]=np.exp(1j*bundle.phase)
            np.testing.assert_allclose(bundle.reference[:,0],vacuum,atol=1e-13,rtol=0)
            self.assertEqual(bundle.identity['scalar_identity_occurrences'],2)
            self.assertEqual(bundle.identity['aggregated_scalar_phase'],2*f.context.scalar_phase)
            self.assertEqual(bundle.identity['strong_symmetric']['step_boundaries_fused'],1)
            self.assertEqual(bundle.identity['reference_optimization']['internal_step_boundaries_fused'],0)
            self.assertEqual(bundle.identity['reference_optimization']['global_cut_breaks_boundary_fusion'],1)
            actual={}
            for method in api.METHODS:
                for axis in api.AXES:
                    wrapper=api.wrapper_for(bundle,r,method=method,axis=axis)
                    obs=c3.base.artificial.induced_observable(wrapper);actual[method,axis]=obs
                    expected=c3.base.artificial.expected_observable(bundle.reference,axis)
                    np.testing.assert_allclose(obs,expected,atol=2e-13,rtol=0)
            np.testing.assert_allclose(actual['reference','X']+1j*actual['reference','Y'],bundle.reference,atol=3e-13,rtol=0)
            with self.assertRaises(ValueError):api.wrapper_for(bundle,api.OrderedRequest(2,(1,0),1),method='reference',axis='X')
            reversed_bundle=api.prepare(f.context,f.root,api.OrderedRequest(2,(1,0),1))
            self.assertGreater(float(np.max(abs(reversed_bundle.reference-bundle.reference))),1e-6)
            bad=api.c3.base.reference_gadget(bundle.later,bundle.earlier,0.)
            wrong=c3.base.artificial.induced_observable(c3.base.artificial.hadamard_wrapper(bad,'X'))
            self.assertGreater(float(np.max(abs(wrong-actual['reference','X']))),1e-4)

    def test_signed_tail_binding_fails_before_emission(self):
        with SyntheticFixture() as f:
            f.steps[0][0].application_sequence[0].coefficient_sign=-1
            with self.assertRaisesRegex(ValueError,'EVENT_SIGN_FRAME_OR_ANGLE'):
                api.prepare(f.context,f.root,api.OrderedRequest(2,(0,1),1))

    def test_context_endpoint_mismatch_is_rejected(self):
        with SyntheticFixture() as f:
            f.context.arm.model.source_identity='wrong'
            with self.assertRaisesRegex(ValueError,'CONTEXT_SOURCE_ENDPOINT_INPUT'):
                api.prepare(f.context,f.root,api.OrderedRequest(2,(0,1),1))

    def test_fixed_C3_entries_against_saved_source_on_two_mode_synthetic_inputs(self):
        original=subprocess.check_output(['git','-C',str(ROOT),'show',
            'cc554cf6daea08df6c20ea8e1e92405259ce49b5:src/trottertracks/track_d_c3_whole_path.py']).decode()
        with SyntheticFixture() as f:
            source=f.root/'old_c3.py';source.write_text(original)
            spec=importlib.util.spec_from_file_location('trottertracks._c3_fixed_synthetic',source)
            old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
            eye=np.eye
            with patch.object(old.np,'eye',side_effect=lambda n,*a,**kw:eye(4 if n==256 else n,*a,**kw)):
                for q in [4,8]:
                    prior=old.prepare(f.context,f.root,q);current=c3.prepare(f.context,f.root,q)
                    self.assertEqual(prior.identity,current.identity)
                    np.testing.assert_allclose(prior.reference,current.reference,atol=1e-13,rtol=0)
                    for name in api.METHODS:
                        np.testing.assert_allclose(Operator(prior.cores[name]).data,Operator(current.cores[name]).data,atol=1e-13,rtol=0)
            with self.assertRaisesRegex(ValueError,'ONLY_FIXED_Q4_Q8'):c3.prepare(None,None,2)

    def test_price_scope_qualification_and_unvalidated_fixture(self):
        r=api.OrderedRequest(2,(0,1),1)
        profile=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,
            basis_gates=['rz','sx','x','cx'],approximation_degree=1.,qubits_initially_zero=False)
        post=dict(status='PASS',arm='reference',axis='X',global_phase_fit=False,
                  metrics={**{k:7 for k in api.METRICS},'qubit_count':17})
        args=dict(method='reference',axis='X',postcheck=post,raw_status='PASS',paired_status='PASS',
                  execution_source='a'*40,compiler=profile)
        value=api.price_record(r,**args)
        self.assertEqual(value['estimate_kind'],'finite_fixture_compiled')
        self.assertIsNone(value['population_expectation']);self.assertIsNone(value['RPE_total'])
        self.assertEqual(value['peak_qubits'],17)
        for changes in [dict(cost_scope='native_kernel_repeat_with_fences'),dict(raw_status='FAIL'),
                        dict(paired_status=None),dict(compiler={**profile,'seed_transpiler':18})]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):api.price_record(r,**{**args,**changes})
        for request in [api.OrderedRequest(4,(0,1,2,3),2),api.OrderedRequest(2,(1,0),1)]:
            with self.assertRaisesRegex(ValueError,'UNVALIDATED_PRICE'):api.price_record(request,**args)
        for method,axis in [('ordinary','X'),('reference','Z')]:
            with self.assertRaises(ValueError):api.price_record(r,**{**args,'method':method,'axis':axis})


if __name__=='__main__':unittest.main()
