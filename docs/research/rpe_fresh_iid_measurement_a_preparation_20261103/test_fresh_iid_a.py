"""Artificial fixtures only. Never prepare/decode the H4 numerical context."""
import ast
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict, FrozenInstanceError
import io
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import PhaseGate,RZGate,RZZGate,UnitaryGate
from qiskit.quantum_info import Operator

import fresh_iid_adapter as a
import run_fresh_iid_a as runner

TOY_NAMESPACE='synthetic_only_fresh_iid_a_tests_v1'


def toy():
    rte=a.legacy_rte()
    components=(rte.RTEComponent('x',.4,.4,1,basis_id='toy-x'),
                rte.RTEComponent('z',.6,.6,-1,basis_id='toy-z'))
    distribution=rte.finite_rte_distribution(.7,2)
    x=np.array([[0,1],[1,0]],complex);z=np.diag([-1.,1.]).astype(complex)
    payload={'psi':np.array([math.sqrt(.3),1j*math.sqrt(.7)]),
             'forward':np.diag(np.exp(1j*np.array([.1,-.1]))),
             'reverse':np.diag(np.exp(1j*np.array([-.2,.2]))),
             'operators':{'x':x,'z':z},'components':[asdict(c) for c in components],
             'distribution':distribution.to_dict()}
    return components,distribution,payload


def spec(q=1,axis='cosine',n=12,target=0.):
    return a.AxisInput(0 if q==1 else 3,q,1,2,axis,n,target,'.007',('.01','.02','.37'),'.1')


def primitive_functions():
    source=a.fixed_bytes(a.CARRIER,'src/trotterlib/df_trotter/ops.py').decode()
    names={'DiagonalEvolutionPrimitives','_as_real','_is_symbolic','_real_or_symbolic',
           '_is_effectively_zero','one_body_diagonal_primitives','df_squared_diagonal_primitives',
           'append_diagonal_primitives','_append_rzz'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    ast.fix_missing_locations(module)
    from dataclasses import dataclass
    ns={'np':np,'math':math,'dataclass':dataclass,'ParameterExpression':ParameterExpression,
        'RZGate':RZGate,'RZZGate':RZZGate,'PhaseGate':PhaseGate,'__name__':__name__}
    exec(compile(module,'<unchanged-legacy-primitives>','exec'),ns)
    return ns


class FreshIIDPreparationTests(unittest.TestCase):
    def test_fixed_metadata_and_old_allocations_transfer_without_reoptimization(self):
        axes=a.fixed_axis_inputs()
        self.assertEqual([(s.q,s.axis,s.shots) for s in axes],
                         [(1,'cosine',510),(1,'sine',510),(8,'cosine',454),(8,'sine',454)])
        with self.assertRaises(FrozenInstanceError):axes[0].shots=1
        self.assertTrue(all(s.r==1 and s.k==2 and s.signal_kind=='attenuated_event_mean_signal' for s in axes))

    def test_seed_keys_unique_reproducible_and_domain_separated(self):
        inputs=[spec(),spec(axis='sine'),spec(q=8),spec(q=8,axis='sine')]
        self.assertEqual(a.check_stream_keys(inputs,namespace=TOY_NAMESPACE),96)
        first=a.stream_seed(1,'cosine',0,'trajectory',namespace=TOY_NAMESPACE)
        self.assertEqual(first,a.stream_seed(1,'cosine',0,'trajectory',namespace=TOY_NAMESPACE))
        self.assertNotEqual(first,a.stream_seed(1,'cosine',0,'measurement',namespace=TOY_NAMESPACE))
        self.assertNotEqual(first,a.stream_seed(1,'sine',0,'trajectory',namespace=TOY_NAMESPACE))
        with self.assertRaises(ValueError):a.check_stream_keys([spec(),spec()],namespace=TOY_NAMESPACE)

    def test_legacy_event_law_order_phase_and_probability_are_preserved(self):
        components,d,_=toy();rte=a.legacy_rte()
        events=rte.enumerate_rte_events(components,d)
        self.assertEqual(len(events),10)
        self.assertAlmostEqual(sum(e.event_probability for e in events),1.)
        self.assertEqual({e.taylor_order for e in events},{0,2})
        for e in events:
            self.assertEqual(e.phase,1 if e.taylor_order==0 else -1)
            self.assertAlmostEqual(e.rotation_angle,math.atan(.7/(e.taylor_order+1)))
            self.assertEqual(e.event_normalization,d.exact_finite_distribution)

    def test_artificial_distribution_mean_has_one_normalization(self):
        components,d,payload=toy();rte=a.legacy_rte()
        events=rte.enumerate_rte_events(components,d)
        mean=sum(e.event_probability*rte.event_unitary(e,payload['operators']) for e in events)
        h=.4*payload['operators']['x']+.6*payload['operators']['z']
        expected=rte.finite_taylor_operator(h,.7,2)/d.exact_finite_distribution
        np.testing.assert_allclose(mean,expected,atol=1e-14)

    def test_every_shot_calls_sampler_even_if_event_sequences_coincide(self):
        components,d,_=toy();event=a.legacy_rte().enumerate_rte_events(components,d)[0]
        sampler=Mock(return_value=(event,))
        kwargs=dict(components=components,distribution=d,signal_fn=lambda *_:.3-.2j,sampler=sampler,namespace=TOY_NAMESPACE)
        first=a.one_shot(spec(),0,**kwargs);second=a.one_shot(spec(),1,**kwargs)
        self.assertEqual(sampler.call_count,2)
        self.assertNotEqual(sampler.call_args_list[0].kwargs['seed'],sampler.call_args_list[1].kwargs['seed'])
        self.assertEqual(first['events'],second['events'])  # Legitimate IID repeats are allowed.
        self.assertEqual(first['raw_conditional_mean'],.3)

    def test_sine_probability_and_no_normalization_reapplication(self):
        components,d,_=toy()
        row=a.one_shot(spec(axis='sine'),0,components=components,distribution=d,
                       signal_fn=lambda *_:.3-.2j,namespace=TOY_NAMESPACE)
        self.assertEqual(row['raw_conditional_mean'],-.2)
        self.assertAlmostEqual(row['p_plus'],.4)
        self.assertEqual(row['outcome'] in (-1,1),True)

    def test_roundoff_is_explicit_and_invalid_means_not_clamped(self):
        raw,used,p,flag=a.conditional_probability(1+2e-13,'cosine')
        self.assertTrue(flag);self.assertGreater(raw,1);self.assertEqual((used,p),(1.,1.))
        for z in (1.001,complex(math.nan,0),complex(0,math.inf)):
            with self.assertRaises(ValueError):a.conditional_probability(z,'cosine')
        self.assertEqual(a.conditional_probability(0,'sine')[2],.5)

    def test_corrected_and_control_labels_are_rejected(self):
        for changes in ({'signal_kind':'corrected_finite_rte_signal'},
                        {'state_label':'effective_partial_s2_eigenstate_control'}):
            with self.assertRaises(ValueError):
                a.AxisInput(**{**asdict(spec()),**changes})

    def test_missing_event_or_shot_records_stop(self):
        components,d,_=toy()
        with self.assertRaises(ValueError):
            a.one_shot(spec(),0,components=components,distribution=d,signal_fn=lambda *_:1,
                       sampler=Mock(return_value=()),namespace=TOY_NAMESPACE)
        with self.assertRaises(ValueError):a.summarize_axis(spec(),[])

    def test_same_synthetic_seed_config_reproduces_both_random_levels(self):
        _,_,payload=toy();context=runner.restore_payload(payload);job=(spec(),tuple(range(12)))
        first=runner.shot_chunk(job,context=context,namespace=TOY_NAMESPACE)
        second=runner.shot_chunk(job,context=context,namespace=TOY_NAMESPACE)
        self.assertEqual(first,second)
        with self.assertRaises(ValueError):runner.shot_chunk((spec(),(0,0)),context=context,namespace=TOY_NAMESPACE)

    def test_process_and_chunk_order_do_not_change_artificial_samples(self):
        _,_,payload=toy();context=runner.restore_payload(payload)
        s=spec(q=8,n=12);serial=runner.shot_chunk((s,tuple(range(12))),context=context,namespace=TOY_NAMESPACE)
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=runner.initialize_worker,initargs=(payload,)) as pool:
            right=pool.submit(runner.shot_chunk,(s,tuple(range(6,12))),namespace=TOY_NAMESPACE)
            left=pool.submit(runner.shot_chunk,(s,tuple(range(6))),namespace=TOY_NAMESPACE)
            parallel=left.result()+right.result()
        self.assertEqual(serial,parallel)

    def test_no_discrepancy_does_not_certify_equivalence_or_RPE_success(self):
        s=spec(n=12,target=1)
        rows=[{'q':1,'axis':'cosine','shot':i,'conditional_mean':1.,'outcome':1,'roundoff_adjusted':False} for i in range(12)]
        result=a.summarize_axis(s,rows)
        self.assertEqual(result['trajectory_conditional_mean_average'],1)
        self.assertEqual(result['measurement_sample_mean'],1)
        self.assertEqual(result['scientific_status'],'UNDETERMINED')
        self.assertFalse(result['equivalence_certified']);self.assertFalse(result['RPE_success_probability_evaluated'])
        self.assertEqual(a.DIAGNOSTIC_COMPARISONS,12)

    def test_a_large_artificial_discrepancy_is_flagged(self):
        s=spec(n=100,target=-1)
        rows=[{'q':1,'axis':'cosine','shot':i,'conditional_mean':1.,'outcome':1,'roundoff_adjusted':False} for i in range(100)]
        result=a.summarize_axis(s,rows)
        self.assertEqual(result['scientific_status'],'INCONSISTENT')
        self.assertFalse(result['checks']['trajectory_to_saved'])
        self.assertTrue(result['checks']['measurement_to_conditional_average'])

    def test_three_levels_of_means_are_kept_separate(self):
        s=spec(n=2,target=.25)
        rows=[{'q':1,'axis':'cosine','shot':i,'conditional_mean':v,'outcome':y,'roundoff_adjusted':False}
              for i,(v,y) in enumerate(((0.,-1),(.5,-1)))]
        result=a.summarize_axis(s,rows)
        self.assertEqual(result['saved_target_mean'],.25)
        self.assertEqual(result['trajectory_conditional_mean_average'],.25)
        self.assertEqual(result['measurement_sample_mean'],-1)

    def test_numeric_basis_order_and_sector_embedding_match_explicit_toy_circuit(self):
        rotation=np.eye(4,dtype=complex);theta=.3
        rotation[1:3,1:3]=[[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
        operations=((UnitaryGate(rotation),(0,1)),(RZGate(.17),(0,)))
        circuit=QuantumCircuit(2)
        for gate,qubits in operations:circuit.append(gate,list(qubits))
        indices=np.array([1,2]);permutation=np.array([1,0]);ordered=indices[permutation]
        actual,leak=runner.restricted_basis(operations,2,indices,permutation)
        np.testing.assert_allclose(actual,Operator(circuit).data[np.ix_(ordered,ordered)],atol=1e-14)
        self.assertEqual(leak,0.)

    def test_numeric_halves_match_legacy_primitives_including_scalar_on_toy(self):
        ns=primitive_functions();identity=np.eye(4,dtype=complex)
        b0=np.diag(np.exp(1j*np.array([.1,.2,-.2,-.1])))
        theta=.3;b1=identity.copy();b1[1:3,1:3]=[[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
        blocks=(SimpleNamespace(basis_id='a',basis_hash='A',runtime_basis_operations=(),kind=0),
                SimpleNamespace(basis_id='b',basis_hash='B',runtime_basis_operations=(),kind=1))
        prep=SimpleNamespace(deterministic_blocks=blocks,constant_coefficient=.13,extracted_identity_coefficient=-.17)
        basis={'a':b0,'b':b1}
        def primitive(block):
            return ns['one_body_diagonal_primitives'](np.array([.2,-.1]),.01) if block.kind==0 else ns['df_squared_diagonal_primitives'](np.array([.3,.8]),-.4,.01)
        actual=runner.numeric_half_operators(prep,lambda key,*_:basis[key],primitive,np.arange(4))
        references=[]
        for reverse in (False,True):
            circuit=QuantumCircuit(2)
            if not reverse:circuit.global_phase=-.02*(.13-.17)
            for block in (tuple(reversed(blocks)) if reverse else blocks):
                circuit.append(UnitaryGate(basis[block.basis_id].conj().T),[0,1])
                ns['append_diagonal_primitives'](circuit,primitive(block))
                circuit.append(UnitaryGate(basis[block.basis_id]),[0,1])
            references.append(Operator(circuit).data)
        for x,y in zip(actual,references):np.testing.assert_allclose(x,y,atol=1e-14)

    def test_default_cli_does_not_enter_science_or_any_rng_draw(self):
        with patch.object(runner,'execute',side_effect=AssertionError('science forbidden')),\
             patch.object(runner,'prepare_context',side_effect=AssertionError('H4 forbidden')),\
             patch.object(np.random,'PCG64',side_effect=AssertionError('RNG forbidden')),redirect_stdout(io.StringIO()) as output:
            runner.main([])
        self.assertEqual(json.loads(output.getvalue())['status'],'PREPARED_NOT_SCIENTIFICALLY_EXECUTED')

    def test_artificial_interrupt_preserves_stop_status_without_context_or_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'mock_output'
            with patch.object(runner,'prepare_context',side_effect=KeyboardInterrupt('synthetic interruption')),\
                 patch.object(a,'one_shot',side_effect=AssertionError('sampling forbidden')),\
                 self.assertRaises(KeyboardInterrupt):
                runner.execute(output)
            record=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(record['scientific_status'],'UNDETERMINED')
            self.assertEqual(record['execution_status'],'INTERRUPTED_OR_INVALID_MANDATORY_STOP')
            self.assertTrue(record['mandatory_STOP'])


if __name__=='__main__':unittest.main()
