"""Fabricated 2-mode metadata and mocked primitive producer only."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj


a=load('_v2_adapter',ROOT/'src/trotterlib/native_df_h6_cost_adapter.py')
graph_api=load('_v2_archive',ROOT/'src/trotterlib/native_df_archive_metadata.py')
model=load('_v2_model',ROOT/'src/trotterlib/native_df_gate_model.py')
schema=load('_v2_schema',ROOT/'src/trotterlib/native_df_metadata_binding.py')
old_fixture=load('_v2_old_fixture',ROOT/'tests/test_native_df_archive_metadata.py')
binding_fixture=load('_v2_binding_fixture',ROOT/'tests/test_native_df_metadata_binding.py')


def fixture():
    archive,ops,runtime,bh=old_fixture.fixture();nodes=archive['codec']['nodes']
    def add(kind, **fields):
        node=dict(type=kind,fields=fields);key=graph_api.digest(node);nodes[key]=node
        return {'ref':key}
    def seq(items):
        node=dict(type='tuple',items=items);key=graph_api.digest(node);nodes[key]=node
        return {'ref':key}
    def h(x):return {'float_hex':float(x).hex()}
    block=add('DFDeterministicOneBodySpec',block_id='one',order_index=0,original_fragment_index=None,
        basis_id='synthetic',basis_hash=bh,basis_change_operations=ops,runtime_basis_operations=runtime,
        diagonal_eigenvalues=seq([h(.2),h(.4)]))
    components=[]
    for i,probability in enumerate((.25,.75)):
        components.append(add('RTEComponent',component_id='C'+str(i),basis_id='synthetic',basis_hash=bh,
            df_fragment_id=0,coefficient_sign=1,is_identity=False,diagonal_pauli_support=seq([i]),
            basis_change_operations=ops,probability=h(probability),coefficient_abs=h(probability)))
    declaration=add('DFTailExtractionMetadata',randomized_component_count=2)
    tail=add('SymbolicRTETail',components=seq(components),extraction_metadata=declaration)
    rte=add('DFRTEEventPreparation',symbolic_tail=tail)
    root=add('DFPartialS2Preparation',num_system_qubits=2,ld=0,hamiltonian_hash='a'*64,
        preparation_hash='b'*64,deterministic_blocks=seq([block]),rte_preparation=rte,
        constant_coefficient=h(.1),extracted_identity_coefficient=h(0.),exact_rte_lambda_r=h(1.))
    archive['codec']['roots']={'fabricated':root}
    archive['codec_digest']=graph_api.digest(archive['codec'])
    return graph_api.FrozenMetadataGraph(archive)


class MockPermit:
    law_policy='EXACT_RENORMALIZATION_OF_SAVED_BINARY64_WEIGHTS'
    def __init__(self,approved=True):
        self.approved=approved;self.used={};self.primitive_cache={};self.occurrence_cache={}
    def require(self,scope):
        if not self.approved or scope!='h6_v2_numeric':raise ValueError('NO_NUMERIC_AUTHORIZATION')
    def verify_producer(self,producer):
        if producer is not PRODUCER:raise ValueError('WRONG_PRODUCER')
    def guard(self):pass
    def take(self,name):self.used[name]=self.used.get(name,0)+1
    def order_law(self,c,d):return .125,{0:Fraction(3,4),2:Fraction(1,4)}


PRODUCER=SimpleNamespace(one_body_diagonal_primitives=lambda coefficients,t:
    SimpleNamespace(rz=tuple((i,-t*v) for i,v in enumerate(coefficients)),rzz=(),global_phase=0.),
    df_squared_diagonal_primitives=lambda *args: (_ for _ in ()).throw(AssertionError('not a one-body fixture')))


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.graph=fixture();self.description=a.preparation_description(self.graph,'fabricated')

    def test_static_description_preserves_hex_without_numerical_evaluation(self):
        d=self.description
        self.assertEqual(d['blocks'][0]['diagonal_coefficients'][0],{'float_hex':float(.2).hex()})
        self.assertFalse(d['primitive_lists_evaluated']);self.assertFalse(d['probabilities_evaluated'])
        self.assertFalse(d['costs_evaluated']);self.assertEqual(len(d['components']),2)

    def test_existing_schema_adapter_keeps_ordered_qubits_and_signed_zero(self):
        basis=a.basis_object(self.description['blocks'][0]['basis'],model,schema)
        self.assertEqual(basis.operations[0].qubits,(1,0))
        self.assertEqual(basis.operations[0].angle.hex(),'-0x0.0p+0')

    def test_missing_root_rejected(self):
        with self.assertRaisesRegex(ValueError,'MISSING_FROZEN_ROOT'):
            a.preparation_description(self.graph,'nonexistent')

    def test_basis_parameter_tampering_rejected(self):
        record=self.description['blocks'][0]['basis'];record['operations'][0]['parameters'][0]='0.0'
        with self.assertRaises(ValueError):a.basis_object(record,model,schema)

    def test_permit_before_producer_or_primitive_work(self):
        p=MockPermit(False)
        with self.assertRaisesRegex(ValueError,'NO_NUMERIC_AUTHORIZATION'):
            a.model_inputs(self.description,model,schema,PRODUCER,p)
        self.assertEqual(p.used,{})

    def test_producer_identity_is_mandatory(self):
        with self.assertRaisesRegex(ValueError,'WRONG_PRODUCER'):
            a.model_inputs(self.description,model,schema,SimpleNamespace(),MockPermit())

    def test_complete_component_law_and_cached_original_primitive_calls(self):
        p=MockPermit();bases,primitive,components=a.model_inputs(self.description,model,schema,PRODUCER,p)
        self.assertEqual([c.probability for c in components],[Fraction(1,4),Fraction(3,4)])
        self.assertEqual(sum(c.probability for c in components),1)
        self.assertIs(primitive(0,.2),primitive(0,.2))
        self.assertEqual(p.used['timed_primitive_records'],1)
        self.assertEqual(primitive(0,-.2).rz[0][1],.04+6.938893903907228e-18)

    def test_unrecognized_probability_renormalization_rejected(self):
        p=MockPermit();p.law_policy='OBSERVED_FREQUENCIES'
        with self.assertRaisesRegex(ValueError,'PROBABILITY_POLICY'):
            a.model_inputs(self.description,model,schema,PRODUCER,p)

    def test_zero_mass_declared_component_rejected(self):
        self.description['components'][0]['probability']={'float_hex':float(0).hex()}
        with self.assertRaisesRegex(ValueError,'NONPOSITIVE'):
            a.model_inputs(self.description,model,schema,PRODUCER,MockPermit())

    def test_missing_component_mass_does_not_become_zero(self):
        self.description['components'].pop()
        with self.assertRaisesRegex(ValueError,'MASS_MISMATCH'):
            a.model_inputs(self.description,model,schema,PRODUCER,MockPermit())

    def test_saved_Taylor_law_all_orders_required(self):
        p=MockPermit();p.order_law=lambda c,d:(.125,{0:Fraction(1)})
        cell=dict(prefix=0,q=2,r=1,K=2,R=2,formula='2nd',method='B2')
        with self.assertRaisesRegex(ValueError,'SUPPORTED_TAYLOR_ORDER'):
            a.evaluate_candidate(self.description,cell,model=model,schema=schema,producer=PRODUCER,
                permit=p,variant='B',axis='cosine',scalar_phase=-.08)

    def test_unseen_positive_order_included_in_artificial_expectation(self):
        p=MockPermit();cell=dict(prefix=0,q=2,r=1,K=2,R=2,formula='2nd',method='B2')
        out=a.evaluate_candidate(self.description,cell,model=model,schema=schema,producer=PRODUCER,
            permit=p,variant='B',axis='cosine',scalar_phase=-.08)
        self.assertGreater(out.upper.rz,0);self.assertGreater(out.transitions,0)

    def test_native_policy_uses_A_prefix_and_B_event(self):
        seen={}
        class Spy:
            def __getattr__(self,name):return getattr(model,name)
            def partial_prefix_circuit(self,*args,**kw):
                seen['prefix']=kw['model'];return model.partial_prefix_circuit(*args,**kw)
            def expected_rte_occurrence(self,*args,**kw):
                seen['event']=kw['model'];return model.expected_rte_occurrence(*args,**kw)
        a.evaluate_candidate(self.description,dict(prefix=0,q=2,r=1,K=2,R=2,formula='2nd',method='B2'),
            model=Spy(),schema=schema,producer=PRODUCER,permit=MockPermit(),
            variant='literal_native',axis='sine',scalar_phase=-.08)
        self.assertEqual(seen,dict(prefix='A',event='B'))

    def test_formula_and_candidate_binding_cannot_change(self):
        cell=dict(prefix=1,q=2,formula='2nd',method='B0')
        with self.assertRaisesRegex(ValueError,'CANDIDATE_PREPARATION'):
            a.evaluate_candidate(self.description,cell,model=model,schema=schema,producer=PRODUCER,
                permit=MockPermit(),variant='A',axis='cosine',scalar_phase=-.08)

    def test_unsupported_PR_S4_rejected(self):
        cell=dict(prefix=0,q=2,r=1,K=2,R=2,formula='4th',method='B2')
        with self.assertRaisesRegex(ValueError,'S2'):
            a.evaluate_candidate(self.description,cell,model=model,schema=schema,producer=PRODUCER,
                permit=MockPermit(),variant='A',axis='cosine',scalar_phase=-.08)

    def test_rotation_classes_distinguish_Clifford_and_nonClifford(self):
        circuit=model.Circuit((model.Gate('rz',(0,),0.),model.Gate('rz',(0,),math.pi/2),
                               model.Gate('rz',(0,),.17),model.Gate('h',(0,))))
        classes=a.rotation_classes(circuit,model)
        self.assertEqual((classes['identity_RZ'],classes['Clifford_RZ'],classes['nonClifford_RZ']),(1,3,1))
        self.assertIsNone(classes['T_count']);self.assertIsNone(classes['Toffoli_count'])

    def binding(self):
        row=binding_fixture.synthetic_binding();raw=b'fabricated receipt contents'
        for k in row:
            if k.endswith('receipt_sha256') or k in ('fragment_order_sha256','basis_registry_sha256','component_law_sha256'):
                row[k]=hashlib.sha256(raw).hexdigest()
        row['receipt_semantics_verified']=True
        return row,raw

    def test_receipt_bytes_and_existing_envelope_jointly_required(self):
        row,raw=self.binding();out=a.validate_full_binding(row,deepcopy(row),schema=schema,receipt_loader=lambda f:raw)
        self.assertEqual(out['status'],'SOURCE_BOUND_RECEIPTS_VALIDATED')
        self.assertFalse(out['receipt_contents_certified'])

    def test_changed_receipt_content_stops(self):
        row,raw=self.binding()
        with self.assertRaisesRegex(ValueError,'RECEIPT_BYTES'):
            a.validate_full_binding(row,deepcopy(row),schema=schema,receipt_loader=lambda f:b'changed')

    def test_hashes_alone_do_not_certify_receipt_semantics(self):
        row,raw=self.binding();row['receipt_semantics_verified']=False
        with self.assertRaisesRegex(ValueError,'CONTENT_VALIDATION'):
            a.validate_full_binding(row,deepcopy(row),schema=schema,receipt_loader=lambda f:raw)

    def pair(self):
        expected=dict(cell_id='FABRICATED',execution_source='a'*40,dataset_seed_source='b'*40,
            input_sha256='c'*64,compiler_hash='d'*64,profile_hash='e'*64,sample_scope='confirmation32',
            replica=0,seed=1,event_digest='f'*64)
        return expected,[dict(expected,axis=ax,RZ=10,CX=20,status='MEASURED_COMPLETE') for ax in ('cosine','sine')]

    def test_paired_axes_have_exact_source_seed_and_scope(self):
        expected,rows=self.pair();self.assertTrue(a.paired_compiled_rows(rows,expected=expected))
        for key in ('execution_source','dataset_seed_source','sample_scope','event_digest','profile_hash'):
            bad=deepcopy(rows);bad[1][key]='changed'
            with self.assertRaises(ValueError):a.paired_compiled_rows(bad,expected=expected)

    def test_unresolved_dataset_source_is_not_execution_source(self):
        expected,rows=self.pair();expected['dataset_seed_source']=None
        with self.assertRaises(ValueError):a.paired_compiled_rows(rows,expected=expected)

    def test_partial_or_missing_axis_not_reused(self):
        expected,rows=self.pair()
        with self.assertRaises(ValueError):a.paired_compiled_rows(rows[:1],expected=expected)
        rows[0]['status']='PARTIAL'
        with self.assertRaises(ValueError):a.paired_compiled_rows(rows,expected=expected)

    def test_affine_mapping_and_residual_strata(self):
        rows=[dict(theory=x,compiled=3+2*x,stratum='PF_SOURCE_A',status='MATCHED_COMPLETE') for x in (10,20,30)]
        out=a.affine_correspondence(rows)
        self.assertEqual((out['intercept'],out['slope']),(3,2));self.assertEqual(out['relative_residuals'],[0,0,0])
        rows[0]['stratum']='PR_SOURCE_B'
        with self.assertRaises(ValueError):a.affine_correspondence(rows)

    def test_plan_H6_only_and_no_execution_authority(self):
        p=a.plan();self.assertFalse(p['scope']['H8']);self.assertFalse(p['approved'])
        self.assertFalse(p['runtime_authorization']);self.assertEqual(p['allowed_cpus'],[])
        self.assertEqual(p['demand']['native_full_wrapper_build'],0);self.assertEqual(p['demand']['new_compile'],0)

    def test_real_runner_refuses_unapproved_grant_before_science(self):
        runner=load('_v2_runner_guard_test',ROOT/'scripts/track_a_v2_runner.py')
        with self.assertRaisesRegex(ValueError,'SEALED_GRANT'):
            runner.verify_grant({},dict(approved_by_user=False),config_sha256='a',source='b',fresh={})

    def test_missing_order_never_becomes_observed_zero_probability(self):
        p=MockPermit();p.order_law=lambda c,d:(.125,{0:Fraction(1),2:Fraction(0)})
        with self.assertRaisesRegex(ValueError,'ZERO_SUPPORTED_TAYLOR_ORDER'):
            a.evaluate_candidate(self.description,dict(prefix=0,q=1,r=1,R=1,K=2,formula='2nd',method='B2'),
                model=model,schema=schema,producer=PRODUCER,permit=p,variant='A',axis='cosine',scalar_phase=0.)

    def test_saved_B0I_short_schema_normalized_using_declared_scope(self):
        item=dict(scope='H6_B0I_FAMILY12',cell_id='B0I_p10_q2_2nd',cell=dict(prefix=10,q=2,formula='2nd'))
        c=a.candidate_from_index(item)
        self.assertEqual(c['method'],'B0I');self.assertIsNone(c['r'])
        self.assertNotIn('method',item['cell'])
        item['cell_id']='B0_p10_q2_2nd'
        with self.assertRaises(ValueError):a.candidate_from_index(item)

    def test_paired_axes_reuse_bound_expectation_and_primitive_metadata(self):
        p=MockPermit();cell=dict(prefix=0,q=2,r=1,R=2,K=2,formula='2nd',method='B2')
        for axis in ('cosine','sine'):
            a.evaluate_candidate(self.description,cell,model=model,schema=schema,producer=PRODUCER,
                permit=p,variant='A',axis=axis,scalar_phase=-.08)
        self.assertEqual(p.used['expected_occurrence_evaluations'],1)
        self.assertEqual(p.used['timed_primitive_records'],1)
        self.assertEqual(p.used['model_candidate_axis_rows'],2)


if __name__=='__main__':unittest.main()
