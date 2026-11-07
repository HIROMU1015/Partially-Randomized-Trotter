"""Invented saved headers only. No matrices, states, Born, shots or compile."""
import ast
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import m_n0_header_binding as b
from trottertracks.algorithm_codesign import m_n0_metadata_execution as e
from trottertracks.algorithm_codesign import m_label_average as m

EVIDENCE=[]


def basis(n, *, theta=.29, empty=False):
    # Deliberately invented matrix-provenance strings, never actual gate matrices.
    ops=[] if empty else [dict(name='rz',qubits=[0],parameters=['0.13'],
                    matrix_sha256=hashlib.sha256(b'invented-rz-no-matrix').hexdigest()),
        dict(name='xx_plus_yy',qubits=[0,1],parameters=[str(theta),'0.47'],
             matrix_sha256=hashlib.sha256(b'invented-xx-no-matrix').hexdigest())]
    described=[b.module('rte').BasisChangeOperation(**r) for r in ops]
    bh=b.module('df_rte_tail').canonical_basis_hash(n,described)
    return dict(basis_id='invented-basis-'+bh,basis_hash=bh,num_system_qubits=n,operations=ops)


def reseal(records):
    p=records['POLICY.json'];d=p['declaration']
    d['definition_sha256']=b.digest(b.canonical(p['definitions']).encode())
    p['policy_fingerprint']=b.digest(b.canonical(d).encode())


def fixture(n=2, *, identity=False, trivial=False):
    original=basis(n,empty=trivial);selected=basis(n,theta=-.31,empty=trivial)
    alternate=basis(n,theta=.52,empty=trivial)
    common=dict(hamiltonian_hash='invented-H',partition_hash='invented-partition',
                preparation_hash='invented-preparation',tail_hash='invented-tail')
    source=dict(source_commit='invented-source',scope=dict(model='invented metadata only',
        system_qubits=n,m=0,q=1,r=1,K=2,delta=.5,coefficient_atol=0.))
    masses=(F(1,8),F(3,8),F(1,2),F(0));components=[];definitions=[]
    for label,mass,support,sign in zip(('a','b','c','zero'),masses,((0,),(n-1,),(0,n-1),(0,)),(1,-1,1,1)):
        is_identity=identity and label=='c'
        md=selected if label in ('a','b','zero') else alternate
        components.append(dict(component_id=label,is_identity=is_identity,
            basis_id=None if is_identity else original['basis_id'],
            basis_hash=None if is_identity else original['basis_hash'],
            diagonal_pauli_support=[] if is_identity else list(support),
            coefficient_sign=-1 if is_identity else sign,coefficient_abs=.2,coefficient_abs_hex=(.2).hex(),
            probability=.25,probability_hex=(.25).hex())) # Not actual Q; binder must ignore it.
        if not is_identity and not any(d['support']==list(support) for d in definitions):
            definitions.append(dict(source_basis_id=original['basis_id'],source_basis_hash=original['basis_hash'],
                                    support=list(support),definition=deepcopy(md)))
    blocks=[]
    for k,kind in enumerate(('one_body','df_fragment')):
        block=dict(block_id='block-'+str(k),block_kind=kind,original_fragment_index=k,rank=n,
            basis_id=original['basis_id'],basis_hash=original['basis_hash'],basis_change_operations=original['operations'],
            num_system_qubits=n,order_index=k)
        if k==0:block['diagonal_eigenvalues']=[0.]*n if trivial else [float(j+1) for j in range(n)]
        else:block.update(diagonal_eta=[0.]*n if trivial else [.5]*n,lam=.75)
        blocks.append(block)
    prep=dict(**common,coefficient_atol=0.,deterministic_blocks=blocks,
        basis_operations={original['basis_hash']:original['operations']},components=components,
        constant_coefficient_hex=(.125).hex(),identity_coefficient_hex=(-.25).hex())
    cfg=dict(config=dict(evolution_time=.5,rte_steps=1,finite_taylor_order=2,
        tail_hash=common['tail_hash'],dimensionless_step_time=.125),
        distribution=dict(orders=[0,2],dimensionless_step_time=.125))
    law=dict(uniform_grid_size=2**53,application_label_order='products_then_rotation',
        Q_definition='Q(order)*product(Q(label_position)); independent uniform-word model',
        orders=[0,2],order_masses=[dict(exact='3/4'),dict(exact='1/4')],
        label_masses=[dict(label=l,exact=str(p)) for l,p in zip(('a','b','c','zero'),masses)])
    policy=dict(definitions=definitions,declaration=dict(schema='a1_new_target_selected_policy_v1',
        source_commit=source['source_commit'],preparation_hash=common['preparation_hash'],tail_hash=common['tail_hash'],
        rule='support_run_le_1',maximum_support_run_length=1,boundary='boundary_optimized',controlled='ordinary_diag_I_U'))
    records=dict(zip(('SOURCE.json','CONFIG.json','LAW.json','POLICY.json','PREPARATION.json'),(source,cfg,law,policy,prep)))
    reseal(records)
    return records


class HeaderBindingTests(unittest.TestCase):
    def test_saved_schema_binds_signed_actual_mass_selected_inverse_and_stable_slot(self):
        records=fixture();bound=b.bind_n0_headers(records);a,neg,c,zero=bound.rows
        self.assertEqual((a.slot_id,neg.slot_id), (b.SLOT,b.SLOT))
        self.assertEqual(a.actual_mass,F(1,8));self.assertEqual(neg.actual_mass,F(3,8))
        self.assertNotEqual(a.actual_mass,F(records['PREPARATION.json']['components'][0]['probability']))
        self.assertEqual((neg.sign,neg.support),(-1,(1,)))
        selected=records['POLICY.json']['definitions'][0]['definition']
        self.assertEqual(a.selected_basis_id,selected['basis_id'])
        self.assertNotEqual(a.selected_basis_id,a.original_basis_id)
        self.assertEqual(a.V.operations,(('xx_plus_yy',((.31).hex(),(.47).hex()),(0,1)),
                                        ('rz',((-.13).hex(),),(0,))))
        self.assertEqual(a.phi.numeric,math.atan(.125/1))
        self.assertEqual(a.scalar_phase.numeric,-.5*1*(.125-.25));self.assertEqual(a.event_phase,m.ZERO)
        self.assertEqual(bound.outer_mass,F(3,4));self.assertEqual(zero.actual_mass,0)
        self.assertEqual(bound.records[-1]['status'],'ZERO_GRID_NOT_GROUPED')
        self.assertEqual(a.system_order,('qiskit-system-0','qiskit-system-1'))

    def test_context_gate_order_and_ordered_global_phases_without_quantum_build(self):
        records=fixture();row=b.bind_n0_headers(records).rows[0]
        self.assertEqual(row.B.operations[:4],(
            ('xx_plus_yy',((-.29).hex(),(.47).hex()),(0,1)),('rz',((-.13).hex(),),(0,)),
            ('rz',((-.25).hex(),),(0,)),('rz',((-.5).hex(),),(1,))))
        self.assertEqual(row.B.global_phase_terms[0],(-.375).hex())
        self.assertEqual(row.A.global_phase_terms,tuple(reversed(row.B.global_phase_terms)))
        self.assertNotEqual(row.A.operations,row.B.operations)
        self.assertEqual(row.B.adjoint().adjoint(),row.B)
        group=m.qualify_group((row,),b.bind_n0_headers(records).law)
        with patch.object(m,'QuantumCircuit',side_effect=AssertionError('forbidden quantum build')):
            with self.assertRaisesRegex(m.Ineligible,'METADATA_ONLY_NO_QUANTUM_BUILD'):
                m.build_instrument(group,'X',feed_forward=True)

    def test_existing_group_adapter_keeps_all_positive_mass_and_exact_q(self):
        bound=b.bind_n0_headers(fixture());positive=tuple(r for r in bound.rows if r.actual_mass>0)
        groups=m.partition_supplied_rows(positive,bound.law)
        self.assertEqual([tuple(r.label for r in g.rows) for g in groups],[('a','b'),('c',)])
        self.assertEqual(groups[0].conditional,(F(1,4),F(3,4)))
        self.assertEqual([g.mass for g in groups],[F(1,2),F(1,2)])
        self.assertEqual(sum(g.mass for g in groups),1)
        changed=replace(positive[1],A=m.LiteralCircuit(2,(),((.1).hex(),)))
        with self.assertRaisesRegex(m.Ineligible,'A_SEQUENCE_MISMATCH'):
            m.qualify_group((positive[0],changed),bound.law)
        for field in ('B','V'):
            altered=replace(positive[1],**{field:m.LiteralCircuit(2,(),((.1).hex(),))})
            with self.assertRaisesRegex(m.Ineligible,field+'_SEQUENCE_MISMATCH'):
                m.qualify_group((positive[0],altered),bound.law)
        result=b.summarize_binding(bound)
        self.assertEqual(result['status'],'NONTRIVIAL_GROUP_FOUND')
        self.assertEqual(result['groups'][0]['full_context_mass'],'3/8')
        self.assertEqual(len(result['groups'][0]['signed_diagonal_functions']),2)
        self.assertEqual(result['zero_grid_labels'],['zero'])
        EVIDENCE.append(result)

    def test_identity_and_explicit_adjoint_metadata_and_eight_wire_width(self):
        identity=b.bind_n0_headers(fixture(identity=True));row=identity.rows[2]
        self.assertEqual((row.support,row.sign,row.V.operations),((),-1,()))
        self.assertIn('identity phase',identity.records[2]['identity_source_rule'])
        trivial=b.summarize_binding(b.bind_n0_headers(fixture(trivial=True)))
        self.assertTrue(all(g['direct_measurement_structure']=='ONLY_KNOWN_DIRECT_MEASUREMENT_STRUCTURE'
                            for g in trivial['groups']))
        wide=b.bind_n0_headers(fixture(8));self.assertEqual(wide.rows[1].support,(7,))
        self.assertEqual(wide.rows[0].A.num_qubits,8)
        self.assertEqual(b.summarize_binding(wide)['status'],'NONTRIVIAL_GROUP_FOUND')

    def test_only_singletons_is_a_slot_result_not_a_general_no_go(self):
        records=fixture()
        for i,entry in enumerate(records['POLICY.json']['definitions']):
            entry['definition']['basis_id']='separate-selected-id-'+str(i)
        reseal(records)
        result=b.summarize_binding(b.bind_n0_headers(records))
        self.assertEqual(result['status'],'NO_NONTRIVIAL_GROUP_IN_SELECTED_SLOT')
        self.assertEqual(len(result['groups']),3)
        self.assertEqual(sum(F(g['conditional_mass']) for g in result['groups']),1)

    def test_inconsistent_unknown_policy_phase_basis_mass_and_order_stop(self):
        def check(mutator,code=None,seal=False):
            records=fixture();mutator(records)
            if seal:reseal(records)
            with self.assertRaises(b.BindingError) as got:b.bind_n0_headers(records)
            if code:self.assertEqual(got.exception.code,code)
            self.assertTrue(got.exception.detail)
        check(lambda r:r['CONFIG.json']['config'].update(evolution_time=.6),'INPUT_OR_POLICY_MISMATCH')
        check(lambda r:r['SOURCE.json']['scope'].update(m=1),'INPUT_OR_POLICY_MISMATCH')
        check(lambda r:r['POLICY.json']['declaration'].update(maximum_support_run_length=2),seal=True)
        check(lambda r:r['POLICY.json']['definitions'][0]['definition']['operations'][0].update(parameters=['0.14']),seal=True)
        check(lambda r:r['PREPARATION.json']['components'][1].update(coefficient_sign=0))
        check(lambda r:r['PREPARATION.json']['components'][1].update(diagonal_pauli_support=[2]))
        check(lambda r:r['PREPARATION.json']['deterministic_blocks'][0].update(order_index=1))
        check(lambda r:r['PREPARATION.json'].update(constant_coefficient_hex='nan'))
        check(lambda r:r['LAW.json']['label_masses'][0].update(exact='1/3'))
        check(lambda r:r['LAW.json'].update(application_label_order='rotation_then_products'))
        check(lambda r:r['POLICY.json']['definitions'][0]['definition']['operations'][0].update(name='opaque'),seal=True)
        with self.assertRaisesRegex(m.Ineligible,'ZERO_ACTUAL_SUPPORT'):
            bound=b.bind_n0_headers(fixture());m.qualify_group((bound.rows[-1],),bound.law)

    def test_mock_one_pass_writes_results_counts_stop_and_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'new-output'
            with (patch.object(e,'preflight_source',return_value={'execution_source_commit':'invented'}) as pre,
                  patch.object(e,'load_fixed_inputs',return_value=(fixture(),{'fixture':'invented'})) as load):
                result=e.run_metadata_pass(root,out,expected_source_commit='a'*40)
            self.assertEqual(result['status'],'NONTRIVIAL_GROUP_FOUND')
            self.assertEqual(pre.call_count,1);self.assertEqual(load.call_count,1)
            counts=json.loads((out/'CALL_COUNTS_FINAL.json').read_text())
            self.assertEqual((counts['metadata_pass'],counts['header_binding'],counts['group_partition'],counts['retries']), (1,1,1,0))
            self.assertTrue(all(counts[k]==0 for k in ('circuit_build','state_or_born','rng_or_shots','compile','variance_N_cost')))
            stop=json.loads((out/'STOP.json').read_text());self.assertFalse(stop['downstream_started'])
            with self.assertRaises(FileExistsError):e.run_metadata_pass(root,out,expected_source_commit='a'*40)

    def test_preflight_mismatch_and_binding_failure_preserve_partial_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/e.P1;inputs.mkdir(parents=True)
            (inputs/'SOURCE.json').write_text('{}')
            with self.assertRaisesRegex(b.BindingError,'input blob mismatch'):e.load_fixed_inputs(root)
            with self.assertRaisesRegex(b.BindingError,'full execution SOURCE'):e.preflight_source(root,'bad')
            with (patch.object(e,'preflight_source',side_effect=b.BindingError('INPUT_OR_POLICY_MISMATCH','mock mismatch')),
                  patch.object(e,'load_fixed_inputs') as load):
                result=e.run_metadata_pass(root,root/'preflight-stop',expected_source_commit='a'*40)
                self.assertEqual(result['status'],'INPUT_OR_POLICY_MISMATCH');load.assert_not_called()
            bad=fixture();bad['PREPARATION.json']['components'][1]['coefficient_sign']=0
            with (patch.object(e,'preflight_source',return_value={'fixture':True}),
                  patch.object(e,'load_fixed_inputs',return_value=(bad,{'fixture':True}))):
                result=e.run_metadata_pass(root,root/'binding-stop',expected_source_commit='a'*40)
            self.assertEqual(result['status'],'BINDING_INCOMPLETE')
            self.assertEqual(result['label'],'b');self.assertEqual(result['obtained_label_records'][0]['label'],'a')
            self.assertTrue((root/'binding-stop/STOP.json').exists())

    def test_execution_source_preflight_accepts_only_bound_code_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in e.SOURCE_FILES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'invented-source-bytes')
            def git_reply(args):
                return ('a'*40+'\n').encode() if args[-1]=='HEAD' else b'invented-source-bytes'
            with (patch.object(e.subprocess,'check_output',side_effect=git_reply),
                  patch.object(e.platform,'python_version',return_value='3.12.3'),
                  patch.object(e.importlib.metadata,'version',side_effect=lambda n:{'numpy':'1.26.4','qiskit':'1.3.0'}[n])):
                result=e.preflight_source(root,'a'*40)
                self.assertEqual(len(result['source_sha256']),len(e.SOURCE_FILES))
                (root/e.SOURCE_FILES[0]).write_bytes(b'changed')
                with self.assertRaisesRegex(b.BindingError,'SOURCE file mismatch'):
                    e.preflight_source(root,'a'*40)
            with patch.object(e.subprocess,'check_output',return_value=('b'*40+'\n').encode()):
                with self.assertRaisesRegex(b.BindingError,'HEAD mismatch'):
                    e.preflight_source(root,'a'*40)

    def test_new_sources_have_no_quantum_state_matrix_rng_or_compile_calls(self):
        for module in (b,e):
            tree=ast.parse(Path(module.__file__).read_text())
            names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
            forbidden={'QuantumCircuit','Statevector','Operator','SavedEngine','selected_circuit',
                       'restore_operations','describe_basis_change_operations','build_instrument',
                       'initialize','evolve','transpile','default_rng','sample','linalg','to_matrix'}
            self.assertFalse(forbidden&(names|attrs))


if __name__=='__main__':unittest.main()
