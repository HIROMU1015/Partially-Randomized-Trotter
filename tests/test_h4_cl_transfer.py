"""Formal/artificial and saved-byte rejection tests; no real H4 numeric work."""
from copy import deepcopy
from fractions import Fraction as Q
import builtins,cmath,importlib.util,json,math,csv,io
from pathlib import Path
import subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('CL_test_binding',ROOT/'src/trotterlib/h4_cl_binding.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
R,T=m.R,m.T
MANIFEST=ROOT/'artifacts/non_df_h4_cl_transfer/2026-10-23/evidence_manifest.json'


def term(name,source='A',target='B',primitives=None,domain='FORMAL',context='axisA'):
    return dict(id=name,domain=domain,context=context,edge=dict(**{'from':source,'to':target},quantity='OPERATOR_NORM',state='operator',scale='unscaled'),primitives=primitives or ['same_cause'],bound_ref='bound:'+name,guarantee='FORMAL_HYPOTHESIS' if domain=='FORMAL' else 'SAVED_UNCHECKED')


class CoverageRules(unittest.TestCase):
    def test_same_name_is_not_same_edge(self):
        r=R.coverage_relation(term('x'),term('y','C','D'))
        self.assertEqual(r['status'],'MISSING_PROOF');self.assertEqual(r['label_collisions'],['same_cause'])
    def test_same_edge_overlap_unresolved(self):
        self.assertEqual(R.coverage_relation(term('x'),term('y'))['status'],'OVERLAP_UNRESOLVED')
    def test_different_names_do_not_prove_disjoint(self):
        self.assertEqual(R.coverage_relation(term('x',primitives=['one']),term('y',primitives=['two']))['status'],'MISSING_PROOF')
    def test_exact_same_bound_can_be_deduplicated(self):
        a,b=term('x'),term('y');b['bound_ref']=a['bound_ref']
        self.assertEqual(R.coverage_relation(a,b,rule='IDENTICAL_BOUND')['status'],'PROVED_COVERED')
    def test_cover_requires_same_proposition(self):
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),term('y'),rule='IDENTICAL_BOUND')
    def test_consecutive_triangle_links_even_shared_cause(self):
        self.assertEqual(R.coverage_relation(term('x'),term('y','B','C'),rule='TELESCOPING_CHAIN')['status'],'PROVED_DISJOINT')
    def test_real_unchecked_chain_not_promoted(self):
        self.assertEqual(R.coverage_relation(term('x',domain='SAVED_H4'),term('y','B','C',domain='SAVED_H4'),rule='TELESCOPING_CHAIN')['status'],'MISSING_PROOF')
    def test_noncontiguous_chain_rejected(self):
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),term('y','C','D'),rule='TELESCOPING_CHAIN')
    def test_mixed_norm_and_signal_rejected(self):
        b=term('y','B','C');b['edge']['quantity']='SIGNAL_ERROR'
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),b,rule='TELESCOPING_CHAIN')
    def test_scaled_and_unscaled_rejected(self):
        b=term('y','B','C');b['edge']['scale']='B_saved'
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),b,rule='TELESCOPING_CHAIN')
    def test_state_ray_scope_rejected(self):
        b=term('y','B','C');b['edge']['state']='raw_state'
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),b,rule='TELESCOPING_CHAIN')
    def test_cross_axis_rejected(self):
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),term('y',context='axisB'))
    def test_cross_domain_rejected(self):
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),term('y',domain='SAVED_H4'))
    def test_unproved_subtraction_cancellation_addition_rejected(self):
        for op in ('SUBTRACT','CANCEL','ADD_C_TOTAL_AND_L','UNRECOGNIZED'):
            with self.subTest(op=op),self.assertRaises(ValueError):R.coverage_relation(term('x'),term('y'),operation=op)
    def test_saved_proof_flag_cannot_certify(self):
        a=term('x',domain='SAVED_H4');a['guarantee']='FORMAL_HYPOTHESIS'
        with self.assertRaises(ValueError):R.validate_term(a)
    def test_duplicate_primitive_and_term_rejected(self):
        a=term('x');a['primitives']=['one','one']
        with self.assertRaises(ValueError):R.validate_term(a)
        with self.assertRaises(ValueError):R.coverage_relation(term('x'),term('x'))
    def test_not_applicable_only_pauli(self):
        self.assertEqual(R.not_applicable('pauli_jw')['status'],'NOT_APPLICABLE')
        with self.assertRaises(ValueError):R.not_applicable('df_native')


class SymbolicTransfer(unittest.TestCase):
    def test_one_body_exact_polynomial(self):
        for n in (1,2,3):self.assertEqual(R.diagonal_identity('ONE_BODY',n)['status'],'FORMAL_LEMMA_CHECKED')
    def test_DF_square_kept_as_square(self):
        for n in (1,2,3):self.assertFalse(R.diagonal_identity('DF_SQUARED',n)['DF_squared_replaced_by_involution'])
    def test_diagonal_phase_sign_mutation_rejected(self):
        for kind in ('ONE_BODY','DF_SQUARED'):
            with self.assertRaises(ValueError):R.diagonal_identity(kind,phase_sign=-1)
    def test_bit_idempotence_only(self):
        b=R.Polynomial.variable('bit:a');x=R.Polynomial.variable('x')
        self.assertEqual((b*b).terms,b.terms);self.assertNotEqual((x*x).terms,x.terms)
    def test_unsupported_diagonal_rejected(self):
        with self.assertRaises(ValueError):R.diagonal_identity('FRAGMENT_IS_PAULI')
    def test_exact_control_diagonal_rules(self):
        for kind in ('CRZ','RZZ','CONTROLLED_RZZ','PHASE'):self.assertEqual(R.controlled_diagonal_identity(kind)['implementation_choice_in_saved_H4'],'NOT_PROVED')
    def test_control_keeps_system_global_phase(self):
        u=T.identity_word(1);u['phase_quarters']=2
        self.assertEqual(R.controlled_word(u,T.identity_word(1),u)['status'],'FORMAL_LEMMA_CHECKED')
        with self.assertRaises(ValueError):R.controlled_word(u,T.identity_word(1),T.identity_word(1))
    def test_zero_branch_not_arbitrary(self):
        u=T.identity_word(1);u['phase_quarters']=1
        with self.assertRaises(ValueError):R.controlled_word(u,u,u)
    def test_inverse_order_and_noncommutation(self):
        self.assertEqual(R.ordered_basis(2,['U1','U2'],['U2','U1'])['status'],'FORMAL_LEMMA_CHECKED')
        with self.assertRaises(ValueError):R.ordered_basis(2,['U1','U2'],['U1','U2'])
    def test_event_even_phase_and_identity_sign(self):
        self.assertEqual([R.event_phase(k,[1]*k,1)['phase_quarters'] for k in (0,2,4)],[0,2,0])
        self.assertEqual(R.event_phase(2,[-1,1],-1,identity=True)['identity_rotation_phase'],'1*alpha')
    def test_unsupported_order_rejected(self):
        with self.assertRaises(ValueError):R.event_phase(3,[1]*3,1)
    def test_hadamard_cosine_and_sine(self):
        for phi in (.0,.3,-.7,math.pi/2):
            z=cmath.exp(1j*phi)
            for axis,shift,expected in [('cosine',1,z.real),('sine',-1j,z.imag)]:
                p0=abs((1+shift*z)/2)**2;p1=abs((1-shift*z)/2)**2
                self.assertAlmostEqual(p0-p1,expected,places=14)
                self.assertEqual(R.wrapper(axis)['binding_to_saved_H4'],'NOT_PROVED')
    def test_sine_phase_order_rejected(self):
        with self.assertRaises(ValueError):T.check_symbolic_rule(dict(rule='HADAMARD_WRAPPER',data=dict(axis='sine',operations=['H','SDG','CONTROLLED_U','H'])))
    def test_Bsaved_not_nominal_and_no_new_values(self):
        r=R.measurement_transfer();self.assertEqual(r['measurement_bound_symbolic'],'B_saved*(eta0+eta1+eta0*eta1)')
        self.assertFalse(r['numerical_B_scaled_bound_evaluated']);self.assertTrue(r['B_saved_not_nominal'])
    def test_both_branch_bound_has_cross_term(self):
        a,b=Q(1,10),Q(1,5);difference=(1+a)*(1+b)-1
        self.assertEqual(difference,a+b+a*b);self.assertGreater(difference,a+b)
        self.assertEqual(R.measurement_transfer(zero_exact=True)['unscaled_bound'],'eta1')
    def test_normalized_ray_factor_must_cancel(self):
        n=Q(1,2);delta=Q(1)
        self.assertEqual(n*delta/n,delta);self.assertLess(n*delta,delta)
        self.assertTrue(R.measurement_transfer()['same_ray_normalization_cancels_state_norm'])
    def test_binary_support_does_not_prove_norm(self):
        r=R.measurement_transfer();self.assertTrue(r['operator_norm_one_not_inferred_from_binary_support'])


class SavedConnections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=T.read_json_bytes(MANIFEST.read_bytes());cls.store=m.TransferStore(ROOT,cls.manifest);cls.output=m.produce(cls.store)
    def test_40_by14_graph(self):
        self.assertEqual(len(self.output['rows']),40)
        for row in self.output['rows']:
            self.assertEqual(len(row['gates']),14);self.assertEqual(len(row['evidence_graph']),14)
    def test_category_vocabulary(self):
        for row in self.output['rows']:
            for gate in row['gates'].values():self.assertIn(gate['category'],m.CATEGORIES)
    def test_L_support_join_is_metadata_only(self):
        self.assertFalse(self.output['local_L_join']['numeric_bounds_compared']);self.assertFalse(self.output['local_L_join']['exact_unitarity_of_rounded_matrices_inferred'])
    def test_all_conditions_remain_blocked(self):
        for row in self.output['rows']:
            self.assertFalse(row['all_scientific_gates_verified']);self.assertIsNone(row['N']);self.assertIsNone(row['G'])
    def test_L_only_used_by_DF_correct_method(self):
        for row in self.output['rows']:
            facts=row['evidence_graph']['local_L']
            if row['target']['case'].startswith('pauli_jw'):self.assertEqual(facts,[])
            else:self.assertEqual(facts[0]['pointer'][-1],row['target']['case'].rsplit('_',1)[1])
    def test_C_and_L_not_combined_or_subtracted(self):
        self.assertEqual(self.output['C_L_combination'],'COMBINATION_BLOCKED')
        self.assertEqual(self.output['new_40_axis_total_bounds'],0);self.assertEqual(self.output['new_B_scaled_bounds'],0)
    def test_old_gate_judgments_unchanged(self):
        for row in self.output['rows']:
            for gate in row['gates'].values():self.assertFalse(gate['original_gate_judgment_modified'])
    def test_no_new_PCG64_requirement(self):self.assertFalse(self.output['unconditional_PCG64_independence_required'])
    def test_all_execution_modes_rejected(self):
        for action in ('EXECUTE','ACCEPT','N_G','COMPILE','RETRY'):
            with self.assertRaises(ValueError):m.no_execution(action)
    def test_bad_manifest_authority_rejected(self):
        bad=deepcopy(self.manifest);bad['execution_authorized']=True
        with self.assertRaises(ValueError):m.TransferStore(ROOT,bad)
    def test_wrong_source_hash_rejected(self):
        bad=deepcopy(self.manifest);bad['files']['L_summary']['sha256']='0'*64
        with self.assertRaises(ValueError):m.TransferStore(ROOT,bad)
    def test_artificial_evidence_cannot_bind_real(self):
        bad=deepcopy(self.manifest);bad['files']['L_summary']['domain']='ARTIFICIAL'
        with self.assertRaises(ValueError):m.TransferStore(ROOT,bad)
    def test_output_tamper_and_cross_axis_rejected(self):
        bad=deepcopy(self.output);bad['rows'][0]['target_digest']=bad['rows'][1]['target_digest']
        with self.assertRaises(ValueError):m.verify(self.store,bad)
    def test_numeric_backends_not_imported_by_adapter(self):
        original=builtins.__import__
        def guarded(name,*a,**kw):
            if name.split('.')[0] in ('numpy','flint','qiskit','scipy'):raise AssertionError('Scientific import forbidden')
            return original(name,*a,**kw)
        with patch('builtins.__import__',side_effect=guarded):
            self.assertEqual(m.verify_L_joins(self.store)['status'],'STATIC_L_IDENTITIES_BOUND')
    def test_CLI_refuses_science_before_missing_manifest(self):
        p=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_h4_cl_transfer.py'),'--mode','EXECUTE'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertIn('NO_SCIENCE_ACCEPTANCE',p.stderr)
    def test_no_saved_numeric_endpoint_comparison(self):
        for row in self.output['rows']:
            self.assertFalse(row['evidence_graph']['combined_budget']['new_total_created'])
        self.assertEqual(self.output['new_H4_numeric_evaluations'],0)


class AdditionalScopeRejections(unittest.TestCase):
    def store(self):return m.TransferStore(ROOT,T.read_json_bytes(MANIFEST.read_bytes()))
    def replace_csv(self,store,role,edit):
        rows=list(csv.DictReader(io.StringIO(store.raw[role].decode())))
        edit(rows)
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        store.raw[role]=text.getvalue().encode()
    def test_forward_inverse_result_scope_swap_rejected(self):
        store=self.store()
        def edit(rows):
            a=next(row for row in rows if row['direction']=='forward')
            b=next(row for row in rows if row['direction']=='inverse' and row['gate_hash']==a['gate_hash'])
            a['result_path'],a['result_SHA256']=b['result_path'],b['result_SHA256']
        self.replace_csv(store,'L_local_csv',edit)
        with self.assertRaises(ValueError):m.verify_L_joins(store)
    def test_basis_inverse_order_mutation_rejected(self):
        store=self.store()
        def edit(rows):
            row=next(r for r in rows if len(json.loads(r['inverse_gate_order_json']))>1)
            row['inverse_gate_order_json']='[]'
        self.replace_csv(store,'L_basis_csv',edit)
        with self.assertRaises(ValueError):m.verify_L_joins(store)
    def test_component_raw_result_swap_rejected(self):
        store=self.store()
        def edit(rows):rows[0]['result_path'],rows[0]['result_SHA256']=rows[1]['result_path'],rows[1]['result_SHA256']
        self.replace_csv(store,'L_support_csv',edit)
        with self.assertRaises(ValueError):m.verify_L_joins(store)
    def test_literal_compiled_scope_relabel_rejected(self):
        store=self.store();output=m.produce(store)
        output['rows'][0]['gates']['wrapper_transfer']['scope']='COMPILED_PROGRAM'
        with self.assertRaises(ValueError):m.verify(store,output)
