"""Invented DF blocks and tiny categorical laws only; no NPZ or molecular run."""
import ast
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from trottertracks.algorithm_codesign import a1_new_target_n1 as n1
from trottertracks.algorithm_codesign.a1_control_variates import EventMetadata
from trottertracks.algorithm_codesign.a1_connections import connect_sampler_inputs, proposed_control_banks
from trottertracks.algorithm_codesign.shared_cpu import module


class N1Correctness(TestCase):
    def test_tail_hash_input_retains_identity_and_exact_zero_components(self):
        tail=module('df_rte_tail'); model=module('df_trotter.model')
        extraction=tail.extract_df_diagonal_tail('invented',
            (model.DFBlock((),np.array([1.,-1.]),.5),),
            fragment_ids=('invented-fragment',),identity_policy='extract_identity_phase')
        payload=json.loads(extraction.canonical_hash_payload_json)
        self.assertEqual(hashlib.sha256(extraction.canonical_hash_payload_json.encode()).hexdigest(),extraction.tail_hash)
        self.assertEqual(len(payload['pre_threshold_components']),4)
        self.assertTrue(any(x['diagonal_pauli_support']==[] for x in payload['pre_threshold_components']))
        self.assertTrue(any(x['coefficient']==0. for x in payload['pre_threshold_components']))
        self.assertTrue(all(c.coefficient_abs>0 and c.diagonal_pauli_support for c in extraction.components))

    def test_empty_tail_preserves_original_canonical_hash_input(self):
        tail=module('df_rte_tail');e=tail.empty_df_tail_extraction('invented-empty',num_system_qubits=2)
        self.assertEqual(n1.digest(e.canonical_hash_payload_json.encode()),e.tail_hash)
        self.assertEqual(json.loads(e.canonical_hash_payload_json)['pre_threshold_components'],[])

    def test_core_change_has_no_numerical_or_existing_hash_ast_change(self):
        root=Path(__file__).resolve().parents[3];path='src/trotterlib/df_rte_tail.py'
        before=ast.parse(subprocess.check_output(['git','show',n1.CONTRACT+':'+path],cwd=root,text=True))
        after=ast.parse((root/path).read_text())
        for node in ast.walk(after):
            if isinstance(node,ast.ClassDef) and node.name=='DFTailExtraction':
                node.body=[x for x in node.body if not (isinstance(x,ast.AnnAssign) and
                    isinstance(x.target,ast.Name) and x.target.id=='canonical_hash_payload_json')]
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='DFTailExtraction':
                node.keywords=[x for x in node.keywords if x.arg!='canonical_hash_payload_json']
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))

    def make_connection(self, probabilities=(.25,.25,.5)):
        components=tuple(NS(component_id=l,probability=p) for l,p in zip(('a','b','c'),probabilities))
        return connect_sampler_inputs(components,NS(orders=(0,2),order_probabilities=(.75,.25)),
            source='invented',validate_components=lambda cs:sum(c.probability for c in cs))

    def test_moments_and_all_strata_match_independent_exhaustive_tiny_law(self):
        c=self.make_connection();result=n1.law_payload(c,a='a',b='b')
        cv,order,strata=proposed_control_banks(c,law_id='invented',a='a',b='b')
        means=[F(0)]*3;joint=[[F(0)]*3 for _ in range(3)];two={x:F(0) for x in result['order_patterns']};five={x:F(0) for x in result['five_patterns']}
        for n,qn in zip(c.law.orders.labels,c.law.orders.masses):
            for labels in product(c.law.labels.labels,repeat=n+1):
                mass=qn
                for label in labels:mass*=c.law.labels.mass(label)
                event=(EventMetadata(n,labels,complex((-1)**(n//2))),)
                g=cv.values(event)
                for i in range(3):
                    means[i]+=mass*g[i]
                    for j in range(3):joint[i][j]+=mass*g[i]*g[j]
                two[order.values(event)]+=mass;five[strata.values(event)]+=mass
        self.assertEqual([F(x['exact']) for x in result['control_means']],means)
        self.assertEqual([[F(x['exact']) for x in row] for row in result['joint_moments']],joint)
        self.assertEqual([F(x['exact']) for x in result['five_strata']],list(five.values()))
        self.assertEqual([F(x['exact']) for x in result['order_strata']],list(two.values()))

    def test_zero_control_grid_support_stops_without_label_substitution(self):
        with self.assertRaises(ValueError):n1.law_payload(self.make_connection((0.,.5,.5)),a='a',b='b')

    def test_same_label_with_wrong_semantic_support_is_rejected(self):
        a=NS(component_id=n1.A,df_fragment_id='df-fragment-3',diagonal_pauli_support=(0,),is_identity=False,coefficient_abs=.25,coefficient_sign=1,basis_hash='mock')
        b=NS(component_id=n1.B,df_fragment_id='df-fragment-6',diagonal_pauli_support=(0,),is_identity=False,coefficient_abs=.5,coefficient_sign=-1,basis_hash='mock')
        payload=dict(pre_threshold_components=[dict(component_id=n1.A,coefficient=.25,basis_hash='mock'),dict(component_id=n1.B,coefficient=-.5,basis_hash='mock')])
        with self.assertRaises(ValueError):n1.check_controls((a,b),payload)

    def test_hash_serializer_rejects_missing_canonical_tail_payload(self):
        @dataclass
        class Fragment:
            weight: float
        part=NS(ld=3,deterministic_fragments=(Fragment(.5),),randomized_fragments=(),ranking_proxy_lambda_r=0.,weight_rule='mock')
        prep=NS(hamiltonian_hash='mockH',partition_hash='mockP',preparation_hash='mockPrep',
            deterministic_blocks=(),tail_extraction=NS(tail_hash='mockTail',canonical_hash_payload_json=''),
            identity_policy='extract_identity_phase',coefficient_atol=0.,basis_construction_policy='fermionic_gaussian_jw',diagonal_sort='descending_abs')
        with self.assertRaisesRegex(ValueError,'canonical tail'):n1.hash_payloads(part,prep,lambda b:b)

    def test_policy_incomplete_and_nonfinite_proofs_are_rejected(self):
        c=NS(basis_id='i',basis_hash='h',diagonal_pauli_support=(0,))
        prep=NS(rte_preparation=NS(symbolic_tail=NS(components=(c,))))
        with self.assertRaisesRegex(ValueError,'coverage'):n1.policy_payload(prep,{},[],source_commit='invented')
        proof=dict(source_basis_id='i',source_basis_hash='h',support=(0,),preserved_columns_max_abs_residual=float('nan'))
        with self.assertRaisesRegex(ValueError,'correspondence'):n1.policy_payload(prep,{('i','h',(0,)):NS()},[proof],source_commit='invented')

    def test_scope_guard_blocks_rng_state_full_build_and_compile_and_restores(self):
        dep=n1.dependencies();original=np.random.default_rng
        with n1.n1_only(dep):
            for action in (lambda:np.random.default_rng(1),lambda:dep.hamiltonian.solve_df_ground_state(None,None),
                           lambda:dep.partial.QiskitDFPartialS2CircuitBuilder.build_step(None,None),
                           lambda:dep.compiler.transpile_and_measure_cost(None,None)):
                with self.assertRaisesRegex(RuntimeError,'N1 scope forbids'):action()
        self.assertIs(np.random.default_rng,original)

    def test_preinput_failure_saves_stop_and_fresh_directory_prevents_retry(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);lock=root/'lock.json';lock.write_text('{}');out=root/'run'
            with patch.object(n1,'dependencies',return_value=NS()),patch.object(n1,'profile',return_value={'different':True}),patch.object(n1.subprocess,'check_output',side_effect=['','mock-source']):
                with self.assertRaisesRegex(ValueError,'profile differs'):n1.run(out,lock,root=root)
            stop=(out/'STOP.json').read_bytes();counts=json.loads((out/'CALL_COUNTS.json').read_text())
            self.assertTrue(all(x==0 for x in counts.values()))
            with patch.object(n1.subprocess,'check_output',return_value=''):
                with self.assertRaises(FileExistsError):n1.run(out,lock,root=root)
            self.assertEqual((out/'STOP.json').read_bytes(),stop)

    def test_invented_preparation_and_policy_run_inside_guard_without_interrogation(self):
        dep=n1.dependencies()
        h=dep.hamiltonian.DFHamiltonian(constant=.1,one_body=np.diag([.2,-.1]),
            lambdas=np.array([.7,.2,.1]),g_matrices=(np.array([[1.,.1],[.1,-.5]]),
              np.array([[.3,.2],[.2,-.1]]),np.diag([.4,.2])),metadata={'scope':'invented'})
        with n1.n1_only(dep):
            part=dep.split.split_df_hamiltonian_by_ld(h,1)
            prep=dep.partial.prepare_df_partial_s2(h,part,identity_policy='extract_identity_phase',coefficient_atol=0.)
            p,q,t=n1.hash_payloads(part,prep,dep.partial._block_hash_payload)
            self.assertEqual(n1.digest(n1.canonical(p).encode()),prep.partition_hash)
            defs,proofs=dep.policy.register_support_restricted_bases(h,prep)
            result=n1.policy_payload(prep,defs,proofs,source_commit='invented')
            self.assertFalse(result['full_interrogation_tested'])
            self.assertLessEqual(result['maximum_preserved_columns_residual'],n1.POLICY_ATOL)
