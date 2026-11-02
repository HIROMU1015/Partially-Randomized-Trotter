"""Phase A: toy/mock checks only; never read a real H6 model or signal."""
from dataclasses import asdict
from fractions import Fraction as Q
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = types.ModuleType('hybrid_extension_test')
p.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules[p.__name__] = p
E = importlib.import_module(p.__name__+'.h6_hybrid_fairness_extension')
P = importlib.import_module(p.__name__+'.pauli_signal_adapter')
F = importlib.import_module(p.__name__+'.fixed_signal_evaluation')


def toy(n=2):
    labels = [('IX', .375), ('IY', -.25), ('ZI', .125), ('YY', -.0625), ('XZ', .03125), ('II', .1875)]
    return P.PauliHamiltonian.from_terms([('I'*(n-2)+s, c) for s, c in labels],
                                        num_qubits=n, model_id='ARTIFICIAL_HYBRID_EXTENSION')


class Checks(unittest.TestCase):
    def test_fixed_four_dispatch_and_old_validators_unchanged(self):
        for cid, values in E.CANDIDATES.items():
            cfg = E.config(cid)
            self.assertEqual((cfg.L_D, cfg.q, cfg.r, cfg.K), values)
        for cid in ('PR_p0078_q8_r8_K2', 'PR_p0000_q8_r64_K2', 'PR_p0684_q1_r32_K4'):
            with self.assertRaises(ValueError): E.config(cid)
        with self.assertRaises(ValueError): E.S.config(78, 8, 16, 2)
        pure = importlib.import_module(p.__name__+'.pauli_rte_event_extension')
        with self.assertRaises(ValueError): pure.candidate_id(78, 8, 16, 2)

    def test_fixed_identity_mock_routes_without_real_law_evaluation(self):
        ham = types.SimpleNamespace(num_qubits=12, terms=(None,)*918, hamiltonian_hash=E.HAMILTONIAN_HASH)
        for cid in E.CANDIDATES:
            with patch.object(E, '_law', return_value='MOCK_ONLY') as call:
                self.assertEqual(E.law_descriptor(ham, cid), 'MOCK_ONLY')
                self.assertEqual(call.call_args.args[1], E.config(cid))
        with patch.object(E, '_law', side_effect=AssertionError('must not evaluate')):
            with self.assertRaises(ValueError): E.law_descriptor(toy(), next(iter(E.CANDIDATES)))
        with self.assertRaises(ValueError): E.bind_saved_inputs(b'wrong', b'wrong')

    def test_original_r8_law_and_split_retained(self):
        h = toy()
        got = E.artificial_law(h, 1, 8, 8, 2)
        original = E.S.law_descriptor(h, E.S.config(1, 8, 8, 2))
        self.assertEqual(got, original)
        for q, r, K in ((8, 16, 2), (8, 32, 2), (8, 64, 2), (1, 16, 4)):
            d = E.artificial_law(h, 2, q, r, K)
            self.assertEqual([x['label'] for x in d['retained']], [s for s,c in h.terms[:2]])
            self.assertEqual([x['label'] for x in d['components']], [s for s,c in h.terms[2:]])
            self.assertEqual(sum(map(Q, d['component_mapping']['exact_masses'])), 1)
            self.assertEqual(sum(map(Q, d['order_mapping']['exact_masses'])), 1)
            self.assertEqual(d['orders'], list(range(0, K+1, 2)))

    def test_real_mean_connection_is_mocked_for_each_fixed_candidate(self):
        numerical = importlib.import_module(p.__name__+'.pauli_vector_tight_enclosure')
        model=dict(n_qubits=12,Hamiltonian_hash=E.HAMILTONIAN_HASH)
        for cid in E.CANDIDATES:
            law=dict(config=asdict(E.config(cid)),Hamiltonian_hash=E.HAMILTONIAN_HASH)
            with patch.object(numerical,'evaluate',return_value='NO_STATE_ACTION') as call:
                self.assertEqual(E.mean_target(model,{},law,cid),'NO_STATE_ACTION')
                self.assertEqual(call.call_args.args,(model,{},law['config'],law))
            bad=dict(law,config=dict(law['config'],r=128))
            with patch.object(numerical,'evaluate',side_effect=AssertionError('must not evaluate')):
                with self.assertRaises(ValueError):E.mean_target(model,{},bad,cid)

    def test_cohort_graph_mock_does_not_generate_real_candidate_seeds(self):
        for cid, (prefix,q,r,K) in E.CANDIDATES.items():
            for cohort in ('exploration','confirmation'):
                with patch.object(E, '_seed', return_value=123) as call:
                    self.assertEqual(E.trajectory_seed(cid,cohort,7),123)
                    self.assertEqual(call.call_args.args,(dict(molecule=6,config=dict(L_D=prefix,q=q,r=r,K=K)),cohort,7))
        for cohort, index in [('other',0),('confirmation',True),('exploration',-1)]:
            with self.assertRaises(ValueError): E.trajectory_seed(next(iter(E.CANDIDATES)),cohort,index)

    def test_extended_toy_body_same_as_original_flat_sampler(self):
        h = toy()
        for prefix,q,r,K in ((1,8,16,2),(1,8,32,2),(1,8,64,2),(2,1,16,4)):
            body,wrapped,meta = E.artificial_trajectory(h,prefix,q,r,K,seed=391)
            original=P.PauliSignalAdapter(h,'PR',prefix,.8/q,r,K).trajectory(q=q,seed=391)
            self.assertEqual(body,original)
            self.assertEqual(meta['event_count'],q*r)
            self.assertEqual(meta['step_seeds'],[391]*q)
            self.assertEqual(set(wrapped),{'cosine','sine'})

    def test_compiled_toy_relative_phase_both_axes_and_nonunit_raw_norm(self):
        from qiskit.quantum_info import Operator, Statevector
        from hybrid_extension_test.rte_compiled_cost import transpile_and_measure_cost
        h=toy(3);raw=np.array([complex((i+1)/32,(2-i)/64) for i in range(8)])
        nu=np.vdot(raw,raw).real
        for prefix,q,r,K in ((1,8,64,2),(2,1,16,4)):
            body,wrapped,meta=E.artificial_trajectory(h,prefix,q,r,K,seed=512)
            ideal=np.vdot(raw,Operator(body).data@raw)/nu
            for axis in F.AXES:
                cost=transpile_and_measure_cost(wrapped[axis],F.default_compiler(),actual_circuit_fingerprint='ARTIFICIAL_'+str((q,r,K,axis)))
                circuit=cost.transpiled_circuit.remove_final_measurements(inplace=False)
                v=Statevector(np.concatenate([raw,np.zeros(8)])).evolve(circuit).data
                measured=(np.vdot(v[:8],v[:8])-np.vdot(v[8:],v[8:])).real/nu
                self.assertAlmostEqual(measured,ideal.real if axis=='cosine' else ideal.imag,places=10)

    def test_existing_shot_account_and_paired_covariance(self):
        from hybrid_extension_test import h6_rte_fairness as old
        target=dict(axes={a:dict(signed_bias_exact='0',u_enc_exact='1/1000000') for a in F.AXES})
        prices={'cosine':'7/3','sine':'11/3'};law={'B_saved_binary64':1.25}
        got=E.account_target(target,prices,'.05',law)
        self.assertEqual(got,old.account(target,prices,'.05',scale=old.scale(Q(1.25))))
        self.assertIsNotNone(got['N_total'])
        stats=E.A.extract('src/trotterlib/fixed_signal_evaluation.py','_statistics',dict(np=np,math=__import__('math'),Fraction=Q))
        samples=[dict(cosine=2,sine=7),dict(cosine=5,sine=4),dict(cosine=8,sine=1)]
        summary=E.A.cost_summary(samples,stats)
        self.assertLess(Q(summary['covariance_exact']['cosine']['sine']),0)
        paired=E.A.paired(samples,dict(cosine=1,sine=1),stats)
        self.assertEqual(paired['standard_error'],0.)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--write-artificial-fixtures':
        h=toy();raw=[[.5,0.],[.125,.0625],[0.,-.125],[.25,0.]]
        reference=dict(reference={k:dict(lower='0',upper='0') for k in ('real','imag')},method='ARTIFICIAL_ZERO_REFERENCE')
        fixtures=[]
        for prefix,q,r,K in ((1,8,16,2),(1,8,32,2),(1,8,64,2),(2,1,16,4)):
            law=E.artificial_law(h,prefix,q,r,K)
            fixtures.append(dict(artificial_qubits=2,law=law,model=dict(state=raw,n_qubits=2,hamiltonian=dict(constant=h.constant,terms=h.terms)),reference=reference))
        Path(sys.argv[2]).write_text(json.dumps(fixtures,indent=2)+'\n')
    else:
        unittest.main()
