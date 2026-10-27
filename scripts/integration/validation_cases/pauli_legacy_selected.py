"""Selected legacy artificial/pure/mock bodies under normal package imports.

Execute against the fixed P2 src and repository root; not a scientific regression.
Original alternate loaders are intentionally not used. No physical model read.
"""
import importlib
import types
import unittest
from dataclasses import asdict
from fractions import Fraction as Q
from unittest.mock import patch
import numpy as np
from trotterlib import pauli_hybrid_rte_comparison as MODULE
from scripts import inventory_pauli_hybrid_rte as INVENTORY
from trotterlib import pauli_rte_event_extension as EVENT
from trotterlib import h6_hybrid_fairness_extension as HYBRID
from trotterlib import pauli_signal_adapter as P

def row(cid, group, G, cohort='exploration', status='CONDITIONAL_GUARD_ELIGIBLE'):
    return dict(molecule='synthetic', epsilon_sig='synthetic', candidate_id=cid, group=group,
                G_exact=G, cohort=cohort, eligibility=status)

def minimum(rows, group='Hybrid', cohort='exploration'):
    return MODULE.saved_class_minimum(rows, molecule='synthetic', epsilon='synthetic',
                                      group=group, cohort=cohort)

class PreparationChecks(unittest.TestCase):
    def test_both_saved_CSV_null_spellings_stay_missing(self):
        self.assertIsNone(INVENTORY.saved_csv_value('null'))
        self.assertIsNone(INVENTORY.saved_csv_value(''))
        self.assertEqual(INVENTORY.saved_csv_value('0'), '0')
        x = dict(G_exact=None, eligibility='BIAS_INELIGIBLE')
        self.assertEqual(INVENTORY.availability(x), 'BIAS_INELIGIBLE')

    def test_cohorts_do_not_mix(self):
        rows = [row('explore', 'Hybrid', '2'), row('confirm', 'Hybrid', '1', 'confirmation')]
        self.assertEqual(minimum(rows)['selected']['candidate_id'], 'explore')
        self.assertEqual(minimum(rows, cohort='confirmation')['selected']['candidate_id'], 'confirm')

    def test_null_never_becomes_zero_or_infinity(self):
        rows = [row('missing', 'Hybrid', None), row('finite', 'Hybrid', '7/3')]
        result = minimum(rows)
        self.assertEqual(result['G_exact'], '7/3')
        self.assertEqual(result['unresolved_ids'], ['missing'])
        self.assertFalse(result['all_registered_parameters_resolved'])

    def test_exact_ties_and_ineligible_prices(self):
        rows = [row('b', 'Hybrid', '4/2'), row('a', 'Hybrid', '2'),
                row('bad', 'Hybrid', '1/100', status='BIAS_INELIGIBLE')]
        self.assertEqual(minimum(rows)['ties'], ['a', 'b'])
        self.assertEqual(minimum(rows)['selected']['candidate_id'], 'a')

    def test_ratio_orientation_and_missing_confirmation(self):
        rows = [row('h', 'Hybrid', '3'), row('r', 'RTE', '2')]
        self.assertEqual(MODULE.hybrid_over_rte(minimum(rows), minimum(rows, 'RTE')), '3/2')
        missing = minimum(rows, 'RTE', 'confirmation')
        self.assertEqual(missing['status'], 'CONFIRMATION_MISSING')
        self.assertIsNone(MODULE.hybrid_over_rte(minimum(rows), missing))

    def test_ratio_rejects_mixed_cohorts(self):
        rows = [row('h', 'Hybrid', '3'), row('r', 'RTE', '2', 'confirmation')]
        with self.assertRaises(ValueError):
            MODULE.hybrid_over_rte(minimum(rows), minimum(rows, 'RTE', 'confirmation'))

    def test_ineligible_confirmed_subset_does_not_disqualify_registered_family(self):
        rows = [row('explore', 'Hybrid', '2'),
                row('old_confirmed', 'Hybrid', None, 'confirmation', 'BIAS_INELIGIBLE')]
        self.assertEqual(minimum(rows, cohort='confirmation')['status'], 'CONFIRMATION_MISSING')

E = EVENT


class EventMetadataChecks(unittest.TestCase):
    def test_domains_do_not_widen_old_search_or_H4(self):
        with self.assertRaises(ValueError):EVENT.S.config(0,8,16,2)
        for args in [(1,8,16,2),(0,4,16,2),(0,8,256,2),(0,8,16,6),(0,8,True,2)]:
            with self.assertRaises(ValueError):EVENT.candidate_id(*args)
        with self.assertRaises(ValueError):EVENT.validate('H4',64,2)
        with self.assertRaises(ValueError):EVENT.validate('H2',16,2)

    def test_seed_graph_unchanged_and_cohorts_separate(self):
        for molecule in ('H4','H6'):
            for cohort in ('exploration','confirmation'):
                got=EVENT.trajectory_seed(molecule,8,2,cohort,0)
                entropy=[20261025,3101 if molecule=='H4' else 6,0,8,8,2,101 if cohort=='exploration' else 202,0]
                self.assertEqual(got,int(np.random.SeedSequence(entropy).generate_state(1,dtype=np.uint64)[0]))
            self.assertNotEqual(EVENT.trajectory_seed(molecule,16,2,'exploration',0),EVENT.trajectory_seed(molecule,16,2,'confirmation',0))


def toy(n=2):
    labels = [('IX', .375), ('IY', -.25), ('ZI', .125), ('YY', -.0625), ('XZ', .03125), ('II', .1875)]
    return P.PauliHamiltonian.from_terms([('I'*(n-2)+s, c) for s, c in labels],
                                        num_qubits=n, model_id='ARTIFICIAL_HYBRID_EXTENSION')
E = HYBRID


class H6MockChecks(unittest.TestCase):
    def test_fixed_four_dispatch_and_old_validators_unchanged(self):
        for cid, values in E.CANDIDATES.items():
            cfg = E.config(cid)
            self.assertEqual((cfg.L_D, cfg.q, cfg.r, cfg.K), values)
        for cid in ('PR_p0078_q8_r8_K2', 'PR_p0000_q8_r64_K2', 'PR_p0684_q1_r32_K4'):
            with self.assertRaises(ValueError): E.config(cid)
        with self.assertRaises(ValueError): E.S.config(78, 8, 16, 2)
        pure = importlib.import_module('trotterlib.pauli_rte_event_extension')
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

    def test_real_mean_connection_is_mocked_for_each_fixed_candidate(self):
        numerical = importlib.import_module('trotterlib.pauli_vector_tight_enclosure')
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
