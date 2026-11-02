"""Artificial numerical and gate checks; no real inputs or transpilation."""
import math
import unittest
from copy import deepcopy
import numpy as np
from scipy.linalg import expm
from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
from trotterlib.df_hamiltonian import DFHamiltonian
from trottertracks.resource_applicability.ax2b_limits import CallBudget
from trottertracks.resource_applicability.ax2b_independent_reference import occupation_df_matrix
from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import recurrence_paths as dense_paths
from trottertracks.resource_applicability.h6_matched_port_v1 import compact_signal
from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
from trottertracks.resource_applicability.h8_sparse_reference_v1 import (
    OccupationCSR, SpinPrimitive, df_operator, reference_pair, exponential_action, recurrence_paths)
from trottertracks.resource_applicability.h8_signal_preflight_contract_v1 import (
    cells, plan, registrations, resources, seed)
from trottertracks.resource_applicability.h48_preflight_contract_v1 import task_caps


def synthetic():
    spatial = np.array([[.7, .2+.1j, -.03j], [.2-.1j, -.4, .11], [.03j, .11, .2]])
    one = np.zeros((6,6), complex); one[::2,::2] = spatial; one[1::2,1::2] = spatial
    sector = PhysicalSector.spin_sector(n_qubits=6, nelec_alpha=1, nelec_beta=1)
    state = np.arange(1, sector.dimension+1)+1j*np.arange(sector.dimension)
    state = state/np.linalg.norm(state)
    return one, sector, state


class SparseTests(unittest.TestCase):
    def test_independent_CSR_all_columns(self):
        one, sector, _ = synthetic(); top = OccupationCSR(sector.basis_indices, 6)
        ref = np.array(occupation_df_matrix(.3, one, [-.7], [one], sector.basis_indices))
        a = top.one(one); op, _ = df_operator(.3, a, [-.7], [a], CallBudget(reference_matvec=100))
        actual = np.column_stack([op@v for v in np.eye(len(ref))])
        self.assertLess(np.linalg.norm(actual-ref), 1e-12)

    def test_complex_spin_primitive_positive_negative_zero(self):
        one, sector, state = synthetic()
        ref = np.array(occupation_df_matrix(0., one, [], [], sector.basis_indices))
        for weight in (None, -.37, .41):
            primitive = SpinPrimitive(one, sector, weight=weight)
            h = ref if weight is None else weight*(ref@ref)
            for t in (0., .8, -.91):
                self.assertLess(np.linalg.norm(primitive(state,t)-expm(-1j*t*h)@state), 1e-12)

    def test_exponential_sparse_pair(self):
        one, sector, state = synthetic(); a = OccupationCSR(sector.basis_indices,6).one(one)
        budget = CallBudget(reference_matvec=10000)
        op, trace = df_operator(.3, a, [.2], [a], budget)
        target = .3*np.eye(len(state))+a.toarray()+.2*(a@a).toarray()
        for t in (.8, -.3):
            actual = exponential_action(op,trace,state,t)
            self.assertLess(np.linalg.norm(actual-expm(-1j*t*target)@state),1e-12)
            signal, _ = reference_pair(op,trace,state,t)
            self.assertLess(abs(signal-np.vdot(state,actual)),1e-12)
        self.assertGreater(budget.used['reference_matvec'],0)

    def test_sparse_budget_before_action(self):
        one, sector, state = synthetic(); a = OccupationCSR(sector.basis_indices,6).one(one)
        budget = CallBudget(reference_matvec=0); op, _ = df_operator(0., a, [], [], budget)
        with self.assertRaisesRegex(RuntimeError,'CALL_BUDGET'):
            op@state

    def test_structure_rejects_spin_mixing(self):
        one, sector, _ = synthetic(); one[0,1] = 1e-30; one[1,0] = 1e-30
        with self.assertRaises(ValueError):
            OccupationCSR(sector.basis_indices,6).one(one)
        with self.assertRaises(ValueError):
            SpinPrimitive(one,sector)

    def test_unequal_spin_coefficients_rejected(self):
        one, sector, _ = synthetic(); one[1,1] += 1e-15
        with self.assertRaises(ValueError):
            SpinPrimitive(one,sector)

    def test_nonfinite_rejected(self):
        one, sector, _ = synthetic(); one[0,0] = np.inf
        with self.assertRaises(ValueError):
            OccupationCSR(sector.basis_indices,6).one(one)
        with self.assertRaises(ValueError):
            SpinPrimitive(one,sector)

    def test_incomplete_sector_rejected(self):
        _, sector, _ = synthetic()
        with self.assertRaises(ValueError):
            OccupationCSR(sector.basis_indices[:-1],6)

    def test_exact_zero_no_cutoff(self):
        one, sector, _ = synthetic(); one[0,4] = 1e-30; one[4,0] = 1e-30
        one[1,5] = 1e-30; one[5,1] = 1e-30
        a = OccupationCSR(sector.basis_indices,6).one(one)
        self.assertTrue(np.any(np.abs(a.data)==1e-30))

    def test_forward_vs_dense_and_Horner(self):
        one, sector, state = synthetic(); a = OccupationCSR(sector.basis_indices,6).one(one)
        dense = a.toarray(); budget = CallBudget(reference_matvec=100000)
        op, trace = df_operator(0., a, [], [], budget)
        c = dict(id='artificial',method='B2',prefix=0,q=2,formula='2nd',r=2,R=4,K=4)
        actions = [SpinPrimitive(one,sector)]
        kw = dict(cell=c,T=.2,scalar=.3,lambda_r=.1,maximum_tail=100)
        s = recurrence_paths(state,actions,(op,trace),**kw)
        d = dense_paths(state,actions,dense,**kw)
        for k in s['vectors']:
            self.assertLess(np.linalg.norm(s['vectors'][k]-d['vectors'][k]),1e-12)
        h = compact_signal(state,actions,lambda v:op@v,cell=c,T=.2,scalar=.3,lambda_r=.1,
                           budget=ActionBudget(100,100))
        for k in h['vectors']:
            self.assertLess(np.linalg.norm(h['vectors'][k]-s['vectors'][k]),1e-12)
        self.assertLess(abs(h['signals']['raw']*math.exp(h['log_B'])-h['signals']['corrected']),1e-12)


class GateTests(unittest.TestCase):
    def test_native_16_mode_primitives_on_artificial_input(self):
        from trottertracks.resource_applicability.ax2b_native_df_v5 import make_native_block_action
        from trottertracks.resource_applicability.ax2b_h4_science_v5 import checked_basis_bridge
        from trottertracks.resource_applicability.ax2a_state_action import project_primitive_checked
        n = 16; one = np.diag(np.repeat(np.arange(1,9)*.03,2)).astype(complex)
        one[0,2] = one[2,0] = one[1,3] = one[3,1] = .07
        ham = DFHamiltonian(.13,one,np.array([-.31]),(one.copy(),),{})
        sector = PhysicalSector.spin_sector(n_qubits=n,nelec_alpha=4,nelec_beta=4)
        state = np.zeros(sector.dimension,complex); state[3] = 1
        full = np.zeros(1<<n,complex);full[sector.basis_indices] = state
        indices, _ = checked_basis_bridge(ham,sector,full,state,1e-12)
        p = _prepare(ham,1)
        for i, b in enumerate(p.deterministic_blocks):
            native = make_native_block_action(b,max_instructions=1000000)
            fast = SpinPrimitive(one,sector,weight=None if i==0 else -.31)
            for t in (.23,-.71):
                actual = project_primitive_checked(lambda v:native(v,t),state,indices,1<<n,leakage_tolerance=1e-12)
                self.assertLess(np.linalg.norm(actual-fast(state,t)),1e-11)


class ContractTests(unittest.TestCase):
    def test_paired_static_budget(self):
        p = plan(); signals = {'signals':[dict(cell=c,signals={'corrected':{'real':1.,'imag':0.}}) for c in cells()]}
        heavy = registrations('a'*40,'HEAVY',signals); rest = registrations('a'*40,'REMAINING',signals)
        self.assertEqual([x['index'] for x in heavy],[3]); self.assertEqual(len(rest),7)
        self.assertEqual({x['task']['cell_id'] for x in heavy+rest},{c['id'] for c in cells()})
        caps = [task_caps(x['task'],p['caps_proposed']) for x in heavy+rest]
        self.assertEqual(sum(c['compile'] for c in caps),20)
        self.assertEqual(sum(c['trajectory'] for c in caps),4)
        self.assertEqual(sum(c['occurrence'] for c in caps),15)
        self.assertEqual(sum(c['control_probe'] for c in caps),40)

    def test_seed_independent_of_slot_and_H6_namespace(self):
        self.assertEqual(seed('a'*40,'c'),seed('a'*40,'c'))
        self.assertNotEqual(seed('a'*40,'c'),seed('b'*40,'c'))
        from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed as old
        self.assertNotEqual(seed('a'*40,'c'),old('c','exploration',0))

    def test_weighted_admission_never_eight_heavy_workers(self):
        from trottertracks.resource_applicability.h6_parallel_contract_v1 import slots
        r = resources('REMAINING',3)
        self.assertEqual(slots(r),3)
        self.assertEqual(r['aggregate_rss_bytes'],76*2**30)
        with self.assertRaises(ValueError):
            resources('REMAINING',8)
        with self.assertRaises(ValueError):
            resources('HEAVY',2)

    def test_no_full_campaign_or_input_regeneration(self):
        p = plan();self.assertFalse(p['H8_full_campaign_authorized'])
        self.assertEqual([p['caps_proposed'][k] for k in ('integral_build','df_decomposition','solver_matvec')],[0,0,0])
        self.assertTrue(any(t<0 for _,t in p['coverage']['physical_times']))
        self.assertEqual(p['compiler_proposed']['optimization_level'],1)


if __name__=='__main__':
    unittest.main()
