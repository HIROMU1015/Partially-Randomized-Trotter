"""Artificial operators only; no molecular input, circuits, sampling or prices.

These checks establish algebraic reuse conditions, not a production Pauli
provider or scientific qualification. The existing finite-RTE primitives and
current PR macro kernel are reused unchanged.
"""
import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
package=types.ModuleType('df_pauli_contract_fixture')
package.__path__=[str(ROOT/'src/trotterlib')]
sys.modules[package.__name__]=package
rte=importlib.import_module(package.__name__+'.rte')
pauli=importlib.import_module(package.__name__+'.pauli_signal_adapter')
import numpy as np
from scipy.linalg import expm
from trottertracks import pr_rpe_signal_provider as provider


def matrix(label):
    return pauli.PauliHamiltonian.from_terms([(label,1.)],num_qubits=len(label)).sparse_operator().toarray()


class ContractTests(unittest.TestCase):
    def setUp(self):
        def forbidden(*args,**kwargs):raise AssertionError('No saved inputs/circuit construction in artificial tests')
        self.guards=[patch.object(np,'load',side_effect=forbidden),patch.object(pauli,'QuantumCircuit',side_effect=forbidden)]
        for g in self.guards:g.start();self.addCleanup(g.stop)

    def test_individual_pauli_leaks_and_projected_pauli_is_not_involutory(self):
        # Three artificial number modes, N=1; q0 is the rightmost label bit.
        # Spectator |001> leaves this sector under XXI or YYI separately.
        xx,yy=matrix('XXI'),matrix('YYI');indices=[1,2,4]
        projector=np.eye(8)[:,indices];outside=np.eye(8)-projector@projector.T
        tail=.3*(xx+yy)
        self.assertGreater(np.linalg.norm(outside@xx@projector),.9)
        np.testing.assert_allclose(outside@tail@projector,0.,atol=1e-15)
        reduced=projector.T@xx@projector
        self.assertGreater(np.linalg.norm(reduced@reduced-np.eye(3)),.9)
        np.testing.assert_allclose(projector.T@xx@xx@projector,np.eye(3),atol=1e-15)
        self.assertGreater(np.linalg.norm(projector.T@expm(-.3j*xx)@projector-expm(-.3j*reduced)),.04)

    def test_event_mean_K0_K2_sign_order_and_dictionary_dependent_normalizer(self):
        x,z,y=matrix('X'),matrix('Z'),matrix('Y')
        dictionaries=[[(.37,x),(-.21,z)],[(.37,x),(-.21,z),(.1,y),(-.1,y)]]
        for cutoff in (0,2):
            for dt in (.08,-.08):
                corrected=[];norms=[]
                for dictionary in dictionaries:
                    lam=math.fsum(abs(c) for c,p in dictionary)
                    components=tuple(rte.RTEComponent(str(i),abs(c)/lam,abs(c),1 if c>0 else -1) for i,(c,p) in enumerate(dictionary))
                    operators={str(i):(1 if c>0 else -1)*p for i,(c,p) in enumerate(dictionary)}
                    distribution=rte.finite_rte_distribution(lam*dt,cutoff)
                    events=rte.enumerate_rte_events(components,distribution)
                    raw=rte.exact_enumerated_event_mean_operator(events,operators)
                    for event in events:
                        u=rte.event_unitary(event,operators)
                        np.testing.assert_allclose(u.conj().T@u,np.eye(2),atol=2e-14)
                    b=distribution.exact_finite_distribution
                    expected=rte.finite_taylor_operator(.37*x-.21*z,dt,cutoff)
                    np.testing.assert_allclose(b*raw,expected,rtol=0,atol=2e-14)
                    corrected.append(b*raw);norms.append(b)
                np.testing.assert_allclose(*corrected,rtol=0,atol=2e-14)
                self.assertGreater(norms[1],norms[0])

    def test_full_mean_can_be_reduced_after_total_tail_sector_closure(self):
        idx=[1,2,4];projector=np.eye(8)[:,idx];eye=np.eye(8)
        blocks=[.07*matrix('IIZ'),.09*matrix('IZZ'),.3*(matrix('XXI')+matrix('YYI'))+.13*eye]
        state=projector@np.array([1.,.3j,-.2]);state/=np.linalg.norm(state)
        tails={1:dict(scalar=.13,lambda_R=.6,source={'fixture':'ARTIFICIAL_ONLY'})}
        full=provider.make_context(blocks,state,.17,tails,binding={'rank':2},source={'fixture':'ARTIFICIAL_ONLY'})
        reduced=provider.make_context([projector.T@a@projector for a in blocks],projector.T@state,.17,tails,binding={'rank':2},source={'fixture':'ARTIFICIAL_ONLY'})
        st=dict(signed_weights=[1.],allocation=[2])
        for signed_time in (.4,-.4):
            c=dict(prefix=1,formula='2nd',T=signed_time,q=2,R_outer=2,K=2)
            f,_=provider._macros(full,c,st);s,_=provider._macros(reduced,c,st)
            for uf,us in zip(f['operators'],s['operators']):
                np.testing.assert_allclose(projector.T@uf@projector,us,rtol=0,atol=2e-14)
            exact,corrected,raw=[np.linalg.matrix_power(u,2)@state for u in f['operators']]
            attenuation=math.exp(-2*f['log_B_per_outer'])
            np.testing.assert_allclose(raw,attenuation*corrected,rtol=0,atol=2e-14)
            phase=np.exp(-1j*signed_time*(.17+.13))
            for i in range(3):
                left=complex(np.vdot(state,np.linalg.matrix_power(f['operators'][i],2)@state*phase))
                right=complex(np.vdot(reduced.state,np.linalg.matrix_power(s['operators'][i],2)@reduced.state*phase))
                self.assertLess(abs(left-right),2e-14)

    def test_global_collection_scalar_and_DF_prefix_are_distinct(self):
        model=pauli.PauliHamiltonian.from_terms([('II',.13),('XI',.4),('XI',-.1),('IZ',-.2)],num_qubits=2)
        self.assertEqual(model.constant,.13)
        self.assertEqual(len(model.terms),2)
        self.assertAlmostEqual(math.fsum(abs(c) for label,c in model.terms),.5)
        # Existing PauliSignalAdapter.ld counts Pauli terms; it is not a DF
        # fragment prefix and cannot replace the current head builder.
        old=pauli.PauliSignalAdapter(model,'PR',1,.1,2,2)
        self.assertEqual(old.metadata()['split_unit'],'nonidentity_Pauli_term')
        self.assertEqual(old.retained,model.terms[:1])


if __name__=='__main__':unittest.main()
