"""Artificial H2 runner connection checks, no molecular science."""
import importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('h2_scope_test',ROOT/'scripts/run_h2_pauli_rte_validation.py');R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
class Checks(unittest.TestCase):
    def test_narrow_domain_and_old_validator_preserved(self):
        for r in [8,16,32]:self.assertEqual(R.validate(8,r,2),f'p000_q8_r{r}_K2')
        for v in [(1,16,2),(8,64,2),(8,16,6)]:
            with self.assertRaises(ValueError):R.validate(*v)
        with self.assertRaises(ValueError):R.H.S.candidate_id(0,8,16,2)
    def test_artificial_r8_descriptor_equals_original(self):
        from hx_study.pauli_signal_adapter import PauliHamiltonian
        from hx_study.fixed_signal_evaluation import FixedSignalConfig
        h=PauliHamiltonian(2,.17,(('IZ',.4),('XX',-.08)),'{}')
        cfg=FixedSignalConfig('pauli_jw',method='PR',T=.8,L_D=0,q=8,r=8,K=4)
        self.assertEqual(R.law_fn(h,cfg),R.H.S.law_descriptor(h,cfg))
    def test_seed_addresses_unchanged_and_cohorts_separate(self):
        c=dict(molecule=2,config=dict(L_D=0,q=2,r=4,K=2))
        a=R.H.seeds(c,'confirmation',17);b=R.H.seeds(c,'exploration',17)
        self.assertNotEqual(a,b)
        import numpy as np
        self.assertEqual(a,int(np.random.SeedSequence([20261025,2,0,2,4,2,202,17]).generate_state(1,dtype=np.uint64)[0]))
    def test_nulls_not_ranked(self):
        a=dict(candidate_id='a',G_exact=None,eligibility='NUMERICALLY_UNDETERMINED',G_SE=None)
        b=dict(candidate_id='b',G_exact='10',eligibility='CONDITIONAL_GUARD_ELIGIBLE',G_SE=1)
        self.assertEqual(R.minimum([a,b]),b);self.assertFalse(R.near(a,b))
    def test_extension_only_if_unresolved_and_own_variance_reducible(self):
        a=dict(candidate_id='a',G_exact='100',eligibility='CONDITIONAL_GUARD_ELIGIBLE',G_SE=3)
        b=dict(candidate_id='b',G_exact='101',eligibility='CONDITIONAL_GUARD_ELIGIBLE',G_SE=2)
        self.assertTrue(R.extend_rule(a,[b],96));a['G_SE']=1;self.assertFalse(R.extend_rule(a,[b],96))
    def test_artificial_extended_laws_generated_for_oracle(self):
        from hx_study.pauli_signal_adapter import PauliHamiltonian
        from hx_study.fixed_signal_evaluation import FixedSignalConfig
        h=PauliHamiltonian(2,.17,(('IZ',.4),('XX',-.08),('YI',.03)),'{}')
        raw=[[1.,0.],[.125,.25],[0.,0.],[0.,.125]];fixtures=[]
        for r,K in [(16,2),(32,4)]:
            law=R.law_fn(h,FixedSignalConfig('pauli_jw',method='PR',T=.8,L_D=0,q=8,r=r,K=K))
            self.assertEqual(law['nominal_polynomial_degree'],K+1)
            fixtures.append(dict(artificial_qubits=2,law=law,model=dict(n_qubits=2,state=raw,hamiltonian=dict(constant=.17,terms=[('IZ',.4),('XX',-.08),('YI',.03)]))))
        path=ROOT/'artifacts/h2_pauli_rte_validation/2026-10-26/artificial_mean_inputs.json'
        path.write_text(json.dumps(fixtures,indent=2)+'\n')
if __name__=='__main__':unittest.main()
