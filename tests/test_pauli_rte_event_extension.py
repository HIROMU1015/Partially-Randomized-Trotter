"""Synthetic 2/3-qubit checks; do not load a molecular model or reference."""
import importlib,json,sys,types,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('event_test');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
E=importlib.import_module(p.__name__+'.pauli_rte_event_extension')
P=importlib.import_module(p.__name__+'.pauli_signal_adapter')
F=importlib.import_module(p.__name__+'.fixed_signal_evaluation')

def toy(n=2):
    terms=[('I'*(n-1)+'X',.3),('I'*(n-1)+'Y',-.17),('Z'+'I'*(n-1),.11),('I'*n,.19)]
    return P.PauliHamiltonian.from_terms(terms,num_qubits=n,model_id='artificial_event_extension')

class Checks(unittest.TestCase):
    def test_original_r8_config_and_law_unchanged(self):
        for K in (2,4):
            self.assertEqual(asdict(E.config(8,K)),asdict(E.S.config(0,8,8,K)))
            h=toy();self.assertEqual(E.law_descriptor(h,8,K),E.S.law_descriptor(h,E.S.config(0,8,8,K)))
    def test_law_counts_angles_normalization_use_existing_adapter(self):
        import math
        for r in (16,32,64,128):
            for K in (2,4):
                h=toy();d=E.law_descriptor(h,r,K);a=P.PauliSignalAdapter(h,'PR',0,.1,r,K)
                self.assertEqual(d['step_time'],.1)
                self.assertEqual(d['orders'],list(range(0,K+1,2)))
                self.assertEqual(d['nominal_polynomial_degree'],K+1)
                self.assertEqual(d['angles'],[math.atan(a.distribution.dimensionless_step_time/(k+1)) for k in d['orders']])
                self.assertEqual(d['B_saved_binary64'],math.exp(8*a.log_B_step))
                self.assertEqual(d['retained'],[])
    def test_domains_do_not_widen_old_search_or_H4(self):
        with self.assertRaises(ValueError):E.S.config(0,8,16,2)
        for args in [(1,8,16,2),(0,4,16,2),(0,8,256,2),(0,8,16,6),(0,8,True,2)]:
            with self.assertRaises(ValueError):E.candidate_id(*args)
        with self.assertRaises(ValueError):E.validate('H4',64,2)
        with self.assertRaises(ValueError):E.validate('H2',16,2)
    def test_seed_graph_unchanged_and_cohorts_separate(self):
        for molecule in ('H4','H6'):
            for cohort in ('exploration','confirmation'):
                got=E.trajectory_seed(molecule,8,2,cohort,0)
                entropy=[20261025,3101 if molecule=='H4' else 6,0,8,8,2,101 if cohort=='exploration' else 202,0]
                self.assertEqual(got,int(np.random.SeedSequence(entropy).generate_state(1,dtype=np.uint64)[0]))
            self.assertNotEqual(E.trajectory_seed(molecule,16,2,'exploration',0),E.trajectory_seed(molecule,16,2,'confirmation',0))
    def test_large_event_count_body_matches_original_sampler_path(self):
        for molecule,r,K in [('H4',16,4),('H6',128,2)]:
            h=toy();body,wrapped,meta=E.trajectory(h,molecule,r,K,'exploration',0)
            expected=P.PauliSignalAdapter(h,'PR',0,.1,r,K).trajectory(q=8,seed=meta['seed'])
            self.assertEqual(body,expected);self.assertEqual(meta['event_count'],8*r)
            self.assertTrue(all(e.taylor_order<=K for e in meta['events']))
            self.assertTrue(all(w.num_qubits==3 for w in wrapped.values()))
    def test_compiled_toy_phase_and_both_measurements(self):
        from qiskit.quantum_info import Statevector,Operator
        from event_test.rte_compiled_cost import transpile_and_measure_cost
        h=toy(3);body,wrapped,meta=E.trajectory(h,'H4',16,4,'confirmation',0)
        raw=np.array([complex((i+1)/16,(2-i)/32) for i in range(8)])
        nu=np.vdot(raw,raw).real;ideal=np.vdot(raw,Operator(body).data@raw)/nu
        for axis in E.A.AXES:
            cost=transpile_and_measure_cost(wrapped[axis],F.default_compiler(),actual_circuit_fingerprint='artificial_'+axis)
            circuit=cost.transpiled_circuit.remove_final_measurements(inplace=False)
            v=Statevector(np.concatenate([raw,np.zeros(8)])).evolve(circuit).data
            measured=(np.vdot(v[:8],v[:8])-np.vdot(v[8:],v[8:])).real/nu
            self.assertAlmostEqual(measured,ideal.real if axis=='cosine' else ideal.imag,places=10)

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--write-mean-fixtures':
        h=toy();raw=[[.5,0.],[.125,.0625],[0.,-.125],[.25,0.]]
        ref=dict(reference={k:dict(lower='0',upper='0') for k in ('real','imag')},method='ARTIFICIAL_ZERO_REFERENCE')
        records=[dict(artificial_qubits=2,law=E.law_descriptor(h,r,K),model=dict(state=raw,n_qubits=2,hamiltonian=dict(constant=h.constant,terms=h.terms)),reference=ref) for r,K in [(16,2),(32,4)]]
        Path(sys.argv[2]).write_text(json.dumps(records,indent=2)+'\n')
    else:unittest.main()
