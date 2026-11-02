"""Author tests: exact algebra, error ledger, raw phases, bounded scope."""
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m5_algebraic_oneq_prototype as p
from run_m5_algebraic_oneq_artificial import fixtures,without_barriers


class AlgebraicTests(unittest.TestCase):
    def test_exact_dyadic_sum_reduces_with_zero_budget(self):
        plan=p.AlgebraicPlanner().plan([('rz',(.125,)),('rz',(.375,))],Fraction())
        self.assertEqual(plan['output'],(('rz',(.5,)),));self.assertEqual(plan['charge'],0)

    def test_nonrepresentable_sum_is_preserved_with_zero_budget(self):
        gates=[('rz',(.1,)),('rz',(.2,))]
        plan=p.AlgebraicPlanner().plan(gates,Fraction())
        self.assertEqual(plan['output'],tuple(gates));self.assertFalse(plan['changed'])
        self.assertEqual(plan['budget_blocked_groups'],1)

    def test_positive_charge_is_exact_angle_rounding_difference(self):
        plan=p.AlgebraicPlanner().plan([('rz',(.1,)),('rz',(.2,))],Fraction.from_float(1e-11))
        exact=Fraction.from_float(.1)+Fraction.from_float(.2)
        self.assertEqual(plan['charge'],abs(exact-Fraction.from_float(plan['output'][0][1][0])))
        self.assertGreater(plan['charge'],0)

    def test_SX_X_identities_do_not_add_global_phase(self):
        planner=p.AlgebraicPlanner()
        self.assertEqual(planner.plan([('sx',()),('sx',())],Fraction())['output'],(('x',()),))
        self.assertEqual(planner.plan([('sx',()),('sx',()),('x',())],Fraction())['output'],())
        self.assertEqual(planner.plan([('x',()),('x',())],Fraction())['output'],())

    def test_exact_cancellation_exposes_further_safe_reduction(self):
        q=fixtures()['exact_cancellation'];out,records,ledger=p.compile_artificial(q,budget_radians=0.,artificial=True)
        self.assertEqual(len(out.data),0);self.assertEqual(ledger.spent,0)
        self.assertLess(np.linalg.norm(Operator(q).data-Operator(out).data),1e-14)

    def test_small_nonzero_turn_is_never_dropped_by_threshold(self):
        for angle in (2.**-55,2.**-40,-2.**-40):
            gates=[('rz',(angle,))];plan=p.AlgebraicPlanner().plan(gates,Fraction.from_float(1e-11))
            self.assertEqual(plan['output'],tuple(gates))

    def test_two_pi_RZ_is_retained_including_controlled_phase(self):
        q=fixtures()['two_pi_not_identity'];out,records,ledger=p.compile_artificial(q,budget_radians=0.,artificial=True)
        self.assertEqual(out.count_ops()['rz'],1)
        self.assertEqual(out.data[0].operation.params[0],2*math.pi)
        self.assertLess(np.linalg.norm(Operator(without_barriers(q).to_gate().control()).data-Operator(out.to_gate().control()).data),1e-13)

    def test_error_budget_is_cumulative_and_cannot_overspend(self):
        ledger=p.AngleLedger(1e-16);planner=p.AlgebraicPlanner();charges=[]
        for _ in range(8):
            plan=planner.plan([('rz',(.1,)),('rz',(.2,))],ledger.remaining)
            ledger.commit(plan['charge']);charges.append(plan['charge'])
        self.assertLessEqual(ledger.spent,ledger.budget)
        self.assertGreater(sum(c>0 for c in charges),0);self.assertLess(sum(c>0 for c in charges),8)
        with self.assertRaises(ValueError):ledger.commit(ledger.remaining+1)

    def test_overflow_falls_back_to_original_angles(self):
        gates=[('rz',(1e308,)),('rz',(1e308,))];plan=p.AlgebraicPlanner().plan(gates,Fraction())
        self.assertEqual(plan['output'],tuple(gates));self.assertEqual(plan['overflow_preserved_groups'],1)

    def test_invalid_angles_and_budgets_fail_closed(self):
        for value in (float('nan'),float('inf'),-1.,True,1):
            with self.subTest(value=value),self.assertRaises(ValueError):p.AngleLedger(value)
        for name,params in [('h',()),('rz',(float('nan'),)),('rz',(1,))]:
            with self.assertRaises(ValueError):p.AlgebraicPlanner().plan([(name,params)],Fraction())

    def test_symbolic_and_H6_sized_circuits_are_refused(self):
        from qiskit.circuit import Parameter
        q=QuantumCircuit(13)
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_SCOPE_ONLY'):p.compile_artificial(q,budget_radians=0.,artificial=True)
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_SCOPE_ONLY'):p.compile_artificial(QuantumCircuit(1),budget_radians=0.)
        q=QuantumCircuit(1);q.rz(Parameter('a'),0)
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_SCOPE_ONLY'):p.compile_artificial(q,budget_radians=0.,artificial=True)

    def test_cache_reuse_does_not_change_plan_or_budget(self):
        planner=p.AlgebraicPlanner();gates=[('rz',(.125,)),('rz',(.375,))]
        a=planner.plan(gates,Fraction());b=planner.plan(gates,Fraction())
        self.assertEqual(a,b);self.assertEqual(planner.misses,1);self.assertEqual(planner.hits,1)

    def test_cached_tokens_are_rechecked_against_remaining_budget(self):
        planner=p.AlgebraicPlanner();gates=[('rz',(.1,)),('rz',(.2,))]
        a=planner.plan(gates,Fraction());b=planner.plan(gates,Fraction.from_float(1e-11))
        self.assertFalse(a['changed']);self.assertTrue(b['changed'])
        self.assertEqual(planner.misses,1);self.assertEqual(planner.hits,1)

    def test_cancelling_large_angles_retains_tiny_exact_sum(self):
        plan=p.AlgebraicPlanner().plan([('rz',(1.7,)),('rz',(2.**-54,)),('rz',(-1.7,))],Fraction())
        self.assertEqual(plan['output'],(('rz',(2.**-54,)),));self.assertEqual(plan['charge'],0)

    def test_ideal_local_RZ_error_respects_exact_angle_charge(self):
        from m5_local_resynthesis_diagnostic import context,action
        gates=[('rz',(.1,)),('rz',(.2,))];plan=p.AlgebraicPlanner().plan(gates,Fraction.from_float(1e-11))
        c=context(120);difference=action(c,gates)-action(c,plan['output'])
        raw=c.sqrt(sum(abs(x)**2 for x in difference));charge=c.mpf(plan['charge'].numerator)/plan['charge'].denominator
        self.assertLessEqual(raw,charge)  # Ideal matrices, not Qiskit floating simulation.

    def test_other_preset_opt1_tasks_still_execute(self):
        q=QuantumCircuit(2);q.cx(0,1);q.cx(0,1);q.rz(.125,0);q.rz(.375,0)
        out,records,ledger=p.compile_artificial(q,budget_radians=0.,artificial=True)
        self.assertEqual(out.count_ops().get('cx',0),0);self.assertEqual(out.count_ops()['rz'],1)

    def test_custom_gate_lookalike_is_refused(self):
        from qiskit.circuit import Gate
        from qiskit.converters import circuit_to_dag
        q=QuantumCircuit(1);q.append(Gate('rz',1,[.2]),[0]);q.rz(.3,0)
        reducer=p.AlgebraicOneQubitPrototype(p.AngleLedger(0.),p.AlgebraicPlanner(),lambda row:None,artificial=True)
        with self.assertRaisesRegex(ValueError,'STANDARD_GATE_IDENTITY'):reducer.run(circuit_to_dag(q))

    def test_full_preset_raw_operators_and_nonincreasing_cost(self):
        for budget in (0.,1e-11):
            for name,q in fixtures().items():
                with self.subTest(budget=budget,name=name):
                    out,records,ledger=p.compile_artificial(q,budget_radians=budget,artificial=True)
                    self.assertLess(np.linalg.norm(Operator(q).data-Operator(out).data),1e-12)
                    self.assertEqual(float(q.global_phase).hex(),float(out.global_phase).hex())
                    self.assertLessEqual(out.count_ops().get('rz',0),q.count_ops().get('rz',0))
                    self.assertLessEqual(len(out.data),len(q.data))
                    self.assertTrue(all(not r['Euler_resynthesis'] and not r['small_angle_threshold'] and r['global_phase_bits_unchanged'] for r in records))

    def test_default_cli_does_not_execute_and_existing_output_is_preserved(self):
        script=Path(__file__).resolve().parents[1]/'scripts/run_m5_algebraic_oneq_artificial.py'
        result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,check=True)
        self.assertIn('ARTIFICIAL_SCOPE_ONLY',result.stdout)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'kept.json';path.write_text('preserved')
            result=subprocess.run([sys.executable,str(script),'--run-artificial','--output',str(path)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(path.read_text(),'preserved')


if __name__=='__main__':unittest.main()
