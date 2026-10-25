"""Artificial inputs only: body correspondence, complex enclosures, accounting."""
from fractions import Fraction as Q
import importlib
from pathlib import Path
import sys
import types
import unittest

pkg = types.ModuleType("b0_test_private")
pkg.__path__ = [str(Path(__file__).resolve().parents[1]/"src/trotterlib")]
sys.modules[pkg.__name__] = pkg
E = importlib.import_module(pkg.__name__+".h4_pauli_b0_execution")


def model(n=2, q=2):
    return dict(n_qubits=n, qubit_order="qiskit_little_endian", step_time=.25,
                state=[[2.,1.]] + [[0.,0.]]*((1<<n)-1),
                hamiltonian=dict(constant=-.25),
                cases=dict(PR=dict(config=dict(q=q,T=.25*q),
                                   blocks=[dict(label="Z"+"I"*(n-1),coefficient=.5),
                                           dict(label="I"*(n-1)+"Z",coefficient=-.125)])))


def reference():
    return dict(reference=dict(real=dict(lower="0",upper="0"),imag=dict(lower="0",upper="0")))


class AccountingTests(unittest.TestCase):
    def target(self, c, s, radius="0"):
        return dict(axes={a:dict(signed_bias_exact=v,u_enc_exact=radius)
                          for a,v in zip(E.AXES,(c,s))})

    def saved(self):
        return [dict(case="pauli_jw_PR",epsilon_sig=e,G_exact="1000000000000000000",
                     G_guard_range_exact=["1000000000000000000","2000000000000000000"],
                     N_cos=1,N_sin=1,N_total=2,cohort_n=256,hypothetical_rho_new="0") for e in E.EPSILONS]

    def test_own_bias_allowance_and_saved_PR_not_recomputed(self):
        saved=self.saved()
        out=E.conditional_results(self.target("0","0","1/1000000000000"),
                                  {a:dict(RZ=10) for a in E.AXES},saved)
        self.assertEqual(out["N_scalar_evaluations"],20)
        self.assertEqual(out["PR_N_G_recomputations"],0)
        for row,old in zip(out["rows"],saved):
            self.assertEqual(row["G_exact"],str(10*row["N_total"]))
            self.assertEqual(row["saved_PR"]["G_exact"],old["G_exact"])
            self.assertEqual(row["independent_guard_order"],"B0_LOWER_THROUGHOUT_INDEPENDENT_GUARDS")
            self.assertIsNone(row["actual_N"])
            self.assertIsNone(row["actual_G"])
            self.assertFalse(row["formal_certificate_accepted"])

    def test_bias_failure_is_null_and_not_a_price_ranking(self):
        out=E.conditional_results(self.target("1","0"),{a:dict(RZ=10) for a in E.AXES},self.saved())
        for row in out["rows"]:
            self.assertIsNone(row["N_total"]);self.assertIsNone(row["G_exact"])
            self.assertEqual(row["rho_zero_order"],"UNDETERMINED")

    def test_precision_escalation_distinguishes_bias_and_width(self):
        self.assertEqual({r["status"] for r in E.precision_decisions(self.target("1","1"))},{"BIAS_EXCEEDS_TOLERANCE"})
        self.assertEqual({r["status"] for r in E.precision_decisions(self.target("0","0"))},{"GUARD_MET"})
        self.assertIn("WIDTH_AMBIGUITY",{r["status"] for r in E.precision_decisions(self.target("0","0","1/10000"))})

    def test_exact_interval_center_and_radius(self):
        self.assertEqual(E.center_radius(dict(lower="-3/7",upper="1/7")),(Q(-1,7),Q(2,7)))
        with self.assertRaises(ValueError):E.center_radius(dict(lower="2",upper="1"))


class ArbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import flint
        except ImportError:raise unittest.SkipTest("Arb runtime only")

    def test_known_diagonal_complex_signal_and_raw_norm(self):
        C=importlib.import_module(pkg.__name__+".h4_arb_bridge")._backend()
        for n in (2,3):
            m=model(n); result=E.target_enclosure(m,reference(),64)
            with C.ctx.workprec(192):
                # On |0...0>, retained sum=.375 and constant=-.25, T=.5.
                exact=C.phase(C.real(Q(-1,16)))
                for key,value in (("real",exact.real),("imag",exact.imag)):
                    bound=result["physical_signal"][key]
                    self.assertLessEqual(Q(bound["lower"]),C.dyadic(value.lower()))
                    self.assertGreaterEqual(Q(bound["upper"]),C.dyadic(value.upper()))
            self.assertEqual(result["raw_norm_exact"],"5")
            self.assertFalse(result["state_renormalized"])
            self.assertEqual(result["state_actions"],1)
            self.assertEqual(result["reference_state_actions"],0)

    def test_noncommuting_Y_and_negative_retained_term(self):
        C=importlib.import_module(pkg.__name__+".h4_arb_bridge")._backend()
        m=model(3);m["cases"]["PR"]["blocks"][1]["label"]="YXI"
        result=E.target_enclosure(m,reference(),64)
        # Independent small dense C oracle, only on this artificial fixture.
        with C.ctx.workprec(192):
            state=C.acb_mat([[C.complex_ball(z)] for z in m["state"]]);I=C.identity(8)
            blocks=[]
            for b in m["cases"]["PR"]["blocks"]:
                a=C.real(Q(b["coefficient"])*Q(m["step_time"])/2)
                blocks.append(I*a.cos()+C.pauli(b["label"])*C.acb(0,-a.sin()))
            out=C.evolve(state,blocks,None,C.phase(C.real(Q(1,16))),2,4)
            exact=C.signal(state,out)/5
            for key,value in (("real",exact.real),("imag",exact.imag)):
                bound=result["physical_signal"][key]
                self.assertLessEqual(Q(bound["lower"]),C.dyadic(value.lower()))
                self.assertGreaterEqual(Q(bound["upper"]),C.dyadic(value.upper()))

    def test_reference_whole_interval_and_norm_mismatch(self):
        m=model();ref=reference();ref["reference"]["real"]=dict(lower="-1/100",upper="1/100")
        out=E.target_enclosure(m,ref,64)
        self.assertGreaterEqual(Q(out["axes"]["cosine"]["u_enc_exact"]),Q(1,500))
        ref["state_norm"]=dict(lower="1",upper="1")
        with self.assertRaises(ValueError):E.target_enclosure(m,ref,64)


class QiskitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import qiskit
        except ImportError:raise unittest.SkipTest("Compiler runtime only")

    def test_saved_body_operand_phase_correspondence_and_failure(self):
        from qiskit import QuantumCircuit
        P=importlib.import_module(pkg.__name__+".pauli_signal_adapter")
        m=model();body=QuantumCircuit(2)
        blocks=m["cases"]["PR"]["blocks"]
        for _ in range(2):
            for ordered in (blocks,reversed(blocks)):
                for b in ordered:P.append_pauli_evolution(body,b["label"],b["coefficient"],.125)
            body.global_phase-=m["hamiltonian"]["constant"]*.25
        prefix=dict(L_D=2,retained_labels=[b["label"] for b in blocks])
        metadata=dict(sidecar=dict(naming=dict(global_phase=float(body.global_phase))))
        self.assertEqual(E.validate_retained_body(body,m,prefix,metadata)["instruction_count"],8)
        body.global_phase+=.1
        with self.assertRaises(ValueError):E.validate_retained_body(body,m,prefix,metadata)


if __name__=="__main__":unittest.main()
