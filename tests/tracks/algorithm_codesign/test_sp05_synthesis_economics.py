"""Focused technical fixtures only; no registered target synthesis/economics."""
from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
from trottertracks.algorithm_codesign.synthesis_placement.economics import (
    angle, bounds, configure, economics, error_guard, exact_sequence, interval,
    key, pai, product, scaled, sequence_matrix, synthesize,
)
from trottertracks.algorithm_codesign.synthesis_placement.launch import (
    consume_marker, validate_binding,
)

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/tracks/algorithm_codesign/run_sp05_synthesis_economics.py"
spec = importlib.util.spec_from_file_location("sp05_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def mid(b):
    return (mp.mpf(b["lo"])+mp.mpf(b["hi"]))/2


def rz(t):
    return mp.matrix([[mp.exp(-mp.j*t/2), 0], [0, mp.exp(mp.j*t/2)]])


def channel(u):
    return mp.matrix([[u[i//2,j//2]*mp.conj(u[i%2,j%2]) for j in range(4)] for i in range(4)])


class Semantics(unittest.TestCase):
    def setUp(self):
        configure()

    def test_common_fast_path_cost_and_error_all_exact_gate_fixtures(self):
        for k in range(8):
            a = angle("pi", k, 4)
            s = exact_sequence(a)
            self.assertEqual(s.count("T")+s.count("t"), k % 2)
            self.assertLess(mp.mpf(error_guard(s, a)["projective_operator_upper"]), mp.mpf("1e-70"))

    def test_nonexact_dyadic_angle_has_no_free_fast_path(self):
        self.assertIsNone(exact_sequence(angle("pi", 1, 32)))

    def test_negative_exact_angle_uses_one_Tdagger(self):
        self.assertEqual(exact_sequence(angle("pi", -1, 4)), "t")

    def test_string_order_matches_actual_exact_ring_decoder(self):
        from pygridsynth.domega_unitary import DOmegaUnitary
        from pygridsynth.quantum_gate import HGate, TGate
        for s in ("HT", "TH", "SHTXW"):
            expected = DOmegaUnitary.from_gates(s).to_complex_matrix
            decoded = sequence_matrix(s)
            for i in range(2):
                for j in range(2):
                    real = bounds(decoded[i][j].real)
                    imag = bounds(decoded[i][j].imag)
                    self.assertLess(abs(expected[i,j] - mp.mpc(mid(real),mid(imag))),mp.mpf("1e-70"))
        self.assertGreater(mp.norm(HGate(0).matrix*TGate(0).matrix-TGate(0).matrix*HGate(0).matrix),0.1)

    def test_actual_synthesizer_identity_fixture(self):
        from pygridsynth.config import GridsynthConfig
        from pygridsynth.gridsynth import gridsynth_gates
        gates = gridsynth_gates(mp.mpf(0), mp.mpf("1e-8"), cfg=GridsynthConfig(dps=80,seed=0,up_to_phase=True))
        self.assertEqual(gates, "")

    def test_wrong_signed_rotation_fails_error_guard(self):
        error = error_guard("T", angle("pi", -1, 4))
        self.assertGreater(mp.mpf(error["projective_operator_upper"]),mp.mpf("0.5"))

    def test_unsupported_gate_string_rejected(self):
        with self.assertRaises(ValueError):
            sequence_matrix("Q")

    def test_pai_signed_channel_semantics_and_normalization(self):
        for numerator in (31,-31,290):
            t = mp.mpf(numerator)/100
            a = angle("rad", numerator,100)
            p = pai(a)
            gs = [mid(b) for b in p["g"]]
            weighted = mp.zeros(4)
            for g,k in zip(gs,p["notch_indices"]):
                weighted += g*channel(rz(k*mp.pi/4))
            self.assertLess(mp.norm(weighted-channel(rz(t))),mp.mpf("1e-65"))
            self.assertLess(abs(sum(gs)-1),mp.mpf("1e-65"))
            self.assertLess(abs(sum(mid(b) for b in p["p"])-1),mp.mpf("1e-65"))
            self.assertGreaterEqual(mp.mpf(p["gamma"]["hi"]),1)

    def test_exact_notch_is_one_branch(self):
        p = pai(angle("pi",3,4))
        self.assertEqual([b["lo"] for b in p["g"]],["1","0","0"])
        self.assertEqual(p["gamma"]["lo"],"1")

    def test_controlled_lowering_signed_joint_angles(self):
        t = mp.mpf("0.31")
        # I tensor Z and Z tensor Z diagonal factors, no circuit construction.
        q1 = mp.diag([mp.exp(-mp.j*t*s/4) for s in (1,-1,1,-1)])
        q2 = mp.diag([mp.exp(mp.j*t*s/4) for s in (1,-1,-1,1)])
        desired = mp.diag([1,1,mp.exp(-mp.j*t/2),mp.exp(mp.j*t/2)])
        self.assertLess(mp.norm(q1*q2-desired),mp.mpf("1e-70"))
        self.assertGreater(mp.norm(q1*q2**-1-desired),mp.mpf("0.1"))

    def test_system_global_phase_cannot_be_discarded_before_control(self):
        t = mp.pi/4
        full = mp.diag([1,1,mp.exp(-mp.j*t/2),mp.exp(mp.j*t/2)])
        incorrectly_controlled_T = mp.diag([1,1,1,mp.exp(mp.j*t)])
        self.assertGreater(mp.norm(full-incorrectly_controlled_T),0.1)
        # Scalar phases introduced on full-joint factors are global only.
        self.assertLess(mp.norm(channel(rz(t))-channel(mp.exp(mp.j*0.2)*rz(t))),mp.mpf("1e-70"))

    def test_joint_moment_is_product_not_sum(self):
        p = pai(angle("rad",31,100))
        q = pai(angle("rad",-17,100))
        result = economics([p,q],[[1,2,1],[2,1,2]],30)
        gamma1, gamma2 = mid(p["gamma"]),mid(q["gamma"])
        self.assertLess(abs(mid(result["weight_second_moment"])-gamma1**2*gamma2**2),mp.mpf("1e-60"))
        self.assertGreater(abs(mid(result["weight_second_moment"])-(gamma1**2+gamma2**2)),0.1)

    def test_equal_branch_cost_incur_sampling_penalty(self):
        p = pai(angle("rad",31,100))
        result = economics([p],[[10,10,10]],10)
        self.assertEqual(result["classification"],"NO_STRICT_TRADEOFF")

    def test_strict_gain_fixture_uses_cost_times_second_moment(self):
        p = pai(angle("rad",31,100))
        result = economics([p],[[0,1,0]],30)
        self.assertEqual(result["classification"],"STRICT_TRADEOFF")
        explicit = mid(p["gamma"])**2*mid(p["p"][1])/30
        self.assertLess(abs(mid(result["J"])-explicit),mp.mpf("1e-60"))

    def test_interval_over_one_is_inconclusive(self):
        p = {"gamma":{"lo":"1","hi":"1.0001"},
             "p":[{"lo":"0.9999","hi":"1"},{"lo":"0","hi":"0"},{"lo":"0","hi":"0"}]}
        self.assertEqual(economics([p],[[1,0,0]],1)["classification"],"NUMERIC_INCONCLUSIVE")

    def test_zero_cost_baseline_is_not_a_positive_result(self):
        self.assertEqual(economics([],[],0)["classification"],"ZERO_COST_BASELINE_NO_STRICT_GAIN")

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValueError):
            economics([pai(angle("rad",31,100))],[[-1,1,1]],10)

    def test_directed_serialization_encloses_endpoints(self):
        x = mp.iv.mpf(1)/3
        b = bounds(x)
        rebuilt = interval(b)
        self.assertTrue(rebuilt.a <= x.a and rebuilt.b >= x.b)


class LaunchAndBudget(unittest.TestCase):
    def setUp(self):
        configure()
        self.contract = json.loads((runner.PREPARATION/"contract_v1.json").read_text())
        self.auth = {"source_commit":"a"*40,"status":"APPROVED_FOR_ONE_SP05_RUN",
                     "science_execution_authorized":True,"runs":1,"retries":0,
                     "mandatory_STOP":True,"contract_sha256":"contract",
                     "explicit_execution_instruction":"source aaaa under fixed contract; execute once and stop"}

    def binding(self,**overrides):
        args = dict(auth=self.auth,contract_hash="contract",head="b"*40,parents=["a"*40],
                    changed=["auth.json"],dirty=False,allowed={"auth.json","receipt.md"})
        args.update(overrides)
        validate_binding(**args)

    def test_authorization_only_child_valid(self):
        self.binding()

    def test_pending_auth_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(auth={**self.auth,"science_execution_authorized":False})

    def test_source_equals_authorization_HEAD_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(head="a"*40)

    def test_changed_science_source_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(changed=["auth.json","science.py"])

    def test_merge_or_other_parent_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(parents=["a"*40,"c"*40])

    def test_dirty_worktree_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(dirty=True)

    def test_missing_explicit_instruction_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(auth={**self.auth,"explicit_execution_instruction":None})

    def test_changed_contract_rejected(self):
        with self.assertRaises(PermissionError):
            self.binding(contract_hash="changed")

    def test_one_shot_marker_cannot_be_consumed_twice(self):
        with tempfile.TemporaryDirectory() as d:
            marker = consume_marker(d,{"test_fixture":True})
            with self.assertRaises(FileExistsError):
                consume_marker(d,{"test_fixture":True})
            self.assertTrue(marker.exists())

    def test_registered_plan_is_small_and_has_no_synthesis(self):
        with patch.object(runner,"synthesize",side_effect=AssertionError("must not synthesize")):
            self.assertLessEqual(len(runner.requests(self.contract)),32)
        self.assertEqual(len(self.contract["targets"]),8)
        self.assertEqual(self.contract["catalogue"]["count"],1)

    def test_key_cap_rejects_before_run(self):
        self.contract["caps"]["synthesis_keys"] = 1
        with self.assertRaises(ValueError):
            runner.requests(self.contract)

    def test_runtime_source_mismatch_rejected(self):
        with patch.object(runner.platform,"python_version",return_value="0"):
            with self.assertRaises(RuntimeError):
                runner.verify_environment({"python":"3.10.12"})

    def test_missing_semantic_pass_rejected(self):
        with self.assertRaises(PermissionError):
            runner.verify_preparation({"focused_tests_passed":False})

    def test_changed_preparation_identity_rejected(self):
        with self.assertRaises(PermissionError):
            runner.verify_preparation({"focused_tests_passed":True,
                "source_sha256":{"scripts/tracks/algorithm_codesign/run_sp05_synthesis_economics.py":"changed"}})

    def test_worker_call_is_bounded_identity_fixture_only(self):
        import time
        row = runner.bounded_key(angle("pi",0),self.contract,time.monotonic(),0)
        self.assertEqual(row["T_count"],0)
        self.assertTrue(row["error_pass"])

    def test_wall_cap_stops_sleeping_worker_without_retry(self):
        import time
        def slow(*args):
            time.sleep(1)
        self.contract["caps"]["per_key_wall_seconds"] = 0.01
        with patch.object(runner,"_worker",slow):
            with self.assertRaises(TimeoutError):
                runner.bounded_key(angle("pi",0),self.contract,time.monotonic(),0)


if __name__ == "__main__":
    unittest.main()
