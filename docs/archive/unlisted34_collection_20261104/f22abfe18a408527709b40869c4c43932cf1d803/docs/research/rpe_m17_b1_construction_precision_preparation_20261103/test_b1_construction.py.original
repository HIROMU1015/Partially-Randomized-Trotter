"""Artificial matrices/scalars only. Real files are checked as bytes/JSON/headers."""
from concurrent.futures import Future, ProcessPoolExecutor
import contextlib
import copy
import io
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import b1_adapter as b
import run_b1_construction as runner

p = b.precision


def toy():
    dist = {"orders": [0, 2], "order_probabilities": [.75, .25], "finite_taylor_order": 2,
            "dimensionless_step_time": .125, "exact_finite_distribution": 1.25}
    scalars = b.replay_scalars(dist, 2)
    arrays = {"A_probability": np.array([[.2, .1j], [-.1j, -.3]], complex),
              "A_coefficient": np.array([[.2, .1j], [-.1j, -.3]], complex),
              "forward": np.array([[1., .2j], [0., 1.]], complex),
              "reverse": np.array([[1., 0.], [.3, 1.]], complex)}
    replay = b.float_replay(arrays, scalars)
    arrays.update({k: v.copy() for k, v in replay.items() if k != "reference_polynomial"})
    return dist, scalars, arrays, np.array([1., 0.], complex), replay


def job(path="law", bits=96, mode="b1a"):
    _, scalars, arrays, psi, replay = toy()
    return {"mode": mode, "path": path, "bits": bits, "q": 4, "psi": psi,
            "forward": arrays["forward"], "reverse": arrays["reverse"],
            "saved_tail": arrays[runner.TAIL_KEYS[path]], "replay_tail": replay[runner.TAIL_KEYS[path]],
            "A": arrays["A_probability" if path == "law" else "A_coefficient"], "scalars": scalars}


class InlinePool:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def submit(self, fn, value):
        future = Future()
        try:
            future.set_result(fn(value))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class B1Tests(unittest.TestCase):
    def test_saved_and_replay_scalar_provenance(self):
        dist, s, *_ = toy()
        self.assertEqual(s["probabilities"], dist["order_probabilities"])
        self.assertEqual(s["theta64"], [math.atan(.125/(n+1)) for n in [0, 2]])
        trace = b.scalar_trace(s)
        self.assertEqual(float.fromhex(trace["binary64_hex"]["theta64"][0]), s["theta64"][0])
        self.assertIn("new SOURCE replay", trace["provenance"]["theta64_cos64_sin64_logB64_normalizer64"])

    def test_binary64_exact_not_decimal_string(self):
        ctx = p.scalar_context()
        self.assertEqual(p.exact_real(ctx, .1), ctx.mpf(.1.as_integer_ratio()[0])/.1.as_integer_ratio()[1])
        self.assertNotEqual(p.exact_real(ctx, .1), ctx.mpf("0.1"))

    def test_component_probability_renormalization_and_signs(self):
        def component(identifier, prob, sign, coefficient):
            return SimpleNamespace(component_id=identifier, probability=prob, coefficient_sign=sign,
                                   coefficient=coefficient, basis_id="toy", diagonal_pauli_support=(0,))
        comps = [component("a", .2, 1, .2), component("b", .3, -1, -.3)]
        a_prob, a_coef, total = b.task_a.original_runner.grouped_averages(comps, comps, .5, [0, 1], lambda _: np.eye(2))
        self.assertEqual(total, .5)
        np.testing.assert_allclose(a_prob, np.diag([-.2, .2]), atol=1e-15)
        np.testing.assert_allclose(a_coef, a_prob, atol=1e-15)

    def test_event_probabilities_are_not_renormalized_again(self):
        dist, *_ = toy()
        dist["order_probabilities"] = [.75, .2500000000000001]
        s = b.replay_scalars(dist, 2)
        self.assertEqual(s["probabilities"], dist["order_probabilities"])

    def test_missing_or_invalid_scalars_stop(self):
        dist, *_ = toy()
        for key, value in [("orders", [0, 1]), ("order_probabilities", [.8, .8]),
                           ("order_probabilities", [-.1, 1.1]), ("dimensionless_step_time", float("nan")),
                           ("exact_finite_distribution", 0.)]:
            wrong = copy.deepcopy(dist)
            wrong[key] = value
            with self.assertRaises(ValueError):
                b.replay_scalars(wrong, 2)
        del dist["dimensionless_step_time"]
        with self.assertRaises(KeyError):
            b.replay_scalars(dist, 2)

    def test_law_phase_rotation_and_inner_power(self):
        j = job()
        ctx = p.scalar_context()
        a = p.exact_matrix(ctx, j["A"])
        event, tail = b.construct(ctx, "law", a, j["scalars"])
        expected = ctx.zeros(2)
        for n, prob, theta in zip([0, 2], [.75, .25], j["scalars"]["theta64"]):
            angle = p.exact_real(ctx, theta)
            expected += p.exact_real(ctx, prob)*(-1)**(n//2)*(ctx.cos(angle)*ctx.eye(2)-ctx.j*ctx.sin(angle)*a)*(a**n)
        self.assertLess(ctx.norm(event-expected), ctx.mpf("1e-80"))
        self.assertLess(ctx.norm(tail-event*event), ctx.mpf("1e-80"))

    def test_reference_degree_K_plus_one_and_normalizer_once(self):
        j = job("reference")
        ctx = p.scalar_context()
        a = p.exact_matrix(ctx, j["A"])
        poly, tail = b.construct(ctx, "reference", a, j["scalars"])
        z = -ctx.j*p.exact_real(ctx, .125)*a
        expected = ctx.eye(2)+z+z*z/2+z*z*z/6
        self.assertLess(ctx.norm(poly-expected), ctx.mpf("1e-80"))
        self.assertLess(ctx.norm(tail-poly*poly/p.exact_real(ctx, 1.25)**2), ctx.mpf("1e-80"))

    def test_zero_time_identity_and_no_extra_phase(self):
        dist, _, arrays, _, _ = toy()
        dist.update(dimensionless_step_time=0., order_probabilities=[1., 0.], exact_finite_distribution=1.)
        scalars = b.replay_scalars(dist, 3)
        ctx = p.scalar_context()
        for path in runner.PATHS:
            one, tail = b.construct(ctx, path, p.exact_matrix(ctx, arrays["A_probability"]), scalars)
            self.assertEqual(one, ctx.eye(2))
            self.assertEqual(tail, ctx.eye(2))

    def test_float_replay_uses_original_independent_functions(self):
        _, s, arrays, _, replay = toy()
        with patch.object(b.task_a.original_runner, "literal_law_mean", wraps=b.task_a.original_runner.literal_law_mean) as called:
            got = b.float_replay(arrays, s)
        called.assert_called_once()
        np.testing.assert_array_equal(got["reference_tail"], replay["reference_tail"])

    def test_noncommuting_order_outer_power_state_not_renormalized(self):
        j = job()
        j["psi"] *= 2
        result = b.evaluate_job(j)
        ctx = p.scalar_context()
        tail = b.restore_matrix(ctx, result["operators"]["tail"])
        f, r = (p.exact_matrix(ctx, j[k]) for k in ["forward", "reverse"])
        psi = p.exact_matrix(ctx, j["psi"], vector=True)
        value = ((r*tail)*f)**4*psi
        expected = sum(ctx.conj(psi[i])*value[i] for i in range(2))
        self.assertLess(abs(p.number(ctx, result["mean"])-expected), ctx.mpf("1e-25"))
        wrong = (f*tail*r)**4*psi
        self.assertGreater(abs(expected-sum(ctx.conj(psi[i])*wrong[i] for i in range(2))), ctx.mpf("1e-4"))

    def test_float_boundary_and_HP_are_distinct(self):
        j = job(bits=160)
        result = b.evaluate_job(j)
        ctx = p.scalar_context()
        hp = b.restore_matrix(ctx, result["operators"]["tail"])
        self.assertGreater(ctx.norm(hp-p.exact_matrix(ctx, j["replay_tail"])), 0)

    def test_operator_serialization_retains_more_than_binary64(self):
        ctx = p.scalar_context()
        m = ctx.matrix([[ctx.mpf(1)+ctx.mpf(2)**-100, ctx.j/7]])
        payload = json.loads(json.dumps(b.matrix_payload(ctx, m)))
        self.assertEqual(b.restore_matrix(ctx, payload), m)
        self.assertNotEqual(b.restore_matrix(ctx, payload)[0, 0], ctx.mpf(float(m[0, 0].real)))

    def test_invalid_matrix_and_unsupported_modes_stop(self):
        for mutate in [lambda j: j.update(mode="b1b"), lambda j: j.update(q=0),
                       lambda j: j.update(A=np.array([[float("nan")]], complex)),
                       lambda j: j.update(forward=np.eye(3, dtype=complex))]:
            j = job()
            mutate(j)
            with self.assertRaises(ValueError):
                b.evaluate_job(j)

    def test_convergence_is_not_bound_and_detects_tail_instability(self):
        previous = {path: b.evaluate_job(job(path, 64)) for path in runner.PATHS}
        current = {path: b.evaluate_job(job(path, 96)) for path in runner.PATHS}
        result = b.convergence(previous, current, 4)
        self.assertTrue(result["converged"])
        self.assertFalse(result["is_error_bound"])
        current["law"]["operators"]["tail"]["entries"][0][0] = "1.5"
        self.assertFalse(b.convergence(previous, current, 4)["converged"])

    def test_threshold_diagnostic_does_not_certify_RPE(self):
        pair = {p: {"mean": {"real": "0", "imag": "0"}} for p in runner.PATHS}
        self.assertEqual(b.precision.gap_relation(pair)["relation_to_original_atol"], "BELOW")
        self.assertIsNone(runner.STATUS["u_num_bound"])
        self.assertFalse(runner.STATUS["RPE_tolerance_certified"])
        self.assertEqual(runner.STATUS["scientific_status"], "UNDETERMINED")

    def test_default_checks_only_bytes_JSON_headers(self):
        with patch.object(np, "load", side_effect=AssertionError("Real arrays forbidden")), patch.object(b, "replay_scalars", side_effect=AssertionError("Real scalar replay forbidden")), patch.object(runner, "execute", side_effect=AssertionError("No default execution")), contextlib.redirect_stdout(io.StringIO()) as printed:
            runner.main([])
        result = json.loads(printed.getvalue())
        self.assertEqual(result["new_real_numeric_calls"], 0)
        self.assertEqual(len(result["inputs"]), 5)

    def test_run_without_output_stops_before_execution(self):
        with patch.object(runner, "execute") as called, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(["--run"])
        called.assert_not_called()

    def test_spawn_parallel_jobs_match_serial(self):
        jobs = [job(path, bits) for bits in [64, 96] for path in runner.PATHS]
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            parallel = list(pool.map(b.evaluate_job, jobs))
        self.assertEqual([x["mean"] for x in parallel], [b.evaluate_job(j)["mean"] for j in jobs])

    def test_precision_extension_is_bounded(self):
        _, s, arrays, psi, replay = toy()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            record = {"high_precision_jobs": []}
            with InlinePool() as pool, zipfile.ZipFile(output/"operators.zip", "x") as archive, patch.object(b, "convergence", return_value={"converged": False, "is_error_bound": False}):
                stage = runner.run_ladder("b1a", arrays, psi, s, replay, (64, 96, 128), pool, record, output, archive)
            self.assertEqual(stage["precision_bits_used"], [64, 96, 128])
            self.assertEqual(len(record["high_precision_jobs"]), 6)
            self.assertFalse(stage["convergence"]["converged"])

    def integration(self, mismatch=False, fail=False, unconverged=False):
        dist, scalars, arrays, psi, replay = toy()
        if mismatch:
            arrays["law_tail_mean"][0, 0] += 1e-7
        prior_pair = {path: b.evaluate_job(job(path, 96, "replay_tail_baseline")) for path in runner.PATHS}
        prior = {"best_pair": runner.compact_pair(prior_pair), "convergence": {"converged": True}}
        meta = {"inputs": [], "precision_bits": (64, 96, 128), "task_a": {"fixed_factors": prior},
                "old_result": {"fixed_literals": {"synthetic": True}, "context": {"distribution": dist}}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"fresh"
            with contextlib.ExitStack() as stack:
                for obj, name, kwargs in [(runner, "source_metadata", {"return_value": {"commit": "synthetic"}}),
                    (runner, "fixed_metadata", {"return_value": meta}),
                    (runner, "load_arrays", {"side_effect": ValueError("synthetic failure") if fail else None, "return_value": (arrays, psi)}),
                    (runner, "ProcessPoolExecutor", {"new": InlinePool}), (runner, "Q", {"new": 4}), (runner, "R", {"new": 2})]:
                    stack.enter_context(patch.object(obj, name, **kwargs))
                if unconverged:
                    stack.enter_context(patch.object(b, "convergence", return_value={"converged": False, "is_error_bound": False}))
                if fail:
                    with self.assertRaises(ValueError):
                        runner.execute(output)
                else:
                    runner.execute(output)
                    with self.assertRaises(FileExistsError):
                        runner.execute(output)
            result = json.loads((output/"RESULT.json").read_text())
            if not fail:
                with zipfile.ZipFile(output/"high_precision_operators.zip") as archive:
                    self.assertEqual(len(archive.namelist()), len(result["high_precision_jobs"]))
            return result

    def test_end_to_end_reuses_saved_baseline(self):
        result = self.integration()
        self.assertEqual(result["execution_status"], "COMPLETED_MANDATORY_STOP")
        self.assertEqual(len(result["high_precision_jobs"]), 4)
        self.assertNotIn("replay_tail_baseline", result)
        self.assertIsNone(result["u_num_bound"])

    def test_end_to_end_conditional_replay_baseline(self):
        result = self.integration(mismatch=True)
        self.assertEqual(len(result["high_precision_jobs"]), 6)
        self.assertIn("replay_tail_baseline", result)
        self.assertIn("not original saved", result["baseline_origin"])
        self.assertEqual(result["baseline_recomputed_paths"], ["law"])

    def test_partial_failure_is_retained(self):
        result = self.integration(fail=True)
        self.assertEqual(result["execution_status"], "INTERRUPTED_OR_INVALID_MANDATORY_STOP")
        self.assertEqual(result["error"]["message"], "synthetic failure")
        self.assertTrue(result["mandatory_STOP"])

    def test_unconverged_replay_baseline_skips_b1a(self):
        result = self.integration(mismatch=True, unconverged=True)
        self.assertEqual(len(result["high_precision_jobs"]), 3)
        self.assertNotIn("b1a", result)
        self.assertIn("B-1a skipped", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
