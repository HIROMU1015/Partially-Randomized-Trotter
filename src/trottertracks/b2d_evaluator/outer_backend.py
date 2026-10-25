"""Dormant outer nodes; reuse DenseBackend validation, exp cache and signals.

No instantiation in artificial preparation tests. The fixed outer execution
context binds the new graph/scope; the legacy 36-row scope remains separate.
"""
import math
from dataclasses import asdict
from fractions import Fraction
from .dense_backend import DenseBackend
from .errors import Stage1Failure
from .outer import INNER, normalization_product
from trottertracks.b2d_adapter._published import symmetric_s2_sequence


class OuterDenseBackend(DenseBackend):
    def _check_rte_config(self, spec, binding, config):
        expected = (self.s.context.tail_id, self.s.hamiltonian_id,
                    self.s.input_data_sha256, self.s.precision, self.s.finite_bit_policy)
        if binding != expected:
            raise Stage1Failure("RTE_INPUT_IDENTITY_MISMATCH")
        tail = self.i.symbolic_tail
        actual = (config.tail_id, config.tail_hash, config.lambda_r, config.evolution_time,
                  config.rte_steps, config.finite_taylor_order, config.truncation_tolerance, config.seed,
                  config.taylor_distribution)
        expected_config = (tail.tail_id, tail.tail_hash, tail.lambda_r, float(spec.duration),
                           spec.rte_steps, spec.cutoff, spec.truncation_tolerance, spec.seed,
                           "rte_even_taylor_paired")
        if actual != expected_config:
            raise Stage1Failure("RTE_SIGNED_CONFIG_MISMATCH")

    def _outer_signals(self, physical, outer, ideal, corrected, raw):
        records = self._signals((physical, ideal, corrected, raw))
        value = self.execution.science_call("signal.expectation",
            lambda: self.signal(outer, self.i.state))
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
        records["exact_outer"] = dict(X=value.real, Y=value.imag, radius=abs(value),
            principal_phase=None, phase_status="NOT_REQUESTED")
        target = complex(records["physical"]["X"], records["physical"]["Y"])
        for name in ("exact_outer", "ideal_tail", "corrected", "raw"):
            signal = complex(records[name]["X"], records[name]["Y"])
            records[name + "_physical_signal_error"] = abs(signal - target)
        records["D_approximation_signal_error"] = abs(
            complex(records["ideal_tail"]["X"], records["ideal_tail"]["Y"]) - value)
        return records

    def _check_error(self, actual, reference, code):
        error = float(self.np.linalg.norm(actual - reference, ord=2))
        if not math.isfinite(error):
            raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
        if error > self.s.numerical_atol:
            raise Stage1Failure(code)
        return error

    def _check_word(self, actual, mathematical_factors, code):
        reference = self.identity.copy()
        for label, time in mathematical_factors:
            reference = reference @ self._exp(label, time)
        return dict(operator_error=self._check_error(actual, reference, code), phase_fit=False)

    def perform(self, kind, args, params):
        if kind.startswith("outer_") and not self.validated:
            raise ValueError("Input correctness must precede outer scientific operations")
        if kind == "outer_rte_moments":
            spec = params[0]
            config, distribution = self.execution.science_call("rte.config",
                lambda: spec.make_config(self.i.symbolic_tail, self.make_rte_config), identity=str(spec))
            if len(params) == 2:
                self._check_rte_config(spec, params[1], config)
                if not hasattr(self, "outer_configs"):
                    self.outer_configs = {}
                self.outer_configs[params] = (config, distribution)
            return self.execution.science_call("rte.moments",
                lambda: self.finite_rte_operator_moments(self.i.normalized_tail, config), identity=str(spec))
        if kind == "outer_tail_bound_check":
            value = float(self.np.linalg.norm(self.i.normalized_tail, ord=2))
            if not math.isfinite(value) or value > 1 + self.s.numerical_atol:
                raise Stage1Failure("RTE_TAIL_BOUND_ASSUMPTION_FAILURE")
            return dict(normalized_tail_norm=value, theoretical_norm_bound=1,
                floating_point_certificate=False, assumption="convex_sum_of_signed_involutions")
        if kind == "outer_rte_check":
            config, distribution = self.outer_configs[params]
            self._check_rte_config(*params, config)
            moments, = args
            expected = math.exp(config.rte_steps * math.log(config.distribution_normalization))
            n, attenuation = moments.normalization_product, moments.attenuation_factor
            if not all(math.isfinite(v) and v > 0 for v in (n, attenuation, expected)):
                raise Stage1Failure("INCONCLUSIVE_NORMALIZATION_RANGE")
            if (not math.isclose(n, expected, rel_tol=0, abs_tol=self.s.numerical_atol)
                    or not math.isclose(n * attenuation, 1, rel_tol=0, abs_tol=self.s.numerical_atol)):
                raise Stage1Failure("NORMALIZATION_CORRECTNESS_FAILURE")
            return dict(config=asdict(config), event_distribution=asdict(distribution),
                normalization_product=n, attenuation=attenuation,
                event_phase="existing_paired_even_order_phase_and_signed_rotation",
                negative_time_policy="signed_finite_polynomial_not_inverse",
                sampling="NONE_ANALYTIC_MOMENTS")
        if kind == "outer_rte_records":
            return [dict(append_occurrence=i, **record) for i, record in enumerate(args)]
        if kind == "outer_taylor_bound":
            from trotterlib.rte import rte_occurrence_truncation_from_config
            config, _ = self.outer_configs[params]
            record = self.execution.science_call("rte.truncation",
                lambda: rte_occurrence_truncation_from_config(str(params[0]), config), identity=str(params[0]))
            return asdict(record)
        if kind == "outer_taylor_total":
            from trotterlib.rte import compose_truncation_residual_bounds
            value = self.execution.science_call("bound.compose", lambda: compose_truncation_residual_bounds(
                [(b["step_truncation_residual_bound"], b["rte_steps"], 1) for b in args]))
            if not math.isfinite(value):
                raise Stage1Failure("INCONCLUSIVE_TAYLOR_BOUND_RANGE")
            return dict(occurrences=[dict(record, append_occurrence=i) for i, record in enumerate(args)], occurrence_count=len(args),
                corrected_vs_ideal_tail_operator_bound=value, complex_signal_bound=value,
                floating_point_uncertainty=None, certificate="NOMINAL_ANALYTIC_ONLY")
        if kind == "outer_bias_qualification":
            from .outer_rte_execution import bias_qualification
            return bias_qualification(args[0])
        if kind == "outer_normalization":
            try:
                return normalization_product(args)
            except ValueError as error:
                raise Stage1Failure("INCONCLUSIVE_NORMALIZATION_RANGE") from error
        if kind == "outer_order_check":
            return self._check_word(args[0], params[0], "OUTER_ORDER_CORRECTNESS_FAILURE")
        if kind == "outer_rescale_check":
            return self._check_error(args[0], args[1], "NORMALIZATION_CORRECTNESS_FAILURE")
        if kind == "outer_signals":
            return self._outer_signals(*args)
        if kind == "outer_d_check":
            stage, = params
            blocks = self.s.context.block_ids
            # Independent mathematical expansion, not adapter append expansion.
            word = tuple((block, stage.tau * Fraction(w) / (1 if block == blocks[-1] else 2))
                for w in symmetric_s2_sequence(INNER[stage.inner_arm]())
                for block in blocks[:-1] + blocks[::-1])
            return self._check_word(args[0], word * stage.repetitions, "INNER_PF_CORRECTNESS_FAILURE")
        if kind == "outer_diagnostics":
            np, d, r = self.np, self.d_matrix, self.r_matrix
            comm = d @ r - r @ d
            def norm(x):
                value = float(np.linalg.norm(x, ord=2))
                if not math.isfinite(value):
                    raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
                return value
            dn, rn = norm(d), norm(r)
            return dict(D_norm=dn, R_norm=rn, R_D_norm_ratio=rn / dn if dn else None,
                DD_R_norm=norm(d @ comm - comm @ d), RR_D_norm=norm(r @ (-comm) - (-comm) @ r),
                weak_perturbation_status="DESCRIPTIVE_ONLY_NO_ADOPTED_THRESHOLD")
        return super().perform(kind, args, params)
