"""Dormant outer nodes; reuse DenseBackend validation, exp cache and signals.

No instantiation in preparation. A future execution context must bind the NEW
graph/scope: the old Stage-1 launcher/scope cannot run this graph.
"""
import math
from fractions import Fraction
from .dense_backend import DenseBackend
from .errors import Stage1Failure
from .outer import INNER, normalization_product
from trottertracks.b2d_adapter._published import symmetric_s2_sequence


class OuterDenseBackend(DenseBackend):
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
            spec, = params
            config, _distribution = self.execution.science_call("rte.config",
                lambda: spec.make_config(self.i.symbolic_tail, self.make_rte_config), identity=str(spec))
            return self.execution.science_call("rte.moments",
                lambda: self.finite_rte_operator_moments(self.i.normalized_tail, config), identity=str(spec))
        if kind == "outer_normalization":
            try:
                return normalization_product(args)
            except ValueError as error:
                raise Stage1Failure("INCONCLUSIVE_NORMALIZATION_RANGE") from error
        if kind == "outer_order_check":
            return self._check_word(args[0], params[0], "OUTER_ORDER_CORRECTNESS_FAILURE")
        if kind == "outer_rescale_check":
            return self._check_error(args[0], args[1], "NORMALIZATION_CORRECTNESS_FAILURE")
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
