"""Dormant small-system numerical connection. Never instantiated in preparation.

The guarded input_package loader supplies matrices/state in ONE documented
sector order. No PF circuit renderer is included. Scientific imports are lazy.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from .errors import Stage1Failure


@dataclass(frozen=True)
class PreparedDenseInputs:
    preparation_id: str
    tail_id: str
    hamiltonian_id: str
    sector_id: str
    state_id: str
    blocks: dict
    physical_hamiltonian: object
    normalized_tail: object
    symbolic_tail: object
    state: object


class DenseBackend:
    is_mock = False

    def __init__(self, settings, inputs: PreparedDenseInputs, *, execution=None):
        # evaluate() checks explicit approval before constructing a backend.
        # Direct construction is also a scientific action requiring approval;
        # these API guards do not replace human authorization or ledger checks.
        settings.validate()
        from .preflight import require_claimed_run
        require_claimed_run(execution).check_settings(settings)
        import numpy as np
        from scipy.linalg import expm
        from trotterlib.rte import make_rte_config, finite_rte_operator_moments
        from trotterlib.finite_rte_signal_validation import (
            _signal, _phase_distance, _array_fingerprint,
        )
        self.s, self.i, self.np, self.expm = settings, inputs, np, expm
        self.make_rte_config = make_rte_config
        self.finite_rte_operator_moments = finite_rte_operator_moments
        self.signal, self.phase_distance = _signal, _phase_distance
        self.array_fingerprint = _array_fingerprint
        self.exp_cache = {}
        self.validated = False
        self.execution = execution

    def _validate(self):
        np, s, i = self.np, self.s, self.i
        actual = (i.preparation_id, i.tail_id, i.hamiltonian_id, i.sector_id, i.state_id)
        expected = (s.context.preparation_id, s.context.tail_id, s.hamiltonian_id, s.sector_id, s.state_id)
        if actual != expected or tuple(i.blocks) != s.context.block_ids:
            raise Stage1Failure("INPUT_IDENTITY_ORDER_MISMATCH")
        if i.symbolic_tail.tail_hash != s.context.tail_id or i.symbolic_tail.lambda_r <= 0:
            raise Stage1Failure("INCONCLUSIVE_RESIDUAL_IDENTITY")
        arrays = {**{key: np.asarray(value) for key, value in i.blocks.items()},
                  "physical": np.asarray(i.physical_hamiltonian),
                  "tail": np.asarray(i.normalized_tail), "state": np.asarray(i.state)}
        digest = hashlib.sha256(json.dumps(
            {key: self.array_fingerprint(value) for key, value in arrays.items()},
            sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if digest != s.input_data_sha256:
            raise Stage1Failure("INPUT_BYTES_MISMATCH")
        dimension = arrays["state"].size
        if arrays["state"].shape != (dimension,) or dimension < 2:
            raise Stage1Failure("SECTOR_STATE_DIMENSION_MISMATCH")
        for label, matrix in arrays.items():
            if not np.isfinite(matrix).all() or matrix.dtype != np.complex128:
                raise Stage1Failure("NUMERICAL_INPUT_PRECISION_FAILURE")
            if label != "state":
                if matrix.shape != (dimension, dimension):
                    raise Stage1Failure("SECTOR_BASIS_DIMENSION_MISMATCH")
                if np.linalg.norm(matrix - matrix.conj().T, ord=2) > s.numerical_atol:
                    raise Stage1Failure("NUMERICAL_HERMITICITY_FAILURE")
        if abs(np.vdot(i.state, i.state) - 1) > s.numerical_atol:
            raise Stage1Failure("STATE_NORMALIZATION_FAILURE")
        self.d_matrix = sum(i.blocks.values())
        self.r_matrix = i.symbolic_tail.lambda_r * i.normalized_tail
        self.scalar = float(sum(t.coefficient for t in s.context.scalar_terms))
        self.identity = np.eye(dimension, dtype=np.complex128)
        identity_error = np.linalg.norm(
            i.physical_hamiltonian - self.d_matrix - self.r_matrix - self.scalar * self.identity, ord=2
        )
        if not math.isfinite(identity_error) or identity_error > s.numerical_atol:
            raise Stage1Failure("HAMILTONIAN_IDENTITY_FAILURE")
        d = tuple(i.blocks.values())
        d_commutators = [float(np.linalg.norm(a @ b - b @ a, ord=2))
                         for j, a in enumerate(d) for b in d[j + 1:]]
        dr_commutator = float(np.linalg.norm(self.d_matrix @ self.r_matrix - self.r_matrix @ self.d_matrix, ord=2))
        if not all(math.isfinite(v) for v in d_commutators+[dr_commutator]):
            raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
        if not d_commutators or max(d_commutators) <= s.numerical_atol or dr_commutator <= s.numerical_atol:
            raise Stage1Failure("INCONCLUSIVE_NONCOMMUTATIVITY")
        self.validated = True
        return {"input_sha256": digest, "H_identity_error": float(identity_error),
                "D_commutator_norms": d_commutators, "DR_commutator_norm": dr_commutator,
                "sector_id": i.sector_id, "state_id": i.state_id}

    def _exp(self, label, time):
        key = (label, time)
        if key not in self.exp_cache:
            matrix = {"exact_D": self.d_matrix, "exact_R": self.r_matrix,
                      "exact_physical": self.i.physical_hamiltonian}.get(label)
            if matrix is None:
                matrix = self.i.blocks[label]
            from .scope import exp_key
            self.exp_cache[key] = self.execution.science_call("backend.expm",
                lambda: self.expm(-1j * float(time) * matrix),
                identity=exp_key(self.execution.scope.graph, label, time))
        return self.exp_cache[key]

    def _signals(self, matrices):
        np, s = self.np, self.s
        names = ("physical", "ideal_tail", "corrected", "raw")
        def expectation(matrix):
            execution = getattr(self, "execution", None)
            return execution.science_call("signal.expectation", lambda: self.signal(matrix,self.i.state)) if execution else self.signal(matrix,self.i.state)
        values = dict(zip(names, (expectation(m) for m in matrices)))
        if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values.values()):
            raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
        def record(value):
            if not s.phase_diagnostics:
                return {"X": value.real, "Y": value.imag, "radius": abs(value),
                        "principal_phase": None, "phase_status": "NOT_REQUESTED"}
            angle = float(np.angle(value))
            eligible = abs(value) > s.radius_floor and np.pi - abs(angle) > s.branch_margin
            return {"X": value.real, "Y": value.imag, "radius": abs(value),
                    "principal_phase": angle if eligible else None,
                    "phase_status": "ELIGIBLE_DIAGNOSTIC" if eligible else "INCONCLUSIVE_RADIUS_OR_BRANCH"}
        records = {name: record(value) for name, value in values.items()}
        records["corrected_physical_signal_error"] = abs(values["corrected"] - values["physical"])
        records["finite_RTE_signal_error"] = abs(values["corrected"] - values["ideal_tail"])
        eligible = all(records[x]["principal_phase"] is not None for x in ("physical", "corrected"))
        records["corrected_physical_phase_distance"] = (
            self.phase_distance(values["corrected"], values["physical"]) if eligible else None
        )
        return records

    def _a4_reference(self, actual, h, n_d):
        # Independent mathematical composition: no adapter block expansion.
        # This DOES NOT assert equivalence to the old validator's append order.
        from trottertracks.b2d_adapter.adapter import published_math_sequences
        np, tau = self.np, h / n_d
        def s2(weight):
            factors = [self._exp(b, tau * weight / 2) for b in self.s.context.block_ids]
            value = self.identity.copy()
            for factor in factors + factors[::-1]:
                value = value @ factor
            return value
        kernel, processor = published_math_sequences()
        def math_word(weights):
            value = self.identity.copy()
            for weight in weights:
                value = value @ s2(weight)
            return value
        p, k = math_word(processor), math_word(kernel)
        expected = p @ np.linalg.matrix_power(k, n_d) @ p.conj().T
        error = float(np.linalg.norm(actual - expected, ord=2))
        inverse_error = float(np.linalg.norm(p.conj().T @ p - self.identity, ord=2))
        if not all(math.isfinite(v) for v in (error,inverse_error)):
            raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
        if max(error, inverse_error) > self.s.numerical_atol:
            raise Stage1Failure("A4_ORDER_OR_INVERSE_CORRECTNESS_FAILURE")
        convention = self.execution.science_call("published.convention",
            lambda:self._published_convention(actual,h,n_d),identity=(str(h),n_d))
        return {"mathematical_reference_error": error, "actual_inverse_error": inverse_error,
                "published_convention":convention,
                "historical_validator_equivalence": "NOT_CERTIFIED_BY_THIS_REFERENCE"}

    def _published_convention(self, actual, h, n_d):
        # Full word, independent of DRun expansion and p @ k**n @ p-adjoint.
        # Upstream documents mathematical P K^n P^-1. Its legacy validator
        # instead left-multiplies the returned list; that is NOT this target.
        from fractions import Fraction
        from ._published_reference import _get_s2_sequence
        weights = _get_s2_sequence("8th(Morales-YP8m8)",kernel_steps=n_d)
        indices = tuple(range(len(self.s.context.block_ids)))
        expected = self.identity.copy()
        for weight in map(Fraction,weights):
            for index in indices[:-1]+indices[::-1]:
                time = h*weight/(n_d*(1 if index==indices[-1] else 2))
                expected = expected @ self._exp(self.s.context.block_ids[index],time)
        error = float(self.np.linalg.norm(actual-expected,ord=2))
        if not math.isfinite(error) or error>self.s.numerical_atol:
            raise Stage1Failure("PUBLISHED_CONVENTION_CORRECTNESS_FAILURE")
        return dict(operator_error=error,reference="UPSTREAM_MATHEMATICAL_FULL_WORD",
                    legacy_direct_append_equivalence="NOT_ASSERTED")

    def perform(self, kind, args, params):
        if kind == "validate_input":
            return self._validate()
        if not self.validated:
            raise ValueError("Input validation must precede every scientific operation.")
        np = self.np
        if kind == "exponential":
            return self._exp(*params)
        if kind == "product_append":
            value = self.identity.copy()
            for matrix in args:
                value = matrix @ value
            return value
        if kind == "power":
            return np.linalg.matrix_power(args[0], params[0])
        if kind == "physical_scalar_phase":
            return np.exp(-1j * float(params[0]) * self.scalar) * args[0]
        if kind == "rte_moments":
            config, _distribution = self.execution.science_call("rte.config", lambda: self.make_rte_config(
                self.i.symbolic_tail, evolution_time=float(params[0]), rte_steps=self.s.r_rte,
                finite_taylor_order=self.s.cutoff, truncation_tolerance=self.s.truncation_tolerance,
                seed=self.s.seed,
            ), identity=str(params[0]))
            return self.execution.science_call("rte.moments", lambda: self.finite_rte_operator_moments(
                self.i.normalized_tail, config), identity=str(params[0]))
        if kind == "moment_field":
            return getattr(args[0], params[0])
        if kind == "normalization":
            product = args[0].normalization_product ** params[0]
            return {"normalization_product": product, "attenuation": 1 / product}
        if kind == "rescale_raw":
            return args[0] * args[1]["normalization_product"]
        if kind == "operator_error":
            value = float(np.linalg.norm(args[0] - args[1], ord=2))
            if not math.isfinite(value):
                raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
            return value
        if kind == "signals":
            return self._signals(args)
        if kind == "a4_reference_check":
            return self._a4_reference(args[0], *params)
        raise ValueError("Unknown evaluation operation: " + kind)
