"""Registered AX-1b cost models. No trotterlib, science IO, circuit or signal code."""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

from .ax1b_contract import FEATURES, RETAIN_PRIORITY, Stop, number, require, digest


@dataclass(frozen=True)
class Features:
    n_det: float | None
    E_rand: float | None
    n_fixed: float | None
    q: float | None

    def __post_init__(self):
        for key in FEATURES[1:]:
            value = getattr(self, key)
            if value is not None:
                number(value, key, positive=(key == "q"))

    @classmethod
    def from_mapping(cls, fields):
        require(set(fields) == set(FEATURES[1:]), "SCHEMA", "feature projection must contain only I1 feature keys")
        return cls(**fields)

    def mapping(self):
        return dict(intercept=1.0, **{k: getattr(self, k) for k in FEATURES[1:]})

    def actions(self):
        if any(getattr(self, k) is None for k in ("n_det", "E_rand", "n_fixed")):
            return dict(status="INPUT_MISSING", A_exact=None, A_ceil=None)
        exact = self.n_det + self.E_rand + self.n_fixed
        ceil = self.n_det + math.ceil(self.E_rand - 1e-15) + self.n_fixed
        number(exact, "A_exact")
        return dict(status="AVAILABLE_INDEX_ONLY", A_exact=exact, A_ceil=ceil, unit="action_count_per_shot")


def _array_module():
    import numpy as np
    return np


def _rms(array):
    np = _array_module()
    largest = float(np.max(np.abs(array)))
    return largest * float(np.sqrt(np.mean((array / largest)**2))) if largest else 0.0


def rank(array):
    np = _array_module()
    singular = np.linalg.svd(array, compute_uv=False)
    count = int(np.count_nonzero(singular > singular[0] * 1e-12)) if len(singular) and singular[0] else 0
    return count, singular.tolist()


def kkt_residual(X, y, theta):
    np = _array_module()
    require(np.all(np.isfinite(theta)) and np.all(theta >= 0), "NUMERICAL_FIT", "negative/nonfinite NNLS coefficient")
    g = X.T @ (X @ theta - y)
    active = theta > 1e-12
    residual = max(float(np.max(np.abs(g[active]))) if np.any(active) else 0.0,
                   float(np.max(np.maximum(-g[~active], 0))) if np.any(~active) else 0.0,
                   float(np.max(np.abs(theta * g))) if len(theta) else 0.0)
    require(math.isfinite(residual) and residual <= 1e-8, "NUMERICAL_FIT", "NNLS KKT residual failed")
    return residual


@dataclass
class CostFit:
    model_id: str
    status: str
    coefficients: dict
    retained: list
    dropped: dict
    scales: dict
    relations: dict
    training_ids: list
    audit: dict

    def predict(self, features):
        if self.status != "FIT_OK":
            return dict(C_pred=None, status=self.status, missing_reason="fit unavailable", extrapolation_flags=[])
        data = features.mapping()
        if self.model_id == "PRED_BASE_SINGLE_COEFF":
            actions = features.actions()
            if actions["A_exact"] is None:
                return dict(C_pred=None, status="INPUT_MISSING", missing_reason="A_exact missing", extrapolation_flags=[])
            prediction=self.coefficients["A_exact"] * actions["A_exact"]
            number(prediction,"predicted cost")
            return dict(C_pred=prediction, status="PREDICTION_OK", missing_reason=None, extrapolation_flags=[])
        missing = [k for k in self.retained if data[k] is None]
        if missing:
            return dict(C_pred=None, status="INPUT_MISSING", missing_reason=missing, extrapolation_flags=[])
        prediction = math.fsum(self.coefficients[k] * data[k] for k in self.retained)
        number(prediction, "predicted cost")
        flags = []
        for key, relation in self.relations.items():
            if data[key] is None:
                flags.append("DROPPED_FEATURE_RELATION_UNCHECKABLE:" + key)
                continue
            actual = data[key] / self.scales[key] if self.scales[key] else data[key]
            expected = math.fsum(relation[k] * data[k] / self.scales[k] for k in self.retained)
            if abs(actual - expected) > 1e-12 * max(1.0, abs(actual), abs(expected)):
                flags.append("OUTSIDE_TRAIN_FEATURE_RELATION:" + key)
        return dict(C_pred=prediction, status="PREDICTION_OK", missing_reason=None, extrapolation_flags=flags)

    def record(self):
        return {**self.__dict__, "training_membership_sha256": digest(sorted(self.training_ids)), "version": 1}


def fit_cost(model_id, features, target, training_ids):
    """Fit caller-provided training C only; no test truth or bias parameter exists."""
    require(model_id in {"PRED_BASE_SINGLE_COEFF", "PRED_BASE_FEW_PARAM"}, "SCHEMA", "unregistered calibrated model")
    require(len(features) == len(target) == len(training_ids) and len(set(training_ids)) == len(training_ids), "INPUT_IDENTITY", "fit row identity")
    np = _array_module()
    y = np.asarray([number(v, "training C") for v in target], dtype=np.float64)
    base = dict(model_id=model_id, coefficients={}, retained=[], dropped={}, scales={}, relations={}, training_ids=list(training_ids), audit={})
    if not len(features):
        return CostFit(status="FIT_UNIDENTIFIABLE", **base)
    if model_id == "PRED_BASE_SINGLE_COEFF":
        actions = [f.actions()["A_exact"] for f in features]
        if any(v is None for v in actions):
            return CostFit(status="INPUT_MISSING", **base)
        x = np.asarray(actions, dtype=np.float64)
        scale_x, scale_y = _rms(x), _rms(y) or 1.0
        if not scale_x or len(x) < 2:
            return CostFit(status="FIT_UNIDENTIFIABLE", **base)
        xs, ys = x / scale_x, y / scale_y
        theta = max(0.0, float(xs @ ys / (xs @ xs)))
        residual = kkt_residual(xs[:, None], ys, np.array([theta]))
        base.update(coefficients={"A_exact": theta * scale_y / scale_x}, retained=["A_exact"],
                    scales={"A_exact": scale_x}, audit=dict(kkt_residual=residual, target_scale=scale_y, weight="equal_candidate"))
        number(base["coefficients"]["A_exact"],"single coefficient")
        return CostFit(status="FIT_OK", **base)
    raw = [f.mapping() for f in features]
    if any(v is None for f in raw for v in f.values()):
        return CostFit(status="INPUT_MISSING", **base)
    X = np.asarray([[f[k] for k in FEATURES] for f in raw], dtype=np.float64)
    scales = {k: _rms(X[:, i]) for i, k in enumerate(FEATURES)}
    scaled = X / np.asarray([scales[k] or 1.0 for k in FEATURES])
    retained, dropped, indices, rank_audit = [], {}, [], {}
    for key in RETAIN_PRIORITY:
        idx = FEATURES.index(key)
        if not scales[key]:
            dropped[key] = "ALL_ZERO_TRAIN_COLUMN"
            continue
        current_rank, singular = rank(scaled[:, indices + [idx]])
        rank_audit[key] = singular
        if current_rank > len(indices):
            retained.append(key)
            indices.append(idx)
        else:
            dropped[key] = "TRAIN_LINEAR_DEPENDENCY"
    base.update(retained=retained, dropped=dropped, scales=scales)
    if not indices or len(X) < 2 * len(indices):
        return CostFit(status="FIT_UNIDENTIFIABLE", **base)
    design = scaled[:, indices]
    relations = {}
    for key in dropped:
        if not scales[key]:
            relations[key] = {k: 0.0 for k in retained}
        else:
            # Pure feature algebra, train only; no target or test data.
            coef = np.linalg.lstsq(design, scaled[:, FEATURES.index(key)], rcond=1e-12)[0]
            relations[key] = dict(zip(retained, coef.tolist()))
    import scipy
    require(scipy.__version__ == "1.14.1", "ENVIRONMENT", "NNLS reference requires SciPy 1.14.1; no fallback")
    from scipy.optimize import nnls
    scale_y = _rms(y) or 1.0
    ys = y / scale_y
    try:
        theta = np.zeros(len(indices)) if not np.any(y) else nnls(design, ys, maxiter=10000, atol=1e-12)[0]
    except (RuntimeError, ValueError) as exc:
        raise Stop("NUMERICAL_FIT", "NNLS failed; no fallback") from exc
    residual = kkt_residual(design, ys, theta)
    coefficients = {k: 0.0 for k in FEATURES}
    coefficients.update({k: float(theta[i] * scale_y / scales[k]) for i, k in enumerate(retained)})
    for value in coefficients.values():
        number(value,"few-param coefficient")
    base.update(coefficients=coefficients, relations=relations,
                audit=dict(rank=len(indices), singular_values_by_addition=rank_audit, target_scale=scale_y,
                           kkt_residual=residual, weight="equal_candidate", backend="scipy.optimize.nnls:1.14.1"))
    return CostFit(status="FIT_OK", **base)


def structural_na():
    return dict(model_id="PRED_BASE_STRUCT_ACCOUNT", version=1, C_pred=None,
                status="REGISTERED_INPUTS_UNAVAILABLE_N_A", missing_reason="basis/ordered events/control/boundary accounting not in allowlist")


def finite_normalization(lambda_R, T, q, r, K):
    """Canonical finite scalar accounting only, matching rte.py paired weights."""
    number(lambda_R, "lambda_R")
    number(T, "T")
    require(all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in [q, r]), "SCHEMA", "positive integer q/r")
    require(isinstance(K, int) and not isinstance(K, bool) and K >= 0 and K % 2 == 0, "SCHEMA", "even finite cutoff")
    tau = lambda_R * T / (q * r)
    require(math.isfinite(tau), "NUMERICAL_FIT", "nonfinite tau")
    orders = list(range(0, K + 1, 2))
    weights = []
    for n in orders:
        if tau == 0:
            weight = 1.0 if n == 0 else 0.0
        else:
            log_taylor = n * math.log(abs(tau)) - math.lgamma(n + 1)
            try:
                weight = math.exp(log_taylor) * math.hypot(1.0, abs(tau) / (n + 1))
            except OverflowError:
                weight = math.inf
        weights.append(weight)
    try:
        b = math.fsum(weights)
    except OverflowError as exc:
        raise Stop("NUMERICAL_FIT","finite distribution normalization overflow; no rescue") from exc
    require(math.isfinite(b) and b > 0, "NUMERICAL_FIT", "finite distribution normalization overflow; no rescue")
    log_B = q * (r * math.log(b))  # M1 source order, not an altered B approximation.
    try:
        B = math.exp(log_B)
    except OverflowError:
        raise Stop("NUMERICAL_FIT", "total B overflow; no source rescue")
    exponent = tau * tau
    upper=math.exp(exponent) if math.isfinite(exponent) and exponent<=math.log(sys.float_info.max) else None
    require(log_B >= 0 and math.isfinite(B), "NUMERICAL_FIT", "invalid B")
    return dict(model_id="AXM1_FINITE_NORMALIZATION", orders=orders, weights=weights, probabilities=[w / b for w in weights],
                tau=tau, b=b, log_B=log_B, B=B, bound_kind="ANALYTIC_UPPER_BOUND_SEPARATE",
                paper_upper_bound=upper, paper_upper_bound_overflowed=upper is None, log_bound_slack=exponent - math.log(b))


def complexity_gate(single, few):
    """Inputs are the four preregistered q-fold summaries, never PM1/M2."""
    default = dict(selected_model="PRED_BASE_SINGLE_COEFF", gate_status="UNDETERMINED")
    if set(single) != {1, 2, 4, 8} or set(few) != set(single):
        return {**default, "reason": "missing q fold"}
    for q in single:
        a, b = single[q], few[q]
        if not a.get("full_coverage") or not b.get("full_coverage") or a.get("support_hash") != b.get("support_hash") or a.get("score") is None or b.get("score") is None:
            return {**default, "reason": "coverage or identifiable score unavailable"}
        number(a["score"], "single score")
        number(b["score"], "few score")
    s = math.fsum(v["score"] for v in single.values()) / 4
    f = math.fsum(v["score"] for v in few.values()) / 4
    passed = s > 0 and s - f >= 0.01 and (s - f) / s >= 0.10 and all(few[q]["score"] - single[q]["score"] <= 0.02 for q in single)
    return dict(selected_model="PRED_BASE_FEW_PARAM" if passed else default["selected_model"],
                gate_status="PASS" if passed else "KEEP_SIMPLE", balanced_single=s, balanced_few=f,
                independent_test=False)
