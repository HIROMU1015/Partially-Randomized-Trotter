"""Prepared scalar kernels; no real-data CLI. Preflight uses synthetic tests only.

These functions do not load results, optimize candidates, recompute shots, sample,
build circuits, or compile. Real saved-value evaluation awaits user authorization.
"""
import ast
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace


def verified_bytes(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('FIXED_SOURCE_HASH_MISMATCH')
    return data


def standalone_module(path, expected_sha256, name):
    """Load a verified library/metadata file without importing its package."""
    verified_bytes(path, expected_sha256)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_functions(path, expected_sha256, names, scope):
    """Reuse original function ASTs; never execute runner-level statements."""
    data = verified_bytes(path, expected_sha256)
    tree = ast.parse(data)
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if any(name not in nodes for name in names):
        raise ValueError('MISSING_FIXED_FUNCTION')
    result = dict(scope)
    module = ast.Module(body=[nodes[name] for name in names], type_ignores=[])
    exec(compile(module, str(path), 'exec'), result)
    return result


def normalization_terms(weights, counts, *, T, q, K, lambda_R,
                        finite_distribution, canonical_tau):
    """A/B/C as separate quantities; this function does not read real inputs."""
    if type(q) is not int or q < 1 or type(K) is not int or K < 0 or K % 2:
        raise ValueError('INTEGER_Q_EVEN_K')
    if len(weights) != len(counts) or not weights:
        raise ValueError('STAGE_ALIGNMENT')
    if any(type(r) is not int or r < 1 for r in counts):
        raise ValueError('POSITIVE_INTEGER_STAGE_COUNTS')
    if not all(math.isfinite(x) for x in [T, lambda_R, *weights]):
        raise ValueError('FINITE_SCALARS')
    prep = SimpleNamespace(rte_preparation=SimpleNamespace(
           symbolic_tail=SimpleNamespace(lambda_r=lambda_R)))
    log_B = 0.0
    tau = []
    for w, r in zip(weights, counts):
        # Keep the original execution's multiplication/division order.
        signed_time = T * w / q
        t = canonical_tau(prep, signed_time, r)
        beta = finite_distribution(t, K).exact_finite_distribution
        log_B += q * r * math.log(beta)
        tau.append(t)
    a_K = 0.5 if K == 0 else 1.0
    delta = T / q
    leading = a_K * q * lambda_R**2 * delta**2 * math.fsum(
              w*w/r for w, r in zip(weights, counts))
    continuous = a_K * lambda_R**2 * T*T / (q * sum(counts)) * (
                 math.fsum(abs(w) for w in weights)**2)
    return dict(exact=log_B, integer_leading=leading, continuous=continuous,
                max_abs_tau=max(map(abs, tau)), counts=tuple(counts),
                events_per_trajectory=q*sum(counts),
                fourth_order_scale=q*math.fsum(r*t**4 for r,t in zip(counts,tau)))


def paired_metric_cost(pairs, N_cos, N_sin, *, paired, cost_for, cohort):
    """Original paired statistics and fixed-N accounting, with no new shot rule."""
    if type(N_cos) is not int or type(N_sin) is not int:
        raise ValueError('SAVED_INTEGER_N_REQUIRED')
    endpoint = dict(candidate=dict(method='PR'), N_cos=N_cos, N_sin=N_sin)
    statistics_payload = paired(pairs)
    return cost_for(endpoint, dict(unit_statistics=statistics_payload, cohort=cohort))


def mean_only_metric_cost(C_cos_literal, C_sin_literal, N_cos, N_sin):
    """No SE inference when only saved means exist; exact literal arithmetic."""
    if any(x is None or x == '' for x in (C_cos_literal, C_sin_literal)):
        raise ValueError('MISSING_PRICE_IS_NOT_ZERO')
    if type(N_cos) is not int or type(N_sin) is not int:
        raise ValueError('SAVED_INTEGER_N_REQUIRED')
    value = N_cos*Fraction(C_cos_literal) + N_sin*Fraction(C_sin_literal)
    return dict(G_exact=str(value), G_SE=None,
                uncertainty='UNKNOWN_FROM_MEANS_ONLY', N_reoptimized=False)


if __name__ == '__main__':
    raise SystemExit('Preflight module only. Real saved-value reanalysis requires '
                     'the user authorization specified in the task, section 2.')
