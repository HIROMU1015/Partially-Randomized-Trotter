"""Small PR-search connections; imports perform no scientific work.

The old sampler, D2 control and accounting remain the implementations.  This
module names candidate/seed identities and exports their actual finite law.
"""
from dataclasses import asdict
from fractions import Fraction as Q
import math

from . import h4_conditional_cng as A


def candidate_id(prefix, q, r, K):
    if (type(prefix) is not int or prefix < 0 or type(q) is not int
            or q not in (1, 2, 4, 8) or type(r) is not int or r not in (2, 4, 8)
            or type(K) is not int or K not in (2, 4)):
        raise ValueError("PR_SEARCH_PARAMETER_DOMAIN")
    return f"p{prefix:03d}_q{q}_r{r}_K{K}"


def config(prefix, q, r, K):
    from .fixed_signal_evaluation import FixedSignalConfig
    candidate_id(prefix, q, r, K)
    return FixedSignalConfig("pauli_jw", method="PR", T=.8, L_D=prefix,
                             q=q, r=r, K=K, cost_samples=8, seed=20261025)


def trajectory_seed(prefix, q, r, K, cohort, index):
    """Deterministic addresses, not a proof of independent PRNG streams."""
    import numpy as np
    candidate_id(prefix, q, r, K)
    tags = {"exploration": 101, "confirmation": 202}
    if cohort not in tags or type(index) is not int or index < 0:
        raise ValueError("COHORT_INDEX")
    return int(np.random.SeedSequence([20261025, 3101, prefix, q, r, K,
                                      tags[cohort], index]).generate_state(1, dtype=np.uint64)[0])


def probability_masses(probabilities):
    # Exact reuse of the frozen C NumPy-choice/CDF graph, without importing its
    # numerical backend into the Qiskit runtime. No draws or RNG replay.
    cdf = A.extract("src/trotterlib/h4_ball_evidence.py", "cdf_binary64",
                    dict(math=math, require=_require))(list(probabilities))
    masses = A.extract("src/trotterlib/h4_ball_evidence.py", "grid_masses",
                       dict(Q=Q, rational=Q, require=_require))(cdf)
    return dict(cdf=cdf, exact_masses=[str(x) for x in masses])


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def law_descriptor(hamiltonian, cfg):
    """New numerical evaluation on a supplied H: real H4 needs permission."""
    import numpy as np
    from .pauli_signal_adapter import PauliSignalAdapter
    if np.__version__!="1.26.4":
        raise ValueError("FROZEN_NUMPY_CHOICE_GRAPH")
    a = PauliSignalAdapter(hamiltonian, cfg.method, cfg.L_D, cfg.T/cfg.q, cfg.r, cfg.K)
    if not a.randomized or cfg.method != "PR" or cfg.product_formula != "S2":
        raise ValueError("NONEMPTY_PR_TAIL_REQUIRED")
    candidate_id(a.ld, cfg.q, cfg.r, cfg.K)
    total = math.fsum(x.probability for x in a.components)
    cp = [float(x.probability/total) for x in a.components]
    dist = a.distribution
    log_B = cfg.q*a.log_B_step
    if not math.isfinite(log_B) or log_B >= 709:
        raise ValueError("NONFINITE_NORMALIZATION_STOP")
    saved_B = math.exp(log_B)
    return dict(schema="pauli_pr_search_law_v1", candidate_id=candidate_id(a.ld,cfg.q,cfg.r,cfg.K),
        config=asdict(cfg), n_system=a.num_qubits, Hamiltonian_hash=hamiltonian.hamiltonian_hash,
        qubit_order="qiskit_little_endian", step_time=cfg.T/cfg.q,
        retained=[dict(label=s,coefficient=c) for s,c in a.retained],
        components=[dict(label=a.labels[x.component_id],sign=x.coefficient_sign,
                         coefficient_abs=x.coefficient_abs) for x in a.components],
        component_mapping=probability_masses(cp), order_mapping=probability_masses(dist.order_probabilities),
        component_probabilities=cp, orders=list(dist.orders),
        order_probabilities=list(dist.order_probabilities),
        angles=[math.atan(dist.dimensionless_step_time/(k+1)) for k in dist.orders],
        scalar_step_binary64=float(-hamiltonian.constant*a.step_time),
        lambda_r=a.lambda_r, B_K=dist.exact_finite_distribution,
        log_B_binary64=log_B, B_saved_binary64=saved_B,
        nominal_polynomial_degree=cfg.K+1,numpy_version=np.__version__,uniform_bits=53,
        IID_uniform_word_assumption=True,
        target_semantics="C_EFFECTIVE_GRID_LAW_EXACT_RETAINED_PRODUCTS_NATIVE_STEP_SCALAR",
        compiled_rho=None, formal_certificate_accepted=False)


def central_trajectory(hamiltonian, cfg, *, seed, audit=None):
    """Same sampler and body builder; no replacement of the RTE algorithm."""
    from .pauli_signal_adapter import PauliSignalAdapter
    from .h4_d2_pauli_control import central_control_body
    from .fixed_signal_evaluation import hadamard_wrapper
    if audit is None:
        cls=PauliSignalAdapter
        kwargs={}
    else:
        from .rte_event_audit import AuditedPauliSignalAdapter
        cls=AuditedPauliSignalAdapter;kwargs=dict(audit=audit)
    a = cls(hamiltonian,cfg.method,cfg.L_D,cfg.T/cfg.q,cfg.r,cfg.K,**kwargs)
    if not a.randomized:
        raise ValueError("NONEMPTY_PR_TAIL_REQUIRED")
    body = a.trajectory(q=cfg.q,seed=seed)
    lift = central_control_body(body)
    wrappers = {axis:hadamard_wrapper(lift.body,axis,already_controlled=True)
                for axis in ("cosine","sine")}
    return body, wrappers, lift.sidecar


def accounting_scale(descriptor):
    # Support is +/- the actual binary64 B used by the estimator, not the
    # mathematical Taylor norm of a different law. Enclose its log outward.
    B = Q(descriptor["B_saved_binary64"])
    if B < 1:
        raise ValueError("POSITIVE_PR_SCALE")
    return A.scale_bounds(B,A.trans_bounds("ln",B)[1])


def account(target, prices, epsilon, descriptor):
    from .h4_pauli_b0_comparison import account as existing_account
    return existing_account(target,prices,epsilon,scale=accounting_scale(descriptor))
