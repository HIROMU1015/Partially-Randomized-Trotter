"""Four-candidate H6 connection; no scientific work on import and no run CLI.

Preparation tests use only the explicit 2/3-qubit artificial entrypoints.
Real H6 calls require the separately permitted Phase B and a fresh fixed batch.
No historical SOURCE, sampler, numerical kernel or accounting is modified.
"""
from dataclasses import asdict
from fractions import Fraction as Q
import hashlib
import json
import math
from . import h4_conditional_cng as A
from . import h4_pauli_pr_search as S

SIGNATURES = ((78, 8, 16, 2), (78, 8, 32, 2), (78, 8, 64, 2), (684, 1, 16, 4))
CANDIDATES = {f'PR_p{p:04d}_q{q}_r{r}_K{K}': (p, q, r, K) for p, q, r, K in SIGNATURES}
MODEL_SHA256 = '6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681'
REFERENCE_SHA256 = '9956466a5cf3286af9960b792b455bb8bacc2532a34ac7a28b470475f92fcb7a'
HAMILTONIAN_HASH = '37b2747eaafacef8b4b470e5f4d6e195ac0602dc8e59953e3b76152d4576c847'


def signature(cid):
    if cid not in CANDIDATES:
        raise ValueError('FIXED_FOUR_H6_HYBRID_CANDIDATES_ONLY')
    return CANDIDATES[cid]


def _kernel_id(prefix, q, r, K):
    # Only the private extraction has the wider domain. r8 is for toy regression.
    if any(type(x) is not int for x in (prefix, q, r, K)) or prefix < 1 or q not in (1, 8) or r not in (8, 16, 32, 64) or K not in (2, 4):
        raise ValueError('HYBRID_KERNEL_PARAMETER_DOMAIN')
    return f'p{prefix:03d}_q{q}_r{r}_K{K}'


def _config(prefix, q, r, K):
    from .fixed_signal_evaluation import FixedSignalConfig
    _kernel_id(prefix, q, r, K)
    # Legacy descriptor metadata only; this does NOT prescribe future n_E/n_C.
    return FixedSignalConfig('pauli_jw', method='PR', T=.8, L_D=prefix,
                             q=q, r=r, K=K, cost_samples=8, seed=20261025)


def config(cid):
    return _config(*signature(cid))


_law = A.extract('src/trotterlib/h4_pauli_pr_search.py', 'law_descriptor',
                 dict(__package__=__package__, asdict=asdict, math=math,
                      candidate_id=_kernel_id, probability_masses=S.probability_masses))
_seed = A.extract('scripts/pauli_h2_h6_study.py', 'seeds', {})


def bind_saved_inputs(model_bytes, reference_bytes):
    """Static byte identity only: no norm, Hamiltonian or signal evaluation."""
    if hashlib.sha256(model_bytes).hexdigest() != MODEL_SHA256 or hashlib.sha256(reference_bytes).hexdigest() != REFERENCE_SHA256:
        raise ValueError('FIXED_H6_INPUT_BYTES')
    return json.loads(model_bytes), json.loads(reference_bytes)


def _real_h6(ham):
    if ham.num_qubits != 12 or len(ham.terms) != 918 or ham.hamiltonian_hash != HAMILTONIAN_HASH:
        raise ValueError('FIXED_H6_HAMILTONIAN')


def law_descriptor(ham, cid):
    signature(cid)
    _real_h6(ham)
    return _law(ham, config(cid))


def trajectory_seed(cid, cohort, index):
    prefix, q, r, K = signature(cid)
    if cohort not in ('exploration', 'confirmation') or type(index) is not int or index < 0:
        raise ValueError('COHORT_INDEX')
    return _seed(dict(molecule=6, config=dict(L_D=prefix, q=q, r=r, K=K)), cohort, index)


def _trajectory(ham, cfg, seed):
    from .rte_event_audit import AuditedPauliSignalAdapter
    from .h4_d2_pauli_control import central_control_body
    from .fixed_signal_evaluation import hadamard_wrapper
    from .pauli_rte_event_extension import Capture
    audit = Capture()
    adapter = AuditedPauliSignalAdapter(ham, cfg.method, cfg.L_D, cfg.T/cfg.q,
                                       cfg.r, cfg.K, audit=audit)
    body = adapter.trajectory(q=cfg.q, seed=seed)
    assert len(audit.events) == cfg.q*cfg.r
    lift = central_control_body(body)
    wrapped = {axis: hadamard_wrapper(lift.body, axis, already_controlled=True) for axis in A.AXES}
    return body, wrapped, dict(seed=seed, event_count=len(audit.events), events=audit.events,
                              step_seeds=audit.step_seeds, control_sidecar=lift.sidecar)


def trajectory(ham, cid, cohort, index):
    _real_h6(ham)
    return _trajectory(ham, config(cid), trajectory_seed(cid, cohort, index))


def mean_target(model, reference, law, cid):
    cfg = config(cid)
    if model['n_qubits'] != 12 or model['Hamiltonian_hash'] != HAMILTONIAN_HASH or law['Hamiltonian_hash'] != HAMILTONIAN_HASH:
        raise ValueError('FIXED_H6_MEAN_INPUT')
    for k in ('L_D', 'q', 'r', 'K', 'T'):
        if law['config'][k] != getattr(cfg, k):
            raise ValueError('LAW_CANDIDATE_MISMATCH')
    from .pauli_vector_tight_enclosure import evaluate
    return evaluate(model, reference, law['config'], law)


def account_target(target, prices, epsilon, law):
    from .h6_rte_fairness import account, scale
    return account(target, prices, epsilon, scale=scale(Q(law['B_saved_binary64'])))


def _artificial(ham, prefix):
    if ham.num_qubits not in (2, 3) or not 0 < prefix < len(ham.terms):
        raise ValueError('ARTIFICIAL_2_3_QUBITS_NONEMPTY_RETAINED_AND_TAIL_ONLY')


def artificial_law(ham, prefix, q, r, K):
    _artificial(ham, prefix)
    return _law(ham, _config(prefix, q, r, K))


def artificial_trajectory(ham, prefix, q, r, K, *, seed):
    _artificial(ham, prefix)
    return _trajectory(ham, _config(prefix, q, r, K), seed)


def artificial_mean(model, reference, law):
    if model['n_qubits'] not in (2, 3) or model['n_qubits'] != law['n_system'] or not 0 < law['config']['L_D'] < len(model['hamiltonian']['terms']):
        raise ValueError('ARTIFICIAL_HYBRID_MEAN_ONLY')
    from .pauli_vector_tight_enclosure import evaluate
    return evaluate(model, reference, law['config'], law)
