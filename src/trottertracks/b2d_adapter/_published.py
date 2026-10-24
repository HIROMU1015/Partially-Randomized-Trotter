"""Unmodified function excerpts from the fixed published-PF SOURCE.

Origin: HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae
Commit: 20d64c2fea1ed2d2188777df4f3d962ee94df486
product_formula.py blob: 70d68691a1dcdedea5447aac4b8bbfa51141d4cb
pf_decomposition.py blob: b0985414adfa2323b7884c0d4683821f91eaec09

Only the stdlib dependency closure for YP8m8 is included. These functions
retain their upstream bodies and floating-point coefficients. No coefficient
search, coefficient repair, circuit construction, or scientific imports.
Sequences are interpreted as MATHEMATICAL product order by this adapter;
the consuming adapter explicitly reverses them for circuit append order.
"""
from typing import Iterable, List, Sequence

UPSTREAM_REPOSITORY = "HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae"
UPSTREAM_COMMIT = "20d64c2fea1ed2d2188777df4f3d962ee94df486"
PRODUCT_FORMULA_BLOB = "70d68691a1dcdedea5447aac4b8bbfa51141d4cb"
PF_DECOMPOSITION_BLOB = "b0985414adfa2323b7884c0d4683821f91eaec09"

def symmetric_s2_sequence(w_list: Sequence[float]) -> list[float]:
    """Convert compact ``[w0, ..., wm]`` weights to the 2m+1 S2 sequence."""
    weights = list(w_list)
    if not weights:
        return []
    return list(reversed(weights[1:])) + [weights[0]] + weights[1:]


def inverse_s2_sequence(sequence: Sequence[float]) -> list[float]:
    """Return the S2-block sequence implementing the inverse operator."""
    return [-weight for weight in reversed(sequence)]


def _iter_s2_steps(num_terms: int, w: float) -> Iterable[tuple[int, float]]:
    """単一 w の S2 対称ステップを生成する。"""
    for i in range(num_terms - 1):
        yield i, w / 2
    yield num_terms - 1, w
    for k in reversed(range(0, num_terms - 1)):
        yield k, w / 2


def morales_2025_yp8m8_kernel_list() -> List[float]:
    """Morales et al. (QIC 2025) の YP8m8 kernel 係数列を返す。

    These are the kernel coefficients in published Table 2.  A complete
    processed formula also requires the processor returned by
    :func:`morales_2025_yp8m8_processor_gamma_list`.
    """
    w1to8 = [
        0.21784176681731006074681969186513,
        0.1947017706053903224022456342907,
        0.18372413281145589944261642180363,
        -0.37307499512657736825709230652023,
        0.15757644257569146373033662060461,
        -0.33342207567391682979227850551172,
        0.51788649682987924281787142226803,
        0.21456475499897766986381219621761,
    ]
    return [1 - 2 * sum(w1to8)] + w1to8


def morales_2025_yp8m8_processor_gamma_list() -> List[float]:
    """Y8m10b と同じ公開論文 Table 2 の YP8m8 processor 係数を返す。

    The paper lists gamma_1 through gamma_9 and specifies gamma_10 by the
    zero-sum condition.
    """
    gamma1to9 = [
        -0.44324901019570126590495430949294,
        0.25459857192003772850622377066944,
        -0.73862036266779261573694538099739,
        -0.00024139614958652134370419495289618,
        0.73873460354125365739379753874964,
        -0.20285971152536085519251666906017,
        0.44989521689676869571827637424046,
        0.29538398007876871184026747505657,
        -0.3364996155865700091428329802017,
    ]
    return gamma1to9 + [-sum(gamma1to9)]


def morales_2025_yp8m8_processor_sequence() -> List[float]:
    """Return the S2-block sequence for the YP8m8 processor P.

    Appendix B defines ``P = Q(t) Q(-t)`` with
    ``Q(t) = S2(gamma_10 t) ... S2(gamma_1 t)``.
    """
    gamma = morales_2025_yp8m8_processor_gamma_list()
    q_plus = list(reversed(gamma))
    q_minus = [-value for value in reversed(gamma)]
    return q_plus + q_minus
