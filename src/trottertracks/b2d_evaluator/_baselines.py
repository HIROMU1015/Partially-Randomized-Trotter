"""Unmodified baseline generator excerpts; no scientific imports.

Source: HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae
Commit: 20d64c2fea1ed2d2188777df4f3d962ee94df486
product_formula.py blob: 70d68691a1dcdedea5447aac4b8bbfa51141d4cb
"""
from typing import List

def trotter_2nd_list() -> List[float]:
    """2次(Trotter)の係数列を返す。"""
    # 係数リストを構築
    w = [1.0]
    return w


def yoshida_4th_list() -> List[float]:  # s3odr4
    """Yoshida 4次の係数列を返す。"""
    # 係数リストを構築
    w = [-1 * (2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3))]
    return w


def morales_2025_y8m10b_list() -> List[float]:
    """Morales et al. (QIC 2025) の Y8m10b 8次係数列を返す。

    Published Table 1, ``Best 8th order for eigenvalue error``.
    DOI: 10.2478/qic-2025-0001.  The returned representation is
    ``[w0, w1, ..., w10]`` with ``w0 = 1 - 2 * sum(w1, ..., w10)``.
    """
    w1to10 = [
        0.10467636532245895252340732579853,
        -0.57896999331780988041471955125778,
        0.57503350160061785946141563279891,
        0.12231011868707029786561397542663,
        0.27793149999039524816733903301747,
        -0.37349605088056728482635987352576,
        0.11575566589480463220616543972403,
        0.1464645610975800618712569230326,
        -0.39443578322284085764474498594073,
        0.44370228726021218923197141183196,
    ]
    return [1 - 2 * sum(w1to10)] + w1to10
