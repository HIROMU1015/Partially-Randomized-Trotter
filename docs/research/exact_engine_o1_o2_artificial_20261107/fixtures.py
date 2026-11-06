"""Handwritten/formulaic rational fixtures. No scientific inputs or PRNG."""
from dataclasses import dataclass
from fractions import Fraction as F

Z = (F(0), F(0))
ONE = (F(1), F(0))


def matrix(kind, k):
    d = 1 << k
    a = [[Z for _ in range(d)] for _ in range(d)]
    for i in range(d):
        a[i][i] = ONE
    if kind == "identity":
        pass
    elif kind == "diagonal":
        for i in range(d):
            a[i][i] = ((F(3, 5), F(4, 5)) if i % 2 else (F(-1), F(0)))
    elif kind == "tiny":
        a[0][1] = (F(1, 2**120), F(-1, 3**80))
    elif kind == "hadamard":
        assert k == 1
        a = [[(F(1, 2), F(0)), (F(1, 2), F(0))],
             [(F(1, 2), F(0)), (F(-1, 2), F(0))]]
    elif kind == "xx_block":
        assert k == 2
        a[1][1] = a[2][2] = (F(3, 5), F(0))
        a[1][2] = a[2][1] = (F(0), F(-4, 5))
    elif kind == "controlled":
        assert k in (2, 3)
        target = matrix("hadamard" if k == 2 else "xx_block", k - 1)
        for i in range(d // 2):
            for j in range(d // 2):
                a[2 * i + 1][2 * j + 1] = target[i][j]
    elif kind == "sparse_complex":
        for i in range(d):
            a[i][(i + 1) % d] = (F((-1)**i, 7), F(i + 1, 11))
    elif kind == "dense":
        a = [[(F((-1)**(i + j) * (i + 1), 2 * d + j + 1),
               F(j + 1, 3 * d + i + 1)) for j in range(d)] for i in range(d)]
    else:
        raise ValueError(kind)
    return tuple(tuple(row) for row in a)


def state(n, kind="general", bits=16):
    d = 1 << n
    if kind == "sparse":
        return tuple((F(3), F(4)) if i in (0, d - 1) else Z for i in range(d))
    if kind == "cancellation":
        return tuple((F((-1)**i, 3), F((-1)**(i // 2), 5)) for i in range(d))
    if kind == "dyadic":
        return tuple((F((1 << bits) + i + 1, 1 << (bits + 3)),
                      F((-1)**i * (i + 1), 1 << (bits + 5))) for i in range(d))
    return tuple((F(i % 5 - 2, 7), F(i % 7 - 3, 11)) for i in range(d))


@dataclass(frozen=True)
class Fixture:
    name: str
    initial: tuple
    matrices: tuple
    gates: tuple  # (matrix index, ordered support)
    ancilla: int


def benchmark_fixtures():
    # Fixed before first performance measurement; never filtered by results.
    specs = (("identity_k1", 5, "identity", 1, "general", 16),
             ("diagonal_k3", 8, "diagonal", 3, "dyadic", 16),
             ("xx_block_k2", 8, "xx_block", 2, "general", 16),
             ("controlled_k3", 8, "controlled", 3, "dyadic", 16),
             ("sparse_complex_k2", 8, "sparse_complex", 2, "general", 16),
             ("dense_k1", 5, "dense", 1, "general", 16),
             ("dense_k3", 5, "dense", 3, "dyadic", 16),
             ("large_bits_diagonal", 7, "diagonal", 3, "dyadic", 512),
             ("tiny_nonzero_k1", 5, "tiny", 1, "sparse", 16))
    fixtures = []
    for name, n, kind, k, state_kind, bits in specs:
        qs = tuple(range(n - 1, n - 1 - k, -1))
        fixtures.append(Fixture(name, state(n, state_kind, bits), (matrix(kind, k),),
                                tuple((0, qs) for _ in range(12)), n - 1))
    fixtures.append(Fixture("mixed_sequence", state(7, "dyadic"),
                            (matrix("hadamard", 1), matrix("xx_block", 2),
                             matrix("diagonal", 3), matrix("controlled", 3)),
                            ((0, (6,)), (1, (6, 0)), (2, (5, 1, 3)),
                             (3, (6, 2, 0))) * 3, 6))
    return tuple(fixtures)


FIXED_WORDS = (0, 1, (1 << 52) - 1, 1 << 52, (1 << 53) - 2, (1 << 53) - 1)
