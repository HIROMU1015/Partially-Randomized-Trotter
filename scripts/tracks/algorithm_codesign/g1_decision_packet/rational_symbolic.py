"""Small exact univariate algebra kernel; no science libraries or solver imports."""
import ast
import itertools
from fractions import Fraction as F
from math import comb


def trim(a):
    a = list(map(F, a)) or [F(0)]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return tuple(a)


def add(a, b):
    return trim([ (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
                  for i in range(max(len(a), len(b))) ])


def mul(a, b):
    c = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y
    return trim(c)


def divmod_poly(a, b):
    a, b = list(trim(a)), trim(b)
    if b == (F(0),):
        raise ZeroDivisionError('zero polynomial')
    out = [F(0)] * max(1, len(a)-len(b)+1)
    while trim(a) != (F(0),) and len(a) >= len(b):
        k, q = len(a)-len(b), a[-1]/b[-1]
        out[k] += q
        for j, v in enumerate(b):
            a[k+j] -= q*v
        a = list(trim(a))
    return trim(out), trim(a)


def gcd(a, b):
    a, b = trim(a), trim(b)
    while b != (F(0),):
        _, r = divmod_poly(a, b)
        a, b = b, r
    return trim([v/a[-1] for v in a])


class RF:
    """Canonical rational function, coefficients ascending in one variable."""
    __slots__ = ('num', 'den')

    def __init__(self, num=0, den=1):
        if isinstance(num, RF):
            if den != 1:
                raise ValueError('use arithmetic for rational function division')
            self.num, self.den = num.num, num.den
            return
        n = trim(num if isinstance(num, (tuple, list)) else [num])
        d = trim(den if isinstance(den, (tuple, list)) else [den])
        if d == (F(0),):
            raise ZeroDivisionError('zero denominator')
        if n == (F(0),):
            self.num, self.den = n, (F(1),)
            return
        common = gcd(n, d)
        n, rn = divmod_poly(n, common)
        d, rd = divmod_poly(d, common)
        if rn != (F(0),) or rd != (F(0),):
            raise ArithmeticError('polynomial cancellation')
        lead = d[-1]
        self.num = trim([v/lead for v in n])
        self.den = trim([v/lead for v in d])

    @staticmethod
    def variable():
        return RF([0, 1])

    def __add__(self, other):
        b = RF(other)
        return RF(add(mul(self.num, b.den), mul(b.num, self.den)), mul(self.den, b.den))

    __radd__ = __add__

    def __neg__(self):
        return RF([-v for v in self.num], self.den)

    def __sub__(self, other):
        return self + -RF(other)

    def __rsub__(self, other):
        return RF(other) + -self

    def __mul__(self, other):
        b = RF(other)
        return RF(mul(self.num, b.num), mul(self.den, b.den))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = RF(other)
        return RF(mul(self.num, b.den), mul(self.den, b.num))

    def __rtruediv__(self, other):
        return RF(other) / self

    def __pow__(self, power):
        if not isinstance(power, int) or abs(power) > 16:
            raise ValueError('small integer power required')
        if power < 0:
            return (RF(1)/self)**(-power)
        out = RF(1)
        for _ in range(power):
            out *= self
        return out

    def __eq__(self, other):
        try:
            b = RF(other)
        except (TypeError, ValueError):
            return False
        return self.num == b.num and self.den == b.den

    def __hash__(self):
        return hash((self.num, self.den))

    def evaluate(self, value):
        value = F(value)
        def horner(coeff):
            out = F(0)
            for v in reversed(coeff):
                out = out*value+v
            return out
        return horner(self.num)/horner(self.den)

    def compose(self, value):
        value = RF(value)
        def horner(coeff):
            out = RF(0)
            for v in reversed(coeff):
                out = out*value+v
            return out
        return horner(self.num)/horner(self.den)

    def json(self):
        return {'numerator_ascending': list(map(str, self.num)),
                'denominator_ascending': list(map(str, self.den))}


def evaluate_ast(node, names):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return RF(node.value)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate_ast(node.operand, names)
    if isinstance(node, ast.BinOp):
        a = evaluate_ast(node.left, names)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int:
                raise ValueError('literal integer exponent only')
            return a**node.right.value
        b = evaluate_ast(node.right, names)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult): return a*b
        if isinstance(node.op, ast.Div): return a/b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'F':
        if not node.keywords and 1 <= len(node.args) <= 2:
            values = [evaluate_ast(a, names) for a in node.args]
            return values[0] if len(values) == 1 else values[0]/values[1]
    raise ValueError('unsupported expression; never eval/import source')


def parse(expression, **names):
    return evaluate_ast(ast.parse(expression, mode='eval').body, names)


def rref(matrix, coefficient_columns):
    a = [[RF(v) for v in row] for row in matrix]
    pivots, row = [], 0
    for col in range(coefficient_columns):
        found = next((i for i in range(row, len(a)) if a[i][col] != 0), None)
        if found is None:
            continue
        a[row], a[found] = a[found], a[row]
        pivot = a[row][col]
        a[row] = [v/pivot for v in a[row]]
        for i in range(len(a)):
            if i != row:
                factor = a[i][col]
                a[i] = [v-factor*w for v, w in zip(a[i], a[row])]
        pivots.append(col)
        row += 1
        if row == len(a): break
    return a, pivots


def solve_many(matrix, rhs):
    n = len(matrix[0])
    rows, pivots = rref([a+b for a, b in zip(matrix, rhs)], n)
    if len(pivots) != n:
        raise ArithmeticError('not a unique symbolic solve')
    return [row[n:] for row in rows[:n]]


def determinant(matrix):
    size = len(matrix)
    if not 1 <= size <= 4 or any(len(row) != size for row in matrix):
        raise ValueError('square symbolic determinant of size 1..4 required')
    total = RF(0)
    for permutation in itertools.permutations(range(size)):
        inversions = sum(permutation[i] > permutation[j] for i in range(size) for j in range(i+1, size))
        product = RF(-1 if inversions % 2 else 1)
        for i, j in enumerate(permutation): product *= matrix[i][j]
        total += product
    return total


def sign_proof(value, domain):
    """Exact sufficient sign certificate, never a sampled sign assertion."""
    value = RF(value)
    def coefficients(poly):
        if domain == 'positive': return list(poly)
        if domain != 'unit': raise ValueError('unsupported domain')
        degree = len(poly)-1
        return [sum((poly[i]*F(comb(k, i), comb(degree, i))
                     for i in range(k+1)), F(0)) for k in range(degree+1)]
    def sign(coeff):
        if all(v == 0 for v in coeff): return 0
        if all(v >= 0 for v in coeff): return 1
        if all(v <= 0 for v in coeff): return -1
        return None
    n, d = coefficients(value.num), coefficients(value.den)
    sn, sd = sign(n), sign(d)
    result = None if sn is None or sd in (None, 0) else sn*sd
    return {'sign': result, 'domain': domain, 'method': 'power-cone' if domain == 'positive' else 'Bernstein-cone',
            'numerator_certificate': list(map(str, n)), 'denominator_certificate': list(map(str, d))}
