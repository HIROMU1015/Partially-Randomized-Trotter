"""UNEXECUTED future correctness: exact rational 2x2 physical-word enumeration.

No binary64 matrix/mean/FWHT helpers used in reference construction. Analytic
Taylor remainder retained. This is a mandatory in-run test, not executed now.
"""
from fractions import Fraction as Q
from itertools import product
import math
from .control import require_permit


def verify_future(permit, kernels):
    require_permit(permit)
    def add(a, b):return (a[0]+b[0], a[1]+b[1])
    def mul(a, b):return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
    def scale(a, c):return [[mul(x, c) for x in row] for row in a]
    def madd(a, b):return [[add(x, y) for x, y in zip(ar, br)] for ar, br in zip(a, b)]
    def mm(a, b):return [[add(mul(a[i][0], b[0][j]), mul(a[i][1], b[1][j])) for j in range(2)] for i in range(2)]
    one = [[(Q(1), Q(0)), (Q(0), Q(0))], [(Q(0), Q(0)), (Q(1), Q(0))]]
    X = [[(Q(0), Q(0)), (Q(1), Q(0))], [(Q(1), Q(0)), (Q(0), Q(0))]]
    Z = [[(Q(1), Q(0)), (Q(0), Q(0))], [(Q(0), Q(0)), (Q(-1), Q(0))]]
    HD = scale(madd(X, Z), (Q(1, 4), Q(0)))
    delta = Q(1, 8); R = 2; steps = 2
    probabilities = [Q(1, 3), Q(2, 3)]
    paulis = [X, Z]
    degree = 32
    def exponential(h, t):
        value = one; term = one
        for j in range(1, degree+1):
            term = scale(mm(term, h), (Q(0), t/j))
            value = madd(value, term)
        return value
    def branch(labels, sign, method):
        U = one
        r = R if method in ['shared_pm', 'ordinary'] else R//2
        dtime = delta/2 if method == 'ordinary' else delta/4
        angle = delta/R if method in ['ordinary', 'independent_pm'] else delta/(2*R)
        D = exponential(HD, sign*dtime)
        for step in range(steps):
            word = one
            for label in labels[step*r:(step+1)*r]:
                word = mm(exponential(paulis[label], sign*angle), word)
            U = mm(mm(mm(D, word), D), U)
        return U
    references = {}; paths = 0
    for method in ['shared_pm', 'ordinary', 'independent_pm']:
        total = (Q(0), Q(0))
        for labels in product(range(2), repeat=R*steps):
            paths += 1
            if paths > permit.config['resources']['max_word_reference_paths']:
                raise RuntimeError('word reference path cap')
            probability = math.prod(probabilities[j] for j in labels)
            if method == 'ordinary':
                amplitude = branch(labels, 1, method)[0][0]
            else:
                plus = labels if method == 'shared_pm' else labels[:R*steps//2]
                minus = labels if method == 'shared_pm' else labels[R*steps//2:]
                Uplus = branch(plus, 1, method); Uminus = branch(minus, -1, method)
                amplitude = add(mul((Uminus[0][0][0], -Uminus[0][0][1]), Uplus[0][0]), mul((Uminus[1][0][0], -Uminus[1][0][1]), Uplus[1][0]))
            total = add(total, mul((probability, Q(0)), amplitude))
        # All gate generator infinity norms <=1/8. Max16 gates in two words.
        local = Q(1, 8)**33/Q(math.factorial(33))*Q(8, 7)
        error_l1 = 2*16*local*(1+local)**16
        for value in [*total,error_l1]:
            if max(abs(value.numerator).bit_length(),value.denominator.bit_length()) > permit.config['resources']['max_exact_integer_bits']:
                raise RuntimeError('word reference exact bit cap')
        references[method] = total, error_l1
    np = kernels.np
    from .numerics import Ball
    hd = Ball(np.array([[.25, .25], [.25, -.25]], dtype=np.complex128), 0.0, .5)
    h = Ball(np.array([[2/3, 1/3], [1/3, -2/3]], dtype=np.complex128), kernels.rational_up(4*abs(Q(1/3)-Q(1, 3))), 1.0)
    plan = kernels.twirl_plan({'X': Q(1, 3), 'Z': Q(2, 3)}, 2)
    probe = Ball(np.array([[1, 2j], [3, -1j]], dtype=np.complex128), 0.0, 8.0)
    transformed = kernels.twirl(probe, plan)
    px = np.array([[0, 1], [1, 0]], dtype=np.complex128); pz = np.diag([1., -1.]).astype(np.complex128)
    direct = kernels.add(kernels.scale(kernels.mul(kernels.mul(Ball(px,0,1),probe),Ball(px,0,1)),1/3,kernels.rational_up(abs(Q(1/3)-Q(1,3)))), kernels.scale(kernels.mul(kernels.mul(Ball(pz,0,1),probe),Ball(pz,0,1)),2/3,kernels.rational_up(abs(Q(2/3)-Q(2,3)))))
    if kernels.norm(transformed.m-direct.m) > transformed.e+direct.e:
        raise RuntimeError('correctness FAIL: Walsh/direct twirl')
    d = kernels.exp_i(hd, delta/4); do = kernels.exp_i(hd, delta/2)
    shared = kernels.eye(2)
    for _ in range(steps):shared = kernels.shared(shared, d, h, delta/(2*R), R, plan)
    ordinary = kernels.power(kernels.mul(kernels.mul(do,kernels.power(kernels.mean(h,delta/R),R)),do),steps)
    v = kernels.mul(kernels.mul(d,kernels.power(kernels.mean(h,delta/R),R//2)),d)
    independent = kernels.mul(kernels.power(v,steps),kernels.power(v,steps))
    results = {}
    state = np.array([1.,0.],dtype=np.complex128)
    for method, matrix in [('shared_pm',shared),('ordinary',ordinary),('independent_pm',independent)]:
        z, radius = kernels.signal(matrix,state,Q(1))
        exact, error = references[method]
        distance_l1 = abs(Q(z.real)-exact[0])+abs(Q(z.imag)-exact[1])
        if distance_l1 > Q(radius)+error:
            raise RuntimeError('correctness FAIL: enumerated physical words '+method)
        results[method] = {'status':'PASS','reference':'exact rational enumerated words with Taylor remainder','binary64_radius':radius,'reference_remainder_num':str(error.numerator),'reference_remainder_den':str(error.denominator)}
    return {'status':'PASS','Walsh_direct_twirl':'PASS','enumerated_words':results,'R':R,'steps':steps,'delta':'1/8'}
