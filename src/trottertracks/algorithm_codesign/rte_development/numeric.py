"""Generic small-matrix diagnostics; never fixes a statistical budget."""
import numpy as np


def finite_target(operators, p, x, degree=5):
    R = sum(float(a)*q for a,q in zip(p, operators))
    out = np.eye(len(R), dtype=complex); term = out.copy()
    for n in range(1, degree+1):
        term = term @ (-1j*float(x)*R)/n
        out = out + term
    return out


def event_operator(e, operators, pauli=None):
    identity = np.eye(len(operators[0]), dtype=complex)
    phi = np.arctan(float(e['ratio']))
    if 'pauli' in e:
        Q = pauli(e['pauli'])
        out = (Q if not e['rotation_sign'] else
               np.cos(phi)*identity-1j*e['rotation_sign']*np.sin(phi)*Q)
    else:
        out = identity
        for i in e['word']: out = out @ operators[i]
        out = (np.cos(phi)*identity-1j*np.sin(phi)*operators[e['child']]) @ out
    return (1j**e['phase_i_power'])*out


def controlled(u):
    out = np.zeros((2*len(u), 2*len(u)), dtype=complex)
    out[::2,::2] = np.eye(len(u)); out[1::2,1::2] = u
    return out


def strict_error(actual, wanted):
    return float(np.linalg.norm(actual-wanted, 2))


class NativeReference:
    """One immutable provider and shared cached native matrices after launch."""
    def __init__(self, cache):
        from .. import g9_matrix as old
        self.old, self.cache = old, cache
        self.operators = old.Q_matrices()
        self.rotation_cache, self.circuit_cache = {}, {}

    def actual(self, gates, realized):
        key = (realized, tuple(tuple(g) for g in gates))
        if key in self.circuit_cache: return self.circuit_cache[key]
        out = np.eye(16, dtype=complex)
        for g in gates:
            if g[0] == 'R':
                k = (realized, g[1], g[2], g[3])
                if k not in self.rotation_cache:
                    if realized: u = self.old.sequence(self.cache[g[2]]['sequence'])
                    else:
                        from fractions import Fraction
                        theta = np.arctan(float(Fraction(g[2])))
                        u = np.diag([np.exp(-1j*theta/2),np.exp(1j*theta/2)])
                    if g[3] == -1: u = u.conj().T
                    self.rotation_cache[k] = self.old.expand(u, g[1], 4)
                gate = self.rotation_cache[k]
            else: gate = self.old.fixed(tuple(g), 4)
            out = gate @ out
        if len(self.circuit_cache) >= 1024: self.circuit_cache.clear()
        self.circuit_cache[key] = out
        return out

    def evaluate(self, e, gates, analytic_error, tolerance):
        u = event_operator(e, self.operators, self.old.pauli)
        wanted = controlled(u)
        ideal = strict_error(self.actual(gates, False), wanted)
        actual = strict_error(self.actual(gates, True), wanted)
        if ideal > tolerance or actual > float(analytic_error)+tolerance:
            raise ArithmeticError('strict controlled phase/order/operator guard failed')
        return u, dict(ideal_error_diagnostic=ideal, actual_error_diagnostic=actual,
                       up_to_phase=False, diagnostics_do_not_set_bias=True)
