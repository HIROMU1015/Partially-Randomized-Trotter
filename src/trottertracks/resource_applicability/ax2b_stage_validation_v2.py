"""Explicit stage records and an independent small high-precision oracle.

No input I/O, sampler or compiler. Intermediate finite means are not normalized.
MP arithmetic uses integer-ratio lifts of the executed binary64 coefficients
and times. Precision/path differences are empirical diagnostics, not bounds.
"""
from __future__ import annotations

import math
import numpy as np

from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_independent_reference import occupation_df_matrix
from .ax2a_state_action import finite_taylor_action


def canonical_cell(cell):
    value = dict(cell)
    value['formula'] = cell.get('formula', cell.get('order'))
    value['r'] = cell['R'] // cell['q'] if cell.get('R') else None
    return value


def traced_signal(state, actions, tail, *, cell, T, scalar, lambda_r, budget):
    """Existing native callbacks/Horner, with bounded per-cell norm records."""
    cell = canonical_cell(cell)
    psi = np.asarray(state, dtype=np.complex128).copy()
    if psi.ndim != 1 or not np.isfinite(psi).all() or abs(np.linalg.norm(psi)-1) > 1e-12:
        raise ValueError('SAVED_STATE_NORMALIZATION')
    schedule = primitive_time_schedule(cell, T=T)
    trace = []
    def record(path, label, before, after, time=None):
        if len(trace) >= 4096:
            raise RuntimeError('STAGE_RECORD_CAP')
        if not np.isfinite(after).all():
            raise ValueError('NONFINITE_STAGE')
        trace.append({'path': path, 'stage': label, 'time': time,
                      'norm_before': float(np.linalg.norm(before)),
                      'norm_after': float(np.linalg.norm(after)),
                      'state_after': [{'real':float(z.real),'imag':float(z.imag)} for z in after]})
        return after
    random = cell['method'] in ('B2', 'B3')
    log_B, b = 0., 1.
    if random:
        from trotterlib.rte import finite_rte_distribution
        if lambda_r <= 0 or tail is None:
            raise ValueError('EMPTY_REGISTERED_TAIL')
        tau = lambda_r * (T / cell['q']) / cell['r']
        b = finite_rte_distribution(tau, cell['K']).exact_finite_distribution
        log_B = cell['q'] * cell['r'] * math.log(b)
    signals, vectors = {}, {}
    for path in ('corrected', 'raw') if random else ('corrected',):
        current = psi.copy()
        for outer in range(cell['q']):
            if random:
                sequence = schedule['ordinary_one_outer_step']
                for i, t in sequence[:len(actions)]:
                    budget.deterministic()
                    current = record(path, f'{outer}:forward:{i}', current, actions[i](current, t), t)
                for micro in range(cell['r']):
                    degree = cell['K'] + 1
                    def counted(v):
                        nonlocal degree
                        result = tail(v)
                        record(path, f'{outer}:micro:{micro}:horner_matvec:{degree}', v, result)
                        degree -= 1
                        return result
                    before = current
                    current = finite_taylor_action(counted, current, tau, cell['K'], budget=budget)
                    current = record(path, f'{outer}:micro:{micro}', before, current)
                    if path == 'raw':
                        current = record(path, f'{outer}:micro:{micro}:divide_b', current, current/b)
                for i, t in sequence[len(actions):]:
                    budget.deterministic()
                    current = record(path, f'{outer}:reverse:{i}', current, actions[i](current, t), t)
            else:
                for j, (i, t) in enumerate(schedule['ordinary_one_outer_step']):
                    budget.deterministic()
                    current = record(path, f'{outer}:pf:{j}:{i}', current, actions[i](current, t), t)
            current = record(path, f'{outer}:scalar', current,
                             np.exp(-1j * scalar * T/cell['q']) * current, T/cell['q'])
        signals[path] = complex(np.vdot(psi, current))
        vectors[path] = current
    return {'signals': signals, 'vectors': vectors, 'trace': trace, 'log_B': log_B,
            'b': b, 'intermediate_norm_max': max([1.] + [r['norm_after'] for r in trace]),
            'normalization_policy': 'initial only; no stage rescaling'}


def mp_cell(ham, basis, saved_state, cell, *, T, scalar, lambda_r=0.,
            extracted_identity=0., dps=80):
    """Independent occupation matrices and forward Taylor recurrence (not Horner).

    H4 small-sector port only. At most 64-dimensional matrices, no full-space
    matrices. All MP stages and inner products remain MP until serialization.
    Executed times/normalization are distinguished from mathematical MP values.
    """
    if type(dps) is not int or dps not in (80, 120) or len(basis) > 64:
        raise ValueError('SMALL_MP_POLICY')
    import mpmath as mp
    with mp.workdps(dps):
        def exact(x):
            z = complex(x)
            if not math.isfinite(z.real) or not math.isfinite(z.imag):
                raise ValueError('NONFINITE_MP_INPUT')
            def lift(v):
                a, b = float(v).as_integer_ratio()
                return mp.mpf(a)/b
            return mp.mpc(lift(z.real), lift(z.imag))
        def text(x):
            return mp.nstr(x, dps)
        def signal(v):
            z = sum(mp.conj(psi[i])*v[i] for i in range(len(basis)))
            return {'real': text(z.real), 'imag': text(z.imag)}
        def matrix(constant, one, weights, blocks):
            return mp.matrix(occupation_df_matrix(constant, one, weights, blocks, basis, convert=exact))
        zero = np.zeros_like(ham.one_body)
        terms = [matrix(0., ham.one_body, (), ())]
        terms += [matrix(0., zero, (w,), (g,)) for w, g in zip(ham.lambdas, ham.g_matrices, strict=True)]
        target = exact(ham.constant)*mp.eye(len(basis)) + sum(terms, mp.zeros(len(basis)))
        if mp.norm(target-target.transpose_conj()) > mp.mpf('1e-25'):
            raise ValueError('MP_TARGET_NOT_HERMITIAN')
        psi = mp.matrix([exact(x) for x in saved_state])
        norm = mp.norm(psi)
        if norm == 0 or abs(norm-1) > mp.mpf('1e-12'):
            raise ValueError('SAVED_STATE_NORMALIZATION')
        psi /= norm
        cell = canonical_cell(cell)
        schedule = primitive_time_schedule(cell, T=T)
        deterministic = terms[:cell['prefix']+1]
        random = cell['method'] in ('B2', 'B3')
        delta = T/cell['q']
        tau = exact(lambda_r * delta/cell['r']).real if random else mp.mpf(0)
        b = sum(abs(tau)**k/mp.factorial(k)*mp.sqrt(1+(tau/(k+1))**2)
                for k in range(0, cell['K']+1, 2)) if random else mp.mpf(1)
        htail = ((sum(terms[cell['prefix']+1:], mp.zeros(len(basis)))
                  -exact(extracted_identity)*mp.eye(len(basis)))/exact(lambda_r)) if random else None
        trace, result = [], {}
        def stage(path, label, before, after, operator_norm=None, time=None):
            if len(trace) >= 4096:
                raise RuntimeError('MP_STAGE_RECORD_CAP')
            trace.append({'path': path, 'stage': label, 'time': time,
                          'norm_before': text(mp.norm(before)), 'norm_after': text(mp.norm(after)),
                          'state_after': [{'real':text(z.real),'imag':text(z.imag)} for z in after],
                          'operator_frobenius_norm': text(operator_norm) if operator_norm is not None else None})
            return after
        def evolve(v, i, t, path, label):
            u = mp.expm(-mp.j*exact(t)*deterministic[i])
            return stage(path, label, v, u*v, mp.norm(u), t)
        for path in ('corrected', 'raw', 'exact_tail') if random else ('corrected',):
            current = psi.copy()
            for outer in range(cell['q']):
                if random:
                    seq = schedule['ordinary_one_outer_step']
                    for i,t in seq[:len(deterministic)]:
                        current = evolve(current,i,t,path,f'{outer}:forward:{i}')
                    for micro in range(cell['r']):
                        if path == 'exact_tail':
                            u = mp.expm(-mp.j*tau*htail)
                        else:
                            power, u = mp.eye(len(basis)), mp.eye(len(basis))
                            for degree in range(1, cell['K']+2):
                                power = (-mp.j*tau/degree)*htail*power
                                u += power
                            if path == 'raw':
                                u /= b
                        current = stage(path,f'{outer}:micro:{micro}',current,u*current,mp.norm(u))
                    for i,t in seq[len(deterministic):]:
                        current = evolve(current,i,t,path,f'{outer}:reverse:{i}')
                else:
                    for j,(i,t) in enumerate(schedule['ordinary_one_outer_step']):
                        current = evolve(current,i,t,path,f'{outer}:pf:{j}:{i}')
                current = stage(path,f'{outer}:scalar',current,
                                mp.exp(-mp.j*exact(scalar)*exact(delta))*current,mp.mpf(1),delta)
            result[path] = signal(current)
        result['reference'] = signal(mp.expm(-mp.j*exact(T)*target)*psi)
        if cell['method'] == 'B0':
            truncated = exact(ham.constant)*mp.eye(len(basis))+sum(deterministic,mp.zeros(len(basis)))
            result['exact_truncated'] = signal(mp.expm(-mp.j*exact(T)*truncated)*psi)
        def difference(left,right):
            a,b = result[left],result[right]
            z = mp.mpc(a['real'],a['imag'])-mp.mpc(b['real'],b['imag'])
            return {'real':text(z.real),'imag':text(z.imag)}
        decomposition = {'total':difference('corrected','reference')}
        if cell['method'] == 'B0':
            decomposition.update(discard=difference('exact_truncated','reference'),
                                 PF=difference('corrected','exact_truncated'))
        elif random:
            decomposition.update(finite=difference('corrected','exact_tail'),
                                 outer_PF=difference('exact_tail','reference'))
        else:
            decomposition['PF'] = decomposition['total']
        return {'dps': dps, 'signals': result, 'signed_bias_decomposition':decomposition, 'trace': trace,
                'b': text(b), 'log_B': text(cell['q']*cell['r']*mp.log(b) if random else 0),
                'state_norm_before': text(norm), 'evidence_kind': 'EMPIRICAL', 'certified': False,
                'coefficient_policy': 'exact integer-ratio lifts of saved binary64; executed binary64 times',
                'normalization_policy': 'initial only; no stage rescaling'}
