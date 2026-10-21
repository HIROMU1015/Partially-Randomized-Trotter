"""Independent MP oracle v3: v2 arithmetic/stages, cell-local operator reuse.

No cross-cell or cross-precision sharing. Raw division operates on copies.
The original frozen v2 implementation is retained for synthetic regression.
"""
from __future__ import annotations
import math
import numpy as np
from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_independent_reference import occupation_df_matrix
from .ax2b_stage_validation_v2 import canonical_cell, traced_signal
from .ax2b_mp_cache_v1 import OperatorCache, oracle_identity

def mp_cell(ham, basis, saved_state, cell, *, T, scalar, lambda_r=0.,
            extracted_identity=0., dps=80, progress=None):
    """Independent occupation matrices and forward Taylor recurrence (not Horner).

    H4 small-sector port only. At most 64-dimensional matrices, no full-space
    matrices. All MP stages and inner products remain MP until serialization.
    Executed times/normalization are distinguished from mathematical MP values.
    """
    if type(dps) is not int or dps not in (80, 120) or len(basis) > 64:
        raise ValueError('SMALL_MP_POLICY')
    import mpmath as mp
    with mp.workdps(dps):
        identity = oracle_identity(ham, basis, saved_state, cell, T=T, scalar=scalar,
                 lambda_r=lambda_r, extracted_identity=extracted_identity, dps=dps, backend=mp.__version__)
        cache = OperatorCache(identity, progress=progress, mp_precision=mp.mp.prec)
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
            cache.stats['logical_stage_actions'] = len(trace)
            return after
        def evolve(v, i, t, path, label):
            u = cache.get(('deterministic', i, float(t).hex()),
                          lambda: mp.expm(-mp.j*exact(t)*deterministic[i]))
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
                            u = cache.get(('exact_tail',), lambda: mp.expm(-mp.j*tau*htail))
                        else:
                            def polynomial():
                                power, u = mp.eye(len(basis)), mp.eye(len(basis))
                                for degree in range(1, cell['K']+2):
                                    power = (-mp.j*tau/degree)*htail*power
                                    u += power
                                return u
                            u = cache.get(('finite_corrected',), polynomial, kind='polynomial')
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
                if progress is not None:
                    progress({'point':'outer_completed','path':path,'last_stage':trace[-1]['stage'],
                              'oracle_work':cache.report()})
            result[path] = signal(current)
        result['reference'] = signal(cache.get(('reference', float(T).hex()), lambda: mp.expm(-mp.j*exact(T)*target))*psi)
        if cell['method'] == 'B0':
            truncated = exact(ham.constant)*mp.eye(len(basis))+sum(deterministic,mp.zeros(len(basis)))
            result['exact_truncated'] = signal(cache.get(('exact_truncated', float(T).hex()), lambda: mp.expm(-mp.j*exact(T)*truncated))*psi)
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
        work = cache.report()
        work.update(logical_stage_actions=len(trace),
                    state_matrix_products=len(trace)-cell['q']*(3 if random else 1)+1+(1 if cell['method'] == 'B0' else 0),
                    polynomial_matrix_products=(cell['K']+1)*work['completed']['polynomial'] if random else 0)
        return {'oracle_work': work, 'dps': dps, 'signals': result, 'signed_bias_decomposition':decomposition, 'trace': trace,
                'b': text(b), 'log_B': text(cell['q']*cell['r']*mp.log(b) if random else 0),
                'state_norm_before': text(norm), 'evidence_kind': 'EMPIRICAL', 'certified': False,
                'coefficient_policy': 'exact integer-ratio lifts of saved binary64; executed binary64 times',
                'normalization_policy': 'initial only; no stage rescaling'}
