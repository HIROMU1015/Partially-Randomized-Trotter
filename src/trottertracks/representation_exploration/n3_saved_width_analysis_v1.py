"""Dormant authorized posthoc evaluator using only already-saved witnesses."""
from fractions import Fraction as F
import resource
import time

from .n3_occupation_transfer_check_v1 import Checker, hash_value
from .n3_saved_width_v1 import Arithmetic, check_interval, compare, encode, interval, minimum


def evaluate_case(case, primary, secondary, source_fingerprint, primary_seal):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    start, cpu = time.perf_counter(), time.process_time()
    checker = Checker()
    checker.audit_case(case, primary, source_fingerprint)
    primary_cache = checker.cache
    checker.secondary(case, primary, secondary, primary_seal)
    # A secondary separator has its own proof pool; restore the primary cache.
    checker.pool, checker.cache = primary['proof_pool'], primary_cache
    h, _, active, d, nu, _ = checker.decode(case)
    A = [[h[i][j] for j in active] for i in active]
    v = [h[i][d] for i in active]
    U, c = checker.f(primary['upper']['U']), checker.f(primary['bounds']['c_star'])
    g = checker.sub(c, U)
    if g <= 0: raise ValueError('COMMON_BOUND_PRECONDITION')
    a = Arithmetic()
    candidates, coverage = {}, {'raw_attempts': 0, 'invalid_PSD_attempts': 0, 'OCC_spectral_entries': 0,
                                'invalid_spectral_entries': 0, 'missing_spectral_entries': 0}

    def add(kind, kappa, s, origin, matrix_parameters, selected=False):
        identity = hash_value({'input': case['id'], 'kind': kind, 'kappa': encode(kappa), 's': encode(s), 'parameters': matrix_parameters})
        if identity in candidates:
            candidates[identity]['origins'].append(origin)
            candidates[identity]['selected_by_generator'] |= selected
            return
        if len(candidates) >= 512: raise ValueError('FIXED_WIDTH_CANDIDATE_CAP')
        q = a.sub(a.mul(kappa, U), s)
        width = interval(g, kappa, q, a)
        check_interval(g, kappa, q, width)
        candidates[identity] = {'id': identity, 'kind': kind, 'kappa': encode(kappa), 's': encode(s), 'q': encode(q),
                                'width': width, 'origins': [origin], 'parameters': matrix_parameters,
                                'selected_by_generator': selected}

    for ri, row in enumerate(primary['raw_rows']):
        for mi, method in enumerate(row['methods']):
            for ai, witness in enumerate(method['attempts']):
                coverage['raw_attempts'] += 1
                ref = {'row': row['row'], 'method': method['method'], 'attempt': ai,
                       'pointer': f'/raw_rows/{ri}/methods/{mi}/attempts/{ai}',
                       'original_delta_goal': row['delta'], 'original_accepted': witness.get('accepted', False)}
                if witness['kind'] == 'norm':
                    beta2 = checker.total(checker.mul(x, x) for x in v)
                    add('NORM', F(0), -beta2, ref, {'definition': 'saved coefficient norm-only identity'})
                elif witness['kind'] == 'old':
                    if not witness.get('PSD_valid'):
                        coverage['invalid_PSD_attempts'] += 1; continue
                    kappa, mu = checker.f(witness['kappa']), checker.f(witness['mu'])
                    M = checker.plus(checker.shift(checker.scale(A, kappa), mu), checker.scale(checker.outer(v), -1))
                    if not checker.proof(M, witness['proof'])[0]: raise ValueError('SAVED_PSD_FLAG_MISMATCH')
                    add('OLD', kappa, checker.mul(nu, mu), ref, {'mu': witness['mu'], 'proof': witness['proof']})
                else:
                    kappa = checker.f(witness['kappa'])
                    K = checker.plus(checker.scale(A, kappa), checker.scale(checker.outer(v), -1))
                    spectra = list(enumerate(witness.get('spectral_attempts', [])))
                    if witness.get('spectral') is not None: spectra.append(('selected', witness['spectral']))
                    if not spectra: coverage['missing_spectral_entries'] += 1
                    for si, spectral in spectra:
                        coverage['OCC_spectral_entries'] += 1
                        lower, valid = checker.spectrum(K, nu, spectral)
                        if not valid:
                            coverage['invalid_spectral_entries'] += 1; continue
                        sr = dict(ref, spectral_entry=si)
                        sr['pointer'] += '/spectral' if si == 'selected' else '/spectral_attempts/' + str(si)
                        add('OCC', kappa, lower, sr, spectral, si == 'selected')
    all_candidates = list(candidates.values())
    groups = {'BASE': [r for r in all_candidates if r['kind'] in ('NORM', 'OLD')],
              'NORM': [r for r in all_candidates if r['kind'] == 'NORM'],
              'FIVE': [r for r in all_candidates if any(o['method'] == 'FIVE' for o in r['origins'])],
              'OLD_DIRECT': [r for r in all_candidates if any(o['method'] == 'OLD_DIRECT' for o in r['origins'])],
              'OCC_RAW_ALL_SAVED_SPECTRA': [r for r in all_candidates if r['kind'] == 'OCC'],
              'OCC_SELECTED_SPECTRA': [r for r in all_candidates if r['kind'] == 'OCC' and r['selected_by_generator']]}
    widths = {name: minimum(rows) for name, rows in groups.items()}
    outcome = compare(widths['OCC_RAW_ALL_SAVED_SPECTRA'], widths['BASE'])
    return {'input_id': case['id'], 'base': case['base'], 'case_sha256': primary['case_sha256'],
            'source_fingerprint': source_fingerprint, 'primary_seal': primary_seal,
            'U': encode(U), 'c_star': encode(c), 'g': encode(g), 'widths': widths, 'comparison': outcome,
            'coverage': coverage, 'candidates': all_candidates,
            'fallback_or_transported_candidates_in_OCC_specific_pool': 0,
            'resources': {'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu,
                          'checker_conditions': checker.checks, 'checker_unique_PSD_matrices': len(checker.cache),
                          'checker_max_rational_bits': checker.max_bits, 'width_arithmetic_operations': a.operations,
                          'width_max_rational_bits': a.max_rational_bits, 'width_max_scaled_integer_bits': a.max_integer_bits,
                          'peak_worker_rss_bytes_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
            'optimizer_calls': 0, 'numerical_eigensolver_calls': 0, 'new_certificate_proposals': 0,
            'interpretation': 'posthoc implications of a finite saved witness set, not family optimum or truth',
            'original_primary_result_changed': False, 'mandatory_stop': True}
