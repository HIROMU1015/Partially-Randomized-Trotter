#!/usr/bin/env python3
"""Post-hoc scalar budget diagnosis from frozen saved data, never Gate0 execution.

stdlib only. Decode four saved NPZ arrays for exact input L1 scales, read saved
Pauli coefficient hex strings, replay/expand scalar error formulas with Fraction.
No frozen module import, permit fabrication, new coefficient, matrix, state or
W/v/rho evaluation. Counterfactual budgets are unadopted diagnostic scenarios.
"""
import ast
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PLAN = 'artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/diagnostic_contract.json'
OUTPUT = 'artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/diagnostic_result.json'
U = F(1, 2**53)


def gamma(n):
    assert n * U < 1
    return n * U / (1 - n * U)


def up(x):
    value = float(x)
    if F.from_float(value) < x:
        value = math.nextafter(value, math.inf)
    assert math.isfinite(value)
    return value


def textnum(x):
    with localcontext() as ctx:
        ctx.prec = 24
        return str(Decimal(x.numerator) / Decimal(x.denominator))


def serial(x):
    if isinstance(x, F):
        return textnum(x)
    if isinstance(x, dict):
        return {k: serial(v) for k, v in x.items()}
    if isinstance(x, list):
        return [serial(v) for v in x]
    return x


def coefficient_alpha(scale, count, global_M=65536):
    return 2 * gamma(24 * count + 64) * scale * (1 + gamma(global_M + 32))


def law_parts(L, lam, alpha, M):
    gd, gv, gn = gamma(8 * 256 + 32), gamma(8), gamma(8 * M + 32)
    ds = gamma(M + 32) * L
    dl = alpha + ds
    lower = lam - dl
    if lower <= 0:
        return {'lambda_lower': lower, 'status': 'DENOMINATOR_UNRESOLVED'}
    ec_parts = {'coefficient_error': alpha / lower,
                'normalization_from_coefficient_error': alpha * L / (lower * lam),
                'normalization_from_lambda_summation': ds * L / (lower * lam)}
    ec = sum(ec_parts.values())
    assembly = 16 * gn * L
    y_parts = {'normalized_coefficient_error': ec, 'matrix_assembly': assembly / lam,
               'matvec': gd * (16 * L + assembly) / lam,
               'division': gv * (1 + gd) * (16 * L + assembly) / lam}
    y = sum(y_parts.values())
    A = (1 + y)**2
    quotient = (2 * gd + gv * (1 + gd)) * A / (1 - gd)
    Qup = (1 + gv) * (1 + gd) * A / (1 - gd)
    W_parts = {'linear_' + key: 2 * value for key, value in y_parts.items()}
    W_parts.update({'quadratic_y_squared': y * y, 'dot_and_quotient': quotient,
                    'final_subtraction': U * (1 + Qup)})
    ew = sum(W_parts.values())
    ratio = L / lam
    ep = ec + gv * ratio
    defect = (gamma(M + 32) + gv) * ratio
    ea = (2 * gd + gv * (1 + gd)) / (1 - gd)
    gm = gamma(4 * M + 32)
    em = ep + (1 + defect) * ea + gm * (1 + defect) * (1 + ea)
    am = (1 + gm) * (1 + defect) * (1 + ea)
    v_parts = {'direct_law': 4 * ep,
               'center_law': 4 * (1 + defect) * ep,
               'expectation_and_center_quotients': 4 * (1 + defect) * (2 + defect) * ea,
               'mean_summation': 4 * gm * (1 + defect)**2 * (1 + ea),
               'center_quadratic': (1 + defect) * (ea + em)**2,
               'centered_variance_arithmetic': gamma(16 * M + 32) * (1 + defect) * (1 + ea + am)**2}
    ev = sum(v_parts.values())
    # Independent unexpanded SOURCE expression agrees exactly with the expansion.
    assert ev == 4 * ep + (1 + defect) * (4 * (ea + em) + (ea + em)**2) + gamma(16 * M + 32) * (1 + defect) * (1 + ea + am)**2
    assert ew == 2 * y + y * y + quotient + U * (1 + Qup)
    return {'status': 'SCALAR_BUDGET_REPLAY_ONLY', 'alpha': alpha, 'lambda_sum_error': ds,
            'lambda_error': dl, 'lambda_lower': lower, 'alpha_over_lambda': alpha / lam,
            'L_over_lambda': ratio, 'ec': ec, 'ec_parts': ec_parts, 'y': y,
            'y_parts': y_parts, 'ep': ep, 'ea': ea, 'W_parts': W_parts, 'v_parts': v_parts,
            'eW': ew, 'ev': ev,
            'W_percent': {k: 100 * v / ew for k, v in W_parts.items()},
            'v_percent': {k: 100 * v / ev for k, v in v_parts.items()}}


def saved_npy(raw, expected_descr, expected_shape):
    stream = io.BytesIO(raw)
    assert stream.read(6) == b'\x93NUMPY'
    version = tuple(stream.read(2))
    assert version in {(1, 0), (2, 0)}
    header_size = struct.unpack('<H' if version == (1, 0) else '<I', stream.read(2 if version == (1, 0) else 4))[0]
    assert header_size <= 4096
    header = ast.literal_eval(stream.read(header_size).decode('latin1'))
    assert header == {'descr': expected_descr, 'fortran_order': False, 'shape': expected_shape}, header
    data = stream.read()
    count = math.prod(expected_shape)
    components = 2 if expected_descr == '<c16' else 1
    assert expected_descr in {'<c16', '<f8'} and len(data) == count * components * 8
    scalars = [F.from_float(x[0]) for x in struct.iter_unpack('<d', data)]
    assert len(scalars) == count * components
    return scalars


def main():
    plan_raw = (ROOT / PLAN).read_bytes()
    plan = json.loads(plan_raw)
    assert plan['record_kind'] == 'SAVED_DATA_SCALAR_DIAGNOSTIC_NOT_GATE0_AUTHORIZATION'
    assert plan['output'] == OUTPUT
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['diagnostic_source_sha256']
    resource.setrlimit(resource.RLIMIT_AS, (plan['memory_bytes'], plan['memory_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE, (plan['output_bytes'], plan['output_bytes']))
    identities = []
    def read(path, expected):
        raw = (ROOT / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, path
        identities.append({'path': path, 'bytes': len(raw), 'sha256': expected})
        return raw
    result_raw = read(plan['saved_result'], plan['saved_result_sha256'])
    r = json.loads(result_raw)
    input_raw = read(plan['saved_input'], plan['saved_input_sha256'])
    contract_raw = read(plan['adopted_contract'], plan['adopted_contract_sha256'])
    config = json.loads(contract_raw)
    assert r['config'] == config and r['status'] == 'TECHNICAL_STOP'
    with zipfile.ZipFile(io.BytesIO(input_raw)) as archive:
        assert all(info.file_size <= 65536 for info in archive.infolist())
        c = saved_npy(archive.read('constant.npy'), '<f8', ())[0]
        h = saved_npy(archive.read('one_body.npy'), '<c16', (8, 8))
        ls = saved_npy(archive.read('lambdas.npy'), '<f8', (12,))
        gs = saved_npy(archive.read('g_matrices.npy'), '<c16', (12, 8, 8))
    G = [sum(abs(x) for x in gs[i * 128:(i + 1) * 128]) for i in range(12)]
    Fh = sum(abs(x) for x in h)
    blocks = [abs(lam) * g * g for lam, g in zip(ls, G)]
    S = abs(c) + Fh + sum(blocks)
    C, M = config['limits']['max_product_contributions'], config['limits']['max_pauli_terms']
    alpha = coefficient_alpha(S, C, M)
    assert up(S) == r['formula_resolution_before_metrics']['input_scale_H_l1_upper']
    assert up(alpha) == r['formula_resolution_before_metrics']['coefficient_l1_error_upper']
    assert plan['C_cases'] == [C, 706837, r['resources']['product_contributions']]
    rows = []
    checks = ['raw input/result/contract identities', 'exact full S matches saved upward value', 'exact alpha matches saved upward value']
    for row in r['rows']:
        n = int(row['split'][1:])
        assert all(row[k] is None for k in ('W', 'v', 'rho'))
        coefficients = [F.from_float(float.fromhex(x['coefficient_hex'])) for x in row['canonical_table']]
        L = sum(abs(x) for x in coefficients)
        lam = F.from_float(row['lambda_R'])
        tail = sum(blocks[n:])
        baseline = law_parts(L, lam, alpha, M)
        saved = row['error_ledger']['formula_resolution']
        for key, diag in [('W_error_budget', 'eW'), ('v_error_budget', 'ev')]:
            assert up(baseline[diag]) == saved['resolved_law_limits'][key], (row['split'], key)
        for key, diag in [('lambda_absolute_error_upper', 'lambda_error'), ('normalized_law_L1_error_upper', 'ep'), ('Pauli_expectation_error_upper', 'ea')]:
            assert up(baseline[diag]) == saved[key], (row['split'], key)
        checks.append(row['split'] + ': five saved scalar budgets exactly replayed and additive decompositions exact')
        cases = []
        for case in plan['cases']:
            scale = S if case['scale'] == 'full_S' else tail
            a = coefficient_alpha(scale, case['C'], M) if case['alpha'] == 'gamma_scale' else F(0)
            law_M = M if case['law_M'] == 'cap' else len(coefficients)
            value = law_parts(L, lam, a, law_M)
            cases.append({'id': case['id'], 'C': case['C'], 'law_M': law_M, 'scale': scale,
                          'alpha': a, 'eW': value.get('eW'), 'ev': value.get('ev'),
                          'W_ceiling_ratio': value['eW'] / F.from_float(config['precision']['eps_W']) if 'eW' in value else None,
                          'v_ceiling_ratio': value['ev'] / F.from_float(config['precision']['eps_v']) if 'ev' in value else None,
                          'illustrative_budget_sufficient': value.get('eW', F(1)) <= F.from_float(config['precision']['eps_W']) and value.get('ev', F(1)) <= F.from_float(config['precision']['eps_v']),
                          'is_validated_error_bound': False, 'formal_contract_changed': False})
        rows.append({'split': row['split'], 'lambda_R': lam, 'canonical_event_count': len(coefficients),
                     'L_exact_sum': L, 'tail_input_scale': tail, 'full_S_over_tail_S': S / tail,
                     'tail_S_over_lambda': tail / lam, 'baseline': baseline, 'cases': cases})
    report = {'schema': 'm1s_gate0_saved_precision_cause_diagnostic_v1',
              'diagnostic_contract_sha256': hashlib.sha256(plan_raw).hexdigest(),
              'evidence': 'POST_HOC_SAVED_DATA_BUDGET_ANALYSIS_NOT_ERROR_CERTIFICATE',
              'identities': identities, 'full_scale': S, 'full_scale_components': {'abs_constant': abs(c), 'one_body_L1': Fh, 'DF_blocks': blocks},
              'global_alpha': alpha, 'gamma_coefficient_count': 24 * C + 64,
              'rows': rows, 'successful_replay_checks': checks, 'NPZ_numeric_members_decoded': ['constant', 'one_body', 'lambdas', 'g_matrices'],
              'state_decodes': 0, 'Pauli_coefficient_generation': 0, 'matrix_operations': 0,
              'W_v_rho_evaluations': 0, 'Gate0_runner_invocations': 0, 'contract_changes': 0,
              'conditional_model_certified': False, 'scientific_GO_or_no_GO_determined': False,
              'mandatory_STOP': True}
    raw = (json.dumps(serial(report), indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    assert len(raw) <= plan['output_bytes']
    with (ROOT / OUTPUT).open('xb') as stream:
        stream.write(raw)
    for item in identities:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
    print(json.dumps({'output': OUTPUT, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows),
                      'baseline_replay_checks': len(checks), 'saved_data_scalar_diagnostic': True,
                      'new_Gate0_or_W_v_rho_runs': 0, 'mandatory_STOP': True}))


if __name__ == '__main__':
    main()
