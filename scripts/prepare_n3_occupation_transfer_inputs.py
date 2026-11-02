#!/usr/bin/env python3
"""Prepare rational coefficient inputs only; no bounds, proposals or truth calls."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/n3_occupation_redesign_preparation/2026-10-12'


def enc(value):
    if isinstance(value, F):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {k: enc(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [enc(v) for v in value]
    return value


def make_inputs():
    bases = [('C1', 6, 2, F(1), 1, F(3, 5), 'chain'),
             ('R1', 7, 3, F(3, 2), 4, F(2, 5), 'ring'),
             ('L1', 8, 3, F(3, 4), 0, F(1, 2), 'ladder'),
             ('S1', 9, 4, F(5, 4), 8, F(1, 3), 'star')]
    cases = []
    for name, m, nu, scale, external, dd, graph in bases:
        active = [i for i in range(m + 1) if i != external]
        edges = set()
        if graph in ('chain', 'ring'):
            edges.update((i, i + 1) for i in range(m - 1))
            if graph == 'ring':
                edges.add((0, m - 1))
        elif graph == 'ladder':
            edges.update((i, i + 2) for i in range(m - 2))
            edges.update((i, i + 1) for i in range(0, m, 2))
        else:
            edges.update((0, i) for i in range(1, m))
        for direction in ('edge_pair', 'distributed_three'):
            h = [[F(0) for _ in range(m + 1)] for _ in range(m + 1)]
            for k, i in enumerate(active):
                h[i][i] = scale * (F((k + 1) ** 2, 10 * m) - F(1, 2) + F((-1) ** k, 37))
            for i, j in sorted(edges):
                h[active[i]][active[j]] = h[active[j]][active[i]] = scale * F((-1) ** (i + j), 18 + (i + j) % 3)
            h[external][external] = scale * dd
            vector = [F(0) for _ in active]
            if direction == 'edge_pair':
                vector[0], vector[-1] = F(3, 5), F(4, 5)
            else:
                for i, z in zip((m // 3, 2 * m // 3, m - 1), (F(2, 3), F(2, 3), F(1, 3))):
                    vector[i] = z
            for i, z in zip(active, vector):
                h[i][external] = h[external][i] = scale * z / 30
            density = []
            for i, j in sorted(edges):
                density.append({'i': active[i], 'j': active[j], 'coefficient': scale * F(1 + (i + j) % 2, 21)})
            for k in (0, m // 2, m - 1):
                density.append({'i': active[k], 'j': external, 'coefficient': scale * F(k + 1, 11 * m)})
            cases.append({'id': name + '_' + direction, 'base': name, 'modes': m + 1,
                          'particles': nu, 'active': active, 'external': external,
                          'scale': scale, 'amplitude': scale / 30, 'direction': direction,
                          'h': h, 'density': density})
    return enc({'schema_version': 1, 'id': 'n3_occupation_transfer_proposed_v1_20261012',
                'status': 'PROPOSED_INPUTS_FIXED_NO_SCIENTIFIC_EXECUTION',
                'basis': 'ordered spinless fermionic occupation modes',
                'geometry': None, 'chemical_basis': None, 'DF_rank': None, 'L_D': None,
                'delta_policy': 'scale/250 per input; identical across methods',
                'cases': cases, 'case_count': len(cases),
                'bounds_or_acceptance_computed': False, 'many_body_truth_computed': False,
                'random_sample_or_statistical_blind': False,
                'previous_12_cases_are_development': True, 'execution_authorized': False})


def main():
    # No source science module import, coefficient-bound or eigensolver call.
    new = make_inputs()
    old = json.loads((ROOT / 'artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json').read_text())
    old_physical = [{k: c[k] for k in ('modes', 'particles', 'active', 'external', 'h', 'density')} for c in old['cases']]
    for case in new['cases']:
        physical = {k: case[k] for k in ('modes', 'particles', 'active', 'external', 'h', 'density')}
        if physical in old_physical:
            raise ValueError('new proposed input duplicates development')
    data = (json.dumps(new, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    destination = OUT / 'proposed_transfer_inputs.json'
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError('refuse replacing already fixed proposed inputs')
    destination.write_bytes(data)
    receipt = {'status': 'COEFFICIENT_INPUT_PREPARATION_ONLY', 'input_sha256': hashlib.sha256(data).hexdigest(),
               'input_bytes': len(data), 'cases': len(new['cases']), 'bases': 4,
               'bound_evaluations': 0, 'sufficient_condition_evaluations': 0, 'ground_solves': 0,
               'independent_transfer_results': 0, 'mandatory_stop': True, 'execution_authorized': False}
    (OUT / 'input_preparation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
