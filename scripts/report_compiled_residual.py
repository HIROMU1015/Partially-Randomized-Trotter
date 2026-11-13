#!/usr/bin/env python3
"""Write the C - S residual SOURCE inventory, design-identifiability analyses and an
artificial B-integration example into a fresh directory (metadata only).

Pinned Git reads only. No sampling, build/compile/transpile, structural recount,
fit, residual statistics of saved data, N/RMSE or G. The example uses mock E[S]
and mock D values on the fixed B configuration to show the record path and B's
rejection of an unqualified E[C]; it is not a price.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import compiled_residual as R  # noqa: E402
from trottertracks import abc_s0_published_intake as S0  # noqa: E402
from trottertracks import rpe_diagnostic_intake as K  # noqa: E402

p = R.p


def write(path, value):
    def default(v):
        if isinstance(v, Fraction):
            return dict(numerator=v.numerator, denominator=v.denominator)
        raise TypeError(type(v))
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=default)
        f.write('\n')


def designs(rounds):
    point = {r['m']: (r['q_PF'], r['R_outer']) for r in rounds}
    c1 = [(8, 4), (8, 8), (16, 4)]
    cases = [
        ('S0_FOCUS_m0_m6_THREE_TERM', [point[0], point[6]], ['intercept', 'q', 'qR'], []),
        ('S0_FOCUS_m0_m6_Q_R_SEPARATE', [point[0], point[6]], ['intercept', 'q', 'R', 'qR'], []),
        ('S0_PATH_m0_m6_Q_R_SEPARATE', [point[m] for m in range(7)], ['intercept', 'q', 'R', 'qR'], []),
        ('S0_PATH_m0_m12_Q_R_SEPARATE', [point[m] for m in range(13)], ['intercept', 'q', 'R', 'qR'], []),
        ('C1_H4_SAVED_DESIGN', c1, ['intercept', 'q', 'qR'], [(16, 8)]),
        ('PROPOSED_ON_PATH_LADDER', [point[1], point[3], point[5]], ['intercept', 'q', 'qR'], [point[2], point[4], point[6], point[7]]),
    ]
    out = []
    for name, cal, features, hold in cases:
        v = R.identifiability(cal, features, hold)
        out.append(dict(name=name, **v))
    return dict(status='DESIGN_RANK_ONLY_NO_DATA', analyses=out,
                note='X/Y are paired responses of the same design point and add no rows. Exact rank only; an exactly '
                     'full rank with R/q nearly constant (m7..m12 use R=2q-1..) is still ill-conditioned for separating q '
                     'from R. These analyses fit nothing.')


def example(store):
    config = store.read(dict(commit=S0.B_RELEASE, path=S0.BDIR + 'CONFIGURATION_V2.json'))
    rows = []
    for m in (0, 6):
        r = config['rounds'][m]
        for axis in p.AXES:
            rows.append(dict(m=m, axis=axis, identity=deepcopy(r['identity']), law_binding=deepcopy(r['law_binding']),
                physical_candidate_id=r['physical_candidate']['id'], profile=deepcopy(config['spec']['profile']),
                source=dict(fixture='MOCK_K1_NOT_A_RESULT'), law_context=dict(law='IDEAL_FINITE_RTE_LAW'),
                checks=dict(K0='MOCK_PASS'), uncertainty=dict(numerical_interval=None), mean_S=1000 + m))
    packet = dict(configuration_id=config['id'], source=dict(fixture='MOCK_K1_NOT_A_RESULT'), kernel_source=dict(fixture='MOCK'),
                  stage='K1', K0_status='PASS', quantity=K.STRUCTURAL, qualified_for_candidate=False,
                  mean_compiled_C=None, mean_C_minus_S=None, rows=rows)
    accepted = K.accept_structural_packet(config, packet)
    records = []; cohorts = {}
    for s in rows:
        r = config['rounds'][s['m']]
        residual = dict(quantity=R.MEAN_D, target='C_MINUS_S', mode='PAIRED_DIRECT_FINITE_COHORT', coverage=R.FULL,
            m=s['m'], axis=s['axis'], identity=deepcopy(r['identity']), law_binding=deepcopy(r['law_binding']),
            physical_candidate_id=r['physical_candidate']['id'], profile=deepcopy(config['spec']['profile']),
            law_context=dict(law='IDEAL_FINITE_RTE_LAW'), mean_D=Fraction(25), expectation='FINITE_COHORT_ONLY', n_paths=2,
            role='DEVELOPMENT', source=dict(fixture='MOCK_PAIRED_D_NOT_A_RESULT'), uncertainty=dict(descriptive_variance=None))
        combined = R.combine(s, residual, config=config)
        records.append(R.b_price_record(combined, config=config, cohort='MOCK_C_MINUS_S', source=dict(fixture='MOCK')))
        cohorts[(r['candidate']['id'], s['axis'])] = 'MOCK_C_MINUS_S'
    assessed = p_assess(config, records, cohorts)
    return dict(status='ARTIFICIAL_MOCK_VALUES_NOT_A_PRICE_OR_RESULT', configuration_id=config['id'],
                structural_packet_accepted=dict(compiled_price_records=accepted['compiled_price_records'],
                                                scientific_price_qualification=accepted['scientific_price_qualification']),
                price_records=records, B_assess=dict(eligible=assessed['eligible'], cost=assessed['cost'],
                reasons=sorted(set(assessed['reasons']))))


def p_assess(config, records, cohorts):
    return S0.B.assess(config, None, records, cohorts=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    store = p.SavedArtifacts(ROOT)
    inventory = R.saved_paired_inventory(store); rounds = R.target_rounds(store)
    write(args.output_dir/'PAIRED_SC_INVENTORY.json', dict(inventory, target_rounds=rounds, read_sources=dict(store.sources)))
    write(args.output_dir/'IDENTIFIABILITY.json', designs(rounds))
    write(args.output_dir/'B_RECORD_EXAMPLE_ARTIFICIAL.json', example(store))


if __name__ == '__main__':
    main()
