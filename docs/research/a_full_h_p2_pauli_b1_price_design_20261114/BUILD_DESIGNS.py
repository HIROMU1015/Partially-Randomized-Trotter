"""Candidate finite price designs (Full-H kappa2 P-2, PR-Pauli B1): metadata + exact rank + artificial SE.

Reads only pinned metadata (kappa2 layout 26d4f23e, Pauli round table 714247619). No price,
signal, sampling, build, compile or fit on real data. SE values are in units of an
artificial per-event noise sigma_e (path variance sigma_e^2 * s); they compare designs,
they are not estimates of real price variance.

    python -B docs/research/a_full_h_p2_pauli_b1_price_design_20261114/BUILD_DESIGNS.py > .../DESIGNS.json
"""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import finite_price_design as D  # noqa: E402

KAPPA = dict(commit='26d4f23e79e91385087669d4ee091632efa0c87c',
             path='docs/research/d_full_h_kappa_schedule_20261114/KAPPA_CANDIDATES.json')
PAULI = dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af',
             path='docs/research/df_head_pauli_tail_all_round_results_20261114/ROUND_SUMMARY.csv')
PROFILE = dict(compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b', qiskit='1.3.0',
               basis=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=17, control='symmetric_directional',
               wrapper='full_Hadamard_with_measurement', P=0)
FULL_H = dict(route='FULL_H_RTE', quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE', kind='DIRECT_COMPILED_FINITE_SAMPLE',
              law=dict(model='ideal_finite_RTE_law', K=2, price_law='PCG64_numpy_float64_CDF_saved_cohort',
                       preparation='9b83b96b:prepare_full_h'),
              implementation=dict(lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1', native='H46_LONG_TIME_NATIVE_COMPOSE',
                                  scalar_quantity='HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY',
                                  scalar_application='ONCE_AT_TOTAL_T'), profile=PROFILE, expectation='FINITE_COHORT_ONLY')


def blob(d):
    return subprocess.check_output(['git', 'show', d['commit'] + ':' + d['path']], cwd=ROOT)


def kappa2_rounds():
    k = json.loads(blob(KAPPA))['candidates'][0]
    assert k['configuration_id'] == '54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e'
    return {r['m']: dict(q=r['q_grouping'], R=r['R_per_group'], s=r['event_count'], tau=r['tau_float']) for r in k['rounds']}


def pauli_rounds():
    rows = list(csv.DictReader(io.StringIO(blob(PAULI).decode())))
    out = {}
    for r in rows:
        ld = 'LD' + r['prefix']
        out.setdefault(ld, {})[int(r['m'])] = dict(q=int(r['qPF']), R=int(r['R_outer']), s=int(r['event_count']),
                                                     T=float(r['T']))
    return out


def cell(cell_id, role, q, R, n, base, **kw):
    return dict(base, cell_id=cell_id, role=role, q=q, R=R, n_paths=n, **kw)


def evaluate(name, plan, targets, note):
    """Rank check via the plan gate; SE only for identified designs (artificial units)."""
    out = dict(design=name, note=note, cells=[{k: c[k] for k in ('cell_id', 'role', 'q', 'R', 'n_paths')}
                                              | ({'composed_of': c['composed_of']} if c.get('composed_of') else {})
                                              for c in plan['cells']])
    cal = [c for c in plan['cells'] if c['role'] == 'CALIBRATION']
    out['identifiability'] = D.identifiability(cal, plan['feature_map'])
    try:
        out['plan'] = D.check_plan(plan)
    except ValueError as error:
        out['plan'] = dict(status='REJECTED_BEFORE_ACQUISITION', reason=str(error))
        return out
    names = D.FEATURE_MAPS[plan['feature_map']][0]
    for sb, label in ((0.0, 'boundary_noise_0'), (5.0, 'boundary_noise_5_sigma_e')):
        cov = D.coefficient_covariance(cal, plan['feature_map'], sigma_event=1.0, sigma_boundary=sb)
        se = {n: float(cov[i, i]) ** .5 for i, n in enumerate(names)}
        pred = {}
        for tname, t in targets.items():
            fv = [D.features(dict(t, cell_id=tname), plan['feature_map'])[n] for n in names]
            se_p = D.prediction_se(cov, fv); s_t = t['q'] * t['R']
            # relative_SE_per_rho = SE / (b*s) for rho = sigma_e/b (unknown real ratio; design comparison only)
            pred[tname] = dict(prediction_SE=se_p, relative_SE_per_rho=se_p / s_t, single_path_event_SD=s_t ** .5)
        out[label] = dict(coefficient_SE=se, prediction=pred)
    return out


def full_h_designs():
    r = kappa2_rounds()
    g = lambda cid, R, n=2: cell(cid, 'CALIBRATION', 1, R, n, FULL_H)
    hold = [cell('M5_ROUND', 'HOLDOUT', r[5]['q'], r[5]['R'], 2, FULL_H),
            cell('M6_ROUND', 'HOLDOUT', r[6]['q'], r[6]['R'], 1, FULL_H)]
    base = [g('G566', 566), g('G4528', 4528), g('G36223', 36223)]
    plan = lambda cells: dict(feature_map='FULL_H_GROUP_V1', cells=cells + hold, design_source=dict(
        path='docs/research/a_full_h_p2_pauli_b1_price_design_20261114/BUILD_DESIGNS.py', status='PROPOSAL_NOT_ADOPTED'))
    targets = {('m%d' % m): dict(q=r[m]['q'], R=r[m]['R']) for m in (5, 6, 7, 9, 12)}
    designs = [
        evaluate('D0_EXISTING_Q1_ONLY', plan(base), targets,
                 'C/D plan: q=1 group cells R in {566,4528,36223} n=2, holdout m5 n2/m6 n1 (18 compiles); c not identified'),
        evaluate('D5_FIXED_R_Q1_Q2_Q4', plan([g('G566', 566), cell('Q2R566', 'CALIBRATION', 2, 566, 2, FULL_H),
                                              cell('Q4R566', 'CALIBRATION', 4, 566, 2, FULL_H)]), targets,
                 'q varies but R fixed: s=566(q-1)+566 is affine in q-1 -> rejected'),
        evaluate('D1_INDEPENDENT_Q2_Q4', plan(base + [cell('Q2R566', 'CALIBRATION', 2, 566, 2, FULL_H),
                                                      cell('Q4R566', 'CALIBRATION', 4, 566, 2, FULL_H)]), targets,
                 'identified, but c rests on independent noisy cells'),
        evaluate('D2_PAIRED_Q2', plan(base + [cell('P2R566', 'CALIBRATION', 2, 566, 1, FULL_H,
                                                   composed_of=['G566#p0#g0', 'G566#p1#g0'])]), targets,
                 'recommended minimum: q=2 cell built from the SAME two R=566 group paths (common random numbers); +2 compiles'),
        evaluate('D3_PAIRED_Q4', plan([g('G566', 566, 4), g('G4528', 4528), g('G36223', 36223),
                                       cell('P4R566', 'CALIBRATION', 4, 566, 1, FULL_H,
                                            composed_of=['G566#p%d#g0' % i for i in range(4)])]), targets,
                 'paired q=4 (3 boundaries) from 4 group paths at R=566; +6 compiles over D0'),
        evaluate('D6_INDEPENDENT_Q8', plan(base + [cell('Q8R566', 'CALIBRATION', 8, 566, 2, FULL_H)]), targets,
                 'one independent 8-group cell at R=566 (7 boundaries); +4 compiles over D0'),
        evaluate('D7_INDEPENDENT_Q8_PLUS_PAIRED_Q2', plan(base + [cell('Q8R566', 'CALIBRATION', 8, 566, 2, FULL_H),
                                                                  cell('P2R566', 'CALIBRATION', 2, 566, 1, FULL_H,
                                                                       composed_of=['G566#p0#g0', 'G566#p1#g0'])]), targets,
                 'D6 plus a paired 2-group cell from the two R=566 group paths (direct boundary-additivity check); +6 compiles'),
        evaluate('D4_SMALL_ROUNDS_AS_CALIBRATION', plan([cell('M%d_CAL' % m, 'CALIBRATION', r[m]['q'], r[m]['R'], 2, FULL_H)
                                                         for m in (2, 3, 4)]), targets,
                 'identified and cheap, but tau/order-2 rate and s range differ from the extrapolation targets'),
    ]
    return dict(rounds=r, designs=designs)


def pauli_designs():
    rounds = pauli_rounds(); out = {}
    for ld, r in rounds.items():
        base = dict(FULL_H, route='PR_PAULI_TAIL', law=dict(model='ideal_finite_RTE_law', K=2,
                    price_law='PCG64_numpy_float64_CDF_saved_cohort', dictionary=ld),
                    implementation=dict(lowering_id='PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF',
                                        native='DF_HEAD_NATIVE_S2', scalar_quantity='HAMILTONIAN_CONSTANT_PLUS_SAVED_S_R',
                                        scalar_application='ONCE_AT_TOTAL_T'))
        plan = lambda cells: dict(feature_map='PR_STEP_V1', cells=cells, design_source=dict(
            path='docs/research/a_full_h_p2_pauli_b1_price_design_20261114/BUILD_DESIGNS.py', status='PROPOSAL_NOT_ADOPTED'))
        targets = {('m%d' % m): dict(q=r[m]['q'], R=r[m]['R']) for m in (5, 6, 9, 12)}
        R6, R9, R12 = r[6]['R'], r[9]['R'], r[12]['R']
        hold = [cell('M5_ROUND', 'HOLDOUT', r[5]['q'], r[5]['R'], 2, base), cell('M6_ROUND', 'HOLDOUT', r[6]['q'], R6, 1, base)]
        designs = [
            evaluate('B1_C_PLAN_M0_TWO_PATHS', plan([cell('M0', 'CALIBRATION', 1, r[0]['R'], 2, base)]), targets,
                     'C/D B1 guide (2 fresh paths x X/Y per dictionary): wiring/variability only, no model'),
            evaluate('B1_CAL_SINGLE_STEP_PLUS_PAIRED', plan(
                [cell('S%d' % R6, 'CALIBRATION', 1, R6, 2, base), cell('S%d' % R9, 'CALIBRATION', 1, R9, 2, base),
                 cell('S%d' % R12, 'CALIBRATION', 1, R12, 2, base),
                 cell('P2R%d' % R6, 'CALIBRATION', 2, R6, 1, base, composed_of=['S%d#p0#g0' % R6, 'S%d#p1#g0' % R6])]
                + hold), targets,
                     'one-step cells at the m6/m9/m12 tail R, a paired 2-step cell for the head/step term, m5/m6 holdout'),
            evaluate('B1_CAL_SINGLE_STEP_PLUS_Q8_AND_PAIRED', plan(
                [cell('S%d' % R6, 'CALIBRATION', 1, R6, 2, base), cell('S%d' % R9, 'CALIBRATION', 1, R9, 2, base),
                 cell('S%d' % R12, 'CALIBRATION', 1, R12, 2, base), cell('Q8R%d' % R6, 'CALIBRATION', 8, R6, 2, base),
                 cell('P2R%d' % R6, 'CALIBRATION', 2, R6, 1, base, composed_of=['S%d#p0#g0' % R6, 'S%d#p1#g0' % R6])]
                + hold), targets,
                     'recommended proposal: as above plus an independent 8-step cell at the m6 tail R (step term precision)'),
        ]
        out[ld] = dict(rounds=r, designs=designs)
    return out


if __name__ == '__main__':
    print(json.dumps(dict(kind='FINITE_PRICE_DESIGN_CANDIDATES_METADATA_ONLY', status='PROPOSAL_NOT_ADOPTED',
                          sources=dict(kappa2=KAPPA, pauli_rounds=PAULI, c_model=D.C_MODEL),
                          sigma_units='artificial per-event noise sigma_e; path variance sigma_e^2*s; not real price variance',
                          full_h=full_h_designs(), pauli=pauli_designs(),
                          science=dict(prices=0, sampling=0, build=0, compile=0, fit_on_real_data=0)),
                     indent=1, default=str))
