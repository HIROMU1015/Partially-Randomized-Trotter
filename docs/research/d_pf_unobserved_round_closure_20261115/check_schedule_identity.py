"""Metadata-only check: are the per-step native inputs bitwise identical across rounds?

Reads the fixed C alignment rows and evaluates the existing schedule and control-plan
functions (h46_m5_plan.schedule, ax2a_control_plan.deterministic_control_plan).
No Hamiltonian, state, circuit build, transpile, saved-circuit read, price prediction
or fit. Usage (repository root):
    python -B docs/research/d_pf_unobserved_round_closure_20261115/check_schedule_identity.py
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src')]
ALIGNMENT = ('0b4b2d941cd669aaa45231e1236bc2d3158a9d12',
             'docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json')
CANDIDATES = {
    'FULL_PF S2_16h_over_7': ('FULL_PF', '2nd'), 'TRUNCATED_B0I_P7 S2_16h_over_7': ('TRUNCATED_PF', '2nd'),
    'FULL_PF Y4_16h_over_3': ('FULL_PF', '4th'), 'TRUNCATED_B0I_P7 Y4_16h_over_3': ('TRUNCATED_PF', '4th'),
    'FULL_PF M5_16h': ('FULL_PF', '4th(m5_best)'), 'TRUNCATED_B0I_P7 M5_16h': ('TRUNCATED_PF', '4th(m5_best)'),
}
ROUNDS = range(4, 13)
STATUS = {4: 'CALIBRATION', 5: 'CALIBRATION', 6: 'OBSERVED_HOLDOUT', 9: 'OBSERVED_HOLDOUT'}


def main():
    spec = importlib.util.spec_from_file_location('m5_plan', ROOT / 'src/trotterlib/h46_m5_plan.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    from trottertracks.resource_applicability.ax2a_control_plan import deterministic_control_plan
    text = subprocess.run(['git', 'show', ':'.join(ALIGNMENT)], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    rows = json.loads(text)['rows']
    checks, groups = [], []

    def check(name, ok, **detail):
        checks.append(dict(check=name, ok=bool(ok), **detail))
        return bool(ok)

    for label, (route, formula) in CANDIDATES.items():
        selected = {}
        for index, row in enumerate(rows):
            if row['candidate'] == label and row['m'] in ROUNDS:
                selected.setdefault(row['m'], {})[row['axis']] = (index, row['physical_candidate'])
        base = selected[4]['X'][1]
        n_blocks = base['prefix'] + 1
        reference = None
        per_round = []
        for m in ROUNDS:
            (ix, px), (iy, py) = selected[m]['X'], selected[m]['Y']
            check('xy_same_physical', px == py, label=label, m=m, rows=[ix, iy])
            check('formula_route', px['formula'] == formula and
                  px['method'] == ('FULL' if route == 'FULL_PF' else 'B0I'), label=label, m=m)
            check('T_exact_power_of_two', float(px['T']).hex() == math.ldexp(float(base['T']), m - 4).hex(),
                  label=label, m=m, T=float(px['T']).hex())
            check('q_doubling', px['q'] == base['q'] << (m - 4), label=label, m=m, q=px['q'])
            steps = core.schedule(formula, px['q'], 'PER_PIECE', 1, T=px['T'])
            per_step = len(core.weights(formula))
            pattern = [s['signed_time'].hex() for s in steps[:per_step]]
            periodic = all(s['signed_time'].hex() == pattern[i % per_step] for i, s in enumerate(steps))
            check('within_round_step_identity', periodic and len(steps) == px['q'] * per_step,
                  label=label, m=m, pieces=len(steps))
            if reference is None:
                reference = pattern
            check('cross_round_piece_identity', pattern == reference, label=label, m=m)
            stage_times = set()
            for value in set(pattern):
                plan = deterministic_control_plan(n_blocks, float.fromhex(value), 1, '2nd')
                stage_times |= {(float.fromhex(value).hex(), st.time.hex()) for st in plan}
                check('control_plan_half_time', all(st.time.hex() == (float.fromhex(value) / 2).hex() for st in plan)
                      and len(plan) == 2 * n_blocks, label=label, m=m, signed_time=value)
            check('actual_width_hex', float(px['delta']).hex() == float(base['delta']).hex(), label=label, m=m)
            per_round.append(dict(m=m, status=STATUS.get(m, 'UNOBSERVED'), physical_id=px['id'], alignment_rows=[ix, iy],
                                  T_hex=float(px['T']).hex(), q=px['q'], signed_pieces=len(steps),
                                  negative_pieces=sum(s['signed_time'] < 0 for s in steps)))
        groups.append(dict(candidate=label, route=route, formula=formula, retained_blocks=n_blocks,
                           pieces_per_step=len(reference), distinct_signed_times=sorted(set(reference)),
                           per_step_signed_time_pattern=reference, rounds=per_round))
    out = dict(schema='D_PF_SCHEDULE_IDENTITY_CHECK_V1', alignment=dict(commit=ALIGNMENT[0], path=ALIGNMENT[1]),
               functions=['src/trotterlib/h46_m5_plan.py:schedule',
                          'src/trottertracks/resource_applicability/ax2a_control_plan.py:deterministic_control_plan'],
               scope='metadata_only_no_circuit_no_transpile_no_saved_circuit_no_prediction',
               checks=len(checks), failed=[c for c in checks if not c['ok']], all_pass=all(c['ok'] for c in checks),
               groups=groups)
    path = Path(__file__).with_name('SCHEDULE_IDENTITY.json')
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(checks=out['checks'], failed=len(out['failed']), all_pass=out['all_pass'])))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
