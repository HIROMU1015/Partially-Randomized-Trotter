"""Check published identities/locators/exact products; no F0 or circuits executed."""
from pathlib import Path
from fractions import Fraction
import csv
import json
import subprocess

B = Path(__file__).resolve().parent.relative_to(Path.cwd())
load = lambda p: json.loads(p.read_text())
e, h = load(B / 'RANGE_EVIDENCE.json'), load(B / 'C_HANDOFF.json')
cache = {}


def resolve(loc):
    key = (loc['commit'], loc['path'])
    if key not in cache:
        data = subprocess.check_output(['git', 'show', ':'.join(key)])
        cache[key] = json.loads(data) if key[1].endswith('.json') else data
    out = cache[key]
    for part in loc.get('selector', []):
        out = out[part]
    return out


assert len(e['rows']) == len(h['rows']) == 26
assert len({(r['m'], r['axis']) for r in e['rows']}) == 26
for i, (r, a) in enumerate(zip(e['rows'], h['rows'])):
    m, axis = divmod(i, 2)
    assert (r['m'], r['axis']) == (m, ('X', 'Y')[axis])
    assert (a['m'], a['axis'], a['physical_id']) == (m, r['axis'], r['physical_id'])
    f0 = resolve(r['F0_path_bound'])
    base = resolve(r['base_axis_index'])
    assert f0['m'] == base['m'] == m
    assert f0['physical_id'] == base['physical_id'] == r['physical_id']
    assert base['axis'] == r['axis'] and f0['event_positions'] == r['q'] * r['R_outer']
    assert a['range_record']['selector'] == ['rows', i]
    assert a['saved_F0_bound'] == r['F0_path_bound']
    assert a['base_axis_index'] == r['base_axis_index']
    us, uc = r['S_upper'], r['C_upper']
    assert r['separate_envelope_residual_interval'] == dict(lower=-us, upper=uc)
    assert r['residual_oscillation_upper'] == us + uc
    assert Fraction(**r['conditional_RZ_shift_upper']) == Fraction(**f0['path_TV_upper']) * (us + uc)
    assert r['actual_PCG64_RZ_shift_upper'] is r['direct_residual_lower'] is r['direct_residual_upper'] is None
    assert all(r[k] is None for k in ('E_C', 'CI', 'N', 'G'))
    if m == 4:
        assert r['finite_compiled_evidence_role'] == 'INDEPENDENT_HOLDOUT'
    elif m in (0, 3, 6):
        assert r['finite_compiled_evidence_role'] == 'CALIBRATION'
for x, y in zip(e['rows'][::2], e['rows'][1::2]):
    assert x['F0_path_bound'] == y['F0_path_bound']
    assert y['S_upper'] - x['S_upper'] == 1 and y['C_upper'] - x['C_upper'] == 5
for loc in e['sources'].values():
    resolve(loc)
    actual = subprocess.check_output(['git', 'rev-parse', loc['commit'] + ':' + loc['path']], text=True).strip()
    assert actual == loc['git_blob']
with (B / 'AXES.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 26
for r, row in zip(e['rows'], rows):
    for k, value in row.items():
        assert value == str(r[k])
assert h['derivation_source']['git_blob'] == subprocess.check_output(
    ['git', 'hash-object', str(B / 'DERIVE_ENVELOPES.py')], text=True).strip()
changed = subprocess.check_output(['git', '-c', 'core.quotepath=false', 'diff', '--name-only', e['F0']['commit']], text=True).splitlines()
allowed = {'PROJECT_MAP.md', 'docs/README.md', 'docs/research/研究概要・現状.md',
           'docs/research/研究ノート/2026-11-15.md', 'artifacts/validation_manifest.json'}
assert all(p in allowed or p.startswith(str(B) + '/') for p in changed)
out = dict(status='PASS', axis_identity_rows=26, fixed_source_locators=len(e['sources']),
           exact_products_checked=26, shared_F0_round_bounds=13,
           finite_roles_preserved=True, original_protected_artifacts_unchanged=True,
           validation_scope='STATIC_LOCATORS_METADATA_AST_AND_AUTHORIZED_BOUND_ARITHMETIC',
           new_paths=0, H4_builds=0, compiles=0, F0_reruns=0,
           full_support_action_test=False, independent_review=False)
(B / 'CHECKS.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(out, ensure_ascii=False))
