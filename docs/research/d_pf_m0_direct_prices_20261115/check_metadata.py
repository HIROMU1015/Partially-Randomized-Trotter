"""Check published m0 bindings, without scientific imports or recomputation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)


def pinned(ref):
    value = json.loads(blob(ref['commit'], ref['path']))
    for k in ref.get('selector', []): value = value[k]
    return value


def main():
    table = json.loads((STUDY/'run01/AXIS_PRICES.json').read_text())
    assert table['counts'] == dict(planned_axes=12, reused=0, new_compiles=12, valid=12, held=0, missing=0, UNKNOWN=0)
    rows = table['rows']; assert len(rows) == 12
    assert len({(r['configuration_id'], r['physical_id'], r['axis'], r['m']) for r in rows}) == 12
    for r in rows:
        align, trial, price, side = (pinned(r[k]) for k in ('alignment', 'trial', 'receipt', 'sidecar'))
        assert (align['configuration_id'], align['axis'], align['m'], align['physical_candidate']['id']) == (r['configuration_id'], r['axis'], 0, r['physical_id'])
        assert trial['candidate'] == align['physical_candidate'] == price['physical_context']['physical_candidate']
        assert (r['T'], r['q'], r['actual_delta']) == (.1, 1, .1)
        assert r['T'].hex() == r['T_binary64_hex'] == r['actual_delta_binary64_hex']
        axis = 'cosine' if r['axis'] == 'X' else 'sine'
        assert r['metrics'] == trial['prices'][axis] == price['metrics']
        assert r['compiled_RZ'] == r['metrics']['rz_count'] == price['mean']['rz_count']
        assert trial['status'] == 'PASS' and trial['cache_hits'] == 0
        assert len(trial['checks']) == 7 and all(c['error'] < c['tolerance'] and c['phase_alignment'] is False for c in trial['checks'])
        assert side['implementation_evidence'] == r['implementation_evidence']
        proof = pinned(side['implementation_evidence'])
        assert all(proof[k] == side[k] for k in proof)
        assert r['execution_SOURCE'] == price['source']['commit'] == table['execution_SOURCE']
        assert price['diagnostic']['model_qualified'] is False and r['expectation'] == 'DETERMINISTIC'
        guard = json.loads((ROOT/r['receipt']['path']).with_name('GUARD_CHECK.json').read_text())
        g = next(c for c in guard['checks'] if c['configuration_id'] == r['configuration_id'])
        assert any(v['m'] == 0 and v['axis'] == r['axis'] for v in g['guard']['checked'])
    for ident in {r['physical_id'] for r in rows}:
        pair = [r for r in rows if r['physical_id'] == ident]
        assert {r['axis'] for r in pair} == {'X', 'Y'}
        assert len({r['wrapper_fingerprint'] for r in pair}) == 2
    for path in ('scripts/run_pf_m0_prices.py', 'scripts/run_full_pf_shortlist_prices.py', 'scripts/run_truncated_p7_prices.py',
                 'scripts/pf_price_stages.py', 'scripts/pf_native_price_fast.py', 'scripts/h46_b3_fast_fingerprint.py'):
        assert (ROOT/path).read_bytes() == blob(table['execution_SOURCE'], path)
    frozen = blob('12372fa6b53c878dc5dd124e036572ba0cd82061', 'docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
    assert hashlib.sha256(frozen).hexdigest() == '845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a'
    assert table['model_predictions'] == table['fit_calls'] == 0 and table['model_judgement'] is False
    assert table['N'] is table['G'] is table['winner'] is None and table['mandatory_stop'] is True
    print('PASS: 12 unique m0 alignment/axis/trial/receipt/proof/guard bindings; original execution SOURCE and frozen bytes preserved. Scientific calls: 0.')


if __name__ == '__main__': main()
