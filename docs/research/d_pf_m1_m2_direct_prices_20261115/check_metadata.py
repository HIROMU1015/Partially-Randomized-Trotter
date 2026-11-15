"""Fixed m1/m2 publication metadata only; no circuit/vector/scientific calls."""
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
    for key in ref.get('selector', []): value = value[key]
    return value


def main():
    table = json.loads((STUDY/'run01/AXIS_PRICES.json').read_text())
    assert table['counts'] == dict(planned_axes=24, reused=0, new_compiles=24, valid=24, held=0, missing=0, UNKNOWN=0)
    rows = table['rows']; assert len(rows) == 24
    assert len({(r['configuration_id'],r['physical_id'],r['axis'],r['m']) for r in rows}) == 24
    assert table['direct_price_coverage_before'] == 72 and table['direct_price_coverage_after'] == 96
    for r in rows:
        a, trial, price, side = (pinned(r[key]) for key in ('alignment','trial','receipt','sidecar'))
        assert (a['configuration_id'],a['axis'],a['m'],a['physical_candidate']['id']) == (r['configuration_id'],r['axis'],r['m'],r['physical_id'])
        assert trial['candidate'] == a['physical_candidate'] == price['physical_context']['physical_candidate']
        T = .2 if r['m'] == 1 else .4
        q = 2 if r['m'] == 2 and r['formula'] == '2nd' else 1
        assert r['m'] in (1,2) and (r['T'],r['q'],r['actual_delta']) == (T,q,T/q)
        assert r['T'].hex() == r['T_binary64_hex'] and r['actual_delta'].hex() == r['actual_delta_binary64_hex']
        axis = 'cosine' if r['axis'] == 'X' else 'sine'
        assert r['metrics'] == trial['prices'][axis] == price['metrics']
        assert r['compiled_RZ'] == r['metrics']['rz_count'] == price['mean']['rz_count']
        assert trial['status'] == 'PASS' and trial['cache_hits'] == 0
        assert len(trial['checks']) == 7 and all(c['error'] < c['tolerance'] and c['phase_alignment'] is False for c in trial['checks'])
        assert side['implementation_evidence'] == r['implementation_evidence']
        proof = pinned(side['implementation_evidence']); assert all(proof[k] == side[k] for k in proof)
        assert r['execution_SOURCE'] == price['source']['commit'] == table['execution_SOURCE']
        assert price['diagnostic']['model_qualified'] is False and price['source']['stage'] == 'M1_M2_DIRECT_ONLY'
        assert r['classification'] == 'DIRECT_COMPILED_DETERMINISTIC' and r['expectation'] == 'DETERMINISTIC'
        guard = json.loads((ROOT/r['receipt']['path']).with_name('GUARD_CHECK.json').read_text())
        group = next(v for v in guard['checks'] if v['configuration_id'] == r['configuration_id'])
        assert any(v['m'] == r['m'] and v['axis'] == r['axis'] for v in group['guard']['checked'])
    for ident in {r['physical_id'] for r in rows}:
        pair = [r for r in rows if r['physical_id'] == ident]
        assert {r['axis'] for r in pair} == {'X','Y'} and len({r['wrapper_fingerprint'] for r in pair}) == 2
    for cfg in {r['configuration_id'] for r in rows if r['formula'] == '2nd'}:
        pair = [r for r in rows if r['configuration_id'] == cfg and r['axis'] == 'X']
        assert len({r['physical_id'] for r in pair}) == len({r['wrapper_fingerprint'] for r in pair}) == 2
        assert {r['actual_delta'] for r in pair} == {.2}
    for path in ('scripts/run_pf_m0_prices.py','scripts/run_full_pf_shortlist_prices.py','scripts/run_truncated_p7_prices.py',
                 'scripts/pf_price_stages.py','scripts/pf_native_price_fast.py','scripts/h46_b3_fast_fingerprint.py'):
        assert (ROOT/path).read_bytes() == blob(table['execution_SOURCE'],path)
    frozen = blob('12372fa6b53c878dc5dd124e036572ba0cd82061','docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
    assert hashlib.sha256(frozen).hexdigest() == '845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a'
    assert table['model_predictions'] == table['fit_calls'] == 0 and table['model_judgement'] is False
    assert table['N'] is table['G'] is table['winner'] is None and table['mandatory_stop'] is True
    print('PASS: 24 m1/m2 alignment/axis/trial/receipt/proof/C guard bindings; S2 equal width keeps distinct T/q/fingerprint. Fixed SOURCE/frozen bytes preserved. Scientific calls: 0.')


if __name__ == '__main__': main()
