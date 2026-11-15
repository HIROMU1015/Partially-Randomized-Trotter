"""Published accepted-price join only; no gate simulation, fit or prediction."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def read(ref):
    v=json.loads(subprocess.check_output(['git','show',ref['commit']+':'+ref['path']],cwd=ROOT))
    for k in ref.get('selector',[]):v=v[k]
    return v


def main():
    table=json.loads((HERE/'run01/AXIS_PRICES.json').read_text());rows=table['rows']
    assert table['counts']==dict(total_axes=26,accepted=10,reused=6,new_compiles=4,new_valid=4,target_unknown=0,held=0,not_acquired=16)
    assert len(rows)==len({(r['configuration_id'],r['physical_id'],r['m'],r['axis']) for r in rows})==26
    accepted=[r for r in rows if r['status']=='ACCEPTED_DIRECT']
    guard=json.loads((HERE/'run01/GUARD_CHECK.json').read_text())['guard']['checked']
    assert {(r['m'],r['axis']) for r in accepted}=={(r['m'],r['axis']) for r in guard}
    for r in rows:
        assert r['configuration_id']==table['configuration_id'] and r['q']==2**r['m'] and r['actual_delta']==.1
        if r['status']=='NOT_ACQUIRED':
            assert r['m']>=5 and r['compiled_RZ'] is r['receipt'] is None;continue
        p,s,t=(read(r[k]) for k in ('receipt','sidecar','trial'))
        axis='cosine' if r['axis']=='X' else 'sine'
        assert p['physical_context']['physical_candidate']==r['physical_candidate']
        assert p['identity']['candidate']['id']==r['original_canonical_id']
        assert p['mean']['rz_count']==t['prices'][axis]['rz_count']==r['compiled_RZ']
        assert p['cohort']==s['cohort']==r['cohort']
        assert p['source']==s['price_source']==r['price_source']
        assert all(g.get('PASS',True) and g['error']<g.get('tolerance',g.get('strictly_below',0)) for g in t['checks'])
        proof=read(s['implementation_evidence'])
        assert all(proof[k]==s[k] for k in proof)
        assert p['classification']=='DIRECT_COMPILED_DETERMINISTIC' and p['diagnostic']['model_qualified'] is False
        if r['acquisition']=='NEW_COMPILE':
            assert t['cache_hits']==0 and len(t['checks'])==7 and t['cohort']=='D_P7_S2_H_DIRECT_V1'
        else:
            assert r['m'] in (0,3,4) and r['original_producer']
            if r['m']==3:assert r['original_cohort'] is None
            if r['m']==4:assert r['cohort']==r['original_cohort']=='EXPLORATION'
    assert all(table[k]==0 for k in ('model_fit','long_round_predictions','Q3_calls','shot_generation'))
    assert table['reference_cost']['value_RZ']==435481090 and table['reference_cost']['scope'].endswith('NOT_WIDTH_h')
    assert table['N'] is table['G'] is table['winner'] is None and table['mandatory_stop']
    print('PASS: ten accepted axes and sixteen unacquired axes; original/new receipt, scalar/gate/provenance and C guard join preserved. No scientific calls.')


if __name__=='__main__':main()
