"""Check exported saved arithmetic; no molecular inputs, fitting or compilation."""
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]


def read(path):return json.loads(path.read_bytes())


def best(rows):
    if any(x['eligible'] is None for x in rows):return None
    good=[x for x in rows if x['eligible']]
    return min(good,key=lambda x:(Fraction(x['G_exact']),x['cell_id'])) if good else None


def scope(row,name):
    cell=row['cell']
    if cell['method']=='B2':return False
    if cell['method']=='B1':return True
    if name=='old_grid_S2':return cell['formula']=='2nd' and cell['prefix'] in (5,10,15)
    if name=='fine_grid_S2':return cell['formula']=='2nd'
    return name=='fine_grid_S2_plus_truncated_S4'


def main():
    receipt=read(HERE/'accounting_export_receipt.json');compressed=(HERE/'candidate_accounting.json.gz').read_bytes()
    assert len(compressed)==receipt['gzip_bytes'] and hashlib.sha256(compressed).hexdigest()==receipt['gzip_sha256']
    raw=gzip.decompress(compressed)
    assert len(raw)==receipt['expanded_bytes'] and hashlib.sha256(raw).hexdigest()==receipt['expanded_sha256']
    rows=json.loads(raw)['rows'];assert len(rows)==4320
    model_path=BASE/'track_a_v1_phase_b_readiness/2026-10-16/frozen_model.json'
    model=read(model_path);model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_sha=='406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211'
    assert model['fitted_after_validation'] is False
    validation=read(BASE/'track_a_v1_phase_b_run01/2026-10-16/validation_complete.json')['rows']
    validated={(x['cell_id'],x['axis']):x for x in validation}
    costs=read(HERE/'axis_costs.json')['rows'];assert len(costs)==410
    assert sum(x['status']=='VALIDATED_MODEL_ESTIMATE' for x in costs)==196
    for item in costs:
        cid=item['cell_id'];key=(cid,item['axis'])
        if item.get('source_kind')=='PHASE_B_REGISTERED_VALIDATION':
            assert int(item['RZ'])==validated[key]['measured'] and item['measured'] is True
        if item['status']=='VALIDATED_MODEL_ESTIMATE':
            method,p,q,formula=cid.split('_');p=int(p[1:]);q=int(q[1:])
            parent=next(x for x in model['models'] if x['stratum']['method']==method and x['stratum']['axis']==item['axis'])
            a,b,d=map(Fraction,parent['coefficients']);pieces=1 if formula=='2nd' else 3
            assert Fraction(item['RZ'])==a+pieces*q*(b+p*d)
            assert item['model_sha256']==model_sha and item['measured'] is False
    for row in rows:
        if row['eligible']:
            a=row['accounting'];axes=row['axis_costs'];assert len(axes)==2
            expected=sum(Fraction(x['RZ'])*a['axes']['real' if x['axis']=='cosine' else 'imag']['shots'] for x in axes)
            assert expected==Fraction(row['G_exact']) and float(expected)==row['G_point']
            assert row['N_total']==sum(x['shots'] for x in a['axes'].values())
        else:assert row['G_exact'] is None and row['G_point'] is None
        assert row['conservative_bias_certificate'] is None
    comparisons=read(HERE/'comparisons.json')['rows'];assert len(comparisons)==60
    table=list(csv.DictReader((HERE/'comparisons.csv').open()));assert len(table)==60
    for item,saved in zip(comparisons,table):
        selected=[x for x in rows if x['epsilon']==item['epsilon'] and x['factor']==item['factor'] and x['bias_rule']==item['bias_rule']]
        det=[x for x in selected if scope(x,item['scope'])]
        pr=[x for x in selected if x.get('sample_scope')=='H6_CONFIRMATION_UNION8']
        assert len(pr)==8 and len(det)=={'old_grid_S2':32,'fine_grid_S2':96,'fine_grid_S2_plus_truncated_S4':136}[item['scope']]
        d=best(det);p=best(pr)
        for actual,expected in [(item['deterministic']['best'],d),(item['confirmed_PR']['best'],p)]:
            if expected is None:assert actual is None
            else:assert actual['cell_id']==expected['cell_id'] and actual['G_exact']==expected['G_exact']
        if d and p:
            ratio=Fraction(p['G_exact'])/Fraction(d['G_exact'])
            assert ratio==Fraction(item['PR_over_deterministic_exact'])
            assert float(100*(1-ratio))==item['PR_reduction_percentage_points']
            assert saved['deterministic_G_exact']==d['G_exact'] and saved['PR_G_exact']==p['G_exact']
        else:
            assert item['PR_over_deterministic'] is None and saved['deterministic_G_exact']==''
            assert item['deterministic']['unresolved_candidates']==12
    terminal=read(HERE/'execution_terminal.json');launcher=read(HERE/'launcher_terminal.json');audit=read(HERE/'completion_audit.json')
    assert terminal['status']==launcher['status']==audit['status']=='COMPLETE_MANDATORY_STOP'
    assert terminal['exit_code']==launcher['native_exit_code']==0 and launcher['all_reaped']
    assert launcher['first_stop'] is None and launcher['cleanup_errors']==[]
    assert terminal['summary']['saved_reproduction_checks']==2170
    assert audit['protected_files_unchanged']==653 and audit['runtime_profile_files_unchanged']==34273
    assert not terminal['next_stage_authorized'] and not audit['next_stage_authorized']
    print('PASS: lossless 4320 rows, frozen costs, measured precedence, all 60 selections/ratios, CSV and terminal consistency')


if __name__=='__main__':main()
