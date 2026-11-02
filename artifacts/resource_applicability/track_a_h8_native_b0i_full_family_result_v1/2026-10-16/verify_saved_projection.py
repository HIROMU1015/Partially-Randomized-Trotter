"""Verify published saved-value tables only. Stdlib, no science or runtime data.

Usage: python -P -B verify_saved_projection.py [BUNDLE_DIRECTORY]
Read-only; creates no files, imports no scientific packages, never launches jobs.
"""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

B=Path(sys.argv[1]) if len(sys.argv)==2 else Path(__file__).resolve().parent
if len(sys.argv)>2:raise SystemExit(__doc__)
def load(name):return json.loads((B/name).read_bytes())
def rat(value):return Fraction(int(value['numerator']),int(value['denominator']))
METRICS=('RZ','CX','RZ_depth','CX_depth','depth','size')
checks=0
for receipt in load('file_receipts_v1.json')['files']:
    path=B/receipt['name'];raw=path.read_bytes()
    assert len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256'],path
    checks+=1
with (B/'axis_compiled_metrics_v1.csv').open(newline='') as f:axes=list(csv.DictReader(f))
assert len(axes)==16
axis_by={(int(r['prefix']),int(r['q']),r['axis']):r for r in axes}
accounting=load('case_accounting_v1.json');assert len(accounting)==120
with (B/'case_eligibility_v1.csv').open(newline='') as f:eligibility=list(csv.DictReader(f))
for row in accounting:
    p,q=row['cell']['prefix'],row['cell']['q'];a=row['accounting']
    if row['status']=='INELIGIBLE':
        assert row['N_total'] is None and row['metrics']=={}
        continue
    assert row['status']=='DERIVED_SAVED_SCALAR' and row['cost_status']=='ACQUIRED'
    assert row['N_total']==a['axes']['real']['shots']+a['axes']['imag']['shots']
    for metric in METRICS:
        point=sum(a['axes'][channel]['shots']*int(axis_by[(p,q,axis)][metric])
            for axis,channel in (('cosine','real'),('sine','imag')))
        assert point==row['metrics'][metric]['G_point'];checks+=1
assert len(eligibility)==120 and sum(r['status']=='DERIVED_SAVED_SCALAR' for r in accounting)==32
correctness=load('case_correctness_v1.json');assert len(correctness)==8
for c in correctness:
    assert c['status']=='PASS' and c['max_error']<=1e-8
    for axis in ('cosine','sine'):
        assert c['fingerprints'][axis]==axis_by[(c['prefix'],c['q'],axis)]['numeric_fingerprint'];checks+=1
baseline=load('saved_confirmation_cost_projection_v1.json')['rows'];assert len(baseline)==260
pr_ids={r['cell_id'] for r in baseline if r['method']=='B2'};assert len(pr_ids)==6
assert all(r['sample_phase']=='confirmation' and r['sample_count']==32 for r in baseline if r['method']=='B2')
all_rows=baseline+accounting
comparison=load('comparison_summary_v1.json')['comparisons'];assert len(comparison)==60
with (B/'comparison_P0_v1.csv').open(newline='') as f:flat=list(csv.DictReader(f))
assert len(flat)==60
for row in comparison:
    e,f,m=row['epsilon'],row['factor'],row['metric']
    good=[r for r in all_rows if (r['epsilon'],float(r['factor']))==(e,f) and r['status']=='DERIVED_SAVED_SCALAR']
    def select(rows):
        if not rows:return None
        selected=min(rows,key=lambda r:(r['metrics'][m]['G_point'],r['cell_id']))
        return dict(cell_id=selected['cell_id'],G_point=selected['metrics'][m]['G_point'])
    det=select([r for r in good if r['method']!='B2']);pr=select([r for r in good if r['method']=='B2'])
    assert det==row['best_registered_deterministic'] and pr==row['confirmed_PR_union6']
    assert pr['G_point']/det['G_point']==row['PR_over_deterministic']<1
    assert 100*(1-row['PR_over_deterministic'])==row['point_reduction_percent']
    selected=next(r for r in good if r['cell_id']==pr['cell_id'])
    assert row['selected_PR_paired_sample_SE']==selected['metrics'][m]['G_standard_error']
    old=next(r for r in flat if (float(r['epsilon']),float(r['factor']),r['metric'])==(e,f,m))
    assert old['deterministic_cell']==det['cell_id'] and old['PR_confirmation_cell']==pr['cell_id']
    assert float(old['PR_over_deterministic'])==row['PR_over_deterministic'];checks+=8
envelopes=load('P_sensitivity_v1.json');assert len(envelopes)==60
segments=0;P_checks=0
for envelope in envelopes:
    e,f,m=envelope['epsilon'],envelope['factor'],envelope['metric']
    lines={r['cell_id']:(r['N_total'],Fraction(r['metrics'][m]['G_point'])) for r in all_rows
        if r['epsilon']==e and float(r['factor'])==f and r['status']=='DERIVED_SAVED_SCALAR'}
    items=envelope['lower_envelope']['segments'];assert rat(items[0]['P_start'])==0 and items[-1]['P_end'] is None
    for i,segment in enumerate(items):
        lo=rat(segment['P_start']);hi=rat(segment['P_end']) if segment['P_end'] is not None else None
        s,b=segment['slope'],rat(segment['intercept'])
        if i:assert rat(items[i-1]['P_end'])==lo
        assert hi is None or hi>lo
        assert sorted(k for k,v in lines.items() if v==(s,b))==segment['cell_ids']
        for other_s,other_b in lines.values():
            assert (other_b-b)+(other_s-s)*lo>=0;P_checks+=1
            if hi is None:assert other_s>=s
            else:assert (other_b-b)+(other_s-s)*hi>=0
            P_checks+=1
        segments+=1
    for vertex in envelope['lower_envelope']['vertices']:
        point=rat(vertex['P']);minimum=min(b+s*point for s,b in lines.values())
        assert sorted(k for k,(s,b) in lines.items() if b+s*point==minimum)==vertex['tied_cells'];P_checks+=1
assert segments==284
audit=load('completion_audit_v1.json')
assert audit['status']=='COMPLETE_MANDATORY_STOP' and audit['compiled_axes']==16 and audit['all_reaped']
assert audit['first_stop'] is None and audit['maximum_native_reference_error']<=1e-8
assert audit['label_correction']['actual_scope'].startswith('H8_CONFIRMATION_UNION6')
assert audit['historical_carry']['historical_unknown_actual'] is None
assert not audit['historical_carry']['reset'] and audit['historical_carry']['refund']==0
print(json.dumps(dict(status='PASS_PUBLIC_SAVED_VALUE_PROJECTION',checks=checks,P_exact_line_checks=P_checks,
    compared_conditions=60,P_segments=segments,NPZ_array_reads=0,scientific_operations=0,
    confirmation_candidates=6,trajectories_per_PR_candidate=32,new_scientific_review=False)))
