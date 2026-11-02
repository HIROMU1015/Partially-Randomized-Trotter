"""Copy saved scalars/CSV fields; no norms, arrays, shots or cost arithmetic.

Run from repository root. This evidence-formatting utility is not a scientific
runner and imports no numerical packages. Ratios/percentages are not recomputed.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = '3a8356a7843d3cb46bb8069955640381b490ac4c'
PRIOR = 'artifacts/research_c_c1_comparison_audit/2026-10-26/saved_comparison.csv'
NEW = 'artifacts/research_c_c1_h4_ca_resources/2026-10-26/candidate_totals.csv'
H2DF = 'artifacts/research_c_c1_h2_h6_science/2026-10-25/H2-run01/result.json'
H4DF = 'artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01/result.json'
H6DF = 'artifacts/research_c_c1_h6_corrected_fit/2026-10-25/state_comparison.csv'
INPUTS = [PRIOR,NEW,H2DF,H4DF,H6DF,
    'artifacts/research_c_c1_h4_ca_resources/2026-10-26/same_s_comparisons.csv',
    'artifacts/research_c_c1_h4_native_fair/2026-10-26/native_metrics.csv']
INPUTS += [f'artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/run01/candidates/{arm}_D_s{s}_r0_K0.json'
    for arm in ('CDF','XDF') for s in (1,4)]


def read(path): return json.loads((ROOT/path).read_text())
def rows(path): return list(csv.DictReader((ROOT/path).open()))


def main():
    # These identities distinguish execution SOURCE from result publication and
    # the later immutable snapshot used for this transcription.
    identities = {
        'H2': ('33c380d8aa792e5119bb3adec5dd07d1e1377214','cc0cdb52570993fa4f14f6eb199b583d0411d585'),
        'H4_OLD': ('002d0dc2e1dafd8cb71735c43253fa054e675f6d','6202f036457940532aeaa2873cd6d9e6f123c0ca'),
        'H4_NEW': ('18045333a1438740160d7b9e44e78139ad59f2b5',BASE),
        'H6': ('55b3d86817308a8d676221982aae13e1a6e3bed4','5385c6ef01879ee8a70734bfc0d5ae8999dd6186')}
    h2 = read(H2DF)['DF_errors']; h4 = read(H4DF)['saved_anchors']
    h6 = {r['state']:r for r in rows(H6DF)}
    out = []
    for row in rows(PRIOR):
        case = 'H4_OLD' if row['system']=='H4' else row['system']
        arm = row['arm']
        if case == 'H2': value, path = h2[arm], H2DF
        elif case == 'H4_OLD': value, path = h4[arm]['norm_plus_allowance_Ha'], H4DF
        else:
            key = 'corrected_CDF14_final' if arm=='CDF' else 'saved_XDF14_before'
            value, path = h6[key]['sector_norm_plus_guard_Ha'], H6DF
        row.update(comparison=case, DF_norm_plus_guard_Ha=str(value), DF_budget_status='PASS',
            df_sum_source_path=path, science_SOURCE=identities[case][0],
            result_publication_commit=identities[case][1], snapshot_commit=BASE)
        # The previous CSV's result_source_commit was a later snapshot, not
        # the computation's SOURCE. Preserve it explicitly as prior snapshot.
        row['prior_evidence_snapshot_commit'] = row.pop('result_source_commit')
        out.append(row)
    old = {r['arm']:r for r in out if r['comparison']=='H4_OLD'}
    for saved in rows(NEW):
        arm,s = saved['candidate'].split(':s'); row = dict(old[arm])
        row.update(comparison='H4_NEW', rank='8', s=s, status=saved['status'],
            energy=saved['energy_error_bound_pilot_Ha'], shots=saved['shots'],
            DF_norm_plus_guard_Ha=saved['DF_norm_plus_guard_Ha'], df_sum_source_path=NEW,
            science_SOURCE=identities['H4_NEW'][0], result_publication_commit=BASE,
            result_source_path=NEW)
        for metric in ['rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size']:
            row[metric] = saved[metric]
        if arm=='XDF':
            path='artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01/prefix_8.json'
            df=read(path);row.update(df_norm=str(df['sector_error_Ha']),df_guard=str(df['allowance_Ha']),df_source_path=path)
        row['prior_evidence_snapshot_commit']=''
        out.append(row)
    assert len(out)==16 and len({(r['comparison'],r['arm'],r['s']) for r in out})==16
    fields=['comparison','system','arm','rank','s','status','df_norm','df_guard',
        'DF_norm_plus_guard_Ha','DF_budget_status','energy','shots','rz_count','rz_depth',
        'cx_count','cx_depth','total_depth','circuit_size','science_SOURCE',
        'result_publication_commit','snapshot_commit','result_source_path','df_source_path',
        'df_sum_source_path','prior_evidence_snapshot_commit']
    with (HERE/'saved_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(out)
    e=read(str((HERE/'evidence_map.json').relative_to(ROOT)))
    for entry in e['documents']:
        actual=subprocess.check_output(['git','rev-parse',BASE+':'+entry['path']],cwd=ROOT,text=True).strip()
        assert actual==entry['github_blob_sha'],entry['path']
        assert subprocess.check_output(['git','hash-object',entry['path']],cwd=ROOT,text=True).strip()==actual
        entry['local_base_blob_sha']=actual;entry['remote_local_match']=True
    paths=INPUTS+[r['df_source_path'] for r in out]
    e.update(base_commit=BASE, transcription='16 saved rows; no new numerical evaluation or aggregation',
        identities={k:dict(science_SOURCE=v[0],result_publication_commit=v[1]) for k,v in identities.items()},
        scalar_input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(set(paths))},
        molecular_science_calls=0, important_scientific_review=False)
    (HERE/'evidence_map.json').write_text(json.dumps(e,indent=2,ensure_ascii=False)+'\n')
    print('16 saved rows transcribed; 13 GitHub document blobs match local base; science calls0')


if __name__=='__main__': main()
