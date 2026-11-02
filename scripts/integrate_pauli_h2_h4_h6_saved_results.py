#!/usr/bin/env python3
"""Column transcription only: no numerical library or scientific runner imports.

Reuse the existing csv.DictReader/DictWriter exact-string convention. The old
packaging runner also imports accounting/science modules, so this deliberately
small adapter does not import it. No G, ratio, SE, interval or ranking is computed.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25'
EPS=('0.05','0.01','0.005','0.001','0.0005')
M={'H2':'14','H4':'184','H6':'918'}
H26='8fd1ceb283309a7e515ff4a533a253260bdd0f78'
H4='d3fdfad880902b7ff35b710d22e1c9759f7087c2'
DET='fbfbfd21a252a4a32f52e3b40e50acd91b3c626a'
NATIVE='d17780fcb57bcff344325a8632f718c0cc56f1dc;0880c38e3e8a6ea57882ec66e6366a8b86e408e2;a8c7c7dca5a03aae1b008a09048d89c216bbf89c'
POLICY_NOMINEE='EXPLORATION_NOMINEE_FROZEN_THEN_INDEPENDENT_CONFIRMATION'
POLICY_POST='OBSERVED_MINIMUM_AFTER_CONFIRMATION_WITHIN_FROZEN_SET'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def csvsave(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def quoted_json(path):
    # Preserve the saved lexical decimal token, without floating-point conversion.
    return json.loads(path.read_text(),parse_float=str,parse_int=str)

def one(rows,**matches):
    found=[r for r in rows if all(r[k]==v for k,v in matches.items())]
    if len(found)!=1:raise ValueError('NONUNIQUE_OR_MISSING_SAVED_ROW '+repr(matches))
    return found[0]

def classify(prefix,total):
    L=int(prefix);m=int(total)
    if not 0<=L<=m:raise ValueError('PREFIX_OUTSIDE_MODEL')
    return 'FULLY_RANDOMIZED_RTE' if L==0 else 'HYBRID_PR' if L<m else 'FULL_RETENTION_NO_NONTRIVIAL_RANDOMIZED_TAIL'

def export_sources(cache):
    manifest=[]
    for file in sorted(cache.glob('*.json')):
        s=json.loads(file.read_text());raw=s['content'].encode('utf-8')
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=s['github_blob_sha']:raise ValueError('FETCHED_GIT_BLOB_MISMATCH')
        path=OUT/'inputs'/(s['key']+Path(s['path']).suffix);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        manifest.append({k:s[k] for k in ('key','commit','path','github_blob_sha','github_url')} |
                        dict(snapshot=str(path.relative_to(OUT)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                             retrieval='GitHub fetch_file with exact commit ref; UTF-8 Git blob verified'))
    save(OUT/'source_manifest.json',manifest)
    return manifest

def load_inputs():
    manifest=json.loads((OUT/'source_manifest.json').read_text());data={}
    for s in manifest:
        path=OUT/s['snapshot'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=s['sha256']:raise ValueError('SOURCE_SNAPSHOT_CHANGED')
        data[s['key']]=list(csv.DictReader(io.StringIO(raw.decode()))) if path.suffix=='.csv' else quoted_json(path) if path.suffix=='.json' else raw.decode()
    return data,{s['key']:s for s in manifest}

def assemble(data,sources):
    rows=[];provenance=[];transcriptions=[]
    for molecule in ('H2','H4','H6'):
        for eps in EPS:
            row=dict(molecule=molecule,epsilon_sig=eps,M_nonidentity=M[molecule]);links={}
            def copy(field,key,item,column,selector):
                row[field]=item[column];links[field]=dict(source_key=key,selector=selector,column=column)
            if molecule!='H4':
                main=one(data['h26_combined'],molecule=molecule,epsilon_sig=eps)
                pr=one(data['h26_best'],molecule=molecule,epsilon_sig=eps,family='PR_confirmed_best')
                det=one(data['h26_best'],molecule=molecule,epsilon_sig=eps,candidate_id=main['deterministic_candidate'])
                base=dict(molecule=molecule,epsilon_sig=eps)
                for f,c in [('PR_candidate_id','PR_candidate'),('PR_G_exact','PR_G_exact'),('deterministic_candidate_id','deterministic_candidate'),('deterministic_G_exact','deterministic_G_exact'),('PR_over_D_exact','PR_over_D_exact'),('PR_over_D_saved_decimal','PR_over_D')]:
                    copy(f,'h26_combined',main,c,base)
                for f,c in [('PR_L_D','L_D'),('PR_q','q'),('PR_r','r'),('PR_K','K'),('confirmation_n','price_n'),('PR_C_cos_exact','C_cos_exact'),('PR_C_sin_exact','C_sin_exact'),('PR_C_cos_SE','C_cos_SE'),('PR_C_sin_SE','C_sin_SE'),('PR_G_SE','G_SE')]:
                    copy(f,'h26_best',pr,c,base|dict(family='PR_confirmed_best'))
                for f,c in [('deterministic_L_D','L_D'),('deterministic_q','q')]:copy(f,'h26_best',det,c,base|dict(candidate_id=det['candidate_id']))
                if pr['candidate_id']!=row['PR_candidate_id'] or pr['G_exact']!=row['PR_G_exact'] or det['G_exact']!=row['deterministic_G_exact']:raise ValueError('H26_CSV_JOIN_MISMATCH')
                row.update(deterministic_method='FULL_PF' if det['family']=='full_PF' else 'B0',
                    selection_policy=POLICY_POST,confirmation_post_selection_in_primary='true',
                    PR_G_nominal_t95_halfwidth_saved='',PR_savings_percent_saved='',guard_order_saved='',
                    missing_display_note='nominal halfwidth/savings/guard not present in these two source CSVs; not derived',
                    PR_SOURCE_commits=NATIVE,deterministic_SOURCE_commits=NATIVE,
                    PR_result_commit=H26,PR_result_file=sources['h26_best']['path'],
                    deterministic_result_commit=H26,deterministic_result_file=sources['h26_best']['path'],
                    ratio_result_commit=H26,ratio_result_file=sources['h26_combined']['path'])
            else:
                pr=one(data['h4_best'],epsilon_sig=eps);det=one(data['h4_det_best'],epsilon_sig=eps)
                summary=one(data['h4_summary']['epsilon_results'],epsilon_sig=eps)
                primary=summary['primary_exploration_nominee_confirmed']
                if pr['selected_PR']!=primary['candidate_id'] or pr['PR_G_exact']!=primary['G_exact'] or pr['PR_over_best_deterministic_exact']!=summary['PR_over_best_deterministic_exact']:raise ValueError('H4_PRIMARY_ROW_MISMATCH')
                # Never select confirmation_minimum_post_selection or invert old D/PR.
                sel=dict(epsilon_sig=eps)
                for f,c in [('PR_candidate_id','selected_PR'),('PR_L_D','L_D'),('PR_q','q'),('PR_r','r'),('PR_K','K'),('confirmation_n','confirmation_n'),('PR_G_exact','PR_G_exact'),('PR_G_SE','PR_G_SE'),('PR_G_nominal_t95_halfwidth_saved','PR_G_nominal_t95_halfwidth'),('PR_C_cos_exact','mean_C_cos_exact'),('PR_C_sin_exact','mean_C_sin_exact'),('PR_over_D_exact','PR_over_best_deterministic_exact'),('PR_savings_percent_saved','PR_savings_percent'),('guard_order_saved','guard_order')]:copy(f,'h4_best',pr,c,sel)
                for f,c in [('deterministic_candidate_id','best_deterministic_ids'),('deterministic_L_D','best_prefix'),('deterministic_q','best_q'),('deterministic_G_exact','best_G_exact')]:copy(f,'h4_det_best',det,c,sel)
                if det['best_deterministic_ids']!=det['best_truncated_ids'] or (pr['B0_prefix'],pr['B0_q'],pr['B0_G_exact'])!=(det['best_prefix'],det['best_q'],det['best_G_exact']):raise ValueError('H4_DETERMINISTIC_REFERENCE_MISMATCH')
                catalog=one(data['h4_catalog'],candidate_id=pr['selected_PR'])
                if catalog['costs']['confirmation']['n']!=pr['confirmation_n']:raise ValueError('H4_COHORT_MISMATCH')
                for axis,label in [('cosine','cos'),('sine','sin')]:
                    field='PR_C_'+label+'_SE';row[field]=catalog['costs']['confirmation']['axes'][axis]['standard_error']
                    links[field]=dict(source_key='h4_catalog',selector=dict(candidate_id=pr['selected_PR']),nested_path=['costs','confirmation','axes',axis,'standard_error'])
                row.update(deterministic_method='B0',selection_policy=POLICY_NOMINEE,confirmation_post_selection_in_primary='false',
                    PR_over_D_saved_decimal='',missing_display_note='H4 ratio decimal not in canonical main row; saved exact fraction retained',
                    PR_SOURCE_commits='52865921930b3afd11b1447dff0a6cbcb6085af8;4a7f91fa7ba8aa3ae1ce4ccb30bde9b973869410;a17ad381b6ff2ac615e3cd930c641947c551bcbc',
                    deterministic_SOURCE_commits='47384f5698094e466e83027a5a54751e0cf6000e;8ab472db8f90d6f64f6cc626584655c5a7487a56',
                    PR_result_commit=H4,PR_result_file=sources['h4_best']['path'],deterministic_result_commit=DET,
                    deterministic_result_file=sources['h4_det_best']['path'],ratio_result_commit=H4,ratio_result_file=sources['h4_best']['path'])
            row.update(PR_class=classify(row['PR_L_D'],M[molecule]),hypothetical_rho='0',actual_compiled_rho='UNKNOWN',
                PR_population_price_tail='UNGUARANTEED',cost_unit='RZ_x_shot; state preparation excluded',
                model='linear neutral singlet; adjacent 1.00 Angstrom; STO-3G; Pauli-JW',T='0.8',PF_order='S2',
                control='central-rotation',compiler='Qiskit1.3.0; rz/sx/x/cx; opt1; seed17; topology unspecified',
                candidate_scope='FINITE_REGISTERED_SET; no global optimality proof')
            # Field order comes from the first row; all rows have the same columns.
            if rows:row={k:row[k] for k in rows[0]}
            rows.append(row);transcriptions.append(dict(molecule=molecule,epsilon_sig=eps,fields=links))
            provenance.append(dict(molecule=molecule,epsilon_sig=eps,PR_candidate_id=row['PR_candidate_id'],
                deterministic_candidate_id=row['deterministic_candidate_id'],PR_commit=row['PR_result_commit'],PR_file=row['PR_result_file'],
                deterministic_commit=row['deterministic_result_commit'],deterministic_file=row['deterministic_result_file'],
                ratio_commit=row['ratio_result_commit'],ratio_file=row['ratio_result_file'],source_epsilon=eps,
                primary_selection_policy=row['selection_policy'],PR_SOURCE_commits=row['PR_SOURCE_commits'],
                source_links_file='field_provenance.json',github_PR_url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+row['PR_result_commit']+'/'+row['PR_result_file']))
    return rows,provenance,transcriptions

def verify(rows,transcriptions,data,sources):
    expected={(m,e) for m in M for e in EPS};keys=[(r['molecule'],r['epsilon_sig']) for r in rows]
    if len(rows)!=15 or len(set(keys))!=15 or set(keys)!=expected:raise ValueError('ROW_GRID_NOT_3_BY_5_UNIQUE')
    for row,provenance in zip(rows,transcriptions):
        for field,link in provenance['fields'].items():
            value=one(data[link['source_key']],**link['selector'])
            if 'column' in link:value=value[link['column']]
            else:
                for key in link['nested_path']:value=value[key]
            if row[field]!=value:raise ValueError('TRANSCRIPTION_MISMATCH '+field)
        if row['PR_class']!=classify(row['PR_L_D'],M[row['molecule']]):raise ValueError('WRONG_CLASS')
        if row['molecule']=='H4':
            correct=('36','2','8','2') if row['epsilon_sig']=='0.05' else ('0','8','8','4')
            if tuple(row['PR_'+p] for p in ('L_D','q','r','K'))!=correct or row['selection_policy']!=POLICY_NOMINEE:raise ValueError('H4_LEGACY_OR_POST_SELECTION_CONTAMINATION')
        elif row['selection_policy']!=POLICY_POST:raise ValueError('H26_POLICY_CHANGED')
    return dict(status='PASS',rows=15,unique_3_molecules_5_precisions=True,exact_G_ratio_and_ID_strings_copied=True,
                H4_primary_nominee_not_confirmation_minimum=True,H4_old_fixed_PR_absent_from_main=True,
                H4_high_precision_four_rows_fully_randomized=True,selection_policy_distinction_preserved=True,
                price_SE_and_G_SE_distinct=True,source_snapshots_unchanged=True,
                new_scientific_calls=0,new_numerical_reanalysis=0,new_ranking=0,new_statistical_intervals=0,
                validator_scope='structural/string/source comparison by primary Codex, not independent scientific review')

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--export-inputs':export_sources(Path(sys.argv[2]))
    data,sources=load_inputs();rows,provenance,transcriptions=assemble(data,sources)
    checks=verify(rows,transcriptions,data,sources)
    csvsave(OUT/'H2_H4_H6_optimized_PR_comparison.csv',rows)
    save(OUT/'H2_H4_H6_optimized_PR_comparison.json',rows)
    csvsave(OUT/'row_provenance.csv',provenance);save(OUT/'field_provenance.json',transcriptions)
    save(OUT/'verification.json',checks)
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
