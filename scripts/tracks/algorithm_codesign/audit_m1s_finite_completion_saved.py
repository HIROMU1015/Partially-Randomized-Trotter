#!/usr/bin/env python3
"""Saved-result audit/formatting only; never import or rerun numerical kernels."""
from pathlib import Path
import ast,collections,csv,hashlib,json,math,subprocess,sys
from fractions import Fraction
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
BASE='artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16'
OUT='artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16'

def main():
    checks=[]
    def check(n,v):
        checks.append({'name':n,'status':'PASS' if v else 'FAIL'})
        if not v:raise AssertionError(n)
    def sha(b):return hashlib.sha256(b).hexdigest()
    def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
    def load(p):
        def pairs(items):
            d={}
            for k,v in items:
                if k in d:raise ValueError(k)
                d[k]=v
            return d
        return json.loads((ROOT/p).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    def rational(d):return Fraction(int(d['numerator_hex'],16),int(d['denominator_hex'],16))
    final_dir=OUT+'/completion_attempt_02';result_path=final_dir+'/result.json';d=load(result_path);c=d['config'];limits=c['resources']
    original=load('artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/finite_signal_contract_v2_adopted.json')
    check('completed_status',d['status']=='PILOT_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT' and 'failure' not in d)
    check('raw_result_sidecar',sha((ROOT/result_path).read_bytes())==(ROOT/final_dir/'result.sha256').read_text().strip())
    check('config_is_fixed_contract',c==load(d['audit']['contract_path']))
    for k in set(original)-{'source_commit','source_sha256','adoption_record'}:check('scientific_contract_field_unchanged:'+k,c[k]==original[k])
    for p,h in c['source_sha256'].items():
        raw=(ROOT/p).read_bytes();ast.parse(raw);check('source_frozen:'+p,sha(raw)==h and raw==git('show',c['source_commit']+':'+p))
    for a,b in [(c['source_commit'],d['audit']['contract_commit']),(d['audit']['contract_commit'],d['audit']['authorization_commit'])]:
        check('strict_S_C_A_order:'+a,a!=b and subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',a,b]).returncode==0)
    context=load(final_dir+'/execution_context.json')
    check('exclusive_context_consumed',context['one_shot_consumed'] is True and context['automatic_retry'] is False and context['audit']==d['audit'])
    sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.algorithm_codesign.m1s_finite_signal_v2.control import planned_rows
    plan=planned_rows(c);descriptor_keys=set(plan[0]);rows=d['rows']
    check('108_unique_rows',len(rows)==108 and len({tuple(r[k] for k in ['split','R','delta','T','method']) for r in rows})==108)
    expected={tuple(r[k] for k in sorted(descriptor_keys)) for r in plan}
    check('fixed_plan_all_rows',expected=={tuple(r[k] for k in sorted(descriptor_keys)) for r in rows})
    check('zero36_nonzero72',sum(r['T']=='0' for r in rows)==36 and sum(r['T']!='0' for r in rows)==72)
    eps=Fraction(1,1<<24);phase_eps=Fraction(1,1<<18)
    for i,r in enumerate(rows):
        check('row_evaluated:'+str(i),r['status']=='NUMERIC_DIAGNOSTIC_ONLY' and len(r['z'])==2 and all(math.isfinite(x) for x in r['z']))
        check('row_signal_precision:'+str(i),0<=Fraction(r['signal_error_radius'])<=eps)
        check('row_evidence:'+str(i),r['evidence']=='CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
        if r['phase_status']=='RESOLVED_PRINCIPAL_PHASE':check('phase_precision:'+str(i),0<=Fraction(r['phase_error_radius_rad'])<=phase_eps)
        else:check('unresolved_phase_null:'+str(i),r['phase_status']=='UNDETERMINED_PHASE' and r['phase_bias_rad'] is None)
        ledger=r['error_ledger'];model=ledger['Hamiltonian_model_difference'];B=rational(model['operator_bound_Ha_exact']);eta=rational(model['signal_bound_abs_exact'])
        check('model_eta_exact:'+str(i),eta==abs(Fraction(r['T']))*B and B>=0)
        check('numerics_model_separate:'+str(i),ledger['numerical_rounding']['precision_gate_excludes_model_drift'] is True and ledger['numerical_rounding']['signal_radius']==r['signal_error_radius'])
        check('no_algorithm_point_claim:'+str(i),ledger['finite_step_algorithm_error']['algorithm_only_error_measured'] is False and ledger['finite_step_algorithm_error']['point_value'] is None)
        check('total_raw_H_phase_label:'+str(i),r['phase_interpretation']=='TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H')
        if r['T']=='0':check('zero_control_encloses_one:'+str(i),abs(complex(*r['z'])-1)<=r['signal_error_radius'])
    check('first_correctness_PASS',d['numerical_correctness']['status']=='PASS' and d['numerical_correctness']['Walsh_direct_twirl']=='PASS')
    for method,v in d['numerical_correctness']['enumerated_words'].items():check('physical_words_PASS:'+method,v['status']=='PASS')
    check('all3_full_twirl_PASS',set(d['full_size_twirl_correctness'])=={'G3','G6','G9'} and all(v['status']=='PASS' for v in d['full_size_twirl_correctness'].values()))
    check('48_contrasts',len(d['comparisons'])==48)
    for i,x in enumerate(d['comparisons']):
        check('same_split_contrast:'+str(i),x['comparison_target']=='COMMON_H_TILDE_WITHIN_SPLIT' and x['model_drift_in_numerical_radius'] is False and x['relative_visibility_improvement'] is None and x['scientific_GO'] is None)
    for key,limit in [('matrix_products','max_matrix_products'),('twirl_applications','max_twirl_applications'),('exact_events','max_exact_events')]:check('resource:'+key,d['resources'][key]<=limits[limit])
    check('recorded_memory_cap',d['resources']['actual_address_space_cap']<=limits['memory_bytes'])
    check('RSS_within_cap',d['resources']['maxrss_linux_KiB']*1024<=limits['memory_bytes'])
    for name,cap in [('result.json','result_bytes'),('runner.log','log_bytes'),('execution_context.json','context_bytes')]:check('file_cap:'+name,(ROOT/final_dir/name).stat().st_size<=limits[cap])
    check('no_scientific_or_platform_GO',d['scientific_GO_or_no_GO_determined'] is False and d['platform_certified'] is False and d['auxiliary_split_ideal_signal_evaluated'] is False)
    first=load(OUT+'/completion_attempt_01/result.json')
    check('failed_attempt_01_preserved',sha((ROOT/OUT/'completion_attempt_01/result.json').read_bytes())=='ef99d39d1457eb874ed6d242a82247ec7b7a7de7a6c213634f18a8bf1cdf9bed' and first['status']=='TECHNICAL_STOP')
    centres={tuple(r[k] for k in ['split','R','delta','T','method']):r['z'] for r in rows}
    check('all16_overlapping_signal_centres_bitwise_unchanged',all([x.hex() for x in r['z']]==[x.hex() for x in centres[tuple(r[k] for k in ['split','R','delta','T','method'])]] for r in first['rows'] if 'z' in r))
    for attempt in ['01','02']:
        directory=BASE+'/attempt_'+attempt
        markers=load(directory+'/external_record_identity.json') if attempt=='02' else load(directory+'/failure_preservation_and_next_repair.json')['external_records']
        for row in markers:check('external_copy_identity:'+row['copy_path'],sha(Path(row['external_path']).read_bytes())==sha((ROOT/row['copy_path']).read_bytes())==row['sha256'])
        preflight=load(directory+'/runtime_preflight.json');completion=load(directory+'/launch_completion.json')
        check('single_CPU_GPU_off:'+attempt,preflight['GPU'] is False and all(x=='1' for x in preflight['thread_environment'].values()))
        check('one_launch_each_attempt:'+attempt,completion['run_attempts']==1 and completion['one_shot_consumed'] is True and completion['child_returncode']==0)
    from jsonschema import Draft202012Validator
    schema=load(c['result_schema']['path']);Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(d)
    check('result_JSON_schema_valid',True)
    summary={'status':d['status'],'model':c['model'],'grid':c['grid'],'SOURCE':c['source_commit'],'contract_commit':d['audit']['contract_commit'],'authorization_commit':d['audit']['authorization_commit'],
      'result_path':result_path,'result_sha256':sha((ROOT/result_path).read_bytes()),'result_bytes':(ROOT/result_path).stat().st_size,
      'completed_rows':len(rows),'phase_status_counts':dict(collections.Counter(r['phase_status'] for r in rows)),
      'max_signal_error_radius':max(r['signal_error_radius'] for r in rows),'signal_precision_target':'2^-24',
      'max_phase_error_radius_rad':max((r.get('phase_error_radius_rad') or 0) for r in rows),'phase_precision_target_rad':'2^-18',
      'contrast_counts_by_split':{s:dict(collections.Counter(x['contrast_status'] for x in d['comparisons'] if x['split']==s)) for s in ['G3','G6','G9']},
      'correctness':'PASS','full_size_twirl':'PASS_ALL3','resources':d['resources'],
      'scientific_attempts':2,'attempt01':'TECHNICAL_STOP_PRECISION_BUDGET; preserved','attempt02':'COMPLETE',
      'science_conditions_and_thresholds_unchanged':True,'evidence':'CONDITIONAL_ENGINEERING_NOT_CERTIFIED','scientific_GO_determined':False,'mandatory_STOP':True}
    target=ROOT/BASE/'final';target.mkdir(exist_ok=True)
    (target/'saved_result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    fields=['split','R','delta','T','method','visibility','visibility_error_radius','signal_error_radius','phase_status','phase_bias_rad','phase_error_radius_rad','total_visibility_loss_relative_to_raw_H']
    with (target/'signals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['z_real','z_imag'],lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in fields}|{'z_real':r['z'][0],'z_imag':r['z'][1]})
    fields=['split','R','delta','T','comparator','visibility_difference_shared_minus_comparator','absolute_error_radius','contrast_status']
    with (target/'contrasts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:x[k] for k in fields} for x in d['comparisons'])
    report={'schema':'m1s_finite_completion_saved_result_audit','PASS':len(checks),'FAIL':0,'checks':checks,
      'new_numerical_kernels_called':0,'new_scientific_runs_by_auditor':0,'basis':'saved JSON/CSV/schema/hashes only','mandatory_STOP':True}
    (target/'saved_result_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'saved_result_audit_PASS':len(checks),'FAIL':0,**{k:summary[k] for k in ['status','result_sha256','completed_rows','max_signal_error_radius','max_phase_error_radius_rad','contrast_counts_by_split']}}))
if __name__=='__main__':main()
