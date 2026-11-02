"""One-shot 450-axis saved-value error analysis. No compile/model/fit APIs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import pf_q_model_diagnostics as d
from prepare_pf_q_model_comparison import csv_text
from h8_phase_b2_contract import axis_check

BASE = 'b777ac7ea16e0a2feff6f8b0864ed766d779770f'
PRIOR = 'artifacts/resource_applicability/track_a_pf_q_model_comparison_preparation/2026-10-20'
OUT = ROOT/'artifacts/resource_applicability/track_a_pf_q_model_diagnostics/2026-10-20'
SOURCE_FILES = ('src/trotterlib/pf_q_model_diagnostics.py', 'scripts/run_pf_q_model_diagnostics.py',
                'scripts/plot_pf_q_model_diagnostics.py', 'scripts/prepare_pf_q_model_comparison.py',
                'scripts/prepare_pf_direct_compile_inventory.py', 'scripts/h8_phase_b2_contract.py',
                'tests/test_pf_q_model_diagnostics.py')
INPUT_FILES = tuple(PRIOR+'/'+name for name in ('axis_correspondence.json', 'saved_prediction_records.json',
                   'model_registry.json', 'model_used_points.json', 'source_inventory.json',
                   'prior_source_inventory.json', 'excluded_48_axes.csv', 'publication_review.json'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def verify(contract):
    for name, digest in contract['input_hashes'].items():
        raw=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest or sha(ROOT/name)!=digest:
            raise ValueError('SAVED_INPUT_CHANGED:'+name)
    for name,digest in contract['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('ANALYSIS_SOURCE_CHANGED:'+name)


def seal():
    if OUT.exists():raise ValueError('FRESH_OUTPUT_REQUIRED')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    rows=json.loads((ROOT/INPUT_FILES[0]).read_text())['rows']
    selected=[r for r in rows if r['comparison_class']=='CONDITIONAL_COMPARISON']
    if Counter(r['system'] for r in selected)!={'H6_DF19':192,'H8_DF28':258}:raise ValueError('FIXED_SCOPE')
    value=dict(status='SEALED_SAVED_VALUE_ANALYSIS_AUTHORIZED',base_commit=BASE,SOURCE=head,
        scope='SAVED_450_AXIS_ERROR_AGGREGATION_VISUALIZATION_ONLY',sealed_UTC=datetime.now(timezone.utc).isoformat(),
        selected_task_ids=[r['task_id'] for r in selected],axes=450,
        input_hashes={n:sha(ROOT/n) for n in INPUT_FILES},source_hashes={n:sha(ROOT/n) for n in SOURCE_FILES},
        definitions=dict(difference='saved prediction minus directly measured RZ',
            absolute_error='abs(difference)',relative_error='difference / measured RZ',
            absolute_relative_error='abs(difference) / measured RZ',percent_multiplier=100,
            threshold='absolute relative error < 1/50 (strict; exactly 2 percent fails)',
            exact_arithmetic='Fraction of saved numeric literals; no model evaluation',
            summary='unweighted axis means; median, p95 type-7; max; observed threshold counts',
            groupings=d.GROUPINGS,axis_dependence='cosine/sine kept separately; no independence assumption',
            no_population_ci=True,seed=None,random_operations=False),
        rendering=dict(figures=['predicted_vs_measured','relative_error_distribution','error_vs_q','error_vs_prefix'],
            formats=['png','svg','pdf'],histogram_bins=20,range='full observed range; no clipping or outlier removal',
            scatter_roles='only selected axes; all existing-use roles retained',
            trend_lines='unweighted means of saved axes, descriptive, no fit'),
        scientific_call_scope=dict(saved_axis_errors=450,aggregate_passes=1,
            new_compile=0,new_sampling=0,new_model_prediction=0,fit=0,shot_cost_reaccounting=0,ranking=0),
        excluded=dict(other_preparation_axes=338,H2_signal_repair=False,old_profile_48_reacquisition=False),
        limitations=['H6 nominal profile mismatch','model-anchor/current runtime equivalence unproven',
            'S4 frozen composition hypothesis','initial 572 global final gate missing; 428 selected axes affected',
            'conditional diagnosis, not separation of intrinsic model and environment effects'],
        detailed_scientific_review=False,final_evidence_freeze=False,next_stage_authorized=False)
    verify(value)
    write('sealed_analysis_contract.json',value)
    write('grant.json',dict(status='APPROVED_BY_CURRENT_EXPLICIT_USER_INSTRUCTION',
        approval='保存済み450軸（H6：192軸、H8：258軸）の数値再解析を明示的に承認',
        SOURCE=head,contract_sha256=sha(OUT/'sealed_analysis_contract.json'),
        scientific_execution_authorized=True,new_publication_authorized=False,
        scope=value['scope'],retry=False,next_stage_authorized=False))
    print(json.dumps(dict(status='SEALED',SOURCE=head,axes=450)))


def run():
    contract=json.loads((OUT/'sealed_analysis_contract.json').read_text());verify(contract)
    grant=json.loads((OUT/'grant.json').read_text())
    if grant['contract_sha256']!=sha(OUT/'sealed_analysis_contract.json') or not grant['scientific_execution_authorized']:
        raise ValueError('APPROVED_GRANT_REQUIRED')
    marker=OUT/'one_shot.marker'
    with marker.open('x') as f:f.write(contract['SOURCE']+'\n')
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
    ledger=dict(SOURCE=contract['SOURCE'],status='STARTED',saved_axis_errors=0,new_compile=0,new_sampling=0,
                new_model_prediction=0,fit=0,shot_cost_reaccounting=0,ranking=0)
    write('ledger.json',ledger)
    try:
        table=json.loads((ROOT/INPUT_FILES[0]).read_text())['rows']
        predictions=json.loads((ROOT/INPUT_FILES[1]).read_text())['rows']
        rows=d.diagnose(table,predictions)
        if [r['task_id'] for r in rows]!=contract['selected_task_ids']:raise ValueError('SEALED_AXIS_ORDER')
        for row in rows:
            legacy=axis_check(row['predicted_RZ_exact'],row['measured_RZ'])
            if (legacy['exact_relative_error']!=row['absolute_relative_error_exact'] or
                    legacy['PASS']!=row['strictly_below_2_percent']):raise ValueError('LEGACY_THRESHOLD_MISMATCH')
            row['residuals_computed']=True
        grouped=d.aggregate(rows)
        write('axis_errors.json',dict(SOURCE=contract['SOURCE'],rows=rows))
        write('axis_errors.csv',csv_text(rows))
        write('grouped_errors.json',dict(SOURCE=contract['SOURCE'],rows=grouped))
        write('grouped_errors.csv',csv_text([dict(grouping=r['grouping'],**r['group'],**{k:v for k,v in r.items() if k not in ('group','grouping')}) for r in grouped]))
        priority=[r for r in rows if r['system']=='H6_DF19' and r['candidate_id']=='B0_p11_q4_2nd']
        write('H6_B0_p11_q4_2nd.json',dict(SOURCE=contract['SOURCE'],rows=priority))
        role_counts={role:sum(r['model_point_class']==role for r in rows) for role in
                     ('TRAINING_USED','VALIDATION_USED','OTHER_COMPARISON_POINT')}
        write('summary.json',dict(SOURCE=contract['SOURCE'],axes=450,systems={s:d.summarize([r for r in rows if r['system']==s]) for s in ('H6_DF19','H8_DF28')},
             all_axes=d.summarize(rows),model_point_class_counts=role_counts,
             original_final_gate_missing_axes=sum(r['original_global_final_gate_missing'] for r in rows),
             analysis_is_conditional=True,model_environment_effects_separated=False,
             cosine_sine_are_not_claimed_independent=True))
        verify(contract)
        ledger.update(status='COMPLETED_SAVED_NUMERICAL_ANALYSIS',saved_axis_errors=450,aggregate_passes=1,
            legacy_criterion_crosschecks=450,wall_seconds=time.monotonic()-started,
            cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime-before.ru_utime,
            process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write('ledger.json',ledger)
        write('numeric_completion.json',dict(status='COMPLETED_CONDITIONAL_DIAGNOSTIC_NOT_VALIDATION_CERTIFICATE',SOURCE=contract['SOURCE'],
             result_hashes={n:sha(OUT/n) for n in ('axis_errors.json','axis_errors.csv','grouped_errors.json','grouped_errors.csv','summary.json','H6_B0_p11_q4_2nd.json')},
             input_and_source_final_gate='PASS',prohibited_calls_all_zero=True,rendering_pending=True))
        print(json.dumps(dict(status=ledger['status'],axes=450,SOURCE=contract['SOURCE'])))
    except BaseException as e:
        ledger.update(status='STOP_TECHNICAL_ERROR',error_type=type(e).__name__,error=str(e));write('ledger.json',ledger)
        write('STOP.json',dict(status='STOP_TECHNICAL_ERROR',error_type=type(e).__name__,error=str(e),automatic_retry=False))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('seal','run'));a=p.parse_args()
    seal() if a.stage=='seal' else run()


if __name__=='__main__':main()
