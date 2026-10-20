"""Sealed one-shot ALL_SAVED458 accounting for the explicitly approved 70 scopes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import pf_selection_cost_comparison as core
from prepare_pf_q_model_comparison import csv_text

BASE='f63065fb636504b78d295f351ffc339370b9a532'
PREP='artifacts/resource_applicability/track_a_pf_selection_comparison_preparation/2026-10-20'
OUT=ROOT/'artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20'
INPUT_NAMES=('saved_condition_rows.json','axis_price_lineage.json','baseline_saved_comparisons.json',
    'condition_coverage.json','candidate_domains.json','source_inventory.json','input_bindings.json',
    'execution_contract_DRAFT.json','technical_source_manifest.json','STOP.json')
SOURCE_FILES=('src/trotterlib/pf_selection_cost_comparison.py','scripts/run_pf_selection_cost_comparison.py',
    'scripts/plot_pf_selection_cost_comparison.py','scripts/check_pf_selection_cost_comparison.py',
    'tests/test_pf_selection_cost_comparison.py','scripts/prepare_pf_q_model_comparison.py',
    'scripts/prepare_pf_direct_compile_inventory.py','src/trotterlib/h8_six_axis_reaccount.py')
PROHIBITED=('new_compile','new_sampling','new_model_prediction','new_fit','shot_recalculation',
            'PR_mean_SE_recalculation','candidate_addition','unresolved_H6_noncancellation_conditions_evaluated')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def input_data():
    return {name:read(ROOT/PREP/name) for name in INPUT_NAMES}


def selected_conditions(data):
    rows=[r for r in data['condition_coverage.json']['rows'] if r['strategy']=='ALL_SAVED458']
    selected=[r for r in rows if r['status']=='CONDITIONAL_INPUTS_READY']
    excluded=[r for r in rows if r['status']!='CONDITIONAL_INPUTS_READY']
    core.require(len(selected)==70 and len(excluded)==30,'EXACT_PREPARED_70_PLUS_EXCLUDED_30')
    core.require(Counter(r['system'] for r in selected)=={'H6_DF19':30,'H8_DF28':40},'EXACT_SYSTEM_SCOPE')
    core.require(all(r['bias_mode']=='signed_signal' for r in selected if r['system']=='H6_DF19'),'H6_SIGNED_ONLY')
    core.require(all(r['system']=='H6_DF19' and r['bias_mode']=='empirical_no_cancellation' for r in excluded),'KEEP_UNRESOLVED_30')
    core.require(len({core.endpoint(r) for r in selected})==30,'EXACT_30_ENDPOINTS')
    for row in selected:
        core.require(row['candidate_ids']==data['candidate_domains.json'][row['system']][row['original_scope']], 'NO_CANDIDATE_DOMAIN_CHANGE')
        core.require(not row['unresolved_PR_candidate_ids'],'SAVED_PR_INPUTS_COMPLETE')
    return selected,excluded


def verify(contract):
    for name,value in contract['input_hashes'].items():
        raw=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        core.require(hashlib.sha256(raw).hexdigest()==value and sha(ROOT/name)==value,'FIXED_INPUT_BYTES_CHANGED:'+name)
    for name,value in contract['source_hashes'].items():
        core.require(sha(ROOT/name)==value,'SEALED_SOURCE_CHANGED:'+name)
    for item in contract['underlying_saved_sources']:
        raw=subprocess.check_output(['git','show',item['commit']+':'+item['path']],cwd=ROOT)
        core.require(hashlib.sha256(raw).hexdigest()==item['sha256'],'UNDERLYING_FIXED_GIT_INPUT_CHANGED:'+item['path'])
        path=ROOT/item['path']
        if path.exists():core.require(sha(path)==item['sha256'],'UNDERLYING_WORKTREE_INPUT_CHANGED:'+item['path'])
    core.require(sha(Path(sys.executable))==contract['runtime']['executable_sha256'] and sys.version==contract['runtime']['python_version'],'NUMERIC_RUNTIME_CHANGED')
    for name,value in contract['runtime']['arithmetic_stdlib_hashes'].items():
        core.require(sha(name)==value,'STDLIB_CHANGED:'+name)


def seal():
    core.require(not OUT.exists(),'FRESH_OUTPUT_REQUIRED')
    data=input_data();conditions,excluded=selected_conditions(data)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for name in SOURCE_FILES:
        raw=subprocess.check_output(['git','show',head+':'+name],cwd=ROOT)
        core.require(hashlib.sha256(raw).hexdigest()==sha(ROOT/name),'COMMITTED_SOURCE_REQUIRED:'+name)
    import fractions,decimal,json.encoder,json.decoder
    modules=(fractions,decimal,json.encoder,json.decoder)
    runtime=dict(executable=sys.executable,executable_sha256=sha(sys.executable),python_version=sys.version,
        platform=platform.platform(),arithmetic_stdlib_hashes={m.__file__:sha(m.__file__) for m in modules},
        process_affinity=sorted(os.sched_getaffinity(0)),new_resource_caps=False,
        scheduling='One indexed exact-arithmetic pass; reuse endpoint totals across scopes; no shared settings modified')
    value=dict(status='SEALED_APPROVED_ALL_SAVED458_70_CONDITIONS',SOURCE=head,base_commit=BASE,
        strategy='ALL_SAVED458',scope_conditions=70,unique_endpoints=30,conditions=conditions,
        condition_ids=[core.condition_id(c) for c in conditions],condition_manifest_sha256=digest(conditions),
        excluded_unresolved_conditions=excluded,excluded_conditions_sha256=digest(excluded),
        input_hashes={PREP+'/'+n:sha(ROOT/PREP/n) for n in INPUT_NAMES},
        underlying_saved_sources=data['source_inventory.json'],
        source_hashes={n:sha(ROOT/n) for n in SOURCE_FILES},runtime=runtime,
        definitions=dict(weighted_cost='N_cosine*saved_RZ_cosine + N_sine*saved_RZ_sine; Fraction exact arithmetic',
            baseline='Saved mixed measured/model policy, not pure-model counterfactual',
            replay_gate='Every eligible baseline PF G, every saved baseline PF winner/tie and PR selection/cost/SE/ratio must agree exactly before replacement',
            selection='Fraction G ascending then lexical candidate_id; keep all exact ties',
            ranks='ordinal with lexical ties and competition rank; strict pairwise inversions and tie changes separately',
            PR='Copy saved selected PR total/SE/cohort without recomputing means, SE, shots or selecting new PR',
            relation='PR_total/PF_total <1:PR_LOWER, >1:PF_LOWER, =1:TIE',
            unchanged_policies=['saved shots','bias eligibility','candidate sets','P=0','T=0.8','control','cohorts']),
        call_scope=dict(PF_accounting_passes=1,baseline_and_replacement_PF_selections=140,scope_rank_comparisons=70,
            **{k:0 for k in PROHIBITED}),
        rendering=dict(figures=['PF_cost_change','PR_PF_ratios'],formats=['PNG','SVG','PDF'],
            no_outlier_removal=True,scope_conditions=70),
        limitations=['H6 profile differs','H8/model runtime identity unproved','S4 composition hypothesis',
            'initial572 missing global final gate (428 of new450 price axes)','old pending-profile48 retained only',
            'conditional historical saved-cohort comparison; no population superiority guarantee'],
        new_publication_authorized=False,detailed_scientific_review=False,final_evidence_freeze=False,
        next_stage_authorized=False,sealed_UTC=datetime.now(timezone.utc).isoformat())
    verify(value)
    write('sealed_contract.json',value)
    write('grant.json',dict(status='APPROVED_BY_EXPLICIT_CURRENT_USER_INSTRUCTION',SOURCE=head,
        strategy='ALL_SAVED458',scientific_execution_authorized=True,contract_sha256=sha(OUT/'sealed_contract.json'),
        approved_scope='入力準備済み70件（重複を除く30条件）の保存shot込み再会計・PF選択・PR/PF比・順位比較',
        retry=False,new_publication_authorized=False,next_stage_authorized=False))
    write('excluded_unresolved30.json',dict(status='EXCLUDED_NOT_EXECUTED',rows=excluded,
        numeric_analysis_performed=False,original_coverage_sha256=digest(excluded)))
    print(json.dumps(dict(status='SEALED',SOURCE=head,conditions=70,endpoints=30)))


def run():
    contract=read(OUT/'sealed_contract.json');grant=read(OUT/'grant.json')
    verify(contract);core.authorize(contract,grant)
    core.require(grant['contract_sha256']==sha(OUT/'sealed_contract.json'),'CONTRACT_GRANT_BINDING')
    with (OUT/'one_shot.marker').open('x') as f:f.write(contract['SOURCE']+'\n')
    started=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF)
    ledger=dict(SOURCE=contract['SOURCE'],status='STARTED',PF_accounting_passes_reserved=1,
        PF_accounting_passes_completed=0,baseline_gate='NOT_COMPLETED',**{k:0 for k in PROHIBITED})
    write('ledger.json',ledger)
    try:
        data=input_data();conditions,excluded=selected_conditions(data)
        core.require(conditions==contract['conditions'] and digest(excluded)==contract['excluded_conditions_sha256'],'SEALED_SCOPE_CHANGED')
        rows=data['saved_condition_rows.json']['rows'];units={u['task_id']:u for u in data['axis_price_lineage.json']['rows']}
        row_digest=digest(rows)
        costs=core.replay_pf(rows,units,{core.endpoint(c) for c in conditions})
        gates=core.baseline_gate(costs,rows,data['baseline_saved_comparisons.json']['rows'],conditions)
        write('baseline_replay_gate.json',dict(status='PASS_COMPLETE_EXACT_BASELINE_GATE',SOURCE=contract['SOURCE'],
            candidate_rows=len(costs),eligible_PF_totals_reproduced=sum(r['eligible'] for r in costs),scope_winner_and_ratio_matches=len(gates),
            all_candidate_and_saved_winner_G_exact=True,PR_totals_SE_cohort_preserved=True,
            replacement_accounting_started=False))
        ledger['baseline_gate']='PASS';write('ledger.json',ledger)
        used=core.replace_costs(costs,units)
        comparisons,ranks,changes=core.compare(costs,conditions,gates)
        core.require(len(comparisons)==70 and digest(rows)==row_digest,'RESULT_COUNT_OR_SAVED_ROW_MUTATION')
        write('candidate_totals.json',dict(SOURCE=contract['SOURCE'],rows=costs))
        write('candidate_totals.csv',csv_text(costs))
        write('selection_comparison.json',dict(SOURCE=contract['SOURCE'],rows=comparisons))
        flat=[]
        for row in comparisons:
            value={k:v for k,v in row.items() if k not in ('baseline','replacement')}
            for side in ('baseline','replacement'):
                value.update({side+'_'+k:v for k,v in row[side].items()})
            flat.append(value)
        write('selection_comparison.csv',csv_text(flat))
        write('candidate_ranks.json',dict(SOURCE=contract['SOURCE'],rows=ranks))
        write('candidate_ranks.csv',csv_text(ranks))
        write('rank_relation_changes.json',dict(SOURCE=contract['SOURCE'],rows=changes))
        write('rank_relation_changes.csv',csv_text(changes))
        group_keys=sorted({(r['system'],r['bias_mode'],r['original_scope']) for r in comparisons})
        summary=dict(SOURCE=contract['SOURCE'],all_scopes=core.summarize(comparisons),
            groups=[dict(system=s,bias_mode=b,original_scope=o,**core.summarize([r for r in comparisons
                if (r['system'],r['bias_mode'],r['original_scope'])==(s,b,o)])) for s,b,o in group_keys],
            unique_endpoints=30,condition_counts_are_overlapping_scopes=True,
            used_axis_prices=dict(Counter(units[tid]['ALL_SAVED458'] for tid in used)),
            eligible_candidate_totals_replayed=sum(r['eligible'] for r in costs),unique_candidate_condition_rows=len(costs),
            model_environment_effects_separated=False,conditional_comparison=True,excluded30_not_evaluated=True)
        write('summary.json',summary)
        write('used_price_evidence.json',dict(SOURCE=contract['SOURCE'],rows=[units[tid] for tid in sorted(used)],
            available_new450=450,available_extra8=8,available_retained102=102))
        verify(contract)
        ledger.update(status='COMPLETED_APPROVED_70_SCOPE_COMPARISON',PF_accounting_passes_completed=1,
            PF_candidate_condition_rows=len(costs),scope_comparisons=70,baseline_and_replacement_PF_selections=140,
            pairwise_rank_comparisons=sum(r['pairwise_comparisons'] for r in comparisons),
            wall_seconds=time.monotonic()-started,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime-cpu.ru_utime,
            process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            input_and_SOURCE_final_gate='PASS')
        write('ledger.json',ledger)
        names=['candidate_totals.json','candidate_totals.csv','selection_comparison.json','selection_comparison.csv',
            'candidate_ranks.json','candidate_ranks.csv','rank_relation_changes.json','rank_relation_changes.csv',
            'summary.json','used_price_evidence.json','baseline_replay_gate.json','excluded_unresolved30.json']
        write('numeric_completion.json',dict(status='NUMERIC_COMPLETE_CONDITIONAL_COMPARISON',SOURCE=contract['SOURCE'],
            result_sha256={n:sha(OUT/n) for n in names},baseline_gate='PASS',input_and_SOURCE_gate='PASS',
            prohibited_calls_zero=True,rendering_pending=True,mandatory_STOP_after_audit=True))
        print(json.dumps(dict(status=ledger['status'],SOURCE=contract['SOURCE'],conditions=70)))
    except BaseException as error:
        ledger.update(status='STOP_NO_AUTOMATIC_RETRY',error_type=type(error).__name__,error=str(error));write('ledger.json',ledger)
        write('STOP.json',dict(status='STOP_NUMERIC_OR_TECHNICAL_MISMATCH',error_type=type(error).__name__,error=str(error),retry=False,next_stage_authorized=False))
        raise


def launch():
    contract=read(OUT/'sealed_contract.json');verify(contract)
    with (OUT/'launch.marker').open('x') as f:f.write(contract['SOURCE']+'\n')
    started=time.monotonic()
    with (OUT/'numeric_stdout.log').open('wb') as stdout,(OUT/'numeric_stderr.log').open('wb') as stderr:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT,stdout=stdout,stderr=stderr)
        write('native_started.json',dict(pid=child.pid,parent_pid=os.getpid(),SOURCE=contract['SOURCE'],started_UTC=datetime.now(timezone.utc).isoformat()))
        pid,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
    write('native_exit_evidence.json',dict(status='PASS_NATIVE_EXIT0_REAP' if code==0 else 'STOP_NATIVE_ERROR',
        SOURCE=contract['SOURCE'],worker_pid=pid,parent_pid=os.getpid(),native_exit_code=code,
        same_parent_wait4=True,reaped=True,wall_seconds=time.monotonic()-started,
        child_CPU_seconds=usage.ru_utime+usage.ru_stime,child_peak_RSS_bytes=usage.ru_maxrss*1024,
        stdout_sha256=sha(OUT/'numeric_stdout.log'),stderr_sha256=sha(OUT/'numeric_stderr.log'),
        automatic_retry=False,live_scientific_workers_after_wait=0))
    if code:raise SystemExit(code)
    print(json.dumps(dict(status='PASS_OWNED_CHILD_NATIVE_EXIT0',pid=pid,wall_seconds=time.monotonic()-started)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('seal','run','launch'));args=parser.parse_args()
    {'seal':seal,'run':run,'launch':launch}[args.stage]()


if __name__=='__main__':main()
