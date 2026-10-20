"""Commit-bound one-shot stage B; fixed stage A shots and previous70 stay intact."""
import argparse
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import h6_noncancellation_stage_b as core

BASE='c585cef8990ef28f4d43c13dfaf2a0d423fda5a7'
B='artifacts/resource_applicability/'
A=B+'track_a_h6_noncancellation_stage_a/2026-10-20/'
PREP=B+'track_a_pf_selection_comparison_preparation/2026-10-20/'
OLD=B+'track_a_pf_selection_cost_comparison/2026-10-20/'
STOPPED=B+'track_a_h6_noncancellation_stage_b/2026-10-20/'
OUT=ROOT/(B+'track_a_h6_noncancellation_stage_b_rerun/2026-10-20')
PYTHON='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
INPUTS=(A+'endpoints.json',A+'summary.json',A+'completion_audit.json',A+'numeric_completion.json',
        A+'STOP.json',PREP+'saved_condition_rows.json',PREP+'axis_price_lineage.json',
        PREP+'baseline_saved_comparisons.json',PREP+'condition_coverage.json',PREP+'candidate_domains.json',
        PREP+'source_inventory.json',OLD+'selection_comparison.json',OLD+'summary.json')
SOURCES=('src/trotterlib/h6_noncancellation_stage_b.py','src/trotterlib/pf_selection_cost_comparison.py',
    'src/trotterlib/h8_six_axis_reaccount.py','scripts/run_h6_noncancellation_stage_b_rerun.py',
    'scripts/check_h6_noncancellation_stage_b_rerun.py','scripts/plot_h6_noncancellation_stage_b_rerun.py',
    'tests/test_h6_noncancellation_stage_b_rerun.py','tests/test_h6_noncancellation_stage_b.py',
    'tests/test_pf_selection_cost_comparison.py')
ZERO=('signal','sampling','compile','model_prediction','fit','shot_recalculation','eligibility_recalculation',
      'PR_mean_SE_or_candidate_recalculation','old70_numeric_reanalysis','retry','candidate_addition')
APPROVAL='''Current user explicitly approves correction of the SOURCE serialization bug
at stopped commit c585cef8990ef28f4d43c13dfaf2a0d423fda5a7, full-output
artificial/regression tests and a fresh SOURCE/contract/grant for exactly the
same H6 noncancellation30 scopes /10 endpoints. Re-run once, verify known712
costs, use accepted stageA34 eligible endpoints, compare PF candidates/q/order/
totals/PR ratios/ranks, copy old70 without reanalysis, save CSV/JSON/figures,
validate numerical consistency/SHA/SOURCE/ledger/native exit/reap/preservation.
Scientific formulas, inputs, candidates, prices and stageA shots/eligibility
stay unchanged. Preserve the stopped SOURCE/partial outputs/consumed grant/
ledger/markers/STOP; no reuse. No stageA/old70 reanalysis, signal, sampling,
compile, model predictions/refit, shots/PR-statistics recalculation, automatic
retry or scope expansion. Stop on anomaly; no detailed scientific review or
final evidence freeze. Publication only within a separately approved scope;
this receipt does not authorize publication of these new files.
'''


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT)
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def write(name,value,*,exclusive=False):
    raw=value.encode() if isinstance(value,str) else encoded(value)
    if exclusive:
        with (OUT/name).open('xb') as f:f.write(raw)
    else:
        temp=OUT/('.'+name+'.'+str(os.getpid())+'.tmp')
        with temp.open('xb') as f:f.write(raw)
        os.replace(temp,OUT/name)


def csv_text(rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='')
    w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':'))
        if isinstance(v,(dict,list,bool)) or v is None else v for k,v in r.items()})
    return s.getvalue()


JSON_OUTPUTS=('joined_rows.json','patched120_lineage.json','candidate_totals.json',
    'selection_comparison30.json','candidate_ranks.json','rank_relation_changes.json',
    'combined_selection100.json','summary.json','numeric_validation.json','used_price_evidence.json')
CSV_OUTPUTS=('candidate_totals','selection_comparison30','candidate_ranks',
    'rank_relation_changes','combined_selection100')


def output_payloads(outputs,source):
    """Validate and serialize every output before writing; preserve matching SOURCE."""
    core.require(set(outputs)==set(JSON_OUTPUTS),'COMPLETE_OUTPUT_BUNDLE_REQUIRED')
    prepared={}
    for name,value in outputs.items():
        core.require(isinstance(value,dict),'OUTPUT_MAPPING_REQUIRED:'+name)
        if 'SOURCE' in value:
            core.require(value['SOURCE']==source,'OUTPUT_SOURCE_MISMATCH:'+name)
        prepared[name]=encoded({'SOURCE':source,**value}).decode('utf-8')
    for name in CSV_OUTPUTS:
        prepared[name+'.csv']=csv_text(outputs[name+'.json']['rows'])
    return prepared


def persist_outputs(outputs,source):
    prepared=output_payloads(outputs,source)
    for name,payload in prepared.items():write(name,payload)
    return list(prepared)


def inputs():return {n:read(ROOT/n) for n in INPUTS}


def metadata(data):
    a=data[A+'endpoints.json']['rows'];summary=data[A+'summary.json']
    core.require(len(a)==120 and summary['eligibility_counts']==dict(ELIGIBLE=34,INELIGIBLE=86,UNRESOLVED=0), 'STAGE_A_ACCEPTED120')
    core.require(data[A+'STOP.json']['status']=='STAGE_A_COMPLETE_MANDATORY_STOP' and
        data[A+'completion_audit.json']['status']=='PASS_STAGE_A_COMPLETE_NATIVE_REAP_PROTECTED_EVIDENCE', 'STAGE_A_COMPLETED_AUDIT')
    conditions=[r for r in data[PREP+'condition_coverage.json']['rows'] if r['system']=='H6_DF19'
                and r['bias_mode']==core.BIAS and r['strategy']=='ALL_SAVED458']
    core.require(len(conditions)==30 and len({core.ep(r) for r in conditions})==10,'EXACT30_SCOPE10_ENDPOINTS')
    domains=data[PREP+'candidate_domains.json']['H6_DF19']
    core.require(sorted(map(len,domains.values()))==[32,96,136],'ORIGINAL32_96_136')
    for c in conditions:core.require(c['candidate_ids']==domains[c['original_scope']],'UNCHANGED_DOMAIN')
    target=[r for r in data[PREP+'saved_condition_rows.json']['rows'] if r['system']=='H6_DF19' and r['bias_mode']==core.BIAS]
    core.require(len(target)==1440 and sum(r['eligible'] is None for r in target)==120,'EXACT_SAVED1440_UNKNOWN120')
    core.require(sum(r['kind']=='PF' and r['eligible'] is True for r in target)==712,'KNOWN_ELIGIBLE712')
    old70=data[OLD+'selection_comparison.json']['rows'];core.require(len(old70)==70,'SAVED70')
    return conditions


def resources():
    path=OUT
    while not path.exists():path=path.parent
    disk=os.statvfs(path)
    mem={r.split(':')[0]:r.split(':')[1].strip() for r in Path('/proc/meminfo').read_text().splitlines()}
    return dict(UTC=datetime.now(timezone.utc).isoformat(),allowed_affinity=sorted(os.sched_getaffinity(0)),
        host_load=list(os.getloadavg()),MemAvailable=mem.get('MemAvailable'),
        output_available_bytes=disk.f_bavail*disk.f_frsize,no_resource_caps_or_affinity_changes=True,
        policy='One indexed stdlib arithmetic child; reuse10 endpoint totals across30 scopes; no useful extra worker overhead',
        shared_environment_and_other_processes_unchanged=True)


def verify(c):
    for name,value in c['source_hashes'].items():
        core.require(sha(ROOT/name)==value and hashlib.sha256(git('show',c['SOURCE']+':'+name)).hexdigest()==value,'SOURCE_CHANGED:'+name)
    for name,value in c['input_hashes'].items():
        core.require(sha(ROOT/name)==value and hashlib.sha256(git('show',BASE+':'+name)).hexdigest()==value,'INPUT_CHANGED:'+name)
    for name,value in c['protected_evidence_hashes'].items():core.require(sha(ROOT/name)==value,'OLD_EVIDENCE_CHANGED:'+name)
    for s in c['underlying_saved_sources']:
        core.require(hashlib.sha256(git('show',s['commit']+':'+s['path'])).hexdigest()==s['sha256'],'UNDERLYING_RESULT_CHANGED')
    core.require(sys.executable==PYTHON and sys.version==c['runtime']['python_version'],'ORIGINAL_PYTHON_REQUIRED')
    for name,value in c['runtime']['file_hashes'].items():core.require(sha(name)==value,'RUNTIME_CHANGED:'+name)


def seal(receipt):
    core.require(not OUT.exists(),'FRESH_STAGE_B_OUTPUT_REQUIRED')
    data=inputs();conditions=metadata(data);tests=read(receipt)
    core.require(tests['status']=='PASS_ARTIFICIAL_AND_COMPARISON_REGRESSION' and tests['real_numeric_evaluations']==0,'PRE_RUN_TESTS_REQUIRED')
    head=git('rev-parse','HEAD').decode().strip()
    for n in SOURCES:
        core.require(git('show',head+':'+n)==(ROOT/n).read_bytes() and tests['source_hashes'][n]==sha(ROOT/n),'COMMITTED_TESTED_SOURCE_REQUIRED')
    protected=[]
    for directory in (A,OLD,B+'track_a_h6_noncancellation_investigation/2026-10-20/',STOPPED):
        protected+=git('ls-tree','-r','--name-only',BASE,directory.rstrip('/')).decode().splitlines()
    runtime={str(Path(sys.executable).resolve())}
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and Path(f).is_file() and not Path(f).resolve().is_relative_to(ROOT):runtime.add(str(Path(f).resolve()))
    for line in Path('/proc/self/maps').read_text().splitlines():
        f=line.split()[-1]
        if f.startswith('/') and Path(f).is_file():runtime.add(f)
    budget=dict(one_shot_primary_passes=1,old_unknown_rows_joined=120,PF_candidate_endpoint_rows=1360,
        known_baseline_weighted_costs=712,new_baseline_weighted_costs=34,replacement_weighted_costs=746,
        baseline_PF_selections=30,replacement_PF_selections=30,scope_rank_comparisons=30,
        pair_relation_comparisons_max=275400,validation_weighted_costs=1492,validation_scopes=30,
        validation_pair_relations_max=275400,combined_saved_result_rows=100)
    c=dict(status='SEALED_APPROVED_STAGE_B30',scope='H6_NONCANCELLATION_STAGE_B_30',SOURCE=head,base_commit=BASE,
        accepted_stage_A_SOURCE=data[A+'summary.json']['SOURCE'],scope_conditions=30,unique_endpoints=10,
        strategy='ALL_SAVED458',conditions=conditions,condition_ids=[core.old.condition_id(c) for c in conditions],
        scientific_call_budget=budget,prohibited_calls={k:0 for k in ZERO},
        input_hashes={n:sha(ROOT/n) for n in INPUTS},source_hashes={n:sha(ROOT/n) for n in SOURCES},
        protected_evidence_hashes={n:sha(ROOT/n) for n in protected},underlying_saved_sources=data[PREP+'source_inventory.json'],
        runtime=dict(executable=sys.executable,python_version=sys.version,platform=platform.platform(),file_hashes={n:sha(n) for n in sorted(runtime)}),
        definitions=dict(baseline='Known712 exact replay plus new34 constructed stageA costs; historical full-domain minimum null',
            cost='Fraction(saved N_cos*RZ_cos + saved N_sin*RZ_sin)',selection='exact total ascending then lexical ID; retain all exact ties',
            fixed='32/96/136 candidate domains; saved PR union8 n32 choice/cost/SE; stageA eligibility/shots',
            missing='Any eligible=None, missing cost/price or baseline mismatch STOP without correction/retry',
            old70='Copy rows, aggregate saved flags/counts and render references; do not rerun old accounting/ranking/ratios'),
        resources_policy=resources(),approval_receipt_sha256=hashlib.sha256(APPROVAL.encode()).hexdigest(),
        one_shot=True,retry=False,new_publication_authorized=False,detailed_scientific_review=False,final_evidence_freeze=False,
        limitations=['H6 profile mismatch','model/runtime identity unproven','S4 composition hypothesis',
            'initial572 global final gate missing; old profile48 retained','finite PR n32 point comparison; no population guarantee'],
        sealed_UTC=datetime.now(timezone.utc).isoformat())
    previous=read(ROOT/(STOPPED+'sealed_contract.json'))
    for key in ('input_hashes','conditions','scientific_call_budget','prohibited_calls','definitions'):
        core.require(c[key]==previous[key],'SCIENTIFIC_CONTRACT_UNCHANGED:'+key)
    for name,value in previous['source_hashes'].items():
        core.require(sha(ROOT/name)==value and hashlib.sha256(git('show',BASE+':'+name)).hexdigest()==value,
                     'STOPPED_SOURCE_PRESERVED:'+name)
    c.update(previous_stopped_SOURCE=previous['SOURCE'],previous_stopped_contract_sha256=sha(ROOT/(STOPPED+'sealed_contract.json')),
        unchanged_scientific_input_conditions_budget_definitions=True,
        correction='Output serialization only: validate any existing SOURCE, then merge one matching key; complete-bundle prevalidation',
        prior_grant_reused=False,new_user_authorized_execution=True)
    verify(c);OUT.mkdir(parents=True,exist_ok=False)
    write('sealed_contract.json',c,exclusive=True);write('user_approval_receipt.txt',APPROVAL,exclusive=True)
    write('artificial_tests.json',tests,exclusive=True)
    write('grant.json',dict(status='CURRENT_USER_APPROVED_STAGE_B_ONLY',SOURCE=head,scope=c['scope'],
        scientific_execution_authorized=True,contract_sha256=sha(OUT/'sealed_contract.json'),retry=False,
        next_stage_authorized=False,new_publication_authorized=False),exclusive=True)
    print(json.dumps(dict(status='SEALED_NOT_RUN',SOURCE=head,scope_conditions=30)))


def run():
    c=read(OUT/'sealed_contract.json');g=read(OUT/'grant.json');verify(c);core.authorize(c,g)
    core.require(g['contract_sha256']==sha(OUT/'sealed_contract.json') and sha(OUT/'user_approval_receipt.txt')==c['approval_receipt_sha256'],'GRANT_BINDING')
    write('one_shot.marker',c['SOURCE']+'\n',exclusive=True)
    ledger=dict(status='STARTED',SOURCE=c['SOURCE'],stages_reserved=[],stages_completed=[],prohibited_calls={k:0 for k in ZERO})
    started=time.monotonic()
    def stage(name,func,*args,**kwargs):
        core.require(name not in ledger['stages_reserved'],'NO_PHASE_RETRY');ledger['stages_reserved'].append(name);write('ledger.json',ledger)
        value=func(*args,**kwargs);ledger['stages_completed'].append(name);write('ledger.json',ledger);return value
    try:
        data=inputs();conditions=metadata(data);core.require(conditions==c['conditions'],'SEALED_CONDITION_LIST')
        a=data[A+'endpoints.json'];original=data[PREP+'saved_condition_rows.json']['rows']
        lineage=dict(SOURCE=a['SOURCE'],file_path=A+'endpoints.json',file_sha256=c['input_hashes'][A+'endpoints.json'],accepted_by_current_user=True)
        rows,patched=stage('join_stage_A',core.join_stage_a,original,a['rows'],lineage)
        units={u['task_id']:u for u in data[PREP+'axis_price_lineage.json']['rows']}
        costs,replay=stage('baseline_costs_and_exact_replay',core.baseline,rows,units)
        core.require(len(costs)==1360 and replay['known_costs_replayed']==712 and replay['newly_constructed_eligible_baseline_costs']==34,'EXACT_BUDGET_COUNTS')
        write('baseline_replay_gate.json',dict(SOURCE=c['SOURCE'],**replay))
        gates=stage('new_completed_baseline_selection',core.baseline_selections,costs,rows,
                    data[PREP+'baseline_saved_comparisons.json']['rows'],conditions)
        used=stage('direct_price_replacement',core.old.replace_costs,costs,units)
        result,ranks,changes=stage('new30_comparison_and_ranks',core.compare,costs,conditions,gates)
        audit=stage('integer_numeric_validation',core.audit,costs,rows,units,conditions,result,ranks,changes)
        core.require(audit['weighted_cost_checks']==1492 and audit['pair_relations_checked']<=c['scientific_call_budget']['validation_pair_relations_max'],'VALIDATION_BUDGET')
        combined,summary=stage('reference_old70_and_aggregate',core.aggregate,result,data[OLD+'selection_comparison.json']['rows'])
        core.require(summary['previous70']==data[OLD+'summary.json']['all_scopes'],'SAVED70_SUMMARY_UNCHANGED')
        summary.update(SOURCE=c['SOURCE'],accepted_stage_A_SOURCE=c['accepted_stage_A_SOURCE'],
            new30_replay_gate=replay,used_price_policy_counts=dict(__import__('collections').Counter(units[t]['ALL_SAVED458'] for t in used)),
            no_unresolved_eligibility_remaining=True)
        outputs={'joined_rows.json':dict(rows=rows),'patched120_lineage.json':dict(rows=patched),
            'candidate_totals.json':dict(rows=costs),'selection_comparison30.json':dict(rows=result),
            'candidate_ranks.json':dict(rows=ranks),'rank_relation_changes.json':dict(rows=changes),
            'combined_selection100.json':dict(rows=combined),'summary.json':summary,'numeric_validation.json':audit,
            'used_price_evidence.json':dict(rows=[units[t] for t in sorted(used)])}
        persist_outputs(outputs,c['SOURCE'])
        verify(c)
        ledger.update(status='COMPLETE_STAGE_B30',actual_calls=dict(one_shot_primary_passes=1,
            old_unknown_rows_joined=len(patched),PF_candidate_endpoint_rows=len(costs),known_baseline_weighted_costs=712,
            new_baseline_weighted_costs=34,replacement_weighted_costs=746,baseline_PF_selections=30,
            replacement_PF_selections=30,scope_rank_comparisons=30,pair_relation_comparisons=sum(r['pairwise_comparisons'] for r in result),
            validation_weighted_costs=audit['weighted_cost_checks'],validation_scopes=30,
            validation_pair_relations=audit['pair_relations_checked'],combined_saved_result_rows=100),
            final_input_SOURCE_runtime_old_evidence_gate='PASS',wall_seconds=time.monotonic()-started)
        write('ledger.json',ledger)
        names=list(outputs)+[n+'.csv' for n in ('candidate_totals','selection_comparison30','candidate_ranks','rank_relation_changes','combined_selection100')]+['baseline_replay_gate.json']
        write('numeric_completion.json',dict(status='NUMERIC_COMPLETE_PENDING_RENDER_NATIVE_AUDIT',SOURCE=c['SOURCE'],
            result_sha256={n:sha(OUT/n) for n in names},old70_reanalysis=False,additional_science_authorized=False))
        write('grant_consumption_receipt.json',dict(status='CONSUMED_COMPLETE',SOURCE=c['SOURCE'],grant_sha256=sha(OUT/'grant.json'),further_execution_authorized=False))
        print(json.dumps(summary))
    except BaseException as error:
        ledger.update(status='STOP_NO_RETRY',error_type=type(error).__name__,error=str(error));write('ledger.json',ledger)
        write('STOP.json',dict(status='STAGE_B_NUMERIC_OR_TECHNICAL_STOP',error=str(error),retry=False,mandatory_STOP=True))
        write('grant_consumption_receipt.json',dict(status='CONSUMED_STOP',further_execution_authorized=False))
        raise


def launch():
    c=read(OUT/'sealed_contract.json');verify(c)
    core.require(not (OUT/'one_shot.marker').exists() and not (OUT/'STOP.json').exists(),'NO_RETRY')
    write('prelaunch_gate.json',dict(status='PASS',resources=resources()),exclusive=True)
    write('launch.marker',c['SOURCE']+'\n',exclusive=True);start=time.monotonic()
    with (OUT/'native_stdout.txt').open('xb') as stdout,(OUT/'native_stderr.txt').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT,stdout=stdout,stderr=stderr)
        write('native_started.json',dict(SOURCE=c['SOURCE'],pid=child.pid,parent_pid=os.getpid()),exclusive=True)
        pid,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
    write('native_exit_evidence.json',dict(SOURCE=c['SOURCE'],native_exit_code=code,same_parent_wait4=True,
        worker_pid=pid,parent_pid=os.getpid(),reaped=True,live_owned_workers=0,wall_seconds=time.monotonic()-start,
        CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,
        stdout_sha256=sha(OUT/'native_stdout.txt'),stderr_sha256=sha(OUT/'native_stderr.txt'),retry=False),exclusive=True)
    if code:
        if not (OUT/'STOP.json').exists():write('STOP.json',dict(status='STAGE_B_NATIVE_STOP',native_exit_code=code,retry=False,mandatory_STOP=True))
        raise SystemExit(code)
    print(json.dumps(dict(status='PASS_NATIVE_EXIT0_SAME_PARENT_REAP',SOURCE=c['SOURCE'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('seal','run','launch'));p.add_argument('--tests-receipt',type=Path);a=p.parse_args()
    if a.stage=='seal':core.require(a.tests_receipt is not None,'TEST_RECEIPT_REQUIRED');seal(a.tests_receipt)
    elif a.stage=='run':run()
    else:launch()


if __name__=='__main__':main()
