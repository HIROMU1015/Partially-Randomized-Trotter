#!/usr/bin/env python3
"""Acquire/seal/preflight/one-shot/audit. Only _child performs saved-value science."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import uuid

from h8_rare_c05.accounting import Budget, atomic, sha
from h8_rare_saved_core import CELLS, analyze_cell
from h8_rare_saved_verify import validate

BASE='f377f8869c697c4846e3614b0ae7bc9d3ad5cb3c'
B_COMMIT='3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb'
C_SOURCE='383dd9068504b8f578944984fc45a60dd4ed18b2'
B_DIR='artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21'
C_DIR='artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21'
OUT='artifacts/resource_applicability/h8_rare_rte_saved_integration/2026-10-21'
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'.rare_integration_private'
SOURCE_PATHS=['scripts/h8_rare_saved_core.py','scripts/h8_rare_saved_verify.py',
              'scripts/h8_rare_saved_run.py','scripts/h8_rare_c05/accounting.py',
              'scripts/h8_rare_c05/sampling.py','scripts/h8_rare_c05/__init__.py',
              'tests/test_h8_rare_saved_integration.py',
              OUT+'/analysis_definition.json','docs/research/h8_rare_rte_saved_integration_contract_20261021.md']
LIMITS={'saved_value_campaign':1,'primary_condition_analysis':2,
        'independent_condition_validation':2,'deterministic_table_export':1}


def require(ok,reason):
    if not ok: raise ValueError(reason)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def read(path): return json.loads(Path(path).read_text())
def h(data): return hashlib.sha256(data).hexdigest()
def public(name,value): atomic(ROOT/OUT/name,value)


def exclusive(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False); f.flush();os.fsync(f.fileno())


def acquisition():
    require(not PRIVATE.exists(),'NEW_PRIVATE_ROOT_ONLY')
    PRIVATE.mkdir(mode=0o700)
    (PRIVATE/'.gitignore').write_text('*\n')
    cache=PRIVATE/'inputs';cache.mkdir()
    records=[]
    for label,commit,directory in [('B',B_COMMIT,B_DIR),('C1',BASE,C_DIR)]:
        paths=git('ls-tree','-r','--name-only','-z',commit,directory).decode().split('\0')
        for path in filter(None,paths):
            data=git('show',commit+':'+path)
            file='inputs/'+str(len(records)).zfill(4)+'.blob'
            (PRIVATE/file).write_bytes(data)
            records.append(dict(label=label,commit=commit,path=path,sha256=h(data),bytes=len(data),cache=file,
                                git_blob_id=git('rev-parse',commit+':'+path).decode().strip()))
        manifest=json.loads(git('show',commit+':'+directory+'/SHA256_manifest.json'))
        for row in manifest['files']:
            require(h(git('show',commit+':'+row['path']))==row['sha256'],'OLD_GIT_MANIFEST_MISMATCH')
    # The pilot execution SOURCE is retrieved without importing its scientific driver.
    cm=json.loads(git('show',BASE+':'+C_DIR+'/SOURCE_manifest.json'))
    require(cm['SOURCE']==C_SOURCE,'C1_SOURCE_COMMIT')
    source_receipts=[]
    for row in cm['files']:
        require(h(git('show',C_SOURCE+':'+row['path']))==row['sha256'],'C1_SOURCE_FILE')
        source_receipts.append(dict(commit=C_SOURCE,**row))
    atomic(PRIVATE/'catalog.json',records)
    atomic(PRIVATE/'old_SOURCE_receipts.json',source_receipts)
    # Protect only this research's saved inputs, not other worktrees' HEAD/status.
    protected=[]
    b_private=ROOT.parent/'pr-rare-rte-stage-b-20261021'/'.stage_b_private'
    for r in read(b_private/'original_reads.json'):
        if r['kind']=='PRIVATE_ORIGINAL':
            p=Path(r['path']);require(p.is_absolute(),'OLD_B_PRIVATE_PATH')
            require(sha(p)==r['sha256'],'OLD_B_PRIVATE_CHANGED')
            protected.append(dict(path=str(p),sha256=r['sha256'],role='B_raw'))
    c_root=ROOT.parent/'h8-rare-rte-c1-pilot-repair-20261021'
    for r in read(c_root/'.c1_repair_private/private_result_inventory.json'):
        p=Path(r['path']);p=p if p.is_absolute() else c_root/p
        require(sha(p)==r['sha256'],'OLD_C1_PRIVATE_CHANGED')
        protected.append(dict(path=str(p),sha256=r['sha256'],role='C1_private_result'))
    for record in records:
        oldroot=ROOT.parent/('pr-rare-rte-stage-b-20261021' if record['label']=='B' else 'h8-rare-rte-c1-pilot-repair-20261021')
        p=oldroot/record['path'];require(sha(p)==record['sha256'],'OLD_TRACKED_INPUT_CHANGED')
        protected.append(dict(path=str(p),sha256=record['sha256'],role='old_tracked_evidence'))
    atomic(PRIVATE/'protected.json',protected)
    public('input_acquisition.json',dict(status='PASS_FIXED_GIT_BLOBS_AND_SAVED_SHA',
           fixed_commits=[B_COMMIT,BASE,C_SOURCE],
           files=[{k:v for k,v in r.items() if k!='cache'} for r in records],
           C1_SOURCE_files=source_receipts, protected_private_files=len(protected),
           protected_inventory_sha256=sha(PRIVATE/'protected.json'),internal_locators_published=False))
    print('PASS_ACQUISITION_METADATA_ONLY',len(records),len(protected))


def load_input(label,name):
    rows=read(PRIVATE/'catalog.json')
    hits=[r for r in rows if r['label']==label and Path(r['path']).name==name]
    require(len(hits)==1,'UNIQUE_FIXED_INPUT')
    r=hits[0];p=PRIVATE/r['cache'];require(sha(p)==r['sha256'],'CACHED_INPUT_CHANGED')
    return p.read_text()


def check_protected():
    rows=read(PRIVATE/'protected.json')
    for r in rows:require(sha(r['path'])==r['sha256'],'OLD_USED_EVIDENCE_CHANGED')
    return len(rows)


def structure():
    b=json.loads(load_input('B','result.json'))
    cells={r['condition']:r for r in b['conditions'] if r['condition'] in CELLS}
    require(set(cells)==set(CELLS),'TWO_FIXED_CELLS')
    c=json.loads(load_input('C1','condition_results.json'))
    require(c['SOURCE']==C_SOURCE,'PILOT_SOURCE')
    for cid,candidate in CELLS.items():
        z=cells[cid]
        pilot=json.loads(load_input('C1','trajectory_costs_'+cid+'.json'))
        summary=json.loads(load_input('C1','pilot_summary_'+cid+'.json'))
        law=json.loads(load_input('C1','law_gate_'+cid+'.json'))
        require(z['candidate']==candidate==pilot['candidate'] and z['n']==48 and len(z['tasks'])==48 and
                len(pilot['rows'])==43 and summary['groups']['S2_SINGLE']['n']==43,'FIXED_COHORT_COUNTS')
        require(pilot['shots']==z['shots'] and law['law']==z['law']['saved_law'] and law['status']=='PASS',
                'FIXED_LAW_SHOTS')
        require(all(r['stratum']=='S0_ONLY' for r in z['tasks']) and
                all(r['stratum']=='S2_SINGLE' for r in pilot['rows']),'NO_POOLING')
        expected={'H8_0.05':('B0I_p11_q1_2nd','1267321444'),
                  'H8_0.0005':('B0I_p15_q4_4th','94193016652860')}[cid]
        require((z['fixed_PF'],z['fixed_PF_G_exact'])==expected,'FIXED_PF_ANCHOR')
        for layer in ('S2_MULTI','S4_PRESENT'):
            require(summary['groups'][layer]['mean'] is None and summary['groups'][layer]['variance'] is None,
                    'UNKNOWN_NOT_ZERO')
        saved=next(r for r in c['conditions'] if r['condition']==cid)
        require(saved['status']=='COMPLETE' and saved['n']==43 and saved['native_exit']['exit_code']==0 and
                saved['native_exit']['reaped'] and saved['actual']==saved['planned'],'COMPLETED_PILOT')
    return dict(status='PASS_STRUCTURE_NO_NEW_SCIENTIFIC_ARITHMETIC',conditions=2,S0_n_each=48,SINGLE_n_each=43)


def closure():
    paths={Path(sys.executable).resolve()}
    for m in tuple(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and Path(f).is_file(): paths.add(Path(f).resolve())
    for line in Path('/proc/self/maps').read_text().splitlines():
        p=line.split()[-1]
        if p.startswith('/') and Path(p).is_file(): paths.add(Path(p).resolve())
    return [dict(path=str(p),sha256=sha(p)) for p in sorted(paths)]


def seal():
    require(not (PRIVATE/'sealed.json').exists(),'ALREADY_SEALED')
    structural=structure()
    b_source=json.loads(load_input('B','result.json'))['SOURCE']
    b_receipts=[]
    for path in ('scripts/rare_rte_stage_b_core.py','scripts/rare_rte_stage_b_verify.py','scripts/run_rare_rte_stage_b.py'):
        data=git('show',b_source+':'+path)
        b_receipts.append(dict(commit=b_source,path=path,sha256=h(data)))
    evidence=read(ROOT/OUT/'input_acquisition.json');evidence['B_SOURCE_files']=b_receipts
    public('input_acquisition.json',evidence)
    protected=read(PRIVATE/'protected.json')
    b_private=ROOT.parent/'pr-rare-rte-stage-b-20261021'/'.stage_b_private'
    c_private=ROOT.parent/'h8-rare-rte-c1-pilot-repair-20261021'/'.c1_repair_private'
    for folder,names in [(b_private,['grant_APPROVED_PRIVATE.json','analysis_ledger.json','ONE_SHOT.json']),
                         (c_private,['grant_H8_0.05.json','grant_H8_0.0005.json'])]:
        for name in names:
            p=folder/name;protected.append(dict(path=str(p),sha256=sha(p),role='old_grant_ledger_marker'))
    atomic(PRIVATE/'protected.json',protected)
    public('preservation_preflight.json',dict(protected_files=len(protected),
           inventory_sha256=sha(PRIVATE/'protected.json'),old_grant_ledger_marker_explicitly_preserved=True,
           unrelated_worktree_HEAD_status_not_frozen=True))
    n=check_protected()
    source=git('rev-parse','HEAD').decode().strip()
    require(git('status','--porcelain','--untracked-files=no')==b'','COMMIT_SOURCE_BEFORE_SEAL')
    rows=[]
    for path in SOURCE_PATHS:
        digest=sha(ROOT/path)
        require(h(git('show',source+':'+path))==digest,'SOURCE_NOT_COMMITTED')
        rows.append(dict(path=path,sha256=digest))
    runtime=closure();atomic(PRIVATE/'runtime.json',runtime)
    public('SOURCE_manifest.json',dict(SOURCE=source,baseline=BASE,files=rows,
           runtime_inventory_sha256=sha(PRIVATE/'runtime.json'),
           runtime_files=[dict(id=i,name=Path(r['path']).name,sha256=r['sha256']) for i,r in enumerate(runtime)]))
    contract=dict(SOURCE=source,definition_sha256=sha(ROOT/OUT/'analysis_definition.json'),
                  catalog_sha256=sha(PRIVATE/'catalog.json'),budget=LIMITS,
                  runtime_inventory_sha256=sha(PRIVATE/'runtime.json'),source_files=rows,
                  explicit_saved_value_analysis_approved=True,one_shot=True,retry=False,resume=False,
                  conditions=CELLS,new_sampling_compile_simulation_signal_allowed=False)
    public('contract.json',contract)
    public('execution_approval_receipt.json',dict(SOURCE=source,contract_sha256=sha(ROOT/OUT/'contract.json'),
           approved=True,authority='CURRENT_USER_EXPLICIT_ONE_SHOT_SAVED_VALUE_INTEGRATION_REQUEST',
           budget=LIMITS,conditions=CELLS,publication_approved=False,
           permitted='S0 n48 plus SINGLE n43 stratified integration and two residual-layer conditional sensitivity only'))
    public('preflight.json',dict(**structural,source_sha_pass=True,input_sha_pass=True,
           runtime_sha_pass=True,preservation_sha_files=n,synthetic_tests=read(ROOT/OUT/'synthetic_validation.json'),
           science_started=False,grant_issued=False,shared_settings_changed=False))
    exclusive(PRIVATE/'sealed.json',dict(SOURCE=source,contract_sha256=sha(ROOT/OUT/'contract.json'),
              approval_sha256=sha(ROOT/OUT/'execution_approval_receipt.json')))
    print('PASS_SEALED_PREFLIGHT',source)


def gate():
    s=read(PRIVATE/'sealed.json');c=read(ROOT/OUT/'contract.json');a=read(ROOT/OUT/'execution_approval_receipt.json')
    require(sha(ROOT/OUT/'contract.json')==s['contract_sha256'] and
            sha(ROOT/OUT/'execution_approval_receipt.json')==s['approval_sha256'],'CONTRACT_APPROVAL_CHANGED')
    require(c['budget']==LIMITS==a['budget'] and c['SOURCE']==a['SOURCE']==s['SOURCE'],'SCOPE_SOURCE')
    require(a['approved'] is True and c['one_shot'] and not c['retry'] and not c['resume'],'AUTHORIZATION')
    require(sha(PRIVATE/'catalog.json')==c['catalog_sha256'] and
            sha(PRIVATE/'runtime.json')==c['runtime_inventory_sha256'],'MANIFEST_CHANGED')
    for r in c['source_files']:require(sha(ROOT/r['path'])==r['sha256'],'SOURCE_CHANGED')
    for r in read(PRIVATE/'runtime.json'):require(sha(r['path'])==r['sha256'],'RUNTIME_CHANGED')
    for r in read(PRIVATE/'catalog.json'):require(sha(PRIVATE/r['cache'])==r['sha256'],'INPUT_CHANGED')
    return s,c


def export_tables(results):
    tables={
      'known_layers.csv': [dict(condition=r['condition'],**v) for r in results for v in r['known_layers']],
      'two_layer_boundary.csv': [dict(condition=r['condition'],**v) for r in results for v in r['two_layer_boundary']['points']],
      'known_error_sensitivity.csv': [dict(condition=r['condition'],**v) for r in results for v in r['known_error_sensitivity']],
      'condition_summary.csv': [{k:r[k] for k in ('condition','candidate','K_known_G','SE_K_known_G','remaining_budget_D_G',
          'residual_mass','residual_mixture_mean_threshold_G','residual_mixture_threshold_over_S0','headroom_in_known_SE')} for r in results]}
    for name,rows in tables.items():
        f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        p=ROOT/OUT/name
        with p.open('x') as out:out.write(f.getvalue());out.flush();os.fsync(out.fileno())


def child():
    s,c=gate();g=read(PRIVATE/'grant.json')
    require(g['SOURCE']==s['SOURCE'] and g['contract_sha256']==s['contract_sha256'] and
            g['approval_sha256']==s['approval_sha256'] and g['budget']==LIMITS and g['one_shot'] and
            not g['retry'] and not g['resume'],'FRESH_GRANT_BINDING')
    exclusive(PRIVATE/'ONE_SHOT.json',dict(consumed=True,nonrefundable=True,grant_sha256=sha(PRIVATE/'grant.json'),
              SOURCE=s['SOURCE'],contract_sha256=s['contract_sha256']))
    ledger=Budget(PRIVATE/'budget',LIMITS)
    actual={k:0 for k in LIMITS};results=[];audits=[]
    def enter(kind,n,key):
        ledger.take(kind,n,key);actual[kind]+=n
        public('calls.json',dict(planned=LIMITS,reserved=ledger.used(),actual_entered=actual,nonrefundable=True))
    started=time.monotonic()
    try:
        enter('saved_value_campaign',1,'campaign')
        b=json.loads(load_input('B','result.json'))
        for cid in CELLS:
            z=next(x for x in b['conditions'] if x['condition']==cid)
            pilot=json.loads(load_input('C1','trajectory_costs_'+cid+'.json'))
            saved=json.loads(load_input('C1','pilot_summary_'+cid+'.json'))
            law=json.loads(load_input('C1','law_gate_'+cid+'.json'))
            enter('primary_condition_analysis',1,cid+'_primary')
            out=analyze_cell(z,pilot,saved,law)
            results.append(out);public('partial_results.json',dict(conditions=results,complete=False))
            enter('independent_condition_validation',1,cid+'_verify')
            audits.append(validate(z,load_input('B','trajectory_costs.csv'),load_input('C1','paired_costs_'+cid+'.csv'),out))
        enter('deterministic_table_export',1,'tables')
        export_tables(results)
        public('result.json',dict(status='COMPLETE_SAVED_STRATIFIED_CONDITIONAL_ANALYSIS',SOURCE=s['SOURCE'],
               conditions=results,cohorts_pooled=False,new_sampling_simulation_signal_circuits_compile=0,
               fixed_candidates_reselected=False,production=False,full_population_ranking_guaranteed=False))
        public('independent_numeric_audit.json',dict(status='PASS',checks=audits,
               same_author_separate_arithmetic=True,separate_reviewer=False,scientific_guarantee=False))
        public('calls.json',dict(planned=LIMITS,reserved=ledger.used(),actual_entered=actual,
               all_entered_calls_completed=True,nonrefundable=True,
               new_sampling=0,new_simulation=0,new_signal=0,new_circuit=0,new_compile=0))
        public('worker_completion.json',dict(status='PASS',elapsed_seconds=time.monotonic()-started,
               SOURCE=s['SOURCE'],scientific_children_spawned=0,all_numerical_gates_passed=True))
    except BaseException:
        (PRIVATE/'exception.txt').write_text(traceback.format_exc())
        public('STOP.json',dict(status='MANDATORY_STOP_FAILURE_NO_RETRY',nonrefundable=True,
               grant_consumed=True,partial_conditions=len(results),next_science_authorized=False))
        raise


def execute():
    s,c=gate();require(not (PRIVATE/'grant.json').exists() and not (PRIVATE/'ONE_SHOT.json').exists(),'FRESH_ONE_SHOT_REQUIRED')
    check_protected();structure()
    # No arbitrary worker/CPU/RAM cap: this exact two-cell arithmetic does not merit a pool.
    mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    public('resource_preflight.json',dict(available_CPU_count=len(os.sched_getaffinity(0)),
           MemAvailable=mem.get('MemAvailable'),load_average=list(os.getloadavg()),
           concurrency_reason='One numerical child; two tiny exact-arithmetic cells; no molecular runtime or pool overhead',
           resource_caps_added=False,shared_settings_changed=False))
    grant=dict(id=str(uuid.uuid4()),SOURCE=s['SOURCE'],contract_sha256=s['contract_sha256'],
               approval_sha256=s['approval_sha256'],budget=LIMITS,one_shot=True,retry=False,resume=False)
    exclusive(PRIVATE/'grant.json',grant)
    public('grant_issue_receipt.json',{k:v for k,v in grant.items() if k!='id'}|{'grant_sha256':sha(PRIVATE/'grant.json')})
    start=time.monotonic()
    with (PRIVATE/'stdout.txt').open('x') as out,(PRIVATE/'stderr.txt').open('x') as err:
        proc=subprocess.Popen([sys.executable,'-B','-S',str(Path(__file__).resolve()),'_child'],cwd=ROOT,stdout=out,stderr=err)
        pid,status,ru=os.wait4(proc.pid,0);proc.returncode=os.waitstatus_to_exitcode(status)
    native=dict(exit_code=proc.returncode,reaped=pid==proc.pid,elapsed_seconds=time.monotonic()-start,
                CPU_seconds=ru.ru_utime+ru.ru_stime,peak_RSS_bytes=ru.ru_maxrss*1024,
                started_children=1,reaped_children=1,remaining_owned_workers=0)
    public('native_exit.json',native)
    n=check_protected();gate()
    consumed=(PRIVATE/'ONE_SHOT.json').exists()
    public('grant_consumption_receipt.json',dict(grant_sha256=sha(PRIVATE/'grant.json'),
           marker_sha256=sha(PRIVATE/'ONE_SHOT.json') if consumed else None,
           consumed=consumed,nonrefundable=True,retry=False,resume=False))
    ok=proc.returncode==0 and read(ROOT/OUT/'worker_completion.json')['status']=='PASS'
    public('completion_audit.json',dict(status='PASS' if ok else 'FAIL_MANDATORY_STOP',
           native=native,protected_file_SHA_matches=n,source_input_runtime_SHA_pass=True,
           one_shot_consumed=consumed,grant_issued=1,grant_reused=False,
           new_molecular_science_calls=0,unknown_costs_imputed=False,
           independent_numeric_pass=ok,full_population_certificate=False))
    public('STOP.json',dict(status='MANDATORY_STOP_COMPLETE' if ok else 'MANDATORY_STOP_FAILURE_NO_RETRY',
           further_science_authorized=False,push_performed=False,production=False))
    print('COMPLETE' if ok else 'STOP_FAILURE',native)
    if not ok: raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['acquire','seal','execute','_child','audit'])
    args=p.parse_args()
    if args.action=='acquire':acquisition()
    elif args.action=='seal':seal()
    elif args.action=='execute':execute()
    elif args.action=='_child':child()
    else:
        gate();n=check_protected()
        require(read(ROOT/OUT/'native_exit.json')['exit_code']==0,'NATIVE_FAILURE')
        require(read(ROOT/OUT/'STOP.json')['status']=='MANDATORY_STOP_COMPLETE','NO_COMPLETION')
        print('PASS_READ_ONLY_FINAL_SHA_AND_NATIVE_AUDIT',n)


if __name__=='__main__':main()
