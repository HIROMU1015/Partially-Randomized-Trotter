"""Static/saved-value provenance audit only; no numerical module import."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12'
BASE='86e5c43d17f8f4ba2e232776b5e4beb7e5db3a98'
SOURCE='41349b73157c7781e0b534cf0fe61e06bf439954'
EXECUTION=ROOT.parent/'track-b-df-correction-comparison-one-shot-execution-20261012'
PREFIX='src/trottertracks/algorithm_codesign/df_correction_comparison/'
CHANGED={PREFIX+'native_evaluation.py',PREFIX+'worker.py',
         'tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py'}


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def load(p):return json.loads((ROOT/p).read_bytes())
def identity(p,root=ROOT):
    b=(root/p).read_bytes()
    return {'path':p,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,data):
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    checks=[]
    def check(name,ok):
        checks.append({'check':name,'passed':bool(ok)})
        if not ok:raise AssertionError(name)
    protected=load('artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/source_manifest_v3.json')['files']
    old_results=load('artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json')['result_file_inventory']
    append=load(str((ART/'documentation_append_receipt_v1.json').relative_to(ROOT)))['records']
    allowed_modified=CHANGED|{r['path'] for r in append}
    check('original execution worktree HEAD retained',subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXECUTION,text=True).strip()=='ce0e286430a2a23869cb9417c82113c98064c72c')
    check('original execution worktree clean',not subprocess.check_output(['git','status','--porcelain=v1'],cwd=EXECUTION))
    for row in protected:
        check('protected original execution source '+row['path'],identity(row['path'],EXECUTION)==row)
        if row['path'] not in allowed_modified:
            check('unchanged current protected file '+row['path'],identity(row['path'])==row)
    for row in old_results:
        check('old result bytes '+row['path'],identity(row['path'])==row)
        check('original result bytes '+row['path'],identity(row['path'],EXECUTION)==row)
    for p in ['src/trotterlib/rte_compiled_cost.py',
        'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json',
        'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/authorization.json',
        PREFIX+'launch.py',PREFIX+'guard.py',
        'artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/source_manifest_v3.json']:
        check('unchanged base bytes '+p,(ROOT/p).read_bytes()==git('show',BASE+':'+p))
    for row in append:
        b=(ROOT/row['path']).read_bytes()
        check('original document prefix retained '+row['path'],
              len(b)==row['before_bytes']+row['appended_bytes'] and
              hashlib.sha256(b[:row['before_bytes']]).hexdigest()==row['before_sha256'] and
              b[:row['before_bytes']]==git('show',BASE+':'+row['path']))
    modified=set(git('diff','--name-only','-z','--diff-filter=MDRT',BASE).decode().rstrip('\0').split('\0'))
    check('preexisting diffs limited to observation/test/docs',modified==CHANGED|{r['path'] for r in append})
    native=ast.parse((ROOT/(PREFIX+'native_evaluation.py')).read_bytes())
    old_native=ast.parse(git('show',BASE+':'+PREFIX+'native_evaluation.py'))
    def predicates(tree):
        return [ast.dump(n.test,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.If)]
    check('native guard predicates unchanged',predicates(native)==predicates(old_native))
    def assignment(tree,name):
        return next(ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    for name in ['raw_u','compiler','reference']:
        check('native expression unchanged '+name,assignment(native,name)==assignment(old_native,name))
    adapter=ast.parse((ROOT/(PREFIX+'fingerprint_diagnostics.py')).read_bytes())
    check('adapter no runtime monkeypatch',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
          and n.func.attr in {'settrace','setprofile','setattr','patch'} for n in ast.walk(adapter)))
    report=load(str((ART/'focused_tests_v3.json').relative_to(ROOT)))
    check('104 focused tests final passed',report['passed'] and report['pytest_counts']=={'tests':104,'failures':0,'errors':0,'skipped':0})
    for row in report['source_files']:check('final test source binding '+row['path'],identity(row['path'])==row)
    check('final tests within guards',report['process_receipt']['peak_sampled_aggregate_RSS_MiB']<=512
          and report['process_receipt']['CPU_seconds_parent_and_reaped_children']<=90)
    for name in ['registered_CD_tasks','registered_E_compile_calls','registered_frame_generation',
                 'Clifford_T_synthesis_calls','production_runner_calls','LP_molecule_NPZ_GPU_trajectory_shots_G10_RA_D0_calls']:
        check('zero scope '+name,report[name]==0)
    for n in (1,2):
        check('earlier failed test evidence retained '+str(n),not load(str((ART/f'focused_tests_v{n}.json').relative_to(ROOT)))['passed'])
    difference=load(str((ART/'off_domain_payload_difference_audit_v1.json').relative_to(ROOT)))
    for row in difference['records']:
        check('saved off-domain byte identity '+str((row['control'],row['controlled'])),row['target_bytes_identical'])
        check('saved off-domain strict point residual '+str((row['control'],row['controlled'])),row['strict_operator_residuals']==[0,0])
        if row['controlled']:
            check('controlled payload differences only automatic names '+str(row['control']),
              len(row['payload_differences'])==4 and all(d['path'].endswith('/operation/name')
              and d['old'].startswith('circuit-') and d['new'].startswith('circuit-') for d in row['payload_differences']))
        else:check('uncontrolled payload identical '+str(row['control']),not row['payload_differences'])
    paths={r['path'] for r in protected}|CHANGED|{PREFIX+'fingerprint_diagnostics.py',
      'tests/tracks/algorithm_codesign/test_df_correction_fingerprint_diagnostics.py',
      'scripts/tracks/algorithm_codesign/verify_df_correction_fingerprint_diagnostics_preparation.py',
      str(Path(__file__).relative_to(ROOT))}
    manifest={'kind':'DIAGNOSTIC_PREPARATION_SOURCE_SNAPSHOT_NOT_LAUNCH_AUTHORIZATION','base_audit_commit':BASE,
      'old_science_source':SOURCE,'files':[identity(p) for p in sorted(paths)],
      'old_manifest_retained':True,'RUN_READY':False,'science_execution_authorized':False,
      'future_launch_manifest_and_result_namespace_require_separate_preparation':True,'mandatory_STOP':True}
    write(ART/'source_manifest_diagnostics_v1.json',manifest)
    status={'kind':'SOURCE_PREPARATION_AND_SAVED_CONTROL_PROVENANCE_AUDIT','base_commit':BASE,
      'checks':checks,'all_checks_passed':all(r['passed'] for r in checks),
      'original_execution_protected_source_files_unchanged':len(protected),'original_result_files_unchanged':len(old_results),
      'new_preparation_modified_existing_code_or_test_paths':sorted(CHANGED),
      'unchanged_current_protected_files':sum(r['path'] not in allowed_modified for r in protected),
      'source_snapshot_files':len(paths),'new_snapshot_sha256':identity(str((ART/'source_manifest_diagnostics_v1.json').relative_to(ROOT)))['sha256'],
      'source_manifest_used_by_old_launch_unchanged':True,'old_science_classification':'TECHNICAL_INCONCLUSIVE',
      'registered_cause':'UNRESOLVED','off_domain_automatic_name_hash_difference_observed':True,
      'science_retry':0,'registered_science_compile_synthesis_calls':0,'repair_adopted':False,
      'RUN_READY':False,'science_execution_authorized':False,'mandatory_STOP':True}
    write(ART/'preparation_audit_v1.json',status)
    print(json.dumps({k:status[k] for k in ['all_checks_passed','source_snapshot_files','RUN_READY','mandatory_STOP']}))


if __name__=='__main__':main()
