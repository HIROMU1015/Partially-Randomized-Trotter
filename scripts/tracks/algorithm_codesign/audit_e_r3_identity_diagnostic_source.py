"""Static/metadata-only source audit; no registered numeric import/evaluation."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12'
BASE='47bb7e8ef67690846ef4d4769a1f3057e0922061'
PREFIX='src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/'


def identity(p,root=ROOT):
    b=(root/p).read_bytes();return {'path':p,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def load(p):return json.loads((ROOT/p).read_bytes())
def write(p,data):
    with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-unfrozen-draft',action='store_true')
    args=parser.parse_args()
    checks=[]
    def check(name,value):
        checks.append({'check':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    old_snapshot=load('artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/source_manifest_diagnostics_v1.json')['files']
    for row in old_snapshot:check('prior diagnostic protected file '+row['path'],identity(row['path'])==row)
    old_results=load('artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json')['result_file_inventory']
    for row in old_results:check('old result '+row['path'],identity(row['path'])==row)
    append=load(str((ART/'documentation_append_receipt_v1.json').relative_to(ROOT)))['records']
    for row in append:
        b=(ROOT/row['path']).read_bytes();original=b[:row['original_bytes']]
        check('append-only '+row['path'],len(b)==row['original_bytes']+row['appended_bytes']
            and hashlib.sha256(original).hexdigest()==row['original_sha256']
            and original==subprocess.check_output(['git','show',BASE+':'+row['path']],cwd=ROOT))
    diff=subprocess.check_output(['git','diff','--name-only','-z','--diff-filter=MDRT',BASE],cwd=ROOT).decode().rstrip('\0').split('\0')
    check('only preexisting docs changed',set(diff)-{''}=={r['path'] for r in append})
    intake=load(str((ART/'review_intake_v1.json').relative_to(ROOT)))
    review=identity(intake['repository_path'])
    check('original GPT review bytes',review['bytes']==intake['bytes'] and review['sha256']==intake['sha256'])
    contract=load(str((ART/'contract_v1.json').relative_to(ROOT)))
    for row in list(contract['inputs'].values())+[contract['legacy_helper']]:
        check('fixed identity '+row['path'],identity(row['path'])==row)
    job=load(contract['inputs']['saved_job']['path']);inputs=load(contract['inputs']['saved_inputs']['path'])
    check('r3 job/input hex metadata matches',job['r']==3 and job['stage']=='native' and job['frames']==inputs['3'])
    check('r3 rank2 metadata shape only',len(job['frames'])==3 and all(len(v)==3 and all(len(r)==2 and all(len(z)==2 for z in r) for r in v) for v in job['frames']))
    check('old marker pointer retained',job['marker_sha256']==contract['old_marker_sha256'])
    prior=load(contract['inputs']['historical_expected_record']['path'])
    expected=next(r['circuit_fingerprint'] for r in prior['central_compilations'] if r['active_modes']==3 and r['controlled'] is False)
    check('historical expected retained',expected==contract['historical_expected_fingerprint'])
    check('fixed order/scope',contract['path_order']==['current','legacy'] and contract['scope']['modes']==3
          and contract['scope']['controlled'] is False and contract['scope']['rank']==[2,2,2])
    old_contract=load('artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json')
    check('same numeric runtime',contract['runtimes']['numeric']==old_contract['runtimes']['numeric'])
    check('resource caps not enlarged from old science native/total',
          contract['caps']['CPU_seconds_total']<=old_contract['caps']['CPU_seconds_total']
          and contract['caps']['CPU_seconds_worker']<=old_contract['caps']['native']['CPU_seconds']
          and contract['caps']['aggregate_RSS_MiB']<=old_contract['caps']['aggregate_RSS_MiB']
          and contract['caps']['worker_RSS_MiB']<=old_contract['caps']['native']['worker_RSS_MiB']
          and contract['caps']['output_bytes']<=old_contract['caps']['output_bytes'])
    check('new wall caps absent',contract['caps']['wall_seconds_total'] is None and contract['caps']['wall_seconds_worker'] is None)
    auth=load(str((ART/'authorization.json').relative_to(ROOT)))
    check('separate authorization pending',auth['diagnostic_execution_authorized'] is False and auth['science_execution_authorized'] is False and auth['runs']==0 and auth['source_commit'] is None)
    check('contract bound in pending auth',auth['contract_sha256']==identity(str((ART/'contract_v1.json').relative_to(ROOT)))['sha256'])
    check('new result namespace absent',not (ROOT/'artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v1').exists())
    evaluation=ast.parse((ROOT/(PREFIX+'evaluation.py')).read_bytes())
    calls=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(evaluation) if isinstance(n,ast.Call)]
    check('one payload/expm syntactic call',calls.count('_circuit_payload')==1 and calls.count('expm')==1)
    check('no native/PR synthesis call in evaluation',not set(calls)&{'transpile','transpile_and_measure_cost','synthesize','cd_task','e_recipe','cd_recipe','pauli_table','Operator','default_rng'})
    old_native=ast.parse((ROOT/'src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py').read_bytes())
    old_expm=next(n for n in ast.walk(old_native) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='expm')
    new_expm=next(n for n in ast.walk(evaluation) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='expm')
    class Alias(ast.NodeTransformer):
        def visit_Attribute(self,n):
            return ast.Name(id='raw_t',ctx=ast.Load()) if isinstance(n.value,ast.Name) and n.value.id=='t' and n.attr=='value' else self.generic_visit(n)
    check('current raw rotation AST unchanged modulo alias',ast.dump(Alias().visit(old_expm),include_attributes=False)==ast.dump(new_expm,include_attributes=False))
    for name in ['launch.py','worker.py']:
        tree=ast.parse((ROOT/(PREFIX+name)).read_bytes())
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        modules=[n.module or '' for n in imports if isinstance(n,ast.ImportFrom)]+[a.name for n in imports if isinstance(n,ast.Import) for a in n.names]
        check('no top-level numeric imports '+name,not any(m.startswith(('numpy','scipy','qiskit','mpmath')) for m in modules))
    tests=load(str((ART/'focused_tests_v2.json').relative_to(ROOT)))
    check('34 final tests passed',tests['passed'] and tests['test_counts']=={'tests':34,'failures':0,'errors':0,'skipped':0})
    for row in tests['source_files']:check('test source binding '+row['path'],identity(row['path'])==row)
    check('only off-domain tests and no native bodies',tests['registered_input_array_or_matrix_or_payload_calls']==0 and tests['native_transpile_body_calls']==0 and tests['prior104_tests_repeated'] is False)
    source_paths={r['path'] for r in old_snapshot}|{r['path'] for r in old_results}|{
        r['path'] for r in contract['inputs'].values()}|{contract['legacy_helper']['path'],intake['repository_path']}
    source_paths|={str(p.relative_to(ROOT)) for p in (ROOT/PREFIX).glob('*.py')}
    source_paths|={'scripts/tracks/algorithm_codesign/run_e_r3_identity_diagnostic.py',
        'scripts/tracks/algorithm_codesign/verify_e_r3_identity_diagnostic_preparation.py',str(Path(__file__).relative_to(ROOT)),
        'tests/tracks/algorithm_codesign/test_e_r3_identity_diagnostic_preparation.py',str((ART/'contract_v1.json').relative_to(ROOT)),
        str((ART/'focused_test_plan_v1.json').relative_to(ROOT)),str((ART/'review_intake_v1.json').relative_to(ROOT))}
    check('manifest no molecule NPZ input',not any(p.lower().endswith('.npz') for p in source_paths))
    manifest={'kind':'E_R3_IDENTITY_ONLY_SOURCE_MANIFEST_V1','base_publication_commit':BASE,
        'files':[identity(p) for p in sorted(source_paths)],'authorization_receipt_and_self_excluded':True,
        'future_source_commit_reported_in_publication_receipt':True,'diagnostic_execution_authorized':False,'mandatory_STOP':True}
    def publish_draft(path,data):
        if not args.refresh_unfrozen_draft:
            write(path,data);return
        check('refresh only at unfrozen review-intake HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='838768a936ea475d75cae81cb35927a03583832b')
        check('refresh before execution authorization',auth['source_commit'] is None and auth['runs']==0 and auth['diagnostic_execution_authorized'] is False)
        checkpoint=load(str((ART/'draft_checkpoint_before_profile_guard/receipt.json').relative_to(ROOT)))
        row=next(r for r in checkpoint['files'] if r['original_path']==str(path.relative_to(ROOT)))
        check('retain previous draft before refresh '+path.name,identity(row['archived_path'])['sha256']==row['sha256']
              and identity(str(path.relative_to(ROOT)))['sha256']==row['sha256'])
        with path.open('w') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    publish_draft(ART/'source_manifest_v1.json',manifest)
    result={'kind':'STATIC_SOURCE_AND_SAVED_METADATA_SCOPE_AUDIT','checks':checks,'all_checks_passed':all(c['passed'] for c in checks),
        'prior_diagnostic_source_files_unchanged':len(old_snapshot),'old_result_files_unchanged':len(old_results),
        'new_source_manifest_files':len(source_paths),'registered_array_matrix_payload_calls':0,'native_transpile_calls':0,
        'source_manifest_sha256':identity(str((ART/'source_manifest_v1.json').relative_to(ROOT)))['sha256'],
        'old_science_classification':'TECHNICAL_INCONCLUSIVE','new_authorization_pending':True,'mandatory_STOP':True}
    publish_draft(ART/'source_scope_audit_v1.json',result)
    print(json.dumps({k:result[k] for k in ['all_checks_passed','prior_diagnostic_source_files_unchanged','old_result_files_unchanged','new_source_manifest_files','mandatory_STOP']}))


if __name__=='__main__':main()
