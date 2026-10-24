"""Verify saved quotations, design links and unchanged historical evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
COMMIT='21fa1d08cf53422bd6e44e832ce8cc453f6d3ebf'
KEY='h4_central_r2_static_design_20261025'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
    evidence=read(BASE/'source_evidence.json')
    assert all(sha(ROOT/p)==h for p,h in evidence['files'].items())
    assert all(sha(Path(p))==h for p,h in evidence['compiler_source_files'].items())
    prior=json.loads(subprocess.check_output(['git','show',COMMIT+':artifacts/validation_manifest.json'],cwd=ROOT))
    current=read(ROOT/'artifacts/validation_manifest.json')
    assert set(current)==set(prior)|{KEY} and all(current[k]==v for k,v in prior.items())
    quoted=read(BASE/'local_results_QUOTED.json');assert quoted['rows']==read(ROOT/quoted['source'])['rows']
    law=read(BASE/'law_QUOTED.json');model=read(ROOT/law['model_path']);result=read(ROOT/law['result_path'])
    assert law['order_effective_probabilities']==result['normalization']['PR']['order_effective_probabilities']
    assert law['component_effective_probabilities']==result['normalization']['PR']['component_effective_probabilities']
    assert law['component_metadata']==model['cases']['PR']['components']
    assert law['angles_binary64']==model['cases']['PR']['law']['angles']
    assert law['saved_normalization']==model['cases']['PR']['saved_normalization']
    art=read(BASE/'artificial_verification.json');assert art['total']==11
    assert all(sha(BASE/p)==h for p,h in art['log_sha256'].items())
    for name,count in [('artificial_design_tests.log',7),('reused_selected_target_tests.log',4)]:
        text=(BASE/name).read_text();assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
    summary=read(BASE/'design_summary.json');contract=read(BASE/'next_contract_DRAFT_DISABLED.json')
    assert all(v==0 for v in summary['current_counts'].values()) and summary['PR_law_rho'] is None
    assert not summary['formal_certificate_accepted'] and summary['mandatory_STOP'] and not summary['automatic_next_stage']
    assert contract['execution_authorized'] is False and contract['dispatch_enabled'] is False
    assert contract['scientific_SOURCE'] is None and contract['launch_command'] is None
    assert not (BASE/'run01').exists() and not (BASE/'grant.json').exists()
    report=ROOT/'docs/non_df_h4_central_r2_design_20261025.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if not target.startswith(('https://','http://')):assert (report.parent/target).resolve().is_file(),target
    allowed={'PROJECT_MAP.md','VALIDATION_STATUS.md','artifacts/validation_manifest.json','docs/README.md','docs/research/研究概要・現状.md'}
    tracked_base=set(subprocess.check_output(['git','ls-tree','-rz','--name-only',COMMIT],cwd=ROOT).decode().split('\0'))
    changed=subprocess.check_output(['git','diff','--name-only','-z',COMMIT],cwd=ROOT).decode().split('\0')
    assert all(p not in tracked_base or p in allowed for p in changed if p),changed
    return dict(status='PASS_STATIC_DESIGN_AND_PRESERVATION',base_commit=COMMIT,
        unchanged_source_and_reference_files=len(evidence['files']),prior_manifest_rows_unchanged=len(prior),
        copied_local_diagnostics=6,artificial_tests_passed=11,new_H4_numerical_calls=0,new_compile=0,new_sampling=0,
        new_scientific_grants=0,full_law_rho=None,formal_certificate_accepted=False,mandatory_STOP=True)

if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
