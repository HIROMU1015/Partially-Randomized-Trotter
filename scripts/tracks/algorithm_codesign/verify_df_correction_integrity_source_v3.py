"""Read-only source/metadata verification, no registered recipe or numerical call."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_comparison.storage import artifact_identity,load_json

BASE='b4aba2b698cdc81710dbdbb5769a1686f716ffe2'
ART=ROOT/'artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11'
PREFIX='src/trottertracks/algorithm_codesign/df_correction_comparison/'
APPEND=['PROJECT_MAP.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','tests/README.md',
        'docs/tracks/algorithm_codesign/README.md','docs/research/研究概要・現状.md',PREFIX+'README.md']
REPAIRS=[PREFIX+n for n in ('worker.py','guard.py','launch.py')]+[
 'scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py',
 'tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    checks=[]
    def check(name,value):
        checks.append({'name':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    def old(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    for name in ['angles.py','certification.py','comparison.py','digital.py','intervals.py','native.py','native_evaluation.py']:
        path=PREFIX+name;check('unchanged_scientific_module.'+name,(ROOT/path).read_bytes()==old(path))
    for path in ['src/trottertracks/algorithm_codesign/numerics.py',
       'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json',
       'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/authorization.json',
       'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/source_manifest_v2.json',
       'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/STOP.json',
       'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/runtime_identity_v2.json']:
        check('unchanged.'+path,(ROOT/path).read_bytes()==old(path))
    for path in APPEND:check('append_only.'+path,(ROOT/path).read_bytes().startswith(old(path)))
    modified=subprocess.check_output(['git','diff','--name-only','-z','--diff-filter=MDRT',BASE],cwd=ROOT).decode().rstrip('\0').split('\0')
    check('only_approved_preexisting_modifications',set(modified)-{''}<=set(APPEND+REPAIRS))
    auth=json.loads((ROOT/'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/authorization.json').read_text())
    check('authorization_false_runs0',auth['science_execution_authorized'] is False and auth['runs']==0 and auth['source_commit'] is None)
    check('retry0_STOP',auth['retries']==0 and auth['mandatory_STOP'])
    check('marker_absent',not (ROOT/'artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/one_shot_marker.json').exists())
    intake=json.loads((ART/'review_intake.json').read_text());b=(ROOT/intake['path']).read_bytes()
    check('review_original',len(b)==intake['bytes'] and hashlib.sha256(b).hexdigest()==intake['sha256'])
    tests=json.loads((ART/'focused_tests_final_v3.json').read_text())
    check('88_focused_passed',tests['exit_code']==0 and tests['tests_passed']==88)
    for row in tests['sources']:check('test_source.'+row['path'],hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'])
    for field in ['registered_CD_tasks','registered_E_compile_calls','Clifford_T_synthesis_calls','registered_frame_generation','molecule_NPZ_GPU_LP_trajectory_shots_G10_RA_D0_calls']:
        check('scope_zero.'+field,tests[field]==0)
    proof=json.loads((ART/'metadata_review_example_verification.json').read_text())
    example=ART/'metadata_review_example/Pauli_table.json'
    identity=artifact_identity(example);data=load_json(example)
    check('fabricated_metadata_reconstruction',identity['bytes']==proof['bytes'] and identity['sha256']==proof['sha256'] and data['science_data'] is False)
    check('metadata_4097_rows',len(data['Pauli_coefficients'])==4097)
    for path in (ART/'metadata_review_example').rglob('*'):
        if path.is_file():check('review_file_size.'+str(path.relative_to(ROOT)),path.stat().st_size<=512*1024)
    manifest=json.loads((ART/'source_manifest_v3.json').read_text())
    for row in manifest['files']:
        path=ROOT/row['path'];b=path.read_bytes()
        check('protected_identity.'+row['path'],len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'])
        if path.suffix=='.py':ast.parse(b)
    result={'kind':'LIMITED_INTEGRITY_REPAIR_SOURCE_METADATA_AUDIT','base_source':BASE,'checks':checks,
      'checks_passed':len(checks),'modified_preexisting_paths':modified,
      'science_contract_and_numerical_modules_byte_identical':True,
      'old_artifacts_results_markers_STOP_authorization_unchanged':True,
      'registered_comparison_synthesis_calls':0,'science_execution_authorized':False,'mandatory_STOP':True}
    if args.output:
        with Path(args.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'checks_passed':len(checks),'science_conditions_unchanged':True,'science_execution_authorized':False,'mandatory_STOP':True}))


if __name__=='__main__':main()
