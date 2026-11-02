"""Read-only stdlib source/contract verifier; never opens a registered fixture."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11'
BASE='73b2102a305f018d22348418adc7939cae63f0b2'
APPEND_ONLY=['PROJECT_MAP.md','docs/README.md','scripts/README.md','src/trotterlib/README.md',
             'tests/README.md','docs/tracks/algorithm_codesign/README.md','docs/research/研究概要・現状.md']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    checks=[]
    def check(name,condition):
        checks.append({'name':name,'pass':bool(condition)})
        if not condition:raise AssertionError(name)
    old=json.loads((ROOT/'artifacts/track_b_df_correction_comparison_preparation/2026-10-11/contract_v1.json').read_text())
    new=json.loads((ART/'contract_v2.json').read_text())
    for section,keys in [('CD',['modes','coefficients','primitive_ranks','primitive_signs','seed','t','deltas','steps','gamma','kappa','RTE_microsteps','paired_cutoff','mean_degree','A_mass','B_mass']),
                         ('confidence',['complex_epsilon','axis_epsilon','total_alpha','axis_task_alpha','tasks','axes']),
                         ('E',['seed','generation_order','active_modes','primitive_ranks','primitive_signs','theta','u','kappa','controlled_masks']),
                         ('T',['gate_precision','version','mpmath','dps','options','wheel_sha256','synthesis_calls_cap','RZ_occurrence_cap_per_circuit'])]:
        for key in keys:check(section+'.'+key+'_unchanged',new[section][key]==old[section][key])
    check('error_maxima_unchanged',new['error']['axis_maxima']==old['error']['axis_maxima'])
    for key in ('wall_seconds_total','CPU_seconds_total','aggregate_RSS_MiB','CD','native','synthesis','output_bytes','log_bytes'):
        check('cap.'+key+'_unchanged',new['caps'][key]==old['caps'][key])
    for key in ('A_controlled','A_uncontrolled','B','phase_reserve','maximum_logical_keys','fill_empty_slots'):
        check('T_selection.'+key+'_unchanged',new['T']['selection'][key]==old['T']['selection'][key])
    check('compiler_unchanged',new['compiler']==old['compiler'])
    auth=json.loads((ART/'authorization.json').read_text())
    check('not_science_authorized',new['science_execution_authorized'] is False and auth['science_execution_authorized'] is False)
    check('runs_zero',auth['runs']==0 and new['runs_authorized_now']==0)
    check('retry_zero',new['retries']==0 and auth['retries']==0)
    check('marker_absent',not (ROOT/'artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/one_shot_marker.json').exists())
    check('mandatory_STOP',new['mandatory_STOP'] and auth['mandatory_STOP'])
    review=ROOT/new['review_original'];check('review_original_identity',hashlib.sha256(review.read_bytes()).hexdigest()==new['review_sha256'])
    for path in APPEND_ONLY:
        original=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
        check('append_only.'+path,(ROOT/path).read_bytes().startswith(original))
    changed=subprocess.check_output(['git','diff','--name-only','-z','--diff-filter=MDRT',BASE],cwd=ROOT).decode().rstrip('\0').split('\0')
    changed=[path for path in changed if path]
    check('all_preexisting_tracked_changes_are_documentation_append_only',set(changed)<=set(APPEND_ONLY))
    files=list((ROOT/'src/trottertracks/algorithm_codesign/df_correction_comparison').glob('*.py'))
    files += [Path(__file__),ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py',ROOT/'tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py']
    for path in files:
        ast.parse(path.read_text());check('syntax.'+str(path.relative_to(ROOT)),True)
    manifest=json.loads((ART/'source_manifest_v2.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['path']
        check('source_identity.'+row['path'],len(p.read_bytes())==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result={'kind':'SOURCE_AND_CONTRACT_METADATA_VERIFICATION_ONLY','base':BASE,'checks':checks,'checks_passed':len(checks),
        'registered_CD_tasks':0,'registered_E_compile_calls':0,'Clifford_T_synthesis_calls':0,
        'molecule_NPZ_GPU_LP_trajectory_calls':0,'old_sources_results_markers_STOPs_unchanged':True,
        'science_execution_authorized':False,'mandatory_STOP':True}
    if args.output:
        with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'checks_passed':len(checks),'registered_comparison_calls':0,'mandatory_STOP':True}))


if __name__=='__main__':main()
