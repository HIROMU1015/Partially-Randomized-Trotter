"""V0 metadata/AST audit. Never import or execute analysis/bounds kernels."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_v1 import control, spec

INDEXES = ['PROJECT_MAP.md', 'docs/README.md', 'scripts/README.md',
           'src/trotterlib/README.md', 'docs/research/研究概要・現状.md', 'VALIDATION_STATUS.md']
MANIFEST = 'artifacts/validation_manifest.json'
MKEY = 'track_b_m1s_v1_v0_preparation_20261017'
DOCS = ['docs/tracks/algorithm_codesign/m1s_v1_saved_signal_contract_candidate_20261017.md',
        'docs/tracks/algorithm_codesign/m1s_v1_arithmetic_coverage_and_resource_design_20261017.md',
        'docs/research/研究ノート/2026-10-17_track_b_M1S_v1_v0_preparation.md']
NEW_PREFIXES = [spec.PACKAGE + '/', spec.ART + '/']
NEW_PATHS = set(DOCS + [
 'docs/research/m1s_rpe_applicability_scientific_design_review_20261017.md',
 'scripts/tracks/algorithm_codesign/run_m1s_saved_signal_v1.py',
 'scripts/tracks/algorithm_codesign/check_m1s_v1_v0_static.py',
 'tests/tracks/algorithm_codesign/test_m1s_v1_v0_static.py'])


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))


def main():
    checks = []
    def check(name, value):
        checks.append({'name': name, 'status': 'PASS' if value else 'FAIL'})
    def load(path):
        return control.read_json((ROOT/path).read_bytes())
    candidate = load(spec.ART + '/execution_contract_candidate.json')
    control.validate_contract(candidate)
    check('candidate_not_adopted', candidate['adopted'] is False)
    check('coverage_not_adopted', candidate['coverage_model_adopted'] is False)
    check('contract_no_execution_grant', candidate['contract_grants_execution'] is False)
    check('direction_only_adopted', candidate['research_direction_adopted'] is True)
    check('scientific_fields_all_null', all(v is None for v in candidate['science'].values()))
    check('no_materiality_or_resource_GO', candidate['scientific_GO'] is None and candidate['native_costs'] is None and candidate['total_resources'] is None)
    check('source_identity_frozen', candidate['source_identity'] is not None)
    if candidate['source_identity'] is not None:
        S = candidate['source_identity']['commit']
        for path, sha in candidate['source_identity']['files'].items():
            raw = (ROOT/path).read_bytes()
            check('source_hash:' + path, hashlib.sha256(raw).hexdigest() == sha)
            check('source_commit_bytes:' + path, raw == git('show', S+':'+path))
        check('source_descends_from_base', subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',spec.BASE,S],capture_output=True).returncode == 0)
    try:
        control.authorize(ROOT, spec.ART + '/execution_contract_candidate.json', None, None)
    except control.ExecutionClosed:
        check('actual_candidate_refused_without_authority_or_engine', True)
    else:
        check('actual_candidate_refused_without_authority_or_engine', False)
    for suffix in ['analysis', 'bounds']:
        check('engine_never_imported:' + suffix, 'trottertracks.algorithm_codesign.m1s_saved_signal_v1.' + suffix not in sys.modules)
    for path in spec.SOURCE_FILES:
        if not path.endswith('.py'):
            continue
        code = (ROOT/path).read_text()
        tree = ast.parse(code)
        check('AST_parse:' + path, isinstance(tree, ast.Module))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported += [node.module or '']
        check('no_old_kernel_or_scientific_dependency:' + path,
              not any(any(x in n for x in ['numpy','scipy','qiskit','mpmath','m1s_finite','rpe_resource','gate0','trotterlib']) for n in imported))
        check('no_eval_exec_bytecode_compile:' + path, not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ['eval','exec','compile'] for n in ast.walk(tree)))
    bounds = (ROOT/spec.PACKAGE/'bounds.py').read_text()
    engine = (ROOT/spec.PACKAGE/'analysis.py').read_text()
    ctrl = (ROOT/spec.PACKAGE/'control.py').read_text()
    for token in ['atan_small', 'sin_positive', '_log_unit', 'ceil_fraction', 'max_bound_steps', 'max_rational_bits']:
        check('declared_bounded_arithmetic:' + token, token in bounds)
    for token in ['saved_binary64', 'as_integer_ratio', 'TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H', 'UNRESOLVED_ABSOLUTE_CONTRAST', 'ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE', 'CEILING_OF_RATIONAL_UPPER_ENCLOSURE']:
        check('required_semantics:' + token, token in engine)
    check('no_model_drift_double_add', "'model_drift_added_again': False" in engine)
    check('only_deferred_engine_import', 'from .analysis import evaluate' in ctrl and ctrl.index('from .analysis import evaluate') > ctrl.index('output.mkdir()'))
    check('one_shot_before_calculation', ctrl.index('output.mkdir()') < ctrl.index('from .analysis import evaluate'))
    check('no_time_caps', all(candidate['resources'][k] is None for k in ['wall_time_seconds','cpu_time_seconds']))
    schema = load(spec.ART + '/result_schema.json')
    check('schema_rows108', schema['properties']['rows']['minItems'] == schema['properties']['rows']['maxItems'] == 108)
    check('schema_contrasts48', schema['properties']['comparisons']['minItems'] == schema['properties']['comparisons']['maxItems'] == 48)
    check('schema_no_scientific_GO', schema['properties']['scientific_GO'] == {'type':'null'})
    intake = load(spec.ART + '/research_direction_intake.json')
    check('intake_not_science_authority', intake['research_direction_adopted'] is True and intake['scientific_execution_authorized'] is False and intake['experiment_contract_adopted'] is False)
    for k in ['execution_authorization_issued','grant_issued','marker_issued']:
        check('none_issued:' + k, intake[k] is False)
    identities = load(spec.ART + '/input_identity.json')['files']
    for r in identities:
        raw = (ROOT/r['path']).read_bytes()
        check('input_hash:' + r['path'], len(raw) == r['bytes'] and hashlib.sha256(raw).hexdigest() == r['sha256'])
        if r['commit']:
            check('input_fixed_commit:' + r['path'], raw == git('show', r['commit']+':'+r['path']))
    for p in INDEXES:
        check('old_index_body_preserved:' + p, (ROOT/p).read_bytes().endswith(git('show', spec.BASE+':'+p)))
    old = control.read_json(git('show',spec.BASE+':'+MANIFEST))
    new = load(MANIFEST)
    check('old_manifest_entries_preserved', all(new.get(k)==v for k,v in old.items()))
    check('only_new_manifest_entry', set(new)-set(old)=={MKEY})
    for path in DOCS:
        text = (ROOT/path).read_text()
        check('document_LF:' + path, text.endswith('\n') and '\r' not in text)
        check('no_trailing_space:' + path, all(x == x.rstrip() for x in text.splitlines()))
    # Audit all inherited tracked bytes, even ones omitted from the shorter identity list.
    all_paths = git('ls-tree','-r','--name-only','-z',spec.BASE).decode().split('\0')
    unchanged = 0
    changed = []
    for p in filter(None, all_paths):
        if p in INDEXES + [MANIFEST]:
            continue
        data = ROOT/p
        if not data.is_file() or data.read_bytes() != git('show',spec.BASE+':'+p):
            changed.append(p)
        else:
            unchanged += 1
    check('all_old_source_results_authority_markers_preserved', not changed)
    differences = set(filter(None, git('diff','--name-only','-z',spec.BASE).decode().split('\0')))
    untracked = set(filter(None, git('ls-files','--others','--exclude-standard','-z').decode().split('\0')))
    paths = differences | untracked
    check('only_V0_paths_changed', all(p in INDEXES+[MANIFEST] or p in NEW_PATHS or any(p.startswith(x) for x in NEW_PREFIXES) for p in paths))
    check('no_new_execution_authority_marker_or_science_result', not any(p.startswith((spec.AUTH_PREFIX,spec.OUTPUT_PREFIX)) for p in paths))
    result = {'schema':'m1s_v1_v0_static_validation','checks':checks,
              'PASS':sum(c['status']=='PASS' for c in checks),
              'FAIL':sum(c['status']=='FAIL' for c in checks),
              'inherited_tracked_files_preserved':unchanged,
              'unexpected_old_changes':changed,
              'scope':'STATIC_AST_METADATA_CLOSED_GUARD_ONLY',
              'numerical_correctness_tested':False,'runtime_validated':False,
              'scientific_calculations':0,'saved_numeric_reanalysis':0,'shot_forecasts':0,
              'mandatory_STOP':True}
    (ROOT/spec.ART/'static_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['PASS','FAIL','inherited_tracked_files_preserved','scope']}))
    return 1 if result['FAIL'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
