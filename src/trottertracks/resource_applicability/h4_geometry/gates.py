"""Production authorization and checkout gates, separate from JSON examples."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from dataclasses import dataclass
from .identity import Stop, require, sha, fingerprint, hash_id

BASE = 'b662dbd72e49fa713a25c716f323843e547e973b'
CONTRACT = 'artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2'
PLAN_SHA = '18aa36a2776d38852657f154a88c381b3299fbc009007f20a2d95fcb865d9f7a'
PLAN_FP = 'c76e8f1f6de5a2625affde38cc471b8214b299f343da2817aadbb3ebabc7d933'
MANIFEST_SHA = '14cc5d0cc4da2b82168a0640cf8ff70ddf382b79810842bab1d7fedfae029f70'
SOURCE_AUDIT = 'artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/source_freeze_v1.json'
DISTANCES = ('0.70', '0.80', '0.90', '1.10', '1.40', '1.60')
ARTIFACT_ANCHOR = '/home/AbeHiromu/projects/partially-randomized-trotter'
RUN_ID = 'track-a-h4-geometry-v2-20261006-run02'
OUTPUT = ARTIFACT_ANCHOR + '/artifacts/resource_applicability/track_a_h4_geometry_execution/' + RUN_ID
PYTHON = '/home/AbeHiromu/venvs/trotter-common/bin/python'
THREAD_ENV = {k: '1' for k in ('PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'OPENBLAS_NUM_THREADS',
              'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS')}
THREAD_ENV['QISKIT_PARALLEL'] = 'false'


def git_blob(root, commit, path):
    require(not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts, 'source path')
    return subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + path])


def verify_contract(root):
    get = lambda p: git_blob(root, BASE, CONTRACT + '/' + p)
    require(sha(get('artifact_manifest_v2.json')) == MANIFEST_SHA, 'contract manifest')
    manifest = json.loads(get('artifact_manifest_v2.json'))
    require(len(manifest['files']) == 37, 'contract inventory')
    for entry in manifest['files']:
        require(sha(git_blob(root, BASE, entry['path'])) == entry['sha256'], 'contract blob mismatch')
    require(sha(get('zero_compute_plan_v2.json')) == PLAN_SHA, 'plan bytes')
    plan = json.loads(get('zero_compute_plan_v2.json'))
    require(plan['plan_fingerprint'] == PLAN_FP and len(plan['templates']) == 218, 'scope')
    return plan


def environment_matches(reference):
    require(sys.executable == PYTHON and sys.version_info[:3] == (3, 12, 3), 'interpreter identity')
    require(all(os.environ.get(k) == v for k, v in THREAD_ENV.items()), 'process-only thread limits')
    require(len(reference) == 45, 'dependency closure')
    observed = {}
    for name, expected in reference.items():
        d = metadata.distribution(name)
        record = d.read_text('RECORD')
        require(record is not None, 'missing dependency RECORD')
        observed[name] = {'version': d.version, 'installed_record_sha256': sha(record.encode())}
        require(observed[name] == expected, 'environment mismatch: ' + name)
    return observed


def compiler_matches(reference):
    # This import is for compiler metadata only, never a molecular dependency.
    import inspect
    from qiskit import transpile
    defaults = {k: repr(v.default) for k, v in inspect.signature(transpile).parameters.items()}
    plugins = sorted([{'group': e.group, 'name': e.name, 'value': e.value}
                      for e in metadata.entry_points() .select() if e.group.startswith('qiskit.')],
                     key=lambda e: (e['group'], e['name'], e['value']))
    require(defaults == reference['inherited_defaults'], 'compiler default mismatch')
    require(plugins == reference['available_plugins_metadata'], 'compiler plugin mismatch')
    require(metadata.version('qiskit') == reference['qiskit_version'] and
            metadata.version('rustworkx') == reference['rustworkx_version'], 'compiler version mismatch')
    require(reference['explicit_options']['num_processes'] == 1, 'compiler parallelism')
    return reference['explicit_options']


@dataclass(frozen=True)
class Permit:
    stage: str
    plan: dict
    source_root: str
    review_digest: str


def structural_gate(plan,authorization,review):
    """Production wire structure only; semantic/binding checks follow separately."""
    specifications=(
        (plan,'h4-native-execution-plan-v1',{'schema_version':str,'stage':str,'run_id':str,'base_commit':str,
          'contract_plan_fingerprint':str,'source_commit':str,'source_hashes':dict,'source_root':str,
          'artifact_anchor':str,'output_root':str,'distances':list,'requested_workers':int,'binding':str,
          'inputs':(dict,type(None)),'generation_freeze_digest':(str,type(None)),'templates':list,
          'compiler_fingerprint':str,'environment_fingerprint':str,'source_audit_sha256':str}),
        (authorization,'h4-native-authorization-v1',{'schema_version':str,'stage':str,'run_id':str,'permission':str,
          'one_shot':bool,'allowed_cpus':list,'result_prior':bool,'plan_fingerprint':str}),
        (review,'h4-native-stage-review-v1',{'schema_version':str,'stage':str,'run_id':str,'approved':bool,
          'plan_fingerprint':str,'authorization_digest':str}))
    for document,version,schema in specifications:
        require(type(document) is dict and set(document)==set(schema),'production schema fields')
        require(document['schema_version']==version,'production schema version')
        for key,expected in schema.items():
            types=expected if isinstance(expected,tuple) else (expected,)
            require(type(document[key]) in types,'production schema type: '+key)


def authorize(stage, plan, authorization, review, *, explicit_launch):
    """Pure negative boundary: callers must not open inputs or create outputs first."""
    require(stage in ('input_generation', 'signal_compile'), 'unknown stage')
    require(explicit_launch is True, 'explicit launch required')
    structural_gate(plan,authorization,review)
    for document in (plan, authorization, review):
        require(document.get('stage') == stage and document.get('run_id') == RUN_ID, 'stage/run mismatch')
    require(plan.get('binding') == ('SOURCE_BOUND' if stage == 'input_generation' else 'INPUT_BOUND'), 'plan binding')
    require(plan.get('base_commit') == BASE and plan.get('contract_plan_fingerprint') == PLAN_FP, 'base contract')
    require(plan.get('distances') == list(DISTANCES), 'six-distance scope')
    require(plan.get('artifact_anchor') == ARTIFACT_ANCHOR and plan.get('output_root') == OUTPUT, 'artifact roots')
    require(isinstance(plan.get('source_root'), str) and Path(plan['source_root']).is_absolute(), 'actual checkout root')
    require(len(plan.get('source_commit', '')) == 40 and all(c in '0123456789abcdef' for c in plan['source_commit']), 'source commit')
    require(bool(plan.get('source_hashes')), 'source closure missing')
    for value in plan['source_hashes'].values():
        hash_id(value)
    hash_id(plan['source_audit_sha256'])
    p = fingerprint('h4-execution-plan-v1', plan)
    require(authorization.get('plan_fingerprint') == p, 'authorization plan binding')
    require(review.get('plan_fingerprint') == p and review.get('authorization_digest') ==
            fingerprint('h4-authorization-v1', authorization), 'independent review binding')
    require(authorization.get('result_prior') is True and review.get('approved') is True, 'review/authorization')
    require(authorization.get('permission') == stage and authorization.get('one_shot') is True, 'separate one-shot permission')
    require(type(plan.get('requested_workers')) is int and 1 <= plan['requested_workers'] <= 12, 'workers')
    require(isinstance(authorization.get('allowed_cpus'), list) and authorization['allowed_cpus'] and
            all(type(c) is int and c >= 0 for c in authorization['allowed_cpus']), 'explicit CPU permission')
    require(len(set(authorization['allowed_cpus'])) == len(authorization['allowed_cpus']), 'duplicate CPU permission')
    if stage == 'input_generation':
        require(plan.get('inputs') is None and plan.get('generation_freeze_digest') is None, 'no placeholder inputs')
    else:
        require(set(plan.get('inputs', {})) == set(DISTANCES), 'all frozen inputs required')
        hash_id(plan.get('generation_freeze_digest'))
        for inp in plan['inputs'].values():
            require(set(inp) == {'file', 'bytes_sha256', 'input', 'H', 'DF', 'state'}, 'input closure')
            for k in ('bytes_sha256', 'input', 'H', 'DF', 'state'):
                hash_id(inp[k])
    return Permit(stage, plan, plan['source_root'], fingerprint('h4-review-v1', review))


def checkout_gate(permit):
    root, plan = Path(permit.source_root), permit.plan
    require(Path(__file__).absolute() == root/'src/trottertracks/resource_applicability/h4_geometry/gates.py', 'loaded source checkout')
    audit_bytes=(root/SOURCE_AUDIT).read_bytes()
    require(sha(audit_bytes)==plan['source_audit_sha256'],'independent source audit binding')
    audit=json.loads(audit_bytes)
    require(audit['source_commit']==plan['source_commit'] and audit['status']=='SOURCE_BLOBS_VERIFIED','actual frozen source audit')
    require({**audit['new_source_hashes'],**audit['namespace_parent_hashes']}==plan['source_hashes'],'complete source closure')
    for path, expected in plan['source_hashes'].items():
        require(path.endswith('.py') and sha(git_blob(root, plan['source_commit'], path)) == expected, 'frozen source blob')
        require(sha((root / path).read_bytes()) == expected, 'actual checkout source mismatch')
    # Full source inventory prevents an unlisted module from becoming a dependency.
    live = {str(p.relative_to(root)) for p in (root/'src/trottertracks/resource_applicability/h4_geometry').glob('*.py')}
    require(live <= set(plan['source_hashes']), 'unbound source module')
    contract = verify_contract(root)
    reference = json.loads(git_blob(root, BASE,
        'artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/environment_binding_audit_v1.json'))
    environment_matches(reference['dependency_observations'])
    sources=json.loads(git_blob(root,BASE,CONTRACT+'/static_source_audit_v2.json'))
    for item in sources['source_findings']:
        if 'absolute_path' in item:
            path=Path(item['absolute_path'])
            require(str(path).startswith('/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages/') and
                    path.suffix=='.py' and sha(path.read_bytes())==item['sha256'],'installed molecular source mismatch')
    for path,expected in audit['installed_source_hashes'].items():
        require(path.startswith('/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages/') and
                path.endswith('.py') and sha(Path(path).read_bytes())==expected,'reviewed installed source closure')
    options = compiler_matches(contract['compiler_environment_reference']['compiler'])
    require(plan.get('compiler_fingerprint') == contract['compiler_environment_reference']['compiler_fingerprint'] and
            plan.get('environment_fingerprint') == contract['compiler_environment_reference']['environment_fingerprint'], 'environment plan binding')
    require(plan.get('templates') == contract['templates'], 'template substitution')
    return contract, options
