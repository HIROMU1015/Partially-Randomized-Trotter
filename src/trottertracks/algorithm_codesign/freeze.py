"""Source-content preparation and launch checks; text files only."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys

from .input_contract import INPUT

EXECUTION_ID = 'bf1-20261005-development-v1'
PREPARATION_REL = 'artifacts/track_b_bf1_preparation/2026-10-05/v2'
DOMAIN_REL = 'artifacts/track_b_bf1_preparation/2026-10-05/v1/domain_manifest.json'
DOMAIN_SHA256 = 'caece92bd2d9f827b0f1f17281865869420bf923273ff294fd30cabd24b51efe'
SCIENCE_OUTPUT_REL = 'artifacts/track_b_bf1_execution/2026-10-05/v1'
REVIEWED_COMMIT = '15465b0b856d80f9cfde495fd3d22434825f1cc8'
REVISION_REVIEWED_COMMIT = '2d0ddafa95aefb272a36a13f259ee0b45bcc80ef'
AUTHORIZATION_REL = 'artifacts/track_b_bf1_authorization/2026-10-05/authorization_v1.json'
AUTHORIZATION_DOC_REL = 'docs/tracks/algorithm_codesign/bf1_execution_authorization_v1.md'
PUBLICATION_SCHEME = 'source_plus_single_authorization_only_child_v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with path.open('xb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n')


def environment():
    packages = ('numpy', 'scipy', 'mpmath', 'pytest', 'qiskit', 'openfermion', 'openfermionpyscf', 'numba')
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in packages},
                threads={p: os.environ.get(p) for p in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                       'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def shared_sources(root):
    """Transitive static relative-import graph, including deferred imports.

    Root __init__ is bypassed, GPU module is a rejecting stub; subpackage
    __init__ files are included. This is an explicit text-source inventory.
    """
    base = root/'src/trotterlib'
    pending = ['pr2_new_series_validation', 'pr2_s0_s1_validation',
               'pr2_matched_accuracy_m1_execution', 'df_rte_tail', 'df_trotter.decompose', 'rte']
    observed, files = set(), set()
    while pending:
        name = pending.pop()
        if name in observed or name == 'df_gpu_statevector':
            continue
        observed.add(name)
        file = base/(name.replace('.', '/')+'.py')
        if not file.is_file():
            file = base/name.replace('.', '/')/'__init__.py'
        if not file.is_file():
            continue
        files.add(str(file.relative_to(root)))
        parts = name.split('.')
        if file.name == '__init__.py':
            package = parts
        else:
            package = parts[:-1]
        for count in range(1, len(parts)):
            pending.append('.'.join(parts[:count]))
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                parent = package[:len(package)-(node.level-1)]
                target = parent+((node.module or '').split('.') if node.module else [])
                if target:
                    pending.append('.'.join(target))
                if not node.module:
                    pending.extend('.'.join(parent+[alias.name]) for alias in node.names)
    return sorted(files)


def source_inventory(root):
    b_files = [str(p.relative_to(root)) for directory in
               ('src/trottertracks', 'scripts/tracks/algorithm_codesign', 'tests/tracks/algorithm_codesign')
               for p in (root/directory).rglob('*.py')]
    documents = ['pyproject.toml', 'AGENTS.md', 'PROJECT_MAP.md', 'docs/README.md',
      'scripts/README.md', 'src/trotterlib/README.md', 'docs/research/研究概要・現状.md',
      'docs/research/研究ノート/2026-10-05_track_b_bf1_preregistration.md',
      'docs/research/研究ノート/2026-10-05_track_b_bf1_execution_gate_revision.md',
      'docs/tracks/algorithm_codesign/README.md',
      'docs/tracks/algorithm_codesign/bf0_mathematical_contract.md',
      'docs/tracks/algorithm_codesign/bf0_prior_art_claim_matrix.md',
      'docs/tracks/algorithm_codesign/bf1_minimal_pilot_proposal.md',
      'docs/tracks/algorithm_codesign/bf0_external_review_request_20261004.md',
      'docs/tracks/algorithm_codesign/bf1_preregistration_v1.md',
      'docs/tracks/algorithm_codesign/bf1_implementation_and_review_20261005.md',
      'docs/tracks/algorithm_codesign/bf1_execution_gate_revision_v2.md']
    return [dict(path=p, bytes=len(data), sha256=sha(data))
            for p in sorted(set(b_files+documents+shared_sources(root)))
            for data in [(root/p).read_bytes()]]


def check_text_path(path):
    value = PurePosixPath(path)
    if value.is_absolute() or '..' in value.parts or value.suffix not in ('.py', '.md', '.toml', '.json'):
        raise ValueError("Source seal accepts explicit repository text paths only")


def committed_text(root, commit, path):
    check_text_path(path)
    return subprocess.check_output(['git', '-C', str(root), 'show', commit+':'+path])


def verify_authorization_child(root, source_commit, authorization):
    """B scheme: one direct, non-merge, authorization-only child of source.

    The authorization binds its parent source SHA; its own SHA is read from
    Git at launch, so publication creates no self-reference.
    """
    try:
        source_oid = git(root, 'rev-parse', '--verify', source_commit+'^{commit}')
    except subprocess.CalledProcessError as exc:
        raise PermissionError('Unknown source commit') from exc
    if source_oid != source_commit:
        raise PermissionError('Authorization must identify the full source commit SHA')
    head = git(root, 'rev-parse', 'HEAD')
    if git(root, 'rev-list', '--parents', '-n', '1', head).split() != [head, source_oid]:
        raise PermissionError('HEAD must be a single authorization-only child of source_commit')
    changes = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-status', '--no-renames',
                                       '-z', source_oid, head]).decode().split('\0')[:-1]
    pairs = list(zip(changes[::2], changes[1::2]))
    allowed = {AUTHORIZATION_REL, AUTHORIZATION_DOC_REL}
    if len(changes) % 2 or not pairs or any(status != 'A' or path not in allowed for status, path in pairs):
        raise PermissionError('Authorization commit changed a non-authorization path or an existing file')
    if ('A', AUTHORIZATION_REL) not in pairs:
        raise PermissionError('Authorization JSON must be added in the authorization commit')
    committed = committed_text(root, head, AUTHORIZATION_REL)
    working = (root/AUTHORIZATION_REL).read_bytes()
    if working != committed or canonical(json.loads(committed)) != canonical(authorization):
        raise PermissionError('Authorization must match the committed, clean JSON')
    return head


def verify_launch(root, plan, authorization, domain_bytes, tests_bytes):
    """Reject draft authorization BEFORE any molecular pathname operation."""
    if authorization.get('science_execution_authorized') is not True:
        raise PermissionError('BF1_EXECUTION_NOT_AUTHORIZED')
    expected = dict(execution_id=EXECUTION_ID, plan_fingerprint=sha(canonical(plan)),
                    domain_sha256=sha(domain_bytes), semantic_report_sha256=sha(tests_bytes),
                    review_verdict='APPROVED_FOR_ONE_BF1_RUN', source_content_sealed=True,
                    run_count=1, mandatory_stop=True, science_retry_authorized=False,
                    automatic_next_stage=None, BF2_authorized=False, input=INPUT,
                    publication_scheme=PUBLICATION_SCHEME)
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError('Authorization does not bind the reviewed plan/source/input: '+key)
    if not authorization.get('explicit_user_execution_instruction'):
        raise PermissionError('Missing separate explicit execution instruction')
    if plan['execution_id'] != EXECUTION_ID or plan['input'] != INPUT:
        raise ValueError('Plan/input contract altered')
    if plan.get('publication_scheme') != PUBLICATION_SCHEME:
        raise ValueError('Plan does not freeze the authorization publication scheme')
    if sha(domain_bytes) != DOMAIN_SHA256:
        raise ValueError('The original domain manifest was altered')
    if not json.loads(tests_bytes)['all_passed']:
        raise ValueError('Semantic preflight did not pass')
    source_commit = authorization.get('source_commit')
    if not source_commit:
        raise PermissionError('Source must be separately committed before execution')
    authorization_commit = verify_authorization_child(root, source_commit, authorization)
    if canonical(json.loads(committed_text(root, source_commit, PREPARATION_REL+'/source_plan.json'))) != canonical(plan):
        raise ValueError('Source plan is not bound to source_commit')
    if committed_text(root, source_commit, DOMAIN_REL) != domain_bytes:
        raise ValueError('Domain is not bound to source_commit')
    if committed_text(root, source_commit, PREPARATION_REL+'/synthetic_semantic_report.json') != tests_bytes:
        raise ValueError('Semantic report is not bound to source_commit')
    for row in plan['sources']:
        check_text_path(row['path'])
        actual = (root/row['path']).read_bytes()
        if sha(actual) != row['sha256']:
            raise ValueError('Source-content mismatch: '+row['path'])
        committed = committed_text(root, source_commit, row['path'])
        if sha(committed) != row['sha256']:
            raise ValueError('Source is not commit-bound: '+row['path'])
    if environment() != plan['environment']:
        raise ValueError('STOP_ENVIRONMENT_MISMATCH')
    return dict(source_commit=source_commit, authorization_commit=authorization_commit,
                publication_scheme=PUBLICATION_SCHEME)
