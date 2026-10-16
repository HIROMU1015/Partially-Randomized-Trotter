"""Metadata checks and separate future authority gate. No numerical imports.

The contract is a candidate. No authority, grant, marker or output is issued by
status/schema/validate-config or this preparation. A future adopted contract and
separate explicit execution approval are both required for run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
INPUT_PATH = 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz'
INPUT_SHA = '3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a'
H_SHA = 'de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424'
SOURCE_PATHS = ('src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/__init__.py', 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/control.py', 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/reference.py', 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/execution.py', 'scripts/tracks/algorithm_codesign/run_m1s_gate0_independent_v1.py', 'src/trottertracks/__init__.py', 'src/trottertracks/algorithm_codesign/__init__.py')
AUTH_PREFIX = 'artifacts/track_b_m1s_independent_execution_authorization/'
OUTPUT_PREFIX = 'artifacts/track_b_m1s_independent_reference_result/'
POLICY = {'schema': 'm1s_gate0_independent_contract_v1', 'status': 'CANDIDATE_NOT_ADOPTED', 'adopted': False, 'contract_grants_execution': False, 'base_result_commit': 'cdcb7f6f151df217960611f502eb1bac8be63f58', 'draft_origin_commit': 'd6d35cfb79b97f9cb218c9169b427aacf6a4a334', 'scope': 'M1S_GATE0_V5_POSTHOC_INDEPENDENT_CORRECTNESS_ONCE', 'input': {'path': 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz', 'sha256': '3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a', 'bytes': 8652}, 'model': {'molecule': 'linear_H4', 'distance_angstrom': '1.00', 'basis': 'sto-3g', 'modes': 8, 'df_rank': 12, 'splits': [3, 6, 9], 'delta_window': None, 'mode_order': 'mode0_MSB_alpha_even_beta_odd', 'state': 'saved_binary64_vector_quotient_by_exact_original_norm2'}, 'comparator': {'path': 'artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json', 'sha256': '8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5', 'bytes': 440470, 'commit': 'cdcb7f6f151df217960611f502eb1bac8be63f58', 'use': 'HASH_BIND_ONLY_UNTIL_ALL_INDEPENDENT_REFERENCES_ARE_FROZEN'}, 'v5_contract': {'path': 'artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/execution_contract_v5_adopted.json', 'sha256': '361e7034850512fcfde76a68ebd5dc3b58de945d1075a696196d75af8ae7547b', 'commit': '50b3669b9253eb889fa4daa87e793646df2d5138'}, 'v5_source': {'commit': '69de022dc8142f703423b1ce0bd42f05e123954e', 'sha256': {'src/trottertracks/algorithm_codesign/m1s_gate0_v5/__init__.py': '289d67f42cd13ca0d9fd919f7a0f1fe98d2c98edc01d8c66d281e315cd5ebbbd', 'src/trottertracks/algorithm_codesign/m1s_gate0_v5/control.py': 'dc027a115ae7b131325bebc9ba13992794daf1a7a216a5de489e691b14dc1022', 'src/trottertracks/algorithm_codesign/m1s_gate0_v5/science.py': '9a1d42916e972cb9b39af462678db1a70888fcab22a5de4f98d11fab0bc1a0ab', 'src/trottertracks/algorithm_codesign/m1s_gate0_v5/budget.py': 'b73f47aa76718abda3f9ef8c4236e9d23cc379c22ce1b87140d4dd3c49638556', 'scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v5.py': '453bda1343048c54726d3a6b8858dd08b6922cdb39b869ed63636cf84a1c6365', 'src/trottertracks/algorithm_codesign/m1s_gate0_tail_bound_spec.py': 'b23c8aade7e3e94c804ad7128ca78c9ade1aa89bd40a7d50963a9c50a8c50732', 'src/trottertracks/__init__.py': '2d51bfa08b2b1e000b86ec53bd463e786529307a7270078600e3b03995e90285', 'src/trottertracks/algorithm_codesign/__init__.py': 'b6503716cfd51d52ca9038e6de5a411eb78380d30451f81df9b81c31b1be090f'}}, 'v5_proof': {'proof_commit': 'b5a185cc0c3aae0e477bfc52b93061e86feb7dec', 'files_sha256': {'docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md': '4d3547493e336c41a0acade491470f8376f872eafd4486c5cd206b8937834156', 'artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/tail_local_model_candidate.json': '88ee296cdf64b166a500671ffed2bad190463c0f4b97adea0c95f386bd1a924c'}, 'coefficient_pipeline_reference_commit': '606036db52da2f080cf846e48824e5e418430762', 'coefficient_pipeline_reference_sha256': '364514fdbb415bf4c43dfac48cda5ea65671323f7c39575f5bec1c7a9840548e'}, 'arithmetic': {'coefficients': 'Gaussian_integer_dyadic_occupation_and_CAR_full_Fock_trace_FWHT', 'metrics': 'exact_integer_moments_and_Fraction; Decimal_32_display_only', 'reference_metric_allowance': '0_EXACT_RATIONAL_IF_SOURCE_CORRECT', 'evidence': 'EXACT_SAVED_BINARY64_ALGEBRA_IF_SOURCE_CORRECT_NOT_FORMALLY_VERIFIED', 'platform_coverage': 'NOT_CERTIFIED'}, 'comparison': {'exact_algebra_and_embedding': '0', 'state_norm2_quality': '2^-40', 'oracle_residual_squared_quality_Ha2': '(1e-8)^2', 'coefficient_L1_Ha': 'gamma_(713+tail_rank)*S_R', 'global_coefficient_L1_Ha': '2*gamma_(24000064)*S*(1+gamma_(65568))', 'HR_matrix_F_Ha': '16*alpha_R; compare exact squared norms', 'probability_mass': '2^-46', 'probability_rounding': 'integer_binary64_nearest_even_cell_of_abs_saved_coeff_over_saved_lambda', 'W_absolute': '2^-24', 'v_absolute': '2^-24', 'rho_absolute': '2^-12', 'rho_defined': 'exact W > 0', 'support_or_sign_discrepancy': 'UNDETERMINED_IF_WITHIN_L1_ENVELOPE_OTHERWISE_FAIL', 'full_HD_saved_path_match': 'UNDETERMINED_NOT_SAVED_BY_V5', 'overall': 'FAIL_precedes_UNDETERMINED_precedes_PASS; missing_full_HD_comparator_prevents_full_scope_PASS', 'identity_scalar_sum': 'gamma_34*(abs(saved_HD_before)+abs(saved_HR_moved)); nearest_even_relation_recorded_only'}, 'resources': {'workers': 1, 'threads_per_library': 1, 'thread_environment': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1'}, 'gpu': False, 'memory_bytes': 2147483648, 'output_bytes': 134217728, 'log_bytes': 134217728, 'context_bytes': 4194304, 'wall_time_seconds': None, 'cpu_time_seconds': None, 'max_exact_integer_bits': 65536, 'max_primitive_events': 500000000, 'max_coefficient_cells': 65536, 'trace_transforms': 7, 'max_matrix_cells': 65536, 'max_transition_cache_entries': 4352, 'max_pauli_cache_entries': 512, 'max_difference_witnesses': 64, 'max_input_decode_bytes': 1048576}, 'runtime': {'python': '3.11.0rc1', 'backend': 'CPython_stdlib_only_no_NumPy_BLAS', 'python_executable_sha256': '320707a1d684b043661306d508d1a8655f507c0bb8e827398cb53680054aa7c8', 'stdlib_source_sha256': {'fractions': 'b11e850e354808b882d13a70a911c29accd1dbdd41757566704e3b7206c74edb', 'decimal': '000c00bad31d126b054c6ec7f3e02b27c0f9a4d579f987d3c4f879cee1bacb81', 'zipfile': 'b3d1f76b7fe20c0b9da8ad587c5f5d3ee681c63f5bb9d99bbe681ff2fc605796', 'struct': '9c231f9497caf513a22dee8f790b07f969b0e45854a0bdd6dd84b492e08c2856', 'ast': '984a4037b30c8b178cc6bf8007d83d68ee5f24b054836547736ef89b6e860722'}}, 'source_commit': None, 'source_sha256': {'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/__init__.py': None, 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/control.py': None, 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/reference.py': None, 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/execution.py': None, 'scripts/tracks/algorithm_codesign/run_m1s_gate0_independent_v1.py': None, 'src/trottertracks/__init__.py': None, 'src/trottertracks/algorithm_codesign/__init__.py': None}, 'adoption_record': {'path': None, 'sha256': None}, 'requires_separate_explicit_user_execution_approval': True, 'execution_adapter': 'IMPLEMENTED_UNEXECUTED', 'automatic_retry': False, 'failure_action': 'PRESERVE_ALL_COMPLETED_REFERENCE_LEDGER_AND_STOP; remaining_splits_UNDETERMINED', 'one_shot_consumed_on_output_reservation': True, 'mandatory_STOP': True}


class ExecutionClosed(PermissionError):
    pass


def read_json(raw):
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('JSON byte cap')
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in b)
    if isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _sha(value, size=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(size)+'}', value) is not None


def _relative(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts or Path(value).as_posix() != value:
        raise ValueError('Repository-relative canonical path required')
    return value


def candidate():
    return read_json(json.dumps(POLICY).encode())


def validate_contract(config, require_adopted=False):
    errors, pending = [], []
    if type(config) is not dict or set(config) != set(POLICY):
        return {'valid': False, 'contract_complete': False, 'actual_execution_permitted': False, 'errors': ['Exact keys required'], 'pending': []}
    variable = {'status', 'adopted', 'source_commit', 'source_sha256', 'adoption_record'}
    for key in set(POLICY)-variable:
        if not _same(config[key], POLICY[key]):
            errors.append('Frozen condition changed: '+key)
    if type(config['adopted']) is not bool or config['status'] not in ('CANDIDATE_NOT_ADOPTED', 'ADOPTED') or (config['status']=='ADOPTED') != config['adopted']:
        errors.append('Invalid adoption classification')
    if config['adopted'] is not True:
        pending.append('validation_contract_adoption')
    if config['source_commit'] is None:
        pending.append('source_commit')
    elif not _sha(config['source_commit'], 40):
        errors.append('Full source commit required')
    if type(config['source_sha256']) is not dict or set(config['source_sha256']) != set(SOURCE_PATHS):
        errors.append('Exact source paths required')
    else:
        for path, digest in config['source_sha256'].items():
            if digest is None:
                pending.append('source_sha256:'+path)
            elif not _sha(digest):
                errors.append('Invalid source hash: '+path)
    record = config['adoption_record']
    if type(record) is not dict or set(record) != {'path','sha256'}:
        errors.append('Adoption record shape')
    elif record['path'] is None or record['sha256'] is None:
        pending.append('adoption_record')
    else:
        try:
            _relative(record['path'])
        except ValueError:
            errors.append('Invalid adoption record path')
        if not _sha(record['sha256']):
            errors.append('Invalid adoption record hash')
    if require_adopted and pending:
        errors.append('Fixed adopted validation contract required')
    return {'valid': not errors, 'contract_complete': not errors and not pending,
            'actual_execution_permitted': False, 'errors': errors, 'pending': pending,
            'numerical_processing_performed': False, 'mandatory_STOP': True}


def _git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args], stderr=subprocess.PIPE)


_SEAL = object()


class _Permit:
    def __init__(self, seal, config, audit):
        if seal is not _SEAL:
            raise ExecutionClosed('Separate validated authority required')
        self._seal, self.config, self.audit = seal, config, audit


def require_execution(permit=None):
    if not isinstance(permit, _Permit) or permit._seal is not _SEAL:
        raise ExecutionClosed('No separately validated independent-verification authority')


def authorize(root, config_path, authority_commit, authority_path):
    root = Path(root).resolve()
    config_path, authority_path = _relative(config_path), _relative(authority_path)
    if not authority_path.startswith(AUTH_PREFIX):
        raise ExecutionClosed('New independent-verification authority namespace required')
    if not _sha(authority_commit, 40):
        raise ExecutionClosed('Explicit full authority commit required')
    head = _git(root,'rev-parse','HEAD').decode().strip()
    if head != authority_commit or _git(root,'status','--porcelain','--untracked-files=all').strip():
        raise ExecutionClosed('Clean fixed authority HEAD required')
    authority_raw = _git(root,'show',head+':'+authority_path)
    if (root/authority_path).read_bytes() != authority_raw:
        raise ExecutionClosed('Authority bytes differ')
    authority = read_json(authority_raw)
    keys = {'schema','scope','explicit_user_scientific_execution_approval','user_approval_record','contract_commit','contract_path','contract_sha256','source_commit','output_directory'}
    if type(authority) is not dict or set(authority) != keys or authority['schema'] != 'm1s_gate0_independent_separate_execution_authorization_v1' or authority['scope'] != POLICY['scope'] or authority['explicit_user_scientific_execution_approval'] is not True or not isinstance(authority['user_approval_record'],str) or not authority['user_approval_record'].strip():
        raise ExecutionClosed('Separate explicit independent-verification approval record required')
    cc = authority['contract_commit']
    if not _sha(cc,40) or cc == head or authority['contract_path'] != config_path:
        raise ExecutionClosed('Earlier adopted validation contract required')
    _git(root,'merge-base','--is-ancestor',cc,head)
    config_raw = _git(root,'show',cc+':'+config_path)
    if (root/config_path).read_bytes() != config_raw or hashlib.sha256(config_raw).hexdigest() != authority['contract_sha256']:
        raise ExecutionClosed('Validation contract byte/hash mismatch')
    config = read_json(config_raw)
    if not validate_contract(config,require_adopted=True)['contract_complete']:
        raise ExecutionClosed('Validation contract not adopted/complete')
    sc = config['source_commit']
    if authority['source_commit'] != sc or sc in (cc,head):
        raise ExecutionClosed('Separate frozen SOURCE commit required')
    _git(root,'merge-base','--is-ancestor',sc,cc)
    _git(root,'merge-base','--is-ancestor',POLICY['base_result_commit'],sc)
    for path, digest in config['source_sha256'].items():
        raw = _git(root,'show',sc+':'+path)
        if (root/path).read_bytes() != raw or hashlib.sha256(raw).hexdigest() != digest:
            raise ExecutionClosed('Independent source identity mismatch')
    adoption = config['adoption_record']
    ar = _git(root,'show',cc+':'+adoption['path'])
    record = read_json(ar)
    if (root/adoption['path']).read_bytes() != ar or hashlib.sha256(ar).hexdigest() != adoption['sha256'] or record.get('record_kind') != 'INDEPENDENT_VALIDATION_CONTRACT_ADOPTION_NOT_EXECUTION_AUTHORIZATION' or record.get('scientific_execution_authorized') is not False:
        raise ExecutionClosed('Validation contract adoption record mismatch')
    bindings = [config['comparator'], config['v5_contract']]
    for binding in bindings:
        raw = _git(root,'show',binding['commit']+':'+binding['path'])
        if (root/binding['path']).read_bytes() != raw or hashlib.sha256(raw).hexdigest() != binding['sha256']:
            raise ExecutionClosed('Frozen v5 comparator/contract identity mismatch')
    for binding, key in [(config['v5_source'],'sha256'),(config['v5_proof'],'files_sha256')]:
        commit = binding.get('commit',binding.get('proof_commit'))
        for path, digest in binding[key].items():
            raw = _git(root,'show',commit+':'+path)
            if (root/path).read_bytes() != raw or hashlib.sha256(raw).hexdigest() != digest:
                raise ExecutionClosed('Frozen v5 SOURCE/proof identity mismatch')
    raw = (root/INPUT_PATH).read_bytes()
    if len(raw) != POLICY['input']['bytes'] or hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ExecutionClosed('Raw input identity mismatch')
    if sys.version.split()[0] != POLICY['runtime']['python'] or not sys.platform.startswith('linux'):
        raise ExecutionClosed('Fixed Python/Linux runtime required')
    if hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() != POLICY['runtime']['python_executable_sha256']:
        raise ExecutionClosed('Fixed Python binary hash required')
    for module, expected in POLICY['runtime']['stdlib_source_sha256'].items():
        origin = importlib.util.find_spec(module).origin
        if hashlib.sha256(Path(origin).read_bytes()).hexdigest() != expected:
            raise ExecutionClosed('Fixed stdlib source hash required: '+module)
    if any(os.environ.get(k) != v for k,v in POLICY['resources']['thread_environment'].items()):
        raise ExecutionClosed('Single-thread environment required')
    output = _relative(authority['output_directory'])
    if not output.startswith(OUTPUT_PREFIX) or not (root/output).resolve().is_relative_to(root) or (root/output).exists():
        raise ExecutionClosed('New exclusive independent-result directory required')
    return _Permit(_SEAL,config,{'authorization_commit':head,'authorization_path':authority_path,'authorization_sha256':hashlib.sha256(authority_raw).hexdigest(),
        'contract_commit':cc,'contract_path':config_path,'contract_sha256':hashlib.sha256(config_raw).hexdigest(),'source_commit':sc,'output_directory':output})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='status',choices=('status','schema','validate-config','run'))
    parser.add_argument('--contract')
    parser.add_argument('--authorization-commit')
    parser.add_argument('--authorization-path')
    args = parser.parse_args(argv)
    try:
        if args.command == 'schema':
            value = candidate()
        elif args.command == 'validate-config':
            value = validate_contract(read_json(Path(args.contract).read_bytes()))
        elif args.command == 'run':
            if not all((args.contract,args.authorization_commit,args.authorization_path)):
                raise ExecutionClosed('Adopted validation contract and separate explicit authority arguments required')
            root = Path(__file__).resolve().parents[4]
            permit = authorize(root,args.contract,args.authorization_commit,args.authorization_path)
            from .execution import run_authorized
            value = run_authorized(root,permit)
        else:
            value = {'status':'IMPLEMENTED_UNEXECUTED_VALIDATION_PREPARATION','actual_execution_permitted':False,'scientific_processing_performed':False,'mandatory_STOP':True}
        print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
        return 0 if value.get('valid',True) else 2
    except (ValueError,TypeError,OSError,KeyError,ExecutionClosed,subprocess.SubprocessError) as exc:
        print(json.dumps({'status':'REFUSED','reason':str(exc),'actual_execution_permitted':False,'mandatory_STOP':True}))
        return 2
