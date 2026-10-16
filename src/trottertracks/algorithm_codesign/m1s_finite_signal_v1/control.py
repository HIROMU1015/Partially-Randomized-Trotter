"""Finite-step metadata only. Contract adoption and execution approval are separate."""
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
from fractions import Fraction
sys.dont_write_bytecode = True
AUTH_PREFIX='artifacts/track_b_m1s_finite_signal_execution_authorization/'
OUTPUT_PREFIX='artifacts/track_b_m1s_finite_signal_result/'
POLICY={'schema': 'm1s_finite_signal_contract_v1', 'status': 'CANDIDATE_NOT_ADOPTED', 'adopted': False, 'coverage_model_adopted': False, 'contract_grants_execution': False, 'scope': 'M1S_FINITE_STEP_COMPLEX_SIGNAL_PILOT_ONCE', 'base_commit': 'cc2f451324b7292ae50cef94ff794d46f225d892', 'model': {'molecule': 'linear_H4', 'distance_angstrom': '1.00', 'basis': 'sto-3g', 'modes': 8, 'df_rank': 12, 'splits': [3, 6, 9], 'mode_order': 'mode0_MSB_alpha_even_beta_odd', 'state': 'saved_binary64_vector_quotient_by_exact_original_norm2', 'delta_window_Ha_inverse': ['1/16', '1/8']}, 'input': {'path': 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz', 'sha256': '3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a', 'bytes': 8652}, 'saved_law': {'path': 'artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json', 'sha256': '8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5', 'bytes': 440470, 'commit': 'cdcb7f6f151df217960611f502eb1bac8be63f58', 'use': 'HASH_BIND_ONLY_UNTIL_ALL_INDEPENDENT_REFERENCES_ARE_FROZEN'}, 'review': {'path': 'docs/research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md', 'sha256': '07864ab817cc5fa8f3b8f2a834619c3ea4af090dec3e553cb7be3438a401f6fb', 'role': 'received GPT scientific interpretation and recommendation; not execution authorization'}, 'independent_result': {'path': 'artifacts/track_b_m1s_independent_reference_result/2026-10-16/v1_one_shot/result.json', 'sha256': '32c73f57a1b3fada6bc185311bd56007628849222ad77e4a33e8383a2d0085ac', 'commit': 'cc2f451324b7292ae50cef94ff794d46f225d892'}, 'grid': {'delta_rel_Ha_inverse': ['1/8', '1/16'], 'R_total_labels_per_step': [2, 4], 'T_rel_Ha_inverse': ['1', '4'], 's_zero_control': True, 'order': ['split', 'R', 'delta', 'T', 'method'], 'methods': ['shared_pm', 'ordinary', 'independent_pm'], 'nonzero_cells': 24, 'nonzero_signals': 72, 'zero_signals': 36}, 'semantics': {'shared': 'r=R; theta=lambda*delta/(2R); Psi(X)=D Phi_theta^R(D X D) D; D=exp(+i delta HD/4)', 'ordinary': 'r=R; angle=lambda*delta/R; O=Do M_angle^R Do; Do=exp(+i delta HD/2); z=<O^s>', 'independent': 'r=R/2 on each branch; angle=lambda*delta/R; V=D M_angle^(R/2) D; z=<(V^s)(V^s)>; never V_dagger', 'reference': '<psi|exp(+i T_rel H_DF_full)|psi>/original_norm2; no exact-eigenstate substitution', 'negative_branch': 'same label order with every sign reversed; not full-word adjoint', 'sampling': 'positions and steps IID; no trajectory Monte Carlo; exact expectation recursion', 'cost': 's*R independent labels for all three; shared has 2*s*R branch event gates; others s*R; conceptual accounting only, no native claim'}, 'target': {'H_full': 'exact saved-binary64 DF Hamiltonian by occupation arithmetic', 'HD': 'exact DF prefix plus exact saved v5 HR identity scalar; newly reconstructed, not old unsaved HD comparator', 'HR_law': 'saved v5 signed nonidentity binary64 coefficient table; lambda=sum exact abs coefficients; p=exact abs(c)/lambda', 'raw_H_vs_realized_split': 'report exact entrywise row-norm bound on HD+HR_law-H_full; do not silently repair HD or renormalize stored probability floats', 'state': 'saved 256-vector, exact original norm2 quotient, mode0 MSB alpha even beta odd; no sector reduction'}, 'precision': {'eps_signal_abs': '2^-24', 'eps_phase_rad': '2^-18', 'phase_visibility_floor': '1/4', 'matrix_exp_degree': 24, 'matrix_exp_scaled_norm_bound': '1/8', 'matrix_exp_max_squarings': 32, 'arithmetic_model': 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED', 'IEEE_binary64_RN': True, 'BLAS_dot_model': 'gamma_(8*d+32)*||Ahat||F*||Bhat||F', 'norm_reduction_model': 'computed_F_norm inflated by gamma_(8*n_cells+64)', 'trig_model': 'sin/cos argument exact-to-float error plus absolute 2^-51 library allowance; unverified assumption', 'radius_model': 'conditional upper norms, propagated operand errors, nextafter outward scalar budgets; not a platform certificate', 'underflow': 'no unaccounted underflow/FTZ/overflow; observed subnormals or nonfinite => technical STOP; zero-producing underflow remains an assumption', 'independent_correctness': 'future one-shot only: exact rational 2x2 physical-word enumeration, degree32 Taylor with analytic remainder, R2/s2/noncommuting HD; direct twirl vs FWHT; full256 literal-tensor monomial twirl probes for every split', 'phase': 'principal Arg(z conj(z_ref)); no unwrap; balls intersecting negative-real branch cut or floor => UNDETERMINED_PHASE', 'comparison': 'difference intervals use sum of signal radii; visibility advantage resolved only if interval excludes zero; no scientific GO threshold', 'relative_ratio': 'null unless denominator loss interval excludes zero; never promote G9 relative ratios below absolute precision', 'scalar_budget_evaluation': 'positive expressions <64 operations: relative 512u headroom plus nextafter; no unaccounted underflow', 'phase_library_model': 'complex abs/hypot relative error <=2u; asin/atan2/log absolute allowance 2^-51 on accepted domain; conditional and unverified'}, 'resources': {'workers': 1, 'gpu': False, 'thread_environment': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1'}, 'memory_bytes': 4294967296, 'result_bytes': 134217728, 'log_bytes': 134217728, 'context_bytes': 4194304, 'wall_time_seconds': None, 'cpu_time_seconds': None, 'max_matrix_dimension': 256, 'max_dense_superoperator_cells': 0, 'max_exact_integer_bits': 65536, 'max_exact_events': 500000000, 'max_matrix_products': 100000, 'max_twirl_applications': 10000, 'max_transition_cache_entries': 4352, 'max_pauli_cache_entries': 512, 'max_input_decode_bytes': 1048576, 'max_matrix_cells': 65536, 'max_saved_law_terms': 512, 'max_word_reference_paths': 48, 'max_word_reference_gates_per_wordpair': 16, 'exact_event_counter_scope': 'reused exact-input kernel ticks; rational toy has separate fixed structural caps'}, 'runtime': {'python': '3.11.0rc1', 'python_executable_sha256': '320707a1d684b043661306d508d1a8655f507c0bb8e827398cb53680054aa7c8', 'numpy': '1.26.4', 'numpy_build_files_sha256': {'numpy.libs/libgfortran-040039e1.so.5.0.0': '14afb3129b1a8b50bc40a3b0820c7f1152ea9bc10121aab152943f7057472886', 'numpy.libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so': '9254d0854dd7615e11de28d771ae408878ca8123a7ac204f21e4cc7a376cc2e5', 'numpy.libs/libquadmath-96973f99.so.0.0.0': '934c22ded0e7d169c4d4678876c96051adf3d94545da962f60b41659b075da3b', 'numpy/__config__.py': '3ae7fe8ec40a0b49635f0b4853615a843326e009db5c043d64f5512c391c07e1', 'numpy/__init__.py': '22cd1535fa14d74ef6f457cca149ffdc80875f460be313b8f895273f78bc402e', 'numpy/core/_multiarray_umath.cpython-311-x86_64-linux-gnu.so': 'a735e4e8355b75c800112af8a5b1731b891b2016ed57067352eea6cab0aa00bc'}}, 'source_commit': None, 'source_sha256': {'src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/__init__.py': None, 'src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/control.py': None, 'src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/exact_input.py': None, 'src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/numerics.py': None, 'src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/word_reference.py': None, 'src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/pilot.py': None, 'scripts/tracks/algorithm_codesign/run_m1s_finite_signal_v1.py': None, 'src/trottertracks/__init__.py': None, 'src/trottertracks/algorithm_codesign/__init__.py': None}, 'adoption_record': {'path': None, 'sha256': None}, 'requires_separate_explicit_execution_approval': True, 'one_shot': True, 'automatic_retry': False, 'failure_action': 'save completed rows and budgets, mark remaining UNDETERMINED, mandatory STOP; phase/contrast unresolved are per-row diagnostics', 'mandatory_STOP': True}

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


class ExecutionClosed(PermissionError):
    pass

def planned_rows(config):
    """Declarative grid/count metadata only; no input values or signal calls."""
    result=[]
    for split in config['model']['splits']:
        for R in config['grid']['R_total_labels_per_step']:
            for delta in config['grid']['delta_rel_Ha_inverse']:
                for T in (['0'] if config['grid']['s_zero_control'] else [])+config['grid']['T_rel_Ha_inverse']:
                    steps=Fraction(T)/Fraction(delta)
                    if steps.denominator!=1:raise ValueError('grid time not an integer step count')
                    for method in config['grid']['methods']:
                        result.append({'split':'G'+str(split),'R':R,'delta':delta,'T':T,'s':int(steps),'method':method,
                          'r_per_active_branch':R//2 if method=='independent_pm' else R,
                          'conceptual_independent_labels':int(steps)*R,
                          'branch_Pauli_exp_events':int(steps)*R*(2 if method=='shared_pm' else 1),
                          'deterministic_half_events':int(steps)*(2 if method=='ordinary' else 4)})
    return result

def validate_contract(value,require_adopted=False):
    errors=[];pending=[]
    if type(value) is not dict or set(value)!=set(POLICY):
        return {'valid':False,'contract_complete':False,'actual_execution_permitted':False,'errors':['exact keys required'],'pending':[]}
    variable={'status','adopted','coverage_model_adopted','source_commit','source_sha256','adoption_record'}
    for k in set(POLICY)-variable:
        if not _same(value[k],POLICY[k]):errors.append('frozen condition changed: '+k)
    for k in ['adopted','coverage_model_adopted']:
        if type(value[k]) is not bool:errors.append('bool required: '+k)
        if value[k] is not True:pending.append(k)
    if value['status'] not in ['CANDIDATE_NOT_ADOPTED','ADOPTED'] or (value['status']=='ADOPTED')!=(value['adopted'] is True):errors.append('invalid status')
    if not _sha(value['source_commit'],40):pending.append('source_commit')
    if type(value['source_sha256']) is not dict or set(value['source_sha256'])!=set(POLICY['source_sha256']):errors.append('source path set')
    else:
        for p,h in value['source_sha256'].items():
            if not _sha(h):pending.append('source hash:'+p)
    record=value['adoption_record']
    if type(record) is not dict or set(record)!={'path','sha256'}:errors.append('adoption shape')
    elif record['path'] is None or record['sha256'] is None:pending.append('adoption_record')
    else:
        try:_relative(record['path'])
        except ValueError:errors.append('adoption path')
        if not _sha(record['sha256']):errors.append('adoption hash')
    if require_adopted and pending:errors.append('adopted complete contract required')
    return {'valid':not errors,'contract_complete':not errors and not pending,'actual_execution_permitted':False,'errors':errors,'pending':pending,'mandatory_STOP':True}

_SEAL=object()
class _Permit:
    def __init__(self,seal,config,audit):
        if seal is not _SEAL:raise ExecutionClosed('separate fixed authority required')
        self.seal=seal;self.config=config;self.audit=audit
def require_permit(permit):
    if not isinstance(permit,_Permit) or permit.seal is not _SEAL:raise ExecutionClosed('no finite-step science permit')
def _git(root,*a):return subprocess.check_output(['git','-C',str(root),*a],stderr=subprocess.PIPE)
def authorize(root,contract_path,authority_commit,authority_path):
    root=Path(root).resolve();contract_path=_relative(contract_path);authority_path=_relative(authority_path)
    if not authority_path.startswith(AUTH_PREFIX):raise ExecutionClosed('new finite-step authority namespace required')
    if not _sha(authority_commit,40):raise ExecutionClosed('full authority SHA required')
    if _git(root,'rev-parse','HEAD').decode().strip()!=authority_commit or _git(root,'status','--porcelain','--untracked-files=all').strip():raise ExecutionClosed('clean fixed authority HEAD required')
    raw=_git(root,'show',authority_commit+':'+authority_path);auth=read_json(raw)
    if (root/authority_path).read_bytes()!=raw:raise ExecutionClosed('authority bytes')
    keys={'schema','scope','explicit_user_scientific_execution_approval','user_approval_record','contract_commit','contract_path','contract_sha256','source_commit','output_directory'}
    if type(auth) is not dict or set(auth)!=keys or auth['schema']!='m1s_finite_signal_separate_execution_authorization_v1' or auth['scope']!=POLICY['scope'] or auth['explicit_user_scientific_execution_approval'] is not True or not isinstance(auth['user_approval_record'],str) or not auth['user_approval_record'].strip():raise ExecutionClosed('separate explicit finite-step approval required')
    C=auth['contract_commit']
    if not _sha(C,40) or C==authority_commit or auth['contract_path']!=contract_path:raise ExecutionClosed('earlier adopted contract required')
    _git(root,'merge-base','--is-ancestor',C,authority_commit)
    cr=_git(root,'show',C+':'+contract_path);config=read_json(cr)
    if (root/contract_path).read_bytes()!=cr or hashlib.sha256(cr).hexdigest()!=auth['contract_sha256'] or not validate_contract(config,True)['contract_complete']:raise ExecutionClosed('adopted contract/hash')
    S=config['source_commit']
    if auth['source_commit']!=S or S in (C,authority_commit):raise ExecutionClosed('separate earlier SOURCE required')
    for earlier,later in [(POLICY['base_commit'],S),(S,C)]:
        if earlier==later:raise ExecutionClosed('strict SOURCE order')
        _git(root,'merge-base','--is-ancestor',earlier,later)
    for p,h in config['source_sha256'].items():
        b=_git(root,'show',S+':'+p)
        if (root/p).read_bytes()!=b or hashlib.sha256(b).hexdigest()!=h:raise ExecutionClosed('SOURCE hash: '+p)
    ar=config['adoption_record'];b=_git(root,'show',C+':'+ar['path']);record=read_json(b)
    if (root/ar['path']).read_bytes()!=b or hashlib.sha256(b).hexdigest()!=ar['sha256'] or record.get('record_kind')!='FINITE_SIGNAL_CONTRACT_ADOPTION_NOT_EXECUTION_AUTHORIZATION' or record.get('scientific_execution_authorized') is not False:raise ExecutionClosed('adoption record')
    for binding in [config['input'],config['saved_law'],config['independent_result'],config['review']]:
        p=binding['path'];b=(root/p).read_bytes()
        if hashlib.sha256(b).hexdigest()!=binding['sha256']:raise ExecutionClosed('input/evidence hash: '+p)
    runtime=config['runtime']
    if sys.version.split()[0]!=runtime['python'] or hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()!=runtime['python_executable_sha256']:raise ExecutionClosed('Python runtime')
    spec=importlib.util.find_spec('numpy');np_root=Path(spec.origin).parent.parent
    for p,h in runtime['numpy_build_files_sha256'].items():
        q=np_root/p
        if hashlib.sha256(q.read_bytes()).hexdigest()!=h:raise ExecutionClosed('NumPy build hash: '+p)
    if any(os.environ.get(k)!=v for k,v in config['resources']['thread_environment'].items()):raise ExecutionClosed('single-thread environment required')
    output=_relative(auth['output_directory'])
    if not output.startswith(OUTPUT_PREFIX) or not (root/output).resolve().is_relative_to(root) or (root/output).exists():raise ExecutionClosed('exclusive new output required')
    return _Permit(_SEAL,config,{'authorization_commit':authority_commit,'authorization_path':authority_path,'authorization_sha256':hashlib.sha256(raw).hexdigest(),'contract_commit':C,'contract_path':contract_path,'contract_sha256':hashlib.sha256(cr).hexdigest(),'source_commit':S,'output_directory':output})

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='status',choices=['status','schema','validate-config','run'])
    parser.add_argument('--contract');parser.add_argument('--authorization-commit');parser.add_argument('--authorization-path')
    args=parser.parse_args(argv)
    try:
        if args.command=='schema':value=POLICY
        elif args.command=='validate-config':value=validate_contract(read_json(Path(args.contract).read_bytes()))
        elif args.command=='run':
            if not all([args.contract,args.authorization_commit,args.authorization_path]):raise ExecutionClosed('adoption and separate approval arguments required')
            root=Path(__file__).resolve().parents[4]
            permit=authorize(root,args.contract,args.authorization_commit,args.authorization_path)
            from .pilot import run_authorized
            value=run_authorized(root,permit)
        else:value={'status':'SOURCE_PREPARATION_CANDIDATE_UNEXECUTED','actual_execution_permitted':False,'mandatory_STOP':True}
        print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));return 0 if value.get('valid',True) else 2
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError,ExecutionClosed) as exc:
        print(json.dumps({'status':'REFUSED','reason':str(exc),'actual_execution_permitted':False,'mandatory_STOP':True}));return 2
