"""Small stdlib-only plan/input gate for the separately approved PS48 study."""
from pathlib import Path
import importlib.metadata
import json
import re
import subprocess
import sys
from .authorization import _canonical, _under, file_sha
from .f2_plan import CONTEXT, COMPILER, THRESHOLDS, SAVED_INPUT_HASHES, case_id

STAGE = 'RECTANGULAR_GIVENS_PS48'
COUNTS = dict(candidate_basis=6, ordered_evolutions=24, full_wrapper_build_requests=48,
    compile_requests=48, raw_operator_checks=48, raw_signal_checks=48,
    compiled_operator_checks=48, compiled_signal_checks=48,
    new_reference_precision_evaluations=0, baseline_build_compile_requests=0)
INPUTS = {p:SAVED_INPUT_HASHES[p] for p in (
    'handoff/f1_bundle.json', 'inputs/physical.json', 'arms/PS.json', 'requirements.lock')}
BASELINE_COMMIT = 'dc7378ab34015287215bcf4ce52147e46d94ecd9'
DESIGN_COMMIT = '70be3c3bae09a8f99fecaa67757da1f8c4887267'
ARTIFACT = 'artifacts/pr_rectangular_givens_preparation/2026-10-24/PLAN.json'


def cases():
    """Fixed identity descriptors, not events or numerical circuit generation."""
    return [dict(id=case_id('PS', i, j, axis), arm='PS', i=i, j=j, axis=axis,
        frame_atom=k, partner=p) for k in range(20,26) for p in (0,2)
        for i,j in ((k,p),(p,k)) for axis in ('X','Y')]


def inspect_preparation(plan_path, source_root, saved_input_root, baseline_root):
    """Read/hash saved bytes only. No QPY decoding, array import or evaluation."""
    root=Path(source_root).resolve(); saved=Path(saved_input_root).resolve()
    baseline=Path(baseline_root).resolve(); plan_bytes=Path(plan_path).read_bytes()
    plan=json.loads(plan_bytes)
    expected=dict(stage=STAGE, context=CONTEXT, compiler=COMPILER, thresholds=THRESHOLDS,
        counts=COUNTS, cases=cases(), saved_B_inputs=INPUTS,
        baseline_commit=BASELINE_COMMIT, design_commit=DESIGN_COMMIT,
        candidate_policy='nature_0.7.2_rectangular_V_transpose_T_first_m_singleton_only',
        approval_requirement='SEPARATE_SCIENTIFIC_APPROVAL_REQUIRED',
        reference_policy='SAVED_F2_90_128_VALIDATED_TARGET_NO_NEW_REFERENCE',
        statistics='UNQUALIFIED_NO_N_NO_N_WEIGHTED_COST',
        angle_policy='BIND_ORIGINAL_SIGNED_CENTRAL_RZ_FROM_QPY_AFTER_APPROVAL',
        raw_operator_atol='1e-10',raw_signal_atol='2e-10',
        internal_support_rule='T=tuple(range(len(original_support)))',
        shared_run_policy='LEGACY_FULL_FRAME_UNCHANGED',
        retry_allowed=False,mandatory_stop=True,wall_time_cap=None,memory_cap=None,
        scientific_executions_in_preparation=0)
    for key,value in expected.items():
        if plan.get(key)!=value: raise ValueError('PS48 plan changed: '+key)
    sources=plan['source_files']
    required={str(p.relative_to(root)) for p in
        (root/'src/trottertracks/pr_dictionary_validation').glob('*.py')}
    required.add('scripts/run_pr_rectangular_givens_saved.py')
    if not required.issubset(sources): raise ValueError('Incomplete SOURCE closure')
    for name,wanted in sources.items():
        if file_sha(_under(root,name).read_bytes())!=wanted: raise ValueError('SOURCE mismatch: '+name)
    for name,wanted in INPUTS.items():
        if file_sha(_under(saved,name).read_bytes())!=wanted: raise ValueError('Saved B mismatch: '+name)
    for name,wanted in plan['public_files'].items():
        if file_sha(_under(root,name).read_bytes())!=wanted: raise ValueError('Saved public result mismatch: '+name)
    expected_paths={f'{folder}/{row["id"]}{suffix}' for row in cases() for folder,suffix in (
        ('build','.qpy'),('build','.json'),('build','.reference128.npy'),('compiled','.json'))}
    if set(plan['baseline_files'])!=expected_paths:
        raise ValueError('Baseline inventory must bind exactly the selected saved artifacts')
    for name,wanted in plan['baseline_files'].items():
        if file_sha(_under(baseline,name).read_bytes())!=wanted:
            raise ValueError('Saved baseline mismatch: '+name)
    for row in cases():
        raw=json.loads(_under(baseline,f'build/{row["id"]}.json').read_text())
        compiled=json.loads(_under(baseline,f'compiled/{row["id"]}.json').read_text())
        if (raw['id'],raw['i'],raw['j'],raw['axis'])!=(row['id'],row['i'],row['j'],row['axis']):
            raise ValueError('Baseline case identity mismatch')
        if raw['reference_precision_evaluations']!=2 or compiled['id']!=row['id']:
            raise ValueError('Saved reference/compile provenance missing')
        if raw['raw_qpy_sha256']!=plan['baseline_files'][f'build/{row["id"]}.qpy']:
            raise ValueError('Saved QPY binding mismatch')
    return dict(plan=plan,plan_sha256=file_sha(plan_bytes),source_root=root,
        saved_input_root=saved,baseline_root=baseline,
        technical_status='READY_FOR_SEPARATE_PS48_SCIENTIFIC_APPROVAL',
        angle_status='QPY_HASH_BOUND_VALUES_UNREAD_UNTIL_SCIENTIFIC_APPROVAL',
        scientific_execution_count=0)


def verify_execution(packet_dir, plan_path, source_root, saved_input_root, baseline_root, output):
    """Fresh scope only. Preparation approval and old F1/F2 grants do not pass."""
    packet=Path(packet_dir).resolve()
    grant_bytes=(packet/'authorization.json').read_bytes(); grant=json.loads(grant_bytes)
    if (grant.get('stage'),grant.get('authority'),grant.get('status'))!=(
            STAGE,'USER_EXPLICIT_RECTANGULAR_PS48','APPROVED_BOUND_ONE_SHOT'):
        raise PermissionError('Separate explicit PS48 scientific approval required')
    if grant.get('retry_allowed') is not False or grant.get('mandatory_stop') is not True:
        raise PermissionError('Fresh one-shot/no-retry/mandatory STOP required')
    approval=(packet/'user_approval.txt').read_bytes()
    if not approval.strip() or grant.get('approval_sha256')!=file_sha(approval):
        raise PermissionError('Unbound scientific approval')
    v=inspect_preparation(plan_path,source_root,saved_input_root,baseline_root)
    if grant.get('plan_sha256')!=v['plan_sha256']:
        raise PermissionError('Unbound PS48 plan')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=v['source_root'],text=True).strip()
    if grant.get('source_commit')!=head: raise PermissionError('SOURCE commit changed')
    if grant.get('source_closure_sha256')!=file_sha(_canonical(v['plan']['source_files'])):
        raise PermissionError('Unbound SOURCE closure')
    if sys.version_info[:3]!=(3,11,9) or sys.prefix==sys.base_prefix:
        raise PermissionError('Dedicated CPython 3.11.9 environment required')
    for name,wanted in v['plan']['dependency_versions'].items():
        if importlib.metadata.version(name)!=wanted: raise PermissionError('Dependency changed: '+name)
    for name,wanted in v['plan']['nature_helper_hashes'].items():
        path=Path(sys.prefix)/'lib/python3.11/site-packages'/name
        if file_sha(path.read_bytes())!=wanted: raise PermissionError('Pinned helper changed: '+name)
    out=Path(output).resolve(); permitted=v['source_root']/'artifacts/pr_rectangular_givens'
    if not out.is_relative_to(permitted) or out==permitted or out.exists():
        raise PermissionError('Fresh dedicated output required')
    if grant.get('output')!=str(out): raise PermissionError('Output binding mismatch')
    token=grant.get('grant_id','')
    if re.fullmatch(r'rectangular_[A-Za-z0-9_-]+',token) is None:
        raise PermissionError('Fresh PS48 grant identity required')
    marker=packet/f'.consumed_{token}'
    if marker.exists(): raise PermissionError('Grant already consumed; no resume')
    v.update(authorization=grant,authorization_digest=file_sha(grant_bytes),
        consumption_path=marker,output=out)
    return v
