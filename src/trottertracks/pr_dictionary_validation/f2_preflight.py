"""Read-only F2 packet checks; no science imports or grant issuance."""
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
from .authorization import file_sha,_canonical,_under
from .f2_plan import ARMS,LENGTHS,CONTEXT,SAVED_INPUT_HASHES,COMPILER,THRESHOLDS,counts
from .f2_statistics import require_adopted_statistics


def inspect_packet(packet_dir,source_root):
    packet=Path(packet_dir).resolve(); root=Path(source_root).resolve()
    c=json.loads((packet/'contract.json').read_bytes())
    if c.get('stage')!='F2' or c.get('comparison')!='FINITE_TARGET_ONLY':
        raise PermissionError('Only F2 finite-target comparison is supported')
    if c.get('context')!=CONTEXT or c.get('lengths')!=LENGTHS or c.get('counts')!=counts():
        raise PermissionError('P0 F2 context or registered integer plan changed')
    if c.get('compiler')!=COMPILER or c.get('thresholds')!=THRESHOLDS:
        raise PermissionError('Compiler profile or P0 numerical gates changed')
    if c.get('input_files')!=SAVED_INPUT_HASHES:
        raise PermissionError('Saved B inputs changed; F1 law/price reuse is forbidden')
    inputs={}
    for name,expected in SAVED_INPUT_HASHES.items():
        b=_under(packet,name).read_bytes()
        if file_sha(b)!=expected: raise PermissionError('Input mismatch: '+name)
        inputs[name]=b
    evidence_files=c.get('qualification_evidence_files',{})
    for name,expected in evidence_files.items():
        if not name.startswith('qualifications/'):
            raise PermissionError('Qualification evidence needs a separate packet namespace')
        b=_under(packet,name).read_bytes()
        if file_sha(b)!=expected: raise PermissionError('Qualification evidence hash mismatch')
        inputs[name]=b
    bundle=json.loads(inputs['handoff/f1_bundle.json'])
    if {a:len(bundle['arms'][a]) for a in ARMS}!=LENGTHS:
        raise PermissionError('Saved B lengths mismatch')
    if bundle.get('literal_prices') or bundle.get('score_intervals'):
        raise PermissionError('Input-only bundle required')
    sources=c.get('source_files',{})
    required={str(p.relative_to(root)) for base in ('src/trotterlib','src/trottertracks')
              for p in (root/base).rglob('*.py')}
    required.add('scripts/run_pr_dictionary_f2_saved.py')
    if not required.issubset(sources): raise PermissionError('Incomplete local SOURCE closure')
    for name,h in sources.items():
        if file_sha(_under(root,name).read_bytes())!=h: raise PermissionError('SOURCE mismatch: '+name)
    if c.get('source_closure_sha256')!=file_sha(_canonical(sources)):
        raise PermissionError('SOURCE closure digest mismatch')
    blockers=[]
    try:
        require_adopted_statistics(c.get('statistics',{}))
        if c['statistics']['sampling_qualification']['evidence'] not in evidence_files:
            blockers.append('IID/range/finite-bit evidence is not bound')
    except (PermissionError,KeyError,ValueError) as e: blockers.append(str(e))
    qual=c.get('statistics',{}).get('interval_qualification',{})
    if qual.get('status')!='QUALIFIED_FOR_REGISTERED_SCALAR_FUNCTIONS':
        blockers.append('Scalar interval/ceil qualification is pending')
    elif (qual.get('source_sha256')!=sources['src/trottertracks/pr_dictionary_validation/intervals.py']
            or qual.get('adapter_sha256')!=sources['src/trottertracks/pr_dictionary_validation/f2_statistics.py']
            or qual.get('environment_lock_sha256')!=SAVED_INPUT_HASHES['requirements.lock']
            or qual.get('evidence') not in evidence_files):
        blockers.append('Interval qualification SOURCE/environment/evidence mismatch')
    if c.get('contract_status')!='USER_APPROVED_F2_FINAL': blockers.append('Final execution contract not adopted')
    if c.get('source_commit') is None: blockers.append('Published final SOURCE commit not bound')
    return dict(contract=c,inputs=inputs,source_root=root,packet=packet,blockers=blockers,
                technical_status='SOURCE_INPUT_PLAN_CHECKED',science_executed=False)


def verify_packet(packet_dir,source_root,output):
    packet=Path(packet_dir).resolve()
    # Refuse absent/preparation/F1 grants before importing any numerical library.
    b=(packet/'authorization.json').read_bytes(); grant=json.loads(b)
    if (grant.get('stage'),grant.get('authority'),grant.get('status'))!=(
            'F2','USER_EXPLICIT_F2','APPROVED_BOUND_ONE_SHOT'):
        raise PermissionError('A new explicit F2 execution authorization is required')
    if grant.get('retry_allowed') is not False or grant.get('mandatory_stop') is not True:
        raise PermissionError('One-shot grant and mandatory STOP required')
    v=inspect_packet(packet_dir,source_root); c=v['contract']; root=v['source_root']
    if v['blockers']: raise PermissionError('; '.join(v['blockers']))
    for name,data in [('contract_sha256',(packet/'contract.json').read_bytes()),
                      ('approval_sha256',(packet/'user_approval.txt').read_bytes()),
                      ('source_closure_sha256',_canonical(c['source_files']))]:
        if grant.get(name)!=file_sha(data): raise PermissionError('Grant binding mismatch: '+name)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=c['source_commit']:
        raise PermissionError('Wrong frozen SOURCE commit')
    if not sys.flags.isolated or sys.prefix==sys.base_prefix or sys.version_info[:3]!=(3,11,9):
        raise PermissionError('Isolated dedicated CPython 3.11.9 environment required')
    for dist,version in c['dependency_versions'].items():
        if importlib.metadata.version(dist)!=version: raise PermissionError('Environment mismatch: '+dist)
    out=Path(output).resolve(); marker=packet/('consumed-'+file_sha(b))
    if not out.is_relative_to(root/'artifacts/pr_algorithm_improvement_phase2d_f2') or not out.parent.is_dir():
        raise PermissionError('Dedicated F2 output parent must be prepared before grant consumption')
    if str(out)!=grant.get('fresh_output_namespace') or out.exists() or marker.exists():
        raise PermissionError('Fresh bound output and unused grant required')
    return dict(v,output=out,consumption_path=marker,authorization_digest=file_sha(b))
