"""Closed, explicitly authorized run10 continuation gates; drafts stay inert."""
import math
from pathlib import Path

from .identity import require, hash_id, sha
from .continuation import PARENT_RUN, PARENT_SOURCE, PARENT_REVIEW, load_receipt, parent_plan
from .resources import WALL_CAP

RESUME_RUN_ID='h4-newhost-signal-compile-20261011-resume01'
ENCODING_SOURCE='3beb3870369eeb8ef76b4ead965c886eba7d16e9'
CHANGED_RUNTIME=frozenset(('identity.py','circuits.py','execution.py','ledger.py','parallel.py','launch_binding.py'))


def resume_run_id(authorization):
    scope=authorization['continuation_amendment']['scope']
    if scope=='run10_encoding_only_resume':return RESUME_RUN_ID
    from .continuation_resume01 import SCOPE, NEXT_RUN
    require(scope==SCOPE, 'closed continuation generation')
    return NEXT_RUN


def structural(plan, authorization):
    present='continuation' in plan
    require(present==('continuation_amendment' in authorization), 'continuation plan/auth pair')
    if not present:return
    value=plan['continuation'];change=authorization['continuation_amendment']
    require(type(value) is dict and set(value)=={'identity_source_commit','receipt','imported_wrappers'} and
            value['identity_source_commit']==PARENT_SOURCE and type(value['imported_wrappers']) is int and
            0<=value['imported_wrappers']<=74784, 'continuation dataset identity/count')
    ref=value['receipt']
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str and
            Path(ref['path']).is_absolute() and Path(ref['path']).is_relative_to('/home/AbeHiromu') and
            '..' not in Path(ref['path']).parts and type(ref['bytes']) is int and 0<ref['bytes']<=64*2**20,
            'continuation fixed receipt reference')
    hash_id(ref['sha256'])
    require(type(change) is dict and set(change)=={'approved','scope','continuation','authority'} and
            type(change['approved']) is bool and change['scope'] in ('run10_encoding_only_resume','resume01_inverse_instruction_resume') and
            change['continuation']==value and type(change['authority']) is dict and
            set(change['authority'])=={'path','sha256'}, 'continuation amendment binding')
    authority=change['authority'];p=Path(authority['path'])
    require(not p.is_absolute() and '..' not in p.parts and str(p) not in ('','.','..'), 'relative continuation authority')
    hash_id(authority['sha256'])
    carry=plan['carry']
    require(set(carry)=={'actual_invocations','charged_bytes','wall_seconds'} and
            type(carry['actual_invocations']) is int and 0<=carry['actual_invocations']<plan['caps']['actual_invocations'] and
            type(carry['charged_bytes']) is int and 0<=carry['charged_bytes']<plan['caps']['output_bytes'] and
            type(carry['wall_seconds']) is float and math.isfinite(carry['wall_seconds']) and
            0<=carry['wall_seconds'] and (plan['caps']['wall_seconds'] is None or carry['wall_seconds']<WALL_CAP),
            'continuation carry within selected caps')


def authorize(plan, authorization):
    structural(plan,authorization)
    require(authorization['continuation_amendment']['approved'] is True and
            plan['run_id']==resume_run_id(authorization) and plan['source_commit']!=PARENT_SOURCE,
            'explicit continuation approval/fresh execution SOURCE')
    require(plan['caps']['actual_invocations']>=74784+plan['carry']['actual_invocations']-plan['continuation']['imported_wrappers'],
            'remaining actual cannot guarantee continuation')


def verify_roots(plan, parent, *, extra_readonly=()):
    new=[Path(plan[k]) for k in ('output_root','control_root')]
    old=[Path(parent[k]) for k in ('output_root','control_root')]
    readonly=[Path(plan[k]) for k in ('input_root','stop_evidence_root','source_root')]
    readonly.append(Path(plan['continuation']['receipt']['path']))
    readonly.extend(Path(p) for p in extra_readonly)
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for a in new for b in old+readonly) and
            not new[0].is_relative_to(new[1]) and not new[1].is_relative_to(new[0]),
            'continuation must use disjoint fresh roots')


def verify_runtime(permit):
    """Verify immutable predecessor before CPU roles, locks or science startup."""
    from . import gates
    from .launch_binding import reference
    plan=permit.plan;authorize(plan,permit.authorization)
    if permit.authorization['continuation_amendment']['scope']=='resume01_inverse_instruction_resume':
        from .continuation_resume01 import verify_runtime as verify_next
        return verify_next(permit)
    parent=parent_plan(plan['source_root'])
    for field in ('inputs','input_root','generation_freeze_digest','templates','contract_plan_fingerprint',
                  'environment_fingerprint','compiler_fingerprint','environment_profile','compiler_profile',
                  'library_cache_profile','memory_pressure_profile','gaussian_synthesis_profile','requested_workers','cpu_proposal','caps'):
        require(plan[field]==parent[field], 'continuation changed run10 science/compiler/resource binding: '+field)
    verify_roots(plan,parent)
    authority=reference(Path(plan['source_root']),permit.authorization['continuation_amendment']['authority'])
    require(authority.get('schema_version')=='h4-user-continuation-authority-v1' and
            authority.get('resume_launch_approved') is True and authority.get('preparation_only') is False and
            authority.get('parent_source')==PARENT_SOURCE and authority.get('parent_review')==PARENT_REVIEW and
            authority.get('preserve_dataset_seeds') is True and authority.get('preserve_consumed_budget') is True and
            isinstance(authority.get('instruction'),str) and bool(authority['instruction'].strip()),
            'user continuation launch authority (preparation alone is insufficient)')
    prefix='src/trottertracks/resource_applicability/h4_geometry/'
    for path,digest in parent['source_hashes'].items():
        if path.startswith(prefix):
            require(sha(gates.git_blob(Path(plan['source_root']),PARENT_SOURCE,path))==digest,
                    'run10 scientific SOURCE blob')
            if Path(path).name not in CHANGED_RUNTIME:
                require(plan['source_hashes'].get(path)==digest, 'continuation changed scientific/monitor module')
    memo=prefix+'encoding_memo.py'
    require(plan['source_hashes'].get(memo)==sha(gates.git_blob(Path(plan['source_root']),ENCODING_SOURCE,memo)),
            'validated encoding memo unchanged')
    store,data=load_receipt(plan['source_root'],plan['continuation']['receipt'])
    require(plan['carry']==data['carry'] and plan['continuation']['imported_wrappers']==data['imported_wrappers'],
            'continuation actual receipt/carry binding')
    require(plan['storage']['prior_charge_bytes']==plan['carry']['charged_bytes'] and
            plan['storage']['cumulative_charge_bound']>=parent['storage']['cumulative_charge_bound']+plan['carry']['charged_bytes'] and
            plan['storage']['required_bytes']>=parent['storage']['required_bytes'] and
            plan['storage']['required_inodes']>=parent['storage']['required_inodes'], 'continuation storage includes carry and full fresh bounds')
    return store
