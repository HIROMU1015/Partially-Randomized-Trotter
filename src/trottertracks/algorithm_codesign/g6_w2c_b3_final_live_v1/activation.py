"""Owner-operated activation for the explicit USER one-shot approval.

A later, explicitly approved activation-only release may set the ID and bind a
USER approval record to the exact new SOURCE/manifest. No external keys/server
are needed. This is an owner-operated research gate, not malicious-tamper
protection. The anchor is a local content binding to the preserved instruction, not an external signature.
"""
import os
from .guard import require
from ..g6_w2c_b3_live_preflight_v1 import spec as S

# Post-512 technical preparation has no new science execution approval.
# The spent run's approved anchor remains intact in its historical SOURCE.
APPROVED_ACTIVATION_ID=None


def _check(c,m):
    # This check is unconditional in the preparation release, before touching
    # any output/store, issuing a permit, importing science or making bits.
    require(type(APPROVED_ACTIVATION_ID) is str and len(APPROVED_ACTIVATION_ID)==64,
            'EXPLICIT_USER_APPROVAL_AND_NEW_ACTIVATION_RECORD_REQUIRED')
    import ast
    from pathlib import Path
    tree=ast.parse(Path(__file__).read_bytes())
    recorded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='APPROVED_ACTIVATION_ID' for t in n.targets))
    require(recorded==APPROVED_ACTIVATION_ID,'ACTIVATION_SOURCE_CONSTANT_MISMATCH')
    require(m['activation_record']==dict(id=APPROVED_ACTIVATION_ID,
        purpose='USER_APPROVED_ONE_SHOT_512_CORRECTNESS'),'ACTIVATION_ANCHOR')
    approval=m['approval_record']
    require(type(approval) is dict and set(approval)=={'approver','explicit_execution_approval',
        'approval_text','approval_text_sha256','manifest_binding','measured_SOURCE'},'USER_APPROVAL_RECORD')
    from hashlib import sha256
    require(approval['approver']=='USER' and approval['explicit_execution_approval'] is True
        and type(approval['approval_text']) is str and bool(approval['approval_text'].strip())
        and approval['approval_text_sha256']==sha256(approval['approval_text'].encode()).hexdigest(),
        'EXPLICIT_USER_APPROVAL_TEXT')
    require(approval['manifest_binding']==S.digest({k:v for k,v in m.items() if k!='approval_record'})
        and approval['measured_SOURCE']==m['measured_SOURCE'],'USER_APPROVAL_TARGET')
    require(m['SCIENCE_RUN_AUTHORIZED'] is True and m['artificial_only'] is False
        and m['run_ready'] is True and m['content_adopted'] is True
        and m['contract_grants_execution'] is True and m['science_release']==m['measured_SOURCE']
        and m['scientific_adoption']=='USER_ADOPTED_512_CORRECTNESS'
        and m['approval_status']=='EXPLICIT_USER_EXECUTION_APPROVED'
        and m['seed_namespace']==S.SEED and m['planned']==512
        and m['retry']==0 and type(m['retry']) is int and m['resume'] is False
        and m['mandatory_STOP'] is True,'APPROVED_FIXED_RUNTIME_SCOPE')
    require(type(m['run_id']) is str and m['run_id'].startswith('B3_CORRECTNESS_512/')
        and m['global_scope']==m['run_id'] and m['global_claim']==S.digest([m['run_id'],S.digest(c)]),
        'APPROVED_NEW_RUN_SCOPE')
    from pathlib import Path
    root=Path(m['state_directory'])
    require(root.is_absolute() and root.resolve()==root and root.is_dir()
        and root.stat().st_uid==os.getuid() and str(root/'output')==m['storage_destination'],
        'APPROVED_OWNED_STATE_OUTPUT')
    return S.digest(approval)


def admit(c,m,state_root):
    binding=_check(c,m)
    require(str(state_root)==m['state_directory'],'ADMISSION_STATE_ROOT')
    from .contract import verify_source_commit
    verify_source_commit(c,m['measured_SOURCE'])
    from .gate import LivePermit
    return LivePermit(m,S.digest(c),binding,os.getpid())


def validate(permit,c,m):
    binding=_check(c,m)
    require(permit.approval_binding==binding and permit.manifest==m
        and permit.parent_pid in (os.getpid(),os.getppid()),'PERMIT_OWNER_LINEAGE')
