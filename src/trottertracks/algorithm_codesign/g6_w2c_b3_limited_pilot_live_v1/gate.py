"""Simple research approval boundary for the exact adopted one-shot pilot."""
from dataclasses import dataclass
from .guard import require
from . import activation
from . import spec as S


@dataclass(frozen=True)
class LivePermit:
    manifest: dict
    candidate_binding: str
    approval_binding: str
    parent_pid: int


def require_live(permit,c,m):
    require(type(permit) is LivePermit,'LIVE_PERMIT_REQUIRED')
    # First, verify independently of the caller's booleans or dataclass fields.
    # Content-bound user approval is required, including for LivePermits.
    activation.validate(permit,c,m)
    require(permit.candidate_binding==S.digest(c) and permit.manifest==m,'LIVE_PERMIT_BINDING')
    require(m['SCIENCE_RUN_AUTHORIZED'] is True and m['artificial_only'] is False,
            'LIVE_APPROVAL_STATE')
    return permit


def dispatch(c,m,state_root):
    permit=activation.admit(c,m,state_root)
    require_live(permit,c,m)
    from .integration import run_live
    return run_live(c,m,state_root,permit)


def assert_spent(permit,c,m):
    require_live(permit,c,m)
    from .run_control import SimpleStore
    store=SimpleStore(m['state_directory'],create=False,permit=permit,candidate=c,manifest=m)
    store._intent_binding(m)
    with store.connect() as db:
        row=db.execute('SELECT c.claim,c.binding,c.status,c.spent,r.owner_pid,r.execution_state FROM claims c JOIN runs r USING(scope) WHERE c.scope=?',(m['global_scope'],)).fetchone()
    require(row==(m['global_claim'],S.digest(m),'SPENT',1,permit.parent_pid,'START_RESERVED/SPENT'),
        'SCIENCE_REQUIRES_PARENT_SPENT_NO_STOP_UNKNOWN')
