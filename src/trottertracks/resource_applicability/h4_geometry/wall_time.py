"""Explicitly approved removal of wall STOP; all resource STOPs remain."""
from pathlib import Path

from .identity import require, hash_id
from .resources import WALL_CAP

TRACE_CAP = 2*2**30
PARAMETERS = dict(schema_version='h4-unlimited-wall-policy-v1', wall_seconds=None,
                  observer_trace_bytes=TRACE_CAP, observer_period_seconds=1,
                  observation_deadline_seconds=5, cumulative_wall_accounting=True,
                  memory_pressure_oom_capacity_identity_stops=True, automatic_retry=False)


def verify(profile):
    require(type(profile) is dict and profile==PARAMETERS, 'closed unlimited-wall profile')
    return profile


def structural(plan, authorization):
    profile=plan.get('wall_time_profile');change=authorization.get('wall_time_amendment')
    require((profile is None)==(change is None), 'wall profile/authority pair')
    cap=plan['caps']['wall_seconds']
    if profile is None:
        require(type(cap) is int and cap==WALL_CAP, 'legacy wall cap unchanged')
        return
    require('continuation' in plan and authorization['continuation_amendment']['scope']=='resume01_inverse_instruction_resume' and
            cap is None, 'unlimited wall only on explicit resume02 continuation')
    require(type(change) is dict and set(change)=={'approved','from','to','profile','authority'} and
            type(change['approved']) is bool and type(change['from']) is int and change['from']==WALL_CAP and
            change['to'] is None and change['profile']==profile, 'wall amendment binding')
    for entry in (profile,change['authority']):
        require(type(entry) is dict and set(entry)=={'path','sha256'} and type(entry['path']) is str and
                bool(entry['path']) and not Path(entry['path']).is_absolute() and '..' not in Path(entry['path']).parts,
                'relative fixed wall policy/authority')
        hash_id(entry['sha256'])


def settings(plan, authorization):
    structural(plan,authorization)
    if plan['caps']['wall_seconds'] is not None:return WALL_CAP, None, False
    require(authorization['wall_time_amendment']['approved'] is True, 'wall removal needs explicit approval')
    return None, TRACE_CAP, True


def verify_runtime(root, plan, authorization, reference):
    cap, trace, unlimited=settings(plan,authorization)
    if not unlimited:return
    profile=verify(reference(root,plan['wall_time_profile']))
    authority=reference(root,authorization['wall_time_amendment']['authority'])
    require(authority.get('schema_version')=='h4-user-unlimited-wall-authority-v1' and
            authority.get('wall_limit_removal_approved') is True and authority.get('accepted_profile')==profile and
            authority.get('preserve_cumulative_accounting') is True and authority.get('other_limits_unchanged') is True and
            isinstance(authority.get('instruction'),str) and bool(authority['instruction'].strip()), 'user wall-removal authority')
