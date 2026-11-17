"""Select a host's published execution profile for an exact route.

Operational slots/spool belong to the worker; scientific profile fields are
bound to the controller's assignment and preserved for its whole attempt.
"""
from copy import deepcopy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def profile_core(profile):
    keys = ('host_id', 'root', 'interpreter', 'route_views', 'dp_workers',
            'prefix_cache_directory', 'runtime')
    return {key: deepcopy(profile[key]) for key in keys if key in profile}


def route_matches(rule, assignment):
    route = assignment['request']['route']
    return (rule.get('route_kind') == route.get('kind')
            and rule.get('view_commit') == route.get('view_commit'))


def select_host_profile(base, assignment):
    matches = [rule for rule in base.get('route_profiles', []) if route_matches(rule, assignment)]
    if len(matches) > 1:
        raise ValueError('Ambiguous route-specific host profile')
    selected = deepcopy(matches[0]['profile'] if matches else base)
    if selected.get('host_id') != base.get('host_id'):
        raise ValueError('Route-specific profile belongs to another host')
    selected.pop('route_profiles', None)
    for key in ('work_dir', 'slots'):
        if key in base:
            selected[key] = deepcopy(base[key])
    bound = assignment.get('host_execution_profile')
    if bound is not None:
        if digest(bound) != assignment.get('host_execution_profile_sha256'):
            raise ValueError('Assignment host execution profile hash differs')
        if profile_core(selected) != bound:
            raise ValueError('Assignment requires a different route-specific host profile; common fallback is forbidden')
    return selected
