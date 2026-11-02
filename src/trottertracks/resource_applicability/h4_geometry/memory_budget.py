"""Closed, user-approved worker memory amendment; legacy defaults remain8GiB."""
from .identity import require

GiB = 2**30
DRIVER_CAP = 8 * GiB
WORKER_CAP = 32 * GiB
OBSERVER_AS = 256 * 2**20
OBSERVER_RSS = 64 * 2**20
HEADROOM = 16 * GiB
PARAMETERS = dict(schema_version='h4-worker-memory-budget-v1',
    driver_AS_RSS=DRIVER_CAP, worker_AS_RSS=WORKER_CAP, workers=4,
    observer_AS=OBSERVER_AS, observer_RSS=OBSERVER_RSS,
    headroom=HEADROOM, admission_bytes=int(152.25 * GiB))


def validate_cap(cap, workers):
    require(type(cap) is int and cap in (DRIVER_CAP, WORKER_CAP), 'closed worker memory cap')
    require(type(workers) is int and 1 <= workers <= 12, 'memory worker count')
    require(cap == DRIVER_CAP or workers == 4, '32GiB amendment requires four workers')
    return cap


def required_available(workers, cap):
    validate_cap(cap, workers)
    return max(120 * GiB, DRIVER_CAP + workers * cap + HEADROOM) + OBSERVER_AS


def verify(profile):
    require(type(profile) is dict and set(profile) == set(PARAMETERS) and
        all(type(profile[k]) is type(v) and profile[k] == v for k, v in PARAMETERS.items()),
        'closed worker memory profile')
    return dict(profile)
