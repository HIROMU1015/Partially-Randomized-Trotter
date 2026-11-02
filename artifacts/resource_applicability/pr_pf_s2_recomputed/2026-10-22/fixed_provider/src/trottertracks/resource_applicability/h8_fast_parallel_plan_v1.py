"""Metadata-only bounded case waves, no CPU grants or process launching."""
from copy import deepcopy

NORMAL_RSS = 9*2**30
HEAVY_RSS = 24*2**30
AGGREGATE = 76*2**30
RESERVE = 4*2**30
WORKER_CPUS = [20, 21, 24, 25, 26, 27, 28, 29]


def classify(receipt):
    """Only complete validated terminal peaks can support a normal proposal."""
    if receipt is None:
        return 'heavy'
    if (receipt.get('complete') is not True or receipt.get('all_reaped') is not True
        or receipt.get('first_stop_reason') is not None or receipt.get('source_input_compiler_bound') is not True):
        return 'heavy'
    rss = receipt.get('peak_rss_bytes'); as_size = receipt.get('peak_as_bytes')
    if type(rss) is not int or type(as_size) is not int or min(rss, as_size) <= 0:
        raise ValueError('INVALID_MEMORY_RECEIPT')
    # Engineering reserve of 25%; not a peak-memory guarantee for unseen tasks.
    return 'normal' if rss*5 <= NORMAL_RSS*4 and as_size*5 <= 32*2**30*4 else 'heavy'


def resource_proposal(memory_class, workers):
    if memory_class not in ('normal', 'heavy') or type(workers) is not int:
        raise ValueError('RESOURCE_PROPOSAL_CLASS')
    rss = NORMAL_RSS if memory_class == 'normal' else HEAVY_RSS
    slots = min(8, (AGGREGATE-RESERVE)//rss)
    if not 1 <= workers <= slots:
        raise ValueError('RESOURCE_PROPOSAL_ADMISSION')
    cpus = WORKER_CPUS[:workers]
    return dict(memory_class=memory_class, workers=workers,
        proposed_worker_cpu_sets=[[c] for c in cpus], proposed_coordinator_cpu_set=[22,23],
        worker_AS_bytes=32*2**30, worker_RSS_guard_bytes=rss,
        aggregate_RSS_bytes=AGGREGATE, aggregate_reservation_bytes=workers*rss+RESERVE,
        job_admission_bytes=96*2**30, threads_per_worker=1,
        allowed_cpus=[], approved=False, runtime_authorization=False)


def case_waves(registrations, receipts=None):
    """Deterministic memory-homogeneous waves; preserve each task/seed exactly.

    Ordering affects execution only, never selection/seed/normalization.
    Unmeasured tasks default to heavy. No mixed-weight live supervisor changes.
    """
    receipts = receipts or {}
    rows = sorted(registrations, key=lambda r: r['index'])
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('DUPLICATE_TASK_INDEX')
    result = []
    for memory_class in ('heavy', 'normal'):
        selected = [deepcopy(r) for r in rows if classify(receipts.get(r['index'])) == memory_class]
        maximum = 3 if memory_class == 'heavy' else 8
        for start in range(0, len(selected), maximum):
            wave = selected[start:start+maximum]
            result.append(dict(registrations=wave, proposal=resource_proposal(memory_class, len(wave))))
    return result
