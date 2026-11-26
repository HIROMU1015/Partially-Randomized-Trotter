"""Receive independent, plan-only preparation assignments in one PBS offer."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import groupby
import subprocess
import threading

from .rpe_preparation_artifacts import worker_cache
from .rpe_preparation_worker import assigned_artifacts, is_preparation
from .rpe_q3_worker import digest, safe_id


ACQUISITION_ERRORS = (OSError, ValueError, KeyError, subprocess.SubprocessError)


def _attempt(worker, assignment):
    return worker.work / safe_id(assignment['job_id']) / safe_id(assignment['attempt_id'])


def _plan_only(worker, assignment):
    prep = assignment['request'].get('preparation') or {}
    return (is_preparation(assignment) and prep.get('provider') == 'HCHAIN_SCOPED'
            and not prep.get('input_git_bundles') and not prep.get('workspaces')
            and isinstance(prep.get('inputs'), dict) and set(prep['inputs']) == {'PLAN.json'}
            and set(assignment.get('artifact_manifest', {})) <= {'PLAN.json'}
            and set(assignment.get('artifact_blobs', {})) <= {prep['inputs']['PLAN.json']}
            and not (_attempt(worker, assignment) / 'ASSIGNMENT.json').exists())


def _stage_plans(worker, group, errors):
    cache = worker_cache(worker.profile)
    staged = []
    for assignment in group:
        try:
            if digest(assignment['request']) != assignment['request_sha256']:
                raise ValueError('Assignment request SHA256 differs')
            plan = assigned_artifacts(assignment)['PLAN.json']
            inline = assignment.get('artifact_blobs', {})
            cache.put({plan: inline[plan]} if plan in inline else {})
            staged.append((plan, assignment))
        except ACQUISITION_ERRORS as error:
            errors.append(error)
    missing = cache.missing(dict.fromkeys(plan for plan, _ in staged))
    if missing:
        try:
            cache.put(worker._post('/v1/artifacts/get', {'hashes': missing})['blobs'])
        except ACQUISITION_ERRORS as error:
            errors.append(error)
    # Missing inputs leave the original assignment eligible for replay. The
    # rest of the batch can use the inputs already verified in the local CAS.
    unavailable = set(cache.missing(dict.fromkeys(plan for plan, _ in staged)))
    if unavailable:
        errors.append(ValueError('Preparation input artifacts not available'))
    return [assignment for plan, assignment in staged if plan not in unavailable]


def _receive_plans(worker, assignments, errors, parallelism):
    try:
        worker._receive_assignment(assignments[0])
        worker._launch_buffered()
    except ACQUISITION_ERRORS as error:
        errors.append(error)

    staged = _stage_plans(worker, assignments[1:], errors)
    if not staged:
        return
    # Release checks only read fixed inputs. Durable ownership and the actual
    # CPU/RAM admission stay on the normal worker thread.
    with ThreadPoolExecutor(max_workers=min(parallelism, len(staged))) as pool:
        pending = {}
        for assignment in staged:
            check = dict(assignment)
            check.pop('artifact_blobs', None)
            pending[pool.submit(worker._validate_acquisition, check)] = assignment
        while pending:
            completed, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
            for proof in completed:
                assignment = pending.pop(proof)
                try:
                    proof.result()
                    worker._receive_assignment(assignment, validate=lambda _: proof.result())
                except ACQUISITION_ERRORS as error:
                    errors.append(error)
            if completed:
                worker._launch_buffered()
            worker._pump_collection()


def receive_assignments(worker, assignments, receive_original, parallelism):
    """Keep Q3, large Git inputs and durable replays on the existing path."""
    if not any(_plan_only(worker, item) for item in assignments):
        return receive_original(assignments)
    # Match the existing fresh-before-replay order and preserve the controller's
    # ordering within each class. Read-only checks may complete out of order.
    ordered = sorted(assignments, key=lambda item: _attempt(worker, item) in worker.records)
    groups = [(plans, list(group)) for plans, group in groupby(ordered, lambda item: _plan_only(worker, item))]
    if not any(plans and len(group) > 1 for plans, group in groups):
        return receive_original(assignments)
    errors = []
    for plans, group in groups:
        try:
            if plans and len(group) > 1:
                _receive_plans(worker, group, errors, parallelism)
            else:
                receive_original(group)
        except ACQUISITION_ERRORS as error:
            errors.append(error)
    if errors:
        raise ValueError(str(errors[0])) from errors[0]


class BatchPreparationAcquisition:
    """Add preparation acquisition while retaining the worker's Q3 behavior."""
    def __init__(self, *args, **kwargs):
        self._artifact_rpc_locks = [threading.Lock() for _ in range(64)]
        super().__init__(*args, **kwargs)

    def _post(self, endpoint, payload):
        # Readiness and acquisition may ask for the same objects at once.
        # Serialize response consumption for that request only.
        if endpoint in ('/v1/artifacts/get', '/v1/artifacts/missing', '/v1/artifacts/put'):
            hashes = sorted(payload.get('hashes', payload.get('blobs', {})))
            lock = self._artifact_rpc_locks[int(digest([endpoint, hashes])[:8], 16) % len(self._artifact_rpc_locks)]
            with lock:
                return super()._post(endpoint, payload)
        return super()._post(endpoint, payload)

    def _receive_assignments(self, assignments):
        return receive_assignments(self, assignments, super()._receive_assignments,
                                   self.preparation_acquisition_workers)
