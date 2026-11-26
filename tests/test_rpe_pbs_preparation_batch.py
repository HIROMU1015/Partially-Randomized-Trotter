"""Synthetic mixed-offer and replay tests for the PBS acquisition path."""
import base64
import hashlib
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trottertracks import rpe_pbs_preparation_batch as batch
from trottertracks.rpe_preparation_artifacts import worker_cache
from trottertracks.rpe_q3_worker import digest


def preparation(name, plan):
    data = plan.encode()
    sha = hashlib.sha256(data).hexdigest()
    request = {'route': {'kind': 'preparation'}, 'preparation': {'provider': 'HCHAIN_SCOPED', 'inputs': {'PLAN.json': sha}}}
    return (dict(job_id=name, attempt_id='fixed', request=request,
                 request_sha256=digest(request), artifact_manifest={'PLAN.json': sha}),
            (sha, base64.b64encode(data).decode()))


def test_mixed_offer_launches_priority_then_checks_and_receives_plans(tmp_path):
    first, first_blob = preparation('first', 'cached plan')
    second, shared = preparation('second', 'shared plan')
    third, _ = preparation('third', 'shared plan')
    fourth, distinct = preparation('fourth', 'distinct plan')
    large, _ = preparation('full-bundle', 'large route')
    large['request']['preparation']['input_git_bundles'] = [{}]
    replay, _ = preparation('replay', 'own existing plan')
    attempt = tmp_path/'replay'/'fixed'
    attempt.mkdir(parents=True)
    (attempt/'ASSIGNMENT.json').touch()
    q3 = dict(job_id='q3', attempt_id='fixed', request={'route': {'kind': 'q3'}})
    worker = SimpleNamespace(work=tmp_path, profile={'work_dir': str(tmp_path)}, records={attempt: replay})
    cache = worker_cache(worker.profile)
    cache.put(dict([first_blob]))
    worker._post = Mock(return_value={'blobs': dict([shared, distinct])})
    order, launched = [], []
    worker._launch_buffered = lambda: launched.append(tuple(order))
    worker._pump_collection = Mock()
    barrier = threading.Barrier(3)
    main_thread = threading.get_ident()

    def validate(assignment):
        assert launched[0] == ('q3', 'first')
        assert not cache.missing(assignment['artifact_manifest'].values())
        assert threading.get_ident() != main_thread
        barrier.wait(timeout=5)

    def receive(assignment, *, validate=None):
        assert threading.get_ident() == main_thread
        if validate:
            validate(assignment)
        order.append(assignment['job_id'])

    worker._validate_acquisition = validate
    worker._receive_assignment = receive
    batch.receive_assignments(worker, [replay, q3, first, second, third, fourth, large],
                              lambda group: [receive(item) for item in group], 4)
    worker._post.assert_called_once()
    endpoint, payload = worker._post.call_args.args
    assert endpoint == '/v1/artifacts/get'
    assert set(payload['hashes']) == {shared[0], distinct[0]}
    assert order[:2] == ['q3', 'first']
    assert set(order[2:5]) == {'second', 'third', 'fourth'}
    assert order[5:] == ['full-bundle', 'replay']


def test_missing_and_invalid_plan_leave_other_assignments_recoverable(tmp_path):
    first, _ = preparation('first', 'bad check')
    missing, _ = preparation('missing', 'missing')
    good, blob = preparation('good', 'cached')
    q3 = dict(job_id='q3', attempt_id='fixed', request={})
    worker = SimpleNamespace(work=tmp_path, profile={'work_dir': str(tmp_path)}, records={},
                             _launch_buffered=Mock(), _pump_collection=Mock(), _validate_acquisition=Mock())
    worker_cache(worker.profile).put(dict([blob]))
    worker._post = Mock(side_effect=TimeoutError('temporary transport failure'))
    received = []

    def receive(assignment, *, validate=None):
        if assignment is first:
            raise ValueError('release check failed')
        received.append(assignment['job_id'])

    worker._receive_assignment = receive
    with pytest.raises(ValueError, match='release check failed'):
        batch.receive_assignments(worker, [first, missing, good, q3],
                                  lambda group: [receive(item) for item in group], 4)
    assert received == ['good', 'q3']
    assert worker._validate_acquisition.call_count == 1
    assert worker._validate_acquisition.call_args.args[0]['attempt_id'] == 'fixed'
