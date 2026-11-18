"""Operational Q3 timing. Never subtract absolute clocks on different hosts.

Clock IDs are boot-scoped on Linux (shared by its forked numerical workers),
process-scoped elsewhere. Legacy controller logs are paired only within their
recorded invocation. This module makes no scientific result/price decisions.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import statistics
import time
import uuid

try:
    _CLOCK = socket.gethostname() + ':' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()
except OSError:
    _CLOCK = socket.gethostname() + ':process:' + str(os.getpid()) + ':' + uuid.uuid4().hex


def stamp():
    return dict(clock_id=_CLOCK, monotonic=time.monotonic(), epoch=time.time())


def append(path, event, **fields):
    """Append small metadata; never serialize numerical DP results."""
    row = dict(stamp(), event=event, pid=os.getpid(), **fields)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return row


@contextmanager
def body_context(module, output):
    """Observe the existing Accelerator API, preserving its return/exception.

    The original __call__, kernel AST guard, numerical inputs and cache context
    remain unchanged. The calling worker's fork inherits this wrapper.
    """
    from unittest.mock import patch
    if module is None or not hasattr(module, 'Accelerator'):
        append(Path(output)/'Q3_BODY_TIMING.jsonl', 'BODY_TIMING_UNAVAILABLE', reason='NO_SUPPORTED_ACCELERATOR')
        yield
        return
    original = module.Accelerator.__call__
    def observed(self, *args, **kwargs):
        call_id = uuid.uuid4().hex
        append(Path(output)/'Q3_BODY_TIMING.jsonl', 'Q3_BODY_START', call_id=call_id)
        try:
            result = original(self, *args, **kwargs)
        except BaseException:
            append(Path(output)/'Q3_BODY_TIMING.jsonl', 'Q3_BODY_END', call_id=call_id, completed=False)
            raise
        append(Path(output)/'Q3_BODY_TIMING.jsonl', 'Q3_BODY_END', call_id=call_id, completed=True)
        return result
    with patch.object(module.Accelerator, '__call__', observed):
        yield


def read_lines(path):
    rows = []
    if not Path(path).exists():
        return rows
    lines=Path(path).read_text().splitlines()
    for number, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live final append can be incomplete; never guess its timestamp.
            if number != len(lines):
                raise
    return rows


def delta(start, end, legacy=False):
    if not start or not end or 'monotonic' not in start or 'monotonic' not in end:
        return None
    same = start.get('clock_id') and start.get('clock_id') == end.get('clock_id')
    same = same or (legacy and start.get('invocation') and start.get('invocation') == end.get('invocation'))
    if not same:
        return None
    value = end['monotonic'] - start['monotonic']
    return value if value >= 0 else None


def q3_body_spans(directory):
    body = read_lines(directory/'Q3_BODY_TIMING.jsonl')
    if not body:
        labels = {'ARTIFICIAL_EVALUATION_START': 'Q3_BODY_START', 'ARTIFICIAL_EVALUATION_RETURNED': 'Q3_BODY_END'}
        body = [dict(row, event=labels[row['stage']], call_id='artificial')
                for row in read_lines(directory/'JOB_STAGES.jsonl') if row.get('stage') in labels]
    starts = {row['call_id']: row for row in body if row.get('event') == 'Q3_BODY_START' and 'call_id' in row}
    ends = {row['call_id']: row for row in body if row.get('event') == 'Q3_BODY_END'
            and 'call_id' in row and row.get('completed', True)}
    pairs = [(start, ends[key]) for key, start in starts.items() if key in ends and delta(start, ends[key]) is not None]
    if not pairs or len({start['clock_id'] for start, _ in pairs}) != 1:
        return None, None, []
    return (min((start for start, _ in pairs), key=lambda row: row['monotonic']),
            max((end for _, end in pairs), key=lambda row: row['monotonic']),
            [delta(start, end) for start, end in pairs])


def union_seconds(intervals):
    total = 0.
    for domain in {d for d, _, _ in intervals}:
        values = sorted((a, b) for d, a, b in intervals if d == domain and b >= a)
        if not values:
            continue
        start, end = values[0]
        for a, b in values[1:]:
            if a <= end:
                end = max(end, b)
            else:
                total += end-start; start, end = a, b
        total += end-start
    return total


def stats(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values) if values else None,
                maximum=max(values) if values else None, minimum=min(values) if values else None)


def preparation_summary(run, events):
    """Typed preparation timing, kept separate from Q3 counts and body spans."""
    launches={}
    for event in events:
        if event.get('event')=='LAUNCH'and event.get('attempt_id'):
            key=(event.get('job')or event.get('job_id'),event['attempt_id'],event['host_id'])
            launches.setdefault(key,event)
    by_job=defaultdict(list)
    for key in launches:by_job[key[0]].append(key)
    grouped=defaultdict(lambda:defaultdict(list));ambiguous=[]
    for event in events:
        job=event.get('job')or event.get('job_id');attempt=event.get('attempt_id')
        choices=[key for key in by_job.get(job,[])if attempt is None or key[1]==attempt]
        if len(choices)!=1:
            if event.get('event')in('ADOPT','RESULT_RECEIVED','TRANSFER_COMPLETE'):
                ambiguous.append(dict(job_id=job,event=event.get('event'),reason='NO_UNIQUE_PREPARATION_ATTEMPT'))
            continue
        grouped[choices[0]][event.get('event')].append(event)
    rows=[];queue_intervals=[]
    for key,launch in sorted(launches.items(),key=lambda item:item[1].get('monotonic',0)):
        job,attempt,host=key;stage=launch.get('stage')or str(launch.get('method','')).split(':',1)[-1]
        ev=grouped[key];first=lambda name:next(iter(ev.get(name,[])),None)
        directory=Path(run)/'jobs'/str(launch.get('method'))/job
        marker=directory/'LAUNCH.json';artifact_ok=None
        if marker.exists():
            saved=json.loads(marker.read_text());artifact_ok=saved.get('attempt_id')==attempt and saved.get('host_id')==host
        stages=read_lines(directory/'JOB_STAGES.jsonl')if artifact_ok is True else[]
        starts=[event for event in stages if event.get('event')=='PREPARATION_BODY_START'and event.get('stage')==stage]
        ends=[event for event in stages if event.get('event')=='PREPARATION_BODY_END'and event.get('stage')==stage]
        pairs=[(start,end)for start,end in zip(starts,ends)if delta(start,end)is not None]
        body=stats([delta(start,end)for start,end in pairs])['total']if pairs else None
        received=first('RESULT_RECEIVED');persisted=first('TRANSFER_COMPLETE');adopt=first('ADOPT');end=received or persisted
        q=launch.get('queue_wait_seconds');domain=launch.get('clock_id')or launch.get('invocation')
        if q is not None:queue_intervals.append((domain,launch['monotonic']-q,launch['monotonic']))
        def transfer_metric(name):return next((event[name]for event in ev.get('TRANSFER_TIMING',[])if event.get(name)is not None),None)
        rows.append(dict(job_id=job,attempt_id=attempt,host=host,stage=stage,method=launch.get('method'),
            artifact_attempt_matches=artifact_ok,A_queue_to_assignment_seconds=q,
            A_start_scope='PREPARATION_TASK_ENQUEUED; declared dependency wait may be included before inputs are ready',
            preparation_body_seconds=body,preparation_body_calls_measured=len(pairs),
            preparation_body_scope='PREPARATION_COMMAND_PROCESS_SPAN; separate from the Q3 accelerator API',
            assignment_to_controller_receive_seconds=delta(launch,end,True),
            D_receive_to_executor_adopt_seconds=delta(end,adopt,True),
            receive_validation_persistence_seconds=(persisted or{}).get('transfer_seconds'),
            receive_boundary='RPC_BODY_RECEIVED'if received else'LEGACY_VALIDATED_PERSISTED_RESULT'if persisted else'MISSING',
            assignment_to_body_same_clock_seconds=delta(launch,pairs[0][0])if pairs else None,
            body_to_controller_receive_same_clock_seconds=delta(pairs[-1][1],received)if pairs else None,
            worker_upload_ACK_roundtrip_seconds=transfer_metric('upload_roundtrip_seconds'),
            SSH_delivery_roundtrip_seconds=transfer_metric('ssh_delivery_roundtrip_seconds'),
            SSH_status_download_roundtrip_seconds=transfer_metric('ssh_status_download_seconds')))
    keys=('A_queue_to_assignment_seconds','preparation_body_seconds','assignment_to_controller_receive_seconds',
          'D_receive_to_executor_adopt_seconds','receive_validation_persistence_seconds','worker_upload_ACK_roundtrip_seconds',
          'SSH_delivery_roundtrip_seconds','SSH_status_download_roundtrip_seconds')
    def aggregate(selected):return dict(count=len(selected),metrics={key:stats([row[key]for row in selected])for key in keys})
    return dict(schema='RPE_PREPARATION_OPERATIONAL_TIMING_V1',attempts=len(rows),rows=rows,
        by_stage={stage:aggregate([row for row in rows if row['stage']==stage])for stage in sorted({row['stage']for row in rows})},
        by_host={host:dict(aggregate([row for row in rows if row['host']==host]),
            by_stage={stage:aggregate([row for row in rows if row['host']==host and row['stage']==stage])
                      for stage in sorted({row['stage']for row in rows if row['host']==host})})for host in sorted({row['host']for row in rows})},
        missing_body_count=sum(row['preparation_body_seconds']is None for row in rows),ambiguous_joins=ambiguous,
        queue_wait_union_seconds=union_seconds(queue_intervals),
        clock_scope='Same-clock deltas only; absolute timestamps from separate hosts are never subtracted',
        scientific_H4_calls_added=0)


def summarize(run, *, timeline=None, public_summary=None):
    """One normal run directory, or a historical published timeline.

    All missing/ambiguous joins stay null. Attempt IDs on old events are filled
    only when that job has exactly one LAUNCH attempt. SEARCH_ADOPT's old
    configuration-only event is linked to the unique pending ADOPT for that
    method/configuration and explicitly labelled as a legacy inference.
    """
    run = Path(run)
    all_events = read_lines(timeline or run/'PROGRESS.jsonl')
    preparation_jobs={event.get('job')or event.get('job_id')for event in all_events
                      if str(event.get('method','')).startswith('PREPARATION:')}
    preparation_jobs.discard(None)
    preparation_events=[event for event in all_events if (event.get('job')or event.get('job_id'))in preparation_jobs
                        or str(event.get('method','')).startswith('PREPARATION:')]
    preparation=preparation_summary(run,preparation_events)
    events=[]
    for event in all_events:
        if (event.get('job')or event.get('job_id'))in preparation_jobs or str(event.get('method','')).startswith('PREPARATION:'):continue
        if event.get('event')=='QUEUE_WAIT_STATE':
            event=dict(event,reasons={job:reason for job,reason in event.get('reasons',{}).items()if job not in preparation_jobs})
        events.append(event)
    launches = defaultdict(list)
    candidate_ready = {}
    first_candidate_assignments = {}
    for e in events:
        candidate = (e.get('invocation'), e.get('method'), e.get('configuration'))
        if e.get('event') == 'CANDIDATE_READY':
            candidate_ready.setdefault(candidate, e)
        if e.get('event') == 'LAUNCH' and e.get('attempt_id'):
            launches[e['job']].append(e)
            if candidate in candidate_ready:
                first_candidate_assignments.setdefault(candidate, e)
    attempts = {}; ambiguous = []; waiting = defaultdict(list)
    wait_intervals = []; pending_adopt = defaultdict(list)
    for e in events:
        name = e.get('event'); job = e.get('job') or e.get('job_id')
        if name == 'DEPENDENCY_WAIT_START':
            waiting[(e.get('invocation'), job)].append(e)
        elif name == 'DEPENDENCY_WAIT_END':
            queue = waiting[(e.get('invocation'), job)]
            if queue:
                start = queue.pop(0)
                if delta(start, e, True) is not None:
                    wait_intervals.append((start.get('clock_id') or start.get('invocation'), start['monotonic'], e['monotonic'], job))
        if name == 'SEARCH_ADOPT' and not job:
            key = (e.get('invocation'), e.get('method'))
            matched = [a for a in pending_adopt[key] if a.get('configuration', '').startswith(e.get('configuration', '!'))]
            if len(matched) == 1:
                job = matched[0]['job']; pending_adopt[key].remove(matched[0])
                e = dict(e, job=job, join='LEGACY_UNIQUE_PENDING_ADOPT_CONFIGURATION')
            else:
                ambiguous.append(dict(event=name, reason='NO_UNIQUE_JOB', monotonic=e.get('monotonic')))
        if not job:
            continue
        choices = {x['attempt_id']: x for x in launches.get(job, [])}
        attempt = e.get('attempt_id')
        if not attempt and len(choices) == 1:
            attempt = next(iter(choices))
        if not attempt:
            if name in ('ADOPT', 'SEARCH_ADOPT', 'TRANSFER_COMPLETE'):
                ambiguous.append(dict(job_id=job, event=name, reason='NO_UNIQUE_ATTEMPT'))
            continue
        launch = choices.get(attempt)
        if not launch:
            continue
        key = (job, attempt, launch['host_id'])
        record = attempts.setdefault(key, dict(launch=launch, events=defaultdict(list)))
        record['events'][name].append(e)
        if name == 'ADOPT':
            pending_adopt[(e.get('invocation'), e.get('method'))].append(dict(e, job=job))
    first_dispatch=min((e['monotonic']for values in launches.values()for e in values),default=None)
    rows=[]; receive_adopt_intervals=[]; critical_intervals=[]; queue_intervals=[]
    first_by_host={}
    for (job, attempt, host), r in sorted(attempts.items(), key=lambda item:item[1]['launch'].get('monotonic', 0)):
        launch=r['launch'];ev=r['events']
        first=lambda name: next(iter(ev.get(name, [])), None)
        received=first('RESULT_RECEIVED');persisted=first('TRANSFER_COMPLETE');adopt=first('ADOPT');search=first('SEARCH_ADOPT')
        end=received or persisted
        directory=run/'jobs'/str(launch.get('method'))/job
        assignment=directory/'LAUNCH.json'
        artifact_ok=None
        if assignment.exists():
            saved=json.loads(assignment.read_text())
            artifact_ok=saved.get('attempt_id')==attempt and saved.get('host_id')==host
        bs, be, spans = q3_body_spans(directory) if artifact_ok is True else (None, None, [])
        body_seconds=delta(bs,be)
        transport=read_lines(directory/'TRANSPORT_TIMING.jsonl') if artifact_ok is True else []
        def transport_event(name,mode=None):return next((x for x in transport if x.get('event')==name and (mode is None or x.get('mode')==mode)),None)
        def transfer_metric(key):return next((x[key]for x in ev.get('TRANSFER_TIMING',[])if x.get(key) is not None),None)
        roundtrip=delta(launch,end,True)
        b=delta(launch,bs);c=delta(be,received)
        overhead=None if roundtrip is None or body_seconds is None else roundtrip-body_seconds
        if overhead is not None and overhead<0:overhead=None
        d=delta(end,adopt,True); ds=delta(end,search,True)
        q=launch.get('queue_wait_seconds');domain=launch.get('clock_id') or launch.get('invocation')
        if q is not None:queue_intervals.append((domain,launch['monotonic']-q,launch['monotonic']))
        if ds is not None:
            receive_adopt_intervals.append((domain,end['monotonic'],search['monotonic']))
            for wd,wa,wb,wjob in wait_intervals:
                a,z=max(wa,end['monotonic']),min(wb,search['monotonic'])
                if wd==domain and wjob==job and z>=a:critical_intervals.append((domain,a,z))
        first_host=host not in first_by_host;first_by_host.setdefault(host,job)
        def bounded(exact):
            if exact is not None:return dict(kind='EXACT_SAME_CLOCK',seconds=exact)
            return dict(kind='BOUNDED' if overhead is not None else 'MISSING',seconds=None,lower_seconds=0 if overhead is not None else None,
                        upper_seconds=overhead,reason='Absolute host clocks not subtracted; B+C shared upper envelope' if overhead is not None else 'Q3 body or controller receive boundary missing')
        candidate = (launch.get('invocation'), launch.get('method'), launch.get('configuration'))
        ready = candidate_ready.get(candidate) if first_candidate_assignments.get(candidate) is launch else None
        prefix_events = read_lines(directory/'DP_PREFIX_EVENTS.jsonl') if artifact_ok is True else []
        row=dict(job_id=job,attempt_id=attempt,host=host,method=launch.get('method'),
            host_phase='FIRST_ASSIGNMENT_TO_HOST' if first_host else 'FOLLOWUP_ASSIGNMENT',
            queue_phase='PRE_FIRST_DISPATCH_ENQUEUE' if q is not None and first_dispatch is not None and launch['monotonic']-q<first_dispatch else 'AFTER_DISPATCH_ENQUEUE',
            artifact_attempt_matches=artifact_ok,
            A_candidate_ready_to_first_assignment_seconds=delta(ready,launch),
            dp_prefix_hits=sum(e.get('event')=='HIT' for e in prefix_events) if prefix_events else None,
            A_queue_to_assignment_seconds=q,A_start_scope='OPTIMIZER_ALLOCATION_ENQUEUED; not physical candidate preparation READY',
            B_assignment_to_Q3_body=bounded(b),Q3_body_seconds=body_seconds,
            Q3_body_calls_measured=len(spans),Q3_body_scope='EXISTING_ACCELERATOR_API_CALL_SPAN_OR_LABELLED_ARTIFICIAL_BODY',
            Q3_API_call_seconds=spans,C_Q3_body_to_controller_receive=bounded(c),
            B_plus_C_non_body_envelope_seconds=overhead,assignment_to_controller_receive_seconds=roundtrip,
            receive_boundary='RPC_BODY_RECEIVED' if received else 'LEGACY_VALIDATED_PERSISTED_RESULT' if persisted else 'MISSING',
            D_receive_to_executor_adopt_seconds=d,D_receive_to_search_adopt_seconds=ds,
            search_join=(search or {}).get('join','EXPLICIT_JOB_ATTEMPT' if search else 'MISSING'),
            receive_validation_persistence_seconds=(persisted or {}).get('transfer_seconds'),
            worker_assignment_durable_to_body_seconds=delta(transport_event('WORKER_ASSIGNMENT_DURABLE'),bs),
            body_to_worker_bundle_ready_seconds=delta(be,transport_event('RESULT_BUNDLE_READY')),
            worker_adapter_execute_process_seconds=delta(transport_event('ADAPTER_PROCESS_START','execute'),transport_event('ADAPTER_PROCESS_END','execute')),
            worker_upload_ACK_roundtrip_seconds=transfer_metric('upload_roundtrip_seconds'),
            SSH_status_download_roundtrip_seconds=transfer_metric('ssh_status_download_seconds'),
            SSH_delivery_roundtrip_seconds=transfer_metric('ssh_delivery_roundtrip_seconds'),
            relay_response_to_submit_seconds=transfer_metric('relay_response_to_submit_seconds'),
            SSH_ACK_roundtrip_seconds=transfer_metric('ssh_ack_roundtrip_seconds'))
        rows.append(row)
    by_host={}
    for host in sorted({r['host'] for r in rows}):
        selected=[r for r in rows if r['host']==host]
        keys=['A_queue_to_assignment_seconds','A_candidate_ready_to_first_assignment_seconds','Q3_body_seconds','assignment_to_controller_receive_seconds',
              'D_receive_to_executor_adopt_seconds','D_receive_to_search_adopt_seconds','receive_validation_persistence_seconds',
              'worker_upload_ACK_roundtrip_seconds','SSH_status_download_roundtrip_seconds',
              'worker_assignment_durable_to_body_seconds','body_to_worker_bundle_ready_seconds','worker_adapter_execute_process_seconds',
              'SSH_delivery_roundtrip_seconds','relay_response_to_submit_seconds','SSH_ACK_roundtrip_seconds']
        by_host[host]=dict(count=len(selected),dp_prefix_hits=sum(r['dp_prefix_hits'] or 0 for r in selected),
            prefix_event_coverage=sum(r['dp_prefix_hits'] is not None for r in selected),
            metrics={k:stats([r[k] for r in selected]) for k in keys},
            first_assignment=[r for r in selected if r['host_phase']=='FIRST_ASSIGNMENT_TO_HOST'],
            initial_queue_metrics={k:stats([r[k] for r in selected if r['queue_phase']=='PRE_FIRST_DISPATCH_ENQUEUE'])for k in keys},
            followup_metrics={k:stats([r[k] for r in selected if r['queue_phase']=='AFTER_DISPATCH_ENQUEUE']) for k in keys},
            per_job=selected)
    # Slot observations are generated under the same controller lock as claim.
    # They classify recorded queue intervals; unobserved time stays unknown.
    slots=[e for e in events if e.get('event')=='QUEUE_WAIT_STATE']
    slot_intervals=defaultdict(list);slot_job_intervals=defaultdict(list)
    for i,e in enumerate(slots):
        stop=next((x for x in slots[i+1:] if x.get('invocation')==e.get('invocation')),None)
        if stop and delta(e,stop,True) is not None:
            for job,reason in e.get('reasons',{}).items():
                terminal=next((x for x in launches.get(job,[]) if x.get('invocation')==e.get('invocation') and e['monotonic']<=x['monotonic']<=stop['monotonic']),None)
                z=terminal['monotonic'] if terminal else stop['monotonic']
                span=(e.get('clock_id') or e['invocation'],e['monotonic'],z)
                slot_intervals[reason].append(span);slot_job_intervals[(job,reason)].append(span)
    for row in rows:
        parts={reason:union_seconds(v)for (job,reason),v in slot_job_intervals.items()if job==row['job_id']}
        row['A_observed_wait_seconds_by_reason']=parts if slots else None
        q=row['A_queue_to_assignment_seconds']
        row['A_unattributed_seconds']=max(0,q-sum(parts.values()))if slots and q is not None else q
    return dict(schema='RPE_Q3_OPERATIONAL_TIMING_V1',scope='OPERATIONAL_LOGS_ONLY_NO_SCIENTIFIC_REANALYSIS',
        run_directory=str(run),input_timeline=str(timeline or run/'PROGRESS.jsonl'),attempts=len(rows),by_host=by_host,rows=rows,
        preparation=preparation,attempt_count_scope='Q3_ONLY; typed preparation attempts are reported separately',
        missing_counts={k:sum(r[k] is None for r in rows) for k in ['Q3_body_seconds','D_receive_to_search_adopt_seconds']},
        ambiguous_joins=ambiguous,queue_wait_job_seconds_sum=sum(r['A_queue_to_assignment_seconds'] or 0 for r in rows),
        queue_wait_union_seconds=union_seconds(queue_intervals),
        D_receive_to_search_job_seconds_sum=sum(r['D_receive_to_search_adopt_seconds'] or 0 for r in rows),
        D_receive_to_search_union_seconds=union_seconds(receive_adopt_intervals),
        D_while_optimizer_waiting_union_seconds=union_seconds(critical_intervals) if wait_intervals else None,
        D_critical_scope='Intersection with recorded demanded-job DEPENDENCY_WAIT; not sum of speculative job latency',
        total_demanded_dependency_wait_union_seconds=union_seconds([(d,a,z)for d,a,z,_ in wait_intervals]) if wait_intervals else None,
        dependency_wait_intervals_measured=len(wait_intervals),
        pre_first_dispatch_queue_seconds=None if first_dispatch is None or not queue_intervals else max(0,first_dispatch-min(a for d,a,z in queue_intervals)),
        queue_wait_state_observations=len(slots),queue_wait_observed_union_by_reason={k:union_seconds(v) for k,v in slot_intervals.items()},
        idle_slot_observations=dict(Counter(observation['idle_reason'] for slot in slots
            for observation in slot.get('hosts',{}).values() if 'idle_reason' in observation)),
        queue_wait_state_scope='Observed controller reservation/eligibility state; no attribution if snapshots absent; first host assignment is not a proof of setup completion',
        published_summary_reference=public_summary,
        scientific_H4_calls_added=0)
