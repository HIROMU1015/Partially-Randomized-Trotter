"""Future isolated one-shot adapter; no kernel or numerical import at module load.

No authorization/marker is issued in preparation. This module is dispatched only
after the control guard binds a separately approved future authority and an
adopted validation contract. Any reserved output remains consumed on failure.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import signal
import sys
import time

from .control import require_execution, POLICY, INPUT_PATH


def _write_exclusive(path, raw, cap):
    if len(raw) > cap:
        raise ValueError('UNDETERMINED: serialized output cap')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False).encode() + b'\n'


def run_authorized(root, permit):
    require_execution(permit)
    root = Path(root).resolve()
    limits = permit.config['resources']
    for kind, requested in ((resource.RLIMIT_AS, limits['memory_bytes']),
                            (resource.RLIMIT_FSIZE, limits['output_bytes'])):
        soft, hard = resource.getrlimit(kind)
        cap = min(requested, hard) if hard != resource.RLIM_INFINITY else requested
        cap = min(cap, soft) if soft != resource.RLIM_INFINITY else cap
        resource.setrlimit(kind, (cap, hard))
    output = root / permit.audit['output_directory']
    output.mkdir(parents=True, exist_ok=False)
    # The directory itself consumes this approved attempt; never delete or retry.
    context = {'audit':permit.audit, 'config':permit.config,
               'attempt_consumed':True, 'automatic_retry':False}
    _write_exclusive(output/'execution_context.json',_json(context),limits['context_bytes'])
    log_fd = os.open(output/'runner.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    saved_fds = (os.dup(1), os.dup(2))
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log_fd,1)
    os.dup2(log_fd,2)
    os.close(log_fd)
    started = time.monotonic()
    reference = None
    report = {'schema':'m1s_gate0_independent_result_v1','status':'UNDETERMINED',
              'available_comparator_status':'UNDETERMINED', 'audit':permit.audit,
              'config':permit.config,'new_scientific_execution':True,
              'coefficient_and_metric_evidence':'EXACT_SAVED_BINARY64_ALGEBRA_IF_SOURCE_CORRECT_NOT_FORMALLY_VERIFIED',
              'v5_platform_coverage_certified':False,'scientific_GO_or_no_GO_determined':False,
              'posthoc_independent_reproducibility_check':True,'rows':[],
              'mandatory_STOP':True,'automatic_retry':False,
              'unperformed':['v5_reexecution','sampling','simulation','spectrum','circuits','compile','finite_step','resources','T1c','Q4','Q5']}
    previous_handlers = {}
    def interrupted(number, frame):
        raise InterruptedError('Independent verification interrupted by signal '+str(number))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig,interrupted)
        from .reference import Reference
        reference = Reference(permit)
        input_raw = (root/INPUT_PATH).read_bytes()
        independent = reference.build(input_raw)
        # Only after all independent quantities/hash are fixed is JSON parsed.
        comparator_raw = (root/POLICY['comparator']['path']).read_bytes()
        report.update(reference.compare_saved(independent,comparator_raw))
        frozen_raw = json.dumps(independent['public'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(frozen_raw).hexdigest() != independent['independent_reference_sha256']:
            raise ValueError('FAIL: independent reference changed during comparison')
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc)
        report['status'] = 'FAIL' if reason.startswith('FAIL:') else 'UNDETERMINED'
        report['failure'] = {'type':type(exc).__name__, 'reason':reason,
                             'category':'RESOURCE_LIMIT' if isinstance(exc,MemoryError) or 'cap' in reason else 'REFERENCE_OR_COMPARISON_STOP'}
    finally:
        for sig, old in previous_handlers.items():
            signal.signal(sig, old)
        if reference is not None:
            report['progress'] = reference.progress
            report['primitive_events'] = reference.events
            report['independent_reference_sha256'] = reference.progress.get('independent_reference_sha256')
        if not report['rows']:
            report['rows'] = list(reference.progress.get('comparisons',[])) if reference is not None else []
        completed = {r['split'] for r in report['rows']}
        report['rows'].extend({'split':s,'available_comparator_status':'UNDETERMINED',
                               'full_requested_scope_status':'UNDETERMINED','reason':'NOT_EVALUATED_AFTER_STOP'}
                              for s in ('G3','G6','G9') if s not in completed)
        report['resources'] = {'elapsed_seconds':time.monotonic()-started,
                               'maxrss_linux_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               'workers':1,'backend':'CPython_stdlib_exact','wall_time_cap':None,
                               'address_space_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],
                               'result_and_log_per_file_limit_bytes':resource.getrlimit(resource.RLIMIT_FSIZE)[0]}
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_fds[0],1)
        os.dup2(saved_fds[1],2)
        os.close(saved_fds[0])
        os.close(saved_fds[1])
    raw = _json(report)
    if len(raw) > limits['output_bytes']:
        # Preserve identity and completed status metadata, never truncate JSON.
        report = {k:report[k] for k in ('schema','audit','config','resources','mandatory_STOP','automatic_retry')}
        report.update(status='UNDETERMINED', failure={'type':'OUTPUT_LIMIT','oversize_payload_sha256':hashlib.sha256(raw).hexdigest()},
                      rows=[{'split':s,'available_comparator_status':'UNDETERMINED','reason':'OUTPUT_LIMIT'} for s in ('G3','G6','G9')])
        raw = _json(report)
    _write_exclusive(output/'result.json',raw,limits['output_bytes'])
    _write_exclusive(output/'result.sha256',(hashlib.sha256(raw).hexdigest()+'\n').encode(),128)
    return {'status':report['status'],'result_directory':str(output),'mandatory_STOP':True}
