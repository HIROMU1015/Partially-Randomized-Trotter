"""One-shot saved-value validation; default and unapproved grants do no analysis."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_signed_bias_contract as contract
import h8_signed_bias as numeric


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def execute(plan, grant):
    permit, input_root, output = contract.authorize(plan, grant)
    # No source/input mutation, children, affinity, compiler, or scientific imports.
    lock = output.with_name(output.name + '.one_shot_lock')
    write_new(lock, dict(scope=contract.SCOPE, plan_sha256=contract.digest(plan),
                         grant_sha256=contract.digest(grant), retry=False))
    output.mkdir(exist_ok=False)
    started, first_stop, interrupt_reason = time.monotonic(), None, None
    result = None
    previous_handlers = {}

    def guard():
        if interrupt_reason is not None:
            raise InterruptedError(interrupt_reason)

    def on_signal(signum, frame):
        nonlocal first_stop, interrupt_reason
        interrupt_reason = 'OWN_PROCESS_SIGNAL:' + str(signum)
        if first_stop is None:
            first_stop = interrupt_reason
        raise InterruptedError(interrupt_reason)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, on_signal)
        dataset, registry, receipts = contract.load_saved(plan['input_manifest'], input_root, permit)
        result = numeric.validate_domain(dataset, registry, permit, guard)
        first_stop = result['first_stop']
        # Detect post-read changes without recomputing or parsing science twice.
        for receipt in receipts:
            contract.read_verified(Path(receipt['path']), receipt)
        for receipt in plan['source_files']:
            contract.read_verified(contract.ROOT/receipt['path'], receipt)
        result.update(SOURCE=plan['SOURCE'], plan_sha256=contract.digest(plan),
                      grant_sha256=contract.digest(grant), input_receipts=receipts,
                      environment=contract.environment(), wall_seconds=time.monotonic()-started,
                      post_execution_binding_verified=True)
        guard()
        write_new(output/'axis_results.json', result)
        terminal = {k: result[k] for k in ('status', 'first_stop', 'SOURCE', 'candidate_count',
                    'axis_count', 'failed_axes', 'wall_seconds', 'next_stage_authorized', 'mandatory_STOP')}
    except (Exception, KeyboardInterrupt) as exc:
        if first_stop is None:
            first_stop = type(exc).__name__ + ':' + str(exc)
        terminal = dict(status='STOP', first_stop=first_stop, SOURCE=plan['SOURCE'],
                        wall_seconds=time.monotonic()-started, next_stage_authorized=False,
                        automatic_retry=False, mandatory_STOP=True)
        if result is not None and not (output/'axis_results.json').exists():
            result.update(status='STOP', first_stop=first_stop, post_execution_binding_verified=False)
            write_new(output/'axis_results.json', result)
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
    terminal.update(child_processes_started=0, all_children_reaped=True, lock_retained=True)
    write_new(output/'terminal.json', terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    if not args.execute_approved:
        if args.plan or args.grant:
            parser.error('--execute-approved required; default reads no saved data')
        print(json.dumps(dict(scope=contract.SCOPE, approved=False, runtime_authorization=False,
                              saved_H8_numerical_analysis=0, next_stage_authorized=False)))
        return 0
    if args.plan is None or args.grant is None:
        parser.error('fixed plan and separately approved grant required')
    try:
        plan = contract.strict(contract.home_path(args.plan).read_bytes())
        grant = contract.strict(contract.home_path(args.grant).read_bytes())
        terminal = execute(plan, grant)
        print(json.dumps(terminal))
        return 0 if terminal['status'] == 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY' else 1
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps(dict(status='STOP_NO_SUCCESS_RECEIPT',
                         first_stop=type(exc).__name__+':'+str(exc), next_stage_authorized=False)))
        return 2


if __name__ == '__main__':
    sys.exit(main())
