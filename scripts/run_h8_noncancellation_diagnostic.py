"""One-shot noncancellation diagnostic; default and draft grants perform no analysis."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_noncancellation_contract as contract
import h8_noncancellation as numeric


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=1, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def execute(plan, grant):
    permit, input_root, output, b2_root = contract.authorize(plan, grant)
    stages = tuple(grant['authorized_stages'])
    # No source/input mutation, children, affinity, compiler or scientific imports.
    lock = output.with_name(output.name + '.one_shot_lock')
    write_new(lock, dict(scope=contract.SCOPE, plan_sha256=contract.digest(plan),
                         grant_sha256=contract.digest(grant), retry=False))
    output.mkdir(exist_ok=False)
    started, first_stop, interrupt_reason = time.monotonic(), None, None
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
        cost = 'COST_COMPARISON' in stages
        dataset, registry, receipts = contract.load_saved(
            plan['input_manifest'], input_root, permit,
            b2_dependency=plan['B2_dependency'] if cost else None, b2_root=b2_root)
        result = numeric.evaluate(dataset, registry, permit, stages=stages, guard=guard)
        # Detect post-read changes without parsing science twice.
        for receipt in receipts:
            contract.read_verified(Path(receipt['path']), receipt)
        for receipt in plan['source_files']:
            contract.read_verified(contract.ROOT/receipt['path'], receipt)
        result.update(SOURCE=plan['SOURCE'], plan_sha256=contract.digest(plan),
                      grant_sha256=contract.digest(grant), input_receipts=receipts,
                      environment=contract.environment(), wall_seconds=time.monotonic()-started,
                      post_execution_binding_verified=True)
        guard()
        write_new(output/'diagnostic_result.json', result)
        terminal = dict(status=result['status'], first_stop=None, SOURCE=plan['SOURCE'],
                        stages_executed=result['stages_executed'], cost_stage=result['cost_stage']['status'],
                        unresolved_axes=result['unresolved_axes'], wall_seconds=result['wall_seconds'],
                        next_stage_authorized=False, automatic_retry=False, mandatory_STOP=True)
    except (Exception, KeyboardInterrupt) as exc:
        if first_stop is None:
            first_stop = type(exc).__name__ + ':' + str(exc)
        terminal = dict(status='STOP', first_stop=first_stop, SOURCE=plan['SOURCE'],
                        wall_seconds=time.monotonic()-started, next_stage_authorized=False,
                        automatic_retry=False, mandatory_STOP=True, partial_result_claimed=False)
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
                              saved_H8_numerical_analysis=0, stages=list(numeric.STAGES),
                              cost_stage_requires_B2_full_PASS_binding=True, next_stage_authorized=False)))
        return 0
    if args.plan is None or args.grant is None:
        parser.error('fixed plan and separately approved grant required')
    try:
        plan = contract.strict(contract.home_path(args.plan).read_bytes())
        grant = contract.strict(contract.home_path(args.grant).read_bytes())
        terminal = execute(plan, grant)
        print(json.dumps(terminal))
        return 0 if terminal['status'].startswith('COMPLETE_') else 1
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps(dict(status='STOP_NO_SUCCESS_RECEIPT',
                              first_stop=type(exc).__name__ + ':' + str(exc), next_stage_authorized=False)))
        return 2


if __name__ == '__main__':
    sys.exit(main())
