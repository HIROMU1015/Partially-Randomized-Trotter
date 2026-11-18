#!/usr/bin/env python3
"""Pull independent Q3 jobs over the coordinator's loopback/SSH-forwarded API."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.rpe_q3_worker import Worker, read, supervise, mailbox
from trottertracks.rpe_worker_storage import report as storage_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', help='Coordinator URL reached locally or through existing SSH forwarding')
    parser.add_argument('--profile', type=Path, help='Host-local worker profile JSON')
    parser.add_argument('--poll-seconds', type=float, default=1.)
    parser.add_argument('--mailbox', choices=('status', 'submit', 'ack', 'result', 'readiness', 'resources', 'artifacts-missing', 'artifacts-get', 'artifacts-put'), help='One SSH mailbox RPC; input/output are JSON')
    parser.add_argument('--once', action='store_true', help='One acquisition poll, then finish/recover and transfer')
    parser.add_argument('--drain', action='store_true', help='Recover owned attempts; acquire no new jobs')
    parser.add_argument('--storage-limit-report', action='store_true', help='Read a storage policy from stdin without starting a worker')
    parser.add_argument('--supervise-attempt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise_attempt:
        return supervise(args.supervise_attempt)
    if args.storage_limit_report:
        print(json.dumps(storage_report(json.load(sys.stdin)), allow_nan=False), flush=True)
        return 0
    if args.mailbox:
        if not args.profile:
            parser.error('--profile is required for mailbox RPC')
        try:
            incoming = sys.stdin.read()
            response = mailbox(read(args.profile), args.mailbox, json.loads(incoming) if incoming.strip() else {})
            print(json.dumps(response, separators=(',', ':'), allow_nan=False), flush=True)
            return 0
        except Exception as error:
            print(json.dumps(dict(ok=False, error=type(error).__name__+': '+str(error))), flush=True)
            return 2
    if not args.controller or not args.profile:
        parser.error('--controller and --profile are required')
    worker = Worker(read(args.profile), args.controller, poll_seconds=args.poll_seconds)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: worker.stop.set())
    worker.run(once=args.once, drain=args.drain)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
