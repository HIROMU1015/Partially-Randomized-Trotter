#!/usr/bin/env python3
"""Local central storage maintenance. Independent of worker jobs and calls."""
import argparse
from pathlib import Path
import time

from compact_rpe_git_objects import compact
from rpe_romutail_cold_storage import archive_old
from trottertracks.rpe_q3_coordinator import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--state-directory', type=Path, required=True)
    args=parser.parse_args()
    maintenance=args.state_directory/'maintenance'
    maintenance.mkdir(parents=True,exist_ok=True)
    # One pass can take minutes. The timer starts the next pass after this one
    # exits. Existing per-worker storage cleanup and routing are independent.
    try:
        compact(args.repository, maintenance)
    except Exception as error:
        print('Git compaction deferred:', str(error), flush=True)
    try:
        result=archive_old(args.state_directory)
    except Exception as error:
        result={'status':'COLD_ARCHIVE_DEFERRED','reason':str(error)[:350]}
    atomic_json(maintenance/'COLD_ARTIFACTS.json',dict(result,epoch=time.time()))
    print(result.get('status'),flush=True)

if __name__=='__main__':
    main()
