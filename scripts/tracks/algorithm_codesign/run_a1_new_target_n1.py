"""Freeze an N1 source/dependency profile, or run the explicitly authorized N1 once."""
import argparse
import json
from pathlib import Path
from trottertracks.algorithm_codesign.a1_new_target_n1 import dependencies, profile, run, write

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-profile',type=Path)
    group.add_argument('--output-dir',type=Path)
    parser.add_argument('--profile-lock',type=Path)
    args=parser.parse_args()
    if args.freeze_profile:
        dependencies()
        write(args.freeze_profile,profile(Path(__file__).resolve().parents[3]))
        print('N1 profile frozen; no scientific input loaded')
    else:
        if args.profile_lock is None:parser.error('--profile-lock required for N1')
        r=run(args.output_dir,args.profile_lock)
        print(json.dumps({k:r[k] for k in ('status','source_commit','identities','positive_grid_support','counts')},indent=2))
