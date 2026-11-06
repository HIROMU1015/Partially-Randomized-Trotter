#!/usr/bin/env python3
"""Freeze N2 profile or perform the authorized single limited N2 transaction."""
import argparse,json
from pathlib import Path
from trottertracks.algorithm_codesign import a1_new_target_n2 as n2
p=argparse.ArgumentParser();p.add_argument('--freeze-profile');p.add_argument('--profile-lock');p.add_argument('--output-dir')
a=p.parse_args();root=Path(__file__).resolve().parents[3]
if a.freeze_profile:
    n2.n1.dependencies();n2.write(a.freeze_profile,n2.profile(root))
elif a.profile_lock and a.output_dir:print(json.dumps(n2.run(a.output_dir,a.profile_lock,root),indent=2))
else:p.error('use --freeze-profile OR --profile-lock with --output-dir')
