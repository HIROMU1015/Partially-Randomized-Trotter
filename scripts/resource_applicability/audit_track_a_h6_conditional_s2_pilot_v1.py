#!/usr/bin/env python3
"""Read-only source/profile/input/carry/law audit; fresh resources remain a launch gate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability import h6_conditional_s2_contract_v1 as c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    m=c.read(c.private_path(a.manifest.absolute()));c.validate_plan(ROOT,m,launch=True)
    print(json.dumps(dict(status='CONDITIONAL_S2_PILOT_LOCAL_BINDING_PASS_NOT_AUTHORIZED',source_commit=m['source_commit'],
        source_files=len(c.sources(ROOT)),groups=8,wrappers=16,workers=1,source_profile_input_carry_law_verified=True,
        manifest_digest=c.digest(m),approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        new_trajectory_calls=0,new_compile_calls=0,cap_amendments=m['cumulative_accounting']['cap_amendments']),sort_keys=True))


if __name__=='__main__':raise SystemExit(main())
