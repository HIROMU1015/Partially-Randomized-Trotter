"""Safe metadata planning CLI. Never dispatches molecular science."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def kernel():
    spec=importlib.util.spec_from_file_location('m5_plan',ROOT/'src/trotterlib/h46_m5_plan.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output');a=p.parse_args()
    if a.execute:
        print(json.dumps({'status':'REFUSED','reason':'SEPARATE_SCIENTIFIC_APPROVAL_AND_SEALED_PRODUCTION_ADAPTER_REQUIRED','sampling':0,'compile':0}));return 2
    value=kernel().specification()
    if a.output:
        with Path(a.output).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    else:print(json.dumps({'status':value['status'],'stage_count':11,'pilot_axes':60,'pilot_trajectories':12,'parallel_P2_lanes':4,'science_started':False}))
    return 0


if __name__=='__main__':raise SystemExit(main())
