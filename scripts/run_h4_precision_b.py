#!/usr/bin/env python3
"""B saved-scalar analysis only. C real-data execution is not supported."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    for name in ['prepare','run','launch','verify']:
        actions.add_argument('--'+name,action='store_true')
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    path=Path(__file__).resolve().parents[1]/'src/trotterlib/h4_precision_control.py'
    spec=importlib.util.spec_from_file_location('b_only_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if args.prepare:
        if args.authorization is None:
            parser.error('--prepare requires the explicit B authorization file')
        result=module.prepare(args.pack,args.authorization)
    else:
        result=getattr(module,'run' if args.run else 'launch' if args.launch else 'verify')(args.pack)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))


if __name__=='__main__':
    main()
