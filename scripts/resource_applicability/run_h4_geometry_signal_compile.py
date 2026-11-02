#!/usr/bin/env python3
"""Future signal/compile-only runner; frozen inputs and separate authorization required."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).absolute().parents[2]/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','authorization','review'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--explicit-launch-signal-compile',action='store_true')
    args=parser.parse_args(argv)
    from trottertracks.resource_applicability.h4_geometry.execution import launch
    result=launch('signal_compile',*[json.loads(Path(p).read_bytes()) for p in (args.plan,args.authorization,args.review)],
                  explicit_launch=args.explicit_launch_signal_compile)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
