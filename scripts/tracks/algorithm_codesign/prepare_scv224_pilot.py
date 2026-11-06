#!/usr/bin/env python3
"""Preparation information only. No real scientific launcher in this SOURCE."""
import argparse
import json
from trottertracks.algorithm_codesign.scv224_pilot import preparation_plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--show-plan',action='store_true')
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    if args.run:
        p.error('PREPARATION_ONLY: 224-shot science is not activated; explicit grant and fixed activation SOURCE required')
    print(json.dumps(preparation_plan(),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
