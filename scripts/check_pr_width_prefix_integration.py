"""Reproduce the integrated-width CLI/queue's actual DP HIT on artificial tables."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'scripts'),str(ROOT/'src')]
from test_pr_width_prefix_execution import acceptance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(acceptance(args.output),indent=2))


if __name__=='__main__':main()
