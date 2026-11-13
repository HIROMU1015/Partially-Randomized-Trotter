"""Approved fixed H4 K0 -> PASS-only K1. No sampler/compiler/price-provider."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--source',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not a.execute:p.error('Use --execute only for the approved scientific scope')
    from trottertracks.h4_structural_kernel import execute_batch
    execute_batch(ROOT,a.source,a.out)


if __name__=='__main__':main()
