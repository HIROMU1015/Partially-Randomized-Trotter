"""Explicit fixed-five-arm/fifteen-point numerical diagnosis; no repair."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trotterlib.b2_p1_strong_pf_h4_diagnosis import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--private-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.private_dir,args.output)
