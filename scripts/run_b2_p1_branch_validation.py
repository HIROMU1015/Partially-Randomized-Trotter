"""Fixed saved-means-only branch reconstruction. No private file arguments."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trotterlib.b2_p1_branch_validation import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.output)
