"""Fixed m=1..16 continuation. Requires the user's separate execution scope."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trotterlib.b2_p1_remaining_rounds import run

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--private-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.private_dir,args.output)
    print(result['status'])
