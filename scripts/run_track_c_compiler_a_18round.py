"""Authorized finite Track C 18round compiler audit, with no price acquisition."""
import argparse
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--common-cache',type=Path,required=True)
parser.add_argument('--out',type=Path)
args=parser.parse_args()
os.environ['TRACK_C_H4_COMMON_CACHE']=str(args.common_cache.resolve())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import track_c_compiler_a_18round as study
if args.out is not None:study.OUT=args.out.resolve()

if __name__=='__main__':study.run()
