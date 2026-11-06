"""Two authorized, independent finite work streams; no old launcher."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('QISKIT_PARALLEL','FALSE')

parser=argparse.ArgumentParser()
parser.add_argument('stream',choices=['A','B'])
parser.add_argument('--common-cache',type=Path,required=True)
parser.add_argument('--out',type=Path)
args=parser.parse_args()
os.environ['TRACK_C_H4_COMMON_CACHE']=str(args.common_cache.resolve())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    if args.stream=='B':
        from trottertracks import track_c_cn_a_price as study
    else:
        from trottertracks import track_c_source_guarantee as study
    if args.out is not None:study.OUT=args.out.resolve()
    study.run()
