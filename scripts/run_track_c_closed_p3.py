"""Run only the newly authorized fixed-H4 Closed-P3 integration namespace."""
import argparse
from pathlib import Path
import os
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('TRACK_C_H4_COMMON_CACHE',str(root/'.cache/closed_p3/inputs/common.pkl'))
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['stage1','correctness','pilot','holdout','finish','fair']);a=parser.parse_args()
if a.stage=='stage1':
    from trottertracks.track_c_closed_p3_validation import stage_one
    stage_one()
elif a.stage=='fair':
    from trottertracks.track_c_closed_p3_fairness import run
    run()
else:
    from trottertracks.track_c_closed_p3_resources import run
    run(a.stage)
