#!/usr/bin/env python3
"""One bounded local extension; fresh independent holdout and final STOP."""
import argparse,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('TRACK_C_H4_COMMON_CACHE',str(root/'.cache/closed_p3/inputs/common.pkl'))
os.environ.setdefault('TRACK_C_CLOSED_P3_JOINT_OPT_OUTPUT',str(root/'artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended'))
from trottertracks.track_c_closed_p3_joint_extend import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','training','correctness','holdout','finish']);run(p.parse_args().stage)
