#!/usr/bin/env python3
"""Independent, finite Closed P3 h/resolution study. See the study REPORT."""
import argparse
from pathlib import Path
import os
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('TRACK_C_H4_COMMON_CACHE',str(root/'.cache/closed_p3/inputs/common.pkl'))
from trottertracks.track_c_closed_p3_joint_opt import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','pilot','correctness','selected-correctness','training','refine','holdout','holdout-more','finish']);run(p.parse_args().stage)
