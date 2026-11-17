#!/usr/bin/env python3
"""Summarize a normal distributed run without opening numerical Q3 results."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_q3_timing import summarize


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory',type=Path)
    p.add_argument('--timeline',type=Path,help='Historical published controller timeline')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.timeline and not a.timeline.exists():p.error('Timeline missing')
    value=summarize(a.run_directory,timeline=a.timeline)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(attempts=value['attempts'],hosts={k:v['count']for k,v in value['by_host'].items()},
        missing=value['missing_counts'],output=str(a.output)),ensure_ascii=False))
    return value
if __name__=='__main__':main()
