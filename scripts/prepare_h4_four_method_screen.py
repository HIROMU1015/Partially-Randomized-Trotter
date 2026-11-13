"""Metadata-only fixed 20-cell screen preparation; no scientific execute flag."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.h4_four_method_screen import plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=plan(ROOT);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:f.write(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(fresh=len(result['fresh']),saved=len(result['reuse']),science_calls=0,mandatory_stop=True)))


if __name__=='__main__':main()
