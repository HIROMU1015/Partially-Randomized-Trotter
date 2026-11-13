"""Default: static 13-round metadata. --execute is future approved science."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--source');a=p.parse_args()
    from trottertracks import h4_structural_schedule as s
    if a.execute:
        if not a.source:p.error('Fixed execution SOURCE required')
        s.execute(ROOT,a.source,a.out)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        s.write(a.out/'COVERAGE.json',s.prepare(ROOT))


if __name__=='__main__':main()
