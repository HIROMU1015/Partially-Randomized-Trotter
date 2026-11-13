"""Export published diagnostics only; no scientific execution option."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.h4_pr_physical_inputs import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(result['status'])


if __name__ == '__main__':
    main()
