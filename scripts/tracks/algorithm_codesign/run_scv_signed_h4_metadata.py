"""Explicit once-only metadata command; no action on import."""
import argparse
from pathlib import Path
from trottertracks.algorithm_codesign.scv_signed_h4_metadata import run_fixed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    result=run_fixed(root,args.source_commit)
    print(result['metadata_structural_eligibility'])
    return 1 if result['metadata_structural_eligibility']=='UNKNOWN_ERROR' else 0


if __name__=='__main__':raise SystemExit(main())
