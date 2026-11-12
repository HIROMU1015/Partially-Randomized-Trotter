"""Fixed S0 -> B V2 metadata only; no scientific evaluation entry point."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_rpe_rmse_s0_v2 as adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Use a new output directory; preserve old results')
    result=adapter.build(ROOT)
    artifacts={'V2_CONFIGURATION.json':result['connection']['configuration'],
               'V2_N_HANDOFF.json':result['connection']['N_packet'],
               'COVERAGE.json':result['local_coverage'],
               'B_EVIDENCE_PLAN.json':result['B_evidence_plan'],
               'SOURCE_INDEX.json':dict(source=adapter.SOURCE,source_files=result['source_files'],
                    source_hash_scope='READ_ARTIFACTS_ONLY_PLUS_PINNED_CODE_REFERENCES',
                    science_calls=0,mandatory_stop=True)}
    args.out.mkdir(parents=True)
    for name,value in artifacts.items():
        (args.out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(configuration_id=result['connection']['configuration']['id'],
        saved_additional_rounds=result['local_coverage']['additional_saved_mean_rounds'],
        missing_mean_rounds=result['local_coverage']['missing_mean_rounds'],science_calls=0,mandatory_stop=True)))


if __name__=='__main__':main()
