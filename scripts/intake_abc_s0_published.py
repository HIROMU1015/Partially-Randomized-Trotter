#!/usr/bin/env python3
"""Published A/C intake only; no S0 regeneration or scientific runner options."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.abc_s0_published_intake import published_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output JSON path; existing files are preserved')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    result = published_result(ROOT)
    plan = result.pop('planner')
    # Keep a small summary of metadata acceptance, not a second canonical/13-round
    # comparison archive. The planner and original packets are already published.
    result['planner_status'] = {k:plan[k] for k in ('status','G','winner','method_coverage',
        'comparison_complete','pending_configuration_ids','scientific_execution_performed','automatic_science_calls')}
    result['sufficient_N_missing_axes'] = len(plan['configurations'][0]['rows'])
    result['blockers'] = plan['configurations'][0]['reasons']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(A_packet_accepted=result['A_packet_accepted'], C_requests=result['C_formal_axis_requests'],
        G=result['G'], winner=result['winner'], scientific_calls=0)))


if __name__ == '__main__':
    main()
