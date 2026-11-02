"""Generate only artificial Stream/run_pair fixtures and seven report examples.

Both input and output directories must be new. No scientific modules or results.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/tracks/algorithm_codesign')]
import b3_offline_artificial_fixtures as fixture
from trottertracks.algorithm_codesign.g6_w2c_b3_offline_cli import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('new_inputs', type=Path)
    p.add_argument('new_reports', type=Path)
    p.add_argument('--test-evidence', type=Path)
    a = p.parse_args()
    fixture.bundle(a.new_inputs, n=18)
    candidate = json.loads((ROOT/'docs/research/track_b_g6_w2c_b3_r1_f01_admission_integration_20261019/technical_contract_candidate.json').read_text())['formal_v2_template_not_executable']
    evidence = json.loads(a.test_evidence.read_text()) if a.test_evidence else None
    v, c, r = run(a.new_inputs/'manifest.json', a.new_reports, source_root=ROOT,
                  acceptance_baseline=candidate, unit_evidence=evidence)
    print(json.dumps({'verification': v['status'], 'cost': c['status'], 'acceptance': r['totals']}))
    return 0 if v['status'] == 'VERIFIED_COMPLETE' and c['status'] == 'COMPLETE_ARTIFICIAL_DESCRIPTIVE_COSTS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
