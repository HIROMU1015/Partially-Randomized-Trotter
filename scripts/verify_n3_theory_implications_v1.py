#!/usr/bin/env python3
"""Check reviewed implications of frozen saved witnesses. Never generate witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_old_family_inputs_v1 import IDS, PREFIX, verify_inputs
from trottertracks.representation_exploration.n3_theory_implications_v1 import saved_implication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_bytes())
    for rel, sha in identity['files'].items():
        path = ROOT / rel
        if not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError('PATH_SCOPE')
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha: raise ValueError('FROZEN_SAVED_BYTES:'+rel)
    verify_inputs(ROOT)
    rows=[]
    for name in IDS:
        row = json.loads((ROOT / PREFIX / 'inputs' / (name+'.json')).read_bytes())
        record = json.loads((ROOT / PREFIX / 'diagnostic1' / (name+'.json')).read_bytes())
        summary = json.loads((ROOT / row['provenance']['summary_path']).read_bytes())
        ans = saved_implication(row['instance'], record, summary)
        if ans['saved_winner_id'] != row['provenance']['saved_OCC_winner_id']:
            raise ValueError('SAVED_WINNER_PROVENANCE')
        ans['inherited_OCC_audit'] = json.loads((ROOT / PREFIX / 'diagnostic1' / (name+'.saved_OCC_audit.json')).read_bytes())
        if ans['inherited_OCC_audit']['status'] != 'FROZEN_U_C_AND_SAVED_OCC_TARGET_CERTIFICATE_PASS':
            raise ValueError('ORIGINAL_OCC_AUDIT')
        rows.append(ans)
    print(json.dumps({'schema_version':1, 'status':'SAVED_ANALYTIC_IMPLICATION_CHECKS_COMPLETE',
                     'base_commit':identity['base_commit'], 'cases':rows,
                     'new_inputs':0, 'new_eigensolver_calls':0, 'new_optimizers':0,
                     'new_certificate_searches':0, 'scientific_batch_launches':0,
                     'mandatory_stop':True, 'next_stage_authorized':False, 'central_theme':None},indent=2))


if __name__ == '__main__': main()
