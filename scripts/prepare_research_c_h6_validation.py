"""Read-only H6 design check. No execution command or numerical imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_h6_validation_preparation as design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path,
        default=ROOT/'docs/research/research_c_h6_conditionfix_contract_20261106.json')
    args = parser.parse_args()
    c = json.loads(args.contract.read_text())
    if c.get('execution_ready') is True:
        from trottertracks.c_h6_contract import preflight
        print(json.dumps(dict(preflight(ROOT,args.contract),executable=False)))
        return
    if (c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT_DRAFT'
            or c['execution_ready'] is not False or c['execution_entry'] is not None
            or c['physical_proposal'] != design.specification()
            or c['workload_ceiling'] != design.workload()
            or c['retry'] != 0 or c['resume'] != 0 or c['mandatory_stop'] is not True):
        raise ValueError('H6_PREPARATION_DESIGN_MISMATCH')
    source = ROOT/c['technical_SOURCE']
    if hashlib.sha256(source.read_bytes()).hexdigest() != c['technical_SOURCE_sha256']:
        raise ValueError('TECHNICAL_SOURCE_MISMATCH')
    manifest = json.loads(source.read_text())
    for name, expected in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('TECHNICAL_FILE_MISMATCH:'+name)
    print(json.dumps(dict(status='PASSIVE_H6_DESIGN_CHECK_PASS',
        executable=False, molecular_arrays_loaded=0, scientific_calls=0,
        maximum_logical_units=c['workload_ceiling']['maximum_logical_units'],
        technical_source_files=len(manifest['files']),
        unresolved=c['decisions_before_execution'], mandatory_stop=True), ensure_ascii=False))


if __name__ == '__main__': main()
