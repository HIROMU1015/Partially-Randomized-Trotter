"""Metadata-only preflight. Intentionally provides no execution command."""
import argparse
import json
import subprocess
from pathlib import Path

import pf_direct_compile_contract as c

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19'
OUT = ROOT/'artifacts/resource_applicability/track_a_pf_direct_compile_technical_preparation/2026-10-20'


def check():
    acceptance = c.read_json(OUT/'scope_acceptance.json')
    require = c.require
    require(acceptance['technical_preparation_authorized'] is True
            and acceptance['acquisition_scope_approved'] is False
            and acceptance['scientific_execution_authorized'] is False, 'AUTHORIZATION_BOUNDARY')
    preservation = c.read_json(OUT/'preservation.json')
    for path, digest in preservation['files'].items():
        require(c.sha(ROOT/path) == digest, 'ORIGINAL_PLAN_CHANGED:'+path)
    proposal = PLAN/'proposed_compile_axes.csv'
    rows = c.parse_proposals(proposal.read_bytes(), preservation['files'][str(proposal.relative_to(ROOT))])
    require(len(rows) == 788, 'UNCHANGED_788_PROPOSED_AXES')
    draft = c.read_json(OUT/'execution_contract_DRAFT.json')
    expected = c.draft(rows, c.sha(proposal), c.sha(PLAN/'system_bindings.json'))
    require(draft == expected, 'ONLY_DORMANT_CONTRACT')
    blocked = False
    try:
        c.authorize_metadata(draft, {}, rows)
    except ValueError as error:
        blocked = str(error) == 'SCIENTIFIC_EXECUTION_NOT_AUTHORIZED'
    require(blocked, 'DRAFT_MUST_REJECT_EXECUTION')
    # Unrelated worktrees/HEAD changes are not gates; only this owned diff is checked.
    allowed={'PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/README.md',
             'docs/research/研究概要・現状.md','scripts/README.md','src/trotterlib/README.md','artifacts/validation_manifest.json'}
    changed={p for p in subprocess.check_output(['git','diff','--name-only','-z',c.PLAN_COMMIT],cwd=ROOT).decode().split('\0') if p}
    owned={str(p.relative_to(ROOT)) for p in OUT.iterdir() if p.is_file()}
    code={'scripts/pf_direct_compile_contract.py','scripts/pf_direct_compile_adapter.py',
          'scripts/check_pf_direct_compile_preparation.py','scripts/prepare_pf_direct_compile_technical.py',
          'tests/test_pf_direct_compile_preparation.py'}
    docs={'docs/research/track_a_pf_direct_compile_technical_preparation_20261020.md',
          'docs/research/研究ノート/2026-10-20_pf_direct_compile_technical_preparation.md'}
    require(changed <= allowed|owned|code|docs, 'UNEXPECTED_OLD_SOURCE_OR_RESULT_CHANGE')
    for path in allowed-{'artifacts/validation_manifest.json'}:
        original=subprocess.check_output(['git','show',c.PLAN_COMMIT+':'+path],cwd=ROOT)
        require((ROOT/path).read_bytes().endswith(original),'OLD_INDEX_TEXT_CHANGED')
    old=json.loads(subprocess.check_output(['git','show',c.PLAN_COMMIT+':artifacts/validation_manifest.json'],cwd=ROOT))
    new=c.read_json(ROOT/'artifacts/validation_manifest.json')
    require(set(new)-set(old)=={'track_a_pf_direct_compile_technical_preparation_20261020'}
            and all(new[key]==value for key,value in old.items()),'OLD_VALIDATION_MANIFEST_CHANGED')
    closure = c.read_json(OUT/'technical_source_closure.json')
    for path, digest in closure['files'].items():
        require(c.sha(ROOT/path) == digest, 'TECHNICAL_SOURCE_CHANGED:'+path)
    return dict(status='PASS_METADATA_ONLY_PRODUCTION_DISPATCH_DISABLED',
        technical_preparation_authorized=True, proposed_axes=788, selected_axes=0,
        acquisition_scope_approved=False, scientific_execution_authorized=False,
        real_input_decodes=0, real_circuit_builds=0, real_compiles=0,
        molecular_loader_bound=False, live_gate_bound=False, worker_witness_bound=False,
        mock_validation_is_not_molecular_validation=True, final_evidence_freeze=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(check(), indent=2))
