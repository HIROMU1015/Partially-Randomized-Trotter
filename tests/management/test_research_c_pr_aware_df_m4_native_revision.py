"""Metadata-only revision regression. No molecular decoding or scientific work."""
import copy,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OLD='docs/research/research_c_pr_aware_df_M4_native128_contract_unapproved_20261028.json'
NEW='docs/research/research_c_pr_aware_df_M4_native128_contract_revision2_20261028.json'

def test_revision_preserves_physical_sampling_numerical_and_call_conditions():
    old=json.loads((ROOT/OLD).read_text());new=json.loads((ROOT/NEW).read_text())
    allowed={'logical_unit_contents','source_manifest','source_manifest_sha256','revision'}
    assert {k:v for k,v in old.items() if k not in allowed}=={k:v for k,v in new.items() if k not in allowed}
    for key in ('trajectory_sample','kernel_build','trajectory_reference'):
        assert '128' in new['logical_unit_contents'][key] and '64' not in new['logical_unit_contents'][key]
    assert new['policy']['r']==new['sampling']['events_per_sample']==128
    assert new['sampling']['samples']==32 and sum(new['call_limits'].values())==332
    assert new['logical_unit_contents']=={k:(v.replace('64','128') if k in ('trajectory_sample','kernel_build','trajectory_reference') else v) for k,v in old['logical_unit_contents'].items()}
    before=json.loads((ROOT/old['source_manifest']).read_text())['files']
    after=json.loads((ROOT/new['source_manifest']).read_text())['files']
    assert all(after[k]==v for k,v in before.items()) and len(after)==len(before)+1
