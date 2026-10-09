"""No-science tests of task identities, caps and fail-closed metadata."""
import pytest
from trottertracks.resource_applicability.ax2b_preflight import (
    REQUIRED_IDENTITIES,expanded_pilot_proposal,preflight_report,seed_for,
)


def test_wrapper_budget_is_complete_and_model_independent():
    p=expanded_pilot_proposal()
    assert p['planned_wrapper_calls']==60
    assert p['planned_wrapper_calls_by_system']=={'H4':28,'H6':32}
    assert len({t['id'] for t in p['wrapper_tasks']})==60
    assert len(p['H6_tasks'])==6 and not p['H8_tasks']
    assert len(p['H4_correctness_cells'])==8
    assert p['unique_random_trajectory_count']==8
    assert p['df_tolerance_adopted'] is False and p['actual_df_rank'] is None
    assert p['assigned_resources'] is None
    assert p['scientific_runner_implemented'] is False


def test_axis_and_control_variants_share_each_explicit_trajectory():
    p=expanded_pilot_proposal()
    random=[t for t in p['wrapper_tasks'] if t['trajectory_seed'] is not None]
    assert len(random)==32
    for task in random:
        shared=[t for t in random if t['trajectory_seed']==task['trajectory_seed']]
        assert len(shared)==4
        assert {t['axis'] for t in shared}=={'cosine','sine'}
        assert {t['control_policy'] for t in shared}=={'ordinary','symmetric_directional'}


def test_preflight_never_grants_authority_even_when_fields_are_present():
    empty=preflight_report()
    assert empty['identity_missing']==list(REQUIRED_IDENTITIES)
    full=preflight_report({name:'synthetic placeholder' for name in REQUIRED_IDENTITIES})
    assert full['identity_presence_complete'] is True
    assert full['certificate_semantics_verified'] is False
    for report in (empty,full):
        assert report['launch_allowed'] is False
        assert report['science_authorized'] is False and report['ax2b_authorized'] is False
        assert report['mandatory_stop'] is True


@pytest.mark.parametrize('replica',[-1,True,1.5])
def test_seed_identity_rejects_non_integer_replica(replica):
    with pytest.raises(ValueError):seed_for('toy',replica)


def test_proposal_is_repeatable_and_returns_independent_objects():
    first=expanded_pilot_proposal();second=expanded_pilot_proposal()
    assert first==second
    first['proposed_caps']['total_wrappers']=0
    assert second['proposed_caps']['total_wrappers']==64
    assert seed_for('toy',0)==seed_for('toy',0)
    assert seed_for('toy',0)!=seed_for('toy',1)
