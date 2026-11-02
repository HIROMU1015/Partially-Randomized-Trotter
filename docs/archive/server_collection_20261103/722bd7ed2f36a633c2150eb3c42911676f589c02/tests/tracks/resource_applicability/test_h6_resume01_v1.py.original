"""Complete-only result reuse; metadata fixtures, no arrays or compilation."""
from copy import deepcopy
import pytest
from trottertracks.resource_applicability import h6_parallel_contract_v1 as c


def base():
    return {'reuse_registry':{'missing':[{'index':i,'task':{'seed':i},'partial_event_digest':'old'} for i in [447,*range(450,592)]]}}


def reuse():return {'accepted':[{'index':i} for i in c.REUSED_INDICES]}


def test_resume_exactly_135_preserves_task_order_and_original_seeds():
    rows=c.registrations(base(),reuse())
    assert len(rows)==135 and rows[0]['index']==456 and rows[1]['index']==458
    assert [x['index'] for x in rows]==[i for i in range(450,592) if i not in c.REUSED_INDICES]
    assert all(x['task']['seed']==x['index'] for x in rows)
    assert not set(x['index'] for x in rows)&set(c.REUSED_INDICES)


@pytest.mark.parametrize('case',['partial','duplicate','missing','extra','reordered'])
def test_partial_or_changed_reuse_set_cannot_change_dispatch(case):
    r=reuse()
    if case=='partial':r['accepted'][-1]['index']=456
    if case=='duplicate':r['accepted'][-1]['index']=450
    if case=='missing':r['accepted'].pop()
    if case=='extra':r['accepted'].append({'index':458})
    if case=='reordered':r['accepted'].reverse()
    with pytest.raises(ValueError):c.registrations(base(),r)


def test_partial_attempts_are_reserved_separately_and_not_refunded():
    history={'conservative_reservation':{'trajectory':4,'occurrence':32},'actual_call_totals':None}
    r={'partial_registered_task_reservation':{'compile':4,'control_probe':4,'trajectory':2,'occurrence':16},'partial_actual_totals':None}
    before=deepcopy((history,r))
    assert c.total_reservations(history,r)=={'compile':12,'control_probe':4,'trajectory':6,'occurrence':48}
    assert (history,r)==before and r['partial_actual_totals'] is None


def test_completed_known_calls_added_once_without_reservation_becoming_total(tmp_path):
    b={'reuse_registry':{'original_signal_calls':{'compile':904,'trajectory':434,'occurrence':1024},'accepted':[]}}
    r={'completed_known_calls':{'compile':14,'trajectory':7,'occurrence':56}}
    counts=c.carry_calls(tmp_path,b,None,roundoff=True,reuse=r)
    assert counts=={'compile':918,'trajectory':443,'occurrence':1096}
    assert counts['compile']+12+270==1200


@pytest.mark.parametrize('case',['old_142_grant','reuse_tamper','reuse_missing','valid'])
def test_new_grant_binds_exact_seven_result_reuse(case):
    m={'source_commit':'a'*40,'base_manifest':{'profile':{'sha256':'b'*64}},
        'registrations':c.registrations(base(),reuse()),'proposed_resources':c.proposed_resources(),
        'exclusive_output':{'absolute_path':'/home/AbeHiromu/ARTIFICIAL'},
        'roundoff_preparation_binding':{'fixture':True},'prior_stop_accounting':{'actual_call_totals':None},
        'verified_reuse':reuse()}
    g=c.grant_draft(m);g.update(approved_by_user=True,server_usage_approved_by_user=True)
    if case=='old_142_grant':g['kind']='H6_PARALLEL_ROUNDOFF_EXPLORATION_V1';g.pop('verified_reuse_digest')
    if case=='reuse_tamper':m['verified_reuse']['accepted'][0]['index']=456
    if case=='reuse_missing':m['verified_reuse']=None
    if case=='valid':c.validate_grant(m,g)
    else:
        with pytest.raises(ValueError):c.validate_grant(m,g)
