"""A fixed H-chain plan yields the ordinary preparation graph and leaf outputs."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_rpe_hchain_preparation as hchain
from trottertracks.rpe_preparation_queue import normalize_preparation_request


def test_fixed_batches_reuse_signal_price_and_audit_provider(tmp_path):
    signal = dict(id="physical-signal", method="PR", R_outer=2)
    anchor = dict(id="physical-anchor", method="PR", R_outer=5, m=None)
    holdout = dict(id="physical-holdout", method="PR", R_outer=8, m=None)
    designs = {"model": dict(anchors=[anchor], holdout=holdout)}
    cases = [dict(rounds=[signal])]
    plan = dict(cases=cases, designs=list(designs.values()), scope=dict(price_paths=4),
                owner="owner", binding={"system": "H8"}, target=-1, compiler="fixed",
                price_cohort="retained-cohort")
    batches = [("signal", [[signal["id"], 0]]), ("audit", [[holdout["id"], 1]]),
               ("compile", [[anchor["id"], 0], [anchor["id"], 1]])]
    settings = dict(source=dict(commit="a"*40, files={}), assets=dict(controller_path="assets"),
                    budgets=dict(signal=1, audit=1, compile=1), approved=True)
    graph = hchain.preparation_graph(plan, "b"*64, batches, **settings)
    requests = [normalize_preparation_request(task["request"]) for task in graph["tasks"]]
    assert len({request["request_id"] for request in requests}) == len(batches)
    assert requests[1]["preparation"]["outputs"] == [
        "prices/physical-holdout-1.json", "prices/physical-holdout-1.events.json.xz"]
    calls = []
    state = SimpleNamespace(cases=cases, binding=plan["binding"], target=plan["target"], compiler=plan["compiler"])

    def price(cell, index, cohort):
        calls.append((cell, index, cohort))
        return {"rz": 10}

    provider = SimpleNamespace(startup=lambda *args: state, calibration_design=lambda *args: designs,
                              signal=lambda cell: {"candidate": cell["id"]}, price=price,
                              write=lambda path, data: path.write_text(str(data)),
                              write_price=lambda path, data: path.write_text(str(data)))
    for stage, targets in batches:
        hchain.run(plan, stage, targets, tmp_path, tmp_path, provider)
    assert calls == [(holdout, 1, "retained-cohort"), (anchor, 0, "retained-cohort"),
                     (anchor, 1, "retained-cohort")]
    assert (tmp_path / "signals/physical-signal.json").is_file()
    assert (tmp_path / "prices/physical-anchor-1.json").is_file()
    with pytest.raises(ValueError, match="more than one batch"):
        hchain.preparation_graph(plan, "b"*64, [*batches, batches[0]], **settings)
    with pytest.raises(ValueError, match="outside the specified plan and stage"):
        list(hchain.leaves(plan, "compile", [[holdout["id"], 1]]))
    assert len(calls) == 3


def test_publish_preparation_adopts_after_the_complete_price_pair(tmp_path):
    artifacts = {"prices/round-0.json": "receipt", "prices/round-0.events.json.xz": "events"}
    spec = dict(provider="HCHAIN_SCOPED", outputs=list(artifacts))
    calls = []
    executor = SimpleNamespace(
        store=SimpleNamespace(row=lambda job: {"request": {"preparation": spec}}),
        preparation_result=lambda *args, **kwargs: dict(status="PREPARATION_COMPLETE", artifacts=artifacts),
        artifact_cache=SimpleNamespace(materialize=lambda files, *args, **kwargs: calls.extend(files.values())),
        adopt_preparation=lambda job: calls.append(job))
    hchain.publish_preparation(executor, "job", tmp_path)
    assert calls == ["events", "receipt", "job"]
    calls.clear()
    spec["outputs"].append("extra-output")
    with pytest.raises(ValueError, match="output names differ"):
        hchain.publish_preparation(executor, "job", tmp_path)
    assert calls == []


def test_startup_mismatch_reports_exact_values_before_any_leaf(tmp_path, capsys):
    cell = dict(id='signal', method='FULL', R_outer=None)
    plan = dict(cases=[dict(rounds=[cell])], scope=dict(price_paths=1), owner='owner',
                binding={'system':'H8'}, target=-1, compiler='fixed',
                verified_saved_preparations={'1':'expected'})
    state = SimpleNamespace(binding=plan['binding'], target=-1, compiler='fixed',
                            preparation_sources={1:'different'})
    provider = SimpleNamespace(startup=lambda *_:state,
        signal=lambda *_:pytest.fail('must not start leaf'))
    with pytest.raises(hchain.StartupMismatch, match='verified_saved_preparations') as error:
        hchain.run(plan, 'signal', [['signal',0]], tmp_path, tmp_path, provider)
    assert error.value.expected == {'1':'expected'}
    assert error.value.observed == {'1':'different'}
    assert 'HCHAIN_STARTUP_MISMATCH' in capsys.readouterr().out
    assert not (tmp_path/'signals').exists()


def test_startup_hold_is_shared_by_batches_of_the_same_fixed_owner(tmp_path):
    from trottertracks.rpe_preparation_worker import hchain_startup_hold
    profile = dict(work_dir=str(tmp_path), host_id='local', interpreter='fixed-python')
    request = dict(preparation=dict(provider='HCHAIN_SCOPED',source={'commit':'fixed'}),
                   original_contract=dict(scope_owner='owner',binding={'system':'H8'}))
    fingerprint = dict(input_artifacts={'PLAN.json':'batch-a'},runtime_paths={},runtime={},tools={})
    first = hchain_startup_hold(profile,request,fingerprint)
    second = hchain_startup_hold(profile,request,dict(fingerprint,input_artifacts={'PLAN.json':'batch-b'}))
    assert first == second
    other = dict(request,original_contract=dict(scope_owner='other',binding={'system':'H8'}))
    assert hchain_startup_hold(profile,other,fingerprint) != first


def test_h8_archived_preparation_identity_checked_without_startup(tmp_path):
    from trottertracks.rpe_preparation_worker import verify_h8_saved_preparation_metadata
    import json
    path=tmp_path/'inputs'/'h8_preparation_export.json';path.parent.mkdir()
    path.write_text(json.dumps({'faithful:p2':{'preparation_hash':'expected'}}))
    spec={'h8_saved_preparation_hashes':{'2':'expected'}}
    facts={'hchain_assets':{'path':str(tmp_path),'files':{'inputs/h8_preparation_export.json':'fixture'}}}
    verify_h8_saved_preparation_metadata(spec,facts)
    with pytest.raises(ValueError,match='prefix 2'):
        verify_h8_saved_preparation_metadata({'h8_saved_preparation_hashes':{'2':'wrong'}},facts)
    with pytest.raises(ValueError,match='not declared'):
        verify_h8_saved_preparation_metadata(spec,{'hchain_assets':{'path':str(tmp_path)}})


def test_check_startup_only_does_not_start_leaf(tmp_path):
    cell=dict(id='signal',method='FULL',R_outer=None)
    plan=dict(cases=[dict(rounds=[cell])],scope=dict(price_paths=1),owner='owner',binding={'system':'H8'},target=-1,compiler='fixed')
    state=SimpleNamespace(binding=plan['binding'],target=-1,compiler='fixed',cases=plan['cases'])
    provider=SimpleNamespace(startup=lambda *_:state,signal=lambda *_:pytest.fail('leaf'))
    assert hchain.run(plan,'signal',[['signal',0]],tmp_path,tmp_path,provider,check_startup=True)['leaf_calls']==0


def test_hchain_fixed_kernel_changes_request_identity_without_changing_old_contract():
    cell=dict(id='fixture',method='FULL',R_outer=None)
    plan=dict(cases=[dict(rounds=[cell])],scope=dict(price_paths=1,system='H8'),
              owner='fixture',binding={'system':'H8'})
    kwargs=dict(source={'commit':'a'*40,'files':{}},assets={'controller_path':'fixture'})
    prior=normalize_preparation_request(hchain.preparation_request(plan,'b'*64,'signal',[['fixture',0]],**kwargs))
    fixed=normalize_preparation_request(hchain.preparation_request(plan,'b'*64,'signal',[['fixture',0]],
                                             openblas_coretype='HASWELL',**kwargs))
    assert 'openblas_coretype' not in prior['preparation']['runtime_requirements']
    assert fixed['preparation']['runtime_requirements']['openblas_coretype']=='HASWELL'
    assert fixed['request_id']!=prior['request_id']


def test_full_startup_checks_every_case_and_design_with_one_startup(tmp_path):
    cells=[dict(id='first',method='FULL',R_outer=None),dict(id='late',method='FULL',R_outer=None)]
    cases=[dict(id=cell['id']+'-case',rounds=[cell]) for cell in cells]
    design=dict(anchors=[cells[0]],holdout=cells[1])
    plan=dict(cases=cases,designs=[design],scope=dict(price_paths=1),binding={'fixture':1},
              owner='owner',target=1,compiler='fixed')
    calls=[]
    state=SimpleNamespace(cases=cases,binding=plan['binding'],target=1,compiler='fixed')
    provider=SimpleNamespace(startup=lambda *_:(calls.append('startup'),state)[1],
                            calibration_design=lambda *_:{'fixture':design},
                            signal=lambda *_:pytest.fail('leaf'),
                            price=lambda *_:pytest.fail('leaf'))
    result=hchain.run(plan,None,None,tmp_path,tmp_path,provider,check_startup=True,check_full_startup=True)
    assert result['schedules']=={'signal':2,'compile':1,'audit':1}
    assert result['leaf_calls']==0 and calls==['startup']
    changed=dict(plan,designs=[dict(anchors=[cells[1]],holdout=cells[0])])
    with pytest.raises(hchain.StartupMismatch,match='calibration_design'):
        hchain.run(changed,None,None,tmp_path,tmp_path,provider,check_startup=True,check_full_startup=True)


def test_provider_contract_versions_are_fixed_by_plan_not_system_name():
    from rpe_hchain_q3 import supported_binding
    cell=dict(id='fixture',method='FULL',R_outer=None)
    binding={'system':'synthetic','rank':3}
    contract=dict(system='synthetic',rank=3,packages={'qiskit':'fixture-Q','numpy':'fixture-N'})
    plan=dict(cases=[dict(rounds=[cell])],scope=dict(price_paths=1,system='synthetic'),owner='fixture',
              binding=binding,provider_contract=contract)
    kwargs=dict(source={'commit':'a'*40,'files':{}},assets={'controller_path':'fixture'})
    request=hchain.preparation_request(plan,'b'*64,'signal',[['fixture',0]],**kwargs)
    assert request['preparation']['runtime_requirements']['packages']==contract['packages']
    assert supported_binding(binding,contract) and not supported_binding(binding)
    assert supported_binding({'system':'H8','rank':28})
    with pytest.raises(ValueError,match='fixed binding'):
        hchain.preparation_request(dict(plan,provider_contract=dict(contract,rank=4)),
                                   'b'*64,'signal',[['fixture',0]],**kwargs)
