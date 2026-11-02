"""Deterministic file-publication races and two-hop metadata resume, no science."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from trottertracks.resource_applicability import h8_resource_map_contract_v1 as c
from trottertracks.resource_applicability import h8_resource_map_handoff_v1 as h
from trottertracks.resource_applicability import h8_resource_map_execution_v1 as ex


@pytest.fixture(autouse=True)
def forbid_science(monkeypatch,tmp_path):
    assert tmp_path.is_relative_to('/home/AbeHiromu')
    import numpy as np
    import qiskit
    def forbidden(*a,**k):raise AssertionError('scientific execution forbidden')
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(qiskit,'transpile',forbidden)


def test_deleted_atomic_temporary_is_tolerated_and_final_counted(monkeypatch,tmp_path):
    final=tmp_path/'diagnostic_0013.json';final.write_bytes(b'final evidence')
    pending=tmp_path/'.pending_diagnostic_0013.json';pending.write_bytes(b'pending')
    original=Path.lstat
    def raced(path,*a,**k):
        if path==pending:pending.unlink()
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'lstat',raced)
    # Fix enumeration order so deletion is between listing and metadata read.
    monkeypatch.setattr(ex.os,'walk',lambda *a,**k:iter([(str(tmp_path),[],[pending.name,final.name])]))
    result=ex.inventory(tmp_path)
    assert result==dict(inodes=2,bytes=14,log_bytes=0,vanished_atomic_temporaries=1)


@pytest.mark.parametrize('name',['cost_result.json','.pending_runtime.log','cache.bin'])
def test_missing_non_atomic_file_fails_closed(monkeypatch,tmp_path,name):
    monkeypatch.setattr(ex.os,'walk',lambda *a,**k:iter([(str(tmp_path),[],[name])]))
    with pytest.raises(FileNotFoundError):ex.inventory(tmp_path)


def test_permission_error_not_hidden(monkeypatch,tmp_path):
    p=tmp_path/'.pending_diagnostic.json';p.write_text('x');old=Path.lstat
    def denied(path,*a,**k):
        if path==p:raise PermissionError('fixture')
        return old(path,*a,**k)
    monkeypatch.setattr(Path,'lstat',denied)
    with pytest.raises(PermissionError):ex.inventory(tmp_path)


@pytest.mark.parametrize('directory',[False,True])
def test_symlink_still_fails_closed(tmp_path,directory):
    target=tmp_path/'target'
    if directory:target.mkdir()
    else:target.write_text('evidence')
    (tmp_path/'link').symlink_to(target,target_is_directory=directory)
    with pytest.raises(ValueError,match='OUTPUT_SYMLINK'):ex.inventory(tmp_path)


def test_existing_temporary_counts_bytes_and_logs(tmp_path):
    (tmp_path/'.pending_diagnostic.json').write_bytes(b'x'*11)
    (tmp_path/'diagnostic.json').write_bytes(b'x'*11)
    (tmp_path/'worker.log').write_bytes(b'y'*5)
    r=ex.inventory(tmp_path);assert (r['bytes'],r['log_bytes'],r['inodes'])==(27,5,3)


def test_write_cap_uses_race_aware_scan(monkeypatch,tmp_path):
    vanished=tmp_path/'.pending_other.json'
    old=ex.os.walk
    def enumeration(output,**kwargs):
        yield str(output),[],[vanished.name]
    monkeypatch.setattr(ex.os,'walk',enumeration)
    w=ex.AtomicWriter(tmp_path,byte_cap=2**20);w.write('receipt.json',{'kept':True})
    assert json.loads((tmp_path/'receipt.json').read_text())=={'kept':True}


def manifests():
    hashes={p:'a'*64 for p in h.REPAIR_CHANGED_PATHS}
    hashes[h.PREFIX+'h8_resource_map_optimization_v1.py']='b'*64
    hashes['science.py']='c'*64
    old=dict(source_commit=h.REPAIR_PARENT_SOURCE,source_hashes=hashes,
        seed_source_commit=h.PARENT_SOURCE,input_identity={'fixed':True},profile={'fixed':True},
        compiler_hash=c.COMPILER,plan=c.plan(),cumulative_carry=c.carry(),
        proposed_resources=c.resources(),observation_policy=c.observation(),
        initial_registrations=c.signal_groups(),call_budget=dict(c.call_budget(),trajectory=835,occurrence=4211))
    new=deepcopy(old);new['source_commit']='d'*40;new['repair_authorization_basis']=h.REPAIR_AUTHORIZATION
    for p in h.REPAIR_CHANGED_PATHS:new['source_hashes'][p]='e'*64
    return old,new


def test_repair_transition_preserves_frozen_adapter():
    old,new=manifests();assert not h.compatibility(new,old)['scientific_source_changed']
    new['source_hashes'][h.PREFIX+'h8_resource_map_optimization_v1.py']='f'*64
    with pytest.raises(ValueError,match='SCIENCE_SOURCE_CHANGED'):h.compatibility(new,old)


def test_prior_budget_extension_is_preserved():
    old,new=manifests();new['resume_parent']={'additional_held_reservations':{'compile':14,'trajectory':8,'occurrence':8}}
    new['call_budget'].update(compile=1722,trajectory=843,occurrence=4219)
    assert h.compatibility(new,old)['extra_reservations']['compile']==14


@pytest.mark.parametrize('tamper',[False,True])
def test_inherited_signal_reused_in_second_resume(monkeypatch,tmp_path,tamper):
    old,new=manifests();parent=tmp_path/'parent';parent.mkdir();first=tmp_path/'first';first.mkdir()
    st=dict(kind='SIGNAL',cell_ids=['toy'])
    # Stored predecessor complete signal remains in its original directory.
    signal=first/'signal.json';signal.write_text('original immutable evidence')
    record=dict(task=st,result={'signals':[{'cell':{'id':'toy'}}]},directory=str(first))
    old['resume_parent']=dict(accepted={'signal':{'0':record}},file_hashes={str(signal):c.file_hash(signal)})
    identity=dict(pid=999999,start_ticks=17,pgrp=999999,session=999999)
    ledger=dict(known={'signal':1},reserved={'compile':2,'trajectory':1,'occurrence':1})
    for name,value in [('manifest.json',old),('coordinator_identity.json',identity),
        ('execution_terminal.json',{'SOURCE':h.REPAIR_PARENT_SOURCE,'ledger':ledger})]:
        (parent/name).write_text(json.dumps(value))
    native=tmp_path/'native.json';native.write_text(json.dumps(dict(identity=identity,wait_reaped=True,exit_code=1)))
    pd=parent/'signal';pd.mkdir()
    (pd/'registered_tasks.json').write_text(json.dumps({'registrations':[dict(index=0,task=st)]}))
    (pd/'execution_terminal.json').write_text(json.dumps(dict(all_dispatched_reaped=True,cleanup_errors=[],retired=[],completed_indices=[0])))
    from trottertracks.resource_applicability import h6_server_resources_v1 as rs
    def absent(pid):raise FileNotFoundError()
    monkeypatch.setattr(rs,'process_identity',absent)
    if tamper:
        signal.write_text('modified')
        with pytest.raises(ValueError,match='ANCESTOR_CHANGED'):c.bind_resume_parent(new,parent,native)
    else:
        c.bind_resume_parent(new,parent,native)
        assert new['resume_parent']['accepted']['signal']['0']==record
        assert new['resume_parent']['parent_ledger']==ledger
        assert new['call_budget']['trajectory']==836
        assert not new['resume_parent']['budget_extension_approved']
