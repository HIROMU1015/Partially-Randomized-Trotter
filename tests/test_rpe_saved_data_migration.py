"""Isolated saved-data migration, restoration and fixed-run compatibility."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from rpe_saved_data_migration import (copy_saved_tree,migrate_git_repository,prepare_evidence_service,inventory,
    stage_git_object_store,copy_compressed_file,restore_compressed_file)
from rpe_romutail_retention import (compact_stopped_provider_sources,pack_stopped_diagnostics,
    retire_stopped_diagnostics,restore_stopped_layout,storage_dependency_inventory)
from trottertracks.rpe_local0_search import Gateway,RequestError,write,load
from test_rpe_local0_search import fixture_config,request


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE,text=True).strip()


def init(root):
    root.mkdir(parents=True);git(root,'init');git(root,'config','user.name','fixture');git(root,'config','user.email','fixture@example.invalid')
    path=root/'src/file.py';path.parent.mkdir();path.write_text('fixed fixture\n');git(root,'add','.');git(root,'commit','-m','fixture')
    return git(root,'rev-parse','HEAD')


def test_git_migration_preserves_unreferenced_objects_and_alternate_switch(tmp_path):
    seed=tmp_path/'seed';source_sha=init(seed)
    orphan=seed/'unreachable';orphan.write_text('historical locator with no ref');oid=git(seed,'hash-object','-w',str(orphan));orphan.unlink()
    original=tmp_path/'publication'
    subprocess.run(['git','clone','--shared',str(seed),str(original)],check=True,capture_output=True)
    local=original/'untracked.json';local.write_text('{"saved":true}')
    alternate=original/'.git/objects/info/alternates';old_alternate=alternate.read_bytes()
    before=inventory(original)
    empty=tmp_path/'empty';init(empty)
    # Fresh independent seed contains the same fixed source file but not the
    # original historical commit and unreachable blob.
    with pytest.raises(ValueError,match='lacks a source object'):
        migrate_git_repository(original,tmp_path/'bad',target_objects=empty/'.git/objects')
    assert not (tmp_path/'bad').exists() and inventory(original)==before
    independent=tmp_path/'independent'
    result=migrate_git_repository(original,independent)
    assert result['status']=='VERIFIED_GIT_COPY' and not (independent/'.git/objects/info/alternates').exists()
    switched=tmp_path/'results-publication'
    assert migrate_git_repository(original,switched,target_objects=independent/'.git/objects')['status']=='VERIFIED_GIT_COPY'
    assert alternate.read_bytes()==old_alternate and inventory(original)==before
    seed.rename(tmp_path/'retired-seed');original.rename(tmp_path/'retired-publication')
    for root in (independent,switched):
        assert git(root,'rev-parse','HEAD')==source_sha
        assert git(root,'cat-file','-p',oid)=='historical locator with no ref'
        assert (root/'untracked.json').read_text()=='{"saved":true}'
        git(root,'fsck','--connectivity-only')
    # Exercise ordinary Git publication plumbing on the migrated copy (no
    # actual result publication, remote interaction or production data).
    git(switched,'config','user.name','fixture');git(switched,'config','user.email','fixture@example.invalid')
    git(switched,'add','untracked.json');git(switched,'commit','-m','synthetic publication fixture')
    assert git(switched,'show','HEAD:untracked.json')=='{"saved":true}'


def test_trial_copy_keeps_bytes_and_flags_external_dependencies(tmp_path):
    trial=tmp_path/'planning';trial.mkdir();(trial/'raw.json').write_bytes(b'{"raw":1}\n')
    os.link(trial/'raw.json',trial/'second.json')
    (trial/'internal').symlink_to('raw.json');external=tmp_path/'other';external.write_text('outside');(trial/'external').symlink_to(external)
    copied=tmp_path/'cold/trial'
    before=inventory(trial);report=copy_saved_tree(trial,copied)
    assert report['external_links']==['external'] and not report['independent']
    assert inventory(copied)==before==inventory(trial)
    assert (copied/'internal').read_bytes()==b'{"raw":1}\n'
    assert not (trial/'raw.json').samefile(copied/'raw.json')
    assert (copied/'raw.json').samefile(copied/'second.json')
    with pytest.raises(ValueError,match='destination already exists'):copy_saved_tree(trial,copied)
    (trial/'external').unlink()
    assert copy_saved_tree(trial,tmp_path/'cold/independent')['independent']


def test_stage_object_store_for_linked_worktrees(tmp_path):
    seed=tmp_path/'planning/receipt-object-view';source_sha=init(seed)
    source=tmp_path/'main';subprocess.run(['git','clone','--shared',str(seed),str(source)],check=True,capture_output=True)
    linked=tmp_path/'linked';git(source,'worktree','add','--detach',str(linked))
    before=inventory(source);alternate=source/'.git/objects/info/alternates'
    staged=stage_git_object_store(linked,tmp_path/'cold/objects.git')
    assert staged['status']=='VERIFIED_OBJECT_STORE' and inventory(source)==before
    # Simulate the switch only on the disposable fixture, including the
    # linked worktree. No path from the real repository is involved.
    alternate.write_text(staged['objects']+'\n');seed.rename(tmp_path/'retired-planning')
    for checkout in (source,linked):
        assert git(checkout,'rev-parse','HEAD')==source_sha
        assert git(checkout,'show',source_sha+':src/file.py')=='fixed fixture'
        git(checkout,'fsck','--connectivity-only')


def test_raw_result_cold_copy_preserves_very_large_integer_bytes(tmp_path):
    original=tmp_path/'HCHAIN_RESULT.json';raw=b'{"integer":'+b'9'*70001+b',"text":"raw"}\n';original.write_bytes(raw)
    archive=tmp_path/'cold/HCHAIN_RESULT.json.xz';result=copy_compressed_file(original,archive)
    assert result['sha256']==hashlib.sha256(raw).hexdigest() and original.read_bytes()==raw
    restored=tmp_path/'restored/HCHAIN_RESULT.json'
    with pytest.raises(ValueError,match='digest differs'):restore_compressed_file(archive,restored,'0'*64)
    assert not restored.exists()
    restore_compressed_file(archive,restored,result['sha256'])
    assert restored.read_bytes()==raw


def test_new_evidence_view_pins_fixed_source_outside_release(tmp_path):
    central=tmp_path/'central';central.mkdir();seed=tmp_path/'objects';sha=init(seed)
    release=central/'releases/fixed';release.parent.mkdir()
    subprocess.run(['git','clone','--shared',str(seed),str(release)],check=True,capture_output=True)
    config=fixture_config(tmp_path);config.update(code_root=str(release),code_source=sha,state_directory=str(central/'state'))
    for key in ('problem','round'):
        original=Path(config[key]);target=release/original.name;original.rename(target);config[key]=str(target)
    old=dict(config);before=inventory(release)
    view=central/'evidence-views/fixed';selected=prepare_evidence_service(config,seed,view,checkout_paths=['src'])
    assert config==old and inventory(release)==before
    gate=Gateway(selected,preflight=lambda *_:{'status':'PASS'},launch=lambda *_:pytest.fail('science launch'))
    result=gate.handle(dict(request(4,6),action='prepare'));run=Path(result['run_directory'])
    assert run.is_symlink() and run.resolve().is_relative_to(view) and not run.resolve().is_relative_to(release)
    pin=load(run/'CONTROLLER_CONFIG.json');assert pin['code_root']==str(view) and pin['code_source']==sha
    assert Path(pin['problem']).is_relative_to(view) and Path(pin['round']).is_relative_to(view)
    rel=(run.resolve()/'ROUND.requested.json').relative_to(view).as_posix()
    git(view,'config','user.name','fixture');git(view,'config','user.email','fixture@example.invalid')
    git(view,'add','--sparse','--',rel);git(view,'commit','-m','artificial locator')
    assert json.loads(git(view,'show','HEAD:'+rel))==load(run/'ROUND.requested.json')
    assert storage_dependency_inventory(central/'state')['entries']
    release.rename(central/'releases/retired')
    assert Path(pin['problem']).is_file() and Path(pin['round']).is_file()
    assert load(run/'CONTROLLER_CONFIG.json')['code_root']==str(view)
    assert git(view,'cat-file','-t',sha)=='commit'
    assert gate.handle(dict(action='status',run_id=run.name))['status']=='STATUS'


def setup_retained(tmp_path):
    config=fixture_config(tmp_path);old=tmp_path/'old-release';script=old/'src/trottertracks/pf_distributed_preparation.py';script.parent.mkdir(parents=True);script.write_text('# old controller without restore\n')
    config['code_root']=str(old)
    seen=[]
    def preflight(run,*_):
        assert (workspace/'src/file.py').read_text()=='fixed fixture\n'
        assert transfer.read_bytes()==original.read_bytes()
        seen.append('preflight');return {'status':'PASS'}
    gate=Gateway(config,preflight=preflight,launch=lambda *_:seen.append('launch'))
    run=Path(gate.handle(dict(request(4,6),action='prepare'))['run_directory'])
    state=Path(config['state_directory']);(state/'CURRENT_TARGET.json').unlink()
    write(run/'LAUNCH.json',{'invocation_id':'old'});write(run/'EXIT.json',{'invocation_id':'old','returncode':0})
    workspace=run/'pf_auto/candidates/one/pf_auto/pf_completion/provider_workspace';sha=init(workspace)
    write(workspace.parent/'PROVIDER_SOURCE.json',{'source':sha});(workspace/'src/local.py').write_text('unique')
    original=run/'jobs/FULL/job/TRANSPORT_TIMING.jsonl';original.parent.mkdir(parents=True);original.write_bytes(b'{"timing":1}\n')
    transfer=run/'distributed/transfers/job/attempt/TRANSPORT_TIMING.jsonl';transfer.parent.mkdir(parents=True);transfer.write_bytes(original.read_bytes())
    nonduplicate=transfer.with_name('JOB_STAGES.jsonl');nonduplicate.write_text('nonduplicate')
    receipt=transfer.with_name('PROCESS_RETURN.json');receipt.write_text('{"returncode":0}')
    db=run/'distributed/QUEUE.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE jobs(job_id TEXT,method TEXT,attempt_id TEXT,state TEXT)')
        conn.execute("INSERT INTO jobs VALUES('job','FULL','attempt','ACCEPTED')")
    return state,run,workspace,original,transfer,nonduplicate,receipt,gate,seen


def test_legacy_source_and_diagnostic_retirement_resume_compatibility(tmp_path):
    state,run,workspace,original,transfer,nonduplicate,receipt,gate,seen=setup_retained(tmp_path)
    before=inventory(workspace);fixed=load(run/'CONTROLLER_CONFIG.json');arch=tmp_path/'diagnostics.xz'
    assert compact_stopped_provider_sources(state,run.name)['compacted']==0
    assert compact_stopped_provider_sources(state,run.name,legacy_restore=True)['compacted']==1
    assert not (workspace/'src/file.py').exists() and (workspace/'src/local.py').read_text()=='unique'
    pack_stopped_diagnostics(state,run.name,arch)
    assert retire_stopped_diagnostics(state,run.name,arch)['retired']==1
    assert not transfer.exists() and original.exists() and nonduplicate.exists() and receipt.exists()
    assert gate.handle(dict(action='status',run_id=run.name))['status']=='STATUS'
    assert not transfer.exists()  # status remains non-mutating
    archive_bytes=arch.read_bytes();arch.write_bytes(archive_bytes+b'changed')
    resume=dict(action='resume',run_id=run.name,instruction='artificial layout recovery')
    with pytest.raises(RequestError,match='archive changed'):
        gate.handle(resume)
    assert seen==[] and not transfer.exists()
    arch.write_bytes(archive_bytes)
    transfer.write_text('conflicting fixture')
    with pytest.raises(RequestError,match='Retained diagnostic path differs'):
        gate.handle(resume)
    assert transfer.read_text()=='conflicting fixture' and seen==[]
    transfer.unlink()
    result=gate.handle(resume)
    assert result['status']=='STARTED' and seen==['preflight','launch']
    assert result['preflight']['saved_layout_recovery']['diagnostic_files']==1
    assert inventory(workspace)==before and load(run/'CONTROLLER_CONFIG.json')==fixed
    assert not (run/'RETAINED_DIAGNOSTICS.json').exists()
    assert transfer.read_bytes()==original.read_bytes()


def test_old_gateway_explicit_restore_keeps_unique_files_and_protects_owner(tmp_path):
    state,run,workspace,original,transfer,nonduplicate,receipt,gate,seen=setup_retained(tmp_path)
    arch=tmp_path/'diagnostics.xz'
    assert compact_stopped_provider_sources(state,run.name,legacy_restore=True)['compacted']==1
    pack_stopped_diagnostics(state,run.name,arch);retire_stopped_diagnostics(state,run.name,arch)
    write(state/'CURRENT_TARGET.json',{'run_id':run.name,'run_directory':str(run)})
    with pytest.raises(ValueError,match='No stopped registered run'):restore_stopped_layout(state,run.name)
    (state/'CURRENT_TARGET.json').unlink()
    assert restore_stopped_layout(state,run.name)=={'provider_workspaces':1,'diagnostic_files':1}
    assert (workspace/'src/file.py').read_text()=='fixed fixture\n' and transfer.read_bytes()==original.read_bytes()
    assert restore_stopped_layout(state,run.name)=={'provider_workspaces':0,'diagnostic_files':0}


def test_unsettled_attempt_blocks_legacy_retention(tmp_path):
    state,run,workspace,original,transfer,nonduplicate,receipt,gate,seen=setup_retained(tmp_path)
    with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
        db.execute("UPDATE jobs SET state='UNKNOWN'")
    result=compact_stopped_provider_sources(state,run.name,legacy_restore=True)
    assert result['compacted']==0 and 'Unsettled worker ownership' in result['skipped'][0]['reason']
    assert (workspace/'src/file.py').is_file() and not (run/'RETAINED_LEGACY_PROVIDER_SOURCE.json').exists()
