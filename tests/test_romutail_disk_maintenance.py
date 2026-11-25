"""Storage-only fixtures: no scientific work or scheduler activity."""
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess

import pytest

from compact_rpe_git_objects import compact
import rpe_romutail_cold_storage as cold


def call_git(root, *args, **kwargs):
    return subprocess.check_output(['git','-C',str(root),*args], **kwargs)


def test_compact_shared_transport_stores_keep_exact_objects(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    call_git(repo,'init','-q')
    call_git(repo,'remote','add','origin','git@github.com:HIROMU1015/synthetic.git')
    common = random.Random(17).randbytes(256*1024)
    original = {}
    for i in range(5):
        raw = common+bytes([i]); oid = call_git(repo,'hash-object','-w','--stdin',input=raw).decode().strip()
        original[oid] = raw
    objects = repo/'.git/objects'
    stores=[]
    for name in ('first','second'):
        base=repo/'.rpe-central/transport-flat'/name
        base.mkdir(parents=True)
        call_git(base,'init','--bare','-q','objects.git')
        (base/'TRANSPORT.json').write_text(json.dumps({'alternate_depth':1,'paths':{'old':str(base/'checkouts/sample')}}))
        store=base/'objects.git/objects'
        stores.append(store)
        for oid in original:
            link=store/oid[:2]/oid[2:]; link.parent.mkdir(exist_ok=True)
            os.link(objects/oid[:2]/oid[2:],link)
    unshared=next(iter(original))
    os.link(objects/unshared[:2]/unshared[2:],tmp_path/'external')
    report=compact(repo,tmp_path/'maintenance',min_age=0,minimum_object_bytes=1,force=True,min_free=0,workers=2)
    assert report['status']=='COMPLETE'
    assert report['reclaimed_bytes'] > 0
    assert report['batches'][0]['stores']==3
    for oid,raw in original.items():
        assert call_git(repo,'cat-file','blob',oid)==raw
        for store in stores:
            assert call_git(store.parent,'cat-file','blob',oid)==raw
            assert (store/oid[:2]/oid[2:]).exists()==(oid==unshared)
    pack=next((objects/'pack').glob('*.pack'))
    assert all((store/'pack'/pack.name).stat().st_ino==pack.stat().st_ino for store in stores)


@pytest.mark.parametrize("manifest_kind", ["current", "legacy_list", "legacy_dict"])
def test_archive_ended_cache_and_restore_on_resume(tmp_path, monkeypatch, manifest_kind):
    state=tmp_path/'central/state'; state.mkdir(parents=True)
    (state/'REGISTRY.lock').touch()
    (state/'CURRENT_TARGET.json').write_text('{"run_id":"different"}')
    run=state/'runs/test-run'; source=run/'distributed/artifacts'; source.mkdir(parents=True)
    launch={'invocation_id':'finished'}
    for name in ('LAUNCH.json','EXIT.json'):(run/name).write_text(json.dumps(launch))
    with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
        db.execute('create table jobs (state text)')
        db.execute("insert into jobs values ('RUNNING')")
    raw=b'fixed artifact'; sha=hashlib.sha256(raw).hexdigest(); path=source/sha[:2]/sha
    path.parent.mkdir();path.write_bytes(raw)
    target=tmp_path/'archive';target.mkdir()
    assert cold.archive_one(state,run,target,min_age=0,minimum_bytes=1) is None
    with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:db.execute("update jobs set state='ACCEPTED'")
    result=cold.archive_one(state,run,target,min_age=0,minimum_bytes=1)
    assert result['state']=='MOVED_VERIFIED' and source.is_symlink() and path.read_bytes()==raw
    if manifest_kind != 'current':
        marker=source.resolve().parent/'MANIFEST.json'
        record=json.loads(marker.read_text()); entries=record.pop('entries')
        if manifest_kind == 'legacy_list':
            record['files_manifest']=entries
        else:
            record['files']={entry['path']:entry for entry in entries}
            record['target']=record.pop('archive'); record.pop('state');record['verified']=True
        marker.write_text(json.dumps(record))
    monkeypatch.setattr(cold,'mounted_separately',lambda *_:True)
    result=cold.restore_for_resume(run,target)
    assert result['status']=='RESTORED_FROM_COLD_STORAGE' and not source.is_symlink() and path.read_bytes()==raw
    assert cold.restore_for_resume(run,target)['status']=='LOCAL_CACHE'
