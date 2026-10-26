"""Losslessly compress completed private event logs; leave active trials alone.

The observed event-log size projects to >200GB for exploration alone, before
confirmation. This saves scratch space without changing any scientific bytes.
"""
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import time
import h46_b3 as b


def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def compress_completed():
    count=0;original=0;compressed=0
    for p in (b.PRIVATE/'cost').glob('*/*/*/events_PRIVATE.json'):
        if not (p.parent/'result.json').exists() or (p.parent/'STOP.json').exists(): continue
        result=b.read(p.parent/'result.json')
        if result['status']!='PASS': continue
        archive=p.with_suffix('.json.gz');temp=archive.with_name(archive.name+'.partial')
        with p.open('rb') as source,gzip.open(temp,'wb',compresslevel=1) as target:
            shutil.copyfileobj(source,target,2**20)
        with p.open('rb') as source,gzip.open(temp,'rb') as restored:
            if digest(source)!=digest(restored): raise ValueError('EVENT_ARCHIVE_BYTE_MISMATCH')
        original+=p.stat().st_size;compressed+=temp.stat().st_size
        temp.replace(archive);p.unlink();count+=1
    return dict(files=count,original_bytes=original,compressed_bytes=compressed)


def inventory():
    """Recover counts even if a previous controller vanished between saves."""
    files=0;original=0;compressed=0
    for p in (b.PRIVATE/'cost').glob('*/*/*/events_PRIVATE.json.gz'):
        if not (p.parent/'result.json').exists() or (p.parent/'STOP.json').exists(): continue
        if b.read(p.parent/'result.json')['status']!='PASS': continue
        with p.open('rb') as stream:
            stream.seek(-4,2);original+=int.from_bytes(stream.read(4),'little')
        files+=1;compressed+=p.stat().st_size
    return dict(files=files,original_bytes=original,compressed_bytes=compressed,
        gzip_ISIZE_is_modulo_2pow32_per_file=True,roundtrip_verified=True,
        active_trials_modified=False,scientific_bytes_changed=False)


def run():
    while True:
        r=compress_completed()
        total=inventory()
        b.save(b.PRIVATE/'event_archive_summary.json',total)
        if (b.PUBLIC/'stages_complete.json').exists():
            # One last scan catches a result written after this scan passed
            # its directory. Already archived files are never recomputed.
            if compress_completed()['files']:
                continue
            total=inventory();break
        time.sleep(30)
    b.save(b.PUBLIC/'event_archive_summary.json',total)


if __name__=='__main__': run()
