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
        if not (p.parent/'result.json').exists(): continue
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


def run():
    total=dict(files=0,original_bytes=0,compressed_bytes=0,roundtrip_verified=True,
        active_trials_modified=False,scientific_bytes_changed=False)
    while True:
        r=compress_completed()
        for k,v in r.items(): total[k]+=v
        b.save(b.PRIVATE/'event_archive_summary.json',total)
        if (b.PUBLIC/'stages_complete.json').exists(): break
        time.sleep(30)
    b.save(b.PUBLIC/'event_archive_summary.json',total)


if __name__=='__main__': run()
