#!/usr/bin/env python3
"""Compress redundant *reachable or unreachable* loose Git objects locally.

Every selected object remains at the identical Git object ID. No history,
refs, alternates, scientific files, or unreachable objects are pruned.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zlib

from trottertracks.rpe_q3_coordinator import atomic_json

GIB = 1024**3


def compact(repository, state, *, min_age=86400, minimum_object_bytes=10*1024**2,
            batch_bytes=2*GIB, pressure_fraction=0.20, force=False):
    repository, state = Path(repository).resolve(), Path(state).resolve()
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1',
               GIT_ALLOW_PROTOCOL='', GIT_TERMINAL_PROMPT='0')
    def git(*args, **kwargs):
        return subprocess.run(['git','-C',str(repository),*args], env=env, check=True,
                              **kwargs)
    remote = git('remote','get-url','origin', text=True, capture_output=True).stdout.strip()
    if not remote.startswith(('git@github.com:HIROMU1015/','https://github.com/HIROMU1015/')):
        return {'status':'NOT_OWNED_RESEARCH_REPOSITORY'}
    directory = Path(git('rev-parse','--git-common-dir',text=True,capture_output=True).stdout.strip())
    if not directory.is_absolute():
        directory = repository/directory
    objects = directory.resolve()/'objects'
    if not objects.is_dir() or objects.is_symlink():
        return {'status':'NO_LOCAL_OBJECT_STORE'}
    state.mkdir(parents=True,exist_ok=True)
    with (state/'GIT_COMPACTION.lock').open('a+b') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'ALREADY_COMPACTING'}
        disk = shutil.disk_usage(objects)
        if not force and disk.free >= disk.total*pressure_fraction:
            return {'status':'ENOUGH_DISK_SPACE','free_bytes':disk.free}
        rejected_path = state/'GIT_COMPACTION_NO_SAVINGS.json'
        try:rejected=set(json.loads(rejected_path.read_text()))
        except (OSError,ValueError):rejected=set()
        groups=defaultdict(list)
        cutoff=time.time()-min_age
        for shard in objects.iterdir():
            if not re.fullmatch('[0-9a-f]{2}',shard.name) or not shard.is_dir() or shard.is_symlink():continue
            for path in shard.iterdir():
                oid=shard.name+path.name
                if not re.fullmatch('[0-9a-f]{40}',oid) or oid in rejected:continue
                try:
                    info=path.lstat()
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size<minimum_object_bytes or info.st_mtime>cutoff:continue
                    with path.open('rb') as stream:raw=zlib.decompressobj().decompress(stream.read(65536),100)
                    header,_,body=raw.partition(b'\0')
                    if not header.startswith(b'blob '):continue
                    kind='bundle' if body.startswith(b'# v') else 'pack' if body.startswith(b'PACK') else 'json' if body.startswith((b'{',b'[')) else 'other'
                    groups[kind].append((info.st_size,info.st_blocks*512,oid,path))
                except (OSError,zlib.error):continue
        if not groups:return {'status':'NO_OLD_LARGE_LOOSE_OBJECTS'}
        group=max(groups.values(),key=lambda rows:sum(row[1] for row in rows))
        rows=[];size=0
        for row in sorted(group,reverse=True):
            if rows and size+row[0]>batch_bytes:break
            rows.append(row);size+=row[0]
        if len(rows)<2:return {'status':'NO_SIMILAR_BATCH'}
        # Two simultaneous copies of the new pack plus a little headroom;
        # refuse to make disk pressure worse when there is not enough space.
        if disk.free < 2*size+GIB:return {'status':'PACK_WORKSPACE_LOW','free_bytes':disk.free}
        ids=[r[2] for r in rows];original=sum(r[1] for r in rows)
        with tempfile.TemporaryDirectory(prefix='git-pack-',dir=state) as work:
            pack=Path(work)/'new.pack';idx=pack.with_suffix('.idx')
            with pack.open('wb') as output:
                subprocess.run(['nice','-n','10','git','-C',str(repository),'pack-objects','--stdout',
                                '--window=5','--depth=10','--threads=1','--window-memory=1g','--compression=1'],
                               env=env,input=('\n'.join(ids)+'\n').encode(),stdout=output,
                               stderr=subprocess.PIPE,check=True,timeout=600)
                output.flush();os.fsync(output.fileno())
            packed=pack.stat().st_size
            if packed>=original*.95:
                atomic_json(rejected_path,sorted(rejected|set(ids)))
                return {'status':'NO_MATERIAL_SAVINGS','objects':len(ids),'old_bytes':original,'packed_bytes':packed}
            key=git('index-pack','--strict','-o',str(idx),str(pack),text=True,capture_output=True,timeout=600).stdout.strip()
            git('verify-pack',str(idx),capture_output=True,timeout=600)
            with idx.open('rb') as stream:
                observed={line.split()[1] for line in git('show-index',stdin=stream,capture_output=True).stdout.decode().splitlines()}
            if observed!=set(ids):raise ValueError('New pack does not contain exactly the selected Git objects')
            with pack.open('rb') as stream:
                installed=git('index-pack','--stdin','--strict',stdin=stream,text=True,capture_output=True,timeout=600).stdout.split()[-1]
            if installed!=key:raise ValueError('Installed pack differs from verified pack')
            installed_index=objects/'pack'/('pack-'+key+'.idx')
            with installed_index.open('rb') as stream:
                after={line.split()[1] for line in git('show-index',stdin=stream,capture_output=True).stdout.decode().splitlines()}
            if after!=observed:raise ValueError('Installed pack index differs')
            before={oid:p.stat().st_blocks*512 for _,_,oid,p in rows if p.is_file()}
            # prune-packed only removes exact redundant loose copies, not
            # unreachable Git history; never invoke git prune or gc here.
            git('prune-packed','-q',timeout=120)
            check=git('cat-file','--batch-check',input='\n'.join(ids)+'\n',text=True,capture_output=True,timeout=120).stdout.splitlines()
            if len(check)!=len(ids) or any(len(line.split())!=3 or line.split()[0]!=oid for oid,line in zip(ids,check)):
                raise ValueError('Git object not readable after compaction')
            freed=sum(before.get(oid,0) for _,_,oid,p in rows if not p.exists())-packed
            return {'status':'PACKED_VERIFIED','objects':len(ids),'old_bytes':original,'packed_bytes':packed,
                    'reclaimed_bytes':freed,'pack':key,'free_bytes':shutil.disk_usage(objects).free}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--state-directory',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(compact(args.repository,args.state_directory)))

if __name__=='__main__':main()
