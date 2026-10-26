"""Persist this study's completed data before reporting a task as finished.

Added after two event archives and one result were found empty following loss
of the former controller. This writes no approval records or shared settings.
"""
import os
from pathlib import Path
import h46_b3 as b

original_save=b.save


def sync_file(path):
    with Path(path).open('rb') as stream: os.fsync(stream.fileno())


def sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def save(path,value):
    path=Path(path)
    temporary=path.with_name('.durable.'+str(os.getpid())+'.'+path.name)
    original_save(temporary,value)
    if path.parent.name=='compiled_cache' and path.suffix=='.json':
        for suffix in ('_wrapper.qpy.gz','_compiled.qpy.gz'):
            qpy=path.with_name(path.stem+suffix)
            if qpy.exists(): sync_file(qpy)
    sync_file(temporary);temporary.replace(path);sync_directory(path.parent)
