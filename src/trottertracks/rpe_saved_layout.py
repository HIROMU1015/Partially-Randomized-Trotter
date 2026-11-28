"""Rehydrate stopped-run retention layouts before the ordinary Resume checks.

Only explicitly retained, byte-verified diagnostic and clean provider paths
are restored. No scientific result, ownership, STOP or budget is rewritten.
"""
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import tempfile

from .rpe_worker_storage import restore_provider_source

DIAGNOSTICS = frozenset(('TRANSPORT_TIMING.jsonl','JOB_STAGES.jsonl','WORKER_COLLECT_RETURN.json'))


def provider_workspaces(run):
    run=Path(run).resolve(strict=True);pf=run/'pf_auto'
    for candidates in (pf/'candidates',*pf.glob('price-recovery/recheck-*/candidates')):
        for workspace in candidates.glob('*/pf_auto/pf_completion/provider_workspace'):
            if workspace.resolve()!=workspace:
                raise ValueError('Provider workspace is not local')
            yield workspace


def restore_providers(run):
    restored=0
    for workspace in provider_workspaces(run):
        marker=workspace.parent/'RETAINED_PROVIDER_SOURCE.json'
        if marker.is_symlink():raise ValueError('Invalid retained provider marker')
        if marker.is_file():
            restore_provider_source(workspace)
            restored+=1
    return restored


def diagnostic_path(name):
    path=Path(name);parts=path.parts
    if (path.is_absolute() or '..' in parts or not parts or parts[-1] not in DIAGNOSTICS
            or not (len(parts)==4 and parts[0]=='jobs' or len(parts)==5 and parts[:2]==('distributed','transfers'))
            or path.as_posix()!=name):
        raise ValueError('Invalid diagnostic archive path')
    return path


def diagnostic_rows(archive, run_id):
    seen=set()
    with lzma.open(archive,'rt',encoding='utf-8') as reader:
        if json.loads(reader.readline())!={'schema':'RPE_DIAGNOSTIC_ARCHIVE_V1','run_id':run_id}:
            raise ValueError('Diagnostic archive identity differs')
        for line in reader:
            row=json.loads(line);relative=diagnostic_path(row['path'])
            if relative in seen:raise ValueError('Duplicate diagnostic archive path')
            raw=base64.b64decode(row['data'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Diagnostic archive digest differs')
            seen.add(relative)
            yield relative,row['sha256'],raw


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def restore_retained_diagnostics(run):
    """Restore only explicitly removed transfer diagnostics, before Resume."""
    run=Path(run).resolve(strict=True);marker=run/'RETAINED_DIAGNOSTICS.json'
    if marker.is_symlink():raise ValueError('Invalid retained diagnostic marker')
    if not marker.exists():return 0
    record=json.loads(marker.read_bytes())
    if record.get('schema')!='RPE_RETAINED_DIAGNOSTICS_V1' or record.get('run_id')!=run.name:
        raise ValueError('Retained diagnostic layout differs')
    archive=Path(record['archive'])
    if not archive.is_absolute() or archive.is_symlink() or file_sha(archive)!=record['archive_sha256']:
        raise ValueError('Retained diagnostic archive changed')
    selected={diagnostic_path(name):sha for name,sha in record['files'].items()}
    if any(path.parts[:2]!=('distributed','transfers') for path in selected):
        raise ValueError('Only transfer diagnostics may be retained')
    # Validate the entire archive and all existing paths before restoring any.
    found=set()
    for relative,digest,raw in diagnostic_rows(archive,run.name):
        if relative not in selected:continue
        path=run/relative
        if digest!=selected[relative] or path.resolve()!=path or (path.exists() and file_sha(path)!=digest):
            raise ValueError('Retained diagnostic path differs: '+str(relative))
        found.add(relative)
    if found!=set(selected):raise ValueError('Retained diagnostic archive incomplete')
    restored=0
    for relative,digest,raw in diagnostic_rows(archive,run.name):
        if relative not in selected:continue
        path=run/relative
        if path.exists():continue
        if path.resolve()!=path:raise ValueError('Diagnostic restore escapes original path')
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.restore-',delete=False) as out:
            temporary=Path(out.name)
            try:
                out.write(raw);out.flush();os.fsync(out.fileno())
                os.link(temporary,path);restored+=1
            finally:temporary.unlink(missing_ok=True)
        sync_directory(path.parent)
    if any((run/path).resolve()!=run/path or file_sha(run/path)!=digest for path,digest in selected.items()):
        raise ValueError('Restored diagnostic bytes differ')
    for parent in { (run/path).parent for path in selected }:
        sync_directory(parent)
    marker.unlink()
    sync_directory(run)
    return restored


def restore_saved_layout(run):
    run=Path(run).resolve(strict=True)
    marker=run/'RETAINED_LEGACY_PROVIDER_SOURCE.json'
    providers=0
    if marker.is_symlink():raise ValueError('Invalid legacy provider retention marker')
    if marker.exists():
        if json.loads(marker.read_bytes())!={'run_id':run.name}:
            raise ValueError('Legacy provider retention identity differs')
        providers=restore_providers(run)
        marker.unlink()
        sync_directory(run)
    return dict(provider_workspaces=providers,diagnostic_files=restore_retained_diagnostics(run))
