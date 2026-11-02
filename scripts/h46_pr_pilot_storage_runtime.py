"""Durable opaque-byte storage for NEW approved DESIGN_PILOT artifacts only.

No scientific replay or retry. A committed manifest can recover byte storage
after a crash, but cannot turn interrupted scientific work into a PASS.
"""
import contextlib
import json
import os
from pathlib import Path
import shutil
import h46_pilot_lossless_storage as codec

POLICY='H46_DESIGN_PILOT_BYTE_ARCHIVE_LEVEL1_V1'
require=codec.require

def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def save(path,value):
    path=Path(path);partial=path.with_name(path.name+'.partial')
    with partial.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(partial,path);sync_dir(path.parent);partial.unlink();sync_dir(path.parent)

def read(path):return json.loads(Path(path).read_text())

def storage_policy():
    return dict(policy=POLICY,codec='zlib_deflate',level=1,lossy=False,
      raw_release='ONLY_NEW_RUN_OWN_RAW_AFTER_DURABLE_MANIFEST_AND_VERIFIED_ARCHIVE',
      existing_raw_changes=False,raw_SHA_is_identity=True,archive_SHA_separate=True,
      native_validation_separate=True,crash_recovery='BYTE_AUDIT_ONLY_NO_SCIENCE_RETRY',
      planning_compressed_factor=4,metadata_bytes_per_trajectory=16*1024*1024,
      planning_proxy_is_upper_bound=False,output_truncation=False)

def own_path(cfg,path):
    """Reject symlinks, outside outputs, preexisting inputs, or unknown names."""
    path=Path(path);output=Path(cfg['output'])
    require(path.is_absolute() and output.is_absolute(),'ABSOLUTE_NEW_OUTPUT')
    require(path==path.resolve() and output==output.resolve(),'NO_OUTPUT_SYMLINK')
    rel=path.relative_to(output)
    require(len(rel.parts)==3 and rel.parts[0] in {t['id'] for t in cfg['tasks']}
      and rel.parts[1] in ('quarantine_cosine','quarantine_sine')
      and rel.parts[2] in ('wrapper.qpy','compiled.qpy','vectors.npz'),'NEW_RAW_SCOPE_ONLY')
    require(cfg['storage']==storage_policy(),'APPROVED_BYTE_STORAGE_POLICY')
    marker=read(cfg['marker'])
    require(marker==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),
            'STORAGE_REQUIRES_CONSUMED_OWN_GRANT')
    return path,rel

def manifest_for(cfg,path,*,verify=True):
    path,rel=own_path(cfg,path);mp=path.with_name(path.name+'.archive.json')
    m=read(mp)
    require(m['policy']==POLICY and m['SOURCE']==cfg['SOURCE']
      and m['plan_sha256']==cfg['plan_sha256'] and m['stage']==cfg['stage']
      and m['logical_path']==str(rel) and m['task_id']==rel.parts[0],
      'ARCHIVE_MANIFEST_BINDING')
    require(m['archive_file']==path.name+'.zlib','ARCHIVE_LOCAL_PATH')
    require(m['codec_receipt']['level']==1 and m['byte_storage_PASS'] is True
      and m['native_scientific_PASS'] is None,'BYTE_PASS_NOT_NATIVE_PASS')
    ap=path.with_name(m['archive_file'])
    require(ap==ap.resolve() and ap.is_file(),'ARCHIVE_FILE_REQUIRED')
    if verify:codec.verify_archive(ap,m['codec_receipt'])
    return m,ap

def release_new_raw(cfg,path,*,fault=None):
    """Idempotent byte-storage recovery; no recompile/resample/validation."""
    path,_=own_path(cfg,path);m,ap=manifest_for(cfg,path)
    sync_dir(ap.parent)
    if path.exists():
        require(codec.digest_file(path)==m['codec_receipt']['original'],'RAW_CHANGED_BEFORE_RELEASE')
        if fault:fault('before_raw_release')
        path.unlink();sync_dir(path.parent)
    rp=path.with_name(path.name+'.released.json')
    expected=dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
      manifest_sha256=codec.digest_file(path.with_name(path.name+'.archive.json'))['sha256'],
      new_raw_absent=True,archive_verified=True,scientific_retry=False)
    if rp.exists():require(read(rp)==expected,'RELEASE_RECEIPT_IDENTITY')
    else:save(rp,expected)
    return m

def write_binary(cfg,path,writer,*,fault=None):
    path,rel=own_path(cfg,path);partial=path.with_name(path.name+'.raw.partial')
    require(not any(p.exists() for p in (path,partial,path.with_name(path.name+'.zlib'),
      path.with_name(path.name+'.archive.json'))),'NEW_ARTIFACT_ONLY_NO_RETRY')
    # A complete raw temporary is budgeted per worker. NPZ bytes are preserved,
    # never reserialized. Failed/partial files remain quarantined for audit.
    with partial.open('xb') as f:
        writer(f);f.flush();os.fsync(f.fileno())
    os.link(partial,path);sync_dir(path.parent);partial.unlink();sync_dir(path.parent)
    if fault:fault('raw_durable')
    raw=codec.digest_file(path)
    receipt=codec.archive_file(path,path.with_name(path.name+'.zlib'),raw,level=1)
    sync_dir(path.parent)
    if fault:fault('archive_durable')
    m=dict(policy=POLICY,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],stage=cfg['stage'],
      task_id=rel.parts[0],logical_path=str(rel),archive_file=path.name+'.zlib',
      codec_receipt=receipt,byte_storage_PASS=True,native_scientific_PASS=None,
      new_raw_release_authorized=True,old_raw_untouched=True)
    save(path.with_name(path.name+'.archive.json'),m)
    if fault:fault('manifest_durable')
    release_new_raw(cfg,path,fault=fault)
    return dict(file=path.name,**raw,archive=path.name+'.zlib',archive_sha256=receipt['stored']['sha256'],
      stored_bytes=receipt['stored']['bytes'],manifest=path.name+'.archive.json',byte_restoration_PASS=True)

@contextlib.contextmanager
def resolve(cfg,logical,destination):
    """Restore only into a NEW dedicated restoration area, never original path."""
    logical,_=own_path(cfg,logical);m,archive=manifest_for(cfg,logical)
    destination=Path(destination)
    rr=Path(cfg['run_root'])/'restoration'
    require(destination==destination.resolve() and destination.is_relative_to(rr)
      and not destination.exists(),'NEW_RESTORATION_DESTINATION_ONLY')
    destination.parent.mkdir(parents=True,exist_ok=True)
    codec.restore_new(archive,m['codec_receipt'],destination);sync_dir(destination.parent)
    try:yield destination
    finally:
        # Restored derivative, never a historical original. Archive remains.
        require(codec.digest_file(destination)==m['codec_receipt']['original'],'RESTORATION_CHANGED')
        destination.unlink();sync_dir(destination.parent)

def audit_task(cfg,task,*,verify=True):
    root=Path(cfg['output'])/task['id'];rows=[]
    for axis in ('cosine','sine'):
        for name in ('wrapper.qpy','compiled.qpy','vectors.npz'):
            path=root/('quarantine_'+axis)/name
            m,ap=manifest_for(cfg,path,verify=verify)
            require(not path.exists() and path.with_name(path.name+'.released.json').exists(),
                    'NEW_RAW_RELEASE_NOT_AUDITED')
            release=read(path.with_name(path.name+'.released.json'))
            require(release['manifest_sha256']==codec.digest_file(path.with_name(path.name+'.archive.json'))['sha256']
              and release['new_raw_absent'] and release['archive_verified'],'RELEASE_MANIFEST_SHA')
            rows.append(dict(logical=m['logical_path'],original=m['codec_receipt']['original'],
              stored=m['codec_receipt']['stored'],archive=ap.name))
    require(not list(root.rglob('*.partial')),'UNCOMMITTED_STORAGE_PARTIAL')
    return dict(status='BYTE_STORAGE_PASS_NOT_SCIENTIFIC_VALIDATION',archives=rows,
      original_bytes=sum(r['original']['bytes'] for r in rows),
      stored_bytes=sum(r['stored']['bytes'] for r in rows),all_raw_SHA_verified=verify,
      archive_count=6,original_raw_retained=False)

def allocation(idle_cpu,available_ram,tasks,proxies,demand):
    """Balance predicted completion using saved timing proxies; no fixed cap."""
    import math
    cpu=math.floor(idle_cpu);systems=tuple(tasks);best=None
    require(len(systems)==2,'TWO_INDEPENDENT_STAGES')
    a,b=systems
    for wa in range(1,min(tasks[a],cpu-1,available_ram//proxies[a])+1):
        wb=min(tasks[b],cpu-wa,(available_ram-wa*proxies[a])//proxies[b])
        if wb<1:continue
        score=(max(demand[a]/wa,demand[b]/wb),-(wa+wb))
        if best is None or score<best[0]:best=(score,{a:wa,b:wb})
    require(best is not None,'JOINT_RESOURCES_CANNOT_ADMIT_BOTH')
    return best[1]
