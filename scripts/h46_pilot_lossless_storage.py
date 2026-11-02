"""Opaque, streaming lossless evidence archive prototype; never deletes source.

No QPY/NPZ deserialization, chemistry, compilation, RNG, or array arithmetic.
This standalone prototype is NOT installed in any scientific runner/contract.
Original receipt hashes describe uncompressed bytes; archive hashes are separate.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib

BLOCK=1024*1024
SCHEMA='H46_OPAQUE_LOSSLESS_ARCHIVE_V1'


def require(value,reason):
    if not value:raise ValueError(reason)


def digest_file(path):
    h=hashlib.sha256();size=0
    with Path(path).open('rb') as f:
        while b:=f.read(BLOCK):h.update(b);size+=len(b)
    return dict(sha256=h.hexdigest(),bytes=size)


def _inflate(path,expected_raw,emit):
    """Bounded buffer and total-size checks; reject truncation/concatenation."""
    state=zlib.decompressobj();h=hashlib.sha256();size=0
    with Path(path).open('rb') as f:
        while block:=f.read(BLOCK):
            require(not state.eof,'TRAILING_ARCHIVE_BYTES')
            while block:
                b=state.decompress(block,BLOCK)
                size+=len(b);require(size<=expected_raw['bytes'],'DECODED_SIZE_EXCEEDED')
                h.update(b);emit(b)
                require(not state.unused_data,'TRAILING_ARCHIVE_BYTES')
                block=state.unconsumed_tail
        require(state.eof,'TRUNCATED_ARCHIVE')
    require(size==expected_raw['bytes'] and h.hexdigest()==expected_raw['sha256'],'RAW_BYTES_OR_SHA_MISMATCH')
    return dict(sha256=h.hexdigest(),bytes=size)


def verify_archive(path,receipt):
    require(receipt['schema']==SCHEMA and receipt['codec']=='zlib_deflate','ARCHIVE_FORMAT')
    require(receipt.get('lossy') is False,'LOSSLESS_REQUIRED')
    require(digest_file(path)==receipt['stored'],'STORED_BYTES_OR_SHA_MISMATCH')
    return _inflate(path,receipt['original'],lambda b:None)


def archive_file(source,destination,expected_original,*,level=1,min_free_bytes=0):
    """Copy one immutable original into a NEW archive and verify raw byte SHA.

    Destination and .partial must not preexist. Failures preserve source and any
    diagnostic partial. No auto-retry, overwrite, delete-original, or hardlink
    substitution of scientific inputs occurs.
    """
    source=Path(source).resolve();destination=Path(destination).resolve()
    require(source!=destination,'INPUT_OUTPUT_DISJOINT')
    require(type(level) is int and 0<=level<=9,'CODEC_LEVEL')
    require(type(min_free_bytes) is int and min_free_bytes>=0,'FREE_SPACE_FLOOR')
    require(type(expected_original['bytes']) is int and expected_original['bytes']>=0,'RAW_SIZE')
    require(len(expected_original['sha256'])==64,'RAW_SHA')
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.partial')
    require(not destination.exists() and not partial.exists(),'DESTINATION_ALREADY_USED')
    before=source.stat();require(before.st_size==expected_original['bytes'],'ORIGINAL_SIZE')
    rawhash=hashlib.sha256();rawsize=0;storedhash=hashlib.sha256();storedsize=0
    compressor=zlib.compressobj(level)
    with source.open('rb') as inp,partial.open('xb') as out:
        def store(b):
            nonlocal storedsize
            require(shutil.disk_usage(destination.parent).free>=min_free_bytes+len(b),'SCRATCH_HEADROOM_LOW_NO_RETRY')
            out.write(b);storedhash.update(b);storedsize+=len(b)
        while block:=inp.read(BLOCK):
            rawhash.update(block);rawsize+=len(block);store(compressor.compress(block))
        store(compressor.flush());out.flush();os.fsync(out.fileno())
    after=source.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'ORIGINAL_CHANGED_DURING_COPY')
    require(dict(sha256=rawhash.hexdigest(),bytes=rawsize)==expected_original,'ORIGINAL_SHA')
    receipt=dict(schema=SCHEMA,codec='zlib_deflate',level=level,zlib_version=zlib.ZLIB_VERSION,
       lossy=False,original=expected_original,stored=dict(sha256=storedhash.hexdigest(),bytes=storedsize),
       byte_restoration_verified=True,source_deleted=False,source_modified=False,
       scientific_adoption=False,operator_or_native_validation_reperformed=False)
    # Verify before publishing. link is atomic/no-overwrite; only our temporary
    # compressed file is unlinked. The original is never touched.
    verify_archive(partial,receipt)
    os.link(partial,destination);partial.unlink()
    return receipt


def restore_new(path,receipt,destination):
    """Restore byte-identical original to a NEW path. No deserialization."""
    path=Path(path).resolve();destination=Path(destination).resolve()
    require(path!=destination,'INPUT_OUTPUT_DISJOINT')
    require(not destination.exists(),'RESTORE_DESTINATION_ALREADY_EXISTS')
    verify_archive(path,receipt)
    partial=destination.with_name(destination.name+'.partial')
    require(not partial.exists(),'RESTORE_PARTIAL_EXISTS')
    with partial.open('xb') as f:
        _inflate(path,receipt['original'],f.write);f.flush();os.fsync(f.fileno())
    os.link(partial,destination);partial.unlink()
    return receipt['original']


def archive_writer(destination,writer,*,level=1,min_free_bytes=0):
    """Serialize into a forward-only compressed sink, not a full raw disk file.

    Prototype for future QPY output. It is NOT bound to the existing native
    adapter. Serializers requiring seek must fail and use a separately budgeted
    raw temporary. Receipt raw SHA still identifies the exact serializer bytes.
    """
    destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.partial')
    require(not destination.exists() and not partial.exists(),'DESTINATION_ALREADY_USED')
    require(type(level) is int and 0<=level<=9,'CODEC_LEVEL')
    require(type(min_free_bytes) is int and min_free_bytes>=0,'FREE_SPACE_FLOOR')
    raw=hashlib.sha256();stored=hashlib.sha256();rawsize=storedsize=0;compressor=zlib.compressobj(level)
    with partial.open('xb') as out:
        def emit(b):
            nonlocal storedsize
            require(shutil.disk_usage(destination.parent).free>=min_free_bytes+len(b),'SCRATCH_HEADROOM_LOW_NO_RETRY')
            out.write(b);stored.update(b);storedsize+=len(b)
        class Sink:
            def write(self,block):
                nonlocal rawsize
                block=memoryview(block);n=len(block)
                for offset in range(0,n,BLOCK):
                    b=block[offset:offset+BLOCK];raw.update(b);rawsize+=len(b);emit(compressor.compress(b))
                return n
            def tell(self):return rawsize
            def flush(self):out.flush()
            def writable(self):return True
            def seekable(self):return False
            def seek(self,*args):raise ValueError('SERIALIZER_SEEK_REQUIRES_SEPARATE_RAW_SCRATCH_POLICY')
        writer(Sink());emit(compressor.flush());out.flush();os.fsync(out.fileno())
    receipt=dict(schema=SCHEMA,codec='zlib_deflate',level=level,zlib_version=zlib.ZLIB_VERSION,lossy=False,
      original=dict(sha256=raw.hexdigest(),bytes=rawsize),stored=dict(sha256=stored.hexdigest(),bytes=storedsize),
      byte_restoration_verified=True,source_deleted=False,source_modified=False,scientific_adoption=False,
      operator_or_native_validation_reperformed=False,full_raw_disk_temporary=False)
    verify_archive(partial,receipt);os.link(partial,destination);partial.unlink()
    return receipt


def planning_case(systems,*,n_each,raw_buffers,compressed_factor,metadata_reserve_per_trajectory,
                  recovery_raw_pair_buffers=1):
    """Storage arithmetic only. Explicit proposals, never output upper bounds.

    systems must be from measured *byte compression*, not scientific cost data.
    raw_buffers maps simultaneous workers to whole raw pair budget (conservative
    post-compress design). Direct-stream design does not require a full raw pair
    buffer, but is intentionally not claimed implemented here.
    """
    require(type(n_each) is int and n_each>=2,'PILOT_N')
    require(compressed_factor>=1 and metadata_reserve_per_trajectory>=0,'PLANNING_FACTOR')
    require(set(raw_buffers)==set(systems) and all(type(v) is int and v>=0 for v in raw_buffers.values()),'RAW_BUFFER_COUNTS')
    count=12*n_each
    final={s:count*(compressed_factor*r['max_measured_compressed_pair_bytes']+
                    metadata_reserve_per_trajectory) for s,r in systems.items()}
    raw_scratch={s:raw_buffers[s]*r['raw_pair_proxy_bytes'] for s,r in systems.items()}
    recovery=recovery_raw_pair_buffers*max(r['raw_pair_proxy_bytes'] for r in systems.values())
    return dict(n_each=n_each,trajectories_per_system=count,final_growth_bytes=final,
      retained_existing_results_deleted=0,raw_worker_scratch_bytes=raw_scratch,
      parallel_peak_growth_bytes=sum(final.values())+sum(raw_scratch.values())+recovery,
      sequential_peak_growth_bytes=sum(final.values())+max(raw_scratch.values())+recovery,
      restoration_reserve_bytes=recovery,compressed_factor_proposed=compressed_factor,
      metadata_reserve_per_trajectory_proposed=metadata_reserve_per_trajectory,
      empirical_DET_compression_only=True,PR_or_rare_output_upper_bound=None,
      scientific_storage_policy_adopted=False)
