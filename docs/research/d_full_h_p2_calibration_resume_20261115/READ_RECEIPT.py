"""Read lossless published receipts and bind D1 evidence; no scientific execution.

C's existing evidence reader accepts the small JSON records. The raw event receipt
is gzip (possibly split to respect GitHub's file limit); decode it explicitly here.
"""
from copy import deepcopy
from pathlib import Path
import gzip
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_full_h_kappa2_p2 as P
import revalidate_full_h_kappa2_p1_run02 as R


def archive_result(directory, part_bytes=64*2**20):
    """Preserve raw bytes; verify lossless gzip and publish bounded byte parts."""
    directory=Path(directory).resolve();directory.relative_to(ROOT)
    raw_path=directory/'RESULT.json';zpath=directory/'RESULT.json.gz'
    manifest_path=directory/'RAW_RECEIPT.json'
    if manifest_path.exists() or zpath.exists():raise FileExistsError('Receipt already archived')
    digest=hashlib.sha256();size=0
    with raw_path.open('rb') as source,zpath.open('xb') as target:
        with gzip.GzipFile(filename='',mode='wb',compresslevel=6,mtime=0,fileobj=target) as zipped:
            for block in iter(lambda:source.read(2**20),b''):
                digest.update(block);size+=len(block);zipped.write(block)
    checked=hashlib.sha256();decoded_size=0
    with gzip.open(zpath,'rb') as decoded:
        for block in iter(lambda:decoded.read(2**20),b''):
            checked.update(block);decoded_size+=len(block)
    if checked.digest()!=digest.digest() or decoded_size!=size:raise ValueError('Lossless archive differs')
    compressed_hash=hashlib.sha256();parts=[]
    with zpath.open('rb') as source:
        for number,block in enumerate(iter(lambda:source.read(part_bytes),b'')):
            compressed_hash.update(block)
            if zpath.stat().st_size>part_bytes:
                part=directory/('RESULT.json.gz.part'+str(number).zfill(3))
                with part.open('xb') as target:target.write(block)
                parts.append(dict(path=part.relative_to(ROOT).as_posix(),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
    manifest=dict(raw_path=raw_path.relative_to(ROOT).as_posix(),raw_bytes=size,raw_sha256=digest.hexdigest(),
        published_path=zpath.relative_to(ROOT).as_posix(),encoding='gzip',compressed_bytes=zpath.stat().st_size,
        compressed_sha256=compressed_hash.hexdigest(),roundtrip_bytes_identical=True,raw_preserved_locally=True)
    if parts:
        manifest['parts']=parts
        with (directory/'.gitignore').open('x') as ignored:ignored.write('RESULT.json.gz\n')
    with manifest_path.open('x') as target:json.dump(manifest,target,indent=2);target.write('\n')
    return manifest


def read_receipt(commit, manifest_path):
    manifest = json.loads(P.P1.blob(commit, manifest_path))
    parts = manifest['parts'] if 'parts' in manifest else [dict(path=manifest['published_path'])]
    compressed = []
    for part in parts:
        raw = P.P1.blob(commit, part['path'])
        if part.get('sha256') and hashlib.sha256(raw).hexdigest() != part['sha256']:
            raise ValueError('Published gzip part differs')
        compressed.append(raw)
    raw = b''.join(compressed)
    if hashlib.sha256(raw).hexdigest() != manifest['compressed_sha256']:
        raise ValueError('Published gzip stream differs')
    decoded = gzip.decompress(raw)
    if len(decoded) != manifest['raw_bytes'] or hashlib.sha256(decoded).hexdigest() != manifest['raw_sha256']:
        raise ValueError('Published raw receipt bytes differ')
    return json.loads(decoded), manifest


def bind(commit, cell_id, directory, plan):
    """The fixed caller's records check, with explicit lossless receipt decoding."""
    result, manifest = read_receipt(commit, directory+'/RAW_RECEIPT.json')
    cell = P.selected(plan, 'CALIBRATION', cell_id)
    if result['status'] != 'FINITE_CELL_COMPLETE' or result['child']['status'] != 'PASS':
        raise ValueError('Complete calibration receipt required')
    produced = P.records(plan, cell, result['child']['paths'], source=result['source'],
                         evidence_path=directory+'/IMPLEMENTATION_EVIDENCE.json')
    for filename, key in (('C_INPUTS.json','C_inputs'), ('PRICES.json','prices'),
                          ('SIDECARS_UNBOUND.json','sidecars'), ('IMPLEMENTATION_EVIDENCE.json','implementation_evidence')):
        if json.loads(P.P1.blob(commit, directory+'/'+filename)) != produced[key]:
            raise ValueError('Published record/receipt/evidence differs: '+filename)
    sides = deepcopy(produced['sidecars']); reader = R.c_guard(); bindings = []
    for side in sides:
        for key in ('implementation_evidence','price_records','C_input'):
            side[key]['commit'] = commit
            reader.evidence(ROOT, side[key])
        side['path_receipts'] = dict(manifest_source=dict(commit=commit,path=directory+'/RAW_RECEIPT.json'),
            decoder='docs/research/d_full_h_p2_calibration_resume_20261115/READ_RECEIPT.py:read_receipt',
            selector=['child','paths'])
        if reader.evidence(ROOT, side['implementation_evidence']) != produced['implementation_evidence'][side['axis']]:
            raise ValueError('C metadata reader returned different evidence')
    for ordinal, price in enumerate(produced['prices']):
        index, axis = price['index'], price['axis']
        receipt = price['compile_receipt']
        path = directory+'/COMPILE_RECORDS/p'+str(index)+'_'+axis+'.json'
        raw = P.P1.blob(commit, path)
        if (hashlib.sha256(raw).hexdigest() != receipt['locator']['sha256']
                or json.loads(raw) != receipt['record']):
            raise ValueError('Published individual compile record differs')
        locator = lambda path, selector: dict(commit=commit, path=path, selector=selector)
        bindings.append(dict(cell_id=cell_id, index=index, axis=axis,
            cohort=price['cohort'], seeds=price['seeds'], physical=price['physical'],
            rz_count=price['metrics']['rz_count'],
            direct_price=locator(directory+'/PRICES.json', [ordinal]),
            compile_record=locator(path, []), compile_record_sha256=receipt['locator']['sha256'],
            implementation_evidence=locator(directory+'/IMPLEMENTATION_EVIDENCE.json', [axis,'paths',index]),
            calibration_record=locator(directory+'/C_INPUTS.json', [0 if axis=='X' else 1]),
            raw_receipt=locator(directory+'/RAW_RECEIPT.json', []),
            decoded_path_selector=['child','paths',index], execution_source=result['source'],
            compile_producer_source=receipt.get('producer_commit', result['source']['commit']),
            compile_reused_at_continuation=receipt['reused'],
            checkpoint=result['child']['paths'][index].get('checkpoint'),
            completed_receipt=result['child']['paths'][index].get('completed_receipt'),
            parents=result['child']['paths'][index].get('parents', [])))
    return dict(status='D1_PUBLICATION_LINKS_CHECKED_USING_C_EVIDENCE_READER_NOT_C_ADMISSION',
        publication=dict(commit=commit,directory=directory), bound_sidecars=sides,
        path_axis_bindings=bindings,
        checked_path_axes=len(produced['prices']), raw_receipt_sha256=manifest['raw_sha256'],
        C_scientific_admission='NOT_PERFORMED', canonical_round_guard_invoked=False, law_expectation=False)
