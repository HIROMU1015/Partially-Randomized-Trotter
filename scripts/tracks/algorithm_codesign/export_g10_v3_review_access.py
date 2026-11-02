"""Extract/verify saved G10 v3 JSON for bounded-size GitHub review access.

Stdlib only. No science imports, runner calls, optimization or resynthesis.
Build creates a new directory exclusively; verify only reads saved files.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RESULT_COMMIT = 'fcd3ea6217bc00b667180cec149a70102d75f07e'
SOURCE_COMMIT = 'b9ed01455351628c9073748f5ba5751aa794b789'
ORIGINAL = Path('artifacts/track_b_g10_degree_result/2026-10-10/v3/result_v1.json')
OUT = Path('artifacts/track_b_g10_v3_review_access/2026-10-10')
ORIGINAL_BYTES = 66842494
ORIGINAL_SHA256 = '64a867dd6cc8f5f535880607616dea72f13af1360c7468f91eef44f79c543a2f'
LIMIT = 480 * 1024
AUDITS = ('saved_output_audit_corrected_v3.json', 'final_saved_output_audit_v3.json')


def safe(root, relative):
    if str(relative).lower().endswith('.npz'):
        raise PermissionError('NPZ rejected before access')
    return root / relative


def parse(data):
    def reject(value):
        raise ValueError('Non-JSON constant: ' + value)
    return json.loads(data, parse_float=Decimal, parse_constant=reject)


def encode(value):
    """JSON with unmodified rational strings and exact decimal numeric values."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('Nonfinite JSON number')
        return str(value)
    if isinstance(value, list):
        return '[' + ','.join(encode(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{' + ','.join(encode(k) + ':' + encode(v) for k, v in value.items()) + '}'
    raise TypeError(type(value).__name__)


def json_bytes(value):
    return (encode(value) + '\n').encode('utf-8')


def ident(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def file_ident(path):
    digest, n = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
            n += len(chunk)
    return {'bytes': n, 'sha256': digest.hexdigest()}


def utf8_slices(data, limit):
    start = 0
    while start < len(data):
        end = min(start + limit, len(data))
        while end < len(data) and data[end] & 0xC0 == 0x80:
            end -= 1
        if end == start:
            raise ValueError('Limit cannot hold one UTF-8 character')
        chunk = data[start:end]
        chunk.decode('utf-8', errors='strict')
        yield start, end, chunk
        start = end


def load_original(root):
    data = safe(root, ORIGINAL).read_bytes()
    assert ident(data) == {'bytes': ORIGINAL_BYTES, 'sha256': ORIGINAL_SHA256}
    result = parse(data)
    assert len(result['rows']) == 17 and result['source_commit'] == SOURCE_COMMIT
    assert result['status'] == 'G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE'
    assert result['retries'] == 0 and result['mandatory_STOP'] is True
    assert result['next_science_authorized'] is False
    authoritative = {}
    for name in AUDITS:
        path = ORIGINAL.parent / name
        saved = parse(safe(root, path).read_bytes())
        assert saved['status'] == 'SAVED_COMPLETION_OUTER_PROCESS_PROVENANCE_AND_ACCOUNTING_PASS'
        assert saved['original_output_identity']['result_v1.json'] == ident(data)
        assert saved['source_commit'] == SOURCE_COMMIT and saved['rows_complete'] == 17
        authoritative[str(path)] = file_ident(safe(root, path))
    return data, result, authoritative


def flatten(value, pointer=''):
    if isinstance(value, dict):
        for key, item in value.items():
            segment = key.replace('~', '~0').replace('/', '~1')
            yield from flatten(item, pointer + '/' + segment)
    else:
        yield pointer, value


def csv_value(value):
    return value if isinstance(value, str) else encode(value)


def build(root):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    assert head == RESULT_COMMIT, 'Build starts from the immutable result commit'
    original, result, audits = load_original(root)
    output = safe(root, OUT)
    output.mkdir(parents=True, exist_ok=False)
    files, raw_parts, row_index = {}, [], []

    def write(relative, data, **mapping):
        assert len(data) <= LIMIT, (relative, len(data))
        data.decode('utf-8', errors='strict')
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        files[str(relative)] = {**ident(data), **mapping}
        return files[str(relative)]

    for order, (start, end, chunk) in enumerate(utf8_slices(original, LIMIT)):
        path = f'raw_parts/part_{order:04d}.txt'
        write(path, chunk, kind='original_utf8_bytes', original_byte_start=start,
              original_byte_end_exclusive=end, order=order)
        raw_parts.append(path)
    metadata = {key: value for key, value in result.items() if key != 'rows'}
    write('result_metadata.json', json_bytes(metadata), kind='top_level_without_rows',
          source_pointer='', excluded_keys=['rows'])
    table_rows, flat_rows, columns = [], [], []
    for index, row in enumerate(result['rows']):
        slug = f'row_{index:02d}_m{row["degree"]}_{row["arm"]}'
        saved = {key: value for key, value in row.items() if key != 'events'}
        metadata_path = f'rows/{slug}.json'
        index_path = f'events/{slug}_index.json'
        write(metadata_path, json_bytes(saved), kind='row_without_events', row_index=index,
              source_pointer=f'/rows/{index}', excluded_keys=['events'])
        parts, batch, batch_size, first = [], [], 0, 0

        def flush():
            nonlocal batch, batch_size, first
            if not batch:
                return
            last = first + len(batch)
            envelope = {'row_index': index, 'degree': row['degree'], 'arm': row['arm'],
                        'event_start': first, 'event_end_exclusive': last,
                        'source_pointer': f'/rows/{index}/events', 'bindings': batch}
            path = f'events/{slug}_part_{len(parts):03d}.json'
            record = write(path, json_bytes(envelope), kind='event_bindings', row_index=index,
                           event_start=first, event_end_exclusive=last,
                           source_pointer=f'/rows/{index}/events', output_pointer='/bindings')
            parts.append({'path': path, **record})
            first = last
            batch, batch_size = [], 0

        for binding in row['events']:
            size = len(encode(binding).encode('utf-8'))
            # Reserve 2 KiB for envelope, commas and LF; never split a binding.
            if batch and batch_size + size + len(batch) + 2048 > LIMIT:
                flush()
            assert size + 2048 <= LIMIT, 'Single binding exceeds access limit'
            batch.append(binding)
            batch_size += size
        flush()
        index_record = {'row_index': index, 'degree': row['degree'], 'arm': row['arm'],
                        'source_pointer': f'/rows/{index}/events', 'event_count': len(row['events']),
                        'metadata_path': metadata_path, 'parts': parts,
                        'event_lookup': 'Select start <= i < end; binding is /bindings/(i-start).'}
        write(index_path, json_bytes(index_record), kind='row_event_index', row_index=index,
              source_pointer=f'/rows/{index}/events')
        row_index.append({key: index_record[key] for key in
                          ['row_index', 'degree', 'arm', 'event_count', 'metadata_path']}
                         | {'event_index_path': index_path, 'parts': parts})
        table_rows.append({'row_index': index, 'source_pointer': f'/rows/{index}',
                           'event_count': len(row['events']), 'event_index_path': index_path,
                           'saved_fields': saved})
        flat = dict(flatten(saved))
        flat_rows.append(flat)
        for key in flat:
            if key not in columns:
                columns.append(key)
    write('resource_table_exact.json', json_bytes({
        'original_result_commit': RESULT_COMMIT, 'original_result_identity': ident(original),
        'rows': table_rows, 'numeric_policy': 'Rational strings unchanged; numeric decimals exact.'
    }), kind='all_row_metadata', source_pointer='/rows', excluded_row_keys=['events'])
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['row_index', 'event_count', *columns])
    for index, flat in enumerate(flat_rows):
        writer.writerow([index, len(result['rows'][index]['events']),
                         *[csv_value(flat[key]) if key in flat else '' for key in columns]])
    write('resource_table_exact.csv', stream.getvalue().encode('utf-8'),
          kind='flat_exact_row_metadata', source_pointer='/rows',
          column_policy='JSON-pointer column names; missing fields blank; rational strings unchanged.')
    manifest = {
        'kind': 'G10_V3_SAVED_PRIMARY_DATA_ACCESS_PARTITION', 'original_result_commit': RESULT_COMMIT,
        'source_commit': SOURCE_COMMIT, 'original_result_path': str(ORIGINAL),
        'original_result_identity': ident(original), 'limit_bytes': LIMIT,
        'mandatory_STOP': True, 'next_science_authorized': False,
        'new_science_calls': 0, 'scientific_values_reoptimized_or_reinterpreted': False,
        'authoritative_audits': audits, 'non_authoritative_initial_audit_used': False,
        'files': files, 'raw_parts_in_concatenation_order': raw_parts, 'rows': row_index,
        'row_count': len(row_index), 'event_count': sum(r['event_count'] for r in row_index),
        'metadata_reconstruction': 'result_metadata.json plus rows/*.json with indexed bindings inserted as events.',
        'raw_reconstruction': 'Concatenate raw parts as bytes, in manifest order; do not add separators or normalize LF.',
        'rational_policy': 'Original string numerator/denominator values copied verbatim; never converted to float.',
        'decimal_policy': 'Parse JSON decimals as Decimal; re-encode exact numeric value. Raw parts preserve original lexical bytes.',
        'self_identity_external': True,
    }
    write('manifest.json', json_bytes(manifest), kind='manifest')
    # The manifest intentionally excludes itself from its own files map.
    return verify(root)


def verify(root):
    original, result, audits = load_original(root)
    output = safe(root, OUT)
    manifest = parse((output / 'manifest.json').read_bytes())
    assert manifest['authoritative_audits'] == audits
    assert manifest['original_result_identity'] == ident(original)
    assert manifest['limit_bytes'] == LIMIT and manifest['row_count'] == 17
    assert manifest['new_science_calls'] == 0 and manifest['mandatory_STOP'] is True
    for relative, record in manifest['files'].items():
        path = safe(output, relative)
        data = path.read_bytes()
        assert ident(data) == {key: record[key] for key in ['bytes', 'sha256']}, relative
        assert len(data) <= LIMIT, relative
        data.decode('utf-8', errors='strict')
    digest, position = hashlib.sha256(), 0
    for order, relative in enumerate(manifest['raw_parts_in_concatenation_order']):
        record = manifest['files'][relative]
        part = safe(output, relative).read_bytes()
        assert record['order'] == order and record['original_byte_start'] == position
        end = position + len(part)
        assert record['original_byte_end_exclusive'] == end and part == original[position:end]
        digest.update(part)
        position = end
    assert position == ORIGINAL_BYTES and digest.hexdigest() == ORIGINAL_SHA256
    assembled = parse((output / 'result_metadata.json').read_bytes())
    table = parse((output / 'resource_table_exact.json').read_bytes())
    rows, event_count, part_count = [], 0, 0
    for index, row_ref in enumerate(manifest['rows']):
        assert row_ref['row_index'] == index
        row = parse(safe(output, row_ref['metadata_path']).read_bytes())
        assert table['rows'][index]['saved_fields'] == row
        assert table['rows'][index]['source_pointer'] == f'/rows/{index}'
        assert table['rows'][index]['event_index_path'] == row_ref['event_index_path']
        assert table['rows'][index]['event_count'] == row_ref['event_count']
        assert row['degree'] == row_ref['degree'] and row['arm'] == row_ref['arm']
        event_index = parse(safe(output, row_ref['event_index_path']).read_bytes())
        assert event_index['parts'] == row_ref['parts'] and event_index['event_count'] == row_ref['event_count']
        events = []
        for record in row_ref['parts']:
            assert record['event_start'] == len(events)
            body = parse(safe(output, record['path']).read_bytes())
            assert body['row_index'] == index and body['source_pointer'] == f'/rows/{index}/events'
            assert body['degree'] == row_ref['degree'] and body['arm'] == row_ref['arm']
            assert body['event_start'] == record['event_start']
            assert body['event_end_exclusive'] == record['event_end_exclusive']
            events.extend(body['bindings'])
            assert record['event_end_exclusive'] == len(events)
            assert body['bindings'] == result['rows'][index]['events'][record['event_start']:len(events)]
            part_count += 1
        assert len(events) == row_ref['event_count']
        row['events'] = events
        assert row == result['rows'][index]
        rows.append(row)
        event_count += len(events)
    assembled['rows'] = rows
    assert assembled == result, 'All top-level, row and event fields preserved'
    assert event_count == manifest['event_count'] == 10936
    with (output / 'resource_table_exact.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 17
    for index, record in enumerate(csv_rows):
        assert int(record['row_index']) == index and int(record['event_count']) == len(rows[index]['events'])
        flat = dict(flatten({key: value for key, value in rows[index].items() if key != 'events'}))
        assert {key: value for key, value in record.items() if key.startswith('/') and value != ''} == {
            key: csv_value(value) for key, value in flat.items()}
    # Exercise UTF-8 cuts with multibyte characters, LF, escape sequences and decimals.
    sample = '{"日本語":"α😀\\n","q":"1/3","d":1.2300e-20}\n'.encode()
    assert b''.join(chunk for _, _, chunk in utf8_slices(sample, 7)) == sample
    assert parse(json_bytes(parse(sample))) == parse(sample)
    byte_limit_files = list(manifest['files']) + ['manifest.json']
    assert all(file_ident(safe(output, name))['bytes'] <= LIMIT for name in byte_limit_files)
    assert file_ident(safe(root, ORIGINAL)) == ident(original)
    return {
        'kind': 'G10_V3_READ_ONLY_REVIEW_ACCESS_IDENTITY_VERIFICATION', 'status': 'PASS',
        'original_result_commit': RESULT_COMMIT, 'source_commit': SOURCE_COMMIT,
        'original_result_identity': ident(original), 'manifest_identity': file_ident(output / 'manifest.json'),
        'partition_limit_bytes': LIMIT, 'raw_parts': len(manifest['raw_parts_in_concatenation_order']),
        'raw_concatenation_byte_for_byte_verified': True, 'each_raw_part_valid_UTF8': True,
        'independent_event_parts': part_count, 'rows_verified': len(rows), 'events_verified': event_count,
        'all_fields_semantically_reconstructed_exactly': True, 'exact_rational_strings_unchanged': True,
        'numeric_decimal_values_preserved': True, 'CSV_all_row_metadata_verified': True,
        'all_partition_files_within_limit': True, 'authoritative_audits': audits,
        'non_authoritative_initial_audit_used': False, 'new_science_calls': 0,
        'reoptimization_or_reclassification': False, 'mandatory_STOP': True, 'next_science_authorized': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'verify'])
    args = parser.parse_args()
    report = build(ROOT) if args.mode == 'build' else verify(ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
