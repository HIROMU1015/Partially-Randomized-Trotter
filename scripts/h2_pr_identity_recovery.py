"""Read-only H2 PR evidence recovery. No scientific imports, grant or execution.

Expected metadata comes from fixed Git objects, never a mutable working tree.
The optional execution root/map must be supplied explicitly. Raw archives are
hashed and streamed through zlib only; QPY/NPZ bodies are never interpreted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[1]
BASE = '4008ce45d26bb858e092d8654187bfcda2dbe180'
ACQUISITION = '4ec26921c6872ed6e8667f167074e1e405cf1a3f'
RECEIVED = 'artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received/'
PREVIOUS = 'artifacts/resource_applicability/track_a_h2_accepted_compare/2026-10-23/'
CONFIG_SUFFIX = 'stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json'
TOKEN = '${NEW_EXECUTION}/'


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def loads(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    def constant(value):
        raise EvidenceError('NONFINITE_JSON_CONSTANT')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified_bytes(path, expected):
    require(type(expected['bytes']) is int and expected['bytes'] >= 0, 'EXPECTED_BYTE_SIZE')
    require(Path(path).stat().st_size == expected['bytes'], 'ORIGINAL_BYTE_SIZE_MISMATCH')
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected['sha256'], 'ORIGINAL_SHA_MISMATCH')
    return raw


def resolve(root, locator):
    require(locator.startswith(TOKEN), 'EXPLICIT_EXECUTION_TOKEN_ONLY')
    suffix = locator[len(TOKEN):]
    require(suffix and not suffix.startswith('/') and
            all(p not in ('', '.', '..') for p in suffix.split('/')), 'UNSAFE_LOCATOR')
    base = Path(root).resolve(strict=True)
    require(base.is_dir(), 'EXECUTION_ROOT_DIRECTORY')
    path = base / PurePosixPath(suffix)
    require(path.resolve().is_relative_to(base), 'LOCATOR_ESCAPES_ROOT')
    return path


def root_from_map(path):
    data = loads(Path(path).read_bytes())
    require(type(data.get('roots')) is dict, 'EXPORTER_ROOT_MAP_SCHEMA')
    roots = [p for p, token in data['roots'].items() if token == 'NEW_EXECUTION']
    require(len(roots) == 1 and Path(roots[0]).is_absolute(), 'UNIQUE_ABSOLUTE_NEW_EXECUTION_MAP')
    return Path(roots[0]).resolve(strict=True)


def check_events(value, anchor, piece_counts):
    """Check saved metadata identity, not event sampling or operator semantics."""
    events = value['events']; expected = anchor['result']['saved']
    require(value['conditional'] is False and value['cohort'] == 'EXPLORATION', 'EVENT_COHORT')
    require(value['event_digest'] == expected['event_digest'] == digest(events), 'EVENT_DIGEST_JOIN')
    require(type(events) is list and len(events) == len(piece_counts), 'EVENT_SIGNED_PIECES')
    for stream, count in zip(events, piece_counts):
        require(type(stream) is list and len(stream) == count, 'EVENT_PIECE_COUNT')
        for event in stream:
            order = event['taylor_order']
            require(type(order) is int and order in (0, 2, 4), 'EVENT_ORDER')
            products = event['product_component_ids']; selected = event['selected_component_ids']
            require(type(products) is list and len(products) == order and
                    selected == products + [event['rotation_component_id']] and
                    # Fixed provider RTEEvent.component_id is a nonempty string.
                    all(type(i) is str and bool(i) for i in selected), 'EVENT_COMPONENTS')
            phase = event['phase']
            require(set(phase) == {'real', 'imag'} and
                    all(type(x) is int or (type(x) is float and math.isfinite(x)) for x in phase.values()), 'EVENT_PHASE')
            sequence = event['application_sequence']
            require(all(type(x['application_index']) is int for x in sequence) and
                    [x['application_index'] for x in sequence] == list(range(order + 1)) and
                    [x['role'] for x in sequence] == ['product'] * order + ['rotation'] and
                    [x['component_id'] for x in sequence] == selected, 'EVENT_APPLICATION_ORDER')
    return dict(task=anchor['task']['id'], events=sum(piece_counts), pieces=len(piece_counts),
                event_digest=expected['event_digest'], status='SAVED_EVENT_CONTENT_MATCH')


def verify_archive(path, stored, original):
    """Bound decompression to the declared original size, without writing raw."""
    require(type(original['bytes']) is int and original['bytes'] >= 0, 'ORIGINAL_RAW_SIZE')
    require(Path(path).stat().st_size == stored['bytes'] and file_digest(path) == stored['sha256'],
            'ARCHIVE_SHA_OR_SIZE')
    decoder = zlib.decompressobj(); h = hashlib.sha256(); size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            pending = block
            while pending:
                raw = decoder.decompress(pending, min(1024 * 1024, original['bytes'] - size + 1))
                h.update(raw); size += len(raw)
                require(size <= original['bytes'], 'RAW_EXCEEDS_DECLARED_SIZE')
                require(not decoder.unused_data, 'ARCHIVE_TRAILING_DATA')
                pending = decoder.unconsumed_tail
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
            'ARCHIVE_INCOMPLETE_OR_TRAILING')
    require(size == original['bytes'] and h.hexdigest() == original['sha256'], 'ORIGINAL_RAW_SHA_OR_SIZE')
    require(file_digest(path) == stored['sha256'], 'ARCHIVE_CHANGED_DURING_READ')
    return dict(status='ARCHIVE_AND_RAW_BYTES_MATCH', raw_sha256=h.hexdigest(),
                raw_bytes=size, scientific_body_decoded=False, raw_file_written=False)


class Catalogue:
    def __init__(self):
        self.origins = {r['id']: r for r in loads(git_bytes(BASE, RECEIVED + 'origin_inventory.json'))['originals']}
        self.tasks = loads(git_bytes(BASE, RECEIVED + 'task_acceptance_67.json'))['tasks']
        self.schedule = {r['task_id']: r for r in loads(git_bytes(BASE, RECEIVED + 'rte_schedule_metadata.json'))['tasks']}
        self.storage = loads(git_bytes(BASE, PREVIOUS + 'evidence_sources/storage_records_e353.json'))['tasks']
        self.runtime = loads(git_bytes(BASE, RECEIVED + 'runtime_provider_manifest.json'))
        self.context = loads(git_bytes(BASE, RECEIVED + 'acquisition_context.json'))
        require(len(self.tasks) == 67 and len({r['task']['id'] for r in self.tasks}) == 67,
                'FIXED_67_TASKS')

    def requests(self):
        ids = {'O00001', 'O00002', 'O00006', 'O00008', 'O00009', 'O00010', 'O00011'}
        by_locator = {r['locator']: r for r in self.origins.values()}
        for row in self.tasks:
            ids.add(row['result']['origin'])
            ids.update(x['origin'] for x in row['saved_correctness'])
            folder = self.origins[row['result']['origin']]['locator'].rsplit('/', 1)[0]
            for index, axis in enumerate(row['axes']):
                ids.add(axis['cost']['origin'])
                require(axis['axis'] == ('cosine', 'sine')[index], 'PAIRED_AXIS_ORDER')
                ids.add(by_locator[folder + '/compiler_phase_guard_' + str(index) + '.json']['id'])
            ids.update(x['origin'] for x in self.schedule[row['task']['id']]['pieces'])
        raw_archives = {}
        for row in self.storage:
            ids.add(row['storage_audit']['origin'])
            for axis in row['axes']:
                for item in axis['archives']:
                    ids.update((item['manifest']['origin'], item['release']['origin']))
                    manifest = item['manifest']['saved']
                    parent = self.origins[item['manifest']['origin']]['locator'].rsplit('/', 1)[0]
                    name = manifest['archive_file']; require(PurePosixPath(name).name == name, 'ARCHIVE_BASENAME')
                    raw_archives[parent + '/' + name] = manifest['codec_receipt']
        ids.update(by_locator[p]['id'] for p in raw_archives)
        rows = []
        for oid in sorted(ids):
            row = self.origins[oid]
            item = dict(id=oid, locator=row['locator'], roles=row['roles'],
                        sha256=row['original_sha256'], bytes=row['bytes'])
            if item['locator'] in raw_archives:
                item['archive'] = raw_archives[item['locator']]
            rows.append(item)
        for row in self.tasks:
            tid = row['task']['id']; pieces = self.schedule[tid]['pieces']
            rows.append(dict(id='EVENT__' + tid, locator=TOKEN + 'stagepacks/H2_PR_EXPLORATION/run01/output/' + tid + '/events_PRIVATE.json',
                roles=['full_events_saved_by_fixed_SOURCE_not_in_public_export'], task=tid,
                event_digest=row['result']['saved']['event_digest'],
                piece_counts=[p['saved']['rte_steps'] for p in pieces],
                sha256=None, bytes=None, raw_file_SHA_previously_recorded=False))
        require(all(r['locator'].startswith(TOKEN) for r in rows), 'REQUESTS_LIMITED_TO_EXECUTION_ROOT')
        return rows

    def verify_config(self, path):
        expected = self.origins['O00002']
        value = loads(verified_bytes(path, dict(bytes=expected['bytes'], sha256=expected['original_sha256'])))
        for key in ('SOURCE', 'stage', 'binding', 'preparation_hash', 'compiler_phase_policy'):
            require(digest(value[key]) == digest(self.context[key]['saved']), 'ORIGINAL_CONFIG_FIELD:' + key)
        require(value['SOURCE'] == ACQUISITION, 'ORIGINAL_ACQUISITION_SOURCE')
        targets = {r['task']['id']: r['task'] for r in self.tasks}
        selected = [t for t in value['tasks'] if t['id'] in targets]
        require(len(selected) == 67 and len({t['id'] for t in selected}) == 67 and
                all(digest(t) == digest(targets[t['id']]) for t in selected), 'ORIGINAL_67_TASK_JOIN')
        return value

    def recover(self, root=None):
        requests = self.requests(); results = []
        if root is not None:
            self.verify_config(resolve(root, TOKEN + CONFIG_SUFFIX))
        anchors = {r['task']['id']: r for r in self.tasks}
        for request in requests:
            result = dict(id=request['id'], locator=request['locator'], status='MISSING_CANONICAL_ROOT')
            if root is not None:
                try:
                    path = resolve(root, request['locator'])
                    if request['id'].startswith('EVENT__'):
                        result.update(check_events(loads(path.read_bytes()), anchors[request['task']], request['piece_counts']))
                    elif 'archive' in request:
                        receipt = request['archive']
                        result.update(verify_archive(path, receipt['stored'], receipt['original']))
                    else:
                        verified_bytes(path, request); result['status'] = 'ORIGINAL_BYTES_MATCH'
                except FileNotFoundError:
                    result['status'] = 'MISSING_ORIGINAL_FILE'
                except (EvidenceError, OSError, ValueError, KeyError, TypeError, zlib.error) as exc:
                    result.update(status='EVIDENCE_REJECTED', reason=type(exc).__name__ + ':' + str(exc) if isinstance(exc, EvidenceError) else type(exc).__name__)
            results.append(result)
        return dict(status='BLOCKED_PR_IDENTITY', PR_reuse='BLOCKED', catalogue_base=BASE,
                    original_acquisition_SOURCE=ACQUISITION, tasks=67, axes=134,
                    canonical_root_provided=root is not None, original_config_SHA_verified=root is not None,
                    requests=requests, recovered=results, scientific_calls=0, scientific_imports=0,
                    raw_or_private_config_copies_written=0, live_grants=0, execution_authorized=False,
                    execution_ready=False, identity_witness_created=False,
                    unresolved=['FULL_HISTORICAL_TO_COMMON_OPERATION_STREAM_JOIN',
                                'ACTUAL_HISTORICAL_RUNTIME_LOAD_ATTESTATION_AND_RUNTIME_BYTE_DIFFERENCE',
                                'COMPILER_AND_CORRECTNESS_SCOPE_IDENTITY'], mandatory_STOP=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execution-root'); group.add_argument('--token-map'); group.add_argument('--config')
    parser.add_argument('--report', required=True)
    args = parser.parse_args(); catalogue = Catalogue(); root = None
    if args.execution_root:
        require(Path(args.execution_root).is_absolute(), 'ABSOLUTE_EXPLICIT_ROOT')
        root = Path(args.execution_root).resolve(strict=True)
    elif args.token_map:
        root = root_from_map(args.token_map)
    elif args.config:
        path = Path(args.config).resolve(strict=True)
        catalogue.verify_config(path)
        require(path.as_posix().endswith('/' + CONFIG_SUFFIX), 'CANONICAL_CONFIG_SUFFIX_REQUIRED')
        root = path.parents[2]
    target = Path(args.report).absolute()
    require(not target.exists() and not target.is_symlink(), 'NEW_REPORT_REQUIRED')
    require(target.resolve().is_relative_to(ROOT), 'REPORT_IN_RECOVERY_WORKTREE')
    if root is not None:
        require(not target.resolve().is_relative_to(root), 'REPORT_OUTSIDE_ORIGINALS')
    result = catalogue.recover(root)
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'PR_reuse', 'tasks', 'axes', 'canonical_root_provided',
                     'scientific_calls', 'live_grants', 'execution_ready', 'mandatory_STOP')}))


if __name__ == '__main__':
    main()
