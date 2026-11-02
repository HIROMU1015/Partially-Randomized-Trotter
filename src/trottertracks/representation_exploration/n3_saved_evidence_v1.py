"""Lossless, bounded JSON views of existing evidence; no scientific evaluation."""
import hashlib
import json
from pathlib import Path

SHARD_CAP = 8192
LITERAL_CAP = 7000
TOTAL_CAP = 32 * 1024 ** 2
BASE_COMMIT = '86f57f05a36e7e6639e827486d300bd049bf2438'
PAYLOAD = 'e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c'
LEDGER = 'artifacts/n3_occupation_transfer_v1/2026-10-12/run1/primary_ledger.json'
LEDGER_SHA = '8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69'
PACK = 'artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json'
PACK_SHA = 'c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4'


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
def sha(data): return hashlib.sha256(data).hexdigest()
def escaped(key): return str(key).replace('~', '~0').replace('/', '~1')


def pointer(value, path):
    if not path: return value
    if not path.startswith('/'): raise ValueError('JSON pointer')
    for key in path[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


class Store:
    def __init__(self, out):
        self.out = Path(out)
        self.nodes = {}
        self.written_bytes = 0
        self.locations = None
        self.flushed = False

    def write(self, path, value):
        data = canonical(value) + b'\n'
        if len(data) > SHARD_CAP or self.written_bytes + len(data) > TOTAL_CAP:
            raise ValueError('extraction output cap')
        target = self.out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.written_bytes += len(data)

    def put(self, value):
        identity = sha(canonical(value))
        if identity in self.nodes: return identity
        if len(canonical(value)) <= LITERAL_CAP:
            node = {'id': identity, 'encoding': 'literal', 'value': value}
        elif isinstance(value, dict):
            node = {'id': identity, 'encoding': 'object', 'members': {k: self.put(v) for k, v in value.items()}}
        elif isinstance(value, list):
            node = {'id': identity, 'encoding': 'array', 'members': [self.put(v) for v in value]}
        else:
            raise ValueError('oversized scalar')
        # Arrays with exceptionally many children use recursive bounded pages.
        if len(canonical(node)) + 1 > SHARD_CAP:
            if node['encoding'] != 'array': raise ValueError('oversized object keys')
            refs = node.pop('members')
            node['encoding'] = 'paged_array'
            node['pages'] = [self.put(refs[i:i + 64]) for i in range(0, len(refs), 64)]
        self.nodes[identity] = node
        return identity

    def flush(self):
        if self.flushed: return
        locations, batch, page_number = {}, {}, 0
        for identity, node in sorted(self.nodes.items()):
            if batch and len(canonical(dict(batch, **{identity: node}))) + 1 > SHARD_CAP:
                path = f'objects/page_{page_number}.json'
                self.write(path, batch)
                locations.update({key: path for key in batch})
                batch = {}; page_number += 1
            batch[identity] = node
        if batch:
            path = f'objects/page_{page_number}.json'; self.write(path, batch)
            locations.update({key: path for key in batch})
        index_pages = []
        items = sorted(locations.items())
        for start in range(0, len(items), 64):
            path = f'objects/index_{start // 64}.json'
            self.write(path, dict(items[start:start + 64])); index_pages.append(path)
        self.write('objects/index.json', {'index_pages': index_pages})
        self.locations = locations
        self.flushed = True

    def get(self, identity, active=None):
        active = set() if active is None else set(active)
        if identity in active: raise ValueError('cyclic fragment')
        active.add(identity)
        if not self.flushed and identity in self.nodes:
            node = self.nodes[identity]
        else:
            if self.locations is None:
                pages = json.loads((self.out / 'objects/index.json').read_bytes())['index_pages']
                self.locations = {}
                for path in pages:
                    self.locations.update(json.loads((self.out / path).read_bytes()))
            node = json.loads((self.out / self.locations[identity]).read_bytes())[identity]
        if node['id'] != identity: raise ValueError('fragment id')
        kind = node['encoding']
        if kind == 'literal': value = node['value']
        elif kind == 'object': value = {k: self.get(v, active) for k, v in node['members'].items()}
        elif kind == 'array': value = [self.get(v, active) for v in node['members']]
        elif kind == 'paged_array':
            refs = [r for page in node['pages'] for r in self.get(page, active)]
            value = [self.get(v, active) for v in refs]
        else: raise ValueError('fragment encoding')
        if sha(canonical(value)) != identity: raise ValueError('lossless fragment hash')
        return value


def extract(root, out):
    root, out = Path(root), Path(out)
    if out.exists(): raise ValueError('fresh extraction directory required')
    ledger_data, pack_data = (root / LEDGER).read_bytes(), (root / PACK).read_bytes()
    if sha(ledger_data) != LEDGER_SHA or sha(pack_data) != PACK_SHA:
        raise ValueError('immutable input identity')
    ledger, pack = json.loads(ledger_data), json.loads(pack_data)
    if len(ledger) != 8 or len(pack['cases']) != 8: raise ValueError('eight cases')
    out.mkdir(parents=True)
    store = Store(out)
    all_entries = []
    counts = {'raw_methods': 0, 'raw_attempts': 0, 'saved_spectral_proposals': 0,
              'saved_valid_spectral_flags': 0, 'proof_pool_records': 0, 'operational_views': 0}
    sources = {LEDGER: ledger, PACK: pack}

    def entry(path, source_path, role):
        value = pointer(sources[source_path], path)
        result = {'source_path': source_path, 'pointer': path, 'role': role, 'value_ref': store.put(value)}
        all_entries.append(result)
        return result

    case_indexes = []
    for ci, (case, inp) in enumerate(zip(ledger, pack['cases'])):
        if case['input_id'] != inp['id']: raise ValueError('case ordering')
        prefix = '/' + str(ci)
        case_entries = [entry('/cases/' + str(ci), PACK, 'input_coefficients'),
                        entry(prefix + '/upper', LEDGER, 'saved_trial_upper'),
                        entry(prefix + '/bounds', LEDGER, 'saved_external_bounds'),
                        entry(prefix + '/standalone_reconstructed_cost', LEDGER, 'saved_cost_estimate'),
                        entry(prefix + '/resources', LEDGER, 'saved_resource_record')]
        for ri, row in enumerate(case['raw_rows']):
            rp = prefix + '/raw_rows/' + str(ri)
            for mi, method in enumerate(row['methods']):
                mp = rp + '/methods/' + str(mi)
                counts['raw_methods'] += 1
                preview = {'input_id': inp['id'], 'row': row['row'], 'method': method['method'],
                           'U': row['U'], 'c': row['c'], 'delta_goal': row['delta'],
                           'original_status': method['status'], 'original_accepted': method['accepted'],
                           'event_pointer': prefix + '/events', 'attempts': []}
                method_entry = entry(mp, LEDGER, 'complete_saved_raw_method')
                case_entries.append(method_entry)
                for ai, attempt in enumerate(method['attempts']):
                    ap = mp + '/attempts/' + str(ai)
                    counts['raw_attempts'] += 1
                    spectra = attempt.get('spectral_attempts', [])
                    counts['saved_spectral_proposals'] += len(spectra)
                    counts['saved_valid_spectral_flags'] += sum(s.get('valid_spectral') is True for s in spectra)
                    ae = entry(ap, LEDGER, 'saved_raw_attempt')
                    case_entries.append(ae)
                    view = {k: attempt[k] for k in ('kind', 'kappa', 'mu', 'PSD_valid', 'accepted', 'status', 'proof') if k in attempt}
                    view.update(pointer=ap, value_ref=ae['value_ref'], spectral_proposals_stored=len(spectra),
                                originally_selected=attempt == method['witness'])
                    if attempt.get('spectral') is not None:
                        view['selected_spectral'] = {k: attempt['spectral'][k] for k in ('tau', 'lower', 'bits', 'inflate_factor', 'Z_proof', 'residual_proof', 'full_trace', 'empty') if k in attempt['spectral']}
                        if 'Z' in attempt['spectral']: view['selected_Z_ref'] = store.put(attempt['spectral']['Z'])
                    preview['attempts'].append(view)
                # Bounded preview pages give humans direct kappa/mu/tau access.
                pages = []
                items = preview.pop('attempts')
                batch = []
                for item in items:
                    proposed = dict(preview, attempts=batch + [item])
                    if len(canonical(proposed)) + 1 > SHARD_CAP and batch:
                        pages.append(batch); batch = []
                    batch.append(item)
                pages.append(batch)
                for pi, batch in enumerate(pages):
                    path = f'cases/{inp["id"]}/{row["row"]}_{method["method"]}_{pi}.json'
                    store.write(path, dict(preview, attempts=batch, full_method_ref=method_entry['value_ref']))
            oe = entry(prefix + '/operational_rows/' + str(ri), LEDGER, 'saved_transport_and_fallback')
            case_entries.append(oe); counts['operational_views'] += 1
        for pid in case['proof_pool']:
            case_entries.append(entry(prefix + '/proof_pool/' + escaped(pid), LEDGER, 'saved_PSD_or_SPD_proof'))
            counts['proof_pool_records'] += 1
        # Cost events are referenced, not copied; no new cost arithmetic is run.
        pages = []
        for start in range(0, len(case_entries), 16):
            path = f'cases/{inp["id"]}/provenance_{start // 16}.json'
            store.write(path, {'input_id': inp['id'], 'entries': case_entries[start:start + 16]})
            pages.append(path)
        cp = f'cases/{inp["id"]}/index.json'
        store.write(cp, {'input_id': inp['id'], 'case_sha256': case['case_sha256'], 'source_fingerprint': case['source_fingerprint'],
                         'source_pointer': prefix, 'provenance_pages': pages, 'saved_events_pointer': prefix + '/events',
                         'derivative_width_evaluations': 0})
        case_indexes.append(cp)
    store.flush()
    # Byte/JSON-pointer identity only: no PSD, eigenvalue or width re-evaluation.
    for e in all_entries:
        if store.get(e['value_ref']) != pointer(sources[e['source_path']], e['pointer']):
            raise ValueError('extraction roundtrip')
    paths = sorted(p for p in out.rglob('*.json'))
    file_pages = []
    for start in range(0, len(paths), 32):
        rows = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in paths[start:start + 32]]
        name = f'inventory_{start // 32}.json'
        store.write(name, rows); file_pages.append(name)
    report = {'schema_version': 1, 'status': 'LOSSLESS_SAVED_EVIDENCE_EXTRACTION_PASS_NO_WIDTHS',
              'payload_commit': PAYLOAD, 'ledger_sha256': LEDGER_SHA, 'input_sha256': PACK_SHA,
              'case_indexes': case_indexes, 'inventory_pages': file_pages, 'counts': counts,
              'roundtrip_pointer_checks': len(all_entries), 'unique_value_fragments': len(store.nodes),
              'shard_cap_bytes': SHARD_CAP, 'total_cap_bytes': TOTAL_CAP,
              'derivative_width_evaluations': 0, 'new_PSD_evaluations': 0, 'new_eigensolver_calls': 0,
              'scientific_batch_reruns': 0, 'new_kappa_or_spectral_candidates': 0,
              'posthoc_evaluation_authorized': False, 'mandatory_stop': True}
    store.write('catalog.json', report)
    return report


def verify(root, out):
    root, out = Path(root), Path(out)
    report = json.loads((out / 'catalog.json').read_bytes())
    sources = {}
    for path, digest in ((LEDGER, LEDGER_SHA), (PACK, PACK_SHA)):
        data = (root / path).read_bytes()
        if sha(data) != digest: raise ValueError('source identity')
        sources[path] = json.loads(data)
    files = []
    for page in report['inventory_pages']:
        for row in json.loads((out / page).read_bytes()):
            data = (out / row['path']).read_bytes()
            if len(data) != row['bytes'] or sha(data) != row['sha256']: raise ValueError('extracted file hash')
            files.append(row['path'])
    store, checks = Store(out), 0
    for case_path in report['case_indexes']:
        case = json.loads((out / case_path).read_bytes())
        for page in case['provenance_pages']:
            for e in json.loads((out / page).read_bytes())['entries']:
                if store.get(e['value_ref']) != pointer(sources[e['source_path']], e['pointer']): raise ValueError('pointer identity')
                checks += 1
    if checks != report['roundtrip_pointer_checks']: raise ValueError('coverage count')
    for path in out.rglob('*.json'):
        if path.stat().st_size > SHARD_CAP: raise ValueError('shard cap')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > TOTAL_CAP: raise ValueError('total cap')
    return {'status': 'LOSSLESS_SAVED_EXTRACTION_VERIFIED', 'pointer_checks': checks, 'inventory_files': len(files),
            'width_evaluations': 0, 'PSD_evaluations': 0, 'eigensolver_calls': 0}
