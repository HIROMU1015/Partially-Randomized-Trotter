"""Journal-only forensic helpers. No scientific imports or calculations."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import tarfile
from types import SimpleNamespace

from .shared_cpu import module

atomic_write_json = module('parallel_validation_executor').atomic_write_json


def parse_records(raw: bytes):
    """Retain a broken final write as evidence; reject malformed middle rows."""
    lines = raw.splitlines(keepends=True)
    rows, tail = [], None
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError('malformed journal record, not an incomplete tail')
            tail = dict(line=index + 1, bytes=len(line),
                        sha256=hashlib.sha256(line).hexdigest(), status='UNKNOWN')
            continue
        if not isinstance(row, dict):
            raise ValueError('journal record must be an object')
        rows.append(row)
    return rows, tail


def inspect_records(parent, worker_files):
    planned, fixed, completed = {}, {}, {}
    for index, row in enumerate(parent, 1):
        target = {'TRIAL_PLANNED': planned, 'PATH_FIXED': fixed,
                  'TRIAL_COMPLETE': completed}.get(row['event'])
        if target is not None:
            ticket = row['ticket']
            if ticket in target:
                raise ValueError('duplicate parent identity')
            target[ticket] = (index, row)
    starts, bits, prices, calls, locations, errors = {}, {}, {}, {}, {}, []
    for filename, rows in worker_files.items():
        for index, row in enumerate(rows, 1):
            ticket, event = row['ticket'], row['event']
            locations[ticket] = dict(file=filename, last_line=index)
            if event == 'WORKER_STARTED':
                if ticket in starts:
                    raise ValueError('duplicate worker start')
                starts[ticket] = row
            elif event == 'CALL':
                calls.setdefault(ticket, []).append(row['name'])
            elif event == 'BIT_SAVED':
                if ticket in bits:
                    raise ValueError('duplicate saved bit')
                bits[ticket] = row
            elif event == 'PRICE_SAVED':
                if ticket not in bits or ticket in prices:
                    raise ValueError('price before bit or duplicate price')
                if bits[ticket]['record'] != row['record']:
                    raise ValueError('bit/price record mismatch')
                prices[ticket] = row
            elif event == 'WORKER_STOP':
                errors.append(row)
    if not set(starts) <= set(planned) or not set(bits) <= set(starts):
        raise ValueError('worker identity missing parent/start')
    if not set(completed) <= set(prices):
        raise ValueError('parent completion lacks durable price')
    unfinished = []
    for ticket in sorted(set(starts) - set(prices)):
        state = 'BIT_DURABLE_PRICE_MISSING' if ticket in bits else 'ENGINE_ATTEMPT_NO_DURABLE_BIT'
        unfinished.append(dict(ticket=ticket, domain=starts[ticket]['domain'],
            trial=starts[ticket]['trial'], calls_entered=calls.get(ticket, []),
            status=state, last_logged_call=(calls.get(ticket) or [None])[-1],
            last_call_return='UNKNOWN: legacy journal records entry only',
            parent_plan_line=planned[ticket][0], **locations[ticket]))
    durations = [row['engine_seconds'] for row in bits.values()]
    return dict(planned=len(planned), paths_fixed=len(fixed), starts=len(starts),
        bits=len(bits), prices=len(prices), parent_completed=len(completed),
        worker_errors=errors, unfinished=unfinished,
        unfinished_by_cell=dict(Counter(r['domain']['cell'] for r in unfinished)),
        calls_entered=dict(Counter(name for names in calls.values() for name in names)),
        saved_by_axis_cell=dict(Counter(r['domain']['axis'] + ':' + r['domain']['cell'] for r in bits.values())),
        last_parent_event=parent[-1] if parent else None,
        last_parent_completion=next((r for r in reversed(parent) if r['event']=='TRIAL_COMPLETE'), None),
        elapsed_engine_seconds_saved_records=dict(min=min(durations), median=statistics.median(durations), max=max(durations)) if durations else {},
        global_wall_clock_order='UNKNOWN: legacy journal rows have no UTC/monotonic timestamp',
        selected_full_build_semantics='ENTRY only; build may have returned and exact gate application may have started',
        new_scientific_calls=0)


def verified_members(bundle: Path):
    manifest = json.loads((bundle / 'JOURNAL_ARCHIVE.json').read_text())
    archive = bundle / manifest['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('archive identity mismatch')
    with tarfile.open(archive, 'r:gz') as tf:
        if set(tf.getnames()) != set(manifest['raw_members']):
            raise ValueError('archive member identity mismatch')
        result = {}
        for member in tf.getmembers():
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError('unsafe archive member')
            raw = tf.extractfile(member).read()
            expected = manifest['raw_members'][member.name]
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError('journal byte identity mismatch')
            result[member.name] = raw
    return result


def frozen_reconcile(source: Path, workspace: Path):
    """Reuse the exact saved metadata function without importing the engine.

    The fixed function requires only these stdlib names and the four method
    names. Its output is isolated in the diagnostic workspace, never old data.
    """
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'reconcile']
    if len(nodes) != 1:
        raise ValueError('fixed reconcile function not found')
    env = dict(Path=Path, json=json, Counter=Counter, resource=resource,
               n3=SimpleNamespace(METHODS=('B0', 'B1-O', 'B1-OL', 'A1')),
               save=atomic_write_json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    env['reconcile'](workspace)
