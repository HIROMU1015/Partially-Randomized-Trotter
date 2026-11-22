"""Bind a requested PR-DF horizon to its parent and reusable full-H reference."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from .pr_df_descriptor_extension import BINDING_KEYS
from .rpe_release_registry import read, write
from .saved_rpe_cost_catalog import frac


def references(root, runtime, prior_runs):
    # The published full-H reference is common to all circuit designs sharing
    # this Hamiltonian, target state and time. Additional bindings come from
    # the release, previous attempts or newly collected PF signals.
    published = dict(
        commit='0423cc1e1010d6fe5fb7888322311720b374f301',
        path='docs/research/rpe_m13_real_validation_20261122/run01/FULL_PF/signal/REFERENCE.json',
        sha256='84d354c05b567353161cd6ff03ba5ebe3948ddf0057591ef1ad643aa8ea4d8fd')
    declared = list(runtime.settings.get('horizon_references', ()))
    collected = []
    catalog = runtime.settings.get('horizon_extensions', {})
    for task in catalog.get('tasks', {}).values():
        for loc in task['request']['preparation'].get('input_hashes', ()):
            if Path(loc['path']).name == 'REFERENCE.json':
                declared.append({k:loc[k] for k in ('commit','path','sha256')})
    for run in dict.fromkeys([runtime.executor.run_dir, *map(Path, prior_runs)]):
        for path in (run/'completion'/'horizons').glob('*/REQUEST.json'):
            if loc := read(path).get('terminal_reference'):
                collected.append(loc)
        for path in (run/'pf_auto'/'horizon').glob('*/LEAF.json'):
            if loc := read(path).get('terminal_reference'):
                collected.append(loc)
    by_binding = {}
    seen = set()
    # Resolve each binding using the highest-priority available source.
    # Preserve conflicts within that tier. The published fallback is a
    # separate experiment and cannot override the run's declared reference.
    tiers = {}
    for tier, loc in ((tier,loc) for tier,group in enumerate((declared,collected,[published])) for loc in group):
        identity = (loc['commit'], loc['path'], loc['sha256'])
        if identity in seen:
            continue
        seen.add(identity)
        try:
            raw = subprocess.check_output(['git', 'show', loc['commit']+':'+loc['path']], cwd=root,
                                          stderr=subprocess.DEVNULL)
            if sha256(raw).hexdigest() != loc['sha256']:
                continue
            value = json.loads(raw)
            if value.get('kind') != 'NEW_D2_TWO_EXPONENTIAL_PATHS' or value.get('reference_state_actions') != 2:
                continue
            binding = {key:value['binding'][key] for key in BINDING_KEYS}
            key = (json.dumps(binding, sort_keys=True), frac(value['T']))
            if tiers.setdefault(key, tier) == tier:
                by_binding.setdefault(key, []).append((loc,value))
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError):
            continue
    return by_binding


def select_reference(config, by_binding):
    binding = {key:config['spec']['binding'][key] for key in BINDING_KEYS}
    last = config['rounds'][-1]
    if last['m'] != config['spec']['M']:
        raise ValueError('Parent final round differs from its declared horizon')
    matches = by_binding.get((json.dumps(binding,sort_keys=True), 2*frac(last['T'])), [])
    if not matches:
        raise ValueError('Full-H reference for the fixed Hamiltonian, state and terminal time is missing')
    # Provenance can differ. The full-H numerical pair and diagnostics must agree.
    content = [{k:v for k,v in evidence.items() if k != 'source'} for _,evidence in matches]
    if any(value != content[0] for value in content[1:]):
        raise ValueError('Collected full-H references disagree for this binding and time')
    return matches[0][0]


def ready_keys(candidates, records, horizons):
    """Reinstall completed terminal candidates scoped by their exact parent."""
    parents = {c['configuration_id']: (c.get('configuration_source'), c.get('signal_source'))
               for c in candidates if c['method'] == 'PR_DF' and c.get('configuration_id')}
    keys = set()
    for record in records:
        request = record.get('request') or {}
        parent = request.get('horizon_parent')
        cid = (request.get('selection') or {}).get('source_configuration_id')
        if (record['method'] == 'PR_DF' and record['status'] == 'READY_FOR_SEARCH'
                and request.get('M') in horizons and parent
                and parents.get(cid) == (parent['configuration'], parent['signal_packet'])):
            keys.add(record['key'])
    return keys


def prepare(root, runtime, completion, candidates, horizons, prior_runs, stop_requested, on_ready=None):
    from .pr_price_rules import GitReader
    from .pr_candidate_completion import Held

    if 13 not in horizons:
        return []
    parents = {c['configuration_id']:c for c in candidates if c['method']=='PR_DF' and
               c.get('configuration_id') and c.get('configuration_source') and c.get('signal_source') and
               c.get('M',12)==12 and not c.get('source_configuration_id')}
    native = {c.get('source_configuration_id') for c in candidates if c.get('M')==13 and c.get('ready')}
    if not parents:
        return []
    catalog = runtime.settings.get('horizon_extensions') or {}
    root = Path(root)
    refs = references(root, runtime, prior_runs)
    reader = GitReader(root)
    prior_owners = {}
    for directory in dict.fromkeys(map(Path, prior_runs)):
        if directory.resolve() == runtime.executor.run_dir.resolve():
            continue
        states = directory/'completion'/'STATE.json'
        if not states.is_file():
            continue
        for record in read(states).get('records', {}).values():
            request = record.get('request') or {}
            owned_parent = (request.get('selection') or {}).get('source_configuration_id')
            if not request.get('horizon_parent') or owned_parent not in parents:
                continue
            packets = prior_owners.setdefault(owned_parent, [])
            packet = (record.get('signal') or {}).get('packet_source')
            if packet and packet not in packets:
                packets.append(packet)
    records = []
    def advance(item):
        parent, c = item
        if parent in native:
            return None
        if Path(parent).name != parent or parent in ('','.','..'):
            raise ValueError('Invalid PR-DF parent configuration ID')
        directory = runtime.executor.run_dir/'completion'/'horizons'/parent
        try:
            registered = catalog.get('bindings', {}).get(parent, {}).get('rounds', {}).get('13', {})
            existing_owner = (bool(registered.get('signal')) or parent in prior_owners or
                any((Path(prior)/'completion'/'horizons'/parent/'REQUEST.json').is_file()
                    for prior in prior_runs if Path(prior).resolve() != runtime.executor.run_dir.resolve()))
            config = reader.read(c['configuration_source'])
            if config['id'] != parent or config['spec']['M'] != 12:
                raise ValueError('Signal parent differs from scoped PR-DF configuration')
            source = dict(configuration=c['configuration_source'],signal_packet=c['signal_source'])
            path = directory/'REQUEST.json'
            if path.is_file():
                request = read(path)
                if request['horizon_parent'] != source:
                    raise ValueError('Fixed horizon parent differs on Resume')
            else:
                ref = select_reference(config, refs)
                request = dict(method='PR_DF',configuration=source['configuration'],M=13,
                               horizon_parent=source,terminal_reference=ref,
                               selection=dict(rule='RPE_SCALED_SHOTS_V1',source_configuration_id=parent))
                write(path,request)
            record = completion.complete(request, existing_signal_only=existing_owner,
                                         saved_packets=prior_owners.get(parent, ()))
            if record['status'] == 'READY_FOR_SEARCH':
                record = completion.recover_horizon_signal(record)
            job_id = record.get('pending_preparation_job')
            report = dict(status=record['status'], reason=record.get('reason'), key=record['key'])
            if signal_job := record.get('horizon_signal_job') or job_id:
                report['job_id'] = signal_job
            write(directory/'STATUS.json', report)
            if record['status'] == 'READY_FOR_SEARCH':
                records.append(record)
                if on_ready is not None:
                    on_ready(record)
            return job_id
        except (Held,ValueError,KeyError,OSError,subprocess.CalledProcessError) as err:
            write(directory/'STATUS.json',dict(status='PREPARATION_INCOMPLETE',reason=str(err)))
        return None

    runtime.follow_preparations(parents.items(), advance, stop_requested)
    return records
