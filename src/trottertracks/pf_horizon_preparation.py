"""Extend admitted PF parents through the normal signal queue and price registry."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

from .pf_distributed_preparation import git, read
from .pr_price_rules import GitReader
from .rpe_release_registry import write

PF_Q3_CONTRACT = ('b4296b110b129d226dbef2743c14f725a27c3bf6',
                  'docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json')


def parent_entry(candidate):
    entry = candidate.get('pf_evidence')
    if entry:
        return entry
    key = candidate.get('key') or {}
    if not (candidate.get('configuration_source') and candidate.get('signal_source')
            and all(k in key for k in ('p','formula','delta_over_h'))):
        return {}
    return dict(method=candidate['method'], p=key['p'], formula=key['formula'], delta_over_h=key['delta_over_h'],
                configuration=candidate['configuration_source'], signal_packet=candidate['signal_source'],
                prices=candidate.get('price_registry_source'), q3_inputs=candidate.get('q3_template_source'))


def parents(candidates, methods, horizons, bindings, existing, saved_signals=None):
    """Only consider parents present in this optimizer's scoped candidate list."""
    from run_rpe_optimizer_q3_job import M13_ROUTES
    if 13 not in horizons:
        return {}
    native = {c['source_configuration_id'] for c in candidates
              if c.get('source_configuration_id') and c.get('M') == 13 and c.get('ready')}
    selected = {}
    for c in candidates:
        cid, method, entry = c.get('configuration_id'), c['method'], parent_entry(c)
        entry = entry or (saved_signals or {}).get(cid, {})
        if (method in methods and (method, 'pf_b_pf6') in M13_ROUTES and cid
                and c.get('M', 12) == 12 and not c.get('source_configuration_id')
                and cid not in bindings and cid not in existing and cid not in native
                and entry.get('method') == method and entry.get('configuration') and entry.get('signal_packet')):
            selected[cid] = entry
    return selected


def saved_parent_signals(root, run, candidates, methods, prior_runs=()):
    """Find collected PF signal packets even when their price stage is unfinished."""
    from contextlib import closing
    from .rpe_preparation_artifacts import ArtifactCache
    from .pf_saved_prices import candidate_roots
    from .rpe_preparation_queue import normalize_preparation_request
    from .rpe_q3_coordinator import digest

    unresolved = {row['configuration_id']: row for row in candidates
                  if row['method'] in methods and row['method'] in ('FULL_PF', 'TRUNCATED_PF')
                  and row.get('M', 12) == 12 and not row.get('source_configuration_id')
                  and row.get('configuration_body') and row.get('configuration_id')
                  and not parent_entry(row)}
    if not unresolved:
        return {}
    reader = GitReader(root)
    collected = {}
    conflicts = set()
    for origin in dict.fromkeys(Path(p).resolve() for p in (run, *prior_runs)):
        db = origin/'distributed'/'QUEUE.sqlite3'
        bases = [base for base in candidate_roots(origin/'pf_auto') if base.is_dir()]
        if not db.is_file() or not bases:
            continue
        artifacts = ArtifactCache(origin/'distributed'/'artifacts')

        def verified_input(folder, name, checksum):
            try:
                content = (folder/name).read_bytes()
            except FileNotFoundError:
                content = artifacts.path(checksum).read_bytes()
            if sha256(content).hexdigest() != checksum:
                raise ValueError('Collected PF signal input or output differs: '+name)
            return json.loads(content)

        with closing(sqlite3.connect(db.as_uri()+'?mode=ro', uri=True)) as conn:
            for base in bases:
                for cid, candidate in unresolved.items():
                    relative = f'docs/research/distributed_selected_pf/{cid}'
                    folder = base/cid/'pf_auto'/'pf_completion'/'provider_workspace'/relative
                    marker = folder/'PF_SIGNALS.DISTRIBUTED.json'
                    if not marker.is_file() or cid in conflicts:
                        continue
                    try:
                        record = read(marker)
                        request = normalize_preparation_request(record['request'])
                        job = record['job_id']
                        if (job != request['request_id']
                                or request['preparation']['provider'] != 'PF_SIGNALS'):
                            continue
                        prefix = 'workspace/'+relative+'/'
                        original_input = verified_input(folder, 'PF_SIGNALS.INPUT.json',
                            request['preparation']['inputs'][prefix+'PF_SIGNALS.INPUT.json'])
                        row = conn.execute(
                            "SELECT state, attempt_id, request_sha256 FROM jobs "
                            "WHERE job_id=? AND method='PREPARATION:signal'", (job,)).fetchone()
                        checksum = digest(request)
                        if not row or row[0] != 'ACCEPTED' or not row[1] or row[2] != checksum:
                            continue
                        directory = origin/'jobs'/'PREPARATION:signal'/job
                        outcome = read(directory/'OUTCOME.json')
                        receipt = read(directory/'PROCESS_RETURN.json')
                        if (outcome['request_id'] != job or outcome['request_sha256'] != checksum
                                or outcome['status'] != 'PREPARATION_COMPLETE'
                                or outcome['source_manifest'] != request['preparation']['source']
                                or receipt['returncode'] != 0 or receipt.get('signal') is not None):
                            continue
                        saved = verified_input(folder, 'PF_SIGNALS.RESULT.json',
                            outcome['artifacts'][prefix+'PF_SIGNALS.RESULT.json'])
                        if saved['status'] != 'SIGNALS_READY':
                            continue
                        plan = original_input['source']['input_plan']
                        if plan['path'] != relative+'/PLAN.json':
                            continue
                        config = dict(plan, selector=['configuration'])
                        packet = dict(commit=saved['publication'], path=relative+'/signal_run01/MEAN_HANDOFF.json')
                        if reader.read(config) != candidate['configuration_body']:
                            continue
                        means = reader.read(packet)
                        if (means['configuration_id'] != cid
                                or [row['m'] for row in means['means']] != list(range(13))):
                            continue
                        if cid in collected:
                            if not _same_saved_document(reader, collected[cid]['signal_packet'], packet):
                                collected.pop(cid)
                                conflicts.add(cid)
                            continue
                        key = candidate['key']
                        collected[cid] = dict(method=candidate['method'], p=key['p'], formula=key['formula'],
                                              delta_over_h=key['delta_over_h'], configuration=config,
                                              signal_packet=packet)
                    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error, subprocess.CalledProcessError):
                        continue
    return collected


def eligible_route(routes, method, parent):
    for route in routes.get(method, ()):
        ids = route.get('configuration_ids')
        if (route.get('kind') == 'pf_b_pf6' and not route.get('horizon_risk_source')
                and (ids in (None, '*') or parent in ids)):
            return route
    raise ValueError('Parent has no matching fixed PF Q3 route')


def publish(workspace, root, path):
    rel = path.relative_to(workspace).as_posix()
    git(workspace, 'add', '--sparse', '--', rel)
    if git(workspace, 'diff', '--cached', '--name-only', '--', rel):
        git(workspace, 'commit', '-qm', 'Freeze collected PF horizon metadata', '--', rel)
    commit = git(workspace, 'log', '-1', '--format=%H', '--', rel)
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),commit],
                   cwd=root, check=True, capture_output=True)
    return dict(commit=commit, path=rel)


def fixed_workspace(path, origin, source):
    if not (path/'.git').exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--shared','--no-checkout',str(origin),str(path)],check=True,capture_output=True)
        git(path, 'sparse-checkout', 'set', 'src', 'scripts')
        git(path, 'checkout', '--detach', source)
    if git(path, 'rev-parse', 'HEAD') != source:
        raise ValueError('Terminal signal workspace has a different fixed provider')


def collected_signal(owners, parent, output):
    """Recover an accepted original attempt; an ambiguous owner cannot launch again."""
    from .pr_candidate_completion import PreparationWait
    from .rpe_q3_coordinator import digest

    matches = []
    for prior, rows in owners:
        jobs = {row['jobs']['13']['signal'] for row in rows if row.get('source_configuration_id') == parent
                and row.get('M') == 13 and row.get('jobs',{}).get('13',{}).get('signal')}
        marker = prior/'pf_auto'/'horizon'/'signal-provider'/'horizon-inputs'/parent/'PF_SIGNALS.DISTRIBUTED.json'
        if marker.is_file():
            jobs.add(read(marker)['job_id'])
        matches.extend((prior,job) for job in jobs)
    matches = list(dict.fromkeys(matches))
    if not matches:
        return False
    if len(matches) != 1:
        raise PreparationWait('Conflicting terminal signal owners: '+str(matches))
    prior, job = matches[0]
    db = prior/'distributed'/'QUEUE.sqlite3'
    if not db.is_file():
        raise PreparationWait('Waiting for the original queue ledger at '+str(prior))
    from contextlib import closing
    with closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)) as conn:
        row = conn.execute('SELECT state, attempt_id, request_sha256 FROM jobs WHERE job_id=? AND method=?',
                           (job,'PREPARATION:signal')).fetchone()
    if not row or row[0] != 'ACCEPTED' or not row[1]:
        raise PreparationWait('Await original terminal signal '+job+' at '+str(prior))
    job_dir = prior/'jobs'/'PREPARATION:signal'/job
    request, outcome = read(job_dir/'REQUEST.json'), read(job_dir/'OUTCOME.json')
    result = read(job_dir/'PROCESS_RETURN.json')
    if (request['request_id'] != job or digest(request) != row[2] or outcome['request_sha256'] != row[2]
            or outcome['request_id'] != job or outcome['status'] != 'PREPARATION_COMPLETE'
            or outcome['source_manifest'] != request['preparation']['source']
            or result['returncode'] != 0 or result.get('signal') is not None):
        raise ValueError('Original terminal signal receipt differs from the accepted job')
    for name in ('PLAN.json','MEAN_HANDOFF.json','REFERENCE.json'):
        hashes = [v for k,v in outcome['artifacts'].items() if k.endswith('/result/'+name)]
        if len(hashes) != 1:
            raise ValueError('Original signal requires one returned '+name)
        checksum, = hashes
        raw = (prior/'distributed'/'artifacts'/checksum[:2]/checksum).read_bytes()
        if sha256(raw).hexdigest() != checksum:
            raise ValueError('Collected terminal evidence hash differs: '+name)
        output.mkdir(exist_ok=True, parents=True)
        destination = output/name
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError('Previously recovered terminal evidence differs: '+name)
        if not destination.exists():
            destination.write_bytes(raw)
    return prior, job


def _same_saved_document(reader, left, right):
    """Require exact selected JSON, including type and numeric representation."""
    if dict(left, selector=left.get('selector', [])) == dict(right, selector=right.get('selector', [])):
        return True
    return json.dumps(reader.read(left), sort_keys=True) == json.dumps(reader.read(right), sort_keys=True)


def parent_q3_contract(reader, route, entry, config):
    """Reuse an admitted case's evaluator or the ordinary PF template contract."""
    from run_rpe_optimizer_q3_job import CONTRACT, DROP, git_json

    fields = ('kernel_source', 'C_source', 'acceleration') + CONTRACT
    parent = config['id']
    radii = [1.0] * len(config['rounds'])

    def fixed_inputs(template, *, price_policy):
        if template['options']['phase_radii'] != radii:
            raise ValueError('Parent Q3 template differs from its fixed phase radii')
        inputs = {key: deepcopy(template[key]) for key in fields}
        if price_policy and 'price_policy' in template:
            inputs['price_policy'] = deepcopy(template['price_policy'])
        return inputs

    template_loc = entry.get('q3_inputs')
    if template_loc:
        templates = [reader.read(template_loc)]
    else:
        templates = [git_json(Path(route['view']), loc) for loc in route['templates']]
    matches = []
    for template in templates:
        for case in template['cases']:
            ids = {case[key] for key in ('configuration_id', 'original_configuration_id') if case.get(key)}
            if parent not in ids:
                continue
            if ids != {parent} or any(not _same_saved_document(reader, case[key], entry[key])
                                      for key in ('configuration', 'signal_packet')):
                raise ValueError('Parent Q3 template refers to different fixed candidate evidence')
            scientific_case = {key: val for key, val in case.items()
                               if key not in DROP | {'configuration', 'signal_packet', 'original_configuration_id'}}
            scientific_case['configuration_id'] = parent
            matches.append((fixed_inputs(template, price_policy=True), scientific_case))
    if matches:
        first = matches[0]
        if any(match != first for match in matches[1:]):
            raise ValueError('Saved parent Q3 cases disagree on their fixed evaluator')
        return first[0], first[1]['evaluation_target']
    if template_loc:
        raise ValueError('Registered Q3 template has no case for its fixed parent')

    # This is the same contract used by the native PF provider when registering
    # a new configuration. It must already be offered by this fixed route.
    for loc, template in zip(route['templates'], templates):
        if (loc['commit'], loc['path']) != PF_Q3_CONTRACT or loc.get('inputs_selector'):
            continue
        packet = reader.read(entry['signal_packet'])
        original = packet['target']
        if (config['spec']['law']['model'] != 'deterministic' or packet['configuration_id'] != parent
                or original['binding'] != config['spec']['binding']
                or [row['m'] for row in packet['means']] != list(range(len(radii)))):
            raise ValueError('Cannot bind the PF evaluator to a different scope or signal law')
        targets = [case['evaluation_target'] for case in template['cases']]
        if not targets:
            break
        target = targets[0]
        if (any(other != target for other in targets)
                or target['binding'] != original['binding']
                or target['energy_ha'] != original['reference_solver_energy_ha']
                or not _same_saved_document(reader, target['original_target_source'], original['source'])):
            raise ValueError('Fixed PF evaluator target differs from the admitted signal')
        return fixed_inputs(template, price_policy=False), deepcopy(target)
    raise ValueError('No saved template case or registered evaluator contract for this PF parent')


def bind_terminal_q3(root, parent, entry, routes, interpreter, leaf, reader, config):
    """Attach the preserved terminal signal to the fixed PF evaluator."""
    route = eligible_route(routes, entry['method'], parent)
    view = Path(route['view'])
    if git(view, 'rev-parse', 'HEAD') != route['view_commit']:
        raise ValueError('Parent binding runtime differs from its fixed route')
    template, evaluation_target = parent_q3_contract(reader, route, entry, config)
    vault = Path(leaf['workspace'])
    cfg_loc = leaf['configuration']
    cfg = reader.read(cfg_loc)
    binding_script = ('import json,sys;sys.path.insert(0,"src");'
                      'from trottertracks import rpe_total_binding as b;'
                      'cfg,loc=json.load(sys.stdin);v=b.create(".",cfg,original_source=loc);'
                      'print(json.dumps([v["id"],v["total_configuration"]["id"]]))')
    env = dict(os.environ)
    alternate = json.dumps(str(vault/'.git'/'objects'))
    env['GIT_ALTERNATE_OBJECT_DIRECTORIES'] = os.pathsep.join(filter(None, (
        alternate, env.get('GIT_ALTERNATE_OBJECT_DIRECTORIES'))))
    res = subprocess.run([interpreter, '-B', '-c', binding_script], cwd=view, env=env,
                         input=json.dumps([cfg, cfg_loc]), text=True, capture_output=True, check=True)
    binding, total = json.loads(res.stdout)
    template['options']['phase_radii'].append(1.0)
    # As in the native PF register path, the optimizer replaces the displayed
    # template allocation with the selected shot vector before Q3.
    allocation = dict(allocation_id='UNUSED_TEMPLATE_REPLACED_BY_OPTIMIZER', shot_policy='USER_SUPPLIED',
                      shots=[dict(m=row['m'], N_X=1, N_Y=1) for row in cfg['rounds']])
    template.update(cases=[dict(label='PF_SEARCH_TEMPLATE_'+cfg['id'][:12], configuration=cfg_loc,
                                configuration_id=cfg['id'], signal_packet=leaf['entry']['signal_packet'],
                                binding_id=binding, total_view_id=total, evaluation_target=evaluation_target,
                                allocation=allocation)],
                    concurrent_configurations=1, maximum_scientific_calls=1)
    tpl = vault/'Q3_INPUTS.json'
    if tpl.exists() and read(tpl) != template:
        raise ValueError('Previously prepared terminal Q3 template differs')
    write(tpl, template)
    q3 = publish(vault, root, tpl)
    extension = dict(leaf['entry'], q3_inputs=q3, binding_id=binding, total_configuration_id=total)
    return dict(leaf, entry=extension, signal_only=False)


def produce(root, runtime, parent, entry, routes, interpreter, stop_requested, owners=(), *, bind_q3=True):
    """Run one qualified missing signal. Reconcile the same marker on Resume."""
    from run_rpe_pf_horizon_signal import ENTRY, PROVIDER
    from run_rpe_optimizer_q3_job import M13_RISK
    from .pf_optimizer_connection import physical_ids
    from .pr_candidate_completion import Held

    if Path(parent).name != parent or parent in ('','.','..'):
        raise ValueError('Invalid parent configuration ID')
    root = Path(root).resolve()
    run = runtime.executor.run_dir
    base = run/'pf_auto'/'horizon'/parent
    reader = GitReader(root)
    config = reader.read(entry['configuration'])
    if config['id'] != parent or config['spec']['M'] != 12 or config['spec']['binding']['system'] != 'H4':
        raise ValueError('Parent differs from qualified H4 PF horizon domain')
    parent_locs = {k:entry[k] for k in ('configuration','signal_packet')}
    checkpoint = base/'LEAF.json'
    if checkpoint.is_file():
        leaf = read(checkpoint)
        if leaf['entry']['horizon_parent'] != parent_locs:
            raise ValueError('Collected horizon has different fixed parent inputs')
        if leaf.get('signal_only') is not True or not bind_q3:
            return leaf
        if job := leaf.get('current_run_signal_job'):
            runtime.executor.adopt_preparation(job)
        if stop_requested():
            raise Held('STOP_REQUESTED')
        return bind_terminal_q3(root, parent, entry, routes, interpreter, leaf, reader, config)
    shared = run/'pf_auto'/'horizon'/'signal-provider'
    # Called serially by the terminal producer. All parents share an immutable
    # science checkout. Each queued request has its own stable payload path.
    fixed_workspace(shared, root, PROVIDER)
    directory = shared/'horizon-inputs'/parent
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory/'REQUEST_INPUT.json'
    caller = Path(__file__).resolve().parents[2]/ENTRY
    args = [interpreter,'-B',str(caller),'--provider-root',str(shared),'--request',str(payload)]
    plan_file = base/'plan'/'PLAN.json'
    if not plan_file.is_file():
        from tempfile import TemporaryDirectory
        write(payload, parent_locs)
        base.mkdir(parents=True, exist_ok=True)
        # Planning only reads Git and the saved parent. The worker resolves
        # execution Python from its profile; older PF launchers only accept
        # their original science entries and cannot run this planner.
        planner = [sys.executable, *args[1:]]
        if source := runtime.settings.get('horizon_transport_source'):
            planner.extend(('--caller-source', source))
        with TemporaryDirectory(dir=base) as scratch:
            completed = subprocess.run([*planner,'--output',scratch,'--mode','plan'],
                                       capture_output=True, text=True)
            if completed.returncode:
                errors = completed.stderr.strip().splitlines()
                reason = errors[-1] if errors else f'exit {completed.returncode}'
                raise ValueError('Terminal PF planning failed: ' + reason)
            write(plan_file, read(Path(scratch)/'PLAN.json'))
    planned = read(plan_file)
    if planned['parent'] != parent_locs or planned['requested_rounds'] != [13]:
        raise ValueError('Saved terminal signal plan differs from current parent')
    physical = planned['physical_round']['id']
    write(payload, dict(parent_locs, physical_id=physical))
    provider_files = ('scripts/prepare_pf_candidate.py','src/trottertracks/pf_evidence_preparation.py')
    manifest = dict(commit=PROVIDER, files={name:sha256((shared/name).read_bytes()).hexdigest() for name in provider_files})
    context = SimpleNamespace(source=PROVIDER, preparation_root=shared, preparation_source_manifest=manifest,
                              log=runtime.executor.log)
    result = directory/'result'
    if stop_requested():
        raise Held('STOP_REQUESTED')
    recovered = collected_signal(owners, parent, result)
    if not recovered:
        runtime.call(context, [*args,'--output',str(result),'--mode','execute'], directory, 'PF_SIGNALS', wait=False)
    plan = read(result/'PLAN.json')
    cfg = plan['configuration']
    packet = read(result/'MEAN_HANDOFF.json')
    if (plan['parent'] != read(payload) or plan['configuration'] != planned['configuration']
            or plan['physical_round'] != planned['physical_round']
            or packet.get('prefix_source') != entry['signal_packet'] or packet['configuration_id'] != cfg['id']
            or [row['m'] for row in packet['means']] != list(range(cfg['spec']['M']+1))
            or packet['means'][-1]['physical_candidate']['id'] != physical
            or physical_ids(root, cfg)[cfg['spec']['M']] != physical):
        raise ValueError('Returned terminal result differs from the fixed parent and plan')
    # Keep just these evidence files in a small per-parent Git repository.
    # Objects needed by the published binding/price guards are borrowed locally.
    vault = base/'evidence'
    if not (vault/'.git').exists():
        subprocess.run(['git','clone','--shared','--no-checkout',str(root),str(vault)],check=True,capture_output=True)
    for name, record in (('CONFIGURATION.json',cfg), ('MEAN_HANDOFF.json',packet), ('PLAN.json',plan)):
        path = vault/name
        if path.exists() and read(path) != record:
            raise ValueError('Saved terminal evidence changed: '+name)
        write(path, record)
    raw_reference = (result/'REFERENCE.json').read_bytes()
    ref_file = vault/'REFERENCE.json'
    if ref_file.is_file() and ref_file.read_bytes() != raw_reference:
        raise ValueError('Collected full-H reference differs on recovery')
    ref_file.write_bytes(raw_reference)
    reference = dict(publish(vault, root, ref_file), sha256=sha256(raw_reference).hexdigest())
    cfg_loc = publish(vault, root, vault/'CONFIGURATION.json')
    sig_loc = publish(vault, root, vault/'MEAN_HANDOFF.json')
    extension = {k:entry[k] for k in ('method','p','formula','delta_over_h')}
    if entry.get('price_scope') is not None:
        extension['price_scope'] = entry['price_scope']
    extension.update(configuration=cfg_loc,signal_packet=sig_loc,
                     horizon_parent=parent_locs,horizon_risk_source=M13_RISK,
                     horizon_source_configuration_id=parent)
    leaf = dict(workspace=str(vault), relative_output='.', configuration=cfg_loc, physical_id=physical,
                plan_sha256=sha256((vault/'PLAN.json').read_bytes()).hexdigest(), entry=extension,
                terminal_reference=reference, signal_only=True)
    if recovered and recovered[0].resolve() == run.resolve():
        leaf['current_run_signal_job'] = recovered[1]
    write(checkpoint, leaf)
    if job := leaf.get('current_run_signal_job'):
        runtime.executor.adopt_preparation(job)
    if not bind_q3:
        return leaf
    if stop_requested():
        raise Held('STOP_REQUESTED')
    return bind_terminal_q3(root, parent, entry, routes, interpreter, leaf, reader, config)


def complete_scoped(root, runtime, candidates, methods, horizons, routes, interpreter, prior_runs,
                    *, stop_requested, on_ready, refresh_requested=None, exploratory=False):
    from .pf_distributed_preparation import complete_terminal_prices
    from .pr_candidate_completion import Held, PreparationWait

    explicit = runtime.settings.get('pf_terminal_price_leaves', {})
    owned = (runtime.settings.get('horizon_extensions') or {}).get('bindings', {})
    run = runtime.executor.run_dir
    base = run/'pf_auto'/'horizon'
    scoped = {c['configuration_id']: entry for c in candidates if c['method'] in methods
              and c.get('configuration_id') and (entry := parent_entry(c))}
    complete_terminal_prices(root, runtime, scoped, selected=explicit,
                             stop_requested=stop_requested, on_ready=on_ready)
    saved_signals = saved_parent_signals(root, run, candidates, methods, prior_runs) if 13 in horizons else {}
    pending = parents(candidates, methods, horizons, {}, explicit, saved_signals)
    state_path = run/'pf_auto'/'TERMINAL_PRICES.json'
    saved = read(state_path) if state_path.exists() else {}
    inherited = []
    owners = []
    current_ledger = run/'HORIZON_PREPARATION.json'
    catalog_rows = read(current_ledger) if current_ledger.is_file() else []
    for path in prior_runs:
        prior = Path(path).resolve()
        if prior != run.resolve() and prior.is_dir():
            receipts = prior/'pf_auto'/'TERMINAL_PRICES.json'
            inherited.append((prior, read(receipts) if receipts.is_file() else {}))
            catalog = prior/'HORIZON_PREPARATION.json'
            owners.append((prior,read(catalog) if catalog.is_file() else []))

    def parent_matches(actual, expected):
        if actual == expected:
            return True
        reader = GitReader(root)
        return actual.keys() == expected.keys() and all(
            _same_saved_document(reader, actual[key], expected[key]) for key in actual)

    def advance(item):
        parent, entry = item
        report = base/parent/'PREPARATION.json'
        try:
            locs = {k:entry[k] for k in ('configuration','signal_packet')}
            if parent in saved and saved[parent]['status'] == 'READY':
                record = saved[parent]
                if (not parent_matches(record['entry']['horizon_parent'], locs)
                        or record.get('parent_prices') != entry.get('prices')):
                    raise Held('Previously adopted terminal prices have different parent inputs')
                on_ready(record['configuration_id'], record['entry'])
                return None
            carried = next((states[parent] for _,states in inherited if states.get(parent, {}).get('status') == 'READY'
                         and states[parent]['entry'].get('horizon_parent') == locs
                         and states[parent].get('parent_prices') == entry.get('prices')), None)
            if carried:
                # Native hydrate checks all physical IDs, price axes and signal bindings again.
                latest = read(state_path) if state_path.exists() else {}
                latest[parent] = carried
                write(state_path, latest)
                on_ready(carried['configuration_id'], carried['entry'])
                return None
            selected_owners = owners
            if (owned.get(parent, {}).get('rounds', {}).get('13') or {}).get('signal'):
                if not any(row.get('source_configuration_id') == parent and row.get('M') == 13
                           and row.get('jobs',{}).get('13',{}).get('signal') for row in catalog_rows):
                    raise Held('Waiting for the catalog-owned terminal request')
                selected_owners = [(run,catalog_rows), *owners]
            leaf_file = base/parent/'LEAF.json'
            leaf = read(leaf_file) if leaf_file.is_file() else None
            if leaf and leaf['entry']['horizon_parent'] != locs:
                original = leaf['entry']['horizon_parent']
                if not parent_matches(original, locs):
                    raise ValueError('Collected horizon has different fixed parent inputs')
                entry = dict(entry, **original)
            elif leaf is None:
                plan_file = base/parent/'plan'/'PLAN.json'
                if plan_file.is_file():
                    original = read(plan_file)['parent']
                    if not parent_matches(original, locs):
                        raise ValueError('Existing terminal signal request has different fixed parent inputs')
                    entry = dict(entry, **original)
            if leaf is None or leaf.get('signal_only') is True:
                leaf = produce(root, runtime, parent, entry, routes, interpreter, stop_requested, selected_owners,
                               bind_q3=exploratory or bool(entry.get('prices')) or parent not in saved_signals)
                write(leaf_file, leaf)
            if exploratory and not leaf.get('signal_only'):
                on_ready(GitReader(root).read(leaf['configuration'])['id'], leaf['entry'])
            if entry.get('prices'):
                if 'price_scope' not in leaf['entry']:
                    leaf = dict(leaf, entry=dict(leaf['entry'],price_scope=GitReader(root).read(entry['prices'])['scope']))
                complete_terminal_prices(root,runtime,{parent:entry},selected={parent:leaf},
                                         stop_requested=stop_requested,on_ready=on_ready)
                latest = read(state_path).get(parent, {})
                write(report, dict(status=latest.get('status','WAITING_FOR_PARENT_PRICES'),
                                   configuration_id=parent, reason=latest.get('reason')))
            else:
                write(report, dict(status='WAITING_FOR_PARENT_PRICES',configuration_id=parent))
        except PreparationWait as error:
            fields = dict(status='WAITING_DISTRIBUTED_PREPARATION', configuration_id=parent, reason=str(error))
            if error.job_id:
                fields['job_id'] = error.job_id
            write(report, fields)
            return error.job_id
        except (Held, ValueError, KeyError, OSError, sqlite3.Error, subprocess.CalledProcessError) as error:
            write(report, dict(status='PREPARATION_INCOMPLETE',configuration_id=parent,reason=str(error)))
        return None

    def yield_watcher():
        return stop_requested() or (refresh_requested is not None and refresh_requested())

    # Feed new work before re-publishing the saved READY entries. A new
    # optimizer view can reuse those receipts while the missing stages run.
    ordered = sorted(pending.items(), key=lambda item: saved.get(item[0], {}).get('status') == 'READY')
    runtime.follow_preparations(ordered, advance, yield_watcher)
