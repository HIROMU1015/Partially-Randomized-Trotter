"""Extend admitted PF parents through the normal signal queue and price registry."""
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


def parents(candidates, methods, horizons, bindings, existing):
    """Only consider parents present in this optimizer's scoped candidate list."""
    from run_rpe_optimizer_q3_job import M13_ROUTES
    if 13 not in horizons:
        return {}
    native = {c['source_configuration_id'] for c in candidates
              if c.get('source_configuration_id') and c.get('M') == 13 and c.get('ready')}
    selected = {}
    for c in candidates:
        cid, method, entry = c.get('configuration_id'), c['method'], parent_entry(c)
        if (method in methods and (method, 'pf_b_pf6') in M13_ROUTES and cid
                and c.get('M', 12) == 12 and not c.get('source_configuration_id')
                and cid not in bindings and cid not in existing and cid not in native
                and entry.get('method') == method and entry.get('configuration') and entry.get('signal_packet')):
            selected[cid] = entry
    return selected


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
    return True


def produce(root, runtime, parent, entry, routes, interpreter, stop_requested, owners=()):
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
        with TemporaryDirectory(dir=base) as scratch:
            subprocess.run([*args,'--output',scratch,'--mode','plan'],check=True,capture_output=True)
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
    if not collected_signal(owners, parent, result):
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
    route = eligible_route(routes, entry['method'], parent)
    view = Path(route['view'])
    if git(view,'rev-parse','HEAD') != route['view_commit']:
        raise ValueError('Parent binding runtime differs from its fixed route')
    binding_script = ('import json,sys;sys.path.insert(0,"src");'
                      'from trottertracks import rpe_total_binding as b;'
                      'cfg,loc=json.load(sys.stdin);v=b.create(".",cfg,original_source=loc);'
                      'print(json.dumps([v["id"],v["total_configuration"]["id"]]))')
    env = dict(os.environ)
    alternate = json.dumps(str(vault/'.git'/'objects'))
    env['GIT_ALTERNATE_OBJECT_DIRECTORIES'] = os.pathsep.join(filter(None, (
        alternate, env.get('GIT_ALTERNATE_OBJECT_DIRECTORIES'))))
    res = subprocess.run([interpreter,'-B','-c',binding_script],cwd=view,env=env,
        input=json.dumps([cfg,cfg_loc]),text=True,capture_output=True,check=True)
    binding,total = json.loads(res.stdout)
    template_loc = entry.get('q3_inputs')
    if not template_loc:
        from run_rpe_optimizer_q3_job import template as resolve_template
        _, template, _ = resolve_template(view, dict(route=route, configuration_id=parent))
    else:
        template = reader.read(template_loc)
    cases = [c for c in template['cases'] if (c.get('configuration_id') or c.get('original_configuration_id')) == parent]
    radii = template['options']['phase_radii']
    if (len(cases) != 1 or cases[0]['signal_packet'] != entry['signal_packet']
            or cases[0]['configuration'] != entry['configuration'] or radii != [1.0]*len(config['rounds'])):
        raise ValueError('Parent Q3 template differs from its fixed configuration, signal or radii')
    template['options']['phase_radii'] = [*radii,1.0]
    template['cases'] = [dict(cases[0], configuration=cfg_loc, configuration_id=cfg['id'],
                               signal_packet=sig_loc, binding_id=binding, total_view_id=total)]
    tpl = vault/'Q3_INPUTS.json'
    write(tpl, template)
    q3 = publish(vault, root, tpl)
    extension = {k:entry[k] for k in ('method','p','formula','delta_over_h')}
    if entry.get('price_scope') is not None:
        extension['price_scope'] = entry['price_scope']
    extension.update(configuration=cfg_loc,signal_packet=sig_loc,q3_inputs=q3,binding_id=binding,
                     total_configuration_id=total,horizon_parent=parent_locs,horizon_risk_source=M13_RISK,
                     horizon_source_configuration_id=parent)
    return dict(workspace=str(vault),relative_output='.',configuration=cfg_loc,physical_id=physical,
                plan_sha256=sha256((vault/'PLAN.json').read_bytes()).hexdigest(),entry=extension,
                terminal_reference=reference)


def complete_scoped(root, runtime, candidates, methods, horizons, routes, interpreter, prior_runs,
                    *, stop_requested, on_ready):
    from .pf_distributed_preparation import complete_terminal_prices
    from .pr_candidate_completion import Held

    explicit = runtime.settings.get('pf_terminal_price_leaves', {})
    owned = (runtime.settings.get('horizon_extensions') or {}).get('bindings', {})
    run = runtime.executor.run_dir
    base = run/'pf_auto'/'horizon'
    scoped = {c['configuration_id']: entry for c in candidates if c['method'] in methods
              and c.get('configuration_id') and (entry := parent_entry(c))}
    complete_terminal_prices(root, runtime, scoped, selected=explicit,
                             stop_requested=stop_requested, on_ready=on_ready)
    pending = parents(candidates, methods, horizons, {}, explicit)
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
    for parent, entry in pending.items():
        if stop_requested():
            return
        report = base/parent/'PREPARATION.json'
        try:
            if parent in saved and saved[parent]['status'] == 'READY':
                record = saved[parent]
                if (record['entry']['horizon_parent'] != {k:entry[k] for k in ('configuration','signal_packet')}
                        or record.get('parent_prices') != entry.get('prices')):
                    raise Held('Previously adopted terminal prices have different parent inputs')
                on_ready(record['configuration_id'], record['entry'])
                continue
            locs = {k:entry[k] for k in ('configuration','signal_packet')}
            carried = next((states[parent] for _,states in inherited if states.get(parent, {}).get('status') == 'READY'
                         and states[parent]['entry'].get('horizon_parent') == locs
                         and states[parent].get('parent_prices') == entry.get('prices')), None)
            if carried:
                # Native hydrate checks all physical IDs, price axes and signal bindings again.
                latest = read(state_path) if state_path.exists() else {}
                latest[parent] = carried
                write(state_path, latest)
                on_ready(carried['configuration_id'], carried['entry'])
                continue
            selected_owners = owners
            if (owned.get(parent, {}).get('rounds', {}).get('13') or {}).get('signal'):
                if not any(row.get('source_configuration_id') == parent and row.get('M') == 13
                           and row.get('jobs',{}).get('13',{}).get('signal') for row in catalog_rows):
                    raise Held('Waiting for the catalog-owned terminal request')
                selected_owners = [(run,catalog_rows), *owners]
            leaf_file = base/parent/'LEAF.json'
            if not leaf_file.is_file():
                leaf = produce(root, runtime, parent, entry, routes, interpreter, stop_requested, selected_owners)
                write(leaf_file, leaf)
            else:
                leaf = read(leaf_file)
                if leaf['entry']['horizon_parent'] != locs:
                    raise ValueError('Collected horizon has different fixed parent inputs')
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
        except (Held, ValueError, KeyError, OSError, sqlite3.Error, subprocess.CalledProcessError) as error:
            write(report, dict(status='PREPARATION_INCOMPLETE',configuration_id=parent,reason=str(error)))
