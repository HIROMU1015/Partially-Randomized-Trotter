"""PF preparation decisions before the existing shot/Q3 search.

Only metadata ranks incomplete candidates. Saved T1/T2 evidence orders requests;
no shift, structural gate count or local diagnostic is a Q3 veto/price bound.
The selected canonical configuration goes to D's existing completion protocol.
"""
from copy import deepcopy
from fractions import Fraction
import json
import hashlib
from pathlib import Path
import shutil

from . import pf_optimizer_connection as PF

RULE='UNSEEN_PREFIX_SAVED_EVIDENCE_V1'
PIECES={'2nd':1,'4th':3,'4th(m5_best)':11}
PRICE_RECHECK_STATES = frozenset(('PRICE_EVIDENCE_REQUIRED', 'PRIOR_OWNER_HOLD',
                                'PRICE_RECHECK_AVAILABLE', 'PRICE_ACTION_UNAVAILABLE',
                                'PRICE_OWNER_WAITING'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('Preserve prior PF selection: '+str(path))
    else:path.write_text(text)


def evaluated_prefixes(context):
    return sorted({p['L_D'] for p in context['catalog']['candidates']
        if p['method']=='TRUNCATED_PF' and p.get('q3_status') in ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD','INCONCLUSIVE') and type(p.get('L_D'))is int})


def inspect(context,methods,*,include_full=False):
    evaluated=evaluated_prefixes(context);rows=[]
    for c in context['candidates']:
        if c['method'] not in methods:continue
        if c['method']!='TRUNCATED_PF' and not (include_full and c['method']=='FULL_PF'
                                               and c.get('configuration_body')):continue
        k=c['key'];e=c.get('priority_evidence')or{};t1=e.get('T1')or{}
        p=k.get('p');width=Fraction(k['delta_over_h']);shift=t1.get('delta_E_trunc_B0I_ha')
        score=(0 if p not in evaluated else 1,-len(e.get('T2')or[]),abs(float(shift))if shift is not None else float('inf'),PIECES.get(k['formula'],999),-width,c['configuration_id'])
        rows.append(dict(configuration_id=c['configuration_id'],method=c['method'],key=deepcopy(k),ready=c['ready'],needs=deepcopy(c['needs']),
            prefix_has_saved_Q3=p in evaluated,priority=dict(unseen_prefix_first=score[0],saved_T2_step_descriptors=len(e.get('T2')or[]),saved_abs_T1_shift=None if shift is None else abs(float(shift)),signed_piece_count=score[3],coarser_requested_width_tiebreak=str(width)),
            evidence=deepcopy(e),_score=score))
    rows.sort(key=lambda r:r['_score'])
    for r in rows:r.pop('_score')
    return rows,evaluated


def select(context,methods,excluded=(),*,include_full=False):
    rows,evaluated=inspect(context,methods,include_full=include_full)
    eligible=[r for r in rows if not r['ready'] and r['needs'] and r['configuration_id'] not in excluded]
    chosen=deepcopy(eligible[0])if eligible else None
    incumbents=[p for p in context['catalog']['candidates']if p['method']=='TRUNCATED_PF'and p['status']=='FEASIBLE'and p.get('conditional_cost')is not None]
    from .saved_rpe_cost_catalog import frac
    incumbent=min(incumbents,key=lambda p:frac(p['conditional_cost']))if incumbents else None
    return dict(schema='PF_AUTOMATIC_PREPARATION_SELECTION_V1',rule=RULE,selection_source='OPTIMIZER_INCOMPLETE_CANDIDATE_QUEUE',
        priority_scope='REQUEST_ORDER_ONLY_NOT_Q3_REJECTION_OR_PRICE_BOUND',candidate_count=len(rows),candidates=rows,
        evaluated_prefixes=evaluated,unevaluated_prefixes=sorted({r['key']['p']for r in rows}-set(evaluated)),
        p10_alias='FULL_PF_CANONICAL_ID; NEVER_A_SECOND_B0I_EVALUATION',
        saved_incumbent=None if incumbent is None else {k:deepcopy(incumbent.get(k))for k in ('configuration_id','L_D','formula','conditional_cost','q3_status','N_X','N_Y')},
        selected=chosen,reason=None if chosen is None else 'UNSEEN_PREFIX_THEN_SAVED_STEP_REUSE_THEN_T1_SOFT_PRIORITY_THEN_SIGNED_PIECES_AND_WIDTH; ALL_OTHER_CANDIDATES_REMAIN_ELIGIBLE')


def _artifact_mode(settings,routes,methods):
    if settings.get('mode')!='artificial':return False
    active=[r for m in methods for r in routes.get(m,[])]
    if not active or any(r.get('kind')!='artificial'for r in active):
        raise ValueError('Artificial PF preparation requires only artificial Q3 routes')
    return True


def _refresh(root,problem,context,methods,registry,artificial):
    from . import rpe_method_optimizer as O
    updated=dict(problem,pf_evidence_registry=registry,artificial_pf_evidence=artificial)
    after=O.prepare(root,updated,methods=methods,preparation_dir=context.get('preparation_dir'))
    return after


def prepare(root,problem,context,spec,run,routes,*,stage_executor=None,excluded=(),selection=None):
    """One automatic finite selection, existing D request, same optimizer context.

    plan never starts science; complete is the existing explicitly authorized
    provider path. artificial is only a fixture route and cannot enter C.
    A saved request/response is reused on resume, including UNKNOWN/STOP.
    """
    root=Path(root);run=Path(run);methods=spec['methods']
    settings=deepcopy(spec.get('pf_completion')or problem.get('pf_completion')or{})
    automatic_scope=problem.get('methods',{}).get('TRUNCATED_PF',{})
    if not settings and ('TRUNCATED_PF' not in methods or automatic_scope.get('p_values') is None):
        return context,routes,None
    settings.setdefault('mode','plan');mode=settings['mode']
    if mode not in ('plan','complete','artificial'):raise ValueError('Unknown PF preparation mode')
    artificial=_artifact_mode(settings,routes,methods)
    folder=run/'pf_auto';folder.mkdir(parents=True,exist_ok=True)
    selection_path=folder/'SELECTION.json'
    manual=bool(settings.get('selected'))
    if manual:
        selection=dict(schema='PF_AUTOMATIC_PREPARATION_SELECTION_V1',selection_source='CALLER_SUPPLIED_SELECTED_LEGACY',selected=settings['selected'])
        write(selection_path,selection);selected=deepcopy(settings['selected']);chosen_id=None
    else:
        if selection_path.exists():selection=json.loads(selection_path.read_text())
        else:
            selection=selection or select(context,methods,excluded,include_full='TRUNCATED_PF' not in methods)
            write(selection_path,selection)
        chosen=selection['selected']
        if chosen is None:return context,routes,dict(status='NO_INCOMPLETE_PF_CANDIDATE',selection=selection)
        matches=[c for c in context['candidates']if c['configuration_id']==chosen['configuration_id']]
        if len(matches)!=1 or matches[0]['key']!=chosen['key']:raise ValueError('Selected PF identity changed on resume')
        candidate=matches[0];chosen_id=candidate['configuration_id']
        if not candidate.get('configuration_body'):raise ValueError('Selected PF candidate lacks canonical metadata')
        selected=[dict(configuration_body=deepcopy(candidate['configuration_body']))]
    if settings.get('max_candidates',1)!=1:raise ValueError('This finite preparation hook selects one candidate per round')
    status_path=folder/'STATUS.json'
    if not settings.get('entry'):
        status=dict(status='PLANNED_D_ENTRY_REQUIRED',selection=selection,selected=selected,required_next='Configure the fixed D provider workspace/entry, then use its existing finite plan; no p/shot selection is required',new_scientific_calls=0)
        write(folder/'D_REQUEST.json',dict(selected=selected,mode='plan'));write(status_path,status)
        return context,routes,status
    completion={k:deepcopy(v)for k,v in settings.items()if k not in ('max_candidates','selected')}
    completion.update(selected=selected,mode='plan'if mode=='plan'else'complete')
    if stage_executor is not None:
        binding=next((r for m in ('FULL_PF','TRUNCATED_PF') for r in routes.get(m,[])
                      if r.get('kind')=='pf_b_pf6' and r.get('view') and r.get('view_commit')),None)
        if binding:
            completion['binding_view']={k:binding[k]for k in ('view','view_commit')}
    completion.setdefault('output_root',str(folder/'provider_output'))
    completion.setdefault('registry_file',str(folder/'REGISTRY.json'))
    helper_spec=dict(spec,methods=sorted({item['configuration_body']['spec']['split'].get('original_PF_method')=='B0I'and'TRUNCATED_PF'or'FULL_PF'for item in selected})if not manual else [m for m in methods if m in PF.METHODS],pf_completion=completion)
    from .pr_candidate_completion import Held, PreparationWait
    try:
        response=PF.complete_selected(root,helper_spec,folder,
            **({'stage_executor':stage_executor}if stage_executor is not None else{}))
    except Held as error:
        # The fixed selector and stage markers retain the owner and request.
        # A candidate lacking an eligible host/receipt must not cancel PR or
        # already READY PF work using the same distributed executor.
        return context,routes,dict(status='PREPARATION_WAITING' if isinstance(error, PreparationWait)
                                   else 'PREPARATION_INCOMPLETE',selection=selection,reason=str(error))
    except PF.CompletionRequestMismatch as error:
        return context,routes,dict(status='PREPARATION_INCOMPLETE',selection=selection,
                                   reason='PF_COMPLETION_REQUEST_MISMATCH',request_path=str(error.path),
                                   differing_fields=error.fields)
    # Keep D's original metadata plan reachable from the caller's fixed results,
    # without changing the producer plan or inventing missing scientific values.
    copied=[]
    for item in response.get('response',{}).get('items',[]):
        source=Path(item.get('output',''))/'PLAN.json'
        if source.is_file():
            target=folder/'D_PLAN.json'
            if target.exists()and target.read_bytes()!=source.read_bytes():raise ValueError('D plan changed on resume')
            if not target.exists():shutil.copyfile(source,target)
            copied.append(str(target))
    if mode=='plan':
        status=dict(status='D_PLAN_ONLY_NO_SCIENCE',selection=selection,response=response,copied_plans=copied,new_scientific_calls=0)
    else:
        context=_refresh(root,problem,context,methods,response['registry_file'],artificial)
        registry=context['pf_evidence_registry'];routes=PF.registry_routes(routes,registry)
        if chosen_id is not None:
            returned=next(c for c in context['candidates']if c['configuration_id']==chosen_id)
            if returned['key']!=selection['selected']['key']:raise ValueError('Preparation returned another p/formula/width')
            ready=returned['ready'];needs=returned['needs']
        else:ready=True;needs=[]
        status=dict(status='READY_RETURNED_TO_SAME_OPTIMIZER'if ready else'PREPARATION_INCOMPLETE',selection=selection,response=response,artificial_only=artificial,remaining_needs=needs)
    price_holds = [row for row in response['response'].get('items', [])
                   if row.get('status') in ('PRICE_EVIDENCE_REQUIRED', 'PRIOR_OWNER_HOLD')]
    if price_holds and mode!='plan':
        item = price_holds[0]
        owner_input = folder/'pf_completion/ORIGINAL_OWNER_REQUEST.json'
        owner = json.loads(owner_input.read_text()) if owner_input.exists() else {}
        directories = (set(owner.get('prior_owner_directories', ())) |
                       set(item.get('original_owner_directories', ())))
        status.update(status=item['status'], reason=item['reason'],
                      price_reconciliation={**item, 'original_owner_directories':
                                            sorted(directories)})
    write(status_path,status)
    return context,routes,status


def repair_metadata_waves(records, repairs):
    """Re-select only a wave whose original owner check failed before a stage."""
    from .rpe_search_continuation import _has_local_owner
    repaired={}
    records=deepcopy(records)
    for repair in repairs:
        path=Path(repair['state_file']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=repair['sha256']:
            raise ValueError('PF metadata repair wave changed')
        original=json.loads(path.read_text());parent=path.parent.parent
        launch=json.loads((parent/'LAUNCH.json').read_text())
        ended=json.loads((parent/'EXIT.json').read_text())
        failure=json.loads((parent/'KAPPA_STATE.json').read_text()).get('stop')or{}
        if (launch.get('invocation_id')!=ended.get('invocation_id') or ended.get('returncode')!=1
                or failure.get('reason')!='RUNTIME_EXCEPTION' or failure.get('error_type')!='CalledProcessError'
                or 'check_pf_preparation_owner.py' not in failure.get('message','')
                or 'returned non-zero exit status 1' not in failure.get('message','')
                or _has_local_owner(launch.get('pid'),parent)):
            raise ValueError('PF metadata repair requires an ended owner-check failure')
        for cid in repair['configuration_ids']:
            row=original[cid];directory=path.parent/'candidates'/cid
            expected={'pf_auto/SELECTION.json','pf_auto/pf_completion/REQUEST.json',
                      'pf_auto/pf_completion/ORIGINAL_OWNER_REQUEST.json'}
            files={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
            if (row.get('status')!='PREPARATION_SCHEDULED' or row.get('configurations')
                    or records.get(cid)!=row or files!=expected or cid not in failure['message']):
                raise ValueError('PF metadata repair would retry a stage or change its selection: '+cid)
            selection=json.loads((directory/'pf_auto/SELECTION.json').read_text())['selected']
            request=json.loads((directory/'pf_auto/pf_completion/REQUEST.json').read_text())
            owner=json.loads((directory/'pf_auto/pf_completion/ORIGINAL_OWNER_REQUEST.json').read_text())
            if (selection['configuration_id']!=cid or selection['key']!=row['key']
                    or [v['configuration_body']['id'] for v in request['selected']]!=[cid]
                    or [v['configuration_body']['id'] for v in owner['selected']]!=[cid]):
                raise ValueError('PF metadata repair candidate identity differs: '+cid)
            repaired[cid]=dict(original_record=deepcopy(row),state_file=str(path),sha256=repair['sha256'],
                               reason='OWNER_CHECK_FAILED_BEFORE_ANY_SCIENTIFIC_STAGE')
            records.pop(cid)
    return records,repaired


def prepare_available(root, problem, context, spec, run, routes, stage_executor, *, stop_requested, announce, wait_for_work=None):
    """Run the next bounded wave as soon as its predecessor is saved."""
    recheck_waiting = True
    previous = None

    def publish_ready(context, routes):
        nonlocal previous
        if context is not previous:
            announce(context, routes)
            previous = context

    publish_ready(context, routes)
    while True:
        context, routes, status = prepare_next(root, problem, context, spec, run, routes, stage_executor,
                                               stop_requested=stop_requested, recheck_waiting=recheck_waiting,
                                               announce=publish_ready)
        recheck_waiting = False
        publish_ready(context, routes)
        if status['status'] != 'PREPARATION_WAVE_PROCESSED' or stop_requested():
            return context, routes, status
        if wait_for_work is not None and not wait_for_work():
            return context, routes, dict(status='PREPARATION_WAVE_YIELDED', records=status['records'])


def restore_price_checks(waves, records):
    """Recover legacy summary rows from provider checks, before normal duplicate checks."""
    from .pf_saved_prices import candidate_roots

    roots = [base for wave in waves for base in reversed(candidate_roots(wave))]
    for cid, row in records.items():
        if (row.get('status') != 'PREPARATION_INCOMPLETE' or row.get('configurations')
                or row.get('reason') != 'PF_EVIDENCE_INCOMPLETE'):
            continue
        # Prefer the current run's detailed receipt, then its older inputs.
        for base in roots:
            folder = base/cid/'pf_auto'
            leaf = folder/'pf_completion'
            if (leaf/'PROVIDER_SOURCE.json').exists() or (leaf/'provider_workspace').exists():
                continue
            if not (leaf/'RESPONSE.json').is_file():
                continue
            try:
                selected = json.loads((folder/'SELECTION.json').read_bytes())['selected']
                check = json.loads((leaf/'ORIGINAL_OWNER_CHECK.json').read_bytes())
                items = json.loads((leaf/'RESPONSE.json').read_bytes())['items']
                owner = json.loads((leaf/'ORIGINAL_OWNER_REQUEST.json').read_bytes())
                if (selected['configuration_id'] != cid or selected['key'] != row['key']
                        or [i['configuration_body']['id'] for i in owner['selected']] != [cid]
                        or check['rows'] != items or len(items) != 1 or items[0]['configuration_id'] != cid
                        or items[0]['status'] not in ('PRIOR_OWNER_HOLD', 'PRICE_EVIDENCE_REQUIRED')):
                    continue
            except (OSError, KeyError, TypeError, ValueError):
                continue
            item = items[0]
            directories = set(owner.get('prior_owner_directories', ())) | set(item.get('original_owner_directories', ()))
            directories.add(str(Path(owner['output_root'])/cid))
            row.update(status=item['status'], reason=item['reason'], method=selected['method'],
                       price_reconciliation=dict(item, original_owner_directories=sorted(directories)),
                       price_prior_owner_directories=sorted(directories), prior_price_check=str(leaf/'ORIGINAL_OWNER_CHECK.json'))
            break


def reconsider_prices(root, run, spec, settings, records):
    from concurrent.futures import ThreadPoolExecutor
    from .pf_saved_prices import (completed_partial, prior_waves, receipt_locators, receipt_objects,
                                  reconciliation_locators, reconciliation_snapshot)

    waves = [run/'pf_auto', *prior_waves(run, spec)]
    restore_price_checks(waves, records)
    pending = [(cid, row) for cid, row in records.items()
               if row['status'] in PRICE_RECHECK_STATES and not row.get('configurations')]
    if not pending:
        return
    audits = receipt_locators(waves, index=True)
    locators = {cid: reconciliation_locators(waves, settings, row, audits=audits) for cid, row in pending}
    objects = receipt_objects(root, (loc['commit']+':'+loc['path']
                                    for receipts in locators.values() for loc in receipts))

    def inspect(item):
        cid, row = item
        fingerprint = reconciliation_snapshot(root, row, locators[cid], objects=objects)
        provider = spec.get('pf_price_recheck_provider')
        if provider:
            fingerprint = provider['source'] + ':' + fingerprint
        elif settings.get('selected_axis_compile') is True:
            fingerprint = 'SELECTED_AXIS_V1:'+fingerprint
        return cid, row, fingerprint

    with ThreadPoolExecutor(max_workers=settings.get('max_candidates', 1)) as pool:
        decisions = list(pool.map(inspect, pending))
    for cid, row, snapshot in decisions:
        if row.get('price_recheck_snapshot') == snapshot:
            continue
        previous = run/row.get('price_recheck_directory', (
            'pf_auto/price-recovery' if row.get('price_receipt_recheck') else 'pf_auto'))
        leaf = previous/'candidates'/cid/'pf_auto/pf_completion'
        prior_outputs = row.get('price_prior_owner_directories', [])
        provider = spec.get('pf_price_recheck_provider')
        if provider:
            # The new provider's default owner paths differ from those of
            # the original D checkout. Retain both duplicate checks.
            original = (Path(root) / settings['entry']).resolve().parents[1]
            prior_outputs = sorted(set(prior_outputs) | {
                str(original / '.pf_evidence_private' / cid),
                str(original / 'docs/research/d_pf_on_demand_evidence_20261116/selected' / cid)})
            row['price_recheck_provider_source'] = provider['source']
        consumed = completed_partial(row, leaf, cid)
        if consumed is None and (leaf/'provider_workspace').exists():
            provider = leaf/'provider_workspace/docs/research/distributed_selected_pf'/cid
            prior_outputs = sorted(set(prior_outputs) | {str(provider)})
        generation = int(row.get('price_recheck_generation', 0)) + 1
        row.update(status='PRICE_RECHECK_AVAILABLE', price_recheck_snapshot=snapshot,
                   price_recheck_generation=generation,
                   price_recheck_directory=f'pf_auto/price-recovery/recheck-{generation}',
                   price_prior_owner_directories=prior_outputs,
                   previously_audited_price_axes=(consumed if consumed is not None else
                                                  row.get('previously_audited_price_axes', [])))


def reconcile_received_preparation(run, records, stage_executor):
    """Continue the same PF leaf from a verified compile or received stage."""
    reconcile = getattr(stage_executor, 'received_after_reservation_stop', None)
    if reconcile is None:
        return
    for cid, row in records.items():
        reason = row.get('reason') or ''
        if (row['status'] != 'PREPARATION_INCOMPLETE' or row.get('configurations')
                or reason.startswith('PREPARATION_RESULT_MISMATCH:')):
            continue
        base = run/row.get('price_recheck_directory', (
            'pf_auto/price-recovery' if row.get('price_receipt_recheck') else 'pf_auto'))
        leaf = base/'candidates'/cid/'pf_auto/pf_completion'
        folder = leaf/'provider_workspace/docs/research/distributed_selected_pf'/cid
        inspect = getattr(stage_executor, 'hold_evidence', None)
        if inspect is not None and (evidence := inspect(folder, reason)) is not None:
            row['preparation_recovery'] = evidence
        if reason.startswith('STAGE_FAILED:'):
            continue
        if (leaf/'RESPONSE.json').exists() or (folder/'STOP.json').exists():
            continue
        if reconcile(folder, reason):
            row.update(status='PREPARATION_WAITING', previous_preparation_hold=reason)
            continue
        needs_audit = getattr(stage_executor, 'needs_original_compile_audit', None)
        for compile_stage, audit in (('PF_COMPILE', 'PF_AUDIT'), ('PF_HOLDOUT_COMPILE', 'PF_HOLDOUT_AUDIT')):
            marker = folder/(compile_stage+'.DISTRIBUTED.json')
            if (needs_audit is None or not marker.is_file()
                    or any((folder/(audit+suffix)).exists() for suffix in (
                        '.INPUT.json', '.DISTRIBUTED.json', '.STARTED.json', '.RETURN.json'))):
                continue
            try:
                job = json.loads(marker.read_bytes())['job_id']
            except (OSError, KeyError, ValueError):
                continue
            if needs_audit(folder, compile_stage, job, audit):
                row.update(status='PREPARATION_WAITING', previous_preparation_hold=reason,
                           preparation_recovery=dict(action='AUDIT_ORIGINAL_COMPILE', job_id=job, next_stage=audit))
                break


def prepare_next(root,problem,context,spec,run,routes,stage_executor,*,stop_requested=None,recheck_waiting=True,
                 announce=None):
    """A durable PF wave, with each candidate and native owner in its own directory."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from .pf_saved_prices import price_axis_actions
    from .rpe_preparation_round import snapshot_context
    from .rpe_release_registry import write as atomic_write

    run=Path(run)
    settings=spec.get('pf_completion') or problem.get('pf_completion') or {}
    limit=settings.get('max_candidates',1)
    if type(limit) is not int or limit<1:raise ValueError('PF max_candidates must be a positive integer')
    if settings.get('selected'):
        result = prepare(root,problem,context,spec,run,routes,stage_executor=stage_executor)
        if announce is not None:
            announce(*result[:2])
        return result
    index=run/'pf_auto'/'WAVES.json'
    if index.exists():
        records=json.loads(index.read_text())
    else:
        from .pf_optimizer_connection import read_source
        previous=settings.get('prior_waves_from')
        records=read_source(root,context['reader'],previous) if previous else {}
        from .rpe_preparation_repair import retry_records
        records,retries=retry_records(records,settings.get('preparation_retry_from',()),pf=True)
        if retries:atomic_write(index.with_name('PREPARATION_REPAIRS.json'),retries)
        records,repaired=repair_metadata_waves(records,settings.get('metadata_repair_from',()))
        if repaired:atomic_write(index.with_name('METADATA_REPAIRS.json'),repaired)
        for row in records.values():
            if row.get('status')=='PREPARATION_SCHEDULED':
                row.update(status='PRIOR_OWNER_HOLD',reason='Reconcile the scheduled wave in its original run',
                           original_wave=previous)
    registry=deepcopy(context['pf_evidence_registry'])
    for record in records.values():
        registry['configurations'].update(record.get('configurations',{}))
    if records:
        restored = registry != context['pf_evidence_registry']
        context=_refresh(root,problem,context,spec['methods'],registry,registry.get('artificial',False))
        routes=PF.registry_routes(routes,registry)
        if restored and announce is not None:
            announce(context, routes)
    recovery = run/'pf_auto'/'price-recovery'
    distributed_completion = stage_executor is not None and settings.get('mode') == 'complete'
    if distributed_completion and recheck_waiting:
        reconcile_received_preparation(run, records, stage_executor)
        for row in records.values():
            if (row.get('reason') == 'PF_COMPLETION_REQUEST_MISMATCH'
                    and not row.get('configurations')
                    and set(row.get('differing_fields', ())) <=
                    {'output_root', 'registry_file', 'recover_saved_price_rounds'}):
                row['status'] = 'PREPARATION_WAITING'
    if distributed_completion and settings.get('recover_saved_price_rounds', True):
        for cid, row in records.items():
            if (row['status'] != 'PREPARATION_INCOMPLETE' or row.get('configurations')
                    or row.get('price_receipt_recheck')):
                continue
            prior = run/'pf_auto'/'candidates'/cid/'pf_auto'/'pf_completion'
            check = prior/'ORIGINAL_OWNER_CHECK.json'
            if not check.is_file() or (prior/'PROVIDER_SOURCE.json').exists() or (prior/'provider_workspace').exists():
                continue
            checked = json.loads(check.read_text())['rows']
            if (len(checked) == 1 and checked[0].get('configuration_id') == cid
                    and checked[0].get('status') == 'PRIOR_OWNER_HOLD'
                    and checked[0].get('reason') == 'ValueError: Review existing receipt/job before new work'):
                atomic_write(recovery/'original'/f'{cid}.json', row)
                # Recheck only before a producer or scientific attempt existed.
                row.update(status='PRICE_RECEIPT_RECHECK_AVAILABLE', price_receipt_recheck=True)
        reconsider_prices(root, run, spec, settings, records)
        for row in records.values():
            if row['status'] in PRICE_RECHECK_STATES and 'price_reconciliation' in row:
                row['price_axis_actions'] = price_axis_actions(row['price_reconciliation'])
        if records:
            atomic_write(index, records)
    if records:
        newly_ready = False
        for candidate in context['candidates']:
            record = records.get(candidate['configuration_id'])
            if candidate['ready'] and record and record['status'] in PRICE_RECHECK_STATES:
                record.update(status='READY_RETURNED_TO_SAME_OPTIMIZER', reason=None)
                record.pop('price_reconciliation', None)
                newly_ready = True
            elif record and record['status'] == 'PREPARATION_INCOMPLETE' and not record.get('reason'):
                record.update(reason='PF_EVIDENCE_INCOMPLETE', remaining_needs=candidate['needs'])
                newly_ready = True
        if newly_ready:
            atomic_write(index, records)
    pending={cid for cid,row in records.items() if row.get('status')=='PREPARATION_SCHEDULED'}
    revisit = {cid for cid, row in records.items() if row['status'] in
               ('PRICE_RECEIPT_RECHECK_AVAILABLE', 'PRICE_RECHECK_AVAILABLE')}
    waiting = {cid for cid, row in records.items() if row['status'] == 'PREPARATION_WAITING'} if recheck_waiting else set()
    selection=select(context,spec['methods'],records.keys()-pending-revisit-waiting,
                     include_full=distributed_completion or limit>1 or 'TRUNCATED_PF' not in spec['methods']
                     or any(records[cid].get('method') == 'FULL_PF' for cid in pending | revisit | waiting))
    if selection['selected'] is None and not pending:
        return context,routes,dict(status='NO_INCOMPLETE_PF_CANDIDATE',records=records)
    workers=limit if stage_executor is not None else 1
    if pending:
        wave=[row for row in selection['candidates'] if row['configuration_id'] in pending]
        if len(wave)!=len(pending) or any(row['key']!=records[row['configuration_id']]['key'] for row in wave):
            raise ValueError('Previously selected PF wave no longer matches its configurations')
    else:
        wave=[];families=set()
        # Widths of the same prefix/formula can share early physical rounds.
        # New independent work gets a turn even if a saved wait is unchanged.
        tiers = {}
        for row in selection['candidates']:
            cid = row['configuration_id']
            if row['ready'] or not row['needs'] or (cid in records and cid not in revisit and cid not in waiting):
                continue
            priority = (cid not in records or 'previous_preparation_hold' not in records[cid],
                        cid in waiting, cid in revisit)
            tiers.setdefault(priority, {}).setdefault(row['key'].get('delta_over_h'), []).append(row)
        from itertools import zip_longest
        interleaved = (row for tier in sorted(tiers) for group in zip_longest(*tiers[tier].values())
                       for row in group if row is not None)
        for row in interleaved:
            family=(row['method'],row['key'].get('p'),row['key']['formula'])
            cid=row['configuration_id']
            if family in families:
                continue
            wave.append(row);families.add(family)
            records.setdefault(cid, {}).update(status='PREPARATION_SCHEDULED',key=row['key'],configurations={})
            if len(wave)==workers:break
        atomic_write(index,records)

    if not wave:
        return context,routes,dict(status='NO_INCOMPLETE_PF_CANDIDATE',records=records)
    def acquire(chosen):
        if stop_requested is not None and stop_requested():return None
        cid=chosen['configuration_id']
        record = records[cid]
        one_settings = dict(settings, max_candidates=1)
        if record.get('price_recheck_directory'):
            base = run/record['price_recheck_directory']
            if record.get('price_recheck_provider_source'):
                provider = spec.get('pf_price_recheck_provider') or {}
                if provider.get('source') != record['price_recheck_provider_source']:
                    raise ValueError('PF price recheck provider changed on resume')
                one_settings.update(entry=str(Path(provider['root'])/'scripts/prepare_pf_candidate.py'),
                                    selected_axis_compile=True)
            one_settings['prior_owner_directories'] = sorted(
                set(settings.get('prior_owner_directories', ())) | set(record['price_prior_owner_directories']))
            one_settings['previously_audited_price_axes'] = record['previously_audited_price_axes']
        else:
            base = recovery if record.get('price_receipt_recheck') else run/'pf_auto'
        one_spec = dict(spec, pf_completion=one_settings)
        after,_,status=prepare(root,problem,snapshot_context(context),one_spec,base/'candidates'/cid,
                              deepcopy(routes),stage_executor=stage_executor,selection=dict(selection,selected=chosen))
        return chosen,status,after['pf_evidence_registry']['configurations'].get(cid)

    updated=False;processed=False
    current = None
    with ThreadPoolExecutor(max_workers=min(workers,len(wave))) as pool:
        for task in as_completed([pool.submit(acquire,row) for row in wave]):
            result=task.result()
            if result is None:continue
            chosen,status,entry=result;cid=chosen['configuration_id']
            disposition = status['status']
            if (distributed_completion and settings.get('recover_saved_price_rounds', True) and
                    disposition in ('PRICE_EVIDENCE_REQUIRED', 'PRIOR_OWNER_HOLD')):
                from .pf_saved_prices import live_price_owners
                disposition = ('PRICE_OWNER_WAITING' if live_price_owners(status['price_reconciliation'])
                               else 'PRICE_ACTION_UNAVAILABLE')
            reason = status.get('reason')
            if reason is None and status.get('remaining_needs'):
                reason = 'PF_EVIDENCE_INCOMPLETE'
            records[cid].update(status=disposition,reason=reason,key=chosen['key'],method=chosen['method'],
                                configurations={cid:entry} if entry is not None else {})
            for field in ('request_path', 'differing_fields', 'price_reconciliation', 'remaining_needs'):
                if field in status:
                    records[cid][field] = status[field]
                else:
                    records[cid].pop(field, None)
            if disposition == 'READY_RETURNED_TO_SAME_OPTIMIZER':
                records[cid].pop('previous_preparation_hold', None)
            if 'price_reconciliation' in status:
                records[cid]['price_axis_actions'] = price_axis_actions(status['price_reconciliation'])
            else:
                records[cid].pop('price_axis_actions', None)
            if entry is not None:
                registry['configurations'][cid]=entry;updated=True
            processed |= cid not in waiting or status['status'] != 'PREPARATION_WAITING'
            atomic_write(index,records)
            if entry is not None and announce is not None:
                # Publish each catalog snapshot after its wave record is durable.
                # The other candidates in this wave keep the original context.
                snapshot = deepcopy(registry)
                current = (_refresh(root, problem, context, spec['methods'], snapshot, snapshot.get('artificial', False)),
                           PF.registry_routes(routes, snapshot))
                announce(*current)
    if current is not None:
        context, routes = current
    elif updated:
        context=_refresh(root,problem,context,spec['methods'],registry,registry.get('artificial',False))
        routes=PF.registry_routes(routes,registry)
    return context,routes,dict(status='PREPARATION_WAVE_PROCESSED' if processed else 'PREPARATION_WAVE_YIELDED',
                              records=records)
