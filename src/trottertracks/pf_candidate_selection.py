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
    from .pr_candidate_completion import Held
    try:
        response=PF.complete_selected(root,helper_spec,folder,
            **({'stage_executor':stage_executor}if stage_executor is not None else{}))
    except Held as error:
        # The fixed selector and stage markers retain the owner and request.
        # A candidate lacking an eligible host/receipt must not cancel PR or
        # already READY PF work using the same distributed executor.
        return context,routes,dict(status='PREPARATION_INCOMPLETE',selection=selection,reason=str(error))
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
    price_holds=[row for row in response['response'].get('items',[]) if row.get('status')=='PRICE_EVIDENCE_REQUIRED']
    if price_holds and mode!='plan':
        status.update(status='PRICE_EVIDENCE_REQUIRED',reason=price_holds[0]['reason'])
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


def prepare_available(root, problem, context, spec, run, routes, stage_executor, *, stop_requested, announce):
    """Run the next bounded wave as soon as its predecessor is saved."""
    while True:
        context, routes, status = prepare_next(root, problem, context, spec, run, routes, stage_executor,
                                               stop_requested=stop_requested)
        announce(context, routes)
        if status['status'] != 'PREPARATION_WAVE_PROCESSED' or stop_requested():
            return context, routes, status


def prepare_next(root,problem,context,spec,run,routes,stage_executor,*,stop_requested=None):
    """A durable PF wave, with each candidate and native owner in its own directory."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from .rpe_preparation_round import snapshot_context
    from .rpe_release_registry import write as atomic_write

    run=Path(run)
    settings=spec.get('pf_completion') or problem.get('pf_completion') or {}
    limit=settings.get('max_candidates',1)
    if type(limit) is not int or limit<1:raise ValueError('PF max_candidates must be a positive integer')
    if settings.get('selected'):
        return prepare(root,problem,context,spec,run,routes,stage_executor=stage_executor)
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
        context=_refresh(root,problem,context,spec['methods'],registry,registry.get('artificial',False))
        routes=PF.registry_routes(routes,registry)
    pending={cid for cid,row in records.items() if row.get('status')=='PREPARATION_SCHEDULED'}
    selection=select(context,spec['methods'],records.keys()-pending,
                     include_full=limit>1 or 'TRUNCATED_PF' not in spec['methods'])
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
        for row in selection['candidates']:
            family=(row['method'],row['key'].get('p'),row['key']['formula'])
            cid=row['configuration_id']
            if row['ready'] or not row['needs'] or cid in records or family in families:
                continue
            wave.append(row);families.add(family)
            records[cid]=dict(status='PREPARATION_SCHEDULED',key=row['key'],configurations={})
            if len(wave)==workers:break
        atomic_write(index,records)

    one_spec=dict(spec,pf_completion=dict(settings,max_candidates=1))
    def acquire(chosen):
        if stop_requested is not None and stop_requested():return None
        cid=chosen['configuration_id']
        after,_,status=prepare(root,problem,snapshot_context(context),one_spec,run/'pf_auto'/'candidates'/cid,
                              deepcopy(routes),stage_executor=stage_executor,selection=dict(selection,selected=chosen))
        return chosen,status,after['pf_evidence_registry']['configurations'].get(cid)

    updated=False;processed=False
    with ThreadPoolExecutor(max_workers=min(workers,len(wave))) as pool:
        for task in as_completed([pool.submit(acquire,row) for row in wave]):
            result=task.result()
            if result is None:continue
            chosen,status,entry=result;cid=chosen['configuration_id']
            records[cid]=dict(status=status['status'],reason=status.get('reason'),key=chosen['key'],
                              configurations={cid:entry} if entry is not None else {})
            if entry is not None:
                registry['configurations'][cid]=entry;updated=True
            processed=True
            atomic_write(index,records)
    if updated:
        context=_refresh(root,problem,context,spec['methods'],registry,registry.get('artificial',False))
        routes=PF.registry_routes(routes,registry)
    return context,routes,dict(status='PREPARATION_WAVE_PROCESSED' if processed else 'PREPARATION_WAVE_YIELDED',
                              records=records)
