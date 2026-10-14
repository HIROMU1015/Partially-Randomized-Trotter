"""Authorized S4: corrected KKT/halfway only; immutable v3 controls reused.

No registered constructor, cost or matrix is evaluated at import/preflight.
All numerical work below is entered only after the source-bound one-shot gate.
"""
from fractions import Fraction as F
from pathlib import Path
import time
from .design import candidates,proposals,ledger,comparisons
from .pipeline import (read_bound,expected,serial,freeze_inventory,acquire,
                       normalize_saved,root_validation,validate_common)
from ..parent_allocation import RootAllocationGenerator,native_ir_g9
from ..parent_allocation_native import checked_entry,native_accounting,tool_binding

TARGETS=(('config_04','ordinary','kkt'),('config_05','ordinary','halfway'),
         ('config_10','closed_P5_full','kkt'),('config_11','closed_P5_full','halfway'))
COMPLETE='PARENT_ALLOCATION_KKT_SUPPLEMENT_COMPLETE_AWAITING_GPT_REVIEW'


def scope(c):
    selected=tuple(tuple(v) for v in c['supplement']['targets'])
    if selected!=TARGETS or c['supplement']['inherited_laws']!=50:
        raise PermissionError('fixed four corrected configurations and fifty controls required')
    return selected


def saved_controls(root,c):
    """Bound source bytes; no old solver, N, ledger, guard or matrix execution."""
    s=c['supplement'];docs={k:read_bound(root,e) for k,e in s['saved_v3_inputs'].items()}
    if (docs['terminal']['status']!='PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
        or docs['terminal']['source_commit']!=s['v3_source']
        or docs['terminal']['execution_HEAD']!=s['v3_authorization']
        or docs['audit']['status']!='SAVED_OUTPUT_PROVENANCE_AUDIT_PASS'
        or docs['audit']['checks_pass']!=918
        or docs['old_contract']['input']!=c['input']
        or docs['old_contract']['precision']!=c['precision']
        or any(docs['old_contract'][k]!=c[k] for k in
               ('epsilon_axis','alpha_axis','alpha_complex','confidence','primitive_error','bias','primary','matrix_tolerance','proposals'))
        or docs['old_contract']['acquisition']['options']!=c['acquisition']['options']):
        raise PermissionError('saved v3 semantics or complete evidence differs')
    manifest={e['path']:e for e in docs['manifest']['files']}
    excluded={v[0] for v in scope(c)};inherited=[]
    for ri,row in enumerate(docs['map']['rows']):
        if row['id'].split(':',1)[0] in excluded:continue
        rel=f'rows/row_{ri:02d}.json';e=manifest[rel]
        bound=dict(e,path=s['v3_result_directory']+'/'+rel)
        original=read_bound(root,bound)
        if original!=row:raise PermissionError('saved control row/map binding')
        inherited.append(dict(row=original,source_row=bound,source_result_directory=s['v3_result_directory']))
    if len(inherited)!=s['inherited_laws']:raise PermissionError('exact unaffected control coverage required')
    return docs,inherited,manifest


def comparison_view(row):
    """Parse saved comparison scalars; do not recalculate inherited prices/N."""
    q=row['ledger'];info=dict(q)
    if 'expected_native_intercept' in q:
        info['expected_native_intercept']={k:F(v) for k,v in q['expected_native_intercept'].items()}
    return dict(row,ledger=info)


def run_pipeline(root,c,identity,writer,guard):
    root=Path(root);scope(c);docs,inherited,manifest=saved_controls(root,c)
    precision={k:(F(v) if k in ('eta','rho') else v) for k,v in c['precision'].items()}
    from ..synthesis_placement.wrapper_launch import verify_runtime
    verify_runtime(root,dict(tool_identity=c['acquisition']['tool_identity']))
    from ..rte_reallocation.numeric import configure
    configure(c['acquisition']['interval_dps'])
    tool=tool_binding((root/c['acquisition']['tool_identity']['path']).read_bytes(),
        expected_tool_sha256=c['acquisition']['tool_identity']['sha256'],
        runtime_executable_sha256=c['acquisition']['runtime_executable_sha256'],
        options=c['acquisition']['options'],interval_dps=c['acquisition']['interval_dps'])
    scale=F(docs['freeze']['scale']);old_cache=docs['cache'];configs=[];candidate_records=[];omitted=[];timing=[]
    p=tuple(map(F,c['input']['p']));x=F(c['input']['tau'])
    for arm,common_id in [('ordinary','config_00'),('closed_P5_full','config_06')]:
        started=time.process_time();e=manifest['events/'+common_id+'.json']
        source=read_bound(root,dict(e,path=c['supplement']['v3_result_directory']+'/'+e['path']))
        for item in source:
            for k in ('coefficient','proposal','weight'):item['event'][k]=F(item['event'][k])
            if item['event'].get('ratio') is not None:item['event']['ratio']=F(item['event']['ratio'])
        make=RootAllocationGenerator.ordinary if arm=='ordinary' else RootAllocationGenerator.closed_p5
        base=make(p,x,**precision);validate_common(source,list(base.reference_events()))
        proxy=[item['native_cost']['T'] for item in source if not item['event']['raw_word']]
        saved_proxy=next(v['proxy_T'] for v in docs['freeze']['candidates'] if v['arm']==arm)
        if proxy!=saved_proxy or len(proxy)!=len(p):raise PermissionError('old common full-event proxy binding')
        cand=candidates(base.root.a,base.root.b,proxy,c['candidates']['KKT'])
        candidate_records.append(dict(arm=arm,proxy_T=proxy,**cand))
        selected={v['label']:v for v in cand['candidates']};rest=[i for i in source if i['event']['raw_word']]
        for config_id,which,label in scope(c):
            if which!=arm:continue
            if label not in selected:
                reason=next((v for v in cand['dropped'] if v['label']==label),
                            dict(label=label,status=cand['kkt']['status']))
                omitted.append(dict(config_id=config_id,reason=reason));continue
            z=selected[label]['allocation'];gen=make(p,x,z,**precision)
            roots=[dict(index=i,event=gen.root_event(i),native_ir=native_ir_g9(gen.root_event(i)),new_root=True)
                   for i in range(len(gen.root.atoms))]
            error=(base.base.B.hi-base.base.B.lo-(base.root.mass.hi-base.root.mass.lo)+gen.root.mass.hi-gen.root.mass.lo)/2
            if error<0:raise ArithmeticError('coefficient certificate subtraction')
            configs.append(dict(config_id=config_id,arm=arm,candidate=label,allocation=z,
                coefficient_error_upper=error,normalizer_interval=gen.B,
                bindings=roots+[dict(item,new_root=False) for item in rest]))
        timing.append(dict(arm=arm,corrected_candidate_CPU_seconds=time.process_time()-started))
    if len(configs)+len(omitted)!=len(TARGETS):raise ArithmeticError('supplement scope coverage')
    if (len(configs)>c['caps']['configurations'] or
        any(len(v['bindings'])>c['caps']['events_per_configuration'] for v in configs) or
        sum(len(v['bindings']) for v in configs)>c['caps']['event_bindings']):raise RuntimeError('supplement configuration cap')
    inventory=freeze_inventory(configs,old_cache,tool,c)
    for key in inventory['keys']:
        if key['acquisition']!='NEW_ONCE':key['acquisition']='REUSE_SAVED_STRICT_CERTIFICATE'
    cf=writer.write('candidate_freeze.json',serial(dict(candidates=candidate_records,scale=scale,
        scale_reused_without_recalculation=True,baseline_T=docs['freeze']['baseline_T'],
        configurations=[{k:v for k,v in cell.items() if k!='bindings'} for cell in configs],
        omitted_configurations=omitted,frozen_before_new_price=True,
        corrected_after_v3_results=True,old_proxy='sum C_i sqrt(z_i^2+b_i^2)',
        corrected_objective='sum sqrt(C_i) sqrt(z_i^2+b_i^2)')))
    iv=writer.write('inventory.json',inventory)
    identity=dict(identity,candidate_freeze_sha256=cf['sha256'],inventory_sha256=iv['sha256'])
    cache={}
    for key in inventory['keys']:
        if key['acquisition']=='NEW_ONCE':continue
        row=checked_entry(old_cache[key['ratio']])
        if row['key']!=key['key'] or row['spec']!=key['spec']:raise PermissionError('saved cache acquisition identity')
        cache[key['ratio']]=row
    if len(cache)>c['caps']['cached_metadata_keys']:raise RuntimeError('cached metadata scope cap')
    writer.write('saved_certificate_reuse.json',dict(keys=[e['key'] for e in cache.values()],
        source=c['supplement']['saved_v3_inputs']['cache'],operator_guard_reruns=0,
        nonroot_native_recalculation=0,unchanged_control_N_recalculation=0))
    fresh,receipts=acquire(root,c,identity,inventory,writer,guard)
    for ratio,saved in fresh.items():cache[ratio]=normalize_saved(saved,expected(ratio,tool,c),old=False,c=c,raw_hashes={})
    if set(cache)!={k['ratio'] for k in inventory['keys']}:raise ArithmeticError('complete key coverage')
    writer.write('guarded_cache.json',cache)
    from ..rte_development.numeric import NativeReference
    ref=NativeReference(cache) if configs else None;root_checks=0
    for cell in configs:
        for item in cell['bindings']:
            guard.check()
            if item['new_root']:
                price=native_accounting(item['native_ir'],cache);item['native_cost']=price
                item['new_validation']=root_validation(item['event'],item['native_ir'],price,ref,c['matrix_tolerance']);root_checks+=2
        if root_checks>c['caps']['new_root_matrix_checks']:raise RuntimeError('root diagnostic cap')
        cell['saved_binding']=writer.write('events/'+cell['config_id']+'.json',serial(cell['bindings']))
    rows=[];dropped=[]
    for cell in configs:
        started=time.process_time();events=[v['event'] for v in cell['bindings']];prices=[v['native_cost'] for v in cell['bindings']]
        laws,not_applicable=proposals(events,[v['T'] for v in prices],scale,precision)
        dropped.append(dict(config_id=cell['config_id'],proposals=not_applicable))
        for law in laws:
            guard.check()
            if len(rows)>=c['caps']['laws']:raise RuntimeError('supplement law cap')
            row=dict(id=cell['config_id']+':'+law['label'],arm=cell['arm'],candidate=cell['candidate'],
                proposal=law['label'],kappa=law['kappa'],allocation=cell['allocation'],saved_events=cell['saved_binding'],
                ledger=ledger(law['events'],prices,cell['coefficient_error_upper'],c),workspace_beyond_system=1,
                law=writer.write(f'laws/law_{len(rows):02d}.json',serial(law['events'])))
            writer.write(f'rows/row_{len(rows):02d}.json',serial(row));rows.append(row)
        timing.append(dict(config_id=cell['config_id'],law_and_ledger_CPU_seconds=time.process_time()-started,online_sampling='NOT_MEASURED'))
    inherited_binding=writer.write('inherited_rows.json',inherited)
    views=[comparison_view(item['row']) for item in inherited]+rows
    writer.write('resource_map.json',serial(dict(rows=rows,inherited_rows=inherited_binding,
        excluded_old_internal_configurations=[v[0] for v in TARGETS],combined_comparisons=comparisons(views),
        C_star=scale,dropped=dropped,omitted_configurations=omitted,classification=COMPLETE,
        classical_setup=timing,new_synthesis_calls=len(fresh),new_root_matrix_checks=root_checks,
        acquisition_process_receipts=receipts,cached_sequences_reused=len(cache)-len(fresh),old_guard_reruns=0,
        quantum_shots_executed=0,retries=0,mandatory_STOP=True,scientific_adoption=False,future_science_authorized=False,
        inherited_ledger_or_native_recomputed=False,shared_acquisition_is_not_selected_algorithm_setup_cost=True)))
    return dict(configurations=len(configs),laws=len(rows),old_guards=0,new_synthesis_calls=len(fresh),
        new_root_matrix_checks=root_checks,inherited_rows=len(inherited),supplemental_scope_complete=True,
        scope_slots=len(TARGETS),omitted_configurations=len(omitted),inventory_sha256=iv['sha256'],candidate_freeze_sha256=cf['sha256'])
