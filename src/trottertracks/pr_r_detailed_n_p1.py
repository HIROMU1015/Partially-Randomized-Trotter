"""Fixed P1 glue around unchanged B RMenuBatch and A action/scalar kernels.

Preflight is static. execute is the explicitly invoked, limited science entry.
No P2/P3, price provider, sampler, optimizer or total cost entry is exposed.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from . import pf_pr_rpe_pipeline as p
from . import pr_r_dependent_n as nmodel
from . import pr_r_normalization_phase0 as phase0
from . import pr_r_screening as screen

PLAN_COMMIT='d33380a8c2fe68d0be1f2c5faaf7fc3b3a956b58'
PLAN_PATH='docs/research/pr_r_detailed_n_preparation_20261111/EXECUTION_PLAN.json'
B_COMMIT='ac3e1ce87b5ed7d9114abc316545366dd097209e'
B_PATH='src/trottertracks/rpe_provider_integration.py'
R_MAP={'H4':{0:[2,4,8],6:[4,128,256]},'H6':{4:[11,16,32],8:[16,128,256]}}
POLICY='R_AWARE_EMPIRICAL_ACTION_COUNT_V1'
NEW_FILES=('src/trottertracks/pr_r_detailed_n_p1.py','scripts/run_pr_r_detailed_n_p1.py')


def blob(root,commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=root)


def preflight(root):
    root=Path(root);store=p.SavedArtifacts(root)
    plan=store.read(dict(commit=PLAN_COMMIT,path=PLAN_PATH))
    package=next(v for v in plan['packages'] if v['id']=='P1_SHORT_MIDDLE')
    rows=[r for r in plan['rows'] if r['package']==package['id']]
    expected=[(s,m,R) for s,rounds in R_MAP.items() for m,Rs in rounds.items() for R in Rs]
    if [(r['system'],r['m'],r['R_outer']) for r in rows]!=expected or len(rows)!=12:
        raise ValueError('Only the fixed twelve P1 candidates are allowed')
    if package['proposed_calls']!=dict(new_finite_means=9,internal_exact_tail_controls=9,
        saved_baseline_scalar_only=3,scalar_phase_radius_N=12,axis_N=24,new_reference_actions=2,
        sampling=0,compile=0,C=0,G=0):raise ValueError('Frozen P1 scientific call scope differs')
    prior=phase0.preflight(root)  # Saved receipts/hashes only, no normalizer/actions.
    sources=dict(prior['sources'])
    for spec in plan['numerical_components']:
        path=spec['path'];value=phase0.sha(root/path)
        if value!=spec['sha256']:raise ValueError('Pinned numerical component changed: '+path)
        sources[path]=value
    for path in NEW_FILES+('src/trottertracks/h4_df10_rpe_batch_provider.py',
        'src/trottertracks/h6_rpe_batch_provider.py','src/trottertracks/h4_df10_rpe_connection.py',
        'src/trottertracks/h6_df19_rpe_connection.py','src/trottertracks/c1_stage3_accuracy.py',
        'src/trotterlib/rpe_resource_accounting.py',
        'src/trottertracks/resource_applicability/ax2b_h6_saved_completion_loader_v1.py',
        'src/trottertracks/resource_applicability/ax2b_independent_reference.py'):
        sources[path]=phase0.sha(root/path)
        if path not in NEW_FILES and (root/path).read_bytes()!=blob(root,PLAN_COMMIT,path):
            raise ValueError('Existing loader/geometry changed: '+path)
    builder=screen.b_metadata(root);menus={};saved_refs={};baselines={}
    law=rows[0]['N_request']['law_binding']['definition']
    for system in R_MAP:
        menu=builder.build_menu(nmodel.saved_task(root,system),R_MAP[system],sampling_law=law)
        menu['N_comparison_policy']=POLICY;menus[system]=menu
        for row in (v for v in rows if v['system']==system):
            target=next(v for v in menu['candidates'] if v['candidate']['id']==row['candidate_id'])
            if not p.same(builder.request_context(menu,target),row['N_request']):
                raise ValueError('P1 canonical identity/law binding differs')
            if row['reference']['saved_pair_shape_present']:
                origin=row['reference']['source'];doc=store.read(origin)
                ref=doc['scope_observed']['references'][str(row['T'])]
                saved_refs[(system,row['T'])]=dict(binding=deepcopy(menu['binding']),T=row['T'],
                    source=dict(origin,field=row['reference']['field'],scientific_source=doc['scientific_source']),
                    evidence=deepcopy(ref))
            if row['baseline'] and row['signal']['baseline_signal_evidence_reusable']:
                origin=row['signal']['saved_source'];doc=store.read(origin)
                records=doc.get('records',doc.get('new_records',[]))
                old=next(v for v in records if v['identity']['candidate']['id']==row['candidate_id'])
                if (not p.same(old['identity'],row['N_request']['identity'])
                        or old['plan']['status']!='CONDITIONAL_ROUND_PASS'):
                    raise ValueError('Saved baseline binding or FAIL; no scientific retry')
                baselines[row['candidate_id']]=dict(record=deepcopy(old),source=deepcopy(origin))
    if len(saved_refs)!=3 or len(baselines)!=3:raise ValueError('P1 saved reference/baseline coverage differs')
    return dict(root=root,plan=plan,package=package,rows=rows,menus=menus,baselines=baselines,
        saved_refs=saved_refs,sources=sources,input_files=prior['input_files'],
        input_hashes=prior['input_hashes'],saved_sources=store.sources,B_source=dict(
            commit=B_COMMIT,path=B_PATH,sha256=hashlib.sha256(blob(root,B_COMMIT,B_PATH)).hexdigest()))


def b_bridge(root,law):
    """Reuse unchanged B function/class ASTs, without importing its C catalog."""
    tree=ast.parse(blob(root,B_COMMIT,B_PATH));wanted={'_law','bound_context_factory','RMenuBatch'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:raise ValueError('Pinned B provider SOURCE differs')
    scope=dict(__name__=__name__,__package__='trottertracks',p=p,nmodel=nmodel,
        deepcopy=deepcopy,LAW=deepcopy(law),N_POLICY=POLICY,r=screen.b_metadata(root),
        saved=SimpleNamespace(H6_RELEASE='f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),B_COMMIT+':'+B_PATH,'exec'),scope)
    return SimpleNamespace(**{n:scope[n] for n in ('bound_context_factory','RMenuBatch')})


def checked_reference(item,binding,T):
    """Same independent-pair numeric gate as A RBatch; no new state action."""
    if not p.same(item['binding'],binding) or item['T']!=T or not item.get('source'):
        raise ValueError('Saved reference SOURCE/binding/time differs')
    ref=deepcopy(item['evidence'])
    for field in ('reference','independent_reference'):
        if any(type(ref[field].get(k)) not in (int,float) or not math.isfinite(ref[field][k]) for k in ('real','imag')):
            raise ValueError('Nonfinite independent reference')
    delta=ref['signal_difference']
    if type(delta) not in (int,float) or not math.isfinite(delta) or not 0<=delta<=1e-10:
        raise ValueError('Reported reference pair numerical gate')
    z=lambda v:complex(v['real'],v['imag'])
    observed=abs(z(ref['reference'])-z(ref['independent_reference']))
    if observed>1e-10:raise ValueError('Observed reference pair numerical gate')
    ref['signal_difference']=max(delta,observed)
    return dict(item,evidence=ref)


def new_reference(ctx,T):
    from . import h6_df19_rpe_d2 as backend
    return backend.reference_pair(ctx,T)


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def execute(packet,source,out):
    root=packet['root'];out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(status='P1_RUNNING',scientific_source=source,plan_source=dict(commit=PLAN_COMMIT,path=PLAN_PATH),
        records=[],references={},batches=[],errors=[],scope_observed=dict(saved_baseline_scalar=0,
            scalar_candidates_completed=0,new_reference_actions_attempted=0,new_reference_actions_completed=0,
            finite_mean_attempts=0,exact_tail_controls_completed=0,context_systems=[]),
        source_files=packet['sources'],input_hashes=packet['input_hashes'],saved_sources=packet['saved_sources'],
        B_source=packet['B_source'],automatic_retry=False,price_calls=0,G_evaluations=0,
        actual_N_qualified=False,full_RPE_qualified=False,total_resources=None,mandatory_stop=True)
    save(out/'progress.json',result)
    lookup={r['candidate_id']:r for r in packet['rows']};menus=packet['menus'];contexts={};refs={}
    B=b_bridge(root,packet['rows'][0]['N_request']['law_binding']['definition'])
    scope=result['scope_observed']
    def context(system):
        loader=B.bound_context_factory(root,system)
        def get(task):
            if system not in contexts:
                contexts[system]=loader(task);scope['context_systems'].append(system)
            return contexts[system]
        return get
    def reference(system):
        def get(ctx,T):
            key=(system,T)
            if key not in refs:
                if key in packet['saved_refs']:item=deepcopy(packet['saved_refs'][key])
                elif key==('H6',1.6):
                    scope['new_reference_actions_attempted']+=2
                    item=dict(binding=deepcopy(menus[system]['binding']),T=T,
                        source=dict(commit=source,kind='P1_EXPLICIT_H6_T1_6_TWO_PATH_REFERENCE'),evidence=new_reference(ctx,T))
                    scope['new_reference_actions_completed']+=2
                else:raise ValueError('No implicit new-reference fallback')
                refs[key]=checked_reference(item,menus[system]['binding'],T)
                result['references'][system+':'+str(T)]=deepcopy(refs[key])
            return deepcopy(refs[key])
        return get
    def attach(n,row):
        p._match(menus[row['system']],next(r for r in menus[row['system']]['candidates'] if r['candidate']['id']==row['candidate_id']),n)
        if not p.same(n['identity'],row['N_request']['identity']):raise ValueError('P1 response identity differs')
        n['law_binding']=deepcopy(row['N_request']['law_binding']);n['N_comparison_policy']=POLICY
        return n
    def batch(system,ids):
        menu=menus[system];selected=[next(r for r in menu['candidates'] if r['candidate']['id']==cid) for cid in ids]
        provider=B.RMenuBatch(menu,nmodel.saved_task(root,system),dict(commit=source,path=NEW_FILES[0]),context(system),reference(system))
        try:return provider(menu,selected)
        finally:
            observed=deepcopy(provider.last_run);result['batches'].append(dict(system=system,candidate_ids=ids,observed=observed))
    try:
        # Reanalyze only the three complete saved baseline signals; no action.
        for cid in packet['package']['execution_order']:
            if cid not in packet['baselines']:continue
            row=lookup[cid];old=packet['baselines'][cid];diag=old['record']['diagnostics']
            task=nmodel.with_round_R(nmodel.saved_task(root,row['system']),{row['m']:row['R_outer']},dict(commit=source))['task']
            target=next(r for r in task['candidates'] if r['candidate']['id']==cid)
            rec=dict(candidate=target['candidate'],signal=deepcopy(diag['signal']),
                **{k:deepcopy(diag[k]) for k in ('signed_weights','allocation','event_count')})
            item=reference(row['system'])(None,row['T'])
            if not p.same(rec['signal']['reference'],item['evidence']['reference']):raise ValueError('Baseline signal/reference differs')
            scope['saved_baseline_scalar']+=1
            n=nmodel.analyze(task,target,rec,item['evidence'],dict(state_energy=diag['reference_solver_energy_ha'],
                state_residual=diag['state_residual_ha']),dict(commit=source,path=NEW_FILES[0],saved_signal=old['source']))
            n['origin']='SAVED_BASELINE_SIGNAL_REANALYZED_MATCHED_R_POLICY';result['records'].append(attach(n,row))
            scope['scalar_candidates_completed']+=1
            save(out/'progress.json',result)
            if n['plan']['status']!='CONDITIONAL_ROUND_PASS':raise ValueError('Saved baseline N unqualified; no retry')
        # Complete the sole baseline lacking independent reference/norm evidence.
        base_id=next(r['candidate_id'] for r in packet['rows'] if r['system']=='H6' and r['m']==4 and r['baseline'])
        fresh=batch('H6',[base_id]);result['records'].extend(fresh)
        if any(n['plan']['status']!='CONDITIONAL_ROUND_PASS' for n in fresh):raise ValueError('H6 m4 baseline unqualified; no new-R calls')
        save(out/'progress.json',result)
        # All four baselines now qualified under the same policy. Independent systems run together.
        def variants(system):
            ids=[r['candidate_id'] for r in packet['rows'] if r['system']==system and not r['baseline']]
            return batch(system,ids)
        with ThreadPoolExecutor(max_workers=len(R_MAP)) as pool:
            futures=[pool.submit(variants,s) for s in R_MAP]
            for future in futures:result['records'].extend(future.result())
        result['status']='P1_COMPLETE_CONDITIONAL_N' if all(n['plan']['status']=='CONDITIONAL_ROUND_PASS' for n in result['records']) else 'P1_STOP_UNQUALIFIED_N'
    except Exception as exc:result.update(status='P1_STOP',errors=[str(exc)])
    # Preserve B's partial records even if a group raised after earlier work.
    known={n['identity']['candidate']['id'] for n in result['records']}
    for observed in result['batches']:
        for n in (observed['observed'] or {}).get('records',[]):
            cid=n['identity']['candidate']['id']
            if cid not in known:result['records'].append(n);known.add(cid)
        for group in (observed['observed'] or {}).get('groups',[]):
            if group:
                scope['finite_mean_attempts']+=group.get('finite_mean_attempts',0)
                scope['exact_tail_controls_completed']+=group.get('PR_exact_tail_controls_completed',0)
    scope['scalar_candidates_completed']+=sum(n['origin']=='EXPLICIT_PR_R_VARIANT_N' for n in result['records'])
    expected=(3,12,2,9,9)
    actual=tuple(scope[k] for k in ('saved_baseline_scalar','scalar_candidates_completed','new_reference_actions_completed','finite_mean_attempts','exact_tail_controls_completed'))
    if result['status']=='P1_COMPLETE_CONDITIONAL_N' and (len(result['records'])!=12 or actual!=expected):
        result.update(status='P1_STOP_SCOPE_MISMATCH',errors=['Completed science scope differs from fixed P1'])
    existing={n['identity']['candidate']['id'] for n in result['records']}
    for row in packet['rows']:
        if row['candidate_id'] not in existing:
            menu=menus[row['system']];target=next(r for r in menu['candidates'] if r['candidate']['id']==row['candidate_id'])
            n=nmodel.failure(menu,target,dict(commit=source),'NOT_EXECUTED_AFTER_P1_STOP','NOT_EXECUTED_AFTER_P1_STOP')
            result['records'].append(attach(n,row))
    order={r['candidate_id']:i for i,r in enumerate(packet['rows'])}
    result['records'].sort(key=lambda n:order[n['identity']['candidate']['id']])
    result['wall_seconds']=time.monotonic()-started
    try:result['source_preserved']=all(phase0.sha(root/path)==h for path,h in packet['sources'].items())
    except OSError:result['source_preserved']=False
    try:result['input_preserved']=all(phase0.sha(v['path'])==v['sha256'] for v in packet['input_files'].values())
    except OSError:result['input_preserved']=False
    if not result['source_preserved'] or not result['input_preserved']:result['status']='P1_STOP_SOURCE_OR_INPUT_CHANGED'
    save(out/'result.json',result)
    save(out/'N_HANDOFF.json',dict(status=result['status'],scientific_source=source,shots=result['records'],
        menus=menus,comparison_policy=POLICY,source_preserved=result['source_preserved'],input_preserved=result['input_preserved'],
        actual_N_qualified=False,full_RPE_qualified=False,prices=[],total_resources=None,mandatory_stop=True))
    with (out/'ROUND_N.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['system','m','T','q_RPE','q_PF','R_outer','candidate_id','status','N_X','N_Y','log_B','rho_lower','beta_det'])
        for n in result['records']:
            i=n['identity'];c=i['candidate'];diag=n['diagnostics'];axis=n['plan'].get('axes',{}).get('X',{})
            writer.writerow([c['system'],i['m'],c['T'],i['q_RPE'],i['q_PF'],c['R_outer'],c['id'],n['plan']['status'],n['plan'].get('N_X'),n['plan'].get('N_Y'),diag.get('signal',{}).get('log_B'),axis.get('radius_lower'),axis.get('beta_deterministic')])
    return result
