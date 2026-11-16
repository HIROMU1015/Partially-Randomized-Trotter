"""Pinned four-method screen plan; imports and plan() do no scientific work.

evaluate_cell() / execute() are explicit future-science leaves. New real data
calls require the user's separate permission; there is no approval-proof system.
"""
from copy import deepcopy
import ast
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from . import h4_pr_m1_rmse as M
from .h4_pr_rmse_completion import INDEX
p = M.p
MENU = '51500b724a75f9535a2c831d50bcfe09eabb9326'
MDIR = 'docs/research/h4_four_method_minimal_menu_20261113/'
D_PATH = 'src/trottertracks/full_baseline_v2_requests.py'
FULL_H = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'
PHYS = dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855', path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
MODULE = 'src/trottertracks/h4_four_method_screen.py'
MEAN = 'src/trottertracks/h4_four_method_screen_mean.py'
RUNNER = 'scripts/prepare_h4_four_method_screen.py'
REFERENCE_ROUNDS = (3, 6, 12)
LANES = (('F2-FINE','2nd',.1), ('F2-COARSE','2nd',.4), ('F4-FINE','4th',.1),
         ('F4-COARSE','4th',.4), ('F5-FINE','4th(m5_best)',.1), ('F5-COARSE','4th(m5_best)',.4))


def _reader(root):
    store = p.SavedArtifacts(root)
    cache = {}
    def read(ref):
        key = ref['commit'], ref['path']
        if key in cache:return cache[key]
        data = M.blob(root,ref['commit'],ref['path'])
        store.sources[ref['commit']+':'+ref['path']] = hashlib.sha256(data).hexdigest()
        cache[key] = json.loads(gzip.decompress(data) if ref['path'].endswith('.gz') else data)
        return cache[key]
    return store, read


def physical_api(root):
    """Reuse D/B's metadata constructors; never import their scientific modules."""
    tree = ast.parse(M.blob(root,MENU,D_PATH))
    literal = {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
               if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None) in
               ('BASE','FULL_H','FULL_H_T20','LIGHT','SAVED_FULL_H_REFERENCE_T','BINDING_KEYS','PREPARATION_KEYS')}
    scope = dict(literal,deepcopy=deepcopy,p=p,math=math,json=json,
                 B=SimpleNamespace(a=SimpleNamespace(exact_scalar=lambda v:Fraction(v['numerator'],v['denominator']) if isinstance(v,dict) else Fraction(v))),
                 SOURCES={
                     'full_h_prepare':dict(commit=FULL_H,path='scripts/h46_full_h_pure_rte.py',functions=['prepare_full_h','context','validate']),
                     'full_h_saved_summary':dict(commit=FULL_H,path=literal['LIGHT']+'input_and_validation_summary.json'),
                     'full_h_saved_selection':dict(commit=FULL_H,path=literal['LIGHT']+'selection.json'),
                 },_blob=lambda commit,path:M.blob(root,commit,path),
                 saved_grid=lambda:dict(q={.8:(1,2,4,8,16),1.6:(1,2,4,8,16)},formulas=('2nd','4th','4th(m5_best)')))
    for name in ('check_binding','check_full_h_preparation','saved_full_h_preparation','_physical'):
        M.function(root,MENU,D_PATH,name,scope)
    return SimpleNamespace(**scope)


def reference_inventory(physics):
    result = {}
    for m in REFERENCE_ROUNDS:
        row = physics['rounds'][m]; d = row['diagnostics']
        result[str(m)] = dict(m=m,T=row['physical_candidate']['T'],binding=deepcopy(physics['binding']),
            source=dict(**PHYS,selector=['rounds',m,'diagnostics','reference_pair']),
            original_diagnostic_source=deepcopy(row['diagnostics_source']),
            fields=sorted(d['reference_pair']),status='SAVED_PAIR_NOT_REEVALUATED')
        if not {'reference','independent_reference','signal_difference'} <= set(d['reference_pair']):
            raise ValueError('Missing saved independent reference pair')
    return result


def pf_structure(c):
    if c['formula']!='4th':return p.structure(c)
    # Existing A helper predates standard Y4; use the existing schedule, not M5.
    weights=list(p._schedule().weights('4th'))
    return dict(formula_id='PF_STANDARD_YOSHIDA4_BINARY64_V1',signed_weights=weights,
        allocation=[0]*len(weights),event_count=0,scalar_policy='scalar_identity_once_at_total_T',
        allocation_policy='ABS_TIME_PER_OUTER_INTEGER_MIN_ONE_PER_PIECE',law_model='deterministic')


def plan(root):
    store,read = _reader(root)
    menu = read(dict(commit=MENU,path=MDIR+'EVIDENCE_METADATA.json'))
    physics = read(PHYS); binding = physics['binding']; api = physical_api(root)
    api.check_binding(menu['common_binding'],binding)
    provenance=read(dict(commit='155529e20a8fac39e00786c40b57830b71203c4f',
        path='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json'))
    if not p.same(provenance['inputs']['H4']['binding'],binding):raise ValueError('Archived PF H/state binding differs')
    prep = api.saved_full_h_preparation('H4'); api.check_full_h_preparation(prep,binding)
    # No Hamiltonian arrays or normalizers are evaluated. Only saved fields.
    full_spec = dict(commit=FULL_H,path=api.LIGHT+'input_and_validation_summary.json')
    read(full_spec)  # Record the unchanged public preparation descriptor SOURCE.
    if not p.same(prep['binding'],binding):raise ValueError('Full-H H/state differs')
    refs = reference_inventory(physics); fresh = []; reuse = []
    implementation_sources=[]
    for path in ('src/trotterlib/h46_m5_plan.py','src/trotterlib/rte.py','scripts/h46_long_time_signal.py',
                 'src/trottertracks/pr_rpe_power_mean.py','src/trottertracks/h4_pr_rmse_completion.py'):
        commit='c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e';data=M.blob(root,commit,path)
        if (Path(root)/path).read_bytes()!=data:raise ValueError('Reused numerical SOURCE changed: '+path)
        implementation_sources.append(dict(commit=commit,path=path,sha256=hashlib.sha256(data).hexdigest()))
    for label,formula,width in LANES:
        saved = next(v for v in menu['full_pf_inventory']['six_proposed_lanes']
                     if v['formula']==formula and float(v['requested_width'])==width)
        for record in saved['matching_saved_rounds']:
            source = record['signal_source']; archived = read(source)[source['key']]
            c = archived['candidate']
            if c['id']!=record['physical_candidate_id'] or c['id']!=p.candidate_id(c):raise ValueError('Saved PF physical ID differs')
            reuse.append(dict(label=label,m=record['m'],physical_candidate=deepcopy(c),source=deepcopy(source),
                              fields=sorted(archived['signal']),status='SAVED_NOT_REANALYZED'))
        for m in (6,12):
            T=.1*(1<<m); q=max(1,math.ceil(T/width))
            c,_,_=api._physical(dict(spec=dict(method='PF',binding=binding,formula=formula,L_D=10)),
                                dict(T=T,q_PF=q,mapping=None))
            fresh.append(dict(label=label,m=m,candidate=c,structure=pf_structure(c),reference=deepcopy(refs[str(m)]),
                              comparison_method='FULL_PF',requested_delta=width))
    for m in (3,4):
        source=menu['truncated_B0I_anchor'] if m==3 else menu['truncated_B0I_m4']
        archived=read(source)[source['key']]
        T=.1*(1<<m);c=dict(system='H4',method='B0I',formula='4th',prefix=7,T=T,q=2 if m==3 else 4,delta=.4,R_outer=None,K=None)
        c['id']=p.candidate_id(c)
        if m==4 and not p.same(archived['candidate'],c):raise ValueError('Saved B0I physical recipe differs')
        reuse.append(dict(label='T4-SCALAR',m=m,physical_candidate=c,source=deepcopy(source),fields=sorted(archived['signal']),
                          status='SAVED_NOT_REANALYZED'))
    for m in (6,12):
        T=.1*(1<<m); q=max(1,math.ceil(T/.4))
        c=dict(system='H4',method='B0I',formula='4th',prefix=7,T=T,q=q,delta=T/q,R_outer=None,K=None)
        c['id']=p.candidate_id(c)
        fresh.append(dict(label='T4-SCALAR',m=m,candidate=c,structure=pf_structure(c),reference=deepcopy(refs[str(m)]),
                          comparison_method='TRUNCATED_PF',requested_delta=.4))
    for K,label in ((2,'R2-ADAPT'),(4,'R4-ADAPT')):
        for m in REFERENCE_ROUNDS:
            T=.1*(1<<m); q=max(1,math.ceil(T/.25)); R=max(1,math.ceil(448*T/2))
            c,_,_=api._physical(dict(spec=dict(method='FULL_H_RTE',binding=binding,finite_K=K)),
                                dict(T=T,mapping=dict(physical_layout=dict(q=q,R_outer=R,event_positions=q*R))))
            fresh.append(dict(label=label,m=m,candidate=c,structure=dict(formula_id='FULL_H_PURE_RTE',signed_weights=[1.],
                allocation=[R],event_count=q*R,scalar_policy='scalar_identity_once_at_total_T',deterministic_blocks=0,
                allocation_policy='ABS_TIME_PER_OUTER',law_model='ideal_finite_RTE_law'),
                reference=deepcopy(refs[str(m)]),comparison_method='FULL_H_RTE',kappa=None,
                preparation_hash=prep['preparation_hash'],layout_policy='FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1'))
    for r in menu['truncated_B0_saved_rounds']:
        reuse.append(dict(label='T2-REUSE',m=r['m'],physical_candidate=deepcopy(r['physical_candidate']),
                          source=deepcopy(r['signal_carrier']),status='SAVED_NOT_REANALYZED'))
    for r in physics['rounds']:
        reuse.append(dict(label='PR-KEEP',m=r['m'],physical_candidate=deepcopy(r['physical_candidate']),
                          source=deepcopy(r['diagnostics_source']),status='SAVED_NOT_REANALYZED'))
    out=dict(schema='H4_FOUR_METHOD_SIGNAL_SCREEN_V1',status='PREPARED_NO_SCIENCE',menu_source=MENU,
        binding=deepcopy(binding),profile=deepcopy(menu['profile']),full_h_preparation=prep,full_h_saved_summary_source=full_spec,
        conditions=dict(h=.1,M=12,sector=[2,2],dimension=36,target='SAVED_DF_H_SAME_SECTOR_GROUND',P=0),
        fresh=fresh,reuse=reuse,references=refs,science_calls=0,scientific_menu_adopted=False,
        maximum_calls=dict(new_complex_means=20,new_full_H_reference_actions=0,full_h_finite_normalizers=6,
            new_SCF_DF_state=0,trajectory_sampling=0,compile=0,N_RMSE_G=0),
        input_index=deepcopy(INDEX),
        source_files=store.sources,implementation_sources=implementation_sources,
        qualification='LOCAL_SCREEN_NOT_GLOBAL_RMSE',canonical_configuration_ids=None,mandatory_stop=True)
    validate(out)
    return out


def validate(planned):
    cells=planned['fresh']; ids=[r['candidate']['id'] for r in cells]
    if len(cells)!=20 or len(set(ids))!=20:raise ValueError('Exactly 20 unique fresh cells required')
    saved_ids={r.get('physical_candidate',{}).get('id') for r in planned['reuse']}
    if set(ids)&saved_ids:raise ValueError('Fresh evaluation duplicates saved physical candidate')
    if planned['binding']['rank']!=10 or planned['binding']['system']!='H4':raise ValueError('H4 DF10 only')
    for r in cells:
        c=r['candidate'];m=r['m'];full=r['comparison_method']=='FULL_H_RTE'
        if c['T']!=.1*(1<<m) or r['reference']['T']!=c['T']:raise ValueError('Time / reference mismatch')
        if full:
            if c['id']!='FULL_H_'+p.candidate_id(c) or c['family']!='FULL_H_PURE_RTE' or c['K'] not in (2,4) or r['structure']['deterministic_blocks']!=0:
                raise ValueError('Full-H is not DF B3')
            if c['q']!=max(1,math.ceil(c['T']/.25)) or c['R_outer']!=max(1,math.ceil(448*c['T']/2)):
                raise ValueError('Fixed explicit layout differs')
            if (r['label'],c['K']) not in (('R2-ADAPT',2),('R4-ADAPT',4)) or m not in REFERENCE_ROUNDS or r['kappa'] is not None:
                raise ValueError('Full-H K / round / kappa scope differs')
            if r['structure']['allocation']!=[c['R_outer']] or r['structure']['event_count']!=c['q']*c['R_outer']:
                raise ValueError('Full-H grouping/allocation differs')
            if (r['structure']['law_model']!='ideal_finite_RTE_law' or
                    r['structure']['signed_weights']!=[1.] or
                    r['preparation_hash']!=planned['full_h_preparation']['preparation_hash']):
                raise ValueError('Full-H law / signed piece / preparation differs')
        elif c['id']!=p.candidate_id(c) or not p.same(r['structure'],pf_structure(c)):
            raise ValueError('PF signed recipe or ID differs')
        else:
            expected={label:(formula,width,10,'FULL') for label,formula,width in LANES}
            expected['T4-SCALAR']=('4th',.4,7,'B0I')
            if r['label'] not in expected or m not in (6,12):raise ValueError('PF screen label/round differs')
            f,width,prefix,method=expected[r['label']]
            if (c['formula'],c['prefix'],c['method'],c['q'])!=(f,prefix,method,max(1,math.ceil(c['T']/width))):
                raise ValueError('PF formula/width/prefix differs')
        if not p.same(r['reference']['binding'],planned['binding']):raise ValueError('Reference H/state differs')
    return True


def evaluate_cell(ctx,row,reference,*,root,preparation):
    """Explicit numerical leaf; called on artificial inputs in preparation tests."""
    from .h4_four_method_screen_mean import evaluate
    return evaluate(ctx,row,reference,root=root,preparation=preparation)


def execute(root,out,source):
    """Future approved fixed screen: exactly 20 cells, no saved-value reanalysis.

    This entry is not exposed by the preparation CLI. SOURCE identifies code,
    not approval. Invoke only after the user permits this finite real run.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    from threading import Event
    import numpy as np
    from . import h4_pr_rmse_completion as previous
    from . import h4_four_method_screen_mean as numerical
    out=Path(out)
    if out.exists():raise ValueError('Fresh scientific output required')
    for path in (MODULE,MEAN):
        if M.blob(root,source,path)!=(Path(root)/path).read_bytes():raise ValueError('Scientific SOURCE differs')
    planned=plan(root);out.mkdir(parents=True)
    def save(name,value):
        with (out/name).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    save('PLAN.json',planned)
    result=dict(source=source,status='STARTED',records=[],errors=[],maximum_new_means=20,
        new_reference_actions=0,automatic_retry=False,N_RMSE_G=None,mandatory_stop=True)
    try:
        if np.__version__!='1.26.4':raise ValueError('Saved numerical runtime NumPy version differs')
        # Existing saved-input loader, no SCF/DF/state generation or circuit work.
        saved,_=previous.load_context(root,planned['binding'])
        index=p.SavedArtifacts(root).read(planned['input_index']);spec=index['T20_reference_files']['H4']
        data=Path(spec['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Saved context changed')
        meta=json.loads(data)
        if not p.same(meta['binding'],planned['binding']):raise ValueError('Saved tail scalar binding differs')
        prep=planned['full_h_preparation']
        if saved.ham.constant!=prep['scalar']['constant']:raise ValueError('Full-H scalar/constant mismatch')
        ctx=numerical.context(saved,root=root,preparation=prep,tail7=meta['prefixes']['7'])
        physics=p.SavedArtifacts(root).read(PHYS)
        for row in planned['fresh']:
            if row['comparison_method']!='FULL_H_RTE':
                numerical.deterministic_step(ctx,row['candidate'],row['structure']['signed_weights'])
        stop=Event()
        def job(row):
            if stop.is_set():raise RuntimeError('NOT_STARTED_AFTER_NUMERICAL_STOP')
            try:
                ref=physics['rounds'][row['m']]['diagnostics']['reference_pair']
                return evaluate_cell(ctx,row,ref,root=root,preparation=prep)
            except Exception:
                stop.set();raise
        # Workers are finite independent cells, no arbitrary CPU/RAM cap.
        workers=min(len(planned['fresh']),len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs={pool.submit(job,row):row for row in planned['fresh']}
            for future in as_completed(jobs):
                row=jobs[future]
                try:
                    record=future.result();record.update(label=row['label'],m=row['m'],
                        comparison_method=row['comparison_method'],reference_source=row['reference']['source'],
                        source=dict(commit=source,module=MODULE,numerical_leaf=MEAN),
                        binding=deepcopy(planned['binding']),profile=deepcopy(planned['profile']))
                    save(row['label']+'_m'+str(row['m'])+'.json',record);result['records'].append(record)
                except Exception as error:
                    stop.set()
                    result['errors'].append(dict(label=row['label'],m=row['m'],error=str(error)))
                    for pending in jobs:pending.cancel()  # Active workers finish; no retry or new cell begins.
        result['status']='STOP_NUMERICAL_OR_BINDING_FAILURE' if result['errors'] else 'LOCAL_SCREEN_COMPLETE_NOT_RMSE_QUALIFIED'
        result['records'].sort(key=lambda r:(r['label'],r['m']))
        result['saved_reuse']=deepcopy(planned['reuse'])  # locators only; no old N or new comparison ranking.
    except Exception as error:
        result['status']='STOP_FAILED_NO_RETRY';result['errors'].append(dict(error=str(error)))
    save('result.json',result)
    return result
