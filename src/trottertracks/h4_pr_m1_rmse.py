"""Fixed m1 saved-scalar preparation and explicit future-science leaf.

prepare() inspects fields/metadata only. diagnose()/execute() perform scientific
reanalysis when called on real data and require separate user permission.
Old S0/P1 sources and the Hoeffding solver are never modified or executed here.
"""
import ast
import cmath
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from . import pr_rpe_rmse_s0_v2 as V
A=V.A;p=V.p
A_RELEASE='75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d'
A_DIR='docs/research/pr_rpe_s0_v2_connection_20261113/metadata01/'
B_RELEASE='030f3f548a35d98a88e62e92a35f132e0243a293'
D_RELEASE='eecaa8030280f8957ad85555bcc89e2daa1a29fa'
CARRIER=dict(commit='89c91c692e1d35de94d2e7afbe890aabe6d80b7e',
             path='docs/research/h4_df10_rpe_batch_results_20261110/run01/result.json',selector=['records',5])
OLD_SCIENCE='90dcc3ed20f30dfab6b22a68cc4adb9461485a98'
CANDIDATE='c00e00dc1e9c414fb7d9'
MODULE='src/trottertracks/h4_pr_m1_rmse.py'
RUNNER='scripts/run_h4_pr_m1_rmse.py'
SOURCE=dict(module=MODULE,A=A_RELEASE,B=B_RELEASE,D=D_RELEASE,saved_carrier=CARRIER,
            scalar_gate=V.S0_SCIENCE,allowance=V.S0_SCIENCE,reference_gate=V.legacy.P1)


def blob(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)


def function(root,commit,path,name,scope):
    node=next(n for n in ast.parse(blob(root,commit,path)).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),commit+':'+path,'exec'),scope)
    return scope[name]


def intake(root):
    """B's S0-only validator; not a validator for newly evaluated m1 packets."""
    b=V.b_metadata(root)
    for path in ('src/trottertracks/pr_rpe_rmse_controller.py','src/trottertracks/pr_rpe_rmse_integration.py'):
        if blob(root,B_RELEASE,path)!=blob(root,V.B_RELEASE,path):raise ValueError('B bridge SOURCE changed')
    scope=dict(B=SimpleNamespace(_validate=b.validate),p=p,deepcopy=deepcopy)
    function(root,B_RELEASE,'src/trottertracks/h4_s0_v2_intake.py','same',scope)
    return function(root,B_RELEASE,'src/trottertracks/h4_s0_v2_intake.py','validate_A_v2_packet',scope)


def _load(root):
    store=p.SavedArtifacts(root)
    for path in ('src/trottertracks/pr_rpe_rmse.py','src/trottertracks/pr_rpe_rmse_pilot.py'):
        if (Path(root)/path).read_bytes()!=blob(root,V.S0_SCIENCE,path):raise ValueError('RMSE/disk numerical SOURCE changed')
    config=store.read(dict(commit=B_RELEASE,path='docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json'))
    ac=store.read(dict(commit=A_RELEASE,path=A_DIR+'V2_CONFIGURATION.json'))
    packet=store.read(dict(commit=A_RELEASE,path=A_DIR+'V2_N_HANDOFF.json'))
    original=store.read(dict(commit=V.S0,path=V.S0_DIR+'N_HANDOFF.json'))
    if not p.same(config,ac):raise ValueError('B official V2 differs from fixed A S0')
    intake(root)(config,packet,original)
    s0=store.read(dict(commit=V.S0,path=V.S0_DIR+'CONFIGURATION.json'))
    schedule=s0['A_schedule'];A.evaluate(schedule)  # metadata only, no means
    request=next(r for r in schedule['rows'] if r['identity']['m']==1 and p.arm(r['identity']['candidate'])=='PR')
    row=config['rounds'][1];i=request['identity']
    if (i['candidate']['id']!=CANDIDATE or not p.same(row['physical_candidate'],i['candidate'])
            or not p.same(row['structure'],i['structure']) or not p.same(row['round_law_definition']['A_declaration'],request['law_binding'])):
        raise ValueError('m1 physical/V2 identity or law differs')
    old=store.read(CARRIER)
    if old['scientific_source']!=OLD_SCIENCE:raise ValueError('m1 original scientific SOURCE differs')
    saved=old['records'][5]
    if (not all(p.same(saved['identity'][k],i[k]) for k in ('candidate','binding','profile','structure'))
            or saved['source']!=dict(commit=OLD_SCIENCE,path='scripts/run_h4_df10_rpe_batch.py')
            or saved['plan']['status']!='CONDITIONAL_ROUND_PASS'):
        raise ValueError('m1 saved identity/SOURCE/old status differs; no retry')
    if not V.legacy.previous.same_source(dict(commit=OLD_SCIENCE,path='src/trotterlib/rte.py'),request['law_binding']['source'],root):
        raise ValueError('m1 finite-RTE law SOURCE differs')
    d=saved['diagnostics'];sig=d['signal']
    if (not p.same(d['candidate'],i['candidate']) or
            any(not p.same(d[k],i['structure'][k]) for k in ('signed_weights','allocation','event_count'))):
        raise ValueError('m1 saved signed structure differs')
    needed=V.legacy.previous.MEAN_FIELDS+('outer_PF_or_discard_bias','finite_RTE_bias')
    if (set(needed)-set(V.legacy.previous.present(sig,needed)) or
            set(V.legacy.previous.ERROR_FIELDS)-set(V.legacy.previous.present(sig.get('allowance_evidence',{}),V.legacy.previous.ERROR_FIELDS))
            or set(V.legacy.previous.REFERENCE_FIELDS)-set(V.legacy.previous.present(d.get('numerical_reference',{}),V.legacy.previous.REFERENCE_FIELDS))
            or any(d.get(k) is None for k in ('reference_solver_energy_ha','state_residual_ha','ground_state_certified'))
            or any(d.get('allowance',{}).get(k) is None for k in ('u','log_B_margin','certified'))
            or 'unconditional_ground_identification_error' not in d.get('error_disk',{})):
        raise ValueError('m1 saved scalar fields missing; no fresh fallback')
    reference=dict(T=i['candidate']['T'],binding=deepcopy(i['binding']),
        source=dict(CARRIER,field='diagnostics.numerical_reference'),evidence=deepcopy(d['numerical_reference']))
    return dict(configuration=config,N_packet=packet,schedule=schedule,request=request,saved=saved,
                reference=reference,source_files=store.sources)


def prepare(root):
    x=_load(root);packet=x['N_packet']
    return dict(status='M1_TECHNICALLY_PREPARED_NOT_EVALUATED',source=deepcopy(SOURCE),
        configuration_id=x['configuration']['id'],B_S0_packet_accepted=True,m=1,
        physical_candidate=deepcopy(x['request']['identity']['candidate']),
        identity=deepcopy(x['configuration']['rounds'][1]['identity']),
        law_binding=deepcopy(x['configuration']['rounds'][1]['law_binding']),
        saved_mean=deepcopy(CARRIER),reference_source=deepcopy(x['reference']['source']),
        numerical_gates_rechecked=False,allowance_applied=False,nominal_N_X=None,nominal_N_Y=None,
        sufficient_N_X=None,sufficient_N_Y=None,global_RMSE_bound=None,
        nominal_coverage=[{k:deepcopy(r[k]) for k in ('m','nominal_N_X','nominal_N_Y','N_X','N_Y','model_status')} for r in packet['rounds']],
        proposed_scientific_scope=dict(saved_candidates=1,local_diagnostics=1,nominal_axis_pairs=1,
            new_finite_means=0,new_exact_tail_controls=0,new_reference_actions=0,new_normalizers=0,
            H_state_array_loads=0,decoder_simulation=0,sampling=0,compile=0,C=0,G=0),
        source_files=x['source_files'],science_calls=0,mandatory_stop=True)


def kernels(root,carrier_source):
    """Reuse frozen allowance prefix and S0 gate; no analyze/Hoeffding call.

    Only pre-point assignments in the original analyze() are compiled. Legacy
    P1 locator variables of the unchanged S0 gate are bound to this carrier,
    so geometry provenance points to H4 batch, never an unrelated P1 record.
    """
    path='src/trottertracks/pr_r_dependent_n.py'
    node=next(n for n in ast.parse(blob(root,V.S0_SCIENCE,path)).body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
    stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='point' for t in n.targets))
    fn=ast.parse('def allowance(row,record,reference): pass').body[0]
    fn.body=node.body[:stop]+[ast.parse("return dict(u=u,log_B_margin=margin,stages=stages,roundoff=roundoff,certified=False,policy='R_AWARE_EMPIRICAL_ACTION_COUNT_V1')").body[0]]
    scope=dict(math=math);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),V.S0_SCIENCE+':'+path,'exec'),scope)
    checked=function(root,V.legacy.P1,'src/trottertracks/pr_r_detailed_n_p1.py','checked_reference',dict(p=p,deepcopy=deepcopy,math=math))
    def geometry(attenuation,raw_error,*,error_source):
        error_source.update(deepcopy(carrier_source))
        return V.legacy.supplied_disk_geometry(attenuation,raw_error,error_source=error_source)
    prefix=carrier_source['path'].removesuffix('run01/result.json')
    proxy=SimpleNamespace(P1=carrier_source['commit'],P1_DIR=prefix,supplied_disk_geometry=geometry)
    diag=function(root,V.S0_SCIENCE,'scripts/run_pr_rpe_rmse_s0.py','numeric_diagnostics',
        dict(p=p,P=proxy,checked_reference=checked,deepcopy=deepcopy,math=math,cmath=cmath))
    return SimpleNamespace(allowance=scope['allowance'],checked_reference=checked,diagnostics=diag)


def diagnose(request,saved,reference,carrier_source,*,root):
    """Scientific leaf on real means; synthetic fixtures only in current task."""
    if (request['identity']['m']!=1 or request['identity']['candidate']['id']!=CANDIDATE
            or p.candidate_id(request['identity']['candidate'])!=CANDIDATE):raise ValueError('Fixed m1 candidate only')
    k=kernels(root,carrier_source);i=request['identity'];d=saved['diagnostics']
    ref=k.checked_reference(reference,i['binding'],i['candidate']['T'])
    allow=k.allowance(dict(candidate=i['candidate'],structure=i['structure']),
        dict(candidate=d['candidate'],signal=d['signal'],**{key:d[key] for key in ('signed_weights','allocation','event_count')}),ref['evidence'])
    # Preserve any larger historical empirical allowance; never weaken old gates.
    old=d['allowance']
    if old.get('certified') is not False or any(type(old.get(v)) not in (int,float) or not math.isfinite(old[v]) or old[v]<0 for v in ('u','log_B_margin')):
        raise ValueError('Original allowance unknown/nonfinite; no zero fill')
    allow.update(u=max(old['u'],allow['u']),log_B_margin=max(old['log_B_margin'],allow['log_B_margin']),
        source=dict(commit=V.S0_SCIENCE,path='src/trottertracks/pr_r_dependent_n.py',function='analyze:allowance_prefix'),
        saved_carrier=deepcopy(carrier_source),previous_allowance=deepcopy(old),rigorous_bound=False)
    bound=deepcopy(saved);bound['diagnostics']['allowance']=allow
    out=k.diagnostics(request,bound,ref)
    out.update(saved_carrier=deepcopy(carrier_source),allowance_adapter=dict(module=MODULE,formula_source=allow['source']))
    return out


def handoff(configuration,schedule,old_packet,preserved_records,new_record,source,*,root,evidence_domain='UNQUALIFIED_MODEL'):
    """Copy m0/m6; allow one nominal m1 result only, never sufficient N."""
    if new_record['identity']['m']!=1 or p.arm(new_record['identity']['candidate'])!='PR':raise ValueError('m1 PR only')
    if (new_record.get('policy_id')!=A.POLICY or any(new_record.get(k) is not None for k in
        ('sufficient_N_X','sufficient_N_Y','actual_N_X','actual_N_Y','N_X','N_Y','plan','actual','scientific_PASS_FAIL','C','G')) or new_record.get('rmse_guaranteed') is not False):
        raise ValueError('No sufficient N or scientific guarantee in this scope')
    if any(new_record.get(k) is not None and (type(new_record[k]) is not int or new_record[k]<1) for k in ('nominal_N_X','nominal_N_Y')):
        raise ValueError('Nominal N must be a positive exact integer or null')
    ev=A.evaluate(schedule);prior={r['request_id']:r for r in preserved_records}
    if len(prior)!=2 or {r['identity']['m'] for r in prior.values()}!={0,6}:raise ValueError('Exactly original S0 records required')
    for m in (0,6):
        v=next(r for r in prior.values() if r['identity']['m']==m);old=old_packet['rounds'][m]
        if (v['nominal_N_X'],v['nominal_N_Y'],v['status'])!=(old['nominal_N_X'],old['nominal_N_Y'],old['model_status']):
            raise ValueError('S0 nominal or status changed')
    merged=dict(prior);merged[new_record['request_id']]=new_record
    if set(merged)-{r['request_id'] for r in ev['records']}:raise ValueError('Foreign SOURCE/whole schedule')
    ev['records']=[deepcopy(merged.get(r['request_id'],r)) for r in ev['records']]
    ap=next(r for r in A.b_handoff(schedule,ev,source,evidence_domain=evidence_domain)['packets'] if r['spec']['method']=='PR')
    out=V.b_metadata(root).n_from_a(configuration,ap,inputs=configuration['spec']['A_input'])
    for row in out['rounds']:
        row.update(source_candidate_id=configuration['rounds'][row['m']]['physical_candidate']['id'],
            diagnostic_source=deepcopy(source if row['m']==1 else old_packet['source'] if row['m'] in (0,6) else None))
    out.update(diagnostic_scope=dict(newly_reanalyzed_rounds=[1],preserved_S0_rounds=[0,6],
        other_rounds_NOT_EVALUATED=True,whole_schedule_RMSE_qualified=False),
        preserved_S0_packet=dict(commit=A_RELEASE,path=A_DIR+'V2_N_HANDOFF.json'),
        B_S0_only_validator_applicable=False,adapter_source=deepcopy(SOURCE))
    return out


def execute(root,out,source):
    """Future explicitly invoked one-candidate science; never called by prepare."""
    out=Path(out)
    if out.exists():raise ValueError('Fresh output required; preserve all old evidence')
    for path in (MODULE,RUNNER):
        if not re.fullmatch('[0-9a-f]{40}',source) or blob(root,source,path)!=(Path(root)/path).read_bytes():
            raise ValueError('Scientific implementation SOURCE not fixed')
    out.mkdir(parents=True);result=dict(status='M1_RUNNING',science_source=source,diagnostics=None,model_record=None,
        new_finite_means=0,new_reference_actions=0,new_normalizers=0,automatic_retry=False,mandatory_stop=True)
    save=lambda name,value:(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        x=_load(root);d=diagnose(x['request'],x['saved'],x['reference'],CARRIER,root=root)
        result['diagnostics']=d
        c=x['request']['identity']['candidate'];record=dict(candidate=c,signal=d['signal'],**{k:x['request']['identity']['structure'][k] for k in ('signed_weights','allocation','event_count')})
        evidence=A.mean_evidence(x['request'],record,CARRIER)
        ev=A.evaluate(x['schedule'],mean_provider=lambda requests:[evidence])
        model=next(r for r in ev['records'] if r['request_id']==x['request']['request_id'])
        result['model_record']=model
        if model['status']!='UNQUALIFIED_MODEL_ASSUMPTIONS' or model['nominal_N_X'] is None:raise ValueError('Numerical/model gate unresolved; no retry')
        original=p.SavedArtifacts(root).read(dict(commit=V.S0,path=V.S0_DIR+'result.json'))
        src=dict(commit=source,path=RUNNER,saved_carrier=CARRIER,scope='H4_M1_SAVED_SCALAR_ONLY')
        packet=handoff(x['configuration'],x['schedule'],x['N_packet'],original['model_records'],model,src,root=root)
        save('N_HANDOFF.json',packet)
        result.update(status='M1_NOMINAL_COMPLETE_UNQUALIFIED',diagnostics=d,model_record=model,
            configuration_id=x['configuration']['id'],source_files=x['source_files'],sufficient_N_X=None,sufficient_N_Y=None,global_RMSE_bound=None)
    except Exception as error:
        result.update(status='STOP_FAILED_NO_FALLBACK',error=str(error));save('result.json',result);raise
    save('result.json',result);return result


def full_pf_metadata(root):
    """Read D's passive catalog/spec builders; do not invent a PF evaluation policy."""
    path='src/trottertracks/full_baseline_v2_requests.py'
    nodes=[n for n in ast.parse(blob(root,D_RELEASE,path)).body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    scope=dict(ast=ast,deepcopy=deepcopy,json=json,math=math,Path=Path,subprocess=subprocess,
        B=SimpleNamespace(p=p),__file__=str(Path(root)/path))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),D_RELEASE+':'+path,'exec'),scope)
    catalog=scope['catalog']()
    return SimpleNamespace(full_pf_spec=scope['full_pf_spec'],catalog={k:catalog['formulas'][k] for k in ('2nd','4th','4th(m5_best)')},
        sources={k:v for k,v in catalog['sources'].items() if k.startswith('pf_')},grid=catalog['grid'])


def full_pf_inventory(root):
    """Field-only carrier catalog; null archived signals remain missing."""
    d=full_pf_metadata(root);store=p.SavedArtifacts(root)
    prov=store.read(dict(commit=V.RAW_RELEASE,path=V.RAW_DIR+'scientific_provenance.json'))
    inventory=V.legacy.pilot_inventory(root);records=[]
    for filename in ('signal_records.json.gz','extension_signal_records.json.gz'):
        path=V.RAW_DIR+filename
        for cid,r in json.loads(gzip.decompress(blob(root,V.RAW_RELEASE,path))).items():
            c=r['candidate']
            if c['method']!='FULL' or c['formula'] not in d.catalog:continue
            binding=prov['inputs'][c['system']]['binding']
            if c['id']!=p.candidate_id(c) or c['prefix']!=binding['rank']:raise ValueError('Full PF metadata differs')
            signal=r.get('signal') or {}
            refs=[v['source'] for v in inventory['references'] if v['T']==c['T'] and p.same(v['binding'],binding)
                and set(V.legacy.previous.REFERENCE_FIELDS)<=set(v['fields'])]
            records.append(dict(candidate=deepcopy(c),source=dict(commit=V.RAW_RELEASE,path=path,selector=cid),
                signal_record_present=r.get('signal') is not None,
                signal_fields=V.legacy.previous.present(signal,V.legacy.previous.MEAN_FIELDS),
                error_fields=V.legacy.previous.present(signal.get('allowance_evidence') or {},V.legacy.previous.ERROR_FIELDS),
                independent_reference_pair_sources=refs,phase_alignment=r.get('phase_alignment'),
                numerical_gate_rechecked=False,nominal_N_X=None,nominal_N_Y=None))
    return dict(source=dict(D=D_RELEASE,A=A_RELEASE),formulas=d.catalog,source_functions=d.sources,
        saved_grid=d.grid,saved_binding={s:v['binding'] for s,v in prov['inputs'].items()},saved_carriers=records,
        configuration_adopted=False,policy_adopted=False,science_calls=0,mandatory_stop=True)
