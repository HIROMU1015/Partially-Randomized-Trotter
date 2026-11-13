"""Fixed remaining-ten PR evidence batch; plan() is metadata-only.

S1/S2 numerical leaves and execute() need separate scientific permission.
No decoder sampling, sufficient-N qualification, price or total-cost entry.
"""
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

from . import h4_pr_m1_rmse as M
from . import pr_rpe_power_mean as power
A=M.A;p=M.p;V=M.V
RELEASE='3e035c0d61386010f59ad7ea792bc7a5cefd18ac'
BD='3216d27a69ac8532d2a22e78f5f94491d202d238'
DIR='docs/research/h4_pr_m1_rmse_results_20261113/'
CONFIG=dict(commit=BD,path='docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json')
PREVIOUS=dict(commit=RELEASE,path=DIR+'run01/N_HANDOFF.json')
S1=(3,4);S2=(2,5,7,8,9,10,11,12);REMAINING=tuple(sorted(S1+S2))
MODULE='src/trottertracks/h4_pr_rmse_completion.py';RUNNER='scripts/run_h4_pr_rmse_completion.py'
SOURCE=dict(module=MODULE,previous=PREVIOUS,B=BD,power_SOURCE=power.SOURCE,
 scalar_SOURCE=V.S0_SCIENCE,law=dict(commit=M.V.legacy.P1,path='src/trotterlib/rte.py'))
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')


def _archive(root):
    store=p.SavedArtifacts(root)
    prov=store.read(dict(commit=V.RAW_RELEASE,path=V.RAW_DIR+'signal_provenance.json'))
    if prov['signal_SOURCE']!=power.SOURCE:raise ValueError('Archived numerical SOURCE differs')
    for path in (power.CORE,power.SIGNAL,'src/trotterlib/h46_m5_plan.py'):
        if hashlib.sha256(M.blob(root,power.SOURCE,path)).hexdigest()!=prov['source_files'][path]:raise ValueError('Archive code SOURCE differs')
    def read(name):return json.loads(gzip.decompress(M.blob(root,V.RAW_RELEASE,V.RAW_DIR+name)))
    return read('signal_records.json.gz'),read('signal_endpoints.json.gz')


def _accepted(root,config,packet):
    """Reuse B's unchanged published m1 intake, not a new packet generator."""
    previous=p.SavedArtifacts(root).read(dict(commit=M.A_RELEASE,path=M.A_DIR+'V2_N_HANDOFF.json'))
    b=V.b_metadata(root);h=SimpleNamespace(same=lambda x,y,label:None if p.same(x,y) else (_ for _ in ()).throw(ValueError(label)),
        validate_A_v2_packet=M.intake(root),A_PACKET=dict(commit=V.S0,path=V.S0_DIR+'N_HANDOFF.json'))
    s0=SimpleNamespace(A_ADAPTER_SOURCE=previous['adapter_source'],CONFIGURATION_ID=config['id'],A_RELEASE=M.A_RELEASE,ADIR=M.A_DIR)
    scope=dict(B=SimpleNamespace(_validate=b.validate),p=p,H=h,S0=s0,Path=Path,re=re,deepcopy=deepcopy,
        M1_ID=M.CANDIDATE,M1_CARRIER=M.CARRIER,__file__=str(Path(root)/'src/trottertracks/rpe_diagnostic_intake.py'))
    check=M.function(root,BD,'src/trottertracks/rpe_diagnostic_intake.py','validate_m1_packet',scope)
    adapter=p.SavedArtifacts(root).read(dict(commit='ac873e7fcc42e937084fa153476cbd87565702b4',path='docs/research/h4_m1_rmse_full_pf_preparation_20261113/PREFLIGHT.json'))['source']
    numerical=dict(commit='ac873e7fcc42e937084fa153476cbd87565702b4',path=M.RUNNER,saved_carrier=M.CARRIER,scope='H4_M1_SAVED_SCALAR_ONLY')
    return check(config,packet,previous,numerical_source=numerical,adapter_source=adapter)


def load_fixed(root):
    store=p.SavedArtifacts(root);config=store.read(CONFIG);packet=store.read(PREVIOUS)
    _accepted(root,config,packet)
    schedule=store.read(dict(commit=V.S0,path=V.S0_DIR+'CONFIGURATION.json'))['A_schedule']
    A.evaluate(schedule)  # no means: whole-identity validation only
    requests={r['identity']['m']:r for r in schedule['rows'] if p.arm(r['identity']['candidate'])=='PR'}
    for m,r in requests.items():
        if not p.same(r['identity']['candidate'],config['rounds'][m]['physical_candidate']) or not p.same(r['law_binding'],config['rounds'][m]['round_law_definition']['A_declaration']):
            raise ValueError('A/B physical schedule or law differs')
    return config,packet,schedule,requests


def _source_norm(cid):
    if cid not in ('045843b2429bece6f29b','3df0cfd3356c358aeda8'):raise ValueError('SOURCE norm proof restricted to saved m3/m4')
    return dict(kind='SOURCE_GATE_LIMIT_NOT_OBSERVED',observed_error=None,upper_limit=1e-9,candidate_id=cid,
        source=dict(commit=power.SOURCE,path=power.SIGNAL,function='evolve_batch'),
        carrier=dict(commit=V.RAW_RELEASE,path=V.RAW_DIR+'signal_records.json.gz',selector=cid),
        scientific_RMSE_qualification=False)


def plan(root):
    config,packet,schedule,requests=load_fixed(root);inv=V.saved_inventory(root)
    raw,endpoints=_archive(root);rows=[]
    binding=schedule['design']['binding'];profile=schedule['design']['profile']
    for m in REMAINING:
        req=requests[m];c=req['identity']['candidate'];row=config['rounds'][m]
        common=dict(m=m,T=c['T'],q_PF=c['q'],R_outer=c['R_outer'],event_count=row['structure']['event_count'],
            physical_candidate=deepcopy(c),identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
            request_id=req['request_id'],nominal_N_X=None,nominal_N_Y=None,N_X=None,N_Y=None,RMSE_bound=None,status='NOT_EVALUATED')
        if m in S1:
            descriptors=[v for v in inv['means'] if v['source']['commit']==V.RAW_RELEASE and p.same(v['candidate'],c)]
            if len(descriptors)!=1 or not p.same(descriptors[0]['binding'],binding) or not p.same(descriptors[0]['profile'],profile):raise ValueError('Archived binding differs')
            r=raw[c['id']]
            ep=[e for e in endpoints if p.same(e['candidate'],c) and e['epsilon']==.0005]
            if r['origin']!='NEW_SAVED_INPUT_FINITE_MEAN' or r['phase_alignment'] is not False or len(ep)!=1 or ep[0]['status']!='ELIGIBLE':
                raise ValueError('Archive successful-return/endpoint metadata unresolved')
            if any(k not in ep[0] for k in ('u','log_B_margin','empirical_not_certified')):raise ValueError('Historical allowance fields missing')
            common.update(stage='S1',saved_mean=deepcopy(descriptors[0]['source']),norm_evidence=_source_norm(c['id']),
                saved_fields=deepcopy(descriptors[0]['fields']),saved_error_fields=deepcopy(descriptors[0]['error_fields']),
                old_allowance_source=dict(commit=V.RAW_RELEASE,path=V.RAW_DIR+'signal_endpoints.json.gz',candidate_id=c['id'],historical_epsilon=.0005),
                reference_action_plan=dict(times=[c['T']],paths=['expm_multiply','expm'],actions=2),
                acceptance='CONDITIONAL_SOURCE_GATE_NOT_OBSERVED_SCALAR; preserves empirical R-aware event allowance on block-boundary norms')
        else:
            refs=[v for v in inv['references'] if v['T']==c['T'] and p.same(v['binding'],binding) and set(V.legacy.previous.REFERENCE_FIELDS)<=set(v['fields'])]
            if len(refs)!=1:raise ValueError('Exact saved reference pair missing/ambiguous; no new-reference fallback')
            common.update(stage='S2',saved_reference=deepcopy(refs[0]['source']),mean_backend='ARCHIVED_SPECTRAL_POLYNOMIAL_POWER',
                reference_actions=0,direct_overlap_validation=m==2)
        rows.append(common)
    store=p.SavedArtifacts(root);idx=store.read(INDEX);inputs=idx['saved_inputs']['H4']['inputs'];context=idx['T20_reference_files']['H4']
    return dict(status='REMAINING_TEN_TECHNICALLY_PREPARED_NO_SCIENCE',source=deepcopy(SOURCE),configuration_id=config['id'],rows=rows,
        preserved_packet=PREVIOUS,preserved_nominals=[{k:r[k] for k in ('m','nominal_N_X','nominal_N_Y')} for r in packet['rounds'] if r['m'] in (0,1,6)],
        H_state_binding=deepcopy(binding),profile=deepcopy(profile),exact_configuration_source=CONFIG,
        context_source=dict(index=INDEX,inputs=inputs,context=context,available={k:Path(v['path']).is_file() for k,v in inputs.items()}),
        S1=dict(rounds=list(S1),reference_actions=4,saved_mean_reanalyses=2,new_means=0,internal_controls_from_saved=2,
            evidence_contract_status='PROPOSED_NOT_SCIENTIFICALLY_ADOPTED',
            decision='Proposed: accept SOURCE guard for conditional local diagnostics, preserving missing observation; otherwise separate same-candidate mean scope'),
        S2=dict(rounds=list(S2),fast_means=8,fast_internal_controls=8,new_reference_actions=0,
            direct_equivalence_means=1,direct_equivalence_round=2,direct_internal_controls=1,scalar_nominal_pairs=8),
        S3=dict(status='BLOCKED_SCIENTIFIC_CONDITIONS_UNDECIDED',rounds=list(range(13)),
            required=['all_raw_means_and_shots','fixed_decoder_alias_prior','unresolved_trial_output_or_loss_contract',
            'target_ground_or_conditional_state_contract','finite_K_PF_bias_and_Minkowski_contract',
            'iid_ideal_or_actual_sampler_qualification','MSE_precision_and_validation_confidence','finite_trial_design'],
            sufficient_N=None,global_RMSE=None,simulation_trials=None),
        science_calls=0,mandatory_stop=True)


class _LocalEvidenceAdapter(ast.NodeTransformer):
    """Only norm-evidence acceptance and the obsolete HP-status precondition differ.

    Original numeric formulas/thresholds are unchanged. check_norm() runs before
    these gates; no missing observed scalar is filled or passed off as measured.
    """
    def visit_Tuple(self,node):
        if {getattr(v,'value',None) for v in node.elts}=={'exact_tail_norm_error','raw_corrected_vector_closure'}:
            return ast.copy_location(ast.Tuple(elts=[ast.Constant('raw_corrected_vector_closure')],ctx=ast.Load()),node)
        return self.generic_visit(node)
    def visit_Compare(self,node):
        if any(isinstance(v,ast.Constant) and v.value=='CONDITIONAL_ROUND_PASS' for v in node.comparators):
            return ast.copy_location(ast.Constant(False),node)
        return self.generic_visit(node)


def check_norm(ev,proof,cid):
    observed=ev.get('exact_tail_norm_error')
    if observed is not None:
        if type(observed) not in (int,float) or not math.isfinite(observed) or not 0<=observed<=1e-9:raise ValueError('Observed norm gate')
        return dict(kind='OBSERVED_SCALAR',observed_error=observed,upper_limit=1e-9)
    if not p.same(proof,_source_norm(cid)):raise ValueError('Missing observed norm and unmatched SOURCE gate')
    return deepcopy(proof)


def _scalar_kernels(root,source):
    base=M.kernels(root,source);transform=_LocalEvidenceAdapter()
    node=next(n for n in ast.parse(M.blob(root,V.S0_SCIENCE,'src/trottertracks/pr_r_dependent_n.py')).body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
    stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='point' for t in n.targets))
    fn=ast.parse('def allowance(row,record,reference): pass').body[0]
    fn.body=node.body[:stop]+[ast.parse("return dict(u=u,log_B_margin=margin,stages=stages,roundoff=roundoff,certified=False,policy='R_AWARE_EMPIRICAL_ACTION_COUNT_V1')").body[0]]
    scope=dict(base.allowance.__globals__);fn=transform.visit(fn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),MODULE+':allowance','exec'),scope)
    dn=next(n for n in ast.parse(M.blob(root,V.S0_SCIENCE,'scripts/run_pr_rpe_rmse_s0.py')).body if isinstance(n,ast.FunctionDef) and n.name=='numeric_diagnostics')
    ds=dict(base.diagnostics.__globals__);dn=transform.visit(dn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[dn],type_ignores=[])),MODULE+':local_gate','exec'),ds)
    return SimpleNamespace(allowance=scope['allowance'],diagnose=ds['numeric_diagnostics'],checked_reference=base.checked_reference)


def diagnose(request,record,reference,state,source,*,root,norm_proof=None,previous_allowance=None):
    i=request['identity'];c=i['candidate'];st=i['structure'];sig=deepcopy(record['signal'])
    if not p.same(record['candidate'],c) or any(not p.same(record[k],st[k]) for k in ('signed_weights','allocation','event_count')):
        raise ValueError('Mean physical structure differs')
    norm=check_norm(sig['allowance_evidence'],norm_proof,c['id']);k=_scalar_kernels(root,source)
    ref=k.checked_reference(reference,i['binding'],c['T'])
    allow=k.allowance(dict(candidate=c,structure=st),record,ref['evidence'])
    if norm['kind']=='SOURCE_GATE_LIMIT_NOT_OBSERVED' and previous_allowance is None:raise ValueError('Archived allowance source required')
    previous=deepcopy(previous_allowance) if previous_allowance is not None else dict(u=0.,log_B_margin=0.,certified=False,kind='NO_PREVIOUS_NUMERIC_ALLOWANCE')
    if previous.get('certified') is not False or any(type(previous.get(v)) not in (int,float) or not math.isfinite(previous[v]) or previous[v]<0 for v in ('u','log_B_margin')):
        raise ValueError('Unknown prior numerical allowance')
    sensitivity=sig['allowance_evidence'].get('normalizer_one_ulp_log_sensitivity')
    if sensitivity is not None and (not math.isfinite(sensitivity) or sensitivity<0):raise ValueError('Normalizer sensitivity unresolved')
    margin=max(allow['log_B_margin'],previous['log_B_margin'])
    if sensitivity is not None:margin=max(margin,sensitivity)
    allow.update(u=max(allow['u'],previous['u']),log_B_margin=margin,
        previous_allowance=previous,source=dict(commit=V.S0_SCIENCE,path='src/trottertracks/pr_r_dependent_n.py',function='analyze:allowance_prefix'),
        norm_evidence=norm,maximum_norm_domain=sig['allowance_evidence'].get('maximum_norm_domain','ARCHIVED_SPECTRAL_BLOCK_BOUNDARIES'),
        logical_event_accounting_preserved=True,normalizer_sensitivity=sensitivity,rigorous_bound=False)
    d=dict(signal=sig,candidate=c,signed_weights=st['signed_weights'],allocation=st['allocation'],event_count=st['event_count'],
        allowance=allow,reference_solver_energy_ha=state['state_energy'],state_residual_ha=state['state_residual'],
        ground_state_certified=False,error_disk=dict(unconditional_ground_identification_error=None))
    output=k.diagnose(request,dict(identity=deepcopy(i),diagnostics=d),ref)
    output.update(norm_evidence=norm,mean_source=deepcopy(source),carrier_status='SOURCE_BOUND_FINITE_MEAN_NOT_OLD_HP_PASS')
    return output


def load_context(root,binding):
    """Existing frozen H4 snapshot loader, without old HP prepare()/N calls."""
    from . import h4_df10_rpe_batch_provider as loader
    store=p.SavedArtifacts(root);index=store.read(INDEX);inputs=index['saved_inputs']['H4']['inputs'];spec=index['T20_reference_files']['H4']
    context=json.loads(Path(spec['path']).read_bytes());receipt=json.loads(Path(inputs['H4_tol_receipt.json']['path']).read_bytes())
    if not p.same(context['binding'],binding) or context['sector_dimension']!=36:raise ValueError('Saved context binding differs')
    packet=dict(input_specs=inputs,context_spec=spec,context_metadata=context,
        state_metadata=dict(state_energy=receipt['state_energy'],state_residual=receipt['state_residual']))
    node=next(n for n in ast.parse(M.blob(root,'ac873e7fcc42e937084fa153476cbd87565702b4','src/trottertracks/h4_df10_rpe_batch_provider.py')).body if isinstance(n,ast.FunctionDef) and n.name=='load_saved_context')
    # io is unused in this loader. Skip its process-wide historical BLAS policy;
    # preserve every snapshot/hash/layout check and numerical assembly step.
    node.body=[n for n in node.body if not (isinstance(n,ast.Import) and any(v.name=='run_h6_df19_rpe_d2' for v in n.names))]
    scope=dict(vars(loader));exec(compile(ast.Module(body=[node],type_ignores=[]),MODULE+':saved_loader','exec'),scope)
    ctx,state=scope['load_saved_context'](root,packet);ctx.scientific_binding=deepcopy(binding)
    return ctx,state


def saved_reference(root,row,binding):
    source=row['saved_reference'];doc=p.SavedArtifacts(root).read(source)
    if doc['scientific_source']!=source['scientific_source']:raise ValueError('Saved reference SOURCE differs')
    evidence=doc['scope_observed']['references'][source['reference_key']]
    return dict(T=row['T'],binding=deepcopy(binding),source=deepcopy(source),evidence=deepcopy(evidence))


def anchored_reference(saved_signal,new_item):
    """Keep the archived center; retain both new paths and compare all three."""
    z=lambda v:complex(v['real'],v['imag'])
    ref=deepcopy(new_item);e=ref['evidence'];anchor=deepcopy(saved_signal['reference'])
    gap=max(abs(z(anchor)-z(e['reference'])),abs(z(anchor)-z(e['independent_reference'])),e['signal_difference'])
    if not math.isfinite(gap) or gap>1e-10:raise ValueError('Archived/new reference anchor gate')
    ref['new_pair_before_anchor']=deepcopy(e)
    e.update(reference=anchor,signal_difference=gap,kind='ARCHIVED_CENTER_WITH_NEW_TWO_PATH_VALIDATION')
    return ref


def join_records(config,schedule,previous,records,source,*,root):
    """Future partial/full remaining-ten packet; never rebuild the old m1 packet."""
    bym={}
    for record in records:
        m=record['identity']['m']
        if (m not in REMAINING or m in bym or p.arm(record['identity']['candidate'])!='PR' or record.get('policy_id')!=A.POLICY
                or any(record.get(k) is not None for k in ('sufficient_N_X','sufficient_N_Y','actual_N_X','actual_N_Y','scientific_PASS_FAIL','C','G','N_X','N_Y','plan'))
                or record.get('rmse_guaranteed') is not False):raise ValueError('Foreign scope or sufficient N promotion')
        bym[m]=record
        if (any(record.get('nominal_N_'+axis) is not None and
                (type(record['nominal_N_'+axis]) is not int or record['nominal_N_'+axis]<1) for axis in p.AXES)
                or record.get('status')!='UNQUALIFIED_MODEL_ASSUMPTIONS'):
            raise ValueError('Only unqualified empirical nominal results can enter this packet')
    if not bym:raise ValueError('No new evidence; reference the old packet directly')
    store=p.SavedArtifacts(root)
    old=store.read(dict(commit=V.S0,path=V.S0_DIR+'result.json'))['model_records']
    old+=[store.read(dict(commit=RELEASE,path=DIR+'run01/result.json'))['model_record']]
    ev=A.evaluate(schedule);all_records={r['request_id']:r for r in old+list(bym.values())}
    if set(all_records)-{r['request_id'] for r in ev['records']}:raise ValueError('Foreign whole-schedule request')
    ev['records']=[deepcopy(all_records.get(r['request_id'],r)) for r in ev['records']]
    ap=next(r for r in A.b_handoff(schedule,ev,source)['packets'] if r['spec']['method']=='PR')
    out=V.b_metadata(root).n_from_a(config,ap,inputs=config['spec']['A_input'])
    for row in out['rounds']:
        m=row['m'];row.update(source_candidate_id=config['rounds'][m]['physical_candidate']['id'],
            diagnostic_source=deepcopy(source if m in bym else previous['rounds'][m]['diagnostic_source']))
        if m not in bym and any(row[k]!=previous['rounds'][m][k] for k in ('nominal_N_X','nominal_N_Y','model_status')):raise ValueError('Preserved m0/m1/m6 changed')
    out.update(previous_packet=deepcopy(PREVIOUS),adapter_source=deepcopy(SOURCE),
        diagnostic_scope=dict(newly_evaluated_rounds=sorted(bym),saved_mean_rounds=sorted(set(bym)&set(S1)),
            new_mean_rounds=sorted(set(bym)&set(S2)),preserved_rounds=[0,1,6],whole_schedule_RMSE_qualified=False),
        B_previous_scope_validator_applicable=False)
    return out


def execute(root,out,source,stage):
    """S1/S2 only after explicit user permission for this fixed finite plan."""
    groups={'S1':S1,'S2':S2,'S1_S2':REMAINING}
    if stage not in groups:raise ValueError('Explicit S1/S2 stage required; no S3 simulation')
    out=Path(out)
    if out.exists():raise ValueError('Fresh output required')
    for path in (MODULE,RUNNER,'src/trottertracks/pr_rpe_power_mean.py'):
        if not re.fullmatch('[0-9a-f]{40}',source) or (Path(root)/path).read_bytes()!=M.blob(root,source,path):raise ValueError('Scientific SOURCE not fixed')
    for path in ('src/trottertracks/pr_rpe_rmse.py','src/trottertracks/pr_rpe_rmse_pilot.py','src/trottertracks/h6_df19_rpe_d2.py','src/trotterlib/rte.py'):
        if (Path(root)/path).read_bytes()!=M.blob(root,'ac873e7fcc42e937084fa153476cbd87565702b4',path):raise ValueError('Reused numerical SOURCE differs')
    out.mkdir(parents=True);save=lambda name,value:(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    result=dict(status='RUNNING',stage=stage,source=source,diagnostics=[],model_records=[],equivalence=None,
        new_reference_actions=0,reference_actions_attempted=0,fast_mean_rounds_attempted=[],
        new_fast_means=0,new_direct_validation_means=0,direct_validations_attempted=0,
        automatic_retry=False,global_RMSE_bound=None,mandatory_stop=True)
    try:
        planned=plan(root);save('PLAN.json',planned)
        config,previous,schedule,requests=load_fixed(root);ctx,state=load_context(root,planned['H_state_binding'])
        raw,endpoints=_archive(root);by_m={r['m']:r for r in planned['rows']};refs={};newsrc=dict(commit=source,path=RUNNER,stage=stage)
        from . import h6_df19_rpe_d2 as action
        if set(groups[stage])&set(S1):
            for m in S1:
                T=requests[m]['identity']['candidate']['T'];result['reference_actions_attempted']+=2
                ev=action.reference_pair(ctx,T);result['new_reference_actions']+=2
                refs[m]=dict(T=T,binding=ctx.scientific_binding,source=dict(newsrc,scope='S1_NEW_REFERENCE_PAIR',m=m),evidence=ev)
        spectral=power.spectral_context(ctx,root) if set(groups[stage])&set(S2) else None
        def one(m):
            req=requests[m];c=req['identity']['candidate'];row=by_m[m];proof=None;prior=None
            if m in S1:
                record=deepcopy(raw[c['id']]);record.update(**{k:req['identity']['structure'][k] for k in ('signed_weights','allocation','event_count')})
                ref=anchored_reference(record['signal'],refs[m]);origin=row['saved_mean'];proof=row['norm_evidence']
                ep=next(v for v in endpoints if p.same(v['candidate'],c) and v['epsilon']==.0005)
                prior=dict(u=ep['u'],log_B_margin=ep['log_B_margin'],certified=False,source=row['old_allowance_source'])
            else:
                ref=saved_reference(root,row,ctx.scientific_binding);M.kernels(root,newsrc).checked_reference(ref,ctx.scientific_binding,c['T'])
                result['fast_mean_rounds_attempted'].append(m)
                record=power.evolve(spectral,dict(candidate=c,structure=req['identity']['structure']),ref['evidence']['reference'],root=root);origin=dict(newsrc,m=m,backend='SPECTRAL_POWER')
            save('round_%02d_mean.json'%m,dict(m=m,mean=record,reference=ref,mean_source=origin))
            d=diagnose(req,record,ref,state,origin,root=root,norm_proof=proof,previous_allowance=prior)
            evidence=A.mean_evidence(req,record,origin)
            return m,record,ref,d,evidence
        # Finite task count and actual affinity set the pool, no arbitrary cap.
        workers=min(len(groups[stage]),len(os.sched_getaffinity(0)))
        candidate_errors=[];supplied_means=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(one,m):m for m in groups[stage]}
            for future in as_completed(futures):
                try:m,record,ref,d,evidence=future.result()
                except Exception as error:
                    candidate_errors.append(dict(m=futures[future],error=str(error)));continue
                result['diagnostics'].append(dict(m=m,diagnostics=d,reference=ref,mean_source=d['mean_source']))
                supplied_means.append(evidence)
                if m in S2:result['new_fast_means']+=1
                save('result.json',result)
                if m==2:
                    try:
                        result['direct_validations_attempted']+=1
                        keys=action.half_step_keys([dict(candidate=record['candidate'])])
                        direct_ctx=SimpleNamespace(**vars(ctx));direct_ctx.half_steps=dict(action.half_step(ctx,k) for k in keys)
                        direct=action.evolve(direct_ctx,record['candidate'],ref['evidence']['reference'])
                        result['new_direct_validation_means']+=1
                        dd=diagnose(requests[2],direct,ref,state,dict(newsrc,m=2,backend='DIRECT_VALIDATION'),root=root)
                        z=lambda v:complex(v['real'],v['imag'])
                        differences={k:abs(z(direct['signal'][k])-z(record['signal'][k])) for k in ('corrected','raw','exact_tail')}
                        limit=math.hypot(d['allowance']['u'],d['allowance']['u'])+math.hypot(dd['allowance']['u'],dd['allowance']['u'])
                        result['equivalence']=dict(round=2,differences=differences,allowance_limit=limit,source_bound_not_rigorous=True,
                            direct_diagnostics=dd,log_B_equal=direct['signal']['log_B']==record['signal']['log_B'])
                        if max(differences.values())>limit or not result['equivalence']['log_B_equal']:
                            candidate_errors.append(dict(m=2,error='Direct/power equivalence gate; no algorithm fallback'))
                    except Exception as error:
                        candidate_errors.append(dict(m=2,error="Direct validation: "+str(error)))
        if candidate_errors:
            result['candidate_errors']=candidate_errors
            raise ValueError('Candidate group failed; preserve completed records; no retry')
        # One batch evaluation of the newly supplied means; old nominal values
        # are copied later and are never passed back through the shot formula.
        evaluated=A.evaluate(schedule,mean_provider=lambda _: supplied_means)
        result['model_records']=[r for r in evaluated['records'] if r['identity']['m'] in groups[stage] and p.arm(r['identity']['candidate'])=='PR']
        if any(r['status']!='UNQUALIFIED_MODEL_ASSUMPTIONS' or r['nominal_N_X'] is None for r in result['model_records']):
            raise ValueError('Model/numerical issue; preserve batch outputs; no retry')
        result['model_records'].sort(key=lambda r:r['identity']['m']);result['diagnostics'].sort(key=lambda r:r['m'])
        packet=join_records(config,schedule,previous,result['model_records'],newsrc,root=root);save('N_HANDOFF.json',packet)
        result['status']='LOCAL_NOMINAL_BATCH_COMPLETE_GLOBAL_UNQUALIFIED'
    except Exception as error:
        result.update(status='STOP_FAILED_NO_RETRY',error=str(error));save('result.json',result);raise
    save('result.json',result);return result
