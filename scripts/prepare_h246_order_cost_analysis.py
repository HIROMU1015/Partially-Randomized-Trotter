"""Collect fixed saved fields and seal an unapproved analysis; no real arithmetic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import h246_order_cost_analysis as c

ADOPTED=c.old6.ADOPTED
SAVED=c.old6.SAVED
ADIR=c.old6.h4.ADOPTED_DIR
SDIR=c.old6.h4.SAVED_DIR
FULL_SOURCES={
 'H2':('9e5ea6b235d7beecc02651b1393b561948e85789','track_a_h2_archived_exploratory_run01'),
 'H4':('868ee7ced6818b2531a3aa9c12498f02bce55dc3','track_a_h4_four_method_exploratory_run01')}


def collect():
    files={};catalog=[]
    def protect(path,role,expected=None):
        p=Path(path).resolve();s=c.sha(p)
        c.require(expected is None or s==expected,'PINNED_INPUT_SHA')
        files[str(p)]=dict(path=str(p),sha256=s,role=role)
    def fixed(commit,path):
        raw=c.old.recovery.git_bytes(commit,path);p=c.ASSETS/'fixed'/commit/path
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():c.require(p.read_bytes()==raw,'NO_OVERWRITE_FIXED_BYTES')
        else:p.write_bytes(raw)
        protect(p,'INPUT:fixed_Git')
        catalog.append(dict(commit=commit,path=path,sha256=c.sha(p),bytes=len(raw)))
        if path.endswith('.gz'):return json.loads(gzip.decompress(raw))
        if path.endswith('.json'):return json.loads(raw)
        return raw
    dataset=fixed(ADOPTED,ADIR+'dataset.json.gz')
    source_catalog=fixed(ADOPTED,ADIR+'source_catalog.json')
    evaluations=fixed(SAVED,SDIR+'evaluation.json')
    for name in ('cost_decomposition_60.csv','candidate_precision_costs_1000.csv',
                 'formula_comparison_30.csv','unit_statistics_36.csv','input_provenance.json','end_audit.json'):
        fixed(SAVED,SDIR+name)
    for name in ('candidate_index.csv','comparisons_within_formula_45.json','precision_rankings_15.json'):
        fixed(ADOPTED,ADIR+name)
    main={}
    for system,(commit,track) in FULL_SOURCES.items():
        prefix='artifacts/resource_applicability/'+track+'/2026-10-24/'
        main[system]=fixed(commit,prefix+'comparison.json')
        for name in ('REPORT.md','four_method_saved_output_table.csv','final_verification.json'):
            fixed(commit,prefix+name)
    h6=fixed(c.BASE,'artifacts/resource_applicability/track_a_h6_saved_four_method/2026-10-24/saved_results_projection.json')
    for name in ('README.md','research_index.csv','scientific_profiles.json','preserved_paper_files.csv'):
        fixed('6dafed86cc82149d55b4e6395382953d5f3631d7',
              'docs/data_collection/2026-10-23_three_environment_integration/'+name)
    h2cfgp=c.ROOT.parent/'h2-archived-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json'
    h4cfgp=c.ROOT.parent/'h4-four-method-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json'
    h2cfg=c.read(h2cfgp);h4cfg=c.read(h4cfgp)
    for p in (h2cfgp,h4cfgp):protect(p,'PROTECTED:completed_contract')
    protect(h2cfg['origin_contract'],'INPUT:accepted_H2_diagnostic_contract')
    origin=c.read(h2cfg['origin_contract'])
    for rel in ('src/trotterlib/h46_m5_plan.py','scripts/h46_new2_core.py','scripts/h46_m5_native.py'):
        original=next(x for x in origin['source_files'] if x['path']==rel)
        c.require(hashlib.sha256(c.old.recovery.git_bytes(SAVED,rel)).hexdigest()==original['sha256'],
                  'SCHEDULE_SOURCE_MATCHES_ACCEPTED_H2_ACQUISITION')
    # These are immutable read-only sources, not authorities for this analysis.
    for cfg in (h2cfg,h4cfg):
        for p in Path(cfg['run_root']).rglob('*.json'):protect(p,'PROTECTED:completed_result_grant_ledger_STOP')
    plan_path=next(r['path'] for r in h4cfg['files'] if
                  r['role']=='PROTECTED:original_analysis_authority_and_results' and
                  r['path'].endswith('plan_APPROVED_PRIVATE.json'))
    protect(plan_path,'INPUT:original_saved_analysis_plan');plan=c.read(plan_path)
    h4inputs=c.read(h4cfg['manifest']);protect(h4cfg['manifest'],'INPUT:H4_accepted_inputs',h4cfg['manifest_sha256'])
    bindings={s:dataset['systems'][s]['scientific_identity_saved']['binding'] for s in ('H2','H4','H6')}
    # Retain only the actual molecular identity fields; no private runtime locations.
    bindings={s:{k:v for k,v in b.items() if k in ('hamiltonian','state','snapshot','rank','rank_policy','basis','system')}
              for s,b in bindings.items()}
    data=dict(schema=c.SERIES+'_INPUTS',kind='SAVED_REAL_VALUES',physical=c.PHYSICAL,targets=c.TARGETS,
              epsilons=c.EPS,bindings=bindings,rows=[],availability=[],structures=[],limitations=c.LIMITS,
              cohort_pooling=False,catalog=catalog)
    for target in c.TARGETS:
        cid=target['id'];system=target['system'];full=target['method']=='FULL'
        original_id=c.old.FULL[cid][0] if system=='H2' and full else cid
        item=next((x for x in dataset['candidates'] if x['candidate_status_saved']['id']==original_id),None) if full else None
        group=None if full else next(g for g in plan['groups'] if g['record']['candidate']['id']==cid)
        saved_group=None if full else next(g for g in evaluations['groups'] if g['candidate']['id']==cid)
        if full:
            signal=item['signal_record_saved']['signal'];raw_pairs=None
        else:
            c.require(all(group['record']['candidate'][k]==target[k] for k in ('id','method','formula','prefix','q')),
                      'ORIGINAL_PR_CANDIDATE')
            c.require(all(group['binding'][k]==bindings[system][k] for k in ('hamiltonian','state','snapshot','rank')),
                      'ORIGINAL_PR_MOLECULAR_BINDING')
            c.require(group['record']['candidate']['K']==4 and group['record']['candidate']['r_policy']=='ABS_TIME_PER_OUTER'
                      and group['record']['candidate']['r_value']==32,'FIXED_RTE_LAW')
            signal=group['record']['signal'];raw_pairs=[]
            for task in group['tasks']:
                p=Path(task['result_path']);protect(p,'INPUT:original_PR_result',task['sha256']);raw=c.read(p)
                c.require(raw['task']==task['task'] and raw['paired_trajectory'] and raw['native_scientific_PASS']
                          and raw['cohort_pooling'] is False and raw['conditional_sampling'] is False and
                          raw['seed_cohort']=='EXPLORATION','ORIGINAL_PAIRED_TRAJECTORY')
                c.require(raw['input_binding']==group['binding'] and raw['SOURCE']==group['acquisition_SOURCE'],
                          'ORIGINAL_PR_SOURCE_INPUT')
                c.require({a['axis'] for a in raw['axis_costs']}=={'cosine','sine'} and
                          all(a['compiler_settings_hash']==c.old6.h4.COMPILER for a in raw['axis_costs']),
                          'SAVED_AXIS_COMPILER')
                raw_pairs.append(dict(pair_id=task['task']['id'],candidate=cid,cohort='EXPLORATION',
                    axes={a['axis']:a['metrics']['rz_count'] for a in raw['axis_costs']}))
            representative=group['tasks'][0];event_path=Path(representative['result_path']).parent/'events_PRIVATE.json'
            protect(event_path,'INPUT:original_first_trajectory_events')
            event_digest=c.read(representative['result_path'])['event_digest']
            event_doc=c.read(event_path)
            c.require(event_doc['event_digest']==event_digest==c.digest(event_doc['events']) and
                      event_doc['cohort']=='EXPLORATION' and event_doc['conditional'] is False,
                      'ORIGINAL_FULL_EVENT_DIGEST')
        stats=None
        accepted=next((r for r in main[system]['methods'] if r['candidate']==cid),None) if system in main else None
        if accepted:stats=accepted['compiled_RZ_statistics']
        structure=dict(target=target,binding=bindings[system],saved_schedule=None,
                       saved_cost_statistics=stats or (saved_group['unit_statistics'] if saved_group else None),
                       saved_strata=None if full else saved_group['strata'])
        if not full:structure.update(representative_event_file=str(event_path),representative_event_digest=event_digest)
        if system=='H2':
            p=Path(h2cfg['origin_run_root'])/'results'/(cid+'_diagnostics.json');protect(p,'INPUT:accepted_H2_diagnostics')
            diagnostic=c.read(p);structure['saved_schedule']=diagnostic['schedule']
        data['structures'].append(structure)
        for ep in c.EPS:
            if system=='H2' and ep!=.0005:
                data['availability'].append(dict(candidate=cid,epsilon=ep,available=False,
                    reason='No saved endpoint under accepted H2 DIAGNOSTICS allowance/reference policy; common table uses different policy'))
                continue
            cohort='COMMON_PF_DETERMINISTIC' if full else 'EXPLORATION'
            if ep==.0005 and accepted:
                a=accepted['analytic_shots'];axes=a['axes'];u=accepted['allowance'];components=signal
                if system=='H2':
                    bias=diagnostic['reference_binding']['bias'];log_b=diagnostic['log_B']
                else:
                    orig=h4inputs['methods'][cid];bias=orig['bias'];log_b=orig['log_B']
                rz={ax:accepted['compiled_RZ_statistics'][ax]['mean'] for ax in ('cosine','sine')}
                G=accepted['G_total'];se=accepted['paired_resource_statistics']['G_standard_error']
                log_for=None;allowance={ax:u[ch] for ax,ch in (('cosine','real'),('sine','imag'))};margin=u['log_B_margin']
                variance=accepted['compiled_RZ_statistics']
                eligible=accepted['eligible']
                pointer=dict(commit=FULL_SOURCES[system][0],candidate=cid,epsilon=ep)
            elif full:
                endpoint=next(x for x in item['endpoints_saved'] if x['epsilon']==ep)
                axes=endpoint['axes'];bias=signal['total_signed_bias'];log_b=signal['log_B'];log_for=None
                rz={ax:endpoint['price'][ax] for ax in ('cosine','sine')};G=endpoint['total_RZ_exact'];se=None
                allowance={ax:endpoint['u'] for ax in ('cosine','sine')};margin=endpoint['log_B_margin']
                variance=None;eligible=endpoint['status']=='ELIGIBLE';pointer=dict(commit=ADOPTED,candidate=original_id,epsilon=ep)
            else:
                endpoint=next(x for x in saved_group['rows'] if x['epsilon']==ep and x['mode']=='USUAL_SIGNED')
                a=endpoint['shot'];axes=a['axes'];bias=a['saved_signed_bias'];log_b=a['log_B'];log_for=a['log_B_for_shots']
                rz=dict(cosine=endpoint['mean_cos_RZ'],sine=endpoint['mean_sin_RZ']);G=endpoint['total_RZ'];se=endpoint['total_RZ_SE']
                allowance={ax:a['u'] for ax in ('cosine','sine')};margin=a['log_B_margin'];variance=saved_group['unit_statistics']
                eligible=endpoint['status']=='ELIGIBLE';pointer=dict(commit=SAVED,candidate=cid,epsilon=ep)
            pairs=raw_pairs if not full else [dict(pair_id=original_id,candidate=cid,cohort=cohort,axes=rz)]
            if accepted and ep==.0005:
                c.require([p['pair_id'] for p in pairs]==accepted['saved_cost_sample_tasks'],'ACCEPTED_COHORT_TASK_ORDER')
            components={k:signal.get(k) for k in ('total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias')}
            components['same_total_as_selected_endpoint']=components['total_signed_bias']==bias
            row=dict(candidate=cid,target=target,system=system,epsilon=ep,binding=bindings[system],unit=c.UNIT,
                cohort=cohort,bias_mode='USUAL_SIGNED',status='ELIGIBLE' if eligible else 'INELIGIBLE',
                N={ax:axes[ch]['shots'] for ax,ch in (('cosine','real'),('sine','imag'))},C=rz,G_saved=G,
                pairs=pairs,uncertainty=dict(G_SE=se,saved_axis_statistics=variance,
                      interpretation='SAVED_WITHIN_TRAJECTORY_COST_VARIABILITY'),source=pointer,
                explanation=dict(log_B_base=log_b,log_B_margin=margin,log_B_for_shots_saved=log_for,
                    signed_bias={ax:bias[ch] for ax,ch in (('cosine','real'),('sine','imag'))},
                    allowance=allowance,headroom={ax:axes[ch]['headroom'] for ax,ch in (('cosine','real'),('sine','imag'))},
                    log_shot_bound={ax:axes[ch]['log_shot_bound'] for ax,ch in (('cosine','real'),('sine','imag'))},
                    integer_status={ax:axes[ch]['integer_status'] for ax,ch in (('cosine','real'),('sine','imag'))},
                    components=components))
            data['rows'].append(row);data['availability'].append(dict(candidate=cid,epsilon=ep,available=True,source=pointer))
            if system=='H6' and ep==.0005:
                quote=next(x for x in h6['methods'] if x['candidate']['id']==cid)
                c.require(row['G_saved']==quote['G_saved'] and row['N']['cosine']==quote['N_cos'] and
                          row['N']['sine']==quote['N_sin'],'H6_PRIOR_RESULT_QUOTE')
    return c.validate(data),files,catalog,fixed


def prepare(source):
    source=subprocess.check_output(['git','rev-parse',source],cwd=c.ROOT,text=True).strip()
    c.ASSETS.mkdir(exist_ok=False);c.OUT.mkdir(parents=True,exist_ok=True);(c.OUT/'cache').mkdir(exist_ok=True)
    data,files,catalog,fixed=collect();sources=[]
    for rel in c.NEW+c.REUSED:
        p=c.ROOT/rel;raw=c.old.recovery.git_bytes(source,rel);c.require(p.read_bytes()==raw,'COMMITTED_RUNNER')
        if rel in c.REUSED:c.require(raw==c.old.recovery.git_bytes(c.BASE,rel),'UNCHANGED_REUSE')
        files[str(p)]=dict(path=str(p),sha256=c.sha(p),role='SOURCE:runner')
        sources.append(dict(root='runner',path=rel,local_path=str(p),sha256=c.sha(p),commit=source))
    for rel in c.REFERENCE_PATHS:
        fixed(SAVED,rel);p=c.ASSETS/'fixed'/SAVED/rel
        sources.append(dict(root='reference',path=rel,local_path=str(p),sha256=c.sha(p),commit=SAVED))
    cfg0=c.read(c.ROOT.parent/'h6-four-method-saved-results-assets-20261024/RANK_contract_PRIVATE.json')
    runtime_rows=[r for r in cfg0['files'] if r['role'].startswith('runtime:')]
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        c.require(all(pool.map(lambda r:c.sha(r['path'])==r['sha256'],runtime_rows)),'RUNTIME_UNCHANGED')
    files.update({r['path']:r for r in runtime_rows});runtime=dict(cfg0['runtime'])
    manifest=c.ASSETS/'inputs_PRIVATE.json';c.save(manifest,data)
    files[str(manifest)]=dict(path=str(manifest),sha256=c.sha(manifest),role='INPUT:fixed_projection')
    cfg=dict(schema=c.SERIES,source_commit=source,targets=c.TARGETS,physical=c.PHYSICAL,budget=c.counts(),
        limitations=c.LIMITS,retry=False,mandatory_STOP=True,files=list(files.values()),source_files=sources,
        runtime=runtime,manifest=str(manifest),manifest_sha256=c.sha(manifest),run_root=str(c.ASSETS/'ANALYSIS_run01'),
        schedule_source=str(c.ASSETS/'fixed'/SAVED/'src/trotterlib/h46_m5_plan.py'))
    cp=c.ASSETS/'ANALYSIS_contract_PRIVATE.json';c.save(cp,cfg)
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    r=subprocess.run([str(c.old.seal.PYTHON),'-I','-S','-B',str(c.ROOT/'scripts/launch_h246_order_cost_analysis.py'),
                      '--contract',str(cp)],env=env,capture_output=True,text=True)
    c.save(c.OUT/'cache/preflight_PRIVATE.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    c.require(r.returncode==0,'METADATA_PREFLIGHT:'+r.stderr[-1500:])
    def public(name,value):
        text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
        c.require('/home/' not in text and '/usr/' not in text,'PUBLIC_NO_INTERNAL_PATHS')
        (c.OUT/name).write_text(text)
    public('metadata_preflight.json',json.loads(r.stdout))
    public('input_availability.json',dict(targets=c.TARGETS,availability=data['availability'],catalog=catalog,
                                         numerical_operations=0,missing_epsilon_policy_substitution=False))
    public('SOURCE.json',dict(source_commit=source,base=c.BASE,files=[{k:v for k,v in r.items() if k!='local_path'} for r in sources]))
    public('execution_contract_DRAFT.json',dict(schema=c.SERIES,SOURCE=source,contract_sha256=c.digest(cfg),
        input_manifest_sha256=c.sha(manifest),targets=c.TARGETS,epsilons=c.EPS,exact_calls=c.counts(),
        exact_ledger_events=sum(c.counts().values()),endpoints=44,formula_contrasts=22,
        scope='A C/N/G additive/product; B source schedule and six first-trajectory witnesses; C saved shot relationship; D fixed epsilon directions',
        no_new_shot_or_cost_acquisition_or_sampling_statistics=True,no_H2_other_epsilon_substitution=True,limitations=c.LIMITS,
        numerical_execution_authorized=False,grants_issued=0,retry=False,mandatory_STOP=True))
    public('authority_templates.json',dict(actual_grants_issued=0,
        approval=dict(issuer='USER_EXPLICIT',execution_authorized=False,contract_sha256=c.digest(cfg),accepted_limitations=c.LIMITS),
        grant=dict(id=None,purpose=c.SERIES,state='INVALID_AWAITING_APPROVAL',budget=c.counts(),
                   contract_sha256=c.digest(cfg),approval_sha256=None,run_root='${H246_ANALYSIS_ASSETS}/ANALYSIS_run01',retry=False)))
    public('STOP.json',dict(status='SEALED_AWAITING_NUMERICAL_ANALYSIS_APPROVAL',actual_calls=dict.fromkeys(c.KINDS,0),
                           new_grants=0,strict_COMPARE='BLOCKED',mandatory_STOP=True))
    print(json.dumps(dict(status='READY_NO_REAL_ANALYSIS',SOURCE=source,contract_sha256=c.digest(cfg),
                         sealed_files=len(files),endpoints=44,contrasts=22,new_analysis_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    prepare(p.parse_args().source)
