"""Bind saved evidence to an unapproved batch; never evaluate real numerical kernels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import h246_reversal_batch as c

EDIR='artifacts/resource_applicability/track_a_h246_reversal_evidence_controls/2026-10-24/'
RDIR='artifacts/resource_applicability/track_a_h246_order_cost_analysis_run01/2026-10-24/'
CONFIRM='3eca46148070ba4d4b0f5815ef45943407cc959e'
CDIR='artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/'


def public(name,value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    c.require('/home/' not in text and '/usr/' not in text,'NO_PRIVATE_PATHS')
    (c.OUT/name).write_text(text)


def prepare(source):
    source=subprocess.check_output(['git','rev-parse',source],cwd=c.ROOT,text=True).strip()
    c.ASSETS.mkdir(exist_ok=False);c.OUT.mkdir(parents=True,exist_ok=True);(c.OUT/'cache').mkdir()
    files={};catalog=[];sources=[];protected={}
    def protect(path,expected=None):
        p=Path(path).resolve();h=c.sha(p);c.require(expected is None or h==expected,'KNOWN_SAVED_INPUT_SHA')
        protected[str(p)]=h
        return p
    def add(path,role,expected=None):
        p=protect(path,expected);files[str(p)]=dict(path=str(p),sha256=protected[str(p)],role=role);return p
    def fixed(commit,path):
        raw=c.prior.old.recovery.git_bytes(commit,path);p=c.ASSETS/'fixed'/commit/path
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);add(p,'INPUT:fixed_Git')
        catalog.append(dict(commit=commit,path=path,sha256=c.sha(p)))
        return json.loads(raw) if path.endswith('.json') else raw.decode()
    oldroot=c.ROOT.parent/'h246-reversal-evidence-controls-20261024'
    for name in ('all_inputs_before_PRIVATE.json','controls_receipts_before_PRIVATE.json'):
        baseline=c.read(oldroot/EDIR/'cache'/name)
        with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
            c.require(all(pool.map(lambda p:c.sha(p)==baseline[p],baseline)),'PREVIOUS_PROTECTED_INPUTS')
        protected.update(baseline)
    controls=fixed(c.BASE,EDIR+'same_q_controls.json')['controls']
    prior_draft=fixed(c.BASE,EDIR+'additional_validation_DRAFT.json')
    h2meta=fixed(c.BASE,EDIR+'H2_missing_precision_inputs.json')
    fixed(c.BASE,EDIR+'independent_confirmation_scope.json')
    pairs_path=c.ROOT.parent/'h246-reversal-evidence-controls-assets-20261024/same_q_paired_inputs_PRIVATE.json'
    add(pairs_path,'INPUT:previous_control_pairs',prior_draft['same_q']['paired_inputs_manifest_sha256']);pairs=c.read(pairs_path)
    oldcfg=c.read(c.prior.ASSETS/'ANALYSIS_contract_PRIVATE.json')
    add(oldcfg['manifest'],'INPUT:accepted_analysis_rows',oldcfg['manifest_sha256']);original=c.read(oldcfg['manifest'])
    quoted=fixed('d2536c4f959bfb4edd7d2deeceb155585fb53eb0',RDIR+'analysis.json')['comparisons']
    data=dict(schema=c.SERIES+'_INPUTS',kind='SAVED_REAL_VALUES',physical=c.PHYSICAL,
        A_targets=c.A_TARGETS,B_targets=c.B_TARGETS,B_epsilons=c.EPS,limitations=c.LIMITS,
        bindings=original['bindings'],A_rows=[],B_bases=[],quoted_main_comparisons=quoted,
        policy_bindings={},account_bindings={},cohort_pooling=False,recompute_main_0005=False)
    for t in c.A_TARGETS:
        cid=t['id'];meta=next(x[f] for x in controls if x['complete_saved_pair'] for f in ('S2','Yoshida4') if x[f]['candidate']==cid)
        pair=next(x for x in pairs if x['candidate']==cid)
        c.require(all(meta[k]==pair[k] for k in ('candidate','system','epsilon','status','C','N')) and meta['G']==pair['G_saved'],'SAVED_CONTROL_PAIR_JOIN')
        binding=dict(reference=meta['reference'],u=meta['u'],log_B=meta['log_B'],margin=meta['log_B_margin'],
            compiler=meta['compiler'],native=meta['native'],K=meta['K'],r_policy=meta['r_policy'],r_value=meta['r_value'])
        data['policy_bindings'][cid]=copy.deepcopy(binding)
        data['A_rows'].append(dict(pair,target=t,binding=data['bindings'][t['system']],unit=c.prior.UNIT,
            cohort=meta['cohort'],bias_mode='USUAL_SIGNED',subscope='A_SAME_Q',policy_binding=binding,
            uncertainty=dict(G_SE=meta['paired_G_SE'],saved_axis_statistics=meta['saved_unit_statistics'],interpretation='SAVED_WITHIN_TRAJECTORY_COST_VARIABILITY'),
            source=meta['source'],saved_bias=meta['bias'],saved_log_B=meta['log_B'],saved_u=meta['u']))
    for t in c.B_TARGETS:
        row=copy.deepcopy(next(r for r in original['rows'] if r['candidate']==t['id'] and r['epsilon']==.0005))
        m=next(x for x in h2meta['candidates'] if x['candidate']==t['id']);e=row['explanation']
        a=dict(diagnostic=dict(bias={ch:e['signed_bias'][ax] for ax,ch in (('cosine','real'),('sine','imag'))},primary=dict(log_B=e['log_B_base'])),
            allowance=dict(real=e['allowance']['cosine'],imag=e['allowance']['sine'],log_B_margin=e['log_B_margin']))
        binding=dict(allowance=a['allowance'],reference=m['reference_binding'],policy='ACCEPTED_H2_DIAGNOSTICS_UNCHANGED',
            bias=m['bias'],log_B=m['log_B'],input_manifest_sha256=m['source_input_manifest_sha256'])
        row.update(subscope='B_H2_PRECISION',policy_binding=binding,account_inputs=a)
        data['B_bases'].append(row);data['policy_bindings'][t['id']]=copy.deepcopy(binding);data['account_bindings'][t['id']]=copy.deepcopy(a)
    c.validate(data)
    # D is a static join only; independent S2 repricing is explicitly outside this batch.
    contracts=fixed(CONFIRM,CDIR+'stage_contracts.json')['contracts']
    samples=list(csv.DictReader(fixed(CONFIRM,CDIR+'fixed_evaluation/paired_samples_118.csv').splitlines()))
    fixed(CONFIRM,CDIR+'input_reconstruction.json');fixed(CONFIRM,CDIR+'reconstruction_end_audit.json')
    fixed(CONFIRM,CDIR+'fixed_evaluation/raw_archive_inventory.json')
    plan_cfg=c.read(c.ROOT.parent/'h4-four-method-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json')
    pp=next(r['path'] for r in plan_cfg['files'] if r['role']=='PROTECTED:original_analysis_authority_and_results' and r['path'].endswith('plan_APPROVED_PRIVATE.json'))
    protect(pp);plan=c.read(pp);dscope=[]
    for system in ('H4','H6'):
        stage=next(x for x in contracts if x['stage']==system+'_CONFIRMATION')
        result=fixed(CONFIRM,CDIR+'fixed_evaluation/'+system+'_fixed_comparison.json')
        forward=fixed(CONFIRM,CDIR+'published_inputs/'+system+'/forward_receipt.json')
        group=next(g for g in result['groups'] if g['candidate']['formula']=='2nd');cid=group['candidate']['id']
        oldgroup=next(g for g in plan['groups'] if g['record']['candidate']['id']==cid)
        selected=[r for r in samples if r['system']==system and r['candidate_id']==cid]
        checks={k:stage['input_binding'][k]==data['bindings'][system][k] for k in ('hamiltonian','state','rank','rank_policy','basis')}
        checks['compiler']=stage['input_binding']['compiler']==c.prior.old6.h4.COMPILER
        checks['preparation']=forward['cut_preparation']==oldgroup['record']['preparation_hash']
        checks['law']=all(group['candidate'][k]==oldgroup['record']['candidate'][k] for k in ('formula','prefix','q','K','r_policy','r_value'))
        checks['paired_samples']=len(selected)==group['statistics']['n'] and all(r['cosine_RZ'] and r['sine_RZ'] and r['SOURCE']==result['SOURCE'] for r in selected)
        native_paths=('src/trotterlib/h46_m5_plan.py','scripts/h46_new2_core.py','scripts/h46_new2_native.py',
            'scripts/h46_m5_native.py','scripts/h46_m5_cost_native.py')
        native_checks={rel:fixed(oldgroup['acquisition_SOURCE'],rel)==fixed('c028f1c2885afdba65ef8c62c075738aadd7323d',rel) for rel in native_paths}
        checks['native_numerical_SOURCE']=all(native_checks.values())
        dscope.append(dict(system=system,S2_candidate=cid,n=group['statistics']['n'],checks=checks,
            input_container_identity=stage['input_identity'],snapshot_equal=stage['input_binding']['snapshot']==data['bindings'][system]['snapshot'],
            preparation_sha256=forward['cut_preparation'],candidate_law=group['candidate'],
            saved_statistics=group['statistics'],saved_epsilon=stage['analysis_scope']['epsilon'],
            status='SAVED_S2_UNIT_COST_REUSE_PREPARATION_PASS_REPRICING_REQUIRES_NEW_CONTRACT' if all(checks.values()) else 'UNRESOLVED_BINDING',
            native_SOURCE_joins=native_checks,receipt_scope='Accepted fixed paired samples and stage audit; not a new strict full-stream/runtime identity certificate',
            no_epsilon_001_G_or_SE_computed=True,independent_cost_repricing_in_current_budget=False,
            native_SOURCE='c028f1c2885afdba65ef8c62c075738aadd7323d',original_scientific_SOURCE='8825391ff73b531be5834bf852b9bf2479d1799b'))
    for rel in ('scripts/h246_reconstructed_contract.py','scripts/h246_confirmation_contract.py','scripts/h246_confirmation_evaluation.py','scripts/run_h246_confirmation.py'):
        fixed('c028f1c2885afdba65ef8c62c075738aadd7323d',rel)
    public('independent_confirmation_preparation.json',dict(status='PREPARATION_ONLY_NOT_AUTHORIZED',S2_saved_scope=dscope,
        Y4_targets=[x for x in c.A_TARGETS if x['method']=='PR' and x['formula']=='4th' and x['q']==4],
        primary_epsilon=.001,Y4_independent_samples_found=False,minimum_new_groups_if_S2_receipts_join=2,
        sample_counts=None,new_axes='2*(n_H4+n_H6), sample sizes not chosen',
        design_needed=['Scientific claim and target Monte Carlo precision/confidence','Saved exploration variance is planning information, not a tail bound','Independent new cohort with private namespace and available exclusions; historical seed coverage limits explicit','Nonadaptive sample size/stop policy, no automatic top-up','Input/preparation/native/compiler receipt join, actual runtime/source and exact internal call scope','S2 epsilon .001 reweighting and any new SE/interval require a separate numerical contract'],
        no_new_sampling_or_compile=True,new_valid_grants=0,adoption_deferred_until_A_B_C=True))
    for rel in c.NEW+c.REUSED:
        p=c.ROOT/rel;raw=c.prior.old.recovery.git_bytes(source,rel);c.require(p.read_bytes()==raw,'COMMITTED_RUNNER')
        if rel in c.REUSED:c.require(raw==c.prior.old.recovery.git_bytes(c.BASE,rel),'UNCHANGED_OLD_NUMERICS')
        add(p,'SOURCE:runner');sources.append(dict(root='runner',path=rel,local_path=str(p),sha256=c.sha(p),commit=source))
    for rel in c.PROVIDER_PATHS:
        p=c.ASSETS/'provider'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(c.prior.old.recovery.git_bytes(c.PROVIDER,rel))
        add(p,'SOURCE:provider');sources.append(dict(root='provider',path=rel,local_path=str(p),sha256=c.sha(p),commit=c.PROVIDER))
    runtime_rows=[r for r in oldcfg['files'] if r['role'].startswith('runtime:')]
    for r in runtime_rows:add(r['path'],r['role'],r['sha256'])
    manifest=c.ASSETS/'inputs_PRIVATE.json';c.save(manifest,data);add(manifest,'INPUT:fixed_projection')
    cfg=dict(schema=c.SERIES,source_commit=source,physical=c.PHYSICAL,A_targets=c.A_TARGETS,B_targets=c.B_TARGETS,
        B_epsilons=c.EPS,limitations=c.LIMITS,budget=c.counts(),retry=False,mandatory_STOP=True,
        source_files=sources,files=list(files.values()),runtime=oldcfg['runtime'],manifest=str(manifest),
        manifest_sha256=c.sha(manifest),run_root=str(c.ASSETS/'BATCH_run01'))
    cp=c.ASSETS/'BATCH_contract_PRIVATE.json';c.save(cp,cfg)
    (c.OUT/'cache/protected_before_PRIVATE.json').write_text(json.dumps(protected,indent=2)+'\n')
    env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    r=subprocess.run([cfg['runtime']['python_target'],'-I','-S','-B',str(c.ROOT/'scripts/launch_h246_reversal_batch.py'),'--contract',str(cp)],env=env,capture_output=True,text=True)
    (c.OUT/'cache/preflight_PRIVATE.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    c.require(r.returncode==0,'METADATA_PREFLIGHT:'+r.stderr[-1500:]);public('metadata_preflight.json',json.loads(r.stdout))
    public('execution_contract_DRAFT.json',dict(schema=c.SERIES,SOURCE=source,contract_sha256=c.digest(cfg),input_manifest_sha256=c.sha(manifest),
        A_targets=c.A_TARGETS,B_targets=c.B_TARGETS,B_epsilons=c.EPS,physical=c.PHYSICAL,
        exact_calls=c.counts(),exact_ledger_events=sum(c.counts().values()),
        subscope_calls=dict(A=dict(A_decomposition=8),B=dict(B_account=16,u_aware_shots=16,axis_headroom=32,B_decomposition=8),shared=dict(bootstrap=1,bundle_load=1,integration_summary=1)),
        H2_main_epsilon_recomputed=False,H2_new_G_SE_computed=False,quoted_old_comparisons=22,
        limitations=c.LIMITS,execution_authorized=False,valid_grants=0,retry=False,mandatory_STOP=True))
    public('SOURCE_and_inputs.json',dict(SOURCE=source,base_commit=c.BASE,source_files=[{k:v for k,v in r.items() if k!='local_path'} for r in sources],
        catalog=catalog,sealed_files=len(files),protected_files=len(protected),new_real_analysis_calls=0))
    public('STOP.json',dict(status='READY_AWAITING_ONE_EXPLICIT_BATCH_APPROVAL',used=dict.fromkeys(c.KINDS,0),valid_grants=0,
        mandatory_STOP=True,strict_COMPARE='BLOCKED'))
    print(json.dumps(dict(status='READY_NO_REAL_ANALYSIS',SOURCE=source,contract_sha256=c.digest(cfg),input_manifest_sha256=c.sha(manifest),sealed_files=len(files),protected_files=len(protected),exact_events=sum(c.counts().values()))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);prepare(p.parse_args().source)
