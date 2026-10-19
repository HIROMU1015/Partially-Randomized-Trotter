"""Read-only joins and draft preparation. Never restores molecular arrays."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import h46_m5_cost_contract as c

PILOT_SOURCE='c7d53f04c3be131078ee91e8bef3c78ca434ed3e'
CT_SOURCE='885bbd7199c9ada26704b7db75b99767734e92e7'
RESULTS='artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20'

def csvrows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def fixed(path):
    p=c.ROOT/path;data=subprocess.check_output(['git','show',c.design.BASE+':'+path],cwd=c.ROOT)
    c.require(p.read_bytes()==data,'BASE_RESULT_BYTES:'+path);return c.read(p)

def verify_ct(cfg,path):
    paths=[Path(r['path']) for r in cfg['sealed_files'] if r['path'].endswith('/scripts/h6_m5_ct01_sine_contract.py')]
    c.require(len(paths)==1 and paths[0].read_bytes()==subprocess.check_output(['git','show',CT_SOURCE+':scripts/h6_m5_ct01_sine_contract.py'],cwd=paths[0].parents[1]),'ORIGINAL_CT01_SOURCE')
    code='import sys;sys.path.insert(0,sys.argv[1]);import h6_m5_ct01_sine_contract as c;v=c.read(sys.argv[2]);c.authorize(v);c.consumed(v)'
    r=subprocess.run([sys.executable,'-B','-c',code,str(paths[0].parent),str(path)],capture_output=True,text=True)
    c.require(r.returncode==0,'CT01_ORIGINAL_SEALS:'+r.stderr[-1200:])

def collect(signal_config,ct_config,private):
    private=Path(private);sig=c.original_signal(signal_config);ct=c.read(ct_config);verify_ct(ct,ct_config)
    records=fixed(RESULTS+'/signal_records.json')['records'];domain=c.design.selection(records);by_id={r['candidate']['id']:r for r in records}
    c.require([r['candidate'] for r in records]==c.read(c.ROOT/'references/h46_m5_signal_domain_v1.json')['rows'],'ORIGINAL88_UNCHANGED')
    original_raw=fixed(RESULTS+'/private_raw_inventory.json');authorization=Path(signal_config).parent
    used=[Path(signal_config),Path(ct_config)]
    for row in original_raw:
        p=authorization/row['path'];c.require(p.stat().st_size==row['bytes'] and c.sha(p)==row['sha256'],'SAVED_RUN02_RAW_SHA');used.append(p)
    pilot_inventory={r['path']:r for r in c.read(c.ROOT/c.signal_contract.PILOT_INVENTORY)}
    reuse=[];excluded=set();exclusion_evidence=[]
    expected=csvrows(c.ROOT/'artifacts/resource_applicability/track_a_h46_m5_minimal_comparison_plan/2026-10-19/deterministic_reuse_candidates.csv')
    for system,info in sig['systems'].items():
        origin=c.signal_contract.verify_original_pilot(info['origin_config']);used.append(Path(info['origin_config']))
        c.require(origin['SOURCE']==PILOT_SOURCE and origin['runtime']==sig['runtime']==ct['runtime'],'UNCHANGED_NATIVE_RUNTIME')
        for path in [origin['excluded_seed_path'],origin['registry_path']]:used.append(Path(path));exclusion_evidence.append(dict(role='old_seed_exclusion_or_registry',sha256=c.sha(path),bytes=Path(path).stat().st_size))
        ex=c.read(origin['excluded_seed_path'])['values'];c.require(all(type(x) is int for x in ex),'KNOWN_SEEDS_INTS');excluded.update(ex)
        for lane in c.read(origin['registry_path'])['lanes'].values():
            for task in lane.values():excluded.update(task.values())
        for candidate_id in info['reuse']:
            data=by_id[candidate_id];candidate=data['candidate'];task=next(t for t in origin['plan']['scope']['tasks'] if all(t['group'][k]==candidate[k] for k in ('method','formula','prefix','q')))
            p=Path(origin['output'])/task['id']
            if system=='H6' and candidate['method']=='FULL':
                recovery=c.read(ct['inputs']['old_config']['path']);ref=c.read(Path(recovery['reference_root'])/'receipt.json')
                c.require(recovery['binding']==origin['binding'] and recovery['inputs']==origin['inputs'] and recovery['runtime']==origin['runtime'],'CT01_RECOVERY_SAME_INPUT_POLICY')
                c.require(ref['full_preparation']==data['preparation_hash'],'CT01_FULL_PREPARATION_MATCH')
                c.require((c.ROOT/'scripts/h46_m5_native.py').read_bytes()==subprocess.check_output(['git','show',recovery['SOURCE']+':scripts/h46_m5_native.py'],cwd=c.ROOT),'CT01_UNMODIFIED_NATIVE_COMPOSITION')
                completion=c.read(Path(ct['run_root'])/'completion_witness.json');exited=c.read(ct['supervisor']['exit']);report=c.read(Path(ct['output'])/'report.json')
                c.require(report['status']=='CT01_SINE_PASS_MANDATORY_STOP' and exited['native_exit']==0 and exited['reaped'],'CT01_RESULT_AND_NATIVE_EXIT')
                c.require(report['old_STOP_preserved'] and report['compiler_policy']=='ORIGINAL_PRESET_OPT1_UNMODIFIED','OLD_STOP_ORIGINAL_COMPILER')
                proofpaths=[Path(ct['output'])/'report.json',Path(ct['run_root'])/'completion_witness.json',Path(ct['supervisor']['exit']),Path(ct['inputs']['old_config']['path']),Path(recovery['reference_root'])/'receipt.json']
                for axis in ('cosine','sine'):
                    receipt_path=Path(ct['output'])/('saved_cosine_acceptance.json' if axis=='cosine' else 'cost_sine.json');raw=c.read(receipt_path);used+=proofpaths+[receipt_path]
                    c.require(raw['status']==('SAVED_COSINE_ACCEPTED_UNDER_CT01' if axis=='cosine' else 'PILOT_AXIS_ACCEPTED_CT01'),'CT01_EXPLICIT_AXIS_ACCEPTANCE')
                    if axis=='cosine':c.require(raw==ct['cosine_acceptance'] and raw['new_PASS'] and not raw['old_PASS'],'CT01_SEPARATE_NOT_OLD_PASS')
                    else:
                        gate=Path(ct['output'])/'compiled_vector_sine.json';used.append(gate);c.require(c.read(gate)['PASS'] and c.read(gate)['strictly_below']==1e-7,'CT01_SINE_GATE')
                    match=next(r for r in expected if r['candidate_id']==candidate_id and r['axis']==axis)
                    c.require(raw['metrics']['rz_count']==int(match['compiled_RZ']) and raw['compiler_settings_hash']==c.design.COMPILER_HASH,'CT01_SAVED_INTEGER_COST_COMPILER')
                    reuse.append(reuse_row(candidate,axis,raw,receipt_path,origin['binding'],data['preparation_hash'],c.sha(Path(ct['output'])/'report.json'),raw.get('original_SOURCE',raw.get('SOURCE')),True))
            else:
                result_path=p/'result.json';result=c.read(result_path);used.append(result_path)
                c.require(result['status']=='PASS' and result['SOURCE']==PILOT_SOURCE and result['binding']==origin['binding'] and result['preparation_hash']==data['preparation_hash'] and result['native_policy']==c.design.NATIVE_POLICY,'PILOT_COST_IDENTITY')
                for axis in ('cosine','sine'):
                    rp=p/('cost_'+axis+'.json');raw=c.read(rp);used.append(rp)
                    entry=pilot_inventory[str(rp.relative_to(origin['run_root']))]
                    c.require(c.sha(rp)==entry['sha256'] and rp.stat().st_size==entry['bytes'],'ORIGINAL_COST_INVENTORY')
                    c.require(raw in result['records'] and raw['compiler_settings_hash']==c.design.COMPILER_HASH and raw['native_policy']==c.design.NATIVE_POLICY,'AXIS_RECEIPT_IN_PASS_RESULT')
                    match=next(r for r in expected if r['candidate_id']==candidate_id and r['axis']==axis)
                    c.require(raw['metrics']['rz_count']==int(match['compiled_RZ']),'EXACT_SAVED_COST_TRANSCRIPTION')
                    reuse.append(reuse_row(candidate,axis,raw,rp,origin['binding'],data['preparation_hash'],c.sha(result_path),PILOT_SOURCE,False))
    require=c.require;require(len(reuse)==12,'EXACT12_REUSE_AXES')
    selection={r['candidate']['id']:r for r in domain}
    for r in reuse:r['eligible_in_current_five_eps']=bool(selection[r['candidate_id']]['eligible_epsilons'])
    manifest=dict(schema='H46_M5_UNIT_COST_PREPARATION_V1',status='DRAFT_USER_DECISIONS_REQUIRED_NO_EXECUTION',signal_run_commit=c.design.BASE,
      signal_SOURCE=sig['SOURCE'],signal_plan_sha256=sig['plan_sha256'],native_policy=c.design.NATIVE_POLICY,compiler=c.design.COMPILER,compiler_settings_hash=c.design.COMPILER_HASH,
      compile_vector_tolerance=1e-7,other_native_tolerance=1e-9,candidate_table=domain,reuse_axes=reuse,sampling_proposal=c.design.sampling_proposal(domain),
      additional_DET_axes={s:2*len(c.design.tasks_for(s+'_DET',domain,{(r['candidate_id'],r['axis']) for r in reuse})) for s in ('H4','H6')},
      shot_policy_adopted=False,AD03='DEFERRED',PF788_contract_separate=True,valid_grants=0,scientific_dispatch=0,mandatory_STOP=True)
    c.save(c.ROOT/c.MANIFEST,manifest)
    c.save(private/'known_excluded_seeds_PRIVATE.json',dict(values=sorted(excluded),evidence=exclusion_evidence,scope='REGISTERED_KNOWN_EXCLUSIONS_NOT_EVERY_FUTURE_OTHER_RESEARCH_SEED',future_fresh_registry_recheck_required=True))
    c.save(private/'used_evidence_seals_PRIVATE.json',[c.seal(p,'readiness_evidence:'+p.name) for p in sorted(set(used))])
    return manifest

def reuse_row(candidate,axis,raw,path,binding,preparation,result_sha,source,ct01):
    return dict(candidate_id=candidate['id'],system=candidate['system'],method=candidate['method'],prefix=candidate['prefix'],q=candidate['q'],axis=axis,
      compiled_RZ=raw['metrics']['rz_count'],receipt_sha256=c.sha(path),PASS_result_sha256=result_sha,cost_SOURCE=source,compiler_settings_hash=raw['compiler_settings_hash'],
      preparation_hash=preparation,input_binding=binding,native_policy=c.design.NATIVE_POLICY,
      profile_identity='SAME_DECLARED_REFERENCE_PROFILE_WITHIN_M5_SERIES;OLD_MAIN_RUNTIME_EQUIVALENCE_NOT_CERTIFIED',
      fingerprint=raw.get('fingerprint',raw.get('wrapper_fingerprint')),identity_join='MATCHED_WITHIN_M5_NATIVE_SERIES',
      CT01_separate_acceptance=ct01,old_STOP_preserved=True,formal_shots_or_main_adoption=False)

def seal_drafts(source,signal_config,private):
    """Seal four disabled configs. PR n/registry remain null until adoption."""
    private=Path(private);origin=c.original_signal(signal_config);evidence=c.read(private/'used_evidence_seals_PRIVATE.json')
    files={r['path']:r for r in origin['sealed_files']}
    for r in evidence:files[r['path']]=r
    # Added stdlib pool/runtime is sealed in addition to historical closure.
    import run_h46_m5_cost,concurrent.futures.process,concurrent.futures.thread
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to('/usr/lib/python3.12'):
            r=c.seal(Path(path).resolve(),'cost_stdlib_runtime');files[r['path']]=r
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel;c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COST_SOURCE_COMMITTED:'+rel)
        r=c.seal(p,'cost_SOURCE:'+rel);files[r['path']]=r
    ex=private/'known_excluded_seeds_PRIVATE.json';r=c.seal(ex,'known_seed_exclusions');files[r['path']]=r
    m=c.manifest();reuse={(r['candidate_id'],r['axis']) for r in m['reuse_axes']}
    for stage in c.STAGES:
        dest=private/('proposal_'+stage);dest.mkdir(exist_ok=False);s,kind=stage.split('_')
        decisions=dict(cost_domain_accepted=False,empirical_signal_selection_limit_accepted=False,resources_policy_accepted=False,formal_shot_policy_adopted=False,
          exploration_design_accepted=False,rare_event_limit_accepted=False,n_by_candidate={r['candidate']['id']:None for r in m['candidate_table'] if r['candidate']['system']==s and r['candidate']['method']=='PR'} if kind=='PR' else {})
        tasks=c.design.tasks_for(stage,m['candidate_table'],reuse) if kind=='DET' else None
        approval=dest/'approval_REQUIRED_PRIVATE.json';c.save(approval,dict(SOURCE=source,stage=stage,execution_authorized=False,decisions=decisions,call_scope=c.design.call_scope(tasks,kind) if tasks is not None else None))
        closure=dict(files);r=c.seal(approval,'future_cost_approval');closure[r['path']]=r
        # Resource planning only: scale prior same-input observation by the
        # registered q ratio. This is explicitly not a memory upper bound.
        observed=6353003861 if s=='H4' else 22557356*1024
        resources=dict(worker_RSS_proxy_bytes=observed*4,proxy_source='PRIOR_SAME_INPUT_NATIVE_Q2_RSS_TIMES_MAX_Q_RATIO4_NOT_UPPER_BOUND',disk_proxy_bytes=sum(r['bytes'] for r in origin['systems'][s]['inputs'].values()),runtime_caps=False,host_PSI_kill=False,live_own_resource_observation=True)
        root=dest/'future_run01';supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
        cfg=dict(SOURCE=source,stage=stage,origin_signal_config=str(Path(signal_config).resolve()),systems=origin['systems'],runtime=origin['runtime'],provider_root=origin['provider_root'],
          compiler=c.design.COMPILER,compiler_settings_hash=c.design.COMPILER_HASH,compile_tolerance=1e-7,other_tolerance=1e-9,
          sealed_files=list(closure.values()),decisions=decisions,tasks=tasks,call_scope=c.design.call_scope(tasks,kind) if tasks is not None else None,
          campaign_id=None,registry_path=None,excluded_seeds_path=str(ex),resources=resources,authorized=False,
          approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval),run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),supervisor=supervisor)
        cfg['pipeline_identity']=c.digest(dict(SOURCE=source,stage=stage,manifest_sha256=c.sha(c.ROOT/c.MANIFEST),decisions=decisions,runtime=cfg['runtime']))
        plan=dict(status='DRAFT_DECISIONS_UNAPPROVED',SOURCE=source,stage=stage,closure_digest=c.digest(cfg['sealed_files']),decisions=decisions,resources=resources,tasks=tasks,call_scope=cfg['call_scope'],run_root=str(root),supervisor=supervisor,approval_receipt_sha256=c.sha(approval),pipeline_identity=cfg['pipeline_identity'],campaign_id=cfg['campaign_id'],registry_path=cfg['registry_path'],excluded_seeds_path=cfg['excluded_seeds_path'])
        pp=dest/'plan_DRAFT_PRIVATE.json';c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
        cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,SOURCE=source,stage=stage,plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],call_scope=cfg['call_scope'],retry=False,next_phase_authorized=False,**{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
        c.verify(cfg);c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID_PRIVATE.json',cfg['grant'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--signal-config',required=True);p.add_argument('--ct-config');p.add_argument('--private',required=True);p.add_argument('--seal-source');a=p.parse_args()
    if a.seal_source:seal_drafts(a.seal_source,a.signal_config,a.private)
    else:
        c.require(a.ct_config,'CT_CONFIG_REQUIRED');m=collect(a.signal_config,a.ct_config,a.private);print(json.dumps(dict(DET_additional_axes=m['additional_DET_axes'],reuse_axes=len(m['reuse_axes']),n_E=None,science_started=False)))
