"""New migration packs; unchanged fixed confirmation numerics and call counts.

Old private packs are not claimed restored. Historical seed coverage is partial.
"""
import copy
import os
from pathlib import Path
import secrets
import subprocess
import sys
from types import SimpleNamespace
import h246_confirmation_contract as original

ROOT=original.ROOT
STAGES=original.STAGES
read,sha,digest,require=original.read,original.sha,original.digest,original.require
save,seal=original.save,original.seal
law,d,storage=original.law,original.d,original.storage
tasks_for,comparison,analysis_scope=original.tasks_for,original.comparison,original.analysis_scope
task_limits,consumed,reserve,manifest=original.task_limits,original.consumed,original.reserve,original.manifest
COMPILE_TOLERANCE,OTHER_TOLERANCE=original.COMPILE_TOLERANCE,original.OTHER_TOLERANCE
OLD_SOURCE='8825391ff73b531be5834bf852b9bf2479d1799b'
NEW_PATHS=('scripts/h246_reconstructed_contract.py','scripts/run_h246_reconstructed.py')
RT=None


def make_packs(private):
    import numpy as np,qiskit
    import concurrent.futures.thread,concurrent.futures._base,queue,_queue,threading
    import symengine
    private=Path(private).resolve()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Original numerical source is kept byte-identical, independently of new contracts.
    for rel,blob in original.source_blobs(OLD_SOURCE).items():
        require((ROOT/rel).read_bytes()==blob,'ORIGINAL_SCIENTIFIC_SOURCE:'+rel)
    require(np.__version__=='1.26.4' and qiskit.__version__=='1.3.0','PINNED_NUMPY_QISKIT')
    recovery=read(private/'recovery_manifest.json')
    provider=next(x for x in recovery if 'provider_root' in x)
    namespace='H246_MIGRATED_CONFIRMATION_20261022_'+source+'_'+secrets.token_hex(32)
    # Reconstruct the old proposed seeds as a known subset only. Never substitute
    # this subset for the unavailable historical 256k exclusion evidence.
    prior=set()
    for stage in STAGES:
        old=law.registry(tasks_for(stage),OLD_SOURCE,'H246_HIGHER_PF_CONFIRMATION_PROPOSED_RUN01_'+OLD_SOURCE,'CONFIRMATION',[])
        prior.update(v for row in old.values() for v in row.values())
    registries={s:law.registry(tasks_for(s),source,namespace,'CONFIRMATION',sorted(prior)) for s in STAGES}
    seeds={s:{v for row in r.values() for v in row.values()} for s,r in registries.items()}
    require(sum(map(len,seeds.values()))==len(set.union(*seeds.values()))==846,'NEW_CROSS_STAGE_SEED_COLLISION')
    capacities=read(ROOT/'artifacts/resource_applicability/track_a_h246_higher_pf_confirmation_readiness/2026-10-21/stage_readiness.json')
    old_binding=read(ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/input_provenance.json')
    paths=[]
    for stage in STAGES:
        system=stage[:2];dest=private/'confirmation'/stage
        dest.mkdir(parents=True,exist_ok=False);root=dest/'run01'
        forward=private/'forward'/system/'run01'
        require(read(forward/'completion.json')['status']=='REFERENCE_AND_FROZEN_SHOT_APPLICABILITY_PASS','FORWARD_COMPLETE')
        binding=copy.deepcopy(next(x['binding'] for x in old_binding['input_metadata'] if x['system']==system))
        if system=='H6':
            p=private/'recovered/H6';inputs={name:seal(p/file,'restored_original_input') for name,file in
                [('H6_snapshot.npz','h6_input_snapshot.npz'),('H6_snapshot_receipt.json','snapshot_receipt.json'),('H6_df_receipt.json','df_receipt.json')]}
            identity='ORIGINAL_H6_SNAPSHOT_SHA_IDENTICAL'
        else:
            p=private/'generation'/system/'run01'
            result=read(p/'comparison.json')
            require(all(result['array_hash_comparisons'].values()) and result['same_hamiltonian_fingerprint'] and
                result['same_state_fingerprint'],'REGENERATED_ALL_SCIENTIFIC_ARRAYS_IDENTICAL')
            inputs={('snapshot' if system=='H2' else 'H4_tol_snapshot.npz'):seal(p/'tol_only_snapshot.npz','regenerated_input'),
                ('snapshot_receipt' if system=='H2' else 'H4_tol_receipt.json'):seal(p/'tol_only_receipt.json','regenerated_input_receipt')}
            identity='REGENERATED_CONTAINER_DIFFERENT_ALL_SCIENTIFIC_ARRAYS_IDENTICAL'
        snapshot=next(r for k,r in inputs.items() if k=='snapshot' or k.endswith('_snapshot.npz'))
        binding['snapshot']=snapshot['sha256']
        rp=dest/'registry_PRIVATE.json';ep=dest/'known_exclusions_PRIVATE.json'
        save(rp,registries[stage])
        excluded=prior|set.union(*(v for s,v in seeds.items() if s!=stage))
        save(ep,dict(values=sorted(excluded),coverage='OLD_SOURCE_PROPOSED_SEEDS_AND_NEW_OTHER_STAGES_ONLY',
            historical_complete_exclusions_available=False,complete_old_seed_noncollision_guaranteed=False))
        approval=dest/'approval_APPROVED_PRIVATE.json'
        tasks=tasks_for(stage);scope=law.call_scope(tasks,'PR');analysis=analysis_scope(stage)
        save(approval,dict(role='user',execution_authorized=True,SOURCE=source,stage=stage,
            old_SOURCE=OLD_SOURCE,tasks=tasks,call_scope=scope,analysis_scope=analysis,
            migration_and_new_namespace_explicitly_authorized=True,
            complete_historical_seed_noncollision_guaranteed=False,retry=False,
            user_instruction='Reconstruct missing materials with original fixed science; execute the fixed 118 trajectories and 236 compile axes, fixed evaluation, audits and mandatory STOP without intermediate reapproval.'))
        activation=dest/'GRANT_ISSUED.json'
        save(activation,dict(SOURCE=source,stage=stage,run_root=str(root),approval_sha256=sha(approval),retry=False))
        runtime=dict(python=sys.executable,qiskit=qiskit.__version__,numpy=np.__version__,provider_root=provider['provider_root'],provider_SOURCE=provider['provider_SOURCE'])
        cfg=dict(SOURCE=source,old_SOURCE=OLD_SOURCE,stage=stage,cohort='CONFIRMATION',
            input_identity=identity,provider_root=provider['provider_root'],runtime=runtime,
            tasks=tasks,comparison=comparison(stage),call_scope=scope,analysis_scope=analysis,
            campaign_id=namespace,registry_path=str(rp),excluded_seeds_path=str(ep),
            complete_historical_seed_noncollision_guaranteed=False,
            systems={system:dict(binding=binding,inputs=inputs,reference_root=str(forward))},
            inputs=inputs,binding=binding,reference_path=str(forward/'reference.npz'),
            preparation_hash=read(forward/'receipt.json')['cut_preparation'],
            signal_path=str(forward/'signals.json'),storage=storage.storage_policy(),
            resources=copy.deepcopy(next(x['resources'] for x in capacities['stages'] if x['stage']==stage)),
            compiler_settings_hash=law.COMPILER_HASH,compile_tolerance=COMPILE_TOLERANCE,other_tolerance=OTHER_TOLERANCE,
            compiler_phase_policy='H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1',
            run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),
            supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('exit','external_exit_witness_PRIVATE.json'),('log','native_PRIVATE.log')]},
            approval_receipt_path=str(approval),approval_receipt_sha256=sha(approval),approval_activation_path=str(activation),authorized=True)
        files={}
        def add(path,role):
            r=seal(Path(path).resolve(),role);files[r['path']]=r
        for rel in (*original.SOURCE_PATHS,*NEW_PATHS):add(ROOT/rel,'scientific_SOURCE:'+rel)
        for p in sorted(Path(provider['provider_root']).rglob('*.py')):add(p,'pinned_provider')
        modules=(np,qiskit,symengine,concurrent.futures.thread,concurrent.futures._base,queue,_queue,threading)
        for module in modules:add(module.__file__,'loaded_runtime_identity')
        # Pin the installed compiler/array binary closure, not unrelated packages.
        for module in (np,qiskit,symengine):
            for p in Path(module.__file__).parent.rglob('*.so'):add(p,'compiler_array_runtime_binary')
        add(Path(sys.executable).resolve(),'fixed_python_binary')
        for r in inputs.values():add(r['path'],'scientific_input')
        add(private/'recovery_manifest.json','original_recovery_provenance')
        if system!='H6':
            for name in ('comparison.json','tol_only_receipt.json'):
                add(private/'generation'/system/'run01'/name,'regenerated_array_identity_provenance')
        for name in ('reference.npz','receipt.json','signals.json','completion.json'):add(forward/name,'restored_forward_input')
        for p,role in [(rp,'new_seed_registry'),(ep,'partial_known_exclusions'),(approval,'explicit_new_approval'),(activation,'one_time_issuance')]:add(p,role)
        cfg['sealed_files']=list(files.values())
        plan={k:v for k,v in cfg.items() if k!='sealed_files'}
        plan['closure_digest']=digest(cfg['sealed_files'])
        pp=dest/'plan_APPROVED_PRIVATE.json';save(pp,plan)
        cfg.update(plan=plan,plan_path=str(pp),plan_sha256=sha(pp))
        cfg['grant']=dict(valid=True,grant_id=stage+'_MIGRATED_RUN01_'+digest(plan),retry=False,
            next_phase_authorized=False,approval_reference=str(approval)+'#sha256='+sha(approval),
            **{k:cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','call_scope','analysis_scope','output','ledger','marker','supervisor')})
        authorize(cfg)
        cp=dest/'config_APPROVED_PRIVATE.json';save(cp,cfg);save(dest/'grant_APPROVED_PRIVATE.json',cfg['grant']);paths.append(str(cp))
    save(private/'confirmation/configs_PRIVATE.json',dict(configs=paths,SOURCE=source,new_namespace=True,old_pack_restored=False,
        trajectories=118,compile_axes=236,complete_historical_seed_noncollision_guaranteed=False))
    return paths


def verify(cfg):
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'MIGRATED_PLAN_SHA')
    require({k:cfg[k] for k in cfg['plan'] if k!='closure_digest'}=={k:v for k,v in cfg['plan'].items() if k!='closure_digest'},'ALL_MIGRATED_PLAN_BINDINGS')
    require(cfg['tasks']==tasks_for(cfg['stage']) and cfg['comparison']==comparison(cfg['stage']) and
        cfg['call_scope']==law.call_scope(cfg['tasks'],'PR') and cfg['analysis_scope']==analysis_scope(cfg['stage']),
        'ORIGINAL_FIXED_CANDIDATES_COUNTS_BUDGET_SHOTS')
    require(cfg['cohort']=='CONFIRMATION' and cfg['old_SOURCE']==OLD_SOURCE,'SCIENTIFIC_SOURCE_COHORT')
    system=cfg['stage'][:2]
    expected=copy.deepcopy(next(x['binding'] for x in read(ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/input_provenance.json')['input_metadata'] if x['system']==system))
    snapshot=next(r for k,r in cfg['inputs'].items() if k=='snapshot' or k.endswith('_snapshot.npz'))
    expected['snapshot']=snapshot['sha256']
    require(cfg['binding']==expected==cfg['systems'][system]['binding'],
        'EXACT_OLD_MODEL_HAMILTONIAN_STATE_WITH_NEW_CONTAINER_SHA')
    rows=cfg['sealed_files'];indexed={r['path']:r for r in rows}
    require(len(rows)==len(indexed) and digest(rows)==cfg['plan']['closure_digest'],'MIGRATED_CLOSURE')
    for r in rows:
        p=Path(r['path']);require(str(p.resolve())==r['path'] and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'SEALED_SHA:'+r['role'])
    for rel,blob in original.source_blobs(cfg['SOURCE']).items():require((ROOT/rel).read_bytes()==blob,'COMMITTED_ORIGINAL_NUMERICS:'+rel)
    for rel in NEW_PATHS:
        require((ROOT/rel).read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'COMMITTED_MIGRATION_CONTRACT:'+rel)
    require(read(cfg['registry_path'])==law.registry(cfg['tasks'],cfg['SOURCE'],cfg['campaign_id'],'CONFIRMATION',read(cfg['excluded_seeds_path'])['values']),'FRESH_REGISTERED_NAMESPACE')
    require(cfg['complete_historical_seed_noncollision_guaranteed'] is False,'NO_INVENTED_HISTORICAL_SEED_COVERAGE')
    root=Path(cfg['run_root']);require(root==root.resolve() and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_DISJOINT_RUN_ROOT')
    require(not any(Path(p).is_relative_to(root) for p in indexed),'READ_INPUT_WRITE_OUTPUT_DISJOINT')
    for k,n in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:require(cfg[k]==str(root/n),'OWN_STAGE_PATH:'+k)
    require(cfg['storage']==storage.storage_policy() and cfg['compiler_settings_hash']==law.COMPILER_HASH and
        cfg['compile_tolerance']==COMPILE_TOLERANCE and cfg['other_tolerance']==OTHER_TOLERANCE,'UNCHANGED_COMPILER_STORAGE_THRESHOLDS')
    require(cfg['runtime']['qiskit']=='1.3.0' and cfg['runtime']['numpy']=='1.26.4' and cfg['runtime']['provider_root']==cfg['provider_root'],'FIXED_MIGRATION_RUNTIME')
    require(cfg['resources']['runtime_caps'] is False and cfg['resources']['host_PSI_kill'] is False and
        cfg['resources']['process_start_method']=='fork','RESOURCE_POLICY')


def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_RECONSTRUCTED_CONFIRMATION_APPROVAL_REQUIRED')
    require(cfg.get('stage') in STAGES and bool(g.get('grant_id')) and g.get('retry') is False and g.get('next_phase_authorized') is False,'ONE_SHOT_PRIMARY_ONLY')
    require(all(g.get(k)==cfg.get(k) for k in ('SOURCE','stage','cohort','plan_sha256','call_scope','analysis_scope','output','ledger','marker','supervisor')),'EXACT_MIGRATED_GRANT')
    approval=read(cfg['approval_receipt_path'])
    require(approval['execution_authorized'] and approval['migration_and_new_namespace_explicitly_authorized'] and
        all(approval[k]==cfg[k] for k in ('SOURCE','stage','tasks','call_scope','analysis_scope')) and
        sha(cfg['approval_receipt_path'])==cfg['approval_receipt_sha256'],'EXPLICIT_RECEIPT_BINDING')
    require(g['approval_reference']==cfg['approval_receipt_path']+'#sha256='+sha(cfg['approval_receipt_path']),'GRANT_APPROVAL_REFERENCE')
    activation=read(cfg['approval_activation_path'])
    require(all(activation[k]==cfg[k] for k in ('SOURCE','stage','run_root')) and activation['approval_sha256']==sha(cfg['approval_receipt_path']),'ONE_TIME_ISSUANCE')
    verify(cfg)


def native_contract(cfg):
    view=SimpleNamespace(**{k:globals()[k] for k in ('ROOT','sha','read','save','require','digest','authorize','verify','consumed','reserve','task_limits','storage','manifest','COMPILE_TOLERANCE','OTHER_TOLERANCE')})
    view.SIGNAL_RECORDS=cfg['signal_path']
    view.design=SimpleNamespace(COHORT='CONFIRMATION',event_stratum=law.event_stratum,
        COMPILER_HASH=law.COMPILER_HASH,NATIVE_POLICY=law.NATIVE_POLICY)
    return view


def runtime(cfg):
    global RT
    import h246_confirmation_runtime as old
    # Preserve the label-transformer helpers themselves. Transforming their own
    # comparison literal would stop the subsequent DESIGN_PILOT -> CONFIRMATION
    # replacement in the native worker.
    if RT is None:RT,_=old.rebind(old,dict(c=sys.modules[__name__],
        rebind=old.rebind,cohort_code=old.cohort_code))
    r=RT.get_runtime(cfg)
    cli=str(ROOT/'scripts/run_h246_reconstructed.py')
    r.runner_namespace['__file__']=cli
    r.runner.supervise.__globals__['__file__']=cli
    return r


def audit(cfg):
    runtime(cfg)
    return RT.audit(cfg)


def evaluate(cfg):
    import h246_confirmation_evaluation as old
    import h246_confirmation_runtime as helper
    new,_=helper.rebind(old,dict(c=sys.modules[__name__]))
    return new.evaluate(cfg)
