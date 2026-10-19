"""Stdlib-only one-shot signal contract. Sampling and compile are outside scope."""
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import h46_m5_tol_contract as old
from h6_m5_phase_repair_contract import sha,read,save,seal,digest,require

ROOT=old.ROOT
DOMAIN='references/h46_m5_signal_domain_v1.json'
SCOPE_ID='H46_M5_COMMON_Q_SIGNAL_ONLY_V1'
BASE='7415e79edf365fcfb60750f1872722e4238a1950'
PILOT_INVENTORY='artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/private_raw_inventory.json'
ALLOWANCE=dict(id='M5_SIGNAL_EMPIRICAL_OBSERVATIONS_V1_PROPOSED',status='PROPOSED_REQUIRES_EXECUTION_ACCEPTANCE',
    floor=1e-9,safety_factor=10.,roundoff_multiplier=64.,log_B_margin_floor=1e-12,
    maximum_u_over_headroom=.01,certified=False,reference_absolute_error=None,
    reference_assumption='CONDITIONAL_ON_SAVED_PILOT_REFERENCE_ACCURACY;NOT_A_ZERO_ERROR_BOUND',
    compile_tolerance_used_for_shots=False)
NEW_PATHS=(DOMAIN,'scripts/h46_m5_signal_contract.py','scripts/h46_m5_signal_adapter.py',
 'scripts/run_h46_m5_signal.py','scripts/prepare_h46_m5_signal.py','tests/test_h46_m5_signal.py',
 'artifacts/resource_applicability/track_a_h46_m5_minimal_comparison_plan/2026-10-19/candidate_domain_DRAFT.json',PILOT_INVENTORY)
SOURCE_PATHS=tuple(dict.fromkeys(old.SOURCE_PATHS+NEW_PATHS+('scripts/run_h6_m5_saved_phase.py',
 'scripts/h6_m5_phase_repair_contract.py')))
EPSILONS=(.05,.01,.005,.001,.0005)


def rows(system=None):
    result=read(ROOT/DOMAIN)['rows']
    return [r for r in result if system is None or r['system']==system]


def scope():
    return dict(scope_id=SCOPE_ID,domain_sha256=sha(ROOT/DOMAIN),candidate_ids=[r['id'] for r in rows()],
      systems=['H4','H6'],input_policy='SAVED_TOL_ONLY_1e-8',T=.8,P=0,control='symmetric_directional',
      native_policy=old.POLICY,formula_id='PF_M5_BEST_20D64C2_BINARY64_V1',PR_policy='ABS_TIME_PER_OUTER',PR_R=32,K=4,
      epsilons=list(EPSILONS),alpha_axis=.025,allowance_factor=1,allowance_policy=ALLOWANCE,
      signal_targets=88,maximum_new_candidate_signal=82,reused_candidate_signal=6,
      saved_snapshot_restores=2,saved_reference_reads=2,new_full_H_reference=0,
      new_sampling=0,new_compile=0,new_circuit_state_action=0,SCF_DF_state_generation=0,
      shot_counts_computed=False,cost_accounting=False,ranking=False,fit=False,optional20=False,
      retry=False,next_phase_authorized=False,AD03='DEFERRED',mandatory_STOP=True)


def verify_original_pilot(config_path):
    """Run the original sealed contract at its original root, in isolation.

    Its absolute source paths remain mandatory. A new signal worktree is never
    substituted into that historical seal. This subprocess imports stdlib only.
    """
    origin=read(config_path)
    paths=[Path(e['path']) for e in origin['sealed_files'] if e['path'].endswith('/scripts/h46_m5_tol_contract.py')]
    require(len(paths)==1,'UNIQUE_ORIGINAL_PILOT_CONTRACT')
    path=paths[0];root=path.parents[1]
    require(path.read_bytes()==subprocess.check_output(['git','show',origin['SOURCE']+':scripts/h46_m5_tol_contract.py'],cwd=root),'ORIGINAL_PILOT_CONTRACT_GIT_BYTES')
    code='import sys;sys.path.insert(0,sys.argv[1]);import h46_m5_tol_contract as c;c.verify_seals(c.read(sys.argv[2]));c.consumed(c.read(sys.argv[2]))'
    result=subprocess.run([sys.executable,'-B','-c',code,str(root/'scripts'),str(Path(config_path).resolve())],capture_output=True,text=True)
    require(result.returncode==0,'ORIGINAL_PILOT_SEALS:'+result.stderr[-1500:])
    return origin


def verify(cfg):
    require(cfg['plan']['scope']==scope() and cfg['plan']['SOURCE']==cfg['SOURCE'],'EXACT_SIGNAL_SCOPE')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'SIGNAL_PLAN_BYTES')
    files=cfg['sealed_files'];indexed={r['path']:r for r in files}
    require(len(files)==len(indexed) and digest(files)==cfg['plan']['closure_digest'],'SIGNAL_CLOSURE')
    for r in files:
        p=Path(r['path']);require(str(p.resolve())==r['path'] and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'SEALED_BYTES:'+r['role'])
    for rel in SOURCE_PATHS:
        path=ROOT/rel
        require(str(path.resolve()) in indexed and path.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'SIGNAL_GIT_SOURCE:'+rel)
    old_domain=read(ROOT/'artifacts/resource_applicability/track_a_h46_m5_minimal_comparison_plan/2026-10-19/candidate_domain_DRAFT.json')
    require(rows()==[r for r in old_domain['rows'] if r['role']=='PRIMARY'],'ADOPTED_88_ONLY')
    require(digest(cfg['systems'])==cfg['plan']['systems_digest'] and digest(cfg['runtime'])==cfg['plan']['runtime_digest'],'SIGNAL_INPUT_RUNTIME_BINDINGS')
    require(cfg['runtime']['provider_root']==cfg['provider_root'],'SIGNAL_PROVIDER')
    require(set(cfg['systems'])=={'H4','H6'},'TWO_SIGNAL_SYSTEMS')
    reused=[]
    original_inventory={r['path']:r for r in read(ROOT/PILOT_INVENTORY)}
    for system,info in cfg['systems'].items():
        origin=verify_original_pilot(info['origin_config'])
        require(origin['SOURCE']=='c7d53f04c3be131078ee91e8bef3c78ca434ed3e' and origin['system']==system and origin['lane']==system+'_TOL_PILOT_DET','ORIGINAL_PILOT_LINEAGE')
        require(info['binding']==origin['binding'] and info['inputs']==origin['inputs'] and info['reference_root']==origin['reference_root'],'SAME_SAVED_INPUTS_REFERENCE')
        require(cfg['runtime']==origin['runtime'] and cfg['provider_root']==origin['provider_root'],'SAME_PILOT_RUNTIME')
        used=[info['origin_config']]+[v['path'] for v in info['inputs'].values()]
        used.extend(str(Path(info['reference_root'])/n) for n in ('reference.npz','receipt.json'))
        ref=read(Path(info['reference_root'])/'receipt.json')
        require(ref['SOURCE']==origin['SOURCE'] and ref['status']=='PASS' and ref['reference_sha256']==sha(Path(info['reference_root'])/'reference.npz') and all(ref[k+'_hash']==info['binding'][k] for k in ('hamiltonian','state')),'SAVED_REFERENCE_IDENTITY')
        for name in ('reference.npz','receipt.json'):
            path=Path(info['reference_root'])/name;original=original_inventory[str(path.relative_to(origin['run_root']))]
            require(sha(path)==original['sha256'] and path.stat().st_size==original['bytes'],'ORIGINAL_REFERENCE_INVENTORY')
        for candidate_id,record in info['reuse'].items():
            group=next(r for r in rows(system) if r['id']==candidate_id)
            matches=[t for t in origin['plan']['scope']['tasks'] if all(t['group'][k]==group[k] for k in ('method','formula','prefix','q'))]
            require(len(matches)==1 and group['kind']=='DET_SIGNAL' and group['q']==2,'EXACT_DETERMINISTIC_REUSE')
            task=matches[0];path=Path(origin['output'])/task['id']/'signal.json'
            require(record==seal(path,'saved_signal:'+candidate_id),'SAVED_SIGNAL_SHA_BINDING')
            original=original_inventory[str(path.relative_to(origin['run_root']))]
            require(record['sha256']==original['sha256'] and record['bytes']==original['bytes'],'ORIGINAL_SIGNAL_INVENTORY')
            used.append(str(path));reused.append(candidate_id)
        require(all(str(Path(p).resolve()) in indexed for p in used),'SIGNAL_USED_INPUTS_SEALED')
    require(len(reused)==len(set(reused))==6,'EXACT_SIX_REUSED_SIGNALS')
    require(cfg['resources']==cfg['plan']['resources'] and cfg['resources']['host_PSI_kill'] is False and cfg['resources']['runtime_caps'] is False,'SIGNAL_RESOURCE_POLICY')
    require(cfg['run_root']==cfg['plan']['run_root'],'SIGNAL_ROOT_BINDING')
    require(cfg['pipeline_identity']==cfg['plan']['pipeline_identity']==digest(dict(SOURCE=cfg['SOURCE'],scope=scope(),runtime=cfg['runtime'],systems_digest=digest(cfg['systems']))),'SIGNAL_PIPELINE_IDENTITY')


def authorize(cfg):
    grant=cfg.get('grant',{})
    require(cfg.get('authorized') is True and grant.get('valid') is True,'EXPLICIT_SIGNAL_EXECUTION_APPROVAL_REQUIRED')
    expected=dict(SOURCE=cfg['SOURCE'],scope_id=SCOPE_ID,plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],maximum_new_signal=82,maximum_compile=0,maximum_sampling=0,retry=False,next_phase_authorized=False)
    require(all(grant.get(k)==v for k,v in expected.items()) and grant.get('grant_id'),'EXACT_SIGNAL_ONLY_GRANT')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_SIGNAL_ROOT')
    for k,n in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:require(cfg[k]==grant.get(k)==str(root/n),'SIGNAL_OUTPUT_PATH:'+k)
    paths={k:str(root.parent/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==grant.get('supervisor')==paths,'SIGNAL_SUPERVISOR')
    p=cfg['approval_receipt_path'];require(p in {r['path'] for r in cfg['sealed_files']} and sha(p)==cfg['approval_receipt_sha256']==cfg['plan']['approval_receipt_sha256'],'NEW_SIGNAL_APPROVAL_BOUND')
    a=read(p);require(a['execution_authorized'] is True and a['SOURCE']==cfg['SOURCE'] and a['scope']==scope() and a['allowance_policy_accepted_for_this_run'] is True,'EXPLICIT_SIGNAL_AND_ALLOWANCE_APPROVAL')
    require(grant.get('approval_reference')==p+'#sha256='+sha(p),'SIGNAL_APPROVAL_REFERENCE')
    require(not any(Path(r['path']).is_relative_to(root) for r in cfg['sealed_files']),'NO_INPUT_OUTPUT_OVERLAP')
    verify(cfg)


def consumed(cfg):require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),'SIGNAL_CONSUMED_MARKER')


def reserve(cfg,row):
    consumed(cfg);require(row in rows(),'REGISTERED_SIGNAL_CANDIDATE')
    save(Path(cfg['ledger'])/(row['id']+'.json'),dict(candidate=row['id'],SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],state='ENTERED_NONREFUNDABLE',new_signal=row['id'] not in cfg['systems'][row['system']]['reuse']))


def eligibility(record,group):
    """Proposed empirical allowance diagnostic; no shots/costs/ranking."""
    e=record['allowance_evidence'];maximum=e['maximum_intermediate_norm'];closure=e['raw_corrected_vector_closure'];logB=record['log_B']
    require(all(math.isfinite(x) and x>=0 for x in (maximum,closure)) and math.isfinite(logB) and logB>=0,'FINITE_SIGNAL_ALLOWANCE_INPUT')
    stages=group['q']*(2*(group['prefix']+1)+(1 if group['method']=='PR' else 0))*(1 if group['formula']=='2nd' else 3 if group['formula']=='4th' else 11)
    roundoff=ALLOWANCE['roundoff_multiplier']*2**-52*(1+maximum)*(1+stages)
    u=max(ALLOWANCE['floor'],ALLOWANCE['safety_factor']*closure+roundoff)
    margin=max(ALLOWANCE['log_B_margin_floor'],ALLOWANCE['roundoff_multiplier']*2**-52*(1+abs(logB))*(1+stages))
    output=[]
    for epsilon in EPSILONS:
        for axis in ('real','imag'):
            bias=record['total_signed_bias'][axis];require(math.isfinite(bias),'FINITE_SIGNED_BIAS')
            headroom=epsilon/math.sqrt(2)-abs(bias)-u
            status='INELIGIBLE' if headroom<=0 else 'UNDETERMINED' if u/headroom>ALLOWANCE['maximum_u_over_headroom'] else 'ELIGIBLE'
            output.append(dict(epsilon=epsilon,axis=axis,status_under_proposed_empirical_policy=status,bias=bias,u=u,headroom=headroom,log_B=logB,log_B_margin=margin,
              conditional_on_saved_reference=True,reference_absolute_error=None,strict_bias_certificate=None,shots=None,cost=None))
    return dict(policy=ALLOWANCE,u=u,roundoff_observation=roundoff,raw_closure_observation=closure,log_B_margin=margin,rows=output)
