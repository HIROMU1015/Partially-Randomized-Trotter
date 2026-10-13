"""Two explicitly requested scopes: fixed-input H4 panel, H8 input preflight.

No imports of numerical packages during plan preparation. No full H8 grant.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import SNAPSHOT, SNAPSHOT_SHA, file_hash, snapshot_binding
from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_h6_input_generation_contract_v1 import plan as old_input_plan
from .h6_matched_contract_v1 import plan as old_signal_plan

RUNNER='scripts/resource_applicability/run_track_a_h48_preflight_v1.py'
PREPARER='scripts/resource_applicability/prepare_track_a_h48_preflight_v1.py'
PREP='artifacts/resource_applicability/track_a_h48_preflight_preparation_v1/2026-10-13/'
CPUS=[20,21,24,25,26,27,28,29]
PACKAGES=('numpy','scipy','numba','qiskit','qiskit-nature','qiskit-algorithms',
          'openfermion','openfermionpyscf','pyscf','h5py')

def h4_cells():
    rows=[]
    for q in (1,2,4,8):
        for formula in ('2nd','4th'):
            rows.append(dict(id=f'B1_p12_q{q}_{formula}',method='B1',prefix=12,q=q,
                             formula=formula,r=None,R=None,K=None))
        r=2 if q==8 else 4
        for K in (2,4):
            rows.append(dict(id=f'B2_p3_q{q}_2nd_r{r}_K{K}',method='B2',prefix=3,q=q,
                             formula='2nd',r=r,R=q*r,K=K))
    return rows

def h4_plan(scope='H4_COST'):
    p=deepcopy(old_signal_plan());p['kind']='H4_COMMON_ANCHOR_PANEL'
    p['target']=dict(model='linear_H4',geometry_angstrom=1.,basis='sto-3g',n_qubits=8,
                     nelec_alpha=2,nelec_beta=2,sector_dimension=36)
    p['actual_rank']=12;p['cells']=h4_cells()
    p['df_policy']={'kind':'EXISTING_FIXED_RANK12_SNAPSHOT_UNCHANGED',
                   'not_asserted_equal_to_tol_only_policy':True,'snapshot_sha256':SNAPSHOT_SHA}
    p['input_policy']='EXISTING_H4_FIXED_BYTES_NO_REGENERATION'
    p['matrix_free']['num_threads']=1
    p['cost_rule'].update(exploratory_random_trajectories=0,confirmation_random_trajectories=32,
        acquire='fixed anchor panel, all cells eligible at any registered epsilon',
        seed_namespace='H4_COMMON_NATIVE_PANEL_V1_20261013',confirmation_top_per_epsilon=0)
    p['sensitivity_cells']=['B1_p12_q1_4th','B2_p3_q2_2nd_r4_K2']
    p['b3_diagnostic'].update(q=[],r=[],K=[])
    pairs=set();schedules=[]
    for c in p['cells']:
        s=primitive_time_schedule(c,T=.8);s['registered_validation_times_v2']=s['unique_primitive_times']
        schedules.append(dict(cell_id=c['id'],schedule=s))
        pairs.update(('one' if i==0 else str(i-1),t) for i,t in s['unique_primitive_times'])
    p['coverage']=dict(cells=schedules,physical_times=sorted(pairs),primitive_actions=3*len(pairs),primitive_probe_count=3)
    p['caps_proposed'].update(compile=528,trajectory=256,occurrence=960,control_probe=72,
         primitive=3*len(pairs),reference_matvec=36,output_bytes=2*2**30,
         solver_matvec=0,integral_build=0,df_decomposition=0)
    p['maximum_cost_groups']=264;p['maximum_primary_wrappers']=528
    p['maximum_sensitivity_wrappers']=4;p['maximum_wrappers']=528
    # 8 deterministic*2 + 8 random*32*2 =528; ordinary anchors add4 ->532.
    # The manuscript estimate 528 already included 16 deterministic +512 random;
    # sensitivity is an explicitly separate addition, never silently uncharged.
    p['caps_proposed']['compile']=532;p['maximum_primary_wrappers']=528;p['maximum_wrappers']=532
    p['formal_accuracy_certified']=False;p['new_comparison_contract_adopted']=False
    if scope=='H4_SIGNAL':
        p['caps_proposed'].update(compile=0,trajectory=0,occurrence=0,control_probe=0)
    elif scope!='H4_COST':raise ValueError('H4_PHASE_SCOPE')
    return p

def h8_input_plan():
    p=deepcopy(old_input_plan())
    p['schema']='h8_preflight_input_plan_v1'
    p['target']=dict(model='linear_H8',geometry_angstrom=1.,basis='sto-3g',
        geometry=[['H',[0.,0.,float(i)]] for i in range(8)],charge=0,multiplicity=1,
        n_qubits=16,nelec_alpha=4,nelec_beta=4,sector_dimension=4900)
    p['df_policy']['accepted_actual_rank']=[2,256]
    p['df_policy']['projection_scope']='same strict integral-to-DF adapter; no fallback if weighted policy is needed'
    p['caps_proposed'].update(address_space_bytes=32*2**30,output_bytes=32*2**20,
        snapshot_expanded_bytes=16*2**20,solver_matvec=10000,solver_matvec_per_action=10000,
        phase_wall_seconds=None,total_wall_seconds=None)
    p['H8_full_campaign_authorized']=False
    p['scope']='input/DF/saved-state feasibility only; no signal/trajectory/transpile'
    return p

def bounded_h8_solver(action,dimension,initial,total_budget,*,solver):
    # Scientific imports only after the worker's launch binding and limits.
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from .ax2b_h6_controller import bounded_matvec
    if dimension!=4900:raise ValueError('H8_SOLVER_DIMENSION')
    v=np.asarray(initial,dtype=np.complex128)
    if v.shape!=(4900,) or not np.isfinite(v).all() or abs(np.linalg.norm(v)-1)>1e-12:
        raise ValueError('H8_INITIAL_STATE')
    apply,counter=bounded_matvec(action,total_budget,name='solver_matvec',per_action_cap=10000)
    op=LinearOperator((dimension,dimension),matvec=apply,rmatvec=apply,dtype=np.complex128)
    return solver(op,k=1,which='SA',tol=1e-12,maxiter=1000,ncv=40,v0=v.copy()),op,counter

def verify_h8_snapshot_roundtrip(path,arrays):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(arrays):raise ValueError('H8_SNAPSHOT_KEYS')
        for k,a in arrays.items():
            b=z[k]
            if b.shape!=a.shape or b.dtype!=a.dtype or not np.array_equal(a,b):
                raise ValueError('H8_SNAPSHOT_ROUNDTRIP:'+k)

def source_paths(root):
    root=Path(root)
    return sorted({str(p.relative_to(root)) for directory in ('src/trotterlib','src/trottertracks')
        for p in (root/directory).rglob('*.py')}|{RUNNER,PREPARER})

def profile():
    return {'python':sys.version.split()[0],
        'packages':{n:importlib.metadata.version(n) for n in PACKAGES},
        'executable':sys.executable,'prefix':sys.prefix,
        'compiler_output_bit_equivalence_to_old_host_verified':False}

def verify_candidate_runtime(root):
    """Fresh runtime subset of the adopted profile, once per coordinator.

    The original 34,273-file installation inventory is preserved; there is
    no repeat full environment campaign or false claim of a complete audit.
    """
    p=Path(root)/'artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json'
    d=json.loads(p.read_text())
    if file_hash(p)!='4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75':
        raise ValueError('ADOPTED_PROFILE_CHANGED')
    prefix=str(Path(sys.prefix).resolve())+'/';rows=[]
    for a in d['critical_file_hashes']:
        name=a['path']
        if name.startswith(prefix) and (name.endswith('.so') or '.so.' in name) and not any(x in name.lower() for x in ('cupy','cuda','nvidia')):
            if file_hash(name)!=a['sha256']:raise ValueError('NATIVE_RUNTIME_CHANGED:'+name)
            rows.append(a)
    native=d['native_lowering_dependencies']
    if file_hash(native['source_path'])!=native['source_sha256']:raise ValueError('NATIVE_LOWERING_CHANGED')
    if not rows:raise ValueError('NATIVE_RUNTIME_SUBSET_EMPTY')
    return {'adopted_profile_sha256':file_hash(p),'runtime_native_files_checked':len(rows),
        'subset_digest':digest(rows),'full_34273_inventory_repeated':False,
        'native_lowering_source_sha256':native['source_sha256'],'compiler_output_bit_equivalence_verified':False}

def resources(workers=8):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('WORKER_COUNT')
    cpus=CPUS[:workers]
    return dict(assigned_cpus=sorted(cpus+[22,23]),coordinator_cpu_set=[22,23],
        cost_cpu_sets=[[c] for c in cpus],cost_workers=workers,cost_threads=1,
        cost_address_space_bytes=32*2**30,worker_rss_stop_bytes=9*2**30,
        coordinator_address_space_bytes=8*2**30,coordinator_hard_address_space_bytes=32*2**30,
        aggregate_rss_bytes=76*2**30,job_ram_budget_bytes=96*2**30,
        coordinator_rss_reserve_bytes=2*2**30,aggregate_margin_bytes=2*2**30,
        gpu=False,memory_is_reservation=False)

def observation_policy():
    return dict(interval_seconds=.2,maximum_observation_seconds=5.,journal_emit_seconds=5.,
        journal_bytes=32*2**20,host_full_psi_stop_percent=5.,cgroup_full_psi_stop_percent=0.,
        fresh_host_available_bytes=int(248.25*2**30),fresh_effective_available_bytes=96*2**30,
        running_host_available_bytes=int(168.25*2**30),running_effective_available_bytes=16*2**30)

def aggregate_caps():
    return dict(output_bytes=2*2**30,log_bytes=2*2**20)

def fixed_seed(source,cell,replica):
    return int.from_bytes(hashlib.sha256(('H4_COMMON_NATIVE_PANEL_V1_20261013|'+source+'|'+cell+'|'+str(replica)).encode()).digest()[:8],'big')

def task_caps(task,master):
    caps=deepcopy(master);policies=2 if task['ordinary'] else 1
    random=bool(task['cell']['R'])
    caps.update(compile=2*policies,trajectory=int(random),occurrence=task['cell']['q'] if random else 0,
                control_probe=4*policies if task['validate'] else 0,primitive=0,reference_matvec=0)
    return caps

def registrations(manifest,signals):
    by={x['cell']['id']:x for x in signals};rows=[]
    for c in manifest['plan']['cells']:
        record=by[c['id']]
        if not any(x.get('1.0',{}).get('cost_acquisition_eligible') is True for x in record['precision_rows'].values()):
            continue
        for replica in range(32 if c['R'] else 1):
            rows.append(dict(index=len(rows),task=dict(cell=c,phase='fixed_anchor_panel',replica=replica,
              seed=fixed_seed(manifest['source_commit'],c['id'],replica),
              ordinary=replica==0 and c['id'] in manifest['plan']['sensitivity_cells'],validate=replica==0,
              expected_signal=record['signals']['corrected'],kind='H4_COST')))
    return rows

def verify_binding(root,m,grant,*,worker=False):
    root=Path(root)
    if (grant.get('approved_by_user') is not True or grant.get('manifest_sha256')!=digest(m)
          or grant.get('H8_full_campaign_authorized') is not False):raise ValueError('NEW_SCOPE_GRANT_REQUIRED')
    if grant.get('authorization_basis')!=['準備ができたら開始してよい','開始対象は1と3（H8小規模事前検証・H4比較補完）']:
        raise ValueError('USER_SCOPE_BINDING')
    if m['scope'] not in ('H4_SIGNAL','H4_COST','H8_INPUT'):raise ValueError('SCOPE')
    if m['profile']!=profile():raise ValueError('PROFILE_CHANGED')
    if len(m['source_commit'])!=40 or set(m['source_hashes'])!=set(source_paths(root)):raise ValueError('SOURCE_CLOSURE')
    for p,h in m['source_hashes'].items():
        if file_hash(root/p)!=h or (not worker and hashlib.sha256(subprocess.check_output(['git','show',m['source_commit']+':'+p],cwd=root)).hexdigest()!=h):
            raise ValueError('SOURCE_CHANGED:'+p)
    if m['scope'].startswith('H4') and snapshot_binding(root)!=m['snapshot']:raise ValueError('H4_SNAPSHOT_CHANGED')
    if m['plan']!=(h8_input_plan() if m['scope']=='H8_INPUT' else h4_plan(m['scope'])):raise ValueError('PLAN_CHANGED')
    if m['proposed_resources']!=resources(8 if m['scope']=='H4_COST' else 1):raise ValueError('RESOURCE_BINDING')
    if m['observation_policy']!=observation_policy():raise ValueError('OBSERVATION_BINDING')
    if m['base_manifest']!={'plan':{'caps_proposed':aggregate_caps()}}:raise ValueError('AGGREGATE_CAPS')
    if m['scope']!='H4_COST' and m['registrations']!=[{'index':0,'task':{'kind':m['scope']}}]:
        raise ValueError('SINGLE_PHASE_REGISTRATION')
    if m['scope']=='H4_COST':
        signal_path=Path(m['signal_summary_path'])
        if file_hash(signal_path)!=m['signal_summary_sha256']:raise ValueError('SIGNAL_RECEIPT_CHANGED')
        saved=json.loads(signal_path.read_text())
        if m['registrations']!=registrations(m,saved['signals']):raise ValueError('COST_REGISTRATION_CHANGED')
    if m['output']!=grant['output'] or not Path(m['output']).is_absolute():raise ValueError('OUTPUT_BINDING')
    out=Path(m['output'])
    if not str(out).startswith('/home/AbeHiromu/') or str(out.resolve())!=m['output']:raise ValueError('HOME_OUTPUT_REQUIRED')
    if out.is_symlink():raise ValueError('OUTPUT_SYMLINK')
    if m.get('H8_full_campaign_authorized') is not False:raise ValueError('H8_FULL_CAMPAIGN_NOT_AUTHORIZED')
    return True
