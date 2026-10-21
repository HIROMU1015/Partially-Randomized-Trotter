"""AX-2B proposed task expansion and fail-closed metadata, stdlib only.

No launcher, no scientific imports, no grant of execution permission.
Technical readiness and scientific authorization are separate fields.
"""
import copy
import hashlib
import json
from pathlib import Path

from .ax2a_preparation import pilot_draft, digest


REQUIRED_IDENTITIES = (
    'H4_snapshot_sha256', 'H4_state_identity', 'H6_snapshot_sha256',
    'H6_state_identity', 'adopted_df_policy', 'primitive_sector_certificate',
    'basis_order_certificate', 'reference_numerical_allowance',
    'compiler_identity', 'assigned_resources', 'scientific_source_freeze',
)


def seed_for(cell_id, replica):
    if type(replica) is not int or replica < 0 or not isinstance(cell_id,str) or not cell_id:
        raise ValueError('Cell ID and nonnegative integer replica required.')
    key=f'track_a_ax2b_v2_cost_seed:{cell_id}:{replica}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4],'big')


def expanded_pilot_proposal():
    original=pilot_draft()
    result=copy.deepcopy(original)
    result['schema_version']='track_a_ax2b_technical_pilot_draft_v2'
    result['status']='DRAFT_NOT_AUTHORIZATION'
    result['science_authorized']=False
    result['ax2b_authorized']=False
    # H4 phase/sector/numerical tasks are distinct from compiled-cost tasks.
    result['H4_correctness_cells']=[
        {'id':'H4_B1_S2_q1','method':'B1','order':'2nd','prefix':12,'q':1,'R':None,'K':None},
        {'id':'H4_B1_S2_q4','method':'B1','order':'2nd','prefix':12,'q':4,'R':None,'K':None},
        {'id':'H4_B0_q4','method':'B0','order':'2nd','prefix':6,'q':4,'R':None,'K':None},
        {'id':'H4_B2_K2','method':'B2','order':'2nd','prefix':6,'q':4,'R':8,'K':2},
        {'id':'H4_B2_K4','method':'B2','order':'2nd','prefix':6,'q':4,'R':8,'K':4},
        {'id':'H4_B3_K6','method':'B3','order':'2nd','prefix':0,'q':4,'R':8,'K':6},
        {'id':'H4_B1_S4_q1','method':'B1','order':'4th','prefix':12,'q':1,'R':None,'K':None},
        {'id':'H4_B1_S4_q4','method':'B1','order':'4th','prefix':12,'q':4,'R':None,'K':None},
    ]
    h4_cost_ids={'H4_B0_q4','H4_B2_K2','H4_B3_K6','H4_B1_S4_q1','H4_B1_S4_q4'}
    cells=[dict(c,system='H4') for c in result['H4_correctness_cells'] if c['id'] in h4_cost_ids]
    cells += [dict(c,system='H6') for c in result['H6_tasks']]
    wrapper_tasks=[]
    seed_ids={}
    for cell in cells:
        replicas=2 if cell['method'] in ('B2','B3') else 1
        for replica in range(replicas):
            seed=seed_for(cell['id'],replica) if replicas==2 else None
            if seed is not None:
                if seed in seed_ids:raise RuntimeError('SEED_COLLISION')
                seed_ids[seed]=(cell['id'],replica)
            for policy in ('ordinary','symmetric_directional'):
                for axis in ('cosine','sine'):
                    wrapper_tasks.append({
                        'id':f"{cell['id']}_rep{replica}_{policy}_{axis}",
                        'cell':cell,'replica':replica,'trajectory_seed':seed,
                        'control_policy':policy,'axis':axis,
                        'scope':'measured full wrapper without state preparation',
                        'draw_reuse':'same explicit trajectory across axes and control policies',
                    })
    caps=result['proposed_caps']
    caps.update(max_untranspiled_instructions_per_wrapper=1000000,
                max_transpiled_instructions_per_wrapper=5000000,
                max_deterministic_actions_per_signal=100000,
                max_ground_solver_matvecs_per_system=20000,
                max_reference_matvecs_per_action=20000,
                max_H4_qubits=8,max_H6_qubits=12,max_outer_steps=8)
    counts={s:sum(t['cell']['system']==s for t in wrapper_tasks) for s in ('H4','H6')}
    if len(wrapper_tasks)>caps['total_wrappers']:
        raise RuntimeError('PROPOSED_COMPILE_BUDGET_EXCEEDED')
    result['wrapper_tasks']=wrapper_tasks
    result['planned_wrapper_calls']=len(wrapper_tasks)
    result['planned_wrapper_calls_by_system']=counts
    result['unique_random_trajectory_count']=len(seed_ids)
    result['proposed_df_tolerance']=1e-8
    result['df_tolerance_adopted']=False
    result['actual_df_rank']=None
    result['compiler_proposal']={'basis_gates':['rz','sx','x','cx'],'optimization_level':1,
                                 'seed_transpiler':17,'coupling_map':None,
                                 'installed_version_must_be_recorded':True}
    result['proposed_state_policy']={'H4':'same legacy saved normalized state',
                                     'H6':'normalized DF state with recorded energy/residual and ground-state status'}
    result['control_correctness_scope']='native lowering synthetic-validated; molecular verification pending'
    result['instruction_cap_scope']='Qiskit instruction counts; primitive depth/gates and memory require watchdog/profile'
    result['reference_cap_scope']='separate hard caps for solver/reference actions; tolerance is not an allowance certificate'
    result['task_binding_requirements']=list(REQUIRED_IDENTITIES)
    result['scientific_runner_implemented']=False
    return result


def preflight_report(bindings=None):
    bindings={} if bindings is None else dict(bindings)
    proposal=expanded_pilot_proposal()
    missing=[name for name in REQUIRED_IDENTITIES if not bindings.get(name)]
    # Accepting a nonempty field records presence only; this is not a
    # scientific certificate validator or an authorization parser.
    return {'schema_version':'track_a_ax2b_preflight_report_v1',
            'status':'PREPARATION_ONLY_LAUNCH_FORBIDDEN',
            'science_authorized':False,'ax2b_authorized':False,
            'launch_allowed':False,'mandatory_stop':True,
            'identity_presence_complete':not missing,'identity_missing':missing,
            'certificate_semantics_verified':False,
            'additional_launch_blockers':['scientific runner/cap enforcement implementation',
                                          'molecular correctness and allowance validation',
                                          'task/source/authorization freeze',
                                          'separate explicit scientific execution authorization'],
            'proposal':proposal,'proposal_digest':digest(proposal)}


def preparation_bundle(root):
    root=Path(root)
    paths=(
        'src/trottertracks/resource_applicability/ax2a_native_df.py',
        'src/trottertracks/resource_applicability/ax2b_preflight.py',
        'src/trottertracks/resource_applicability/ax2a_control_plan.py',
        'src/trottertracks/resource_applicability/ax2a_state_action.py',
        'src/trotterlib/df_partial_s2.py','src/trotterlib/df_trotter/ops.py',
        'src/trotterlib/df_trotter/circuit.py','src/trotterlib/df_rte_qiskit.py',
        'src/trotterlib/rpe_hadamard_interrogation.py','src/trotterlib/rte_compiled_cost.py',
    )
    report=preflight_report()
    report['source_hashes']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    report['new_scientific_calculation_count']=0
    return report
