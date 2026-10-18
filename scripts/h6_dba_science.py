"""Independent H6 B/A execution stages; D never imports this module."""
from copy import deepcopy
import gc
import math
from pathlib import Path
import statistics
import h6_dba_contract as c
import h6_dba_native as native
import h8_key_compile_parallel as pool
import h6_dba_sampling as sampling


def checked_results(receipt):
    c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'],
              'OWNED_WORKERS_NATIVE_CLEANUP')
    rows=[]
    for h in receipt['workers']:
        path=Path(h['directory'])/'task_result.json'
        c.require(h['exit_code']==0 and h['reaped'] and c.sha(path)==h['result_sha256'],'NATIVE_RESULT_IDENTITY')
        rows.append(c.read(path))
    return rows


def event_oracle(steps, m, native_module, qstate):
    from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import explicit_event_action
    prep=steps[0].preparation;v=qstate.copy()
    actions=[native_module.make_native_block_action(b,max_instructions=1000000) for b in prep.deterministic_blocks]
    for step in steps:
        for action in actions:v=action(v,step.step_time/2)
        for event in step.rte_occurrence.events:
            v=explicit_event_action(v,event,prep.rte_preparation.basis_registry)
        for action in reversed(actions):v=action(v,step.step_time/2)
    v *= m.np.exp(-1j*(prep.constant_coefficient+prep.extracted_identity_coefficient)*.8)
    return complex(m.np.vdot(qstate,v))


def rare(config, grant, output):
    m,n,ham,sector,state,qstate,preps,compiler=native.load(config)
    prepared=preps['faithful:p5']
    c.require(prepared.deterministic_fragment_indices==tuple(range(5)),'SAVED_PR_PREFIX')
    groups=c.plan()['B']['groups'];cpus=grant['worker_cpus']
    # Avoid accidental reuse of the origin exploration/confirmation and pilot masters.
    from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed as old_seed
    from trottertracks.resource_applicability.h6_matched_contract_v1 import cells as old_cells
    occupied={old_seed(x['id'],phase,i) for x in old_cells() if x['method']=='B2'
              for phase,size in [('exploration',8),('confirmation',32)] for i in range(size)}
    occupied.update(sampling.legacy_pilot_seed(i) for i in range(8))
    tasks=[]
    for group in groups:
        for i in range(8):
            seed=c.seed('B',group['cell']['id'],group['stratum'],i,config['binding']['input'])
            c.require(seed not in occupied,'OLD_MASTER_SEED_COLLISION');occupied.add(seed)
            tasks.append(dict(id=group['id']+'_'+str(i),group=group,replica=i,seed=seed))
    c.save(output/'seed_registry.json',dict(tasks=tasks,
        collision_audit='all origin PR exploration/confirmation master seeds plus conditional pilot masters',
        candidate_pairing_assumed=False))
    def callback(task,directory):
        writer=c.Writer(directory);cell=task['group']['cell']
        steps,receipt=sampling.steps(prepared,cell,task['group']['stratum'],task['seed'])
        writer.write('trajectory.json',receipt)
        evolution=n.partial_native_from_step_requests(steps,control_policy='symmetric_directional',max_instructions=1000000)
        if task['replica']==0:
            z=native.control(evolution,m,qstate,writer)
            independent=event_oracle(steps,m,n,qstate)
            discrepancy=abs(z-independent)
            m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=discrepancy)
            c.require(math.isfinite(discrepancy) and discrepancy<1e-8,'ALGEBRAIC_EVENT_ORACLE_MISMATCH')
            writer.write('event_oracle.json',dict(status='PASS',absolute_discrepancy=discrepancy,
                independent_of_event_circuit_lowering=True,independent_of_orbital_decomposition=False))
        rows=native.compile_pair(evolution,m,compiler,config,writer)
        c.require(c.digest([e.to_dict() for s in steps for e in s.rte_occurrence.events])==receipt['event_digest'],
                  'PAIRED_EVENT_SEQUENCE_CHANGED')
        return dict(task_id=task['id'],group=task['group']['id'],replica=task['replica'],seed=task['seed'],
                    event_digest=receipt['event_digest'],n2=receipt['n2'],n4=receipt['n4'],rows=rows)
    def batch(selected,name):
        return pool.map_owned(selected,callback,cpus=cpus,limits=dict(max_workers=len(cpus)),
            directory=output/name,deadline=math.inf,font_seed=config['font_seed'])
    first=batch([t for t in tasks if t['replica']==0],'first_control_workers')
    rows=checked_results(first)
    c.save(output/'first_control_gate.json',dict(status='PASS',groups=2,all_native_reaped=True))
    rest=batch([t for t in tasks if t['replica']!=0],'remaining_workers')
    rows+=checked_results(rest)
    summaries=[]
    for group in groups:
        selected=[r for r in rows if r['group']==group['id']]
        c.require(len(selected)==8 and {r['replica'] for r in selected}==set(range(8)),'N8_COMPLETENESS')
        for axis in ('cosine','sine'):
            v=[next(a['RZ'] for a in r['rows'] if a['axis']==axis) for r in selected]
            summaries.append(dict(group=group['id'],axis=axis,n=8,mean=statistics.mean(v),
                sample_SE=statistics.stdev(v)/math.sqrt(8),min=min(v),max=max(v),
                population_mean_certified=False))
    c.save(output/'worker_receipts.json',dict(first=first,remaining=rest))
    return dict(stage='B',status='COMPLETE_MANDATORY_STOP',groups=groups,trajectories=rows,summaries=summaries,
        actual_calls=dict(native_compile=32,control_probe=8,trajectory=16,occurrence=128,RTE_event=256),
        K4_s2_multi='UNRESOLVED_NOT_MEASURED',unknown_cost_zero_filled=False,
        cost_reaccounting=False,population_mean_certified=False,new_signal=False)


def execute(stage,config,grant,output):
    if stage=='B':return rare(config,grant,output)
    from h6_dba_signal import signal
    return signal(config,grant,output)
