"""Approval-only H2 P adapter over the completed H2 native compilation kernel."""
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

import h2_p_contract as c


def compile_one(port,m,settings,task,directory):
    import h2_panels_science as inherited
    proxy=SimpleNamespace(require=c.require,save=c.save,digest=c.digest,Writer=c.base.Writer,
        stats=c.base.stats,check_pair=c.base.check_pair,BINDING=c.base.BINDING,plan=c.plan)
    function=FunctionType(inherited.compile_one.__code__,dict(inherited.__dict__,c=proxy))
    result=function(port,m,settings,task,directory,'P')
    c.require(result['seed']==task['seed'] and result['preparation_hash']==task['preparation_hash'],
        'FIXED_P_TASK_PREPARATION_SEED')
    c.save(Path(directory)/'confirmation_seed_identity.json',dict(master_seed=task['seed'],
        step_seeds=task['step_seeds'],cell_id=task['cell']['id'],replica=task['replica'],
        seed_plan_digest=task['seed_plan_digest'],cohort=c.COHORT,redraw=False))
    return result


def context(spec,output,calls,loader=None):
    config=spec['native_config'];sys.path.insert(0,str(Path(config['provider_root'])/'src'))
    sys.path.insert(0,spec['A_kernel_scripts'])
    import h4_native_science as h
    from h2_tol_only_input import load
    from h4_dp_science import compiler
    m=h.provider();calls.take('saved_input_load')
    ham,sector,full,state,md=(loader or load)(config)
    c.require((ham.n_qubits,len(ham.lambdas),sector.dimension)==(4,3,4)
        and md['hamiltonian_hash']==c.base.BINDING['hamiltonian']
        and md['state_hash']==c.base.BINDING['state'],'P_SAVED_H2_IDENTITY')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    calls.take('preparation');prep=m._prepare(ham,1)
    c.require(prep.hamiltonian_hash==c.base.BINDING['hamiltonian']
        and prep.exact_rte_lambda_r>0 and prep.deterministic_fragment_indices==(0,)
        and all(ep['preparation_hash']==prep.preparation_hash for ep in spec['endpoints']),
        'P_ORIGINAL_PREFIX1_PREPARATION')
    c.save(output/'preparation_receipt.json',dict(preparation_hash=prep.preparation_hash,
        hamiltonian_hash=prep.hamiltonian_hash,prefix=1,saved_input_load=1,preparation=1,
        SCF_DF_state_solver_signal_reference=0))
    return SimpleNamespace(ham=ham,preps={x['id']:prep for x in c.cells()},qstate=qstate),m,compiler(m,config)


def execute(spec,grant,output):
    c.authorize(spec,grant)
    from h2_panels_science import batch
    calls=c.base.Calls(output,spec['calls']);port,m,settings=context(spec,output,calls)
    registry=c.read(spec['seed_registry']);endpoints={e['cell_id']:e for e in spec['endpoints']}
    tasks=[dict(t,expected=endpoints[t['cell']['id']]['signal'],
        preparation_hash=endpoints[t['cell']['id']]['preparation_hash'],seed_plan_digest=spec['seed_plan_digest'])
        for t in registry['tasks']]
    for key in ('trajectory','native_evolution','wrapper_compile','occurrence','RTE_event',
                'control_probe','independent_event_oracle'):calls.take(key,c.CALLS[key])
    callback=lambda task,directory:compile_one(port,m,settings,task,directory)
    first=[t for t in tasks if t['replica']==0];rest=[t for t in tasks if t['replica']!=0]
    measured=batch(spec,first,callback,output/'first_control_workers')
    measured+=batch(spec,rest,callback,output/'remaining_workers')
    c.check_measurements(measured,registry,spec['endpoints'])
    calls.take('endpoint_accounting_pass');accounting=c.account(measured,registry,spec['endpoints'])
    c.require(calls.used==c.CALLS,'EXACT_P_NONREFUNDABLE_CALLS')
    return dict(scope=c.SCOPE,stage='P',status='COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],
        binding=c.base.BINDING,spec_digest=c.digest(spec),seed_plan_digest=spec['seed_plan_digest'],
        measurements=measured,accounting=accounting,calls_reserved=calls.used,calls_maximum=c.CALLS,
        saved_signals_shots_PF_unchanged=True,old_cohorts_pooled=False,additional_sampling=0,
        automatic_retry=False,next_stage_authorized=False,population_mean_certified=False)
