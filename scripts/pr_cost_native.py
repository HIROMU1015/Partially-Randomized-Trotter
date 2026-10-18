"""Lazy A_COST backend; reuses frozen native lowering, RTE law and owned pool.

Never imported by metadata preparation. Real loaders are called only after the
outer SOURCE/input/runtime/grant gate. All H4/H6/H8 inputs remain private.
"""
import hashlib
import importlib
import inspect
import math
import random
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pr_cost_contract as c


def seed(system,binding,cell,replica,source):
    key=['PR_A_COST_20261018',system,binding,cell,'EXPLORATION_N8',replica,source]
    return int(c.digest(key)[:16],16)


def seed_hierarchy(master,q):
    # Frozen direct_master_to_step_seeds_v1 from df_partial_s2_repeated.
    if q==1:return [master]
    rng=random.Random(master)
    return [master]+[rng.randrange(0,2**63) for _ in range(q)]


def steps(preparation,cell,master_seed):
    """Same make_rte_config and repeated-request API as the original MatchedPort."""
    from trotterlib.rte import make_rte_config
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    from trotterlib.df_partial_s2_repeated import make_df_partial_s2_repeated_request
    delta=.8/cell['q'];tau=preparation.exact_rte_lambda_r*delta/cell['r']
    config,distribution=make_rte_config(preparation.rte_preparation.symbolic_tail,
        evolution_time=delta,rte_steps=cell['r'],finite_taylor_order=cell['K'],
        truncation_tolerance=_explicit_cutoff_tolerance(tau,cell['K']),seed=master_seed)
    request=make_df_partial_s2_repeated_request(preparation,step_time=delta,repetition_count=cell['q'],
        rte_config=config,rte_distribution=distribution,seed=master_seed,controlled=True,
        ancilla_qubit=preparation.num_system_qubits,construction_policy='raw_concatenation')
    return tuple(request.iter_step_requests()),distribution


def event_values(requests):
    return [[event.to_dict() for event in step.rte_occurrence.events] for step in requests]


def control(evolution,requests,context,writer,tolerance):
    """Frozen low/high branch, X/Y wrapper and algebraic event oracle checks."""
    m=context.m;state=context.qstate;np=m.np;n=state.size.bit_length()-1
    low=np.concatenate((state,np.zeros_like(state)));high=np.concatenate((np.zeros_like(state),state))
    a=m.simulate_statevector(evolution.circuit,low);b=m.simulate_statevector(evolution.circuit,high)
    z=complex(np.vdot(state,b[len(state):]))
    errors=[float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(state)])),abs(float(np.linalg.norm(b))-1)]
    for axis,value in [('cosine',z.real),('sine',z.imag)]:
        wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=1000000)
        errors.append(abs(m.hadamard_expectation(m.simulate_statevector(wrapper,low),n)-value))
    # Existing full-time oracle includes original constant + extracted identity once.
    from h6_dba_science import event_oracle
    errors.append(abs(z-event_oracle(requests,m,context.native,state)))
    maximum=max(errors)
    c.require(all(math.isfinite(x) for x in errors) and maximum<tolerance,'NATIVE_SCALAR_CONTROL_EVENT_ORACLE')
    writer.write('control.json',dict(status='PASS',max_error=maximum,tolerance=tolerance,
        control_probes=4,independent_event_oracles=1,full_Fock_scalar_once=True,
        reference_rotated=False,ordinary_ancilla_branch=True))
    return z


def compile_task(context,task,directory,*,synthetic=False):
    cell=task['cell'];m=context.m;prep=context.preparations[cell['prefix']]
    c.core.cell_key(cell)
    writer=Writer(directory);budget=c.Budget(directory,dict(trajectory=1,occurrence=cell['q'],
        RTE_event=cell['R'],native_evolution=1,wrapper_compile=2,control_probe=4,independent_event_oracle=1))
    budget.take('trajectory');budget.take('occurrence',cell['q']);budget.take('RTE_event',cell['R'])
    request,distribution=steps(prep,cell,task['seed'])
    expected=seed_hierarchy(task['seed'],cell['q'])
    c.require([s.seed for s in request]==(expected if cell['q']==1 else expected[1:]),'FROZEN_SEED_HIERARCHY')
    event_digest=c.digest(event_values(request))
    distribution_data=distribution.to_dict()
    writer.write('trajectory.json',dict(cell=cell,seed=task['seed'],event_digest=event_digest,
        events=event_values(request),original_RTE_distribution=distribution_data,
        cohort='EXPLORATION_N8',conditional=False,quantum_weights_changed=False))
    # A_SIGNAL normalization is compared against the original finite law, not refit.
    log_B=cell['R']*math.log(distribution.exact_finite_distribution)
    c.require(math.isfinite(log_B) and abs(log_B-task['saved_log_B'])<=task['log_B_tolerance'],
              'A_SIGNAL_RTE_NORMALIZATION_BINDING')
    c.require(prep.preparation_hash==task['preparation_hash'],'EXACT_PREPARATION_IDENTITY')
    budget.take('native_evolution')
    evolution=context.native.partial_native_from_step_requests(request,control_policy='symmetric_directional',max_instructions=1000000)
    if task['replica']==0:
        budget.take('control_probe',4);budget.take('independent_event_oracle')
        control(evolution,request,context,writer,task['control_tolerance'])
    pairs={}
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    with use_fingerprint_reuse():
        for axis in c.core.AXES:
            c.require(c.digest(event_values(request))==event_digest,'PAIRED_EVENT_SEQUENCE_CHANGED')
            wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
            fp=m.canonical_qiskit_circuit_fingerprint(wrapper);budget.take('wrapper_compile')
            start=time.monotonic()
            result=m.transpile_and_measure_cost(wrapper,context.compiler,actual_circuit_fingerprint=fp)
            c.require(result.compiler_settings_hash==context.binding['compiler']
                      and result.actual_circuit_fingerprint==fp
                      and m.canonical_qiskit_circuit_fingerprint(wrapper)==fp,'COMPILE_IDENTITY')
            metrics={key:getattr(result,name) for key,name in dict(RZ='rz_count',CX='cx_count',
                depth='total_depth',RZ_depth='rz_depth',CX_depth='cx_depth',size='circuit_size').items()}
            c.require(all(type(v) is int and v>=0 for v in metrics.values()) and metrics['RZ']>0,'INTEGER_COMPILED_COST')
            pairs[axis]=dict(metrics,seed=task['seed'],event_digest=event_digest,
                compiler_hash=result.compiler_settings_hash,fingerprint=fp,wall_seconds=time.monotonic()-start)
            writer.write('measured_'+axis+'.json',pairs[axis])
    return dict(cell_id=cell['id'],replica=task['replica'],seed=task['seed'],event_digest=event_digest,
        binding=context.binding,axes=pairs,cohort='EXPLORATION_N8',synthetic=synthetic,
        calls=budget.used,normalization_unchanged=True)


class Writer:
    def __init__(self,directory):self.directory=Path(directory)
    def write(self,name,value):
        c.require(Path(name).name==name,'EXCLUSIVE_FILE_NAME');c.save(self.directory/name,value)


def load_context(system,config,cells,output):
    """Restore per-system archived blocks. Do not call signal/reference solvers."""
    for directory in reversed(config['import_paths']):sys.path.insert(0,directory)
    old=c.read(config['native_config']);writer=Writer(output)
    for key in ('hamiltonian','state','compiler'):
        c.require(old['binding'][key]==config['binding'][key],'ORIGINAL_NATIVE_CONFIG_BINDING')
    c.require(old['binding'].get('snapshot',old['binding'].get('input'))==config['binding']['snapshot'],
              'ORIGINAL_SNAPSHOT_BINDING')
    m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    if system=='H8':
        loader=importlib.import_module('h8_ab_native')
        m,native,ham,sector,state,indices,qstate,preparations,compiler=loader.load(old,writer,
            sorted({x['prefix'] for x in cells}))
    elif system=='H6':
        loader=importlib.import_module('h6_dba_native')
        m,native,ham,sector,state,qstate,preps,compiler=loader.load(old)
        view=importlib.import_module('h6_dba_signal').prefix_view
        preparations={p:(preps['faithful:p5'] if p==5 else view(preps['faithful:p19'],ham,p,m))
                      for p in {x['prefix'] for x in cells}}
    else:
        # The plan explicitly registers at most three original native preparations
        # when an archive is unavailable. This is not SCF/DF/state generation.
        from trottertracks.resource_applicability.h4_common_panel_port_v1 import h4_loader
        ham,sector,full,state,md,_=h4_loader(Path(old['mapping']['snapshot']),
            c.read(old['mapping']['snapshot_receipt']),c.read(old['mapping']['df_receipt']))
        _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
        c.require(md['hamiltonian_hash']==config['binding']['hamiltonian']
                  and md['state_hash']==config['binding']['state'],'H4_SAVED_STATE_IDENTITY')
        if config.get('H4_preparation_archive'):
            codec=importlib.import_module('trottertracks.resource_applicability.h6_frozen_preparation_v1')
            archive=config['H4_preparation_archive']
            receipt=codec.read_archive(archive['path'],sha256=archive['sha256'])
            c.require(receipt['binding']==archive['binding'],'H4_ARCHIVE_SOURCE_INPUT_BINDING')
            preps=codec.decode_preparations(receipt['codec'])
            preparations={p:preps['faithful:p'+str(p)] for p in {x['prefix'] for x in cells}}
        else:
            c.require(config.get('H4_reconstruct_original') is True,'EXPLICIT_H4_PREPARATION_MODE')
            path=inspect.getsourcefile(m._prepare)
            c.require(c.sha(path)=='7528ebae754e14ee623d28212a668e4edc237575775a78d0d2bf4d34dc53c293',
                      'ORIGINAL_H4_PREPARATION_BUILDER_SOURCE')
            prefixes=sorted({x['prefix'] for x in cells})
            c.require(set(prefixes)<={2,3,4},'REGISTERED_H4_PREFIX_PREPARATION_BUDGET')
            writer.write('H4_preparation_reconstruction_reserved.json',dict(prefixes=prefixes,
                original_builder_sha256=c.sha(path),SCF_DF_state_solver=0,nonrefundable=True))
            preparations={p:m._prepare(ham,p) for p in prefixes}
        cp=old['compiler_options'];import qiskit
        compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,
                                    cp['seed_transpiler'],qiskit.__version__)
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    c.require(compiler_settings_hash(compiler)==config['binding']['compiler'],'RESTORED_COMPILER')
    for p,prep in preparations.items():
        c.require(prep.hamiltonian_hash==config['binding']['hamiltonian']
                  and prep.deterministic_fragment_indices==tuple(range(p)),'RESTORED_PREPARATION_BINDING')
        expected=config.get('preparation_identities',{}).get(str(p))
        c.require((expected is not None and prep.preparation_hash==expected) or
                  (system=='H4' and config.get('H4_reconstruct_original') is True
                   and not config.get('H4_preparation_archive') and expected is None),
                  'PREPARATION_ARCHIVE_OR_REGISTERED_BUILDER_IDENTITY')
    writer.write('preparation_receipt.json',dict(prefixes={str(p):dict(preparation_hash=v.preparation_hash,
        hamiltonian_hash=v.hamiltonian_hash,scalar=v.constant_coefficient,tail_identity=v.extracted_identity_coefficient)
        for p,v in preparations.items()},SCF_DF_state_solver=0))
    return SimpleNamespace(m=m,native=native,qstate=qstate,preparations=preparations,compiler=compiler,
                           binding=config['binding'])


def execute(system,config,grant,output):
    p=c.plan(system,'A_COST');saved=c.read(config['mapping']['A_signal'])
    registry=c.read(c.ROOT/c.REGISTRY)['systems'][system]
    expected_records={r['cell']['id']:r['record_sha256'] for r in registry['records']}
    c.require({r['cell']['id']:c.digest(r) for r in saved['signals']}==expected_records,'PINNED_A_SIGNAL_RECORDS')
    selected=c.core.acquisition_candidates(saved['signals'],p['cells'])['selected']
    context=load_context(system,config,selected,output)
    signals={x['cell']['id']:x for x in saved['signals']}
    excluded=set(c.read(config['mapping']['seed_exclusions'])['master_and_derived_seed_values'])
    tasks=[]
    for cell in selected:
        record=signals[cell['id']]
        for replica in range(p['n']):
            value=seed(system,config['binding'],cell,replica,grant['SOURCE'])
            hierarchy=seed_hierarchy(value,cell['q'])
            c.require(len(hierarchy)==len(set(hierarchy)) and set(hierarchy).isdisjoint(excluded),
                      'OLD_OR_NEW_SEED_COLLISION');excluded.update(hierarchy)
            tasks.append(dict(id=cell['id']+'_'+str(replica),cell=cell,replica=replica,seed=value,
                preparation_hash=context.preparations[cell['prefix']].preparation_hash,
                saved_log_B=record['log_B'],log_B_tolerance=record['allowance']['log_B_margin'],
                control_tolerance=1e-9 if system=='H4' else 1e-8))
    import h8_key_compile_parallel as pool
    from h6_dba_science import checked_results
    def callback(task,directory):return compile_task(context,task,directory)
    def batch(items,name):
        receipt=pool.map_owned(items,callback,cpus=grant['allowed_cpus'],
            limits=dict(max_workers=len(grant['allowed_cpus'])),directory=output/name,
            deadline=math.inf,font_seed=config['font_seed'])
        c.save(output/(name+'_native_receipt.json'),receipt)
        return checked_results(receipt)
    rows=batch([t for t in tasks if t['replica']==0],'first_control_workers')
    rows+=batch([t for t in tasks if t['replica']!=0],'remaining_workers')
    c.require(len(rows)==len(tasks),'COMPLETE_A_COST_COVERAGE')
    return dict(status='COMPLETE_MANDATORY_STOP',system=system,stage='A_COST',paired_costs=rows,
        normalization_unchanged=True,original_n32_mixed=False,new_signal=0,new_quantum_shots=0,
        new_ranking=0,next_stage_authorized=False,automatic_retry=False)
