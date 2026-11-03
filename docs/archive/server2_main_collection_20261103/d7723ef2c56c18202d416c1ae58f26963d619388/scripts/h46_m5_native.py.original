"""Lazy native pilot: fixed signed S2 composition, one scalar, paired wrappers.

Only the authorized launcher calls this module. Toy tests inject the lowerer;
real preparation, RNG, statevectors and transpile never run during preparation.
"""
import gc
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

import h46_m5_contract as c

CACHE={}


def make_requests(group,prep,seeds,lambda_R,*,factory,tolerance,checkpoint=lambda phase:None):
    result=[]
    for s in c.core.schedule(group['formula'],group['q'],group['policy'],group['r_value']):
        checkpoint('sample_signed_piece')
        r=factory(prep,step_time=s['signed_time'],rte_steps=s['rte_steps'],
            truncation_tolerance=tolerance(lambda_R*s['signed_time']/s['rte_steps'],4),
            finite_taylor_order=4,seed=seeds[f"{s['outer']}:{s['piece']}"],controlled=True,
            ancilla_qubit=prep.num_system_qubits,pf_label='2nd')
        c.require(len(r.rte_occurrence.events)==s['rte_steps'],'REGISTERED_RTE_EVENT_COUNT')
        result.append(r)
    return tuple(result)


def compose(group,preparation,requests,scalar,api,make_circuit,fingerprint):
    """Use existing S2 native lowering; scalar is applied once after all pieces."""
    policy=group['policy'] or 'PER_PIECE';density=group['r_value'] or 1
    steps=c.core.schedule(group['formula'],group['q'],policy,density)
    if group['method']=='PR':c.require(len(requests)==len(steps),'REQUEST_SCHEDULE_COUNT')
    else:c.require(not requests,'DETERMINISTIC_HAS_NO_RNG_REQUEST')
    n=preparation.num_system_qubits;qc=make_circuit(n+1);bound=1
    if group['method']=='PR':
        from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
        builder=QiskitDFRTEEventCircuitBuilder(basis_registry=preparation.rte_preparation.basis_registry)
    for i,s in enumerate(steps):
        blocks=preparation.deterministic_blocks
        base=sum(api.block_instruction_bound(b,mode) for b in blocks
                 for mode in ('UNCONTROLLED','DIRECTIONAL'))
        if group['method']=='PR':
            r=requests[i]
            c.require(r.step_time==s['signed_time'] and r.controlled and r.ancilla_qubit==n
                and r.preparation.preparation_hash==preparation.preparation_hash,'SIGNED_REQUEST_IDENTITY')
            tail=builder.build_sequence(r.rte_occurrence,basis_plan=r.rte_basis_plan).circuit
            cap=base+len(tail.data)+1
            part=api.build_partial_native(blocks,num_system_qubits=n,T=s['signed_time'],q=1,
                scalar=0.,controlled_tails=[tail],control_policy='symmetric_directional',max_instructions=cap)
        else:
            part=api.build_deterministic_native(blocks,num_system_qubits=n,T=s['signed_time'],q=1,
                formula='2nd',scalar=0.,control_policy='symmetric_directional',max_instructions=base+1)
        qc.compose(part.circuit,inplace=True);bound+=part.instruction_upper_bound
    if scalar*.8!=0:qc.p(-scalar*.8,n)
    return api.NativeEvolution(qc,n,.8,group['q'],group['formula'],
        'symmetric_directional',bound,fingerprint(qc))


def install(config):
    c.require(config['runtime']['provider_root']==config['provider_root'],'PROVIDER_PATH_BINDING')
    root=Path(config['provider_root']).resolve()
    if str(root/'src') not in sys.path:sys.path.insert(0,str(root/'src'))
    class NoGPU:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('cupy','cupyx','cuquantum'):raise ImportError('CPU-only pilot')
    if not any(isinstance(x,NoGPU) for x in sys.meta_path):sys.meta_path.insert(0,NoGPU())
    import qiskit
    from functools import partial
    import trotterlib.rte_compiled_cost as costs
    c.require(qiskit.__version__=='1.3.0','QISKIT_VERSION')
    costs.transpile=partial(qiskit.transpile,num_processes=1)
    for name,mod in list(sys.modules.items()):
        if name.startswith(('trotterlib.','trottertracks.')) and getattr(mod,'__file__',None):
            c.require(Path(mod.__file__).resolve().is_relative_to(root),'IMPORTED_PROVIDER_ORIGIN')


def context(config,*,reference_job=False):
    key=(config['SOURCE'],config['system'],c.core.digest(config['inputs']))
    if key in CACHE:return CACHE[key]
    install(config)
    import numpy as np
    from h46_new2_native import OccupationOracle,prepare
    from h46_new2_core import spectrum,matrix_only_tail_coefficients
    if config['system']=='H4':
        from trotterlib.pr2_new_series_validation import _load_snapshot_once
        ham,sector,full,state,md,_=_load_snapshot_once(Path(config['inputs']['H4_snapshot.npz']['path']))
    else:
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,full,state,md,_=load_h6_snapshot(Path(config['inputs']['H6_snapshot.npz']['path']),
            c.read(config['inputs']['H6_snapshot_receipt.json']['path']),
            c.read(config['inputs']['H6_df_receipt.json']['path']))
    binding=config['binding'];system=c.core.SYSTEMS[config['system']]
    c.require(md['hamiltonian_hash']==binding['hamiltonian'] and md['state_hash']==binding['state'],'SAVED_HAMILTONIAN_STATE')
    c.require(ham.n_blocks==system['rank'] and ham.n_qubits==(8 if config['system']=='H4' else 12),'FIXED_LAYOUT')
    fullprep=prepare(ham,ham.n_blocks);cut=prepare(ham,system['anchor'])
    oracle=OccupationOracle(ham,full,prefix=ham.n_blocks)
    qstate=np.zeros(1<<ham.n_qubits,complex);qstate[oracle.qindices]=oracle.initial
    c.require(abs(np.linalg.norm(qstate)-1)<1e-12,'SAVED_STATE_NORMALIZATION')
    matrices=[(spec[1]*spec[0])@spec[2] for spec in oracle.specs]
    reference_spec=None
    tail_identity,lam=matrix_only_tail_coefficients(ham,system['anchor'])
    c.require(abs(cut.extracted_identity_coefficient-tail_identity)<1e-9
        and abs(cut.exact_rte_lambda_r-lam)<1e-9,'FULL_FOCK_SCALAR_LAMBDA')
    c.require(tuple(cut.deterministic_fragment_indices)==tuple(range(system['anchor']))
        and tuple(fullprep.deterministic_fragment_indices)==tuple(range(ham.n_blocks)),'GENERATION_ORDER')
    tail=sum(matrices[system['anchor']+1:])-tail_identity*np.eye(len(oracle.initial))
    tail_spec=spectrum(tail)
    if reference_job:
        reference_spec=spectrum(sum(matrices))
        reference=reference_spec[1]@(np.exp(-.8j*reference_spec[0])*(reference_spec[2]@oracle.initial))
        reference*=np.exp(-.8j*ham.constant)
    else:
        path=Path(config['reference_root'])/'reference.npz';receipt=c.read(path.parent/'receipt.json')
        c.require(c.sha(path)==receipt['reference_sha256'],'IMMUTABLE_FULL_H_REFERENCE_BYTES')
        with np.load(path,allow_pickle=False) as saved:
            reference=saved['reference'].copy()
            c.require(np.array_equal(saved['qstate'],qstate),'SHARED_REFERENCE_STATE')
    CACHE[key]=SimpleNamespace(np=np,ham=ham,oracle=oracle,qstate=qstate,fullprep=fullprep,
        cut=cut,reference=reference,reference_spec=reference_spec,tail_spec=tail_spec,
        scalar=ham.constant,tail_identity=tail_identity,lambda_R=lam)
    return CACHE[key]


def preflight(config):
    """One immutable full-H reference job/system; no SCF/DF/state generation."""
    c.authorize(config);c.consumed(config)
    c.require(config['plan']['scope']['reference_jobs']==1,'REGISTERED_DET_REFERENCE_JOB_ONLY')
    out=Path(config['reference_root']);out.mkdir(parents=True,exist_ok=False)
    c.save(out/'reserved.json',dict(SOURCE=config['SOURCE'],system=config['system'],
        full_H_reference_jobs=1,nonrefundable=True,SCF_DF_state_generation=0))
    ctx=context(config,reference_job=True);np=ctx.np
    file=out/'reference.npz'
    with file.open('xb') as f:np.savez(f,reference=ctx.reference,qstate=ctx.qstate)
    receipt=dict(status='PASS',system=config['system'],SOURCE=config['SOURCE'],
        snapshot=config['binding']['snapshot'],reference_sha256=c.sha(file),
        state_hash=config['binding']['state'],hamiltonian_hash=config['binding']['hamiltonian'],
        reference_policy='FULL_H_DF_CONSTANT_ONCE_OCCUPATION_SECTOR_NO_PHASE_ALIGNMENT',
        reference_norm=float(np.linalg.norm(ctx.reference)),
        full_preparation=ctx.fullprep.preparation_hash,cut_preparation=ctx.cut.preparation_hash,
        scalar=ctx.scalar,tail_identity=ctx.tail_identity,lambda_R=ctx.lambda_R,
        SCF_DF_state_generation=0)
    c.require(abs(receipt['reference_norm']-1)<1e-9,'REFERENCE_NORM')
    c.save(out/'receipt.json',receipt);return receipt


def signal(ctx,group):
    """Exact deterministic/finite-RTE means; no random trajectories or shots."""
    from h46_new2_core import polynomial,spectral_apply,complex_record
    from trotterlib.rte import finite_rte_distribution
    np=ctx.np;count=group['prefix']+1;specs=ctx.oracle.specs[:count]
    vectors=np.repeat(ctx.oracle.initial[:,None],3,axis=1);log_B=0.;maximum=1.
    steps=c.core.schedule(group['formula'],group['q'],group['policy'] or 'PER_PIECE',group['r_value'] or 1)
    for s in steps:
        t=s['signed_time']
        for spec in specs:vectors=spectral_apply(spec,np.exp(-.5j*t*spec[0]),vectors)
        if group['method']=='PR':
            r=s['rte_steps'];b=finite_rte_distribution(ctx.lambda_R*t/r,4).exact_finite_distribution
            f=polynomial(ctx.tail_spec[0],t/r,4)**r
            factors=np.column_stack((np.exp(-1j*t*ctx.tail_spec[0]),f,f/(b**r)))
            vectors=ctx.tail_spec[1]@(factors*(ctx.tail_spec[2]@vectors));log_B+=r*math.log(b)
        for spec in reversed(specs):vectors=spectral_apply(spec,np.exp(-.5j*t*spec[0]),vectors)
        maximum=max(maximum,float(np.max(np.linalg.norm(vectors,axis=0))))
    scalar=ctx.scalar+(ctx.tail_identity if group['method'] in ('PR','B0I') else 0.)
    vectors*=np.exp(-.8j*scalar)
    closure=float(np.linalg.norm(vectors[:,2]-math.exp(-log_B)*vectors[:,1]))
    c.require(math.isfinite(log_B) and np.isfinite(vectors).all() and closure<1e-9,'FINITE_MEAN_NORMALIZATION')
    reference=complex(np.vdot(ctx.oracle.initial,ctx.reference))
    exact=complex(np.vdot(ctx.oracle.initial,vectors[:,0]));corrected=complex(np.vdot(ctx.oracle.initial,vectors[:,1]))
    record=dict(reference=complex_record(reference),exact_tail=complex_record(exact),
        corrected=complex_record(corrected),raw=complex_record(np.vdot(ctx.oracle.initial,vectors[:,2])),
        total_signed_bias=complex_record(corrected-reference),outer_PF_or_discard_bias=complex_record(exact-reference),
        finite_RTE_bias=complex_record(corrected-exact),log_B=log_B,
        allowance_evidence=dict(kind='EMPIRICAL_DIAGNOSTICS_ONLY_NOT_ADOPTED_FOR_SHOTS',
            raw_corrected_vector_closure=closure,maximum_intermediate_norm=maximum,
            strict_bias_guarantee=None,shot_allowance=None,headroom_eligibility=None))
    return record


def run_task(config,task,stop):
    c.authorize(config);c.consumed(config)
    c.require(task in config['plan']['scope']['tasks'],'FIXED_REGISTERED_TASK_BEFORE_SCIENCE')
    install(config)
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    # Existing exact canonical-byte reuse, isolated to this disposable process.
    with use_fingerprint_reuse() as cache:
        result=_run_task(config,task,stop,cache.fingerprint)
        c.save(Path(config['output'])/task['id']/'fingerprint_cache.json',cache.snapshot())
        return result


def _run_task(config,task,stop,fp):
    out=Path(config['output'])/task['id'];out.mkdir(exist_ok=False)
    checks=[];started=time.monotonic()
    def checkpoint(phase):
        if stop.is_set():raise RuntimeError('OWN_LANE_STOP_NO_FURTHER_CALLS')
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,elapsed=time.monotonic()-started))+'\n')
    def check(value,name):
        value=float(value);checks.append(dict(check=name,error=value))
        c.require(math.isfinite(value) and value<1e-9,'NUMERIC_DISAGREEMENT:'+name)
    try:
        checkpoint('context');ctx=context(config);np=ctx.np;g=task['group']
        ref=c.read(Path(config['reference_root'])/'receipt.json')
        c.require(ref['status']=='PASS' and ref['SOURCE']==config['SOURCE']
            and ref['snapshot']==config['binding']['snapshot']
            and ref['full_preparation']==ctx.fullprep.preparation_hash
            and ref['cut_preparation']==ctx.cut.preparation_hash
            and ref['reference_sha256']==c.sha(Path(config['reference_root'])/'reference.npz'),'SHARED_REFERENCE_IDENTITY')
        import qiskit
        from qiskit import QuantumCircuit
        import trottertracks.resource_applicability.ax2b_native_df_v5 as native
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
        from trotterlib.rte import CompilerSettings
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        from trotterlib.df_trotter.circuit import simulate_statevector
        prep=ctx.fullprep if g['method']=='FULL' else ctx.cut
        # B0 uses the identical deterministic block prefix but never the tail scalar.
        scalar=ctx.scalar+(ctx.tail_identity if g['method'] in ('PR','B0I') else 0.)
        requests=[];events=[];log_B=0.
        if g['method']=='PR':
            seeds=c.read(config['registry_path'])['lanes'][config['lane']][task['id']]
            requests=make_requests(g,prep,seeds,ctx.lambda_R,factory=make_df_partial_s2_step_request,
                tolerance=_explicit_cutoff_tolerance,checkpoint=checkpoint)
            for r in requests:
                events.append([e.to_dict() for e in r.rte_occurrence.events])
                log_B+=len(r.rte_occurrence.events)*math.log(r.rte_distribution.exact_finite_distribution)
        event_digest=c.core.digest(events);c.save(out/'events_PRIVATE.json',dict(events=events,conditional=False,event_digest=event_digest))
        checkpoint('native_build');evolution=compose(g,prep,requests,scalar,native,QuantumCircuit,fp)
        low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)];high=np.r_[np.zeros_like(ctx.qstate),ctx.qstate]
        record=None;z=None
        if task['sample']==0:
            checkpoint('signal_and_controls');record=signal(ctx,g);check(abs(log_B-record['log_B']),'normalization_log_B')
            if g['method']=='PR':expected=ctx.oracle.trajectory(requests)
            else:
                value=ctx.oracle.initial.copy()
                for s in c.core.schedule(g['formula'],g['q'],'PER_PIECE',1):
                    for spec in list(ctx.oracle.specs[:g['prefix']+1])+list(reversed(ctx.oracle.specs[:g['prefix']+1])):
                        value=spec[1]@(np.exp(-.5j*s['signed_time']*spec[0])*(spec[2]@value))
                value*=np.exp(-.8j*scalar);expected=np.zeros_like(ctx.qstate);expected[ctx.oracle.qindices]=value
            check(np.linalg.norm(simulate_statevector(evolution.circuit,low)-low),'native_low')
            actual=simulate_statevector(evolution.circuit,high)
            check(np.linalg.norm(actual[:len(ctx.qstate)]),'native_leakage')
            check(np.linalg.norm(actual[len(ctx.qstate):]-expected),'independent_occupation_vector')
            z=complex(np.vdot(ctx.qstate,expected))
            c.save(out/'signal.json',record)
        compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
        rows=[]
        for axis in ('cosine','sine'):
            checkpoint('compile_'+axis)
            wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
                max_instructions=len(evolution.circuit.data)+4)
            fingerprint=fp(wrapper);c.reserve(config,task,axis,fingerprint)
            cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
            c.require(cost.compiler_settings_hash==c.COMPILER_HASH and cost.actual_circuit_fingerprint==fingerprint
                and fp(wrapper)==fingerprint,'COMPILED_IDENTITY')
            if task['sample']==0:
                before=simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
                observed=float(np.vdot(before[:len(ctx.qstate)],before[:len(ctx.qstate)]).real
                    -np.vdot(before[len(ctx.qstate):],before[len(ctx.qstate):]).real)
                check(abs(observed-(z.real if axis=='cosine' else z.imag)),'hadamard_'+axis)
                after=simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
                check(np.linalg.norm(after-before),'compiled_vector_'+axis)
            metrics={k:getattr(cost,k) for k in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')}
            c.require(all(type(v) is int and v>=0 for v in metrics.values()) and metrics['rz_count']>0,'INTEGER_NATIVE_COST')
            row=dict(axis=axis,metrics=metrics,event_digest=event_digest,fingerprint=fingerprint,
                compiler_settings_hash=cost.compiler_settings_hash,native_policy=c.POLICY)
            c.save(out/('cost_'+axis+'.json'),row);rows.append(row);del cost,wrapper;gc.collect()
        checkpoint('final_seal');c.verify_seals(config)
        result=dict(status='PASS',task=task['id'],lane=config['lane'],SOURCE=config['SOURCE'],
            plan_sha256=config['plan_sha256'],binding=config['binding'],native_policy=c.POLICY,
            cost_compatibility='NEW_PILOT_NATIVE_SERIES_OLD_MAIN_COSTS_NOT_YET_JOINED',
            old_main_profile_is_reference_only=True,old_main_costs_reused=False,
            sample=task['sample'],conditional=False,seed_scope='M5_TECHNICAL_PILOT_NOT_CONFIRMATION',
            preparation_hash=prep.preparation_hash,scalar=scalar,log_B=log_B,
            records=rows,checks=checks,shot_accounting=False,ranking=False,strict_bias_guarantee=None,
            wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        c.save(out/'result.json',result);return result
    except BaseException as exc:
        stop.set();result=dict(status='STOP',lane=config['lane'],task=task['id'],reason=str(exc),
            traceback=traceback.format_exc(),checks=checks,retry=False)
        c.save(out/'STOP.json',result);return result
