"""One H6 trajectory shard; same frozen scientific kernel and legacy seed SOURCE."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def verify(config):
    for key in ('inputs','previous_results'):
        for row in config[key].values():
            if Path(row['path']).stat().st_size!=row['bytes'] or sha(row['path'])!=row['sha256']:
                raise ValueError('INPUT_BYTES_OR_SHA')
    for rel,h in config['provider_source_hashes'].items():
        if sha(Path(config['provider_root'])/rel)!=h:raise ValueError('PROVIDER_SOURCE')
    for rel,h in config['source_hashes'].items():
        if sha(Path(config['source_root'])/rel)!=h:raise ValueError('ADAPTER_SOURCE')
    if sha(config['plan_path'])!=config['plan_sha256']:raise ValueError('PLAN_BINDING')


def run(c,group,out,*,sample_index):
    import numpy as np
    import qiskit
    from functools import partial
    import trotterlib.rte_compiled_cost as compiled_provider
    compiled_provider.transpile=partial(qiskit.transpile,num_processes=1)
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    from trotterlib.df_rte_circuit import DFRTEEventCircuitRequest
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
    from h46_new2_native import prepare,make_requests,native_evolution,OccupationOracle,forced_event_request
    from h46_new2_core import weights,allocation,matrix_only_tail_coefficients
    from d0_frozen_shot_rule_v1 import u_aware_shots
    system,label,prefix,q=(group[k] for k in ('system','label','prefix','q'))
    started=time.monotonic();plan=c['plan'];checks=[];records=[]
    def progress(phase,**kw):
        print(json.dumps(dict(group=group['id'],phase=phase,elapsed=time.monotonic()-started,**kw)),flush=True)
    def check(error,name):
        error=float(error)
        checks.append(dict(check=name,error=error))
        if not math.isfinite(error) or error>plan['numeric_tolerance']:
            write(out/'failed_validation_checks.json',checks)
            raise ValueError(f'{name}: {error:.17g}')
    progress('saved_input_load')
    if system=='H4':
        from trotterlib.pr2_new_series_validation import _load_snapshot_once
        ham,sector,full,state,metadata,_=_load_snapshot_once(Path(c['inputs']['H4_snapshot.npz']['path']))
    else:
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,full,state,metadata,_=load_h6_snapshot(Path(c['inputs']['H6_snapshot.npz']['path']),
            json.loads(Path(c['inputs']['H6_snapshot_receipt.json']['path']).read_text()),
            json.loads(Path(c['inputs']['H6_df_receipt.json']['path']).read_text()))
    previous=json.loads(Path(c['previous_results'][system]['path']).read_text())
    old=next(r for r in previous['records'] if r['prefix']==prefix and r['label']==label and r['q']==q)
    mean=next(r for r in old['finite_rows'] if r['cutoff']==4)
    if previous['SOURCE']!=c['previous_SOURCE'] or previous['conditions']['T']!=plan['T']:
        raise ValueError('PREVIOUS_NUMERICAL_SERIES_BINDING')
    progress('native_preparation')
    prep=prepare(ham,prefix)
    scalar,lam=matrix_only_tail_coefficients(ham,prefix)
    check(abs(prep.extracted_identity_coefficient-scalar),'full_Fock_scalar')
    check(abs(prep.exact_rte_lambda_r-lam),'full_Fock_lambda')
    check(abs(old['lambda_R']-lam),'previous_lambda')
    check(abs(old['full_Fock_identity_scalar']-ham.constant-scalar),'previous_scalar')
    if tuple(prep.deterministic_fragment_indices)!=tuple(range(prefix)):
        raise ValueError('GENERATION_ORDER_CHANGED')
    bias={a:abs(mean['total_signed_bias'][a]) for a in ('real','imag')}
    shots=u_aware_shots(bias,{a:plan['empirical_allowance'] for a in bias},
        epsilon=plan['epsilon'],log_B_upper=mean['log_B']+plan['log_B_empirical_margin'],
        evidence_kind='EMPIRICAL',evidence_ref=c['previous_results'][system]['sha256'],alpha_axis=.025)
    if shots['eligibility_under_declared_allowance']!='ELIGIBLE':raise ValueError('FIXED_PRECISION_INELIGIBLE')
    write(out/'shot_accounting.json',shots)
    progress('independent_occupation_oracle')
    oracle=OccupationOracle(ham,full,prefix=prefix)
    qstate=np.zeros(1<<ham.n_qubits,complex);qstate[oracle.qindices]=oracle.initial
    low=np.r_[qstate,np.zeros_like(qstate)];high=np.r_[np.zeros_like(qstate),qstate]
    compiler=CompilerSettings(tuple(plan['compiler']['basis_gates']),None,None,
        plan['compiler']['optimization_level'],None,None,plan['compiler']['seed_transpiler'],qiskit.__version__)
    event_hist={};negative_events=0
    for sample in (sample_index,):
        def seed_for(a,b):
            raw=f"{c['seed_SOURCE']}|{group['id']}|{sample}|{a}|{b}".encode()
            return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')
        requests=make_requests(prep,label,q,T=plan['T'],total_steps=32,seed_for=seed_for)
        log_B=math.fsum(len(r.rte_occurrence.events)*math.log(r.rte_distribution.exact_finite_distribution) for r in requests)
        check(abs(log_B-mean['log_B']),'sample_normalization')
        events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
        event_digest=hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest()
        # Private exact replay record; no raw trajectory or circuit is staged.
        write(out/f'events_{sample:02}.json',dict(events=events,seed_policy='SOURCE/group/sample/outer/piece SHA256 first64bits',conditional=False))
        for r in requests:
            for e in r.rte_occurrence.events:
                event_hist[str(e.taylor_order)]=event_hist.get(str(e.taylor_order),0)+1
                negative_events+=int(r.step_time<0)
        progress('native_build',sample=sample)
        evolution=native_evolution(requests,label,q,plan['T'])
        z=None
        if sample==0:
            expected=oracle.trajectory(requests)
            check(np.linalg.norm(simulate_statevector(evolution.circuit,low)-low),'native_control_low')
            actual=simulate_statevector(evolution.circuit,high)
            check(np.linalg.norm(actual[:len(qstate)]),'native_control_leakage')
            check(np.linalg.norm(actual[len(qstate):]-expected),'native_occupation_trajectory')
            z=complex(np.vdot(qstate,expected))
        paired={}
        for axis in ('cosine','sine'):
            wrapper=build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=10**7)
            fp=canonical_qiskit_circuit_fingerprint(wrapper)
            start=time.monotonic()
            progress('compile_enter',sample=sample,axis=axis,instructions=len(wrapper.data))
            with (out/'compile_attempts.jsonl').open('a') as f:
                f.write(json.dumps(dict(sample=sample,axis=axis,event_digest=event_digest,fingerprint=fp))+'\n')
            cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fp)
            wall=time.monotonic()-start
            if sample==0:
                unmeasured=wrapper.remove_final_measurements(inplace=False)
                before=simulate_statevector(unmeasured,low)
                native_mean=float(np.vdot(before[:len(qstate)],before[:len(qstate)]).real-
                                  np.vdot(before[len(qstate):],before[len(qstate):]).real)
                check(abs(native_mean-(z.real if axis=='cosine' else z.imag)),'native_hadamard_'+axis)
                after=simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
                check(np.linalg.norm(after-before),'compiled_statevector_'+axis)
            fields=('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')
            row=dict(sample=sample,axis=axis,event_digest=event_digest,
                metrics={k:int(getattr(cost,k)) for k in fields},wall_seconds=wall,
                untranspiled_instructions=len(wrapper.data),fingerprint=fp,
                compiler_settings_hash=cost.compiler_settings_hash)
            records.append(row);paired[axis]=row['metrics']['rz_count']
            write(out/f'cost_{sample:02}_{axis}.json',row)
            progress('compile_complete',sample=sample,axis=axis,RZ=row['metrics']['rz_count'],wall=wall)
            del cost,wrapper;gc.collect()
        del evolution;gc.collect()
    if label=='4th(new_2)' and sample_index==0:
        progress('forced_rare_correctness')
        builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
        components={v.component_id:v for v in prep.rte_preparation.symbolic_tail.components}
        for sign in (-1,1):
            r=next(r for r in requests if r.step_time*sign>0)
            for order in (0,2,4):
                e=forced_event_request(r,order)
                req=DFRTEEventCircuitRequest(e,prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=ham.n_qubits)
                circuit=builder.build_event(req).circuit
                expected=np.zeros_like(qstate);expected[oracle.qindices]=oracle.event(e,components,oracle.initial)
                actual=simulate_statevector(circuit,high)
                check(np.linalg.norm(actual[len(qstate):]-expected),f'forced_order{order}_sign{sign}')
                check(np.linalg.norm(simulate_statevector(circuit,low)-low),f'forced_low_order{order}_sign{sign}')
    totals=[]
    for sample in (sample_index,):
        rows={r['axis']:r for r in records if r['sample']==sample}
        totals.append(sum(shots['axes'][a]['shots']*rows[axis]['metrics']['rz_count']
            for a,axis in (('real','cosine'),('imag','sine'))))
    output=dict(status='COMPLETE_DIAGNOSTIC',SOURCE=c['SOURCE'],provider_SOURCE=c['provider_SOURCE'],
        plan_sha256=c['plan_sha256'],group=group,conditions=dict(geometry_angstrom=1.,basis='STO-3G',
        rank=ham.n_blocks,n_qubits=ham.n_qubits,original_sector_dimension=sector.dimension,
        oracle_particle_number_dimension=len(oracle.basis),T=.8,K=4,total_RTE_steps=q*32),
        preparation_hash=prep.preparation_hash,scalar=ham.constant+scalar,lambda_R=lam,
        previous_exact_mean=mean,shots=shots,validation_checks=checks,records=records,
        observed_order_histogram=event_hist,negative_time_event_count=negative_events,
        forced_events_enter_cost_sample=False,compiled_invocations=len(records),
        RZ_shot_weighted_sample_costs=totals,RZ_shot_weighted_mean=statistics.mean(totals),
        RZ_shot_weighted_SE=None,shard_sample_index=sample_index,seed_SOURCE=c['seed_SOURCE'],
        population_mean_certified=False,quantum_measurement_shots_sampled=0,
        imported_provider_modules={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
            if name.startswith(('trotterlib.','trottertracks.')) and getattr(module,'__file__',None)},
        state_preparation_cost=0,full_RPE_energy_cost_evaluated=False,
        wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    verify(c)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--group',required=True);parser.add_argument('--sample',type=int,choices=range(8),required=True)
    args=parser.parse_args();c=json.loads(Path(args.config).read_text())
    sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,str(Path(c['provider_root'])/'src'))
    class NoGPU:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('cupy','cupyx','cuquantum'):raise ImportError('CPU-only diagnostic')
    sys.meta_path.insert(0,NoGPU())
    group=next(g for g in c['plan']['groups'] if g['id']==args.group)
    out=Path(c['output'])/args.group/('sample_%02d'%args.sample);out.mkdir(parents=True,exist_ok=False)
    try:
        if c['authorized'] is not True:raise ValueError('NOT_AUTHORIZED')
        verify(c);result=run(c,group,out,sample_index=args.sample);write(out/'result.json',result)
    except BaseException as e:
        write(out/'STOP.json',dict(SOURCE=c['SOURCE'],group=args.group,type=type(e).__name__,reason=str(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
