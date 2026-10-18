"""Stage 4: replay fixed saved events, bind native circuits, compile only if needed."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction as F
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import resource


def decode_event(rte, raw):
    data=dict(raw);apps=[]
    for value in raw['application_sequence']:
        a=dict(value)
        a['basis_change_operations']=tuple(rte.BasisChangeOperation(name=o['name'],qubits=tuple(o['qubits']),
            parameters=tuple(o['parameters']),matrix_sha256=o['matrix_sha256']) for o in a['basis_change_operations'])
        a['diagonal_pauli_support']=None if a['diagonal_pauli_support'] is None else tuple(a['diagonal_pauli_support'])
        apps.append(rte.RTEEventApplication(**a))
    data['application_sequence']=tuple(apps)
    for name in ('product_component_ids','selected_component_ids','df_fragment_ids'):data[name]=tuple(data[name])
    data['basis_reuse_intervals']=tuple(rte.BasisReuseInterval(**v) for v in data['basis_reuse_intervals'])
    data['phase']=complex(data['phase']['real'],data['phase']['imag'])
    return rte.RTEEvent(**data)


def reusable(saved, *, candidate, axis, fingerprint, event_digest, compiler):
    return (saved.get('cell_id')==candidate and saved.get('axis')==axis and
            saved.get('control')=='symmetric_directional' and saved.get('fingerprint')==fingerprint and
            saved.get('event_digest')==event_digest and saved.get('compiler_hash')==compiler)


def replay_requests(x,p,c,task,trajectory,rte):
    events=tuple(tuple(decode_event(rte,e) for e in occ) for occ in trajectory['events'])
    if len(events)!=c['q'] or any(len(occ)!=c['r'] for occ in events):raise ValueError('Q_R_EVENT_COUNT')
    if x.lib.digest([[e.to_dict() for e in occ] for occ in events])!=trajectory['event_digest']:
        raise ValueError('EVENT_ROUNDTRIP')
    circuit_api=importlib.import_module('trotterlib.df_rte_circuit');partial=importlib.import_module('trotterlib.df_partial_s2')
    tolerance=importlib.import_module('trottertracks.resource_applicability.ax2b_molecular_ports_v3')._explicit_cutoff_tolerance
    tau=p.exact_rte_lambda_r*(.8/c['q'])/c['r']
    config,dist=rte.make_rte_config(p.rte_preparation.symbolic_tail,evolution_time=.8/c['q'],rte_steps=c['r'],finite_taylor_order=c['K'],truncation_tolerance=tolerance(tau,c['K']),seed=task['seed'])
    if dist.to_dict()!=trajectory['finite_distribution']:raise ValueError('SAVED_EVENT_FINITE_LAW')
    steps=[]
    for occ in events:
        for e in occ:p.rte_preparation.resolve_event_basis_definitions(e)
        request=circuit_api.DFRTEEventSequenceCircuitRequest(events=occ,component_specs=p.rte_preparation.component_specs,
            controlled=True,ancilla_qubit=p.num_system_qubits,tail_id=config.tail_id,tail_hash=config.tail_hash,occurrence_rte_steps=c['r'])
        steps.append(partial.DFPartialS2StepRequest(preparation=p,step_time=.8/c['q'],rte_config=config,
            rte_distribution=dist,rte_occurrence=request,controlled=True,ancilla_qubit=p.num_system_qubits,seed=task['seed']))
    return events,steps


def take(x, output, kind, candidate, axis=None):
    import fcntl
    # Shared only among this one-shot's own workers. Reserve before every call.
    with (output/'calls.jsonl').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        old=[json.loads(line) for line in f]
        caps={'compile':14,'control_probe':28}
        if kind not in caps or sum(r['kind']==kind for r in old)>=caps[kind]:
            raise ValueError('FIXED_CALL_CAP')
        if kind=='compile':
            system_cap=8 if candidate.startswith('H4:') else 6
            if sum(r['kind']==kind and r['candidate'].split(':')[0]==candidate.split(':')[0] for r in old)>=system_cap:
                raise ValueError('FIXED_SYSTEM_COMPILE_CAP')
        f.seek(0,2);f.write(json.dumps(dict(kind=kind,candidate=candidate,axis=axis))+'\n');f.flush();os.fsync(f.fileno())


def paired_theory(x,p,c,events,ops):
    m=x.m;rows=[];random=c['method']=='B2';n=p['n'];bases=tuple(b['basis'] for b in p['blocks'])
    scalar=p['constant']+(p['extracted'] if random else 0.)
    by_id={a['component_id']:a for a in p['components']}
    def lookup(a):return by_id[a.component_id]['basis']
    for model in ('A','B','NATIVE_MIXED'):
        pm='A' if model=='NATIVE_MIXED' else model;tm='B' if model=='NATIVE_MIXED' else model
        if random:
            prefix=m.partial_prefix_circuit(bases,x.factory(p,ops),.8,c['q'],n,'symmetric_directional',-scalar*.8,pm)
            tail=sum((m.count(m.existing_rte_occurrence(occ,lookup,n,tm)) for occ in events),m.Cost())
        else:
            prefix=m.deterministic_circuit(bases,x.factory(p,ops),.8,c['q'],c['formula'],'symmetric_directional',n,-scalar*.8,pm)
            tail=m.Cost()
        for axis,quad in [('cosine','X'),('sine','Y')]:
            cost=m.count(m.hadamard_wrapper(prefix,n,quad))+tail
            rows.append(dict(model=model,axis=axis,cost=x.packed_cost(cost)))
    return rows


def raw_lowered(x,circuit):
    total=x.m.Cost();unknown=[];hist=Counter()
    for item in circuit.data:
        op=item.operation;hist[op.name]+=1
        if op.name=='measure':continue
        try:
            g=x.m.Gate(op.name,tuple(circuit.find_bit(q).index for q in item.qubits),*[float(p) for p in op.params])
            total+=x.m.gate_cost(g)
        except ValueError:unknown.append(op.name)
    return dict(histogram=dict(hist),cost=None if unknown else x.packed_cost(total),
                unknown_primitives=sorted(set(unknown)),measurement_excluded=True)


def worker(x,plan,index,output):
    job=plan['jobs'][index];c=job['candidate'];system=job['system'];cid=c['id'];n=8 if system=='H4' else 12
    own=output/(system+'_'+cid);own.mkdir(exist_ok=False)
    terminal=dict(status='MANDATORY_STOP_FAILURE',first_stop=None,candidate_id=cid,system=system,mandatory_stop=True)
    start=time.monotonic()
    try:
        rte,ops=x.provider_modules(plan)
        codec=importlib.import_module('trottertracks.resource_applicability.h6_frozen_preparation_v1')
        raw=x.read(plan['inputs']['h4_codec' if system=='H4' else 'h6_archive']['path'])
        preps=codec.decode_preparations(raw if system=='H4' else raw['codec'])
        p=preps['faithful:p'+str(c['prefix'])];meta=x.from_live(p)
        identity=plan['system_inputs'][system]
        if p.hamiltonian_hash!=identity['hamiltonian_hash'] or p.deterministic_fragment_indices!=tuple(range(c['prefix'])):
            raise ValueError('HAMILTONIAN_PREFIX_BINDING')
        if system=='H4':loader=importlib.import_module('trottertracks.resource_applicability.h4_common_panel_port_v1').h4_loader
        else:loader=importlib.import_module('trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1').load_h6_snapshot
        inputs=plan['inputs']
        ham,sector,full,state,metadata,_=loader(Path(inputs[system+'_snapshot']['path']),x.read(inputs[system+'_snapshot_receipt']['path']),x.read(inputs[system+'_df_receipt']['path']))
        if metadata['hamiltonian_hash']!=p.hamiltonian_hash or metadata['state_hash']!=identity['state_hash']:
            raise ValueError('SAVED_MODEL_STATE_BINDING')
        bridge=importlib.import_module('trottertracks.resource_applicability.ax2b_h4_science_v5')
        _,qstate=bridge.checked_basis_bridge(ham,sector,full,state,1e-12)
        trajectory=x.read(inputs[job['trajectory_input']]['path']);event_digest=x.lib.digest(trajectory['events'])
        if trajectory['event_digest']!=event_digest:raise ValueError('SAVED_EVENT_DIGEST')
        events=None;steps=None
        if c['method']=='B2':
            events,steps=replay_requests(x,p,c,job['task'],trajectory,rte)
        theory=paired_theory(x,meta,c,events,ops)
        x.write(own/'paired_theory.json',theory)
        native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
        reuse=importlib.import_module('trottertracks.resource_applicability.h6_hash_reuse_v1')
        cost_api=importlib.import_module('trotterlib.rte_compiled_cost')
        simulate=importlib.import_module('trotterlib.df_trotter.circuit').simulate_statevector
        expectation=importlib.import_module('trottertracks.resource_applicability.ax2b_molecular_ports_v3').hadamard_expectation
        import numpy as np
        import qiskit
        cp=plan['compiler'];settings=rte.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
        compiler=cost_api.compiler_settings_hash(settings)
        old_terminal=x.read(inputs[job['terminal_input']]['path'])
        if old_terminal['status'] not in ('COMPLETE','TASK_COMPLETE') or old_terminal.get('reason') is not None:
            raise ValueError('INCOMPLETE_SAVED_NATIVE_MEASUREMENT')
        print(json.dumps(dict(phase='NATIVE_BUILD',system=system,candidate=cid)),flush=True)
        with reuse.use_fingerprint_reuse() as cache:
            evolution=(native.partial_native_from_step_requests(steps,control_policy='symmetric_directional',max_instructions=plan['max_instructions']) if steps else
                native.build_deterministic_native(p.deterministic_blocks,num_system_qubits=n,T=.8,q=c['q'],formula=c['formula'],scalar=p.constant_coefficient,
                    control_policy='symmetric_directional',max_instructions=plan['max_instructions']))
            label=system+':'+cid;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
            checks=[]
            print(json.dumps(dict(phase='CONTROL',system=system,candidate=cid)),flush=True)
            take(x,output,'control_probe',label,'branch0');a=simulate(evolution.circuit,low)
            take(x,output,'control_probe',label,'branch1');b=simulate(evolution.circuit,high)
            checks.extend((float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(qstate)]))))
            z=complex(np.vdot(qstate,b[len(qstate):]));metrics=[];compiled=0;reused=0
            if not steps and 'expected_signal' in job['task']:
                target=job['task']['expected_signal'];checks.append(abs(z-complex(target['real'],target['imag'])))
            for axis,value in (('cosine',z.real),('sine',z.imag)):
                probe=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=plan['max_instructions'])
                take(x,output,'control_probe',label,axis);checks.append(abs(expectation(simulate(probe,low),n)-value))
                del probe
                wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=plan['max_instructions'])
                fingerprint=cache.fingerprint(wrapper);raw_count=raw_lowered(x,wrapper)
                saved=x.read(inputs[job[axis+'_input']]['path'])
                same=reusable(saved,candidate=cid,axis=axis,fingerprint=fingerprint,event_digest=event_digest,compiler=compiler)
                if max(checks)>plan['control_tolerance']:raise ValueError('NATIVE_CONTROL_SEMANTICS')
                if same:
                    measured=saved['metrics'];reused+=1;kind='SAVED_COMPILED_EXACT_WRAPPER_DIGEST'
                else:
                    print(json.dumps(dict(phase='COMPILE',system=system,candidate=cid,axis=axis)),flush=True)
                    take(x,output,'compile',label,axis)
                    cost=cost_api.transpile_and_measure_cost(wrapper,settings,actual_circuit_fingerprint=fingerprint)
                    measured={k:getattr(cost,v) for k,v in dict(RZ='rz_count',CX='cx_count',RZ_depth='rz_depth',CX_depth='cx_depth',depth='total_depth',size='circuit_size').items()}
                    compiled+=1;kind='NEW_COHORT_COMPILED';del cost
                record=dict(system=system,candidate_id=cid,axis=axis,metrics=measured,fingerprint=fingerprint,event_digest=event_digest,
                    preparation_hash=p.preparation_hash,compiler_hash=compiler,compiled_origin=kind,old_fingerprint=saved['fingerprint'],old_fingerprint_match=same,
                    saved_sample_replica=job['task']['replica'],saved_sample_phase=job['task']['phase'],seed=job['task']['seed'],
                    seed_namespace='REPLAY_ORIGINAL_SAVED_EVENT_NO_RANDOM_DRAW',sample_size=1,theory_kind='PAIRED_EVENT_SAMPLE' if steps else 'DETERMINISTIC',
                    raw_native=raw_count,high_level_instructions=len(wrapper.data),compiler_runtime_bit_equivalence_to_old_host_certified=False,
                    native_scalar_phase_hex=float(-.8*(p.constant_coefficient+(p.extracted_identity_coefficient if steps else 0.))).hex())
                x.write(own/('wrapper_'+axis+'.json'),record);metrics.append(record);del wrapper
            x.write(own/'control.json',dict(probes=4,max_error=max(checks),errors=checks,event_digest=event_digest,
                branch1_signal=dict(real=z.real,imag=z.imag),random_signal_is_ensemble=False,finite_sample_mean_claimed=False))
            x.write(own/'hash_reuse.json',cache.snapshot())
        terminal.update(status='COMPLETE_MANDATORY_STOP',new_compile=compiled,reused_compile=reused,new_sampling=0,control_probes=4,wrappers=2)
    except BaseException as exc:terminal['first_stop']=repr(exc);raise
    finally:
        terminal.update(wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        x.write(own/'terminal.json',terminal)


def coordinator(x,plan,output,plan_path,grant_path):
    pending=list(range(len(plan['jobs'])));active={};completed=[];native=[];first=None
    identity=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    x.write(output/'parent_identity.json',identity)
    with (output/'worker_resources.jsonl').open('x') as trace:
        while pending or active:
            while pending and len(active)<plan['workers'] and first is None:
                i=pending.pop(0);base=output/('worker_'+str(i));base.mkdir(exist_ok=False)
                out=(base/'stdout.log').open('xb');err=(base/'stderr.log').open('xb')
                command=[plan['python'],'-P','-B',str(Path(__file__).with_name('h46_cost_execution.py')),'--plan',plan_path,'--grant',grant_path,'--worker',str(i)]
                child=subprocess.Popen(command,stdout=out,stderr=err,env=dict(os.environ,**plan['process_environment']))
                start=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                active[i]=(child,out,err,start);x.journal(output,'worker_dispatch',index=i,pid=child.pid,starttime=start)
            for i,(child,out,err,start) in list(active.items()):
                code=child.poll()
                if code is None:
                    try:trace.write(json.dumps(dict(index=i,pid=child.pid,status=Path(f'/proc/{child.pid}/status').read_text()))+'\n')
                    except FileNotFoundError:pass
                    continue
                code=child.wait();out.close();err.close()
                native.append(dict(index=i,pid=child.pid,starttime=start,returncode=code,wait_reap=True,own_process_remaining=Path(f'/proc/{child.pid}').exists()))
                completed.append(i);del active[i]
                if code!=0 and first is None:first=dict(index=i,returncode=code)
            trace.flush()
            if first is not None:
                pending.clear()
                for child,_,_,start in active.values():
                    try:
                        if Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]==start:child.terminate()
                    except ProcessLookupError:pass
                    except FileNotFoundError:pass
            if active:time.sleep(1)
    x.write(output/'native_workers.json',native)
    if first is not None:raise RuntimeError('FIRST_WORKER_STOP:'+json.dumps(first))
    if len(completed)!=7:raise ValueError('FIXED_SEVEN_COHORT_COVERAGE')
    records=[x.read(output/(j['system']+'_'+j['candidate']['id'])/'terminal.json') for j in plan['jobs']]
    x.journal(output,'fixed_panel_synthesis_reserved',expectation_rows=84,same_event_rows=84,pf_pr_rows=360)
    summary=x.load('_h46_summary',Path(__file__).with_name('h46_cost_summary.py')).summarize(x,plan,output)
    return dict(candidates=7,wrappers=14,new_compile=sum(r['new_compile'] for r in records),
        reused_compile=sum(r['reused_compile'] for r in records),new_sampling=0,control_probes=28,native_workers_reaped=7,**summary)
