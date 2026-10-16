"""Parallel scheduling of the original V1 signal/oracle/allowance operations.

One shared snapshot/reference/archive; deduplicated primitive validation; one
worker per truncation prefix (shared exact reference); two PR diagnostic tasks.
No compiler, sampler, changed recurrence, or scientific-precision shortcut.
"""
from copy import copy,deepcopy
import json
from pathlib import Path
import time

import track_a_phase_a_parallel as pool
from track_a_v1_signal import make_overlays,outer_diagnostics


def task_plan(cells,times):
    primitives=[dict(id='primitive_'+str(i),kind='primitive',block=i,
                     times=sorted(t for j,t in times if i==j),
                     counts=dict(primitive=3*sum(j==i for j,t in times)))
                for i in sorted({i for i,t in times})]
    signals=[]
    for prefix in sorted({c['prefix'] for c in cells}):
        selected=[c for c in cells if c['prefix']==prefix]
        signals.append(dict(id='prefix_'+str(prefix),kind='signals',prefix=prefix,cells=selected,
            counts=dict(new_B0_signals=len(selected),analytic_B0I_overlays=len(selected),
                        exact_truncated_reference_pairs=1,tail_scalar=1)))
    diagnostics=[dict(id='outer_prefix_'+str(p),kind='diagnostic',prefix=p,
                      counts=dict(outer_S2_diagnostics=4)) for p in (3,4)]
    return primitives,signals+diagnostics


def evaluate_prepared(config,permit,contract,writer,port,whole,ham,state,overlay,m,schedules):
    permit.require('signal')
    np=m.np
    times={(i,t) for c in port.cells for i,t in schedules.primitive_time_schedule(c,T=.8)['unique_primitive_times']}
    primitive_tasks,signal_tasks=task_plan(port.cells,times)
    # Populate native callables and spectra once and inherit them read-only.
    largest=max(port.cells,key=lambda c:c['prefix']);prepared=port.preps[largest['id']]
    native=port.actions(prepared);oracle=port.actions(prepared,oracle=True)
    for c in port.cells:port.native_actions[port.preps[c['id']].preparation_hash]=native[:c['prefix']+1]
    probes=[state,np.eye(1,len(state),0,dtype=complex)[0],np.eye(1,len(state),len(state)-1,dtype=complex)[0]]
    deadlines=permit.started+permit.grant['resources']['wall_seconds']

    def callback(task,directory):
        child=permit.child(directory,task['counts'])
        child_writer=type(writer)(directory,child.guard)
        local=copy(port);local.writer=child_writer;local.progress=child
        local.native_actions=dict(port.native_actions);local.spectra=dict(port.spectra)
        local.truncated_references={};local.completed=0;local.calls=deepcopy(port.calls)
        if task['kind']=='primitive':
            i=task['block'];errors=[];spectral=[]
            key='one' if i==0 else i-1
            for t in task['times']:
                exact=m.expm(-1j*t*local.primitive_matrix(key))
                for vector in probes:
                    child.take('primitive')
                    e=float(np.linalg.norm(native[i](vector,t)-exact@vector))
                    se=float(np.linalg.norm(oracle[i](vector,t)-exact@vector))
                    m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(e,se))
                    errors.append(e);spectral.append(se)
            result=dict(kind='primitive',max_native_expm_error=max(errors,default=0.),
                max_spectral_expm_error=max(spectral,default=0.),actual_time_count=len(task['times']))
        elif task['kind']=='signals':
            local.cells=task['cells'];local.plan=dict(port.plan,cells=local.cells)
            actual=local.cell_signal;independent=local.oracle_signal
            def cell_signal(c,**kw):
                child.take('new_B0_signals');return actual(c,**kw)
            def oracle_signal(c):
                if c['prefix'] not in local.truncated_references:child.take('exact_truncated_reference_pairs')
                return independent(c)
            local.cell_signal=cell_signal;local.oracle_signal=oracle_signal
            local.correctness()
            rows=child.read_output('signal_summary.json')['signals']
            corrected=make_overlays(local,rows,child,child_writer,ham,overlay,m)
            child_writer.write('prefix_results.json',dict(signals=rows+corrected))
            result=dict(kind='signals',file='prefix_results.json',rows=len(rows)+len(corrected))
        else:
            rows=outer_diagnostics(local,whole,ham,state,child,m,contract.QS,(task['prefix'],))
            child_writer.write('outer_results.json',dict(rows=rows))
            result=dict(kind='diagnostic',file='outer_results.json',rows=len(rows))
        child.guard()
        if child.used!=task['counts']:raise RuntimeError('WORKER_EXPECTED_CALL_COUNTS_CHANGED')
        return dict(result=result,calls=child.used)

    def batch(tasks,name):
        receipt=pool.map_owned(tasks,callback,cpus=permit.grant['worker_cpus'],
            limits=permit.grant['resources'],directory=writer.directory/name,deadline=deadlines,
            guard=permit.guard,reserve=lambda task:permit.reserve(task['id'],task['counts']),
            font_seed=config['font_seed'],prior_child_CPU_seconds=permit.child_CPU_seconds)
        permit.child_CPU_seconds=receipt['child_CPU_seconds']
        results=[]
        for h in receipt['workers']:
            directory=Path(h['directory']);value=json.loads((directory/'task_result.json').read_bytes())
            permit.merge_reserved(h['task_id'],value['calls'])
            results.append((directory,value['result']))
        return receipt,results

    validation,primitive_results=batch(primitive_tasks,'primitive_workers')
    rows=[row for directory,row in primitive_results]
    if sum(r['actual_time_count'] for r in rows)!=len(times):raise RuntimeError('PRIMITIVE_TIME_COVERAGE_CHANGED')
    writer.write('primitive_validation.json',dict(actual_time_count=len(times),probe_count=3,
        max_native_expm_error=max(r['max_native_expm_error'] for r in rows),
        max_spectral_expm_error=max(r['max_spectral_expm_error'] for r in rows),
        all_registered_times=True,includes_negative_yoshida_times=True,certified=False,
        parallel_native_exits_verified=True))
    signal_receipt,results=batch(signal_tasks,'signal_workers')
    signals=[];diagnostics=[]
    for directory,row in results:
        values=json.loads((directory/row['file']).read_bytes())
        if row['kind']=='signals':signals.extend(values['signals'])
        else:diagnostics.extend(values['rows'])
    order={c['id']:i for i,c in enumerate(contract.registrations()['additional'])}
    signals.sort(key=lambda row:order[row['cell']['id']]);diagnostics.sort(key=lambda row:(row['prefix'],row['q']))
    if len(signals)!=104 or len(diagnostics)!=8:raise RuntimeError('PHASE_A_SIGNAL_COMPLETENESS')
    writer.write('outer_S2_diagnostics.json',dict(rows=diagnostics))
    writer.write('V1_signal_complete.json',dict(status='COMPLETE_MANDATORY_STOP',signals=signals,
        diagnostic_count=len(diagnostics),call_counts=permit.used,conservative_bias_certificate=None,
        new_native_full_wrapper=0,new_compile=0,new_sampling=0,
        parallel=dict(primitive_workers=len(primitive_tasks),signal_tasks=len(signal_tasks),
                      maximum_concurrency=len(permit.grant['worker_cpus']),
                      child_CPU_seconds=permit.child_CPU_seconds,all_reaped=True),
        next_stage_authorized=False))
