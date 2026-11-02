"""Fixed C-1 pilot: technical runtime probe and one-shot representative-first driver.

No environment/thread settings, CPU affinity, shared packages or old evidence are
changed. Science starts only through the newly bound approval/grant gate.
"""
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from h8_rare_c05.accounting import (Budget, atomic, begin_approved, read, sha,
    paired_pilot_summary, independent_pilot_moments)
from h8_rare_c05.adapter import Adapter
from h8_rare_c05.parallel import map_fork, single_thread
from h8_rare_c05.provider import OriginalProvider
from h8_rare_c05.resources import Admission
from h8_rare_c05.sampling import digest, require
from h8_rare_c1_contract import budget as fixed_budget

BASE='77bf6472f901feab8cdc4871369bc8ddfc04be7b'


def process_memory(pid=None):
    pid=pid or os.getpid()
    p=Path('/proc')/str(pid)
    values={}
    for line in (p/'smaps_rollup').read_text().splitlines():
        if ':' in line:
            key,value=line.split(':',1)
            try:values[key]=int(value.split()[0])*1024
            except (ValueError,IndexError):pass
    return dict(RSS=values.get('Rss'),PSS=values.get('Pss'),
        private=sum(values.get(k,0) for k in ('Private_Clean','Private_Dirty','Private_Hugetlb')),
        threads=len(list((p/'task').iterdir())))


def library_record(library):
    """Read existing equivalences; preserve entry/parameter/instruction order.

    Does not ask gates to construct their .definition or compile any circuit.
    Unsupported runtime API fails closed instead of inventing a signature.
    """
    rows=[]
    for key in library.keys():
        entries=library._get_equivalences(key)
        out=[]
        for item in entries:
            circ=item.circuit
            out.append(dict(params=[str(p) for p in item.params],
                circuit_parameters=[str(p) for p in circ.parameters],
                global_phase=str(circ.global_phase),qubits=circ.num_qubits,clbits=circ.num_clbits,
                instructions=[dict(name=i.operation.name,params=[str(p) for p in i.operation.params],
                    qargs=[circ.find_bit(q).index for q in i.qubits],
                    cargs=[circ.find_bit(c).index for c in i.clbits]) for i in circ.data]))
        rows.append(dict(name=key.name,num_qubits=key.num_qubits,entries=out))
    require(bool(rows),'EMPTY_SESSION_LIBRARY')
    return dict(schema='C1_EXISTING_SESSION_EQUIVALENCES_V1',keys=rows)


def live_signature():
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    return digest(library_record(SessionEquivalenceLibrary))


def require_safe_parent(expected_library=None):
    require(single_thread(),'UNSAFE_MULTITHREADED_FORK_PARENT')
    if expected_library is not None:
        require(live_signature()==expected_library,'LIVE_LIBRARY_CHANGED')


def bind_imports(config):
    sys.dont_write_bytecode=True
    for row in config['byte_rows']:
        require(sha(row['path'])==row['sha256'],'FROZEN_BYTES_CHANGED')
    for path in reversed(config['modules']['import_paths']):sys.path.insert(0,path)
    for name,wanted in config['modules']['files'].items():
        module=importlib.import_module(name)
        require(sha(module.__file__)==wanted,'SHADOWED_RUNTIME_MODULE')
    import qiskit
    require(qiskit.__version__=='1.3.0','QISKIT_VERSION')
    require_safe_parent()
    value=library_record(importlib.import_module('qiskit.circuit.equivalence_library').SessionEquivalenceLibrary)
    return value


def runtime_probe(config,output):
    """Imports existing runtime only: no molecular load, law, RNG or circuits."""
    output=Path(output);output.mkdir(exist_ok=False)
    start=time.monotonic()
    try:
        value=bind_imports(config)
        atomic(output/'session_library.json',value)
        signature=digest(value)
        require_safe_parent(signature)
        def child(task,directory,width):
            require_safe_parent(signature)
            return dict(library=live_signature(),memory=process_memory(),scientific_calls=0)
        receipt=map_fork([dict(id='runtime_identity_only')],child,output/'fork_probe',
            lambda:dict(cpu=1,memory=process_memory()['RSS'],io=1),
            lambda task,slots:dict(cpu=1,memory=process_memory()['RSS'],io=1,axis_width=1))
        require(receipt['status']=='PASS' and receipt['all_reaped'],'TECHNICAL_FORK_PROBE')
        atomic(output/'probe.json',dict(status='PASS_TECHNICAL_RUNTIME_IMPORT_LIBRARY_FORK',
            library_signature=signature,memory=process_memory(),elapsed=time.monotonic()-start,
            source_input_runtime_bytes=True,science_calls=0,thread_policy_changed=False,
            owned_child_reaped=True))
        return 0
    except BaseException as exc:
        atomic(output/'probe.json',dict(status='STOP_TECHNICAL_RUNTIME_GATE',
            reason=type(exc).__name__+':'+str(exc),memory=process_memory(),
            elapsed=time.monotonic()-start,science_calls=0,valid_grant_consumed=False,
            traceback=traceback.format_exc(),thread_policy_changed=False,retry=False))
        return 2


def scope_gate(c,tasks,registry):
    expected=('B2_p8_q2_2nd_r4_K4',2) if c['condition']=='H8_0.05' else ('B2_p7_q16_2nd_r4_K4',16)
    require(c['condition'] in ('H8_0.05','H8_0.0005') and (c['candidate'],c['q'])==expected,'CONDITION')
    require(c['strata']==['S2_SINGLE'] and c['n_by_stratum']==dict(S2_SINGLE=43,S2_MULTI=None,S4_PRESENT=None),'FIXED_N43_SINGLE')
    require(c['frozen_scientific_budget']==fixed_budget(c['q'],43),'FIXED_CALL_BUDGET')
    require(not c['cost_cache_enabled'] and not c['production_authorized'],'FIXED_PILOT_ONLY')
    require(len(tasks)==43 and {t['replica'] for t in tasks}==set(range(43)),'43_REGISTERED_REPLICAS')
    require(len({t['id'] for t in tasks})==43 and set(registry)=={t['id'] for t in tasks},'REGISTRY_IDS')
    for t in tasks:
        require(t['candidate']==c['candidate'] and t['q']==c['q'] and t['r']==4 and t['K']==4
            and t['phase']=='variance_load_pilot' and t['stratum']=='S2_SINGLE','TASK_SCOPE')
    require(digest(dict(tasks=tasks,streams=registry))==c['private_task_registry_digest'],'PRIVATE_REGISTRY_SHA')
    first=next(t for t in tasks if t['replica']==0)
    require(c['validation_task_ids']==[first['id']],'PRE_REGISTERED_CONTROL')
    return first


def verify_rows(adapter,tasks,receipt):
    require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['adopted_descendants'],'NATIVE_BATCH')
    require({h['id'] for h in receipt['workers']}=={t['id'] for t in tasks},'EXACT_BATCH_IDS')
    for h in receipt['workers']:
        require(h['exit_code']==0 and h['reaped'],'NATIVE_EXIT_REAP')
        root=Path(h['directory']);row=h['result'];trace=read(root/'trajectory.json')
        require(read(root/'result.json')==row and row['task_id']==h['id'],'RESULT_RECEIPT')
        orders=row['orders'];q=adapter.c['q']
        require(len(orders)==4*q and orders.count(2)==1 and orders.count(4)==0 and set(orders)<={0,2},'SINGLE_ORDER_COUNTS')
        require(trace['event_digest']==row['event_digest']==digest(trace['events']),'COMPLETE_EVENT_DIGEST')
        require(len(trace['events'])==q and all(len(s)==4 for s in trace['events']),'COMPLETE_EVENT_COUNT')
        require(set(row['axes'])=={'cosine','sine'},'PAIRED_AXES')
        for axis,item in row['axes'].items():
            require(item['axis']==axis and item['event_digest']==row['event_digest'] and not item['cache_hit'],'PAIRED_EVENT_CACHE')
            require(item['compiler']==adapter.c['original_binding']['compiler'],'FIXED_COMPILER')
            require(all(type(v) is int and v>=0 for v in item['metrics'].values()) and item['metrics']['RZ']>0,'COMPILED_METRICS')
        native=read(root/'axes/native_receipt.json')
        require(native['status']=='PASS' and native['all_reaped'] and all(x['exit_code']==0 and x['reaped'] for x in native['workers']),'AXES_EXIT_REAP')


def execute_batches(adapter,tasks,registry,root,admission,library_gate=lambda:None):
    """No second dispatch if representative/control/native verification fails."""
    root=Path(root);first=scope_gate(adapter.c,tasks,registry)
    all_receipts=[];observations=[];last=[0.]
    def observe(rows):
        admission.observe(rows)
        if time.monotonic()-last[0]>=1:
            values=[]
            for h in rows:
                try:values.append(dict(task=h['id'],**process_memory(h['pid'])))
                except (FileNotFoundError,ProcessLookupError):pass
            observations.append(dict(monotonic=time.monotonic(),workers=values,capacity=admission.capacity()))
            atomic(root/'resource_observations.json',observations);last[0]=time.monotonic()
    for name,selected in [('representative',[first]),('remaining',[t for t in tasks if t['replica']!=0])]:
        library_gate()
        def callback(task,directory,width):
            library_gate();start=time.monotonic();before=process_memory()
            row=adapter.trajectory(task,directory,width,registry[task['id']])
            row['load']=dict(elapsed=time.monotonic()-start,before=before,after=process_memory(),axis_width=width)
            library_gate();return row
        receipt=map_fork(selected,callback,root/name,admission.capacity,admission.demand,observe=observe)
        all_receipts.append(receipt);atomic(root/'batch_receipts.json',all_receipts)
        verify_rows(adapter,selected,receipt)
        if name=='representative':
            control=read(Path(receipt['workers'][0]['directory'])/'control.json')
            require(control['status']=='PASS' and control['tolerance']==1e-8 and control['max_error']<1e-8
                and control['control_probes']==4 and control['independent_event_oracles']==1
                and control['full_Fock_scalar_once'] and not control['reference_rotated'],'FIRST_CONTROL_GATE')
    rows=[h['result'] for r in all_receipts for h in r['workers']]
    expected=dict(adapter.c['frozen_scientific_budget']);expected.pop('pilot_accounting');expected.pop('independent_numeric_validation')
    require(adapter.budget.used()==expected,'EXACT_BEFORE_ACCOUNTING_CALLS')
    adapter.budget.take('pilot_accounting',1,'pilot_accounting')
    summary=paired_pilot_summary(rows,adapter.c['shots'],adapter.c['candidate'],adapter.c['phase'])
    atomic(root/'pilot_summary.json',summary)
    adapter.budget.take('independent_numeric_validation',1,'independent_numeric_validation')
    audit=independent_pilot_moments(rows,adapter.c['shots'],summary)
    atomic(root/'independent_numeric_audit.json',audit)
    require(adapter.budget.used()==adapter.c['frozen_scientific_budget'],'FINAL_EXACT_CALLS')
    adapter.provider.final_gate();library_gate()
    atomic(root/'result.json',dict(condition=adapter.c['condition'],candidate=adapter.c['candidate'],n=43,
        rows=rows,summary=summary,numeric_audit=audit,old_cohorts_pooled=False,
        population_guarantee=False,unknown_strata=['S2_MULTI','S4_PRESENT'],production=False))
    with (root/'paired_costs.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['task','axis','RZ','CX','depth','RZ_depth','CX_depth','size','paired_G'])
        for row in rows:
            G=sum(adapter.c['shots'][a]*row['axes'][a]['metrics']['RZ'] for a in ('cosine','sine'))
            for a in ('cosine','sine'):
                v=row['axes'][a]['metrics'];writer.writerow([row['task_id'],a]+[v[k] for k in ['RZ','CX','depth','RZ_depth','CX_depth','size']]+[G])
    atomic(root/'completion.json',dict(status='COMPLETE_C1_CONDITION_MANDATORY_STOP',n=43,compile=86,
        calls=adapter.budget.used(),all_owned_reaped=True,source_input_runtime_gate=True,library_gate=True))
    return summary


def run_condition(config,root):
    root=Path(root);root.mkdir(exist_ok=False)
    c=read(config['contract']);tasks=config['tasks'];registry=config['registry']
    scope_gate(c,tasks,registry)
    b=None
    try:
        probe=read(config['technical_probe'])
        require(probe['status']=='PASS_TECHNICAL_RUNTIME_IMPORT_LIBRARY_FORK','TECHNICAL_PREFLIGHT_NOT_PASS')
        demand=config['resource_evidence']
        require(demand.get('provenance') and demand.get('includes_two_axes_and_control_oracle') is True,'RESOURCE_DEMAND_UNSUBSTANTIATED')
        admission=Admission(demand['private_peak_bytes'],demand['threads_per_axis'],demand['IO_profile_capacity'])
        cap=admission.capacity();cost=admission.demand(tasks[0],cap)
        require(cap['memory']>=cost['memory'] and cap['cpu']>=cost['cpu'] and cap['io']>=cost['io'],'FRESH_REPRESENTATIVE_RESOURCES')
        value=bind_imports(config);signature=digest(value)
        require(signature==probe['library_signature'],'RUNTIME_PROBE_IDENTITY_CHANGED')
        permit=begin_approved(config['contract'],config['grant'],root/'one_shot_consumed.json',config['approval'],config['byte_rows'])
        b=Budget(root/'budget',c['frozen_scientific_budget'])
        cell=dict(prefix=8 if c['q']==2 else 7,q=c['q'],r=4,K=4)
        provider=OriginalProvider.load_parent(permit,config['adapter_config'],[cell],root/'restored',config['modules'],config['byte_rows'],b)
        require_safe_parent(signature)
        adapter=Adapter(provider,permit,c,b)
        adapter.initialize_law(config['law_root_seed'])
        expected=read(config['saved_law_bindings'])['conditions'][c['condition']]['saved_law']
        require(adapter.law.to_dict()==expected,'SAVED_TAU_NORMALIZATION_SUPPORT')
        require(adapter.cell['q']==c['q'] and adapter.cell['r']==4 and adapter.cell['K']==4,'LAW_Q_R_K_T')
        summary=execute_batches(adapter,tasks,registry,root,admission,lambda:require_safe_parent(signature))
        return summary
    except BaseException as exc:
        atomic(root/'STOP.json',dict(status='MANDATORY_STOP_C1',reason=type(exc).__name__+':'+str(exc),
            traceback=traceback.format_exc(),reserved={} if b is None else b.used(),
            marker_consumed=(root/'one_shot_consumed.json').exists(),retry=False,resume=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--runtime-probe',type=Path);mode.add_argument('--run',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.runtime_probe:return runtime_probe(read(args.runtime_probe),args.output)
    run_condition(read(args.run),args.output);return 0


if __name__=='__main__':raise SystemExit(main())
