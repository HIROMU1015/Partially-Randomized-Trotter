"""Grant-only H2 adapters over frozen forward, native DF, RTE and compile code.

No molecular/scientific import at module import; real entrypoint authorizes first.
"""
from copy import copy, deepcopy
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import h2_panels_contract as c


def degenerate(cell, lam):
    c.require(math.isfinite(lam) and lam>=0, 'INVALID_TAIL_NORMALIZATION')
    if lam: return None
    return dict(cell=deepcopy(cell),status='DEGENERATE_EMPTY_TAIL',lambda_r=lam,
        precision_rows={str(e):{'1.0':dict(cost_acquisition_eligible=False,
            eligibility_under_declared_allowance='INELIGIBLE',reason='DEGENERATE_EMPTY_TAIL',
            axes={},N_total=None)} for e in c.EPS},signal_evaluated=False,
        sampling=0,compile=0,no_PF_substitution=True)


def setup(spec, cells, output, calls, loader=None):
    config=spec['native_config']
    sys.path.insert(0,str(Path(config['provider_root'])/'src'))
    sys.path.insert(0,spec['A_kernel_scripts'])
    import h4_native_science as h
    from h2_tol_only_input import load
    m=h.provider(); p=h.accounting_plan()
    p.update(T=.8,actual_rank=3,cells=cells,epsilons=c.EPS,kind=c.SCOPE,
        df_policy=dict(kind='SAVED_TOL_ONLY_1E-8_RANK3',snapshot_sha256=c.BINDING['snapshot']),
        input_policy='ACCEPTED_F_SAVED_BYTES_NO_REGENERATION',matrix_free=dict(backend='python'))
    p['numerical']['sensitivity_factors']=[1.]
    cap=p['caps_proposed']
    cap['tail_matvecs_corrected_and_raw']['B2']=max([2*x['R']*(x['K']+1) for x in cells if x['R']]+[0])
    cap['deterministic_actions_per_cell']=max(x['q']*len(m.primitive_time_schedule(x,T=.8)
        ['ordinary_one_outer_step'])*(2 if x['R'] else 1) for x in cells)
    cap['reference_matvec']=0
    port=m.MatchedPort(Path(config['provider_root']),dict(kind=c.SCOPE,plan=p),c.Writer(output),h.Progress())
    port.ham,port.sector,full,state,md=(loader or load)(config)
    c.require((port.ham.n_qubits,len(port.ham.lambdas),port.sector.dimension)==(4,3,4), 'H2_LAYOUT')
    c.require(md['hamiltonian_hash']==c.BINDING['hamiltonian'] and md['state_hash']==c.BINDING['state'],
              'H2_ACCEPTED_F_IDENTITY')
    port.qindices,port.qstate=m.checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    port.state=state.copy(); port.saved_state=state.copy()
    port.spectra={}; port.native_actions={}; port.tails={}; port.oracle_tails={}; port.truncated_references={}
    if spec['stage']=='FP':
        calls.take('full_reference_primary_columns',4)
        op,_=m.df_linear_operator(port.ham,port.sector,backend='python')
        eye=m.np.eye(4,dtype=complex); matrix=m.np.column_stack([op@v for v in eye])
        calls.take('full_reference_independent_columns',4)
        oracle=m.np.column_stack([m.occupation_column(port.ham,port.sector.basis_indices,int(i))
                                 for i in port.sector.basis_indices])
        error=float(m.np.linalg.norm(matrix-oracle)); c.require(error<1e-9,'H2_REFERENCE_OCCUPATION_ORACLE')
        calls.take('full_reference_pair')
        port.reference,discrepancy=m.reference_pair(matrix,state,.8)
        port.reference_error=max(error,*discrepancy.values())
        reference=dict(binding=c.BINDING,T=.8,reference=m.complex_record(port.reference),
            error=port.reference_error,expm_eigh=discrepancy,occupation_discrepancy=error)
    else:
        fp=c.read(spec['dependencies']['FP']['result']); reference=fp['reference']
        c.require(reference['binding']==c.BINDING and reference['T']==.8,'SAVED_FP_REFERENCE_BINDING')
        port.reference=complex(reference['reference']['real'],reference['reference']['imag'])
        port.reference_error=reference['error']
    cache={}
    for x in cells:
        key=(x['method']=='B0',x['prefix'])
        if key not in cache:
            calls.take('preparation'); cache[key]=(m._prepare_discard if key[0] else m._prepare)(port.ham,key[1])
        port.preps[x['id']]=cache[key]
    c.save(output/'preparations.json',dict(binding=c.BINDING,SCF_DF_state_solver=0,
        rows={cid:dict(preparation_hash=v.preparation_hash,hamiltonian_hash=v.hamiltonian_hash,
            lambda_r=v.exact_rte_lambda_r) for cid,v in port.preps.items()}))
    # Static schedule count must equal the actual original fragment/time pairs.
    coverage={(('one' if i==0 else str(i-1)),t) for x in cells
        for i,t in m.primitive_time_schedule(x,T=.8)['unique_primitive_times']}
    count=3*len(coverage); c.require(count<=spec['primitive_probes'],'SEALED_PRIMITIVE_BUDGET')
    p['coverage']['primitive_actions']=cap['primitive']=count
    port.calls.limits['primitive']=count
    for prep in cache.values(): port.actions(prep)
    calls.take('primitive_spectrum',len({('one' if i==0 else str(i-1)) for x in cells
        for i,t in m.primitive_time_schedule(x,T=.8)['unique_primitive_times']}))
    for prep in cache.values(): port.actions(prep,oracle=True)
    calls.take('primitive_probe',count); port.primitives()
    # Share invariant truncated references and normalized tail matrices via fork/COW.
    for prefix in sorted({x['prefix'] for x in cells if x['method']=='B0'}):
        if spec['stage']!='FP' and str(prefix) in fp['truncated_references']:
            saved=fp['truncated_references'][str(prefix)]
            port.truncated_references[prefix]=(complex(saved['signal']['real'],saved['signal']['imag']),saved['error'])
            continue
        calls.take('truncated_reference_pair')
        matrix=m.np.asarray(m.occupation_df_matrix(port.ham.constant,port.ham.one_body,
            port.ham.lambdas[:prefix],port.ham.g_matrices[:prefix],port.sector.basis_indices),dtype=complex)
        port.truncated_references[prefix]=m.reference_pair(matrix,state,.8)
    for prefix in sorted({x['prefix'] for x in cells if x['R']}):
        prep=next(port.preps[x['id']] for x in cells if x['R'] and x['prefix']==prefix)
        if degenerate(next(x for x in cells if x['R'] and x['prefix']==prefix),prep.exact_rte_lambda_r): continue
        ii=prep.randomized_block_indices; calls.take('independent_tail_matrix')
        port.oracle_tails[prefix]=m.np.asarray(m.occupation_df_matrix(-prep.extracted_identity_coefficient,
            m.np.zeros_like(port.ham.one_body),port.ham.lambdas[list(ii)],
            tuple(port.ham.g_matrices[i] for i in ii),port.sector.basis_indices),dtype=complex)/prep.exact_rte_lambda_r
    return port,m,h,reference


def evaluate(port,task,directory):
    local=copy(port); local.plan=deepcopy(port.plan); local.cells=[task['cell']]
    local.plan['cells']=local.cells; local.calls=deepcopy(port.calls)
    local.writer=c.Writer(directory); local.completed=0
    c.save(directory/'signal_reserved.json',dict(cell=task['cell'],nonrefundable=True))
    local.correctness()
    return c.read(directory/'signal_summary.json')['signals'][0]


def compile_one(port,m,compiler,task,directory,stage):
    """Same native kernels; dynamic H2 qubit count and nonnegative RZ metrics."""
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    import h4_native_science as h
    x=task['cell']; prep=port.preps[x['id'].replace('B0I_','B0_',1)]
    writer=c.Writer(directory); events=None; event_digest=c.digest(None)
    c.save(directory/'native_reserved.json',dict(task=task,nonrefundable=True))
    if x['R']:
        import pr_cost_native as kernel
        requests,distribution=kernel.steps(prep,x,task['seed'])
        hierarchy=c.stats.seed_hierarchy(task['seed'],x['q'])
        c.require([s.seed for s in requests]==(hierarchy if x['q']==1 else hierarchy[1:]), 'FROZEN_SEED_HIERARCHY')
        events=kernel.event_values(requests); event_digest=c.digest(events)
        log_B=x['R']*math.log(distribution.exact_finite_distribution)
        record=task['expected']
        c.require(math.isfinite(log_B) and abs(log_B-record['log_B'])<=record['allowance']['log_B_margin'],
                  'SAVED_ENSEMBLE_NORMALIZATION')
        c.save(directory/'trajectory.json',dict(cell=x,seed=task['seed'],events=events,event_digest=event_digest,
            original_RTE_distribution=distribution.to_dict(),conditional=False,
            cohort=c.plan(stage)['cohort'],quantum_weights_changed=False))
        native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
        evolution=native.partial_native_from_step_requests(requests,control_policy='symmetric_directional',max_instructions=1000000)
        if task['replica']==0:
            context=SimpleNamespace(m=m,native=native,qstate=port.qstate)
            kernel.control(evolution,requests,context,writer,1e-9)
    else:
        scalar=task['expected'].get('scalar_receipt',{}).get('scalar',prep.constant_coefficient)
        evolution=m.build_deterministic_native(prep.deterministic_blocks,num_system_qubits=port.ham.n_qubits,
            T=.8,q=x['q'],formula=x['formula'],scalar=scalar,control_policy='symmetric_directional',max_instructions=1000000)
        h.native_control(evolution,m,port.qstate,task['expected']['signals']['corrected'],writer)
    pair={}
    with use_fingerprint_reuse():
        for axis in ('cosine','sine'):
            if events is not None: c.require(c.digest(kernel.event_values(requests))==event_digest, 'PAIRED_EVENTS_UNCHANGED')
            wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
            fp=m.canonical_qiskit_circuit_fingerprint(wrapper)
            c.save(directory/('compile_'+axis+'_reserved.json'),dict(fingerprint=fp,event_digest=event_digest,nonrefundable=True))
            measured=m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fp)
            c.require(measured.compiler_settings_hash==c.BINDING['compiler'] and
                measured.actual_circuit_fingerprint==fp and m.canonical_qiskit_circuit_fingerprint(wrapper)==fp,'COMPILE_IDENTITY')
            metrics={k:getattr(measured,v) for k,v in dict(RZ='rz_count',CX='cx_count',depth='total_depth',
                RZ_depth='rz_depth',CX_depth='cx_depth',size='circuit_size').items()}
            c.require(all(type(v) is int and v>=0 for v in metrics.values()),'NONNEGATIVE_COMPILED_METRICS')
            pair[axis]=dict(metrics,seed=task.get('seed'),event_digest=event_digest,compiler_hash=c.BINDING['compiler'],fingerprint=fp)
            c.save(directory/('measured_'+axis+'.json'),pair[axis])
    result=dict(cell_id=x['id'],replica=task['replica'],seed=task.get('seed'),event_digest=event_digest,
        axes=pair,cohort=c.plan(stage)['cohort'],preparation_hash=prep.preparation_hash,
        stratum=c.stats.event_stratum(events) if events is not None else None)
    c.check_pair(result)
    return result


def batch(spec,tasks,callback,directory):
    if not tasks: return []
    import h4_native_science as h
    rows,receipt=h.batch(tasks,callback,dict(worker_cpus=spec['worker_cpus'],font_seed=None),directory)
    c.save(directory/'accepted_native_receipt.json',receipt)
    return rows


def execute(spec,grant,output):
    c.authorize(spec,grant)
    stage=spec['stage']; calls=c.Calls(output,spec['calls']); cells=c.registrations()[stage]
    raw=[x for x in cells if x['method']!='B0I']
    port,m,h,reference=setup(spec,raw,output,calls)
    saved=c.read(spec['dependencies']['FP']['result']) if stage!='FP' else {}
    old={r['cell']['id']:r for r in saved.get('signals',[])}
    records={}; pending=[]
    for x in raw:
        if x['id'] in old:
            c.require(saved['preparation_identities'][x['id']]==port.preps[x['id']].preparation_hash,'REUSED_FP_PREPARATION')
            records[x['id']]=deepcopy(old[x['id']]); continue
        d=degenerate(x,port.preps[x['id']].exact_rte_lambda_r) if x['R'] else None
        if d: records[x['id']]=d
        else: pending.append(dict(id=x['id'],cell=x))
    calls.take('base_signal',len(pending)); calls.take('independent_forward_signal',len(pending))
    calls.take('new_signed_shot_endpoints',5*len(pending))
    for row in batch(spec,pending,lambda t,d:evaluate(port,t,d),output/'signal_workers'): records[row['cell']['id']]=row
    from h4_tol_main_science import overlay
    for x in cells:
        if x['method']!='B0I': continue
        if x['id'] in old: records[x['id']]=deepcopy(old[x['id']])
        else:
            calls.take('B0I_overlay'); calls.take('new_signed_shot_endpoints',5)
            records[x['id']]=overlay(records[x['id'].replace('B0I_','B0_',1)],port,m)
    ordered=[records[x['id']] for x in cells]
    c.save(output/'signals.json',dict(signals=ordered,binding=c.BINDING,reused_signal_ids=sorted(set(records)&set(old))))
    eligible={x['id'] for x in c.selected(ordered,cells)}
    pairs={}; tasks=[]
    if stage=='D':
        for row in saved['measurements']:
            if row['cell_id'] in {x['id'] for x in cells if not x['R']}:
                c.check_pair(row); pairs[row['cell_id']]=[deepcopy(row)]
    if stage in ('FP','A'):
        registry=c.read(spec['seed_registry'])
        for task in registry['tasks']:
            cid=task['cell']['id']
            if records[cid].get('status')=='DEGENERATE_EMPTY_TAIL': continue
            if stage=='A' and cid not in eligible: continue
            tasks.append(dict(task,expected=records[cid]))
    if stage!='A':
        for x in cells:
            if x['R'] or x['id'] in pairs or (stage=='D' and x['id'] not in eligible): continue
            tasks.append(dict(id=x['id'],cell=x,replica=0,expected=records[x['id']]))
    calls.take('native_evolution',len(tasks)); calls.take('wrapper_compile',2*len(tasks))
    pr=[t for t in tasks if t['cell']['R']]
    calls.take('trajectory',len(pr)); calls.take('occurrence',sum(t['cell']['q'] for t in pr))
    calls.take('RTE_event',sum(t['cell']['R'] for t in pr))
    calls.take('control_probe',4*sum(t['replica']==0 for t in tasks))
    calls.take('independent_event_oracle',sum(t['replica']==0 for t in pr))
    from h4_dp_science import compiler
    settings=compiler(m,spec['native_config'])
    callback=lambda t,d:compile_one(port,m,settings,t,d,stage)
    first=[t for t in tasks if t['replica']==0]; rest=[t for t in tasks if t['replica']!=0]
    measured=batch(spec,first,callback,output/'first_control_workers')
    measured+=batch(spec,rest,callback,output/'remaining_workers')
    c.require(len(measured)==len(tasks) and {(r['cell_id'],r['replica']) for r in measured}==
        {(t['cell']['id'],t['replica']) for t in tasks}, 'COMPLETE_NATIVE_MEASUREMENTS')
    for r in measured: pairs.setdefault(r['cell_id'],[]).append(r)
    accounting=None
    if stage!='FP':
        calls.take('endpoint_accounting_pass')
        d=c.read(spec['dependencies']['D']['result']) if stage=='A' else None
        accounting=c.account(ordered,pairs,stage,d)
    return dict(scope=c.SCOPE,stage=stage,status='COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],
        binding=c.BINDING,signals=ordered,reference=reference,measurements=measured,accounting=accounting,
        truncated_references={str(p):dict(signal=m.complex_record(z),error=e)
            for p,(z,e) in port.truncated_references.items()},
        preparation_identities={cid:p.preparation_hash for cid,p in port.preps.items()},
        calls_reserved=calls.used,calls_maximum=calls.limits,reservations_are_nonrefundable=True,
        FP_PR_cost_pooled_with_A=False,automatic_retry=False,next_stage_authorized=False)
