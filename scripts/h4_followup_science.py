"""Grant-only adapters reusing the sealed H4 native/conditional/accounting paths."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import statistics
import time
import h4_followup_contract as c
import h4_native_science as h


def setup(config,cells,T,output,grant,*,tol=False):
    m=h.provider();p=h.accounting_plan();p.update(T=T,cells=cells)
    p['matrix_free']['num_threads']=1
    cap=p['caps_proposed'];cap.update(trajectory=64,occurrence=512,compile=144,control_probe=40)
    cap['tail_matvecs_corrected_and_raw']['B2']=max([2*x['R']*(x['K']+1) for x in cells if x['R']]+[0])
    # Primitive identities are deduplicated by actual fragment and physical time.
    coverage={('one' if i==0 else str(i-1),t) for x in cells
              for i,t in m.primitive_time_schedule(x,T=T)['unique_primitive_times']}
    p['coverage']['primitive_actions']=cap['primitive']=3*len(coverage)
    port=m.MatchedPort(Path(config['provider_root']),dict(kind=c.SCOPE,plan=p),c.Writer(output),h.Progress())
    if tol:
        from h4_tol_only_input import load
        port.ham,port.sector,full,state,md=load(config)
    else:
        from trottertracks.resource_applicability.h4_common_panel_port_v1 import h4_loader
        port.ham,port.sector,full,state,md,_=h4_loader(Path(config['mapping']['snapshot']),
            c.read(config['mapping']['snapshot_receipt']),c.read(config['mapping']['df_receipt']))
    c.require(port.ham.n_qubits==8 and port.sector.dimension==36,'H4_LAYOUT')
    port.qindices,port.qstate=m.checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    port.state=state.copy();port.saved_state=state.copy()
    port.spectra={};port.native_actions={};port.tails={};port.oracle_tails={};port.truncated_references={}
    if T==.8 and not tol:
        ref=c.read(config['mapping']['saved_reference']);c.require(md==ref['metadata'],'REFERENCE_BINDING')
        port.reference=complex(ref['reference']['real'],ref['reference']['imag'])
        port.reference_error=ref['independent_occupation_all_columns_error']+ref['expm_eigh_agreement']['signal_difference']
        c.save(output/'reference.json',dict(T=T,reused=True,reference=ref['reference'],metadata=md))
    else:
        def reference(task,directory):
            op,_=m.df_linear_operator(port.ham,port.sector,**p['matrix_free'])
            matrix,count=m.bounded_sector_matrix(lambda v:op@v,36,port.calls,per_action_cap=36)
            error=max(float(m.np.linalg.norm(matrix[:,i]-m.occupation_column(port.ham,port.sector.basis_indices,int(ix))))
                      for i,ix in enumerate(port.sector.basis_indices))
            z,pair=m.reference_pair(matrix,state,T)
            c.require(error<1e-9 and pair['signal_difference']<1e-9,'REFERENCE_ORACLE_MISMATCH')
            return dict(T=T,reference=m.complex_record(z),independent_column_error=error,pair=pair,
                        error=error+pair['signal_difference'],matvecs=count,metadata=md,reused=False)
        refs,receipt=h.batch([dict(id='reference')],reference,grant,output/'reference_worker')
        ref=refs[0];port.reference=complex(ref['reference']['real'],ref['reference']['imag']);port.reference_error=ref['error']
        c.save(output/'reference.json',ref);c.save(output/'reference_worker_receipt.json',receipt)
    cache={}
    for x in cells:
        key=(x['method']=='B0',x['prefix'])
        if key not in cache:cache[key]=(m._prepare_discard if key[0] else m._prepare)(port.ham,key[1])
        port.preps[x['id']]=cache[key]
    for prep in cache.values():port.actions(prep);port.actions(prep,oracle=True)
    port.primitives()
    c.save(output/'preparations.json',dict(T=T,metadata=md,rows=[dict(cell=x,hash=port.preps[x['id']].preparation_hash,
        partition=port.preps[x['id']].partition_hash) for x in cells],new_input=tol))
    return port


def overlay(record,port):
    """Same empirical budget policy as H6 fixed-delta, generic H4 prefix/time."""
    if port.plan['T']==.8:return h.overlay(record,port.ham,port.plan)
    import cmath
    import h6_native_b0i_bridge_v1 as bridge
    T=port.plan['T'];p=record['cell']['prefix'];ham=port.ham
    exact=sum((bridge.exact_identity(g,w) for g,w in zip(ham.g_matrices[p:],ham.lambdas[p:])),Fraction())
    C=float(exact);scalar=float(ham.constant+C)
    error=bridge.upper_float(abs(Fraction(C)-exact)+abs(Fraction(scalar)-Fraction(ham.constant)-Fraction(C)))
    out=deepcopy(record);out['cell']=c.cell('B0I',p,record['cell']['q'],record['cell']['formula'])
    phase=cmath.exp(-1j*T*C)
    for key in ('corrected','exact_truncated'):
        z=record['signals'][key];out['signals'][key]=h.provider().complex_record(phase*complex(z['real'],z['imag']))
    u=record['allowance'];co,si=abs(math.cos(T*C)),abs(math.sin(T*C))
    for axis,other in [('real','imag'),('imag','real')]:out['allowance'][axis]=co*u[axis]+si*u[other]+u[axis]+T*error+1e-12
    values={k:complex(v['real'],v['imag']) for k,v in out['signals'].items()}
    out['error_decomposition']=h.provider().differences(values,port.reference,'B0')
    out['scalar_receipt']=dict(C=C,C_error=error,scalar=scalar,T=T,exact_numerator=str(exact.numerator),exact_denominator=str(exact.denominator))
    out['provenance']=dict(base_cell=record['cell']['id'],analytic_phase_overlay=True,reference_rotated=False)
    for key in ('independent_forward_signals','signal_discrepancies','state_discrepancies','B_raw_corrected_difference'):out.pop(key,None)
    return h.decorate(out,port.plan)


def event_oracle(steps,m,qstate):
    from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import explicit_event_action
    from trottertracks.resource_applicability.ax2b_native_df_v5 import make_native_block_action
    prep=steps[0].preparation;v=qstate.copy()
    actions=[make_native_block_action(b,max_instructions=1000000) for b in prep.deterministic_blocks]
    for step in steps:
        for action in actions:v=action(v,step.step_time/2)
        for event in step.rte_occurrence.events:v=explicit_event_action(v,event,prep.rte_preparation.basis_registry)
        for action in reversed(actions):v=action(v,step.step_time/2)
    v*=m.np.exp(-1j*(prep.constant_coefficient+prep.extracted_identity_coefficient)*sum(s.step_time for s in steps))
    return complex(m.np.vdot(qstate,v))


def compile_task(port,config,task,directory):
    m=h.provider();cell=task['cell'];writer=c.Writer(directory);T=port.plan['T']
    raw=cell['id'].replace('B0I_','B0_',1);prep=port.preps[raw]
    writer.write('build_reserved.json',dict(task=task,nonrefundable=True))
    if cell['R']:
        if task.get('stratum'):
            import h6_dba_sampling as sampling
            steps,receipt=sampling.steps(prep,cell,task['stratum'],task['seed'])
        else:
            steps=port.sampled_steps(cell,task['seed'])
            receipt=dict(master_seed=task['seed'],events=[[e.to_dict() for e in s.rte_occurrence.events] for s in steps])
            receipt['event_digest']=c.digest(receipt['events'])
        writer.write('trajectory.json',receipt)
        evolution=m.partial_native_from_step_requests(steps,control_policy='symmetric_directional',max_instructions=1000000)
    else:
        steps=None;receipt=dict(event_digest=c.digest(None))
        scalar=task['expected'].get('scalar_receipt',{}).get('scalar',prep.constant_coefficient)
        evolution=m.build_deterministic_native(prep.deterministic_blocks,num_system_qubits=8,T=T,q=cell['q'],formula=cell['formula'],
            scalar=scalar,control_policy='symmetric_directional',max_instructions=1000000)
    if task['replica']==0:
        expected=m.complex_record(event_oracle(steps,m,port.qstate)) if steps else task['expected']['signals']['corrected']
        h.native_control(evolution,m,port.qstate,expected,writer)
    import qiskit
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    cp=config['compiler_options'];compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    c.require(qiskit.__version__=='1.3.0' and compiler_settings_hash(compiler)==config['binding']['compiler'],'COMPILER_IDENTITY')
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    rows=[]
    with use_fingerprint_reuse():
        for axis in ('cosine','sine'):
            if steps:
                events=[[e.to_dict() for e in s.rte_occurrence.events] for s in steps]
                bound_events=[e for occurrence in events for e in occurrence] if task.get('stratum') else events
                c.require(c.digest(bound_events)==receipt['event_digest'],'PAIRED_EVENTS_CHANGED')
            wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
            fingerprint=m.canonical_qiskit_circuit_fingerprint(wrapper)
            writer.write('compile_'+axis+'_reserved.json',dict(fingerprint=fingerprint,nonrefundable=True,event_digest=receipt['event_digest']))
            start=time.monotonic();result=m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
            c.require(result.compiler_settings_hash==config['binding']['compiler'] and result.actual_circuit_fingerprint==fingerprint,'COMPILED_IDENTITY')
            metrics={k:getattr(result,v) for k,v in dict(RZ='rz_count',CX='cx_count',RZ_depth='rz_depth',CX_depth='cx_depth',depth='total_depth',size='circuit_size').items()}
            c.require(all(type(v) is int and v>=0 for v in metrics.values()) and metrics['RZ']>0,'MEASURED_INTEGER_METRICS')
            row=dict(axis=axis,metrics=metrics,compiler_hash=result.compiler_settings_hash,fingerprint=fingerprint,seconds=time.monotonic()-start)
            writer.write('measured_'+axis+'.json',row);rows.append(row)
            del result,wrapper
    return dict(cell_id=cell['id'],replica=task['replica'],seed=task.get('seed'),stratum=task.get('stratum'),rows=rows,event_digest=receipt['event_digest'],
                control_probes=4 if task['replica']==0 else 0)


def signal_cells(port,cells,grant,output):
    def callback(task,directory):
        record=h.evaluate(port,task['cell'],directory)
        return dict(base=record,overlay=overlay(record,port) if task['cell']['method']=='B0' else None)
    pairs,receipt=h.batch([dict(id=x['id'],cell=x) for x in cells],callback,grant,output/'signal_workers')
    records=[r for pair in pairs for r in (pair['base'],pair['overlay']) if r is not None]
    c.save(output/'signals.json',dict(signals=records));c.save(output/'signal_receipt.json',receipt)
    return records


def boundary(config,grant,output):
    cells=c.plan()['R']['cells'];port=setup(config,cells,.8,output,grant)
    records=signal_cells(port,cells,grant,output)
    return dict(stage='R',status='COMPLETE_MANDATORY_STOP',signals=records,actual_calls=dict(signal=12,compile=0,sampling=0),
                cost_advantage_evaluated=False,global_optimum_certified=False)


def checked_tasks(tasks,config,output):
    import csv
    with Path(config['mapping']['old_metrics']).open() as f:occupied={int(r['seed']) for r in csv.DictReader(f) if r['seed'] not in ('','None')}
    seeds=[t['seed'] for t in tasks if t.get('seed') is not None]
    c.require(len(seeds)==len(set(seeds)) and not (set(seeds)&occupied),'OLD_OR_NEW_SEED_COLLISION')
    c.save(output/'seed_registry.json',dict(tasks=tasks,namespace=c.SCOPE,legacy_master_collision_checked=True,
        candidate_pairing_assumed=False,old_exploration_confirmation_mixed=False))


def run_compile(port,config,tasks,grant,output):
    checked_tasks(tasks,config,output)
    def callback(task,directory):return compile_task(port,config,task,directory)
    first,r1=h.batch([t for t in tasks if t['replica']==0],callback,grant,output/'first_control_workers')
    rest=[t for t in tasks if t['replica']!=0]
    more,r2=h.batch(rest,callback,grant,output/'remaining_workers') if rest else ([],None)
    c.save(output/'compile_receipts.json',dict(first=r1,remaining=r2))
    rows=first+more;c.require(len(rows)==len(tasks),'EXACT_COMPILE_COVERAGE')
    return rows


def rare(config,grant,output):
    groups=c.plan()['B']['groups'];cells=list({x['cell']['id']:x['cell'] for x in groups}.values())
    port=setup(config,cells,.8,output,grant)
    tasks=[dict(id=g['id']+'_'+str(i),cell=g['cell'],stratum=g['stratum'],replica=i,
                seed=c.seed('B',g['id'],i,config['binding']['hamiltonian'])) for g in groups for i in range(8)]
    rows=run_compile(port,config,tasks,grant,output);summaries=[]
    for g in groups:
        for axis in ('cosine','sine'):
            v=[r['metrics']['RZ'] for t in rows if t['cell_id']==g['cell']['id'] and t['stratum']==g['stratum'] for r in t['rows'] if r['axis']==axis]
            c.require(len(v)==8,'RARE_GROUP_N8')
            summaries.append(dict(group=g['id'],axis=axis,n=8,mean=statistics.mean(v),sample_SE=statistics.stdev(v)/math.sqrt(8),min=min(v),max=max(v)))
    return dict(stage='B',status='COMPLETE_MANDATORY_STOP',groups=groups,measurements=rows,summaries=summaries,
        actual_calls=dict(trajectory=32,compile=64,control_probe=16),conditional_only=True,
        unconditional_reaccounting=False,population_mean_certified=False,unknown_cost_zero_filled=False)


def saved_panel(config,*,policy=False):
    records,costs=h.old_inputs(config);d=c.read(config['mapping']['saved_D'])
    end=c.read(config['mapping']['saved_D_native_exit']);gate=c.read(config['mapping']['saved_D_final_gate'])
    c.require(end['exit_code']==0 and end['wait4_complete'] and end['same_parent_reaped'] and
        gate['status']=='PASS' and end['SOURCE']==gate['SOURCE'] and d['status']=='COMPLETE_MANDATORY_STOP','D_RESULT_NORMAL_COMPLETION')
    # D full result keeps accounting rows; the separate immutable signals file is authoritative.
    records=h.old_inputs(config)[0]+c.read(config['mapping']['saved_D_signals'])['signals']
    for row in d['measurements']:costs[row['cell_id']]=[{r['axis']:r['RZ'] for r in row['rows']}]
    cells=c.panel(4,12,policy=policy);byid={r['cell']['id']:r for r in records}
    c.require(all(x['id'] in byid and x['id'] in costs for x in cells),'LEGACY_PANEL_REUSE')
    return [byid[x['id']] for x in cells],{x['id']:costs[x['id']] for x in cells}


def panel_run(config,grant,output,T,q,*,tol=False):
    rank=c.read(config['mapping']['tol_receipt'])['rank'] if tol else 12
    cells=c.panel(q,rank,policy=tol)
    raw={x['id'].replace('B0I_','B0_',1):dict(x,method='B0',id=x['id'].replace('B0I_','B0_',1)) if x['method']=='B0I' else x for x in cells}
    port=setup(config,list(raw.values()),T,output,grant,tol=tol)
    records=signal_cells(port,list(raw.values()),grant,output);byid={r['cell']['id']:r for r in records}
    tasks=[]
    for x in cells:
        for i in range(32 if x['R'] else 1):
            tasks.append(dict(id=x['id']+'_'+str(i),cell=x,replica=i,expected=byid[x['id']],
                seed=c.seed('FP' if tol else 'T',[T,x['id']],i,config['binding']['hamiltonian']) if x['R'] else None))
    measured=run_compile(port,config,tasks,grant,output);costs={x['id']:[] for x in cells}
    for row in sorted(measured,key=lambda x:x['replica']):costs[row['cell_id']].append({r['axis']:r['metrics']['RZ'] for r in row['rows']})
    result=h.account([byid[x['id']] for x in cells],costs,port.plan,scope='PREDECLARED_FIVE_CELL_PANEL_ONLY')
    result.update(T=T,q=q,delta=T/q,signals=[byid[x['id']] for x in cells],measurements=measured,
        input_policy='TOL_ONLY_REGENERATED_STATE' if tol else 'LEGACY_FIXED_RANK12_UNCHANGED',rank=rank,
        actual_calls=dict(compile=2*len(tasks),trajectory=sum(x['cell']['R'] is not None for x in tasks)))
    return result


def execute(stage,config,grant,output):
    grant=dict(grant,font_seed=config['font_seed'])
    if stage=='F':
        from h4_tol_only_input import generate
        return generate(config,grant,output)
    if stage=='R':return boundary(config,grant,output)
    if stage=='B':return rare(config,grant,output)
    old,costs=saved_panel(config,policy=stage=='FP');p=h.accounting_plan()
    saved=h.account([h.decorate(deepcopy(r),p) for r in old],costs,p,scope='SAVED_LEGACY_FIVE_CELL_PANEL_ONLY')
    if stage=='FP':return dict(stage=stage,status='COMPLETE_MANDATORY_STOP',legacy=saved,new=panel_run(config,grant,output,.8,4,tol=True),
        limitation='new SCF orbitals and saved state; differences are not attributed solely to DF truncation')
    panels=[]
    for T,q in ((.4,2),(1.6,8)):
        directory=output/('T_'+str(T));directory.mkdir();panels.append(panel_run(config,grant,directory,T,q))
    return dict(stage='T',status='COMPLETE_MANDATORY_STOP',saved_T08=saved,panels=panels,
        interpretation='fixed delta .2; T and accumulated outer steps vary together; not fixed-q comparison',new_T08_computation=0)
