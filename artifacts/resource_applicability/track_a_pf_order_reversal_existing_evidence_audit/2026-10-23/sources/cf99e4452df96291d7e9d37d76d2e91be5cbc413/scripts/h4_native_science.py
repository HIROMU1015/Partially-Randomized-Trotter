"""Authorized H4 adapters; reuse the frozen native/forward/accounting implementations."""
from copy import copy, deepcopy
import csv
from fractions import Fraction
import importlib
import math
from pathlib import Path
import statistics
import time
import h4_native_contract as c
import h8_key_compile_parallel as pool


class Progress:
    def phase(self,*args,**kwargs):pass
    def update(self,*args,**kwargs):pass


def provider():
    return importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')


def accounting_plan():
    from trottertracks.resource_applicability.h48_preflight_contract_v1 import h4_plan
    plan=h4_plan('H4_SIGNAL');plan['epsilons']=c.EPS
    return plan


def decorate(record, plan):
    m=provider()
    record['precision_rows']=m.precision_rows(record,plan)
    record['noncancellation_precision_rows']=m.precision_rows(c.noncancel(record),plan)
    record['strict_bias_certificate']=None
    return record


def old_inputs(config):
    signals=c.read(config['mapping']['old_signals'])['signals']
    c.require(len(signals)==16 and len({s['cell']['id'] for s in signals})==16,'ORIGINAL_FIXED16')
    with Path(config['mapping']['old_metrics']).open() as f:rows=list(csv.DictReader(f))
    grouped={s['cell']['id']:{} for s in signals}
    for row in rows:
        c.require(row['compiler_hash']==config['binding']['compiler'],'SAVED_COMPILER_IDENTITY')
        if row['control']!='symmetric_directional':continue
        id_,rep,axis=row['cell_id'],int(row['replica']),row['axis']
        c.require(id_ in grouped and axis in ('cosine','sine'),'SAVED_COST_SCOPE')
        pair=grouped[id_].setdefault(rep,{})
        c.require(axis not in pair,'DUPLICATE_PAIRED_AXIS')
        pair[axis]=dict(RZ=int(row['RZ']),seed=row['seed'],event_digest=row['event_digest'])
    result={}
    for signal in signals:
        id_=signal['cell']['id'];pairs=grouped[id_];n=32 if signal['cell']['method']=='B2' else 1
        c.require(set(pairs)==set(range(n)),'ORIGINAL_REPLICA_COVERAGE')
        result[id_]=[]
        for rep,pair in sorted(pairs.items()):
            c.require(set(pair)=={'cosine','sine'},'PAIRED_COSINE_SINE_REQUIRED')
            a,b=pair['cosine'],pair['sine']
            c.require(a['seed']==b['seed'] and a['event_digest']==b['event_digest'],'PAIRED_TRAJECTORY_BINDING')
            result[id_].append({axis:v['RZ'] for axis,v in pair.items()})
    return signals,result


def account(records, costs, plan, *, scope):
    rows=[];comparisons=[]
    for mode,field in [('USUAL_SIGNED','precision_rows'),('EMPIRICAL_NONCANCELLATION','noncancellation_precision_rows')]:
        for eps in plan['epsilons']:
            for factor in c.FACTORS:
                endpoint=[]
                for record in records:
                    id_=record['cell']['id'];a=record[field][str(eps)][str(factor)]
                    row=dict(cell_id=id_,method=record['cell']['method'],bias_mode=mode,epsilon=eps,
                        allowance_factor=factor,eligible=a['cost_acquisition_eligible'],G_RZ_point=None,
                        G_RZ_SE=None,N_total=a['N_total'],cost_provenance='MISSING' if id_ not in costs else 'MEASURED_NATIVE_FULL_WRAPPER')
                    if a['cost_acquisition_eligible'] and id_ in costs:
                        weights={axis:a['axes'][channel]['shots'] for axis,channel in [('cosine','real'),('sine','imag')]}
                        totals=[sum(weights[axis]*pair[axis] for axis in weights) for pair in costs[id_]]
                        exact=Fraction(sum(totals),len(totals))
                        row.update(G_RZ_point=float(exact),G_RZ_exact=str(exact),
                            G_RZ_SE=statistics.stdev(totals)/math.sqrt(len(totals)) if len(totals)>1 else None,
                            n=len(totals),shots=weights,population_mean_certified=False)
                    endpoint.append(row);rows.append(row)
                det=[r for r in endpoint if r['method'] in ('B0','B0I','B1') and r['G_RZ_point'] is not None]
                pr=[r for r in endpoint if r['method']=='B2' and r['G_RZ_point'] is not None]
                missing=[r['cell_id'] for r in endpoint if r['eligible'] and r['G_RZ_point'] is None]
                if det and pr and not missing:
                    bestd=min(det,key=lambda r:(Fraction(r['G_RZ_exact']),r['cell_id']))
                    bestp=min(pr,key=lambda r:(Fraction(r['G_RZ_exact']),r['cell_id']))
                    ratio=float(Fraction(bestp['G_RZ_exact'])/Fraction(bestd['G_RZ_exact']))
                    comparisons.append(dict(bias_mode=mode,epsilon=eps,allowance_factor=factor,
                        deterministic=bestd,PR=bestp,PR_to_det_ratio=ratio,PR_reduction=1-ratio,
                        comparison_scope=scope,finite_sample_point_only=True))
                else:comparisons.append(dict(bias_mode=mode,epsilon=eps,allowance_factor=factor,
                        status='UNRESOLVED_OR_NO_ELIGIBLE_PAIR',missing_costs=missing,PR_reduction=None,comparison_scope=scope))
    return dict(rows=rows,comparisons=comparisons,scope=scope,strict_bias_certificate=None,
                population_mean_certified=False,unmeasured_cost_zero_filled=False)


def noncancellation(config,grant,output):
    records,costs=old_inputs(config);plan=accounting_plan();m=provider();count=0
    for record in records:
        reproduced=m.precision_rows(record,plan)
        for eps,variants in record['precision_rows'].items():
            for factor,saved in variants.items():
                c.require(reproduced[eps][factor]==saved,'ORIGINAL_USUAL_ACCOUNTING_REPRODUCTION')
                count+=1
    c.require(count==128,'ORIGINAL_128_ACCOUNTING_GATE')
    c.save(output/'usual_reproduction_gate.json',dict(status='PASS',precision_rows=count,paired_PR=8*32))
    records=[decorate(deepcopy(r),plan) for r in records]
    result=account(records,costs,plan,scope='ORIGINAL_H4_NATIVE_FIXED16_ONLY')
    # Verify the four saved selected costs without changing their selection.
    old=c.read(config['mapping']['old_summary'])
    for endpoint in old['comparisons']:
        for field in ('PF_point','PR_point_in_fixed_panel'):
            saved=endpoint[field]
            actual=next(r for r in result['rows'] if r['cell_id']==saved['cell_id'] and r['epsilon']==endpoint['epsilon']
                        and r['bias_mode']=='USUAL_SIGNED' and r['allowance_factor']==1.)
            c.require(actual['G_RZ_point']==saved['G_RZ_point'],'ORIGINAL_PAIRED_COST_REPRODUCTION')
    c.save(output/'paired_cost_reproduction_gate.json',dict(status='PASS',saved_selected_costs=8))
    result.update(stage='N',signals=records,new_signal=0,new_compile=0,new_sampling=0,status='COMPLETE_MANDATORY_STOP')
    return result


def setup(config,cells,output):
    m=provider()
    from trottertracks.resource_applicability.h4_common_panel_port_v1 import h4_loader
    plan=accounting_plan();plan['cells']=cells
    plan['matrix_free']['num_threads']=1
    plan['caps_proposed']['tail_matvecs_corrected_and_raw']['B2']=max([2*x['R']*(x['K']+1) for x in cells if x['R']]+[0])
    writer=c.Writer(output)
    port=m.MatchedPort(Path(config['provider_root']),dict(kind='H4_NATIVE_TARGETED',plan=plan),writer,Progress())
    port.ham,port.sector,full,state,md,_=h4_loader(Path(config['mapping']['snapshot']),
        c.read(config['mapping']['snapshot_receipt']),c.read(config['mapping']['df_receipt']))
    c.require((port.ham.n_qubits,len(port.ham.lambdas),port.sector.dimension)==(8,12,36),'H4_LAYOUT')
    c.require(md['hamiltonian_hash']==config['binding']['hamiltonian'] and md['state_hash']==config['binding']['state'],'H4_INPUT_IDENTITY')
    port.qindices,port.qstate=m.checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    port.state=state.copy();port.saved_state=state.copy()
    ref=c.read(config['mapping']['saved_reference'])
    c.require(ref['metadata']==md and ref['state_renormalized'] is False,'SAVED_REFERENCE_BINDING')
    port.reference=complex(ref['reference']['real'],ref['reference']['imag'])
    port.reference_error=ref['independent_occupation_all_columns_error']+ref['expm_eigh_agreement']['signal_difference']
    old=c.read(config['mapping']['old_signals'])['signals']
    c.require(all(x['reference']==ref['reference'] for x in old),'ORIGINAL_REFERENCE_CLOSURE')
    port.spectra={};port.native_actions={};port.tails={};port.oracle_tails={};port.truncated_references={}
    cache={}
    for cell in cells:
        key=(cell['method']=='B0',cell['prefix'])
        if key not in cache:cache[key]=(m._prepare_discard if key[0] else m._prepare)(port.ham,key[1])
        port.preps[cell['id']]=cache[key]
    # Warm only Qiskit/native blocks and tiny spectral caches. Production numba/OpenMP
    # operators are initialized inside disposable children, never in the fork parent.
    for prep in cache.values():port.actions(prep);port.actions(prep,oracle=True)
    coverage=set()
    for cell in cells:
        for i,t in m.primitive_time_schedule(cell,T=.8)['unique_primitive_times']:
            coverage.add((str(i),t))
    port.plan['coverage']['primitive_actions']=3*len(coverage)
    port.primitives()
    writer.write('input_reference_reuse.json',dict(reference=ref['reference'],new_full_H_reference=0,
        snapshot_sha256=config['binding']['snapshot'],state_regenerated=False,SCF_DF_regenerated=False,
        primitive_coverage_pairs=len(coverage),new_Gaussian_preparations=len(cache)))
    return port


def batch(tasks, callback, grant, directory):
    receipt=pool.map_owned(tasks,callback,cpus=grant['worker_cpus'],limits=dict(max_workers=len(grant['worker_cpus'])),
        directory=directory,deadline=math.inf,font_seed=grant['font_seed'])
    c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'],'OWNED_WORKERS_REAP')
    rows=[]
    for worker in receipt['workers']:
        path=Path(worker['directory'])/'task_result.json'
        c.require(worker['exit_code']==0 and worker['reaped'] and c.sha(path)==worker['result_sha256'],'WORKER_RESULT_BINDING')
        rows.append(c.read(path))
    return rows,receipt


def evaluate(port,cell,directory):
    local=copy(port);local.plan=deepcopy(port.plan);local.cells=[cell];local.plan['cells']=[cell]
    local.calls=deepcopy(port.calls);local.writer=c.Writer(directory);local.completed=0
    local.tails={};local.truncated_references={}
    local.correctness()
    row=c.read(directory/'signal_summary.json')['signals'][0]
    return decorate(row,local.plan)


def overlay(base,ham,plan):
    import h6_native_b0i_bridge_v1 as bridge
    m=provider();prefix=base['cell']['prefix']
    exact=sum((bridge.exact_identity(G,lam) for G,lam in zip(ham.g_matrices[prefix:],ham.lambdas[prefix:])),Fraction())
    C=float(exact);scalar=float(ham.constant+C)
    error=bridge.upper_float(abs(Fraction(C)-exact)+abs(Fraction(scalar)-Fraction(ham.constant)-Fraction(C)))
    phase_error=bridge.upper_float(abs(Fraction(-.8*C)+Fraction(.8)*Fraction(C)))
    effective=deepcopy(base);effective['error_components']=bridge.conservative_saved_error_components(effective)
    out=deepcopy(base)
    out.update(bridge.saved_signal_overlay(effective,C=C,C_error=error,
        rotation_roundoff=max(1e-12,64*m.np.finfo(float).eps*(1+abs(.8*C))),phase_roundoff=phase_error))
    out['cell']=dict(base['cell'],method='B0I',id=base['cell']['id'].replace('B0_','B0I_',1))
    values={k:complex(v['real'],v['imag']) for k,v in out['signals'].items()}
    out['error_decomposition']=m.differences(values,complex(out['reference']['real'],out['reference']['imag']),'B0')
    out['scalar_receipt']=dict(C=C,C_error=error,scalar=scalar,exact_numerator=str(exact.numerator),
        exact_denominator=str(exact.denominator),policy='EXACT_STORED_FULL_FOCK_TAIL_IDENTITY',certified_input_error=False)
    out['provenance']=dict(base_cell=base['cell']['id'],analytic_phase_overlay=True,new_signal_recurrence=False)
    return decorate(out,plan)


def native_control(evolution,m,qstate,expected,writer):
    n=len(qstate).bit_length()-1
    c.require(len(qstate)==2**n,'CONTROL_STATE_LAYOUT')
    low=m.np.concatenate((qstate,m.np.zeros_like(qstate)));high=m.np.concatenate((m.np.zeros_like(qstate),qstate))
    a=m.simulate_statevector(evolution.circuit,low);b=m.simulate_statevector(evolution.circuit,high)
    z=complex(m.np.vdot(qstate,b[len(qstate):]))
    errors=[float(m.np.linalg.norm(a-low)),float(m.np.linalg.norm(b[:len(qstate)])),
            abs(float(m.np.linalg.norm(b))-1),abs(z-complex(expected['real'],expected['imag']))]
    for axis,value in [('cosine',z.real),('sine',z.imag)]:
        wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=1000000)
        errors.append(abs(m.hadamard_expectation(m.simulate_statevector(wrapper,low),n)-value))
    maximum=max(errors);m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=maximum)
    c.require(math.isfinite(maximum) and maximum<1e-9,'H4_NATIVE_CONTROL_OR_SCALAR_MISMATCH')
    writer.write('control.json',dict(status='PASS',probes=4,max_error=maximum,signal=m.complex_record(z),certified=False))


def deterministic(config,grant,output):
    cells=[x for x in c.plan()['D']['cells'] if x['method']=='B0']
    port=setup(config,cells,output);m=provider()
    def signal_task(task,directory):
        base=evaluate(port,task['cell'],directory)
        return dict(B0=base,B0I=overlay(base,port.ham,port.plan))
    pairs,receipt=batch([dict(id=x['id'],cell=x) for x in cells],signal_task,grant,output/'signal_workers')
    records=[r[k] for r in pairs for k in ('B0','B0I')]
    c.require(len(records)==96,'D_SIGNAL_SCOPE')
    c.save(output/'signals.json',dict(signals=records,base_evaluations=48,analytic_overlays=48))
    original,old_costs=old_inputs(config)
    anchors={s['cell']['id']:old_costs[s['cell']['id']][0] for s in original if s['cell']['method']=='B1'}
    selected,evidence=c.select_direct(records,anchors)
    c.save(output/'selection.json',dict(selected=selected,endpoints=evidence,maximum_compile=20,
        proxy_is_validated_cost_model=False,unmeasured_optimum_certified=False))
    import qiskit
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    cp=config['compiler_options']
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    c.require(qiskit.__version__=='1.3.0' and compiler_settings_hash(compiler)==config['binding']['compiler'],'LIVE_COMPILER')
    by_id={r['cell']['id']:r for r in records}
    def compile_task(task,directory):
        record=by_id[task['id']];cell=record['cell'];writer=c.Writer(directory)
        base_id=cell['id'].replace('B0I_','B0_',1);prep=port.preps[base_id]
        scalar=record['scalar_receipt']['scalar'] if cell['method']=='B0I' else prep.constant_coefficient
        writer.write('build_reserved.json',dict(cell=cell,scalar_hex=float(scalar).hex(),control='symmetric_directional'))
        evolution=m.build_deterministic_native(prep.deterministic_blocks,num_system_qubits=8,T=.8,q=cell['q'],
            formula=cell['formula'],scalar=scalar,control_policy='symmetric_directional',max_instructions=1000000)
        native_control(evolution,m,port.qstate,record['signals']['corrected'],writer)
        rows=[]
        for axis in ('cosine','sine'):
            wrapper=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
            fingerprint=m.canonical_qiskit_circuit_fingerprint(wrapper)
            writer.write('compile_'+axis+'_reserved.json',dict(axis=axis,fingerprint=fingerprint,nonrefundable=True))
            start=time.monotonic();result=m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
            c.require(result.compiler_settings_hash==config['binding']['compiler'] and result.actual_circuit_fingerprint==fingerprint,
                'NATIVE_COMPILE_IDENTITY')
            row=dict(axis=axis,RZ=result.rz_count,CX=result.cx_count,RZ_depth=result.rz_depth,CX_depth=result.cx_depth,
                depth=result.total_depth,size=result.circuit_size,compiler_hash=result.compiler_settings_hash,
                fingerprint=fingerprint,compile_seconds=time.monotonic()-start)
            c.require(all(type(row[key]) is int and row[key]>=0 for key in ('RZ','CX','RZ_depth','CX_depth','depth','size'))
                and row['RZ']>0,'MEASURED_INTEGER_METRICS')
            writer.write('measured_'+axis+'.json',row);rows.append(row)
            del result,wrapper
        return dict(cell_id=cell['id'],rows=rows,control_probes=4)
    measured,cost_receipt=batch([dict(id=x) for x in selected],compile_task,grant,output/'compile_workers')
    c.save(output/'worker_receipts.json',dict(signal=receipt,compile=cost_receipt))
    costs=dict(old_costs)
    for record in measured:costs[record['cell_id']]=[{r['axis']:r['RZ'] for r in record['rows']}]
    # Only registered, directly measured additions enter this explicit subset comparison.
    result=account([decorate(deepcopy(r),port.plan) for r in original]+[by_id[x] for x in selected],costs,
        port.plan,scope='ORIGINAL_FIXED16_PLUS_REGISTERED_MEASURED_D_SUBSET')
    result.update(stage='D',status='COMPLETE_MANDATORY_STOP',selected=selected,measurements=measured,
        all_D_signal_count=96,unmeasured_D_count=96-len(selected),all_D_domain_minimum=None,
        actual_calls=dict(base_signal=48,analytic_B0I_overlay=48,native_compile=2*len(selected),control_probe=4*len(selected)),
        model_fit=False,model_validation_claim=False,A_cost_stage_started=False)
    return result


def boundary(config,grant,output):
    cells=c.plan()['A']['cells'];port=setup(config,cells,output)
    def callback(task,directory):return evaluate(port,task['cell'],directory)
    rows,receipt=batch([dict(id=x['id'],cell=x) for x in cells],callback,grant,output/'signal_workers')
    c.require(len(rows)==16 and {r['cell']['id'] for r in rows}=={x['id'] for x in cells},'A_EXACT_SCOPE')
    c.save(output/'worker_receipts.json',dict(signal=receipt))
    return dict(stage='A',status='COMPLETE_MANDATORY_STOP',signals=rows,signal_count=16,
        native_compile=0,new_sampling=0,new_full_H_reference=0,cost_acquisition_auto_started=False,
        strict_bias_certificate=None,global_optimum_certified=False)


def execute(stage,config,grant,output):
    grant=dict(grant,font_seed=config['font_seed'])
    if stage=='N':return noncancellation(config,grant,output)
    if stage=='D':return deterministic(config,grant,output)
    return boundary(config,grant,output)
