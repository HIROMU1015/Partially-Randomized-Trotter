"""Saved-operation native B0 adapter; every scientific API is behind the permit."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import time

import h8_b0_anchor_contract as c
import h8_b0_anchor_runtime as r
import h8_strengthened_export as exported
import track_a_phase_a_parallel as pool


def restore(export,permit,*,factory,inspect,one_type,fragment_type,n=16,rank=28):
    permit.require(c.SCOPE)
    permit.take('saved_preparation_decode')
    blocks=[]
    for i,row in enumerate(export['faithful:p'+str(rank)]['blocks']):
        exported.validate_block(row,i,n)
        operations=tuple((factory(o['name'],tuple(float(p) for p in o['parameters'])),tuple(o['qubits']))
                         for o in row['operations'])
        described,metadata=inspect(operations)
        if metadata!=row['operations']:raise ValueError('RESTORED_BASIS_MATRIX_OR_ORDER_CHANGED')
        common=dict(block_id=row['block_id'],basis_id=row['basis_id'],basis_hash=row['basis_hash'],
            basis_change_operations=described,num_system_qubits=n,order_index=i,runtime_basis_operations=operations)
        values=tuple(exported.real_hex(x) for x in row['diagonal_hex'])
        blocks.append(one_type(**common,block_kind='one_body',original_fragment_index=None,rank=None,
            diagonal_eigenvalues=values) if i==0 else fragment_type(**common,block_kind='df_fragment',
            original_fragment_index=i-1,rank=i-1,diagonal_eta=values,lam=exported.real_hex(row['lambda_hex'])))
    if len(blocks)!=rank+1:raise ValueError('RESTORED_BLOCK_COUNT')
    return tuple(blocks)


def observe_prefix(blocks,export,prefix,expected,inspect,*,n=16):
    observed=[]
    for i,b in enumerate(blocks[:prefix+1]):
        _,ops=inspect(b.runtime_basis_operations)
        h=c.digest(dict(num_system_qubits=n,operations=ops,hash_policy='ordered_local_basis_operations_v1'))
        observed.append(dict(block_id=b.block_id,basis_id='df-basis-'+h,basis_hash=h,operations=ops,
            diagonal_hex=[float(v).hex() for v in (b.diagonal_eigenvalues if i==0 else b.diagonal_eta)],
            lambda_hex=None if i==0 else float(b.lam).hex()))
    root=export['discard:p'+str(prefix)]
    if (observed!=root['blocks'] or c.digest(observed)!=expected['ordered_block_description_sha256'] or
            root['preparation_hash']!=expected['saved_preparation_hash'] or
            root['constant_hex']!=expected['constant_hex'] or len(observed)!=expected['block_count']):
        raise ValueError('COST_RUNTIME_PREPARATION_IDENTITY')
    return deepcopy(expected)


def build_case(case,blocks,scalar,qstate,saved,permit,native,m,*,n=16,cap=4_000_000):
    permit.require(c.SCOPE)
    permit.take('native_evolution_build')
    evolution=native.build_deterministic_native(blocks,num_system_qubits=n,T=.8,q=case['q'],
        formula='2nd',scalar=scalar,control_policy='symmetric_directional',max_instructions=cap)
    np=m.np;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
    permit.take('control_probe');a=m.simulate_statevector(evolution.circuit,low)
    permit.take('control_probe');b=m.simulate_statevector(evolution.circuit,high)
    overlap=complex(np.vdot(qstate,b[len(qstate):]))
    targets=[complex(saved[k]['real'],saved[k]['imag']) for k in ('corrected','independent_forward')]
    errors=[float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(qstate)])),
            *(abs(overlap-t) for t in targets)]
    wrappers={};fingerprints={}
    for axis in ('cosine','sine'):
        permit.take('measured_wrapper_build')
        wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=cap)
        permit.take('probe_wrapper_copy');probe=wrapper.remove_final_measurements(inplace=False)
        permit.take('control_probe')
        measured=m.hadamard_expectation(m.simulate_statevector(probe,low),n)
        errors.append(abs(measured-(overlap.real if axis=='cosine' else overlap.imag)))
        wrappers[axis]=wrapper;fingerprints[axis]=m.canonical_qiskit_circuit_fingerprint(wrapper)
        del probe
    if any(not math.isfinite(e) or e<0 or e>=1e-8 for e in errors):raise ValueError('CONTROL_VALIDATION_1E8')
    return wrappers,fingerprints,max(errors)


def compile_wrapper(wrapper,fingerprint,m,compiler,*,cap,before_compile=lambda:None):
    if m.canonical_qiskit_circuit_fingerprint(wrapper)!=fingerprint:raise ValueError('WRAPPER_CHANGED_BEFORE_COMPILE')
    started=time.monotonic()
    before_compile()
    cost=m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
    after=m.canonical_qiskit_circuit_fingerprint(wrapper)
    if (cost.compiler_settings_hash!=c.COMPILER or cost.actual_circuit_fingerprint!=fingerprint or after!=fingerprint
            or len(cost.transpiled_circuit.data)>cap):raise ValueError('COMPILER_WRAPPER_IDENTITY_OR_CAP')
    if type(cost.rz_count)is not int or cost.rz_count<=0:raise ValueError('NONPOSITIVE_MEASURED_RZ')
    return dict(measured_RZ=cost.rz_count,CX=cost.cx_count,RZ_depth=cost.rz_depth,CX_depth=cost.cx_depth,
        depth=cost.total_depth,size=cost.circuit_size,fingerprint_before=fingerprint,fingerprint_after=after,
        compiler_hash=cost.compiler_settings_hash,compile_wall_seconds=time.monotonic()-started)


def run_waves(plan,permit,writer,build,compile_axis,*,map_owned=pool.map_owned,font_seed=None):
    allrows=[];prior_cpu=0.;rsc=plan['resources'];by_id={x['id']:x for x in plan['cases']}
    for number,ids in enumerate(plan['waves']):
        permit.require(c.SCOPE)
        # Finish BOTH case validations before admitting ANY compile in a wave.
        prepared={cid:build(by_id[cid]) for cid in ids}
        tasks=[dict(id=cid+'_'+axis,case_id=cid,axis=axis,counts={'compile':1}) for cid in ids for axis in ('cosine','sine')]
        def callback(task,directory):
            return compile_axis(task,prepared[task['case_id']],directory)
        limits=dict(max_workers=4,AS_bytes=rsc['worker_AS_bytes'],RSS_bytes=rsc['worker_RSS_bytes'],
            worker_CPU_seconds=rsc['worker_CPU_seconds'],aggregate_CPU_seconds=rsc['aggregate_CPU_seconds'])
        directory=permit.output/('wave_'+str(number+1))
        receipt=map_owned(tasks,callback,cpus=permit.grant['worker_cpus'],limits=limits,directory=directory,
            deadline=permit.started+rsc['wall_seconds'],guard=permit.guard,reserve=permit.reserve,
            font_seed=font_seed,prior_child_CPU_seconds=prior_cpu)
        if receipt['status']!='PASS' or receipt['all_reaped'] is not True:raise ValueError('WAVE_NOT_REAPED')
        workers={w['task_id']:w for w in receipt['workers']}
        if set(workers)!={t['id'] for t in tasks}:raise ValueError('EXACT_WAVE_WORKERS')
        for task in tasks:
            worker=workers[task['id']]
            if worker['exit_code']!=0 or not worker['reaped']:raise ValueError('WORKER_NATIVE_EXIT')
            path=Path(worker['directory'])/'task_result.json';raw=path.read_bytes()
            h=__import__('hashlib').sha256(raw).hexdigest()
            if worker.get('result_sha256',h)!=h:raise ValueError('WORKER_RESULT_CHANGED')
            row=json.loads(raw)
            if (row['case_id'],row['axis'])!=(task['case_id'],task['axis']):raise ValueError('TASK_RESULT_IDENTITY')
            row.update(native_exit_code=0,reaped=True,worker_identity=worker['identity'],result_sha256=h)
            permit.settle(task['id']);allrows.append(row)
        prior_cpu=receipt['child_CPU_seconds']
        writer.write('wave_'+str(number+1)+'_accepted.json',dict(rows=allrows[-4:],all_reaped=True,
            child_CPU_seconds=prior_cpu,next_stage_authorized=False))
        del prepared
    return allrows


def evaluate(config,permit,writer):
    permit.require(c.SCOPE)  # Before any scientific import or molecular loader.
    plan=c.plan();paths=config['input_mapping']
    m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    loader=importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1')
    native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    specs=importlib.import_module('trotterlib.df_partial_s2');basis=importlib.import_module('trotterlib.df_rte_tail')
    gates=importlib.import_module('qiskit.circuit.library');qiskit=importlib.import_module('qiskit')
    for module in (m,loader,native,specs,basis):
        if not Path(module.__file__).resolve().is_relative_to(Path(config['provider_root'])/'src'):
            raise ValueError('SCIENTIFIC_PROVIDER_IMPORT_ORIGIN')
    permit.take('snapshot_load')
    ham,sector,full,state,md,_=loader.load_h8_snapshot(Path(paths['h8_input_snapshot.npz']),
        r.read(paths['snapshot_receipt.json']),r.read(paths['df_receipt.json']))
    if ((ham.n_qubits,len(ham.lambdas),sector.dimension)!=(16,28,4900) or
            md['hamiltonian_hash']!=config['scientific_binding']['hamiltonian'] or
            md['state_hash']!=config['scientific_binding']['state']):raise ValueError('H8_LAYOUT_INPUT_IDENTITY')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    export=r.read(paths['h8_preparation_export'])
    exported.validate_export(export,expected_hashes=config['expected_preparation_hashes'])
    def factory(name,params):return {'rz':gates.RZGate,'xx_plus_yy':gates.XXPlusYYGate}[name](*params)
    def inspect(ops):
        d=basis.describe_basis_change_operations(ops)
        return d,json.loads(json.dumps([asdict(x) for x in d]))
    blocks=restore(export,permit,factory=factory,inspect=inspect,
        one_type=specs.DFDeterministicOneBodySpec,fragment_type=specs.DFDeterministicFragmentSpec)
    if any(float(ham.lambdas[i]).hex()!=b.lam.hex() for i,b in enumerate(blocks[1:])):
        raise ValueError('BLOCK_LAMBDA_IDENTITY')
    views={}
    for p in (14,21):
        permit.take('prefix_view');expected=plan['binding']['prefix_preparations'][str(p)]
        observed=observe_prefix(blocks,export,p,expected,inspect)
        if float(ham.constant).hex()!=expected['constant_hex']:raise ValueError('SCALAR_IDENTITY')
        views[p]=(blocks[:p+1],float.fromhex(expected['constant_hex']),observed)
    targets={row['case_id']:row for row in r.read(c.ROOT/c.TARGET_PATH)['cases']}
    cp=plan['compiler_options']
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    def build(case):
        bs,scalar,identity=views[case['prefix']]
        wrappers,fingerprints,error=build_case(case,bs,scalar,qstate,targets[case['id']],permit,native,m)
        writer.write(case['id']+'_control_preparation.json',dict(case=case,preparation_binding=identity,
            observed_ordered_block_description_sha256=identity['ordered_block_description_sha256'],
            basis_observed_from_restored_live_operations=True,scalar_policy=plan['scalar_policy'],
            max_control_error=error,wrapper_fingerprints=fingerprints,profile_sha256=c.PROFILE,SOURCE=config['SOURCE']))
        return dict(wrappers=wrappers,fingerprints=fingerprints,error=error,case=case,identity=identity)
    def compile_axis(task,built,directory):
        def entered():
            r.Writer(directory).write('compile_entered.json',dict(task=task,SOURCE=config['SOURCE'],scope=c.SCOPE,
                profile_sha256=c.PROFILE,preparation_binding=built['identity'],compiler_hash=c.COMPILER,
                fingerprint_before=built['fingerprints'][task['axis']],budget_reserved_in_parent=True,
                API_call_entered_not_completion=True))
        row=compile_wrapper(built['wrappers'][task['axis']],built['fingerprints'][task['axis']],m,compiler,
            cap=plan['transpiled_cap'],before_compile=entered)
        row.update(case_id=task['case_id'],axis=task['axis'],role=built['case']['role'],SOURCE=config['SOURCE'],
            scope=c.SCOPE,profile_sha256=c.PROFILE,control=plan['control'],scalar_policy=plan['scalar_policy'],
            preparation_binding=built['identity'],max_control_error=built['error'])
        return row
    rows=run_waves(plan,permit,writer,build,compile_axis,font_seed=config['font_seed'])
    for role in ('TRAIN','HOLDOUT'):
        writer.write(role.lower()+'_measurements.json',dict(role=role,axes=[x for x in rows if x['role']==role],fit_performed=False))
    return rows
