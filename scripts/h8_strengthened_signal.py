"""H8 sparse signal adapter; numerical imports occur only after permit.require.

Reuse the fixed provider's primitive, spin, CSR, scalar and allowance policies.
No SCF/DF/state solver, new full-H exponential, PR law, sampler or compiler.
"""
from copy import copy,deepcopy
from dataclasses import asdict
from fractions import Fraction
import importlib
import math
from pathlib import Path

import h8_strengthened_contract as contract
import h8_strengthened_export as exported
import h8_strengthened_runtime as runtime
import track_a_phase_a_parallel as pool


def saved_reference(receipt,binding):
    md=receipt['metadata']
    if (md['hamiltonian_hash']!=binding['hamiltonian'] or md['state_hash']!=binding['state'] or
            receipt['state_renormalized'] is not False or
            receipt['numerical_allowance_certified'] is not False):
        raise ValueError('SAVED_REFERENCE_IDENTITY')
    z=complex(receipt['reference']['real'],receipt['reference']['imag'])
    errors=receipt['production_CSR_probe_errors']
    difference=receipt['reference_agreement']['signal_difference']
    if (not math.isfinite(z.real+z.imag) or len(errors)!=3 or
            any(not math.isfinite(e) or e<0 or e>1e-10 for e in errors) or
            not math.isfinite(difference) or not 0<=difference<=1e-10):
        raise ValueError('SAVED_REFERENCE_ERROR_RECEIPT')
    return z,max(errors)+difference


def checked_scalar(saved,prefix,ham,overlay,permit):
    if prefix==21:
        row=saved
        if row.get('full_Fock_identity') is not True or row.get('new_scientific_certificate') is not False:
            raise ValueError('SAVED_P21_SCALAR_POLICY')
        exact=Fraction(int(row['exact_stored_scalar_numerator']),int(row['exact_stored_scalar_denominator']))
        C=row['C'];error=row['scalar_error_upper']
        if (not math.isfinite(C) or not math.isfinite(error) or error<0 or
                Fraction(error)<abs(Fraction(C)-exact)):
            raise ValueError('SAVED_P21_SCALAR_ARITHMETIC')
        return C,error,dict(policy='REUSED_SAVED_P21_FULL_FOCK_SCALAR',receipt=row)
    permit.take('tail_scalar')
    exact=sum((overlay.exact_identity(G,lam) for G,lam in
               zip(ham.g_matrices[prefix:],ham.lambdas[prefix:])),Fraction())
    C=float(exact);error=overlay.upper_float(abs(Fraction(C)-exact))
    return C,error,dict(policy='EXACT_STORED_BINARY64_TAIL_IDENTITY',C=C,
        numerator=str(exact.numerator),denominator=str(exact.denominator),scalar_error_upper=error,
        input_operator_error_certificate=False)


def overlay_record(record,scalar,port,permit,overlay,m):
    permit.take('B0I_analytic_overlays')
    C,C_error,receipt=scalar
    effective=deepcopy(record)
    effective['error_components']=overlay.conservative_saved_error_components(effective)
    rounding=max(1e-12,64*m.np.finfo(float).eps*(1+abs(.8*C)))
    phase_error=overlay.upper_float(abs(Fraction(-.8*C)+Fraction(.8)*Fraction(C)))
    out=overlay.saved_signal_overlay(effective,C=C,C_error=C_error,
            rotation_roundoff=rounding,phase_roundoff=phase_error)
    out['cell']=dict(record['cell'],method='B0I',id=record['cell']['id'].replace('B0_','B0I_',1))
    out['scalar_receipt']=receipt
    out['error_decomposition']=m.differences(
        {k:complex(v['real'],v['imag']) for k,v in out['signals'].items()},port.reference,'B0')
    out['precision_rows']=m.precision_rows(out,port.plan)
    out['independent_forward_signals']={k:dict(real=(z:=complex(v['real'],v['imag'])*
        m.np.exp(-1j*.8*C)).real,imag=z.imag) for k,v in record['independent_forward_signals'].items()}
    return out


def run_batches(config,permit,writer,port,callback):
    receipts=[];outputs=[]
    limits=dict(permit.grant['resources'],AS_bytes=permit.grant['resources']['worker_AS_bytes'],
                RSS_bytes=permit.grant['resources']['worker_RSS_bytes'])
    for tasks,name in zip(contract.tasks(),('primitive_workers','prefix_workers')):
        receipt=pool.map_owned(tasks,callback,cpus=permit.grant['worker_cpus'],limits=limits,
            directory=writer.directory/name,deadline=permit.started+limits['wall_seconds'],
            guard=permit.guard,reserve=permit.reserve,font_seed=config['font_seed'],
            prior_child_CPU_seconds=permit.child_CPU_seconds)
        permit.child_CPU_seconds=receipt['child_CPU_seconds']
        receipts.append(receipt)
        for worker in receipt['workers']:
            directory=Path(worker['directory']);result=runtime.read(directory/'task_result.json')
            permit.settle(worker['task_id'],result['calls'])
            outputs.append((directory,result))
    return receipts,outputs


def evaluate(config,permit,writer):
    permit.require('signal')
    # All scientific imports remain behind the SOURCE/input/runtime/grant gate.
    m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    loader=importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1')
    core=importlib.import_module('trottertracks.resource_applicability.h8_resource_map_port_v1')
    sparse=importlib.import_module('trottertracks.resource_applicability.h8_sparse_reference_v1')
    molecular=importlib.import_module('trottertracks.resource_applicability.ax2b_molecular_ports_v3')
    schedules=importlib.import_module('trottertracks.resource_applicability.ax2b_h6_contract')
    specs=importlib.import_module('trotterlib.df_partial_s2')
    basis=importlib.import_module('trotterlib.df_rte_tail')
    gates=importlib.import_module('qiskit.circuit.library')
    overlay=importlib.import_module('h6_native_b0i_bridge_v1')
    np=m.np;paths=config['input_mapping']
    permit.take('snapshot_load')
    ham,sector,full,state,md,_=loader.load_h8_snapshot(Path(paths['h8_input_snapshot.npz']),
                runtime.read(paths['snapshot_receipt.json']),runtime.read(paths['df_receipt.json']))
    if ((ham.n_qubits,len(ham.lambdas),sector.dimension)!=(16,28,4900) or
            md['hamiltonian_hash']!=config['binding']['hamiltonian'] or
            md['state_hash']!=config['binding']['state']):raise ValueError('H8_FIXED_LAYOUT_IDENTITY')
    export=runtime.read(paths['h8_preparation_export'])
    audit=exported.validate_export(export,expected_hashes=config['expected_preparation_hashes'])
    def factory(name,params):return {'rz':gates.RZGate,'xx_plus_yy':gates.XXPlusYYGate}[name](*params)
    def inspect(ops):
        described=basis.describe_basis_change_operations(ops)
        return described,json_roundtrip([asdict(x) for x in described])
    blocks=exported.restore_blocks(export,permit,gate_factory=factory,inspect_operations=inspect,
        one_type=specs.DFDeterministicOneBodySpec,fragment_type=specs.DFDeterministicFragmentSpec)
    for i,b in enumerate(blocks[1:]):
        if float(ham.lambdas[i]).hex()!=b.lam.hex():raise ValueError('SAVED_BLOCK_LAMBDA_IDENTITY')
    if float(ham.constant).hex()!=export['faithful:p28']['constant_hex']:
        raise ValueError('SAVED_CONSTANT_IDENTITY')
    plan=contract.plan();cs=contract.cells()
    provider_plan=dict(T=.8,cells=cs,numerical=plan['numerical'],epsilons=config['epsilons'],
        shot_rule=dict(alpha_axis=.025),caps_proposed=dict(primitive=20000,control_probe=0,
        compile=0,trajectory=0,occurrence=0,reference_matvec=50000000,
        reference_matvec_per_action=100000,deterministic_actions_per_cell=100000,
        untranspiled_instructions=1000000,tail_matvecs_corrected_and_raw={'B0':0}))
    port=core.H8CoreSignalPort(Path(config['provider_root']),dict(kind='H8_STRENGTHENED',plan=provider_plan),writer,permit)
    port.ham,port.sector=ham,sector
    port.qindices,port.qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    port.state=state.copy();port.saved_state=state.copy()
    port.reference,port.reference_error=saved_reference(runtime.read(paths['full_H_reference']),config['binding'])
    port.calls=permit
    port.tails={};port.oracle_tails={};port.spin={};port.primitive_ops={};port.truncated_references={}
    topology=sparse.OccupationCSR(sector.basis_indices,16)
    port.one=topology.one(ham.one_body);port.blocks=tuple(topology.one(g) for g in ham.g_matrices)
    del topology
    for c in cs:
        port.preps[c['id']]=exported.prefix_view(blocks,export['faithful:p28'],c['prefix'],
                                               export_sha256=config['export_sha256'])
        if contract.schedule(c)['ordinary_one_outer_step']!=schedules.primitive_time_schedule(c,T=.8)['ordinary_one_outer_step']:
            raise ValueError('PROVIDER_SCHEDULE_CHANGED')
    largest=max(cs,key=lambda c:c['prefix']);view=port.preps[largest['id']]
    fast=port.actions(view);native=molecular.MolecularPort.actions(port,view)
    # Cache the small independent spin eigensystems once; never regenerate the
    # saved Gaussian/preparation. CSR operators are made afresh inside workers.
    writer.write('preparation_reference_binding.json',dict(export=audit,
        source=config['SOURCE'],reference_receipt_sha256=config['reference_sha256'],
        saved_reference_reused=True,new_full_H_reference=0,
        prefix_views={str(p):port.preps[next(c['id'] for c in cs if c['prefix']==p)].preparation_hash
                      for p in sorted({c['prefix'] for c in cs})},certified=False))
    probes=[state,np.eye(1,len(state),0,dtype=complex)[0],np.eye(1,len(state),len(state)-1,dtype=complex)[0]]
    saved_scalar=runtime.read(paths['p21_scalar_receipts'])
    selected=[r for r in saved_scalar if r['prefix']==21 and r['q']==1]
    if len(selected)!=1:raise ValueError('SAVED_P21_SCALAR_UNIQUE')

    def callback(task,directory):
        child=permit.child(directory,task['counts']);local=copy(port)
        local.progress=child;local.calls=child;local.writer=runtime.Writer(directory,child.guard)
        local.primitive_ops={};local.truncated_references={};local.completed=0
        # Each expm action has an independent matvec ceiling in addition to
        # the durable global task reservations (48M total <= 50M phase cap).
        original=sparse.exponential_action;active=[None]
        old_take=child.take
        def take(name,count=1):
            if name=='reference_matvec' and active[0] is not None and \
                    child.used.get(name,0)+count-active[0]>100000:
                raise ValueError('REFERENCE_MATVEC_PER_ACTION_CAP')
            return old_take(name,count)
        child.take=take
        def bounded(op,trace,vector,t):
            prior=active[0];active[0]=child.used.get('reference_matvec',0)
            try:return original(op,trace,vector,t)
            finally:active[0]=prior
        sparse.exponential_action=bounded
        try:
            if task['kind']=='primitive':
                i=task['block'];op,trace=local.oracle_primitive('one' if i==0 else i-1)
                errors=[]
                for t in task['times']:
                    for v in probes:
                        child.take('primitive');a=native[i](v,t);f=fast[i](v,t)
                        o=bounded(op,trace,v,t)
                        error=max(float(np.linalg.norm(a-f)),float(np.linalg.norm(f-o)))
                        m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=error)
                        errors.append(error)
                result=dict(kind='primitive',block=i,time_count=len(task['times']),max_error=max(errors))
            else:
                p=task['prefix'];local.cells=task['cells'];local.plan=dict(port.plan,cells=local.cells)
                actual,oracle=local.cell_signal,local.oracle_signal
                def counted(c,**kw):child.take('B0_signals');return actual(c,**kw)
                def independent(c):
                    if c['prefix'] not in local.truncated_references:child.take('truncated_reference_pairs')
                    return oracle(c)
                local.cell_signal,local.oracle_signal=counted,independent
                scalar=checked_scalar(selected[0]['scalar'],p,ham,overlay,child)
                local.correctness()
                rows=runtime.read(directory/'signal_summary.json')['signals']
                out=[overlay_record(r,scalar,local,child,overlay,m) for r in rows]
                for row in out:local.writer.write(row['cell']['id']+'_signal.json',row)
                local.writer.write('prefix_results.json',dict(signals=rows+out))
                result=dict(kind='signals',prefix=p,rows=len(rows)+len(out),file='prefix_results.json')
            child.guard(force=True)
            return dict(result=result,calls=child.used)
        finally:sparse.exponential_action=original

    receipts,outputs=run_batches(config,permit,writer,port,callback)
    signals=[];primitive=[]
    for directory,result in outputs:
        if result['result']['kind']=='signals':signals+=runtime.read(directory/'prefix_results.json')['signals']
        else:primitive.append(result['result'])
    expected={c['id'] for c in plan['registrations']['additional_deterministic']}
    if len(signals)!=152 or {r['cell']['id'] for r in signals}!=expected:
        raise ValueError('H8_SIGNAL_COMPLETENESS')
    writer.write('primitive_validation.json',dict(records=primitive,probe_count=3,
        probe_actions=sum(v['time_count']*3 for v in primitive),negative_S4_times_included=True,certified=False))
    signals.sort(key=lambda r:r['cell']['id'])
    writer.write('H8_signal_complete.json',dict(status='COMPLETE_MANDATORY_STOP',signals=signals,
        empirical_allowance=True,conservative_bias_certificate=None,new_compile=0,new_sampling=0,
        new_full_H_reference=0,all_workers_reaped=all(r['all_reaped'] for r in receipts),
        call_counts=permit.used,irrevocable_reservations=permit.reserved,
        actual_settlements=permit.settled,next_stage_authorized=False))


def json_roundtrip(value):
    import json
    return json.loads(json.dumps(value,allow_nan=False))
