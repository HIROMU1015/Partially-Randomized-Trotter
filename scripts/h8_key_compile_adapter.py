"""Replay saved preparation; never fit a model or generate a scalar/reference."""
from dataclasses import asdict
from fractions import Fraction
import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import h8_key_compile_contract as c
import h8_key_compile_controls as controls
import h8_key_compile_parallel as pool


def scalar_for(case, constant_hex, target):
    original=float.fromhex(constant_hex);c.require(math.isfinite(original),'ORIGINAL_SCALAR')
    c.require(target['case_id']==case['id'],'SCALAR_CASE_IDENTITY')
    saved=target['scalar_receipt']
    if case['method']=='B0':
        c.require(saved is None,'B0_NO_TAIL_SCALAR');return original
    c.require(case['method']=='B0I' and saved['policy']=='EXACT_STORED_BINARY64_TAIL_IDENTITY' and
        saved['input_operator_error_certificate']is False and math.isfinite(saved['C']) and
        Fraction(saved['numerator'])/Fraction(saved['denominator'])>0 and
        float(Fraction(saved['numerator'])/Fraction(saved['denominator']))==saved['C'],
        'OWN_PREFIX_SAVED_FULL_FOCK_SCALAR')
    result=original+saved['C'];c.require(math.isfinite(result),'NONFINITE_TOTAL_SCALAR');return result


def serial_axes(permit, build, compile_axis, *, map_owned=pool.map_owned):
    b=c.binding();rows=[];ordinal=0
    for case in b['cases']:
        built=build(case)  # One native evolution per case; controls precede compile.
        for axis in [a for a in b['axes'] if a['case_id']==case['id']]:
            ordinal+=1;permit.take('compile')  # Nonrefundable reservation before dispatch.
            task=dict(id=case['id']+'_'+axis['axis'],case_id=case['id'],axis=axis['axis'])
            receipt=map_owned([task],lambda t,d:compile_axis(t,built,d),
                cpus=permit.grant['worker_cpus'][:1],limits=dict(max_workers=1),
                directory=permit.output/('compile_'+str(ordinal).zfill(2)),deadline=math.inf,
                font_seed=permit.config.get('font_seed'))
            c.require(receipt['status']=='PASS' and receipt['all_reaped']is True and not receipt['cleanup_errors'] and
                      len(receipt['workers'])==1,'COMPILE_CLEANUP_UNCONFIRMED')
            w=receipt['workers'][0];path=Path(w['directory'])/'task_result.json';row=c.read(path)
            c.require(w['exit_code']==0 and w['reaped']is True and w['task_id']==task['id'] and
                c.sha(path)==w['result_sha256'] and (row['case_id'],row['axis'])==(case['id'],axis['axis']),
                'COMPILE_NATIVE_RESULT_IDENTITY')
            permit.take('model_error_check')
            row.update(native_exit_code=0,reaped=True,worker_identity=w['identity'],
                peak_RSS_bytes=w['peak_RSS_bytes'],native_CPU_seconds=w['cpu_seconds'],
                result_sha256=w['result_sha256'],model_sha256=c.MODEL,
                model_check=c.axis_check(axis['exact_prediction'],row['measured_RZ']))
            # Persist a failed measurement before rejecting it. Never label it accepted.
            c.save(permit.output/('axis_'+str(ordinal).zfill(2)+'_measured.json'),row);rows.append(row)
            c.require(row['model_check']['PASS'],'AXIS_MODEL_ERROR_GE_2_PERCENT')
        del built;gc.collect()
    return rows


def evaluate(config, grant, output):
    # The launcher checked OS affinity before narrowing the direct driver child.
    # This entry must be that child; its source/config/grant are rechecked below.
    c.require(os.sched_getaffinity(0)=={grant['driver_cpu']},'OWN_DRIVER_CPU')
    c.authorize(config,grant,affinity=set(grant['allowed_cpus']))
    permit=c.Permit(config,grant,output,affinity=set(grant['allowed_cpus']));b=c.binding();paths=config['input_mapping']
    model=c.read(paths['B1_frozen_models']);terminal=c.read(paths['B1_phase_terminal'])
    c.require(c.digest(model)==c.MODEL and terminal['status']=='COMPLETE_MANDATORY_STOP' and
              terminal['model_sha256']==c.MODEL,'FROZEN_B1_MODEL')
    for axis in b['axes']:
        case=next(x for x in b['cases'] if x['id']==axis['case_id'])
        coefficients=next(m['coefficients'] for m in model['models'] if
                          (m['method'],m['axis'])==(case['method'],axis['axis']))
        a,d,e=map(Fraction,coefficients);mult=3 if case['formula']=='4th' else 1
        c.require(str(a+mult*case['q']*(d+e*case['prefix']))==axis['exact_prediction'],'FIXED_PREDICTION')
    sys.path.insert(0,str(Path(config['provider_root'])/'src'))
    modules=[importlib.import_module(name) for name in (
        'trottertracks.resource_applicability.h6_matched_port_v1',
        'trottertracks.resource_applicability.h8_signal_preflight_port_v1',
        'trottertracks.resource_applicability.ax2b_native_df_v5','trotterlib.df_partial_s2','trotterlib.df_rte_tail')]
    m,loader,native,specs,basis=modules
    for module in modules:
        c.require(Path(module.__file__).resolve().is_relative_to(Path(config['provider_root'])/'src'),'PINNED_PROVIDER_IMPORT')
    import qiskit
    from qiskit import qpy
    from qiskit.circuit.library import RZGate,XXPlusYYGate
    import h8_phase_b2_adapter as legacy
    import h8_b0_anchor_adapter as common
    import h8_strengthened_export as exported
    permit.take('snapshot_load')
    ham,sector,full,state,md,_=loader.load_h8_snapshot(Path(paths['h8_input_snapshot.npz']),
        c.read(paths['snapshot_receipt.json']),c.read(paths['df_receipt.json']))
    c.require((ham.n_qubits,len(ham.lambdas),sector.dimension)==(16,28,4900) and
        md['hamiltonian_hash']==b['scientific_binding']['hamiltonian'] and
        md['state_hash']==b['scientific_binding']['state'],'H8_SCIENTIFIC_LAYOUT')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    export=c.read(paths['h8_preparation_export'])
    exported.validate_export(export,expected_hashes=b['expected_preparation_hashes'])
    def factory(name,params):return {'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params)
    def inspect(ops):
        described=basis.describe_basis_change_operations(ops)
        return described,json.loads(json.dumps([asdict(x) for x in described]))
    def require_legacy(scope):
        c.require(scope==common.c.SCOPE,'ONLY_SAVED_REPLAY_CAPABILITY');permit.require(c.SCOPE)
    blocks=common.restore(export,SimpleNamespace(require=require_legacy,take=permit.take),factory=factory,
        inspect=inspect,one_type=specs.DFDeterministicOneBodySpec,fragment_type=specs.DFDeterministicFragmentSpec)
    c.require(all(float(ham.lambdas[i]).hex()==block.lam.hex() for i,block in enumerate(blocks[1:])),'BLOCK_LAMBDA_IDENTITY')
    views={}
    for prefix in (11,15):
        permit.take('prefix_view');expected=b['prefix_preparations'][str(prefix)]
        identity=legacy.observe_prefix(blocks,export,prefix,expected,inspect)
        c.require(float(ham.constant).hex()==identity['constant_hex'],'CONSTANT_IDENTITY')
        views[prefix]=(blocks[:prefix+1],identity)
    cp=b['compiler_options']
    c.require(qiskit.__version__==cp['qiskit_version'],'QISKIT_VERSION_CHANGED')
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    targets={x['case_id']:x for x in b['targets']}
    def build(case):
        bs,identity=views[case['prefix']];scalar=scalar_for(case,identity['constant_hex'],targets[case['id']])
        built=controls.build_case(case,bs,scalar,qstate,targets[case['id']],permit,native,m,
                                  cap=case['measured_wrapper_allocation_bound'])
        built.update(case=case,identity=identity,scalar_hex=scalar.hex())
        c.save(permit.output/(case['id']+'_control_preparation.json'),dict(case=case,preparation_binding=identity,
            scalar_hex=scalar.hex(),max_control_error=built['max_control_error'],fingerprints=built['fingerprints'],
            spools=built['spools'],SOURCE=config['SOURCE'],profile_sha256=c.PROFILE))
        return built
    def compile_axis(task,built,directory):
        # The parent reserves qpy_load before dispatch; counters never depend on fork writes.
        spool=built['spools'][task['axis']];fp=built['fingerprints'][task['axis']]
        wrapper=controls.load_wrapper(spool,output,qpy)
        def entered():c.save(directory/'compile_entered.json',dict(task=task,SOURCE=config['SOURCE'],
            model_sha256=c.MODEL,compiler_hash=c.COMPILER,fingerprint=fp,reserved_in_parent=True))
        row=common.compile_wrapper(wrapper,fp,m,compiler,cap=math.inf,before_compile=entered)
        row.update(case_id=task['case_id'],axis=task['axis'],SOURCE=config['SOURCE'],profile_sha256=c.PROFILE,
            preparation_binding=built['identity'],scalar_hex=built['scalar_hex'],max_control_error=built['max_control_error'])
        return row
    def map_compile(*args,**kwargs):
        permit.take('qpy_load');return pool.map_owned(*args,**kwargs)
    rows=serial_axes(permit,build,compile_axis,map_owned=map_compile)
    result=c.complete(rows,permit.calls,config['SOURCE'])
    result.update(SOURCE=config['SOURCE'],rows=rows,actual_calls=permit.calls)
    return result
