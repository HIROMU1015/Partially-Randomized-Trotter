"""Saved H8 prefix replay, parallel controls/QPY, and serial per-axis validation."""
from dataclasses import asdict
from fractions import Fraction
import gc
import importlib
import json
import math
from pathlib import Path
import sys
import h8_dba_contract as c
import h8_key_compile_controls as controls
import h8_key_compile_parallel as pool


def scalar_for(case,constant,target):
    original=float.fromhex(constant)
    if case['method']=='B0':
        c.require(target['scalar_receipt'] is None,'B0_ORIGINAL_SCALAR_ONCE');return original
    saved=target['scalar_receipt']
    c.require(saved['policy']=='EXACT_STORED_BINARY64_TAIL_IDENTITY'
              and float(Fraction(saved['numerator'])/Fraction(saved['denominator']))==saved['C']
              and math.isfinite(saved['C']),'OWN_PREFIX_EXACT_STORED_SCALAR')
    return original+saved['C']


def load(config,permit):
    sys.path.insert(0,str(Path(config['provider_root'])/'src'))
    m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    loader=importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1')
    native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    specs=importlib.import_module('trotterlib.df_partial_s2')
    basis=importlib.import_module('trotterlib.df_rte_tail')
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    import h8_b0_anchor_adapter as replay
    import h8_strengthened_export as exported
    from qiskit.circuit.library import RZGate,XXPlusYYGate
    import qiskit
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    paths=config['mapping'];permit.take('snapshot_load')
    ham,sector,full,state,md,_=loader.load_h8_snapshot(Path(paths['h8_input_snapshot.npz']),
        c.read(paths['snapshot_receipt.json']),c.read(paths['df_receipt.json']))
    c.require((ham.n_qubits,len(ham.lambdas),sector.dimension)==(16,28,4900),'H8_LAYOUT')
    c.require(md['hamiltonian_hash']==config['binding']['hamiltonian']
              and md['state_hash']==config['binding']['state'],'H8_INPUT_STATE_IDENTITY')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    export=c.read(paths['h8_preparation_export'])
    exported.validate_export(export,expected_hashes=config['expected_preparation_hashes'])
    def factory(name,params):return {'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params)
    def inspect(ops):
        described=basis.describe_basis_change_operations(ops)
        return described,json.loads(json.dumps([asdict(x) for x in described]))
    blocks=replay.restore(export,permit,factory=factory,inspect=inspect,
        one_type=specs.DFDeterministicOneBodySpec,fragment_type=specs.DFDeterministicFragmentSpec)
    c.require(all(float(ham.lambdas[i]).hex()==b.lam.hex() for i,b in enumerate(blocks[1:])),'BLOCK_LAMBDAS')
    cp=config['compiler_options'];c.require(qiskit.__version__=='1.3.0','QISKIT_VERSION')
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,
                                cp['seed_transpiler'],qiskit.__version__)
    c.require(compiler_settings_hash(compiler)==c.COMPILER,'LIVE_COMPILER_SETTINGS')
    return m,native,qstate,blocks,export,compiler,use_fingerprint_reuse


def measured_worker(permit,task,built,m,compiler,directory):
    from qiskit import qpy
    import h8_b0_anchor_adapter as measured
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    wrapper=controls.load_wrapper(built['spools'][task['axis']],permit.output,qpy)
    with use_fingerprint_reuse() as reuse:
        row=measured.compile_wrapper(wrapper,built['fingerprints'][task['axis']],m,compiler,cap=math.inf,
            before_compile=lambda:c.save(directory/'compile_entered.json',dict(task=task,SOURCE=permit.grant['SOURCE'])))
        row['fingerprint_reuse']=reuse.snapshot()
    return dict(row,case_id=task['case_id'],axis=task['axis'])


def evaluate(config,grant,output):
    permit=c.Permit(config,grant);case=next(x for x in c.plan()['D'] if x['id']==grant['case_id'])
    target=config['targets'][case['id']]
    m,native,qstate,blocks,export,compiler,reuse=load(config,permit)
    whole=export['faithful:p28'];scalar=scalar_for(case,whole['constant_hex'],target)
    view=blocks[:case['prefix']+1]
    c.require(tuple(b.original_fragment_index for b in view[1:])==tuple(range(case['prefix'])),'PREFIX_GENERATION_ORDER')
    identity=dict(full_preparation_hash=whole['preparation_hash'],prefix=case['prefix'],
        saved_full_blocks_sha256=c.digest(whole['blocks'][:case['prefix']+1]),
        constant_hex=whole['constant_hex'],scalar_hex=scalar.hex(),profile_sha256=c.PROFILE)
    c.save(output/'preparation_binding.json',identity)
    with reuse():
        built=controls.build_case(case,view,scalar,qstate,target,permit,native,m,
            n=16,cap=4_000_000,worker_cpus=grant['worker_cpus'])
    c.save(output/'control_preparation.json',dict(case=case,identity=identity,**built))
    rows=[]
    for axis in c.plan()['axes']:
        c.require(not Path(grant['shared_model_STOP']).exists(),'SHARED_MODEL_HYPOTHESIS_ALREADY_FAILED')
        task=dict(id=case['id']+'_'+axis,case_id=case['id'],axis=axis)
        permit.take('qpy_load');permit.take('compile')
        receipt=pool.map_owned([task],lambda t,d:measured_worker(permit,t,built,m,compiler,d),
            cpus=grant['worker_cpus'][:1],limits=dict(max_workers=1),directory=output/('compile_'+axis),
            deadline=math.inf,font_seed=config['font_seed'])
        c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'], 'NATIVE_COMPILE_CLEANUP')
        worker=receipt['workers'][0];path=Path(worker['directory'])/'task_result.json';row=c.read(path)
        c.require(worker['exit_code']==0 and worker['reaped'] and c.sha(path)==worker['result_sha256']
            and (row['case_id'],row['axis'])==(case['id'],axis),'COMPILE_RESULT_IDENTITY')
        permit.take('model_error_check')
        prediction=config['predictions'][case['id']][axis]
        row.update(SOURCE=grant['SOURCE'],prediction=prediction,model_check=c.axis_check(prediction,row['measured_RZ']),
            model_sha256=c.MODEL,profile_sha256=c.PROFILE,native_exit_code=0,reaped=True,
            max_control_error=built['max_control_error'],preparation_binding=identity)
        c.save(output/('measured_'+axis+'.json'),row);rows.append(row)
        if not row['model_check']['PASS']:
            marker=Path(grant['shared_model_STOP'])
            if not marker.exists():c.save(marker,dict(case_id=case['id'],axis=axis,reason='MODEL_ERROR_GE_2_PERCENT'))
            raise ValueError('AXIS_MODEL_ERROR_GE_2_PERCENT')
        gc.collect()
    c.require(permit.calls==c.CAPS,'EXACT_D_CALL_COMPLETENESS')
    return dict(status='COMPLETE_MANDATORY_STOP',case=case,rows=rows,calls=permit.calls,
        SOURCE=grant['SOURCE'],model_refit=False,cost_reaccounting=False,new_signal=False,new_sampling=False)
