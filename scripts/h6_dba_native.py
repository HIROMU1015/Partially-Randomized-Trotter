"""Lazy H6 native adapters. Read saved preparation; retain compiler and control."""
import gc
import importlib
import math
from pathlib import Path
import time
import h6_dba_contract as c


def load(config):
    m = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    codec = importlib.import_module('trottertracks.resource_applicability.h6_roundoff_preparation_v1')
    native = importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    ham, sector, full, state, md, _ = m.load_h6_snapshot(Path(config['mapping']['h6_input_snapshot.npz']),
        c.read(config['mapping']['snapshot_receipt.json']), c.read(config['mapping']['df_receipt.json']))
    c.require((ham.n_qubits, len(ham.lambdas), sector.dimension) == (12,19,400), 'H6_LAYOUT')
    c.require(md['hamiltonian_hash'] == config['binding']['hamiltonian']
              and md['state_hash'] == config['binding']['state'], 'HAMILTONIAN_STATE_IDENTITY')
    _, qstate = m.checked_basis_bridge(ham,sector,full,state,1e-12)
    preps, archive = codec.read_preparations(config['archive']['path'],config['archive']['sha256'])
    c.require(archive['binding'] == config['archive']['binding'], 'ARCHIVE_IDENTITY')
    whole = preps['faithful:p19']
    c.require(whole.hamiltonian_hash == config['binding']['hamiltonian']
              and whole.deterministic_fragment_indices == tuple(range(19)), 'FULL_PREPARATION')
    import qiskit
    cp = config['compiler_options']
    c.require(qiskit.__version__ == '1.3.0', 'QISKIT_VERSION')
    compiler = m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,
                                  cp['seed_transpiler'],qiskit.__version__)
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    c.require(compiler_settings_hash(compiler) == config['binding']['compiler'], 'COMPILER_PROFILE')
    return m,native,ham,sector,state,qstate,preps,compiler


def control(evolution, m, qstate, writer, expected=None):
    np = m.np; n = 12
    low = np.concatenate((qstate,np.zeros_like(qstate)))
    high = np.concatenate((np.zeros_like(qstate),qstate))
    a,b = m.simulate_statevector(evolution.circuit,low),m.simulate_statevector(evolution.circuit,high)
    c.require(np.isfinite(a).all() and np.isfinite(b).all(), 'NONFINITE_NATIVE_CONTROL')
    z = complex(np.vdot(qstate,b[len(qstate):]))
    errors = [float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(qstate)])),
              abs(float(np.linalg.norm(b))-1)]
    if expected is not None:
        errors.append(abs(z-complex(expected['real'],expected['imag'])))
    for axis,value in [('cosine',z.real),('sine',z.imag)]:
        probe = m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=1000000)
        errors.append(abs(m.hadamard_expectation(m.simulate_statevector(probe,low),n)-value))
        del probe
    maximum = max(errors)
    m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=maximum)
    c.require(math.isfinite(maximum) and maximum < 1e-8, 'NATIVE_CONTROL_DISAGREEMENT')
    writer.write('control.json',dict(status='PASS', probes=4, max_error=maximum,
        signal=dict(real=z.real,imag=z.imag), reference_policy='saved deterministic signal' if expected else
        'paired native branches/X/Y; independently reconstructed event oracle additionally required',
        certified=False))
    return z


def compile_pair(evolution, m, compiler, config, writer, *, predictions=None):
    rows = []
    for axis in ('cosine','sine'):
        wrapper = m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
        fp = m.canonical_qiskit_circuit_fingerprint(wrapper)
        writer.write('compile_'+axis+'_reserved.json',dict(axis=axis,fingerprint=fp,
            input_instructions=len(wrapper.data), nonrefundable=True))
        start = time.monotonic()
        result = m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fp)
        c.require(result.compiler_settings_hash == config['binding']['compiler']
                  and result.actual_circuit_fingerprint == fp
                  and m.canonical_qiskit_circuit_fingerprint(wrapper) == fp, 'COMPILE_IDENTITY')
        row = dict(axis=axis,RZ=result.rz_count,CX=result.cx_count,fingerprint=fp,
                   compiler_hash=result.compiler_settings_hash,compile_seconds=time.monotonic()-start)
        c.require(type(row['RZ']) is int and row['RZ'] > 0, 'INTEGER_RZ')
        if predictions:
            row['prediction'] = predictions[axis]
            row['model_check'] = c.axis_check(predictions[axis],row['RZ'])
        writer.write('measured_'+axis+'.json',row);rows.append(row)
        if predictions:
            c.require(row['model_check']['PASS'], 'AXIS_MODEL_ERROR_GE_2_PERCENT')
        del result,wrapper;gc.collect()
    return rows


def direct(config, directory):
    writer = c.Writer(directory)
    m,native,ham,sector,state,qstate,preps,compiler = load(config)
    saved = c.read(config['mapping']['D_saved_signal'])
    c.require(saved['cell']['id'] == c.plan()['D']['cell_id'], 'REGISTERED_D_CELL')
    whole = preps['faithful:p19'];cell = saved['cell']
    scalar = whole.constant_coefficient + saved['scalar_receipt']['C']
    writer.write('build_reserved.json',dict(cell=cell,scalar_hex=scalar.hex(),
        archive_sha256=config['archive']['sha256'],control='symmetric_directional'))
    evolution = native.build_deterministic_native(whole.deterministic_blocks[:12],num_system_qubits=12,
        T=.8,q=4,formula='4th',scalar=scalar,control_policy='symmetric_directional',max_instructions=1000000)
    control(evolution,m,qstate,writer,expected=saved['signals']['corrected'])
    rows = compile_pair(evolution,m,compiler,config,writer,predictions=config['D_predictions'])
    return dict(stage='D',status='COMPLETE_MANDATORY_STOP',cell_id=cell['id'],rows=rows,
                calls=dict(native_compile=2,control_probe=4),new_signal=False,new_fit=False,
                cost_reaccounting=False,population_mean_certified=False)
