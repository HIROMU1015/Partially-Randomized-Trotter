"""Only authorized workers evaluate the registered E local fixture."""
from fractions import Fraction as F
import math
import numpy as np
import mpmath as mp
from .intervals import configure, iv, upper, bounds
from .certification import Ball, comm, pauli_table, frobenius_upper, verified_rotation, gamma
from .comparison import certified_primitives
from .digital import largest_remainder, weight_point
from .native import pauli_ir, ir_counts, circuit_from_ir, native_inventory


def serialize_frames(frames):
    return [[[ [float(z.real).hex(),float(z.imag).hex()] for z in row] for row in frame] for frame in frames]


def deserialize_frames(rows):
    return [np.array([[complex(float.fromhex(a),float.fromhex(b)) for a,b in row] for row in frame]) for frame in rows]


def operator_numeric_guard(circuit, ideal, ideal_radius):
    """Conditional on faithful libm and local <=2-qubit Operator arithmetic.

    128 floating operations per native gate overestimates local 2/4-wide dot
    products. Ideal reference radius and native phase are returned separately.
    No residual alone is called a certificate.
    """
    from qiskit.quantum_info import Operator
    actual=Operator(circuit).data
    rounding=gamma(128*len(circuit.data)+64)*upper(mp.iv.sqrt(iv(len(actual))))
    residual=frobenius_upper(actual-ideal)+gamma(2)*(frobenius_upper(actual)+frobenius_upper(ideal))
    return residual+rounding+ideal_radius, {'float_residual_diagnostic':float(np.linalg.norm(actual-ideal,'fro')),
        'reference_radius':str(ideal_radius),'Operator_forward_guard':str(rounding),
        'model':'IEEE_binary64_standard_BLAS_and_faithful_libm_4ulp_local_gate_Operator'}


def native_fixture(r, frames, prior, on_compile, *, on_diagnostic):
    configure()
    from scipy.linalg import expm, block_diag
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    from .fingerprint_diagnostics import diagnostic_transpile_and_measure_cost
    if qiskit.__version__ != '1.3.0': raise PermissionError('Compiler version changed')
    qs, frame_certificates = certified_primitives(frames,(1,-1,1))
    t=comm(qs[0],comm(qs[1],qs[2]))
    table, proxy, radius, coefficients=pauli_table(t,number_conserving=True,traceless=True)
    ell=sum(abs(v) for v in table.values())
    if ell==0: raise ArithmeticError('Registered E zero computational generator; STOP')
    alpha_iv=mp.iv.atan2(iv(ell/F(128)),iv(1))
    a,b=bounds(alpha_iv);alpha=float((a+b)/2)
    alpha_error=max(abs(F(alpha)-a),abs(F(alpha)-b))
    probabilities={k:abs(v)/ell for k,v in table.items()}
    law=largest_remainder(probabilities)
    law_l1=sum(abs(law[k]-probabilities[k]) for k in law)
    compiler=CompilerSettings(basis_gates=('rz','sx','x','cx'),backend_name=None,
        coupling_map=None,optimization_level=0,layout_method=None,routing_method=None,
        transpiler_seed=20261011,qiskit_version=qiskit.__version__)
    result={'active_modes':r,'frame_certificates':frame_certificates,'Pauli_coefficients':coefficients,
            'Pauli_proxy_radius':str(radius),'ell_computational':str(ell),
            'ideal_ell_interval':{'lo':str(max(F(0),ell-(1<<r)*radius)), 'hi':str(ell+(1<<r)*radius)},
            'alpha_interval':{'lo':str(a),'hi':str(b)},'native_alpha_raw':str(F(alpha)),
            'conditional_law_L1':str(law_l1),'native':[],'all_B_event_costs':[], 'compile_calls':0}
    w,weight_error=weight_point(F(1,128),ell)
    result['conditional_correction_accounting']={
        'A_executed_weight':'1','A_second_moment':'1','A_range':'1',
        'B_executed_weight':str(w),'B_second_moment':str(w*w),'B_range':str(w),
        'B_weight_radius':str(weight_error),
        'B_mean_proxy_to_ideal_error_upper':str(radius/F(128)),
        'B_conditional_digital_bias_upper':str(w*law_l1),
        'B_finite_linearization_to_exact_rotation_bound':str(upper(mp.iv.exp(iv(F(4,128)))-1-iv(F(4,128)))),
        'outer_coin_rate':'UNSPECIFIED_LOCAL_CONDITIONAL_NOT_CD_RATE',
        'original_RTE_normalization':'NOT_AN_E_DOMAIN_QUANTITY',
        'PR_shot_count':None}
    # Keep the current float DAG unchanged; observation does not repair identity.
    raw_u=expm(1j*.03125*t.value/4)
    reference=verified_rotation(t,F(1,128))
    for controlled in (False,True):
        target=block_diag(np.eye(1<<r),raw_u) if controlled else raw_u
        qc=QuantumCircuit(r+int(controlled));qc.append(UnitaryGate(target),qc.qubits)
        result['compile_calls']+=1;on_compile(result)
        expected=next(p for p in prior if p['active_modes']==r and p['controlled']==controlled)
        compiled=diagnostic_transpile_and_measure_cost(qc,compiler,
            expected=expected['circuit_fingerprint'],
            arrays={**{f'frame_{i}':v for i,v in enumerate(frames)},
                    **{f'Q_{i}':q.value for i,q in enumerate(qs)},
                    'T':t.value,'raw_U':raw_u,'input_target':target},
            context={'active_modes':r,'controlled':controlled,'arm':'A',
                     'guard_location':'native_fixture.whole_input_identity'},
            save=lambda record:on_diagnostic(controlled,record))
        if len(compiled.transpiled_circuit.data)>200000:raise RuntimeError('Native instruction cap')
        if compiled.actual_circuit_fingerprint!=expected['circuit_fingerprint']:
            raise ArithmeticError('Frozen E whole fingerprint mismatch; no synthesis')
        ideal=block_diag(np.eye(1<<r),reference.value) if controlled else reference.value
        error,ledger=operator_numeric_guard(compiled.transpiled_circuit,ideal,reference.error)
        inventory=native_inventory(compiled.transpiled_circuit)
        error+=F(inventory['angle_lowering_error_upper'])+F(inventory['global_phase_error_upper'])
        if error>F(1,5000000):raise ArithmeticError('Whole native numerical allowance not certified')
        inventory['numeric_operator_error_upper']=str(error)
        result['native'].append({'arm':'A','controlled':controlled,'inventory':inventory,
            'circuit_fingerprint':compiled.actual_circuit_fingerprint,'compiler_settings_hash':compiled.compiler_settings_hash,
            'depth':compiled.total_depth,'numeric_operator_error_upper':str(error),'numeric_ledger':ledger})
    # Full per-label analytical IR prices; representative compiles validate only.
    representatives={}
    for label,beta in table.items():
        if not beta: continue
        weight=sum(c!='I' for c in label)
        representatives.setdefault(weight,label)
        for controlled in (False,True):
            ir=pauli_ir(label,(1 if beta>0 else -1)*alpha,controlled)
            price=ir_counts(ir)
            result['all_B_event_costs'].append({'label':label,'controlled':controlled,
                'ideal_conditional_probability':str(probabilities[label]),
                'executed_conditional_probability':str(law[label]),'sign':1 if beta>0 else -1, **price})
    for weight,label in sorted(representatives.items()):
        for controlled in (False,True):
            beta=table[label];sign=1 if beta>0 else -1
            ir=pauli_ir(label,sign*alpha,controlled);qc=circuit_from_ir(ir)
            result['compile_calls']+=1;on_compile(result)
            compiled=transpile_and_measure_cost(qc,compiler)
            expected_counts=ir_counts(ir)['gate_counts']
            if dict(compiled.posttranspile_gate_counts)!=expected_counts:
                raise ArithmeticError('Pauli lowering count mismatch; no compiler switching')
            from .certification import pauli_matrix
            p=pauli_matrix(label)
            target=math.cos(alpha)*np.eye(1<<r)+1j*sign*math.sin(alpha)*p
            if controlled:target=block_diag(np.eye(1<<r),target)
            error,ledger=operator_numeric_guard(compiled.transpiled_circuit,target,alpha_error+gamma(16)*(1<<r))
            inv=native_inventory(compiled.transpiled_circuit,semantic_alpha=ell/F(128))
            error+=F(inv['angle_lowering_error_upper'])+F(inv['global_phase_error_upper'])
            if error>F(1,5000000):raise ArithmeticError('Pauli native numerical allowance not certified')
            inv['numeric_operator_error_upper']=str(error)
            result['native'].append({'arm':'B','label':label,'support_weight':weight,'controlled':controlled,
                'inventory':inv,'depth':compiled.total_depth,'numeric_operator_error_upper':str(error),'numeric_ledger':ledger})
    # Each label's T requirements are the same signed arbitrary angle classes
    # and exact Clifford basis; do not transplant a representative's 1Q count.
    result['T_semantic_angles']=[{'kind':'atan','value':str(ell/F(128)),'scale':str(s)} for s in (-1,1,-2,2)]
    max_b_instructions=max(row['1Q']+row['CX'] for row in result['all_B_event_costs'])
    b_uniform=(gamma(128*max_b_instructions+128)*upper(mp.iv.sqrt(iv(1<<(r+1))))
               +2*alpha_error+F(1,10**14))
    if b_uniform>F(1,5000000):raise ArithmeticError('Uniform B native numerical allowance failed')
    result['B_local_numeric_uniform_upper']=str(b_uniform)
    result['basis']='active-frame identity local centre only'
    result['CD_cost_reuse']=False
    result['whole_PR_cost']=None
    return result
