"""Conditional K2 product correctness only; no price-cohort observation."""
import numpy as np
import h8_full_h_pure_rte as s


def main():
    ctx=s.context('conditional_product_compile')
    row=s.read(s.PUBLIC/'selection.json')['candidates'][0];prep=ctx.preps[0]
    from trotterlib.rte import finite_rte_distribution,_make_event
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder,DFRTEEventCircuitRequest
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fingerprint
    import h8_compiler_phase_bookkeeping as phase
    from qiskit.circuit.library import CZGate
    # Old Stage C observed order0 only. CZ is needed by K2 product terms.
    # In this separate process extend only the diagnostic gate allow-list;
    # its unchanged local Operator check must certify zero additional phase.
    gate=CZGate();_,cz=phase.check_gate((phase.gate_key(gate),gate))
    s.require(cz['extra_phase_quarters']==0 and cz['error']<1e-10,'CZ_OPERATOR_CHECK')
    phase.NONBASIS=phase.NONBASIS|{'cz'}
    cs=prep.rte_preparation.symbolic_tail.components
    one=next(i for i,c in enumerate(cs) if c.df_fragment_id=='full-h-one-body')
    two=max((i for i,c in enumerate(cs) if c.df_fragment_id!='full-h-one-body' and len(c.diagonal_pauli_support)==2),key=lambda i:cs[i].probability)
    dist=finite_rte_distribution(prep.exact_rte_lambda_r*1.6/row['s'],2)
    event=_make_event([one,two,one],cs,dist,dist.orders.index(2))
    qc=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry).build_event(
        DFRTEEventCircuitRequest(event,prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=16)).circuit
    expected=ctx.oracle.event(ctx.qstate,event)
    evolution=ctx.native.NativeEvolution(qc,16,1.6/row['s'],1,'2nd','symmetric_directional',len(qc.data)+1,fingerprint(qc))
    low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)];checks=[]
    for axis in ('cosine','sine'):
        wrapper=ctx.native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=len(qc.data)+4)
        compiled=transpile_and_measure_cost(wrapper,ctx.compiler,actual_circuit_fingerprint=fingerprint(wrapper))
        proof=phase.certify(wrapper,{},workers=1)
        corrected,accounting=phase.correct_phase(wrapper,compiled.transpiled_circuit,proof)
        before=simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
        after=simulate_statevector(corrected.remove_final_measurements(inplace=False),low)
        factor=1. if axis=='cosine' else -1j
        independent=np.r_[(ctx.qstate+factor*expected)/2,(ctx.qstate-factor*expected)/2]
        native_error=float(np.linalg.norm(before-independent));compiled_error=float(np.linalg.norm(after-before))
        s.require(native_error<1e-8 and compiled_error<1e-7,'CONDITIONAL_PRODUCT_COMPILED_ACTION')
        checks.append(dict(axis=axis,native_error=native_error,compiled_error=compiled_error,
            operator_maximum_error=proof['maximum_error'],phase_accounting=accounting))
    result=dict(status='PASS',candidate=row,conditional=True,statistical_samples=0,in_price_samples=False,
        controls=['one_body_Z','DF_ZZ_product','order2','CZ'],cz_operator=cz,checks=checks,
        base_phase_SOURCE='18ce41942051065fa097644a8a937bd11adba708',
        diagnostic_allow_list_extension=['cz'],original_phase_helper_file_unchanged=True,
        live_price_SOURCE_unchanged=True,phase_alignment=False)
    s.save(s.PUBLIC/'conditional_product_compile_validation.json',result);print(result)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        s.save(s.PUBLIC/'conditional_product_compile_STOP.json',dict(reason=str(exc),type=type(exc).__name__))
        raise
