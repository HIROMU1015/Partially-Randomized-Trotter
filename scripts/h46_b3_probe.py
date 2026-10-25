"""B3 reconstruction and conditional signed-event checks, outside MC samples."""
from types import SimpleNamespace
import h46_b3 as b


def run():
    b.setup()
    import numpy as np
    from h46_m5_contract import core
    from h46_new2_native import forced_event_request
    from trotterlib.rte import finite_rte_distribution
    from trotterlib.df_rte_circuit import DFRTEEventCircuitRequest
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from trotterlib.df_trotter.circuit import simulate_statevector
    checks=[];reconstruction=[]
    for system in b.signal.PREFIX:
        ctx=b.native_context(system,[0]);prep=ctx.preparations[0]
        oracle=ctx.oracle;v=oracle.initial
        exact=np.zeros_like(v)
        for weight,g in zip(ctx.ham.lambdas,ctx.ham.g_matrices):
            m=oracle.one_body(g);exact+=weight*(m@(m@v))
        symbolic=prep.extracted_identity_coefficient*v
        for comp in prep.rte_preparation.symbolic_tail.components:
            symbolic=symbolic+comp.coefficient_abs*oracle.component(comp,v)
        error=float(np.linalg.norm(exact-symbolic))
        if error>1e-9: raise ValueError('B3_HAMILTONIAN_RECONSTRUCTION')
        reconstruction.append(dict(system=system,tail_action_error=error,one_body_retained=True,
            DF_fragments=len(prep.randomized_block_indices),scalar=prep.extracted_identity_coefficient,
            lambda_R=prep.exact_rte_lambda_r,input_regeneration=0))
        builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
        components={c.component_id:c for c in prep.rte_preparation.symbolic_tail.components}
        steps=core.schedule('4th(m5_best)',1,'ABS_TIME_PER_OUTER',16,T=1.2)
        for sign in (-1,1):
            step=next(s for s in steps if s['signed_time']*sign>0)
            for order in (2,4,6):
                tau=prep.exact_rte_lambda_r*step['signed_time']/step['rte_steps']
                event=forced_event_request(SimpleNamespace(preparation=prep,rte_distribution=finite_rte_distribution(tau,order)),order)
                circuit=builder.build_event(DFRTEEventCircuitRequest(event,prep.rte_preparation.component_specs,
                    controlled=True,ancilla_qubit=ctx.ham.n_qubits)).circuit
                low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)];high=np.r_[np.zeros_like(ctx.qstate),ctx.qstate]
                expected=np.zeros_like(ctx.qstate);expected[oracle.qindices]=oracle.event(event,components,v)
                actual=simulate_statevector(circuit,high)
                errors=[float(np.linalg.norm(actual[len(ctx.qstate):]-expected)),
                        float(np.linalg.norm(actual[:len(ctx.qstate)])),
                        float(np.linalg.norm(simulate_statevector(circuit,low)-low))]
                if max(errors)>1e-9: raise ValueError('SIGNED_CONDITIONAL_EVENT')
                # An explicit actual inverse, including order phase.
                back=simulate_statevector(circuit.inverse(),actual)
                adjoint_error=float(np.linalg.norm(back-high))
                if adjoint_error>1e-9: raise ValueError('ACTUAL_ADJOINT')
                checks.append(dict(system=system,sign=sign,order=order,errors=errors,
                    actual_adjoint_error=adjoint_error,conditional=True,included_in_cost_samples=False))
    result=dict(reconstruction=reconstruction,conditional_checks=checks,phase_alignment=False,
        new_random_trajectories=0,compile_calls=0,independent_occupation_algebra=True)
    b.save(b.PUBLIC/'B3_correctness.json',result)
    return result


if __name__=='__main__':
    import json
    print(json.dumps(run()))
