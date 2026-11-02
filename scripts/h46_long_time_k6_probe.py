"""New K6-only conditional gate check; never enters a cost sample."""
import json
from types import SimpleNamespace
import h46_long_time_signal as study


def run():
    study.install()
    import numpy as np
    from h46_long_time_native import native_context
    from h46_m5_contract import core
    from h46_new2_native import forced_event_request
    from trotterlib.rte import finite_rte_distribution
    from trotterlib.df_rte_circuit import DFRTEEventCircuitRequest
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from trotterlib.df_trotter.circuit import simulate_statevector
    checks=[]
    for system,prefix in [('H4',3),('H6',5)]:
        ctx=native_context(system,[prefix]);prep=ctx.preparations[prefix]
        builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
        components={v.component_id:v for v in prep.rte_preparation.symbolic_tail.components}
        steps=core.schedule('4th(m5_best)',1,'ABS_TIME_PER_OUTER',16,T=1.2)
        for sign in (-1,1):
            step=next(s for s in steps if s['signed_time']*sign>0)
            tau=prep.exact_rte_lambda_r*(step['signed_time']/step['rte_steps'])
            request=SimpleNamespace(preparation=prep,rte_distribution=finite_rte_distribution(tau,6))
            event=forced_event_request(request,6)
            req=DFRTEEventCircuitRequest(event,prep.rte_preparation.component_specs,
                                       controlled=True,ancilla_qubit=ctx.ham.n_qubits)
            circuit=builder.build_event(req).circuit
            expected=np.zeros_like(ctx.qstate)
            expected[ctx.oracle.qindices]=ctx.oracle.event(event,components,ctx.oracle.initial)
            low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)]
            high=np.r_[np.zeros_like(ctx.qstate),ctx.qstate]
            actual=simulate_statevector(circuit,high)
            error=float(np.linalg.norm(actual[len(ctx.qstate):]-expected))
            leakage=float(np.linalg.norm(actual[:len(ctx.qstate)]))
            low_error=float(np.linalg.norm(simulate_statevector(circuit,low)-low))
            if max(error,leakage,low_error)>1e-9:
                raise ValueError('CONDITIONAL_K6_GATE_DISAGREEMENT')
            checks.append(dict(system=system,prefix=prefix,sign=sign,order=6,
                vector_error=error,leakage=leakage,low_control_error=low_error,phase_alignment=False,
                conditional=True,included_in_cost_samples=False))
    result=dict(checks=checks,sampling_calls=0,compile_calls=0,
                reason='K6 is newly added; reuse existing forced-event/independent occupation oracle',
                existing_K0_K2_K4_checks_repeated=False)
    study.save(study.PUBLIC/'conditional_K6_gate_checks.json',result);return result


if __name__=='__main__':
    print(json.dumps(run()))
