"""Separate C7-4 native-gate numerical comparison; no molecular builders."""
from uuid import uuid4
import numpy as np
from .track_d_c7_3_acceleration import SmallUnitary
from .track_d_state_action_acceleration import _NativeExecutor


class NativeSmallUnitary(SmallUnitary):
    """Keep P0's native gates for official Aer instead of generic matrices.

    Existing SmallUnitary has no conversion hook; this narrow adapter reuses
    its backend and native executor without editing the historical SOURCE.
    No compiler, renormalization, phase fitting or incomplete-column checks.
    """
    def __call__(self,circuit):
        from qiskit_aer.jobs import AerJob
        from threadpoolctl import threadpool_limits
        if circuit.num_qubits>9:raise ValueError('FULL_REFERENCE_OPERATOR_FORBIDDEN')
        if circuit.num_clbits or circuit.parameters:raise ValueError('BOUND_UNITARY_REQUIRED')
        if any(i.operation.name not in ('rz','sx','sxdg','x','cx') for i in circuit.data):
            raise ValueError('NATIVE_P0_DOMAIN_ONLY')
        body=circuit.copy();phase=float(body.global_phase);body.global_phase=0.;body.save_unitary()
        with _NativeExecutor(self.threads) as executor,threadpool_limits(limits=1,user_api='blas'):
            job=AerJob(self.backend,str(uuid4()),self.backend._execute_circuits_job,
                circuits=[body],parameter_binds=None,run_options={'shots':1},executor=executor)
            job.submit();result=job.result()
            if not result.success:raise ValueError('NATIVE_UNITARY_FAIL')
            value=np.asarray(result.get_unitary(0),dtype=np.complex128).copy()
        value*=np.exp(1j*phase)
        return value


def artificial_block(width):
    from qiskit import QuantumCircuit
    c=QuantumCircuit(width)
    for i in range(width):
        c.rz(((-1)**i)*(.17+.03*i),i);c.sx(i)
    for i in range(width-1):c.cx(i,i+1)
    c.rz(-1e-12,width-1);c.rz(0.,0);c.x(0);c.global_phase=.123
    return c


def residual(a,b):
    d=np.asarray(a,dtype=np.clongdouble)-np.asarray(b,dtype=np.clongdouble)
    maximum=float(np.max(abs(d)));frobenius=float(np.sqrt(np.sum(d.real*d.real+d.imag*d.imag)))
    return dict(max_entry=maximum,frobenius=frobenius,component_tolerance=1e-10,
        full_matrix_frobenius_diagnostic_tolerance=2e-10,
        component_PASS=maximum<=1e-10,frobenius_diagnostic_PASS=frobenius<=2e-10,
        phase_fit=False,renormalization=False)


def toy_case(task):
    """Fixed independent artificial full-matrix comparison, not an H4 q."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from threadpoolctl import threadpool_limits
    width,repetitions=task;block=artificial_block(width)
    phase=float(block.global_phase);body=block.copy();body.global_phase=0.
    c=QuantumCircuit(width)
    for _ in range(repetitions):c.compose(body,inplace=True,copy=False)
    c.global_phase=float(np.longdouble(phase)*repetitions)
    with threadpool_limits(limits=1):
        # Extra-precision multiplication of a short Qiskit/binary64 block.
        # It reduces accumulation error but is not an exact interval oracle.
        short=Operator(body).data.astype(np.clongdouble)
        target=np.linalg.matrix_power(short,repetitions)*np.exp(np.clongdouble(1j)*np.longdouble(phase)*repetitions)
        generic=SmallUnitary(1)(c);native=NativeSmallUnitary(1)(c)
        raw=Operator(block).data
        inverse=Operator(block.inverse()).data
    return dict(width=width,block_repetitions=repetitions,operations=len(c.data),
        generic_vs_extended_shortblock_power=residual(generic,target),
        native_vs_extended_shortblock_power=residual(native,target),
        generic_vs_native=residual(generic,native),
        generic_unitarity=residual(generic.conj().T@generic,np.eye(1<<width)),
        native_unitarity=residual(native.conj().T@native,np.eye(1<<width)),
        short_actual_inverse=residual(inverse@raw,np.eye(1<<width)),
        reference_qualification='longdouble product of a binary64 short block; short-block error is repeated, no exact error enclosure',
        source_h4_loaded=False,cost_circuit_changed=False,threshold_relaxed=False)


def short_compile(width):
    from qiskit.quantum_info import Operator
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    block=artificial_block(width);compiled=compile_checked(block,CompilerProfile())
    old=Operator(block).data;new=NativeSmallUnitary(1)(compiled)
    return dict(width=width,compile_calls=1,artificial_only=True,
        raw_size=len(block.data),compiled_size=len(compiled.data),
        raw_vs_compiled=residual(new,old),compiler='unchanged official P0/phase compiler')
