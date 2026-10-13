"""Distinguish common global phase from an ancilla-relative phase error."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scripts.diagnostics.audit_saved_rte_basis_cost import phase_diagnostic, restore


def test_single_global_phase_is_reported_with_strict_failure():
    raw=np.eye(4,dtype=complex)
    result=phase_diagnostic(raw,-raw)
    assert result['strict_max_abs_difference']==2
    assert result['common_phase_max_abs_residual']==0
    assert result['common_phase']['real']==-1
    assert result['branch_specific_phase_alignment'] is False


def test_relative_ancilla_sign_cannot_be_removed_by_common_alignment():
    raw=np.eye(4,dtype=complex)
    native=np.diag([1,1,-1,-1]).astype(complex)
    result=phase_diagnostic(raw,native)
    assert result['common_phase_max_abs_residual']>1
    assert result['ancilla_one_block_residual']>1


def test_saved_native_ir_retains_explicit_global_phase():
    circuit=QuantumCircuit(2);circuit.sx(0);circuit.rz(.29,1);circuit.cx(0,1)
    circuit.global_phase=.71
    ir={'qubits':2,'global_phase':.71,'operations':[
        {'name':'sx','qubits':[0],'parameters':[]},
        {'name':'rz','qubits':[1],'parameters':[.29]},
        {'name':'cx','qubits':[0,1],'parameters':[]}]}
    assert np.max(np.abs(Operator(circuit).data-Operator(restore(ir)).data))<1e-12
