"""Phase-strict off-domain controls. No registered input or synthesis."""
from fractions import Fraction as F
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from trottertracks.algorithm_codesign.df_correction_et_phase_v7 import acceptance, native_evaluation
from trottertracks.algorithm_codesign.df_correction_et_numeric_v6 import acceptance as old
from trottertracks.algorithm_codesign.df_correction_comparison import intervals
from trottertracks.algorithm_codesign.df_correction_comparison.native import t_completion


@pytest.fixture(autouse=True)
def configure():
    intervals.configure()


@pytest.mark.parametrize('phase', [.073, -.043, .19])
def test_unsupported_raw_phase_preserved_not_rounded_and_T_remains_unknown(phase):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(2)
    qc.x(0);qc.cx(0, 1);qc.global_phase = phase
    target = Operator(qc).data.copy()
    previous = old.native_inventory(qc)
    current = acceptance.native_inventory(qc)
    assert not current['phase_supported']
    assert current['ordered_native_ir'] == previous['ordered_native_ir']
    # Qiskit normalizes a negative phase modulo 2pi on assignment; the exact
    # stored binary64 phase, rather than the caller's pre-normalization value,
    # is the input to both the raw Operator and native IR.
    assert current['global_phase_raw'] == str(F(float(qc.global_phase)))
    assert current['nearest_W_phase_distance_upper'] == previous['global_phase_error_upper']
    assert current['global_phase_error_upper'] == '0'
    assert current['scalar_phase_implementation_cost_unknown'] is True
    assert current['global_phase_lowering_applied'] is False
    error, _ = native_evaluation.operator_numeric_guard(qc, target, F(0))
    error += F(current['angle_lowering_error_upper'])+F(current['global_phase_error_upper'])
    assert error < F(1,5000000)
    current['numeric_operator_error_upper'] = str(error)
    priced = t_completion(current, {})
    assert priced['actual_total_T'] is None
    assert priced['fixed_plan_completion_T_lower'] is None
    assert priced['full_circuit_synthesis_certified'] is False
    assert priced['classification'] == 'T_INCONCLUSIVE'
    # Native E correctness does not turn unknown phase implementation into free T.
    assert priced['all_keys_covered'] is True


@pytest.mark.parametrize('mutation', ['phase', 'RZ_sign', 'CX_wires'])
def test_wrong_phase_sign_or_wires_still_rejected_with_unsupported_raw_phase(mutation):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    def circuit(reverse=False, sign=1, phase=.073):
        q = QuantumCircuit(2);q.rz(sign*.117,0)
        q.cx(1,0) if reverse else q.cx(0,1)
        q.global_phase = phase
        return q
    target = Operator(circuit()).data.copy()
    bad = circuit(reverse=mutation=='CX_wires', sign=-1 if mutation=='RZ_sign' else 1,
                  phase=.099 if mutation=='phase' else .073)
    inv = acceptance.native_inventory(bad)
    assert inv['phase_supported'] is False
    error, _ = native_evaluation.operator_numeric_guard(bad, target, F(0))
    assert error > F(1,5000000)


@pytest.mark.parametrize('phase', [0., math.pi/4, math.pi/2])
def test_supported_W_lowering_distance_is_still_charged(phase):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1);qc.x(0);qc.global_phase = phase
    before = old.native_inventory(qc)
    after = acceptance.native_inventory(qc)
    assert after['phase_supported'] and after['global_phase_lowering_applied']
    assert after['global_phase_error_upper'] == before['global_phase_error_upper']
    assert after['nearest_W_phase_distance_upper'] == before['global_phase_error_upper']
    assert after['scalar_phase_implementation_cost_unknown'] is False
