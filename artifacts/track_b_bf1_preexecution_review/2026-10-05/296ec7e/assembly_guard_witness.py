"""Synthetic assembly-guard counterexample; no BF-1 science input or search.

Run with the frozen CPU interpreter, PYTHONDONTWRITEBYTECODE=1 and all four
BLAS/Numba thread settings equal to one. This script prints JSON only.
It does not modify the source, domain, preparation packet or one-shot registry.
"""
import json
import math
from pathlib import Path
import sys

import mpmath as mp
import numpy as np

ROOT = Path(__file__).absolute().parents[4]
sys.path.insert(0, str(ROOT / 'src'))

from trottertracks.algorithm_codesign.domain import Point
from trottertracks.algorithm_codesign.pilot import Evaluator, Task


def main():
    mp.mp.dps = 80
    domain_path = ROOT / 'artifacts/track_b_bf1_preparation/2026-10-05/v1/domain_manifest.json'
    frozen = json.loads(domain_path.read_text())
    point = Point.from_record(next(p for p in frozen['initial_points'] if p['label'] == 'Suzuki5'))
    reference_generator = 5.0
    supplied_generator = reference_generator + 1e-6
    # Sterbenz subtraction is exact here. Round the positive error bound up.
    assembly_bound = math.nextafter(abs(supplied_generator - reference_generator), math.inf)
    matrices = {f'D{i}': np.zeros((1, 1), dtype=complex) for i in range(4)}
    matrices['R'] = np.array([[supplied_generator]], dtype=complex)
    evaluator = Evaluator(Task(matrices, 0., supplied_generator,
                               np.array([1.], dtype=complex), assembly_bound))
    actual = evaluator.finite_cell(point, 1, 5, 2)
    T = mp.mpf('.8')
    x = mp.mpf(reference_generator)
    weights = [w.value(point.basis) for w in point.weights]

    def P3(v):
        return mp.fsum(v**n / mp.factorial(n) for n in range(4))

    finite_reference = mp.fprod(P3(-1j*T*w*x) for w in weights)
    target_reference = mp.exp(-1j*T*x)
    correct_bias = np.array([float(abs(mp.re(finite_reference - target_reference))),
                             float(abs(mp.im(finite_reference - target_reference)))])
    bias_error = np.abs(np.array(actual['bias']) - correct_bias)
    report = dict(
        schema='bf1_synthetic_assembly_guard_witness_v1',
        source_commit='296ec7e4c025f09e4bfda96e56e08d32388b81c5',
        scope='synthetic_1x1_assembly_perturbation_not_H4_science',
        point=point.label, coefficient_identity=point.identity,
        T='.8', q=1, R_bud=5, K=2, allocation=actual['allocation'],
        deterministic_generators='four zero 1x1 matrices',
        reference_tail_generator=reference_generator,
        supplied_tail_generator=supplied_generator,
        actual_generator_error=abs(supplied_generator-reference_generator),
        supplied_assembly_bound=assembly_bound,
        supplied_bound_covers_generator_error=assembly_bound >= abs(supplied_generator-reference_generator),
        reported_bias=actual['bias'], reference_bias=correct_bias.tolist(),
        reported_u_signal=actual['u_signal'], absolute_bias_error_by_axis=bias_error.tolist(),
        error_exceeds_reported_u_by_axis=(bias_error > actual['u_signal']).tolist(),
        max_error_over_u=float(np.max(bias_error)/actual['u_signal']),
        witness_found=bool(np.any(bias_error > actual['u_signal'])),
        implication='The current assembly-to-signal guard fails this synthetic perturbation; no claim about actual H4 errors.',
        molecular_input_operations=0, science_signals=0, coefficient_searches=0,
        trajectories=0, circuits_built=0, compilations=0, gpu_queries=0,
        science_execution_authorized=False, BF1_executed=False)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
