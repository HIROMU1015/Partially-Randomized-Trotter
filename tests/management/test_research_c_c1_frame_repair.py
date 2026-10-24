"""Artificial full-mixing CAR checks, including a fresh parallel process.

No molecular packet, saved fit/export, optimizer or circuit is read or run.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_stage1 as stage
from trottertracks import c1_general_k_adapter as adapter


def artificial_frames(complex_rotation=False):
    """Eight dense 4-orbital rotations and signed, non-square general J."""
    matrices, rotations = [], []
    for i in range(8):
        U = np.eye(4)
        for step, (p, q) in enumerate([(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]):
            angle = (i+1)*(step+2)/11
            c, t = np.cos(angle), np.sin(angle)
            rotation = np.eye(4)
            rotation[p, p] = rotation[q, q] = c
            rotation[p, q], rotation[q, p] = t, -t
            U = U@rotation
        if complex_rotation:
            U = np.diag(np.exp(1j*np.arange(4)/7))@U
        J = np.arange(1., 17).reshape(4, 4)/31
        J = (J+J.T)*(i+1)/10
        J[0, 1] = J[1, 0] = -.1*(i+1)
        matrices.append(J)
        rotations.append(U)
    return np.array(matrices), np.array(rotations)


def make_model(Js, Us):
    return adapter.from_ffsim_number_arrays(
        one_body_tensor=np.zeros((4, 4)), diag_coulomb_mats=Js,
        orbital_rotations=Us, constant=0., z_representation=False,
        one_body_role='ffsim_corrected_kappa', frame_ids=[f'F{i}' for i in range(8)],
        group_ids=[f'G{i}' for i in range(8)], source_identity='ARTIFICIAL',
        input_identity='ARTIFICIAL', enable_general_k=True)


def loop_reference(J, U, bilinears):
    """Direct spatial density sums; no einsum, K expansion, or reshaping."""
    densities = []
    for i in range(4):
        density = np.zeros_like(bilinears[0, 0])
        for p in range(4):
            for q in range(4):
                for spin in (0, 1):
                    density += U[p, i]*U[q, i].conjugate()*bilinears[2*p+spin, 2*q+spin]
        densities.append(density)
    result = np.zeros_like(densities[0])
    for i in range(4):
        for j in range(4):
            result += .5*J[i, j]*(densities[i]@densities[j])
    return result


def legacy_raw_frame(J, U, bilinears):
    """Frozen raw_frame contractions from f65bbba4d (test-only reproducer)."""
    block = bilinears[0::2, 0::2]+bilinears[1::2, 1::2]
    densities = np.einsum('pi,qi,pqab->iab', U, U.conj(), block, optimize=True)
    weighted = np.einsum('ij,jab->iab', J, densities, optimize=True)
    return .5*np.einsum('iab,ibc->ac', densities, weighted, optimize=True)


def parallel_probe(legacy=False):
    """Cold concurrent raw and K calls; compare after workers have completed."""
    Js, Us = artificial_frames()
    model = make_model(Js, Us)
    bilinears = stage.car_bilinears(8)
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    workers = min(len(Js), affinity)
    threads = max(1, affinity//workers)

    def pair(i):
        raw = legacy_raw_frame(Js[i], Us[i], bilinears) if legacy else stage.raw_frame(Js[i], Us[i], bilinears)[0]
        return raw, stage.candidate_frame(model.frames[i])[0]

    with threadpool_limits(limits=threads), ThreadPoolExecutor(max_workers=workers) as executor:
        parallel = list(executor.map(pair, range(len(Js))))
    serial = [stage.raw_frame(Js[i], Us[i], bilinears)[0] for i in range(len(Js))]
    # Check against an explicit coefficient/pair loop as well, after concurrency.
    with threadpool_limits(limits=affinity):
        references = [loop_reference(J, U, bilinears) for J, U in zip(Js, Us)]
    return dict(legacy=legacy, workers=workers, blas_threads=threads,
                raw_version=stage.RAW_FRAME_VERSION,
                frames=[dict(frame=i, raw_vs_serial=float(np.max(np.abs(raw-serial[i]))),
                             raw_vs_loop=float(np.max(np.abs(raw-references[i]))),
                             K_vs_loop=float(np.max(np.abs(candidate-references[i]))),
                             raw_vs_K=float(np.max(np.abs(raw-candidate))))
                        for i, (raw, candidate) in enumerate(parallel)])


@pytest.mark.parametrize('complex_rotation', [False, True])
def test_full_mixing_general_J_matches_direct_CAR_loop_and_K(complex_rotation):
    Js, Us = artificial_frames(complex_rotation)
    model = make_model(Js, Us)
    bilinears = stage.car_bilinears(8)
    affinity = len(os.sched_getaffinity(0))
    with threadpool_limits(limits=affinity):
        for i in (0, 5, 7):
            # Non-contiguous inputs exercise the explicit reshape boundaries.
            J, U = Js[i][::-1, ::-1], Us[i][:, ::-1]
            raw, scale = stage.raw_frame(J, U, bilinears)
            expected = loop_reference(J, U, bilinears)
            np.testing.assert_allclose(raw, expected, atol=3e-13, rtol=0)
            unpermuted = stage.raw_frame(Js[i], Us[i], bilinears)[0]
            candidate = stage.candidate_frame(model.frames[i])[0]
            np.testing.assert_allclose(raw, unpermuted, atol=3e-13, rtol=0)
            np.testing.assert_allclose(unpermuted, candidate, atol=3e-13, rtol=0)
            assert np.isfinite(scale) and raw.dtype == np.complex128


def test_fresh_parallel_process_agrees_with_serial_loop_and_candidate():
    # A separate interpreter avoids hiding the former cold-start race behind
    # earlier serial calls. This asserts the fixed path, never a flaky old FAIL.
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe'],
                           check=True, capture_output=True, text=True)
    record = json.loads(child.stdout)
    assert record['raw_version'] == stage.RAW_FRAME_VERSION
    assert all(max(row[k] for k in ('raw_vs_serial', 'raw_vs_loop', 'K_vs_loop', 'raw_vs_K')) < 3e-13
               for row in record['frames'])


def test_eight_frame_end_to_end_full256_sector36_and_inventory():
    Js, Us = artificial_frames()
    h = np.diag([-.3, -.1, .2, .4]); g = np.zeros((4,)*4)
    result = stage.evaluate(h, g, .375, h.copy(), Js, Us, np.kron(h, np.eye(2)),
                            np.array([.2]), np.eye(8)[None], nalpha=2, nbeta=2,
                            source_identity='ARTIFICIAL', input_identity='ARTIFICIAL')
    assert result['status'] == 'STAGE1_REPRESENTATION_PASS'
    assert result['calls'] == stage.expected_evaluation_calls(8, 36)
    assert result['raw_frame_version'] == stage.RAW_FRAME_VERSION
    assert sum(row['gate'] for row in result['records']) == 37
    assert all(row['passed'] for row in result['records'] if row['gate'])
    assert sum(result['calls'].values())+8 == 147


if __name__ == '__main__':
    assert sys.argv[1:] in (['--probe'], ['--legacy-probe'])
    print(json.dumps(parallel_probe(legacy=sys.argv[1] == '--legacy-probe')))
