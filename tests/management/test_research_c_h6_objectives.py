"""Six-orbital artificial objective/codec test; no molecule or fit is run."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_near_transfer as transfer
from trottertracks import c_pr_aware_df as pr
from trottertracks import c_near_objective as entry
from trottertracks.c1_h6_initialization import reconstruct


@pytest.mark.parametrize('arm', ['PR', 'ENTRY'])
def test_six_orbital_normalization_codec_and_gradient(arm):
    import jax
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(7331)
    z = rng.normal(size=(14, 6, 6)); j = (z+z.transpose(0, 2, 1))/2
    u = np.stack([np.linalg.qr(x)[0] for x in rng.normal(size=(14, 6, 6))])
    e = rng.normal(size=(14, 6, 6)); e = (e+e.transpose(0, 2, 1))/2
    g = reconstruct(j, u)+.01*reconstruct(e, np.tile(np.eye(6), (14, 1, 1)))
    system = transfer.System(6, 14, 4, 1., 'a'*64)
    with patch('numpy.load', side_effect=AssertionError('NO_SAVED_MOLECULE')):
        specs = transfer.objective_specs(g, j, u, system, strength=.1, epsilon=1e-8)
        a, b, x, _, _ = pr.corrected.codec.roundtrip(j, u)
        np.testing.assert_allclose(reconstruct(a, b), reconstruct(j, u), atol=1e-11, rtol=0)
        oracle = (pr if arm == 'PR' else entry).make_value_and_grad(specs[arm])
        value, gradient = oracle(x, g)
        assert float(value) == pytest.approx(1.1, abs=2e-10)
        assert np.asarray(gradient).shape == (504,) and np.isfinite(gradient).all()
        direction = rng.normal(size=len(x)); direction /= np.linalg.norm(direction)
        h = 1e-6
        fd = (float(oracle(x+h*direction, g)[0])-float(oracle(x-h*direction, g)[0]))/(2*h)
        assert fd == pytest.approx(np.asarray(gradient)@direction, rel=2e-5, abs=2e-5)
