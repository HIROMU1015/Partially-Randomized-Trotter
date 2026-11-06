"""Independent Decimal reference checks on invented small arrays only."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_research_c_h6_entry_stability import decimal_reference
from test_research_c_h6_factored_newton import chart
from trottertracks import c_h6_fit_conditioning as fit
from trottertracks import c_h6_conic_diagnostic as conic


@pytest.mark.parametrize('arm', ['PR', 'ENTRY'])
@pytest.mark.parametrize('asymmetric_target', [False, True])
def test_decimal_primal_and_J_gradient_against_existing_independent_contractions(arm, asymmetric_target):
    c = chart(arm)
    if asymmetric_target:
        # Preserve each actual target entry, including rounding asymmetries.
        c.p.g = c.p.g.copy()
        c.p.g[0, 1, 1, 0] += 1e-5
    J = c.point(np.zeros(c.dimension))[1]
    parts = fit.near.parts_gradient_J if arm == 'ENTRY' else fit.inner.stable.precision.parts_gradient_J
    value, gradient, residual = parts(c.p.g, J, c.p.mapping.U, c.p.spec, extended=True)
    ref = decimal_reference(c, J, digits=60)
    assert ref['objective'] == pytest.approx(value['objective'], abs=1e-12)
    np.testing.assert_allclose(ref['gradient'], gradient, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(ref['residual'], residual, atol=1e-18, rtol=1e-12)
    assert not ref['optimized'] and not ref['scientific_acceptance']


@pytest.mark.parametrize('arm', ['PR', 'ENTRY'])
def test_cone_telemetry_is_the_coordinate_point_used_before_polish(arm):
    c = chart(arm); observed = {}; endpoint = {}
    conic.solve_and_polish(c,
        record_conic=lambda metadata, z: observed.update(z=z),
        record_endpoint=lambda metadata, arrays: endpoint.update(arrays))
    np.testing.assert_array_equal(c.point(observed['z'])[1], endpoint['starting_J'])
    # At a small representable point, the unrounded reference agrees too.
    affine = decimal_reference(c, None, affine_z=observed['z'], digits=60)
    actual = decimal_reference(c, endpoint['starting_J'], digits=60)
    np.testing.assert_allclose(affine['gradient'], actual['gradient'], atol=1e-9)
