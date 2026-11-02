"""Real ffsim pack/unpack/JAX, artificial tensors; no optimizer iterations."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_switch as s


def test_smooth_AD_parameter_layout_and_tensor_roundtrip(monkeypatch):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,'minimize',lambda *a,**kw:pytest.fail('NO OPTIMIZER'))
    from ffsim.linalg.util import df_tensors_from_params_jax
    from trottertracks.c1_real_orthogonal_codec import roundtrip
    J=np.array([[[1.,-.4],[-.4,-.3]],[[.2,.9],[.9,-.7]],[[0.,0.],[0.,0.]]])
    U=np.array([[[0.,-1.],[1.,0.]],[[-1.,0.],[0.,1.]],[[-1.,0.],[0.,-1.]]])
    JJ,UU,x,report,_=roundtrip(J,U)
    spec=s.Partition(s.exact_keys(UU,1),ld=1,epsilon=.02)
    def loss(v):
        a,_=df_tensors_from_params_jax(v,3,2,None,real=True)
        z=s.smooth_terms(a,spec,jnp)
        return (z['mass']/2.3)**2+.7*z['switch_fraction']
    value,grad=jax.jit(jax.value_and_grad(loss))(jnp.asarray(x))
    assert np.isfinite(value) and np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[:3],np.zeros(3)) # fixed-group proxy gives no U gradient
    for i in range(len(x)):
        h=2e-6;d=np.zeros_like(x);d[i]=h
        fd=float((loss(jnp.asarray(x+d))-loss(jnp.asarray(x-d)))/(2*h))
        assert float(grad[i])==pytest.approx(fd,abs=1e-7,rel=1e-6)
    assert report['tensor_frobenius']<report['tensor_allowance']
    zero=jnp.zeros_like(JJ)
    zgrad=jax.grad(lambda a:s.smooth_terms(a,spec,jnp)['switch_fraction'])(zero)
    assert np.isfinite(zgrad).all()
