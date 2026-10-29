"""Actual ffsim 0.0.83 loss/codec; artificial arrays, no fitter/iteration."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def test_real_ffsim_pack_loss_and_shared_J_U_gradient_pullback(monkeypatch):
    import ffsim
    import jax
    import jax.numpy as jnp
    import scipy.optimize
    import importlib
    from trottertracks.c1_real_orthogonal_codec import roundtrip
    from trottertracks.c1_h6_initialization import reconstruct
    from trottertracks.c_df_shared_basis import merge
    def forbidden(*args,**kwargs):
        raise AssertionError('NO FIT OR OPTIMIZER')
    monkeypatch.setattr(scipy.optimize,'minimize',forbidden)
    monkeypatch.setattr(ffsim.linalg,'double_factorized',forbidden)
    assert jax.config.jax_enable_x64
    J=np.array([[[.7,-.2],[-.2,-.3]],[[.1,.3],[.3,-.2]],[[-.4,.2],[.2,.5]]])
    t=.43;u=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    U=np.array([u,u,[[0.,-1.],[1.,0.]]])
    A,V,info=merge(J,U,((0,1),(2,)),ld=0)
    _,_,x,_,_=roundtrip(J,U);_,_,y,_,_=roundtrip(A,V)
    assert len(x)==3*2**2 and len(y)==2*2**2
    target=reconstruct(A,V)+.013*np.ones((2,)*4)
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    full=d._make_compressed_value_and_grad(3,2,None)
    reduced=d._make_compressed_value_and_grad(2,2,None)
    f,g=full(jnp.asarray(x),jnp.asarray(target))
    fr,gr=reduced(jnp.asarray(y),jnp.asarray(target))
    np.testing.assert_allclose(f,fr,atol=1e-14,rtol=0)
    a=np.array(g)[3:].reshape(3,3);b=np.array(gr)[2:].reshape(2,3)
    np.testing.assert_allclose(a[0],a[1],atol=1e-13)
    np.testing.assert_allclose(a[[0,2]],b,atol=1e-13)
    # Chain rule for one shared rotation: sum its original-fragment derivatives.
    np.testing.assert_allclose([g[0]+g[1],g[2]],gr[:2],atol=1e-13)
    direction=np.zeros_like(x);direction[3:6]=[.3,-.7,.4];direction[6:9]=-np.array([.3,-.7,.4])
    assert abs(float(np.dot(g,direction)))<1e-13
    for h in [-.25,.25]:
        moved,_=full(jnp.asarray(x+h*direction),jnp.asarray(target))
        np.testing.assert_allclose(moved,f,atol=1e-14,rtol=0)
    assert info['effective_J_null_dimensions']==3
