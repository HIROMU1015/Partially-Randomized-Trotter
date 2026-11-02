"""Invented tensors only; synthetic SciPy iterations are explicitly permitted."""
import importlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from trottertracks import c_rcdf_first_order as m


def artificial(n, rank=3):
    rng = np.random.default_rng(841+n+rank)
    a = rng.normal(size=(rank,n,n))*.04
    J = a+a.transpose(0,2,1)
    U = np.repeat(np.eye(n)[None],rank,axis=0)
    U[0,:,0] *= -1
    U[1,:2,:2] = -np.eye(2)
    J,U,x,_,_ = m.rc.pr.corrected.codec.roundtrip(J,U)
    g = m.rc.pr.corrected.codec.reconstruct(J,U)+1e-3*np.einsum(
        "pq,rs->pqrs",np.eye(n),np.eye(n))
    return g,dict(J=J,U=U,x=x)


@pytest.mark.parametrize("n",[2,4])
def test_real_ffsim_gradients_parts_strength_and_identity_null(n):
    g,point = artificial(n)
    r = m.rc.initial_spec(point["J"],ld=1,loss_scale=.003)
    p = m.rc.pr.Objective(3,n,1,.003,.7,.1)
    calls = []
    def call(kind,label,f):
        calls.append((kind,label)); return f()
    with patch.object(scipy.optimize,"minimize",side_effect=AssertionError("NO_FIT")):
        values = m.evaluate_points(g,{k:point for k in m.POINTS},r,p,call)
    assert len(calls)==9 and len(set(calls))==9
    result = m.component_diagnostic(point["J"],values["initial"],r,
                                  gtol=1e-5,atol=1e-8,rtol=1e-7)
    assert result["gradients"]["rc2_penalty"]["U_l2"] < 1e-12
    assert abs(result["identity_null_direction"]["loss_slope"]) < 1e-12
    assert result["identity_null_direction"]["rc2_penalty_slope"] <= 0
    assert result["CA_achieved"] is None
    for name, base in (("rc2",m.rc.make_value_and_grad(r)),):
        v,d = base(point["x"],g)
        r2 = m.rc.Objective(3,n,1,.003,r.l2_scale,.2)
        v2,d2 = m.rc.make_value_and_grad(r2)(point["x"],g)
        vb,db = values["initial"]["base"]
        np.testing.assert_allclose(d2-db,2*(d-db),atol=1e-12)
        assert float(v2-vb)==pytest.approx(2*float(v-vb))


@pytest.mark.parametrize("n",[2,4])
def test_tensor_null_transfer_and_l2_can_increase_tail(n):
    rng = np.random.default_rng(12+n)
    a = rng.normal(size=(n,n)); a = a+a.T
    U = np.array([np.eye(n),scipy.linalg.expm(.3*(a-np.diag(np.diag(a)))
        *np.subtract.outer(np.arange(n),np.arange(n)))])
    J = np.array([a,.3*a])
    shifted = J+np.array([.2,-.2])[:,None,None]
    rec = m.rc.pr.corrected.codec.reconstruct
    np.testing.assert_allclose(rec(J,U),rec(shifted,U),atol=1e-14)
    # Identical bases: splitting a head core into the tail halves P but raises lambda.
    U[:] = np.eye(n)
    head = np.array([a,np.zeros_like(a)]); balanced = np.array([a/2,a/2])
    np.testing.assert_allclose(rec(head,U),rec(balanced,U),atol=1e-14)
    assert m.rc.full_l2(balanced,np)==pytest.approx(m.rc.full_l2(head,np)/2)
    assert np.abs(m.rc.pr.coefficients(head[1:],np)).sum()==0
    assert np.abs(m.rc.pr.coefficients(balanced[1:],np)).sum()>0


def test_actual_scipy_relative_stop_can_precede_gradient_stop():
    def solve(offset):
        def objective(x):
            return offset+.5*x[0]**2+.5e-4*x[1]**2, np.array([x[0],1e-4*x[1]])
        return scipy.optimize.minimize(objective,[1.,1.],method="L-BFGS-B",
                                       jac=True,callback=None,options=None)
    shifted = solve(1e9); plain = solve(0.); small = solve(.1)
    assert shifted.success and "REL_REDUCTION" in shifted.message
    assert np.max(abs(shifted.jac))>1e-5
    assert np.max(abs(plain.jac))<=1e-5 and np.max(abs(small.jac))<=1e-5
    # Denominator floor: a small constant does not rescale this final-pair rule.
    for offset in (0.,.1):
        previous,final = .02+offset,.02-1e-10+offset
        assert max(abs(previous),abs(final),1)==1
        assert previous-final==pytest.approx(1e-10,abs=1e-16)
    # ftol=0 is not a promise of gradient convergence: zero decrease still qualifies.
    assert (1.-1.)/max(1.,1.,1.) <= 0.


def test_fidelity_mismatch_and_component_failure():
    _,point = artificial(2)
    m.validate_point(point,3,2,rotation_atol=1e-10)
    bad = {**point,"x":point["x"].copy()}; bad["x"][-1]+=.1
    with pytest.raises(ValueError,match="FIDELITY"):
        m.validate_point(bad,3,2,rotation_atol=1e-10)
    spec = m.rc.Objective(3,2,1,1.,1.)
    vals = {k:(0.,np.zeros(12)) for k in ("base","rc2","pr")}
    with pytest.raises(ValueError,match="GRADIENT_IDENTITY"):
        m.component_diagnostic(point["J"],vals,spec,gtol=1e-5,atol=1e-8,rtol=1e-7)


def test_passive_import_has_no_numeric_or_solver_import():
    code = ("import sys;sys.path.insert(0,'src');"
        "import trottertracks.c_rcdf_first_order;"
        "assert not {'numpy','jax','ffsim','scipy.optimize'}&set(sys.modules)")
    subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT,check=True)


def test_zero_direction_and_threshold_margin_remain_unresolved():
    spec = m.rc.Objective(3,2,1,1.,1.)
    v = np.full(12,1e-5)
    values = {k:(0.,v.copy()) for k in ("base","rc2","pr")}
    result = m.component_diagnostic(np.zeros((3,2,2)),values,spec,
                                   gtol=1e-5,atol=1e-8,rtol=1e-7)
    assert result["first_order"]["rc2"]=="UNRESOLVED_THRESHOLD_MARGIN"
    assert result["identity_null_direction"]["zero_direction"]
    assert result["competition"]["rc2"]["cosine"] is None


def test_nonfinite_evaluation_stops_without_more_calls(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    g,point = artificial(2)
    spec = m.rc.Objective(3,2,1,1.,1.)
    pspec = m.rc.pr.Objective(3,2,1,1.,1.,.1)
    def factory(*args):
        return lambda x,g: (float("nan"),np.zeros_like(x))
    calls = Calls(tmp_path,m.LIMITS,m.MAX_CALLS)
    with patch.object(m.rc.pr,"make_value_and_grad",factory),\
            patch.object(m.rc,"make_value_and_grad",factory):
        with pytest.raises(ValueError,match="NONFINITE"):
            m.evaluate_points(g,{k:point for k in m.POINTS},spec,pspec,calls)
    assert calls.stop.is_set() and sum(calls.counts.values())==1


def test_byte_preflight_detects_contract_source_input_and_environment(tmp_path):
    from unittest.mock import Mock
    root = tmp_path
    (root/"saved.bin").write_bytes(b"artificial bytes; never decoded")
    (root/"source.py").write_text("# artificial source\n")
    m.io.save(root/"SOURCE.json",dict(files={"source.py":m.io.sha(root/"source.py")}))
    c = dict(scope=m.SCOPE,policy=m.POLICY,call_limits=m.LIMITS,
        maximum_logical_units=14,fresh_output=m.FAMILY+"/run01",
        points={k:"saved.bin" for k in m.POINTS},input_snapshot="saved.bin",
        saved_inputs={"saved.bin":m.io.sha(root/"saved.bin")},
        source_manifest="SOURCE.json",source_manifest_sha256=m.io.sha(root/"SOURCE.json"),
        packages={"synthetic-package":"1"},ffsim_sources={"fake":"digest"})
    # No ffsim files required: the contract fixture has no installed source pins.
    c["ffsim_sources"] = {}
    m.io.save(root/"contract.json",c)
    with patch.object(m.importlib.metadata,"version",return_value="1"),\
            patch.object(m.importlib.metadata,"distribution",return_value=Mock()):
        m.preflight(root,root/"contract.json")
        with patch.object(m.importlib.metadata,"version",return_value="2"):
            with pytest.raises(ValueError,match="ENVIRONMENT"):
                m.preflight(root,root/"contract.json")
        (root/"source.py").write_text("# changed\n")
        with pytest.raises(ValueError,match="SOURCE_OR_INPUT"):
            m.preflight(root,root/"contract.json")
        (root/"source.py").write_text("# artificial source\n")
        (root/"saved.bin").write_bytes(b"changed")
        with pytest.raises(ValueError,match="SOURCE_OR_INPUT"):
            m.preflight(root,root/"contract.json")


@pytest.mark.parametrize("failure",[None,"schema","gradient"])
def test_artificial_end_to_end_fresh_stop_no_optimizer(tmp_path,failure):
    from trottertracks import c_pr_aware_df_pilot as old
    g,point = artificial(4,8)
    root = tmp_path; output = root/m.FAMILY/"run01"
    packet = {**point,"x":point["x"].copy()}
    if failure=="schema": packet["x"][-1]+=.1
    c = dict(fresh_output=m.FAMILY+"/run01",source_manifest="source.json",
        input_snapshot="g.npz",points={k:k+".npz" for k in m.POINTS})
    m.io.save(root/"source.json",dict(files={}))
    def load(path):
        return dict(g=g) if path.name=="g.npz" else dict(packet)
    original = m.component_diagnostic
    def diagnose(*args,**kwargs):
        if failure=="gradient": raise ValueError("INVENTED_DIAGNOSTIC_FAIL")
        return original(*args,**kwargs)
    with patch.object(m,"preflight",return_value=c),patch.object(old,"old_input",load),\
            patch.object(scipy.optimize,"minimize",side_effect=AssertionError("NO_FIT")),\
            patch.object(m,"component_diagnostic",side_effect=diagnose):
        if failure:
            with pytest.raises(ValueError): m.run(root,root/"contract.json",output)
        else: m.run(root,root/"contract.json",output)
        with pytest.raises(ValueError,match="FRESH"):
            m.run(root,root/"contract.json",output)
    stop = m.io.read(output/"STOP.json")
    assert stop["mandatory_stop"] and stop["retry"]==stop["resume"]==0
    assert stop["scientific_units"]==(2 if failure=="schema" else 14)
    assert stop["failed"] == bool(failure)
    if failure=="gradient": assert (output/"pr_final_pr.npz").exists()
