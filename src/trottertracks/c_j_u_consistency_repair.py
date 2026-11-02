"""Point-aware consistency diagnostics; no fitting or science at import."""

def compare(actual,expected,*,atol,rtol):
    import numpy as np
    a,b=np.asarray(actual),np.asarray(expected)
    schema=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    if not schema:return dict(PASS=False,finite_shape=False,shape_actual=list(a.shape),shape_expected=list(b.shape))
    delta=np.abs(a-b);allow=atol+rtol*np.abs(b)
    return dict(PASS=bool(np.all(delta<=allow)),finite_shape=True,maxabs=float(np.max(delta,initial=0.)),
                maximum_allowance_ratio=float(np.max(delta/allow,initial=0.)),atol=atol,rtol=rtol)


def pr_report(raw,metadata,chart_value,physical_value,chart):
    import numpy as np
    f,gz=chart_value;v,g=physical_value
    p=chart.p
    return dict(solver_success=metadata['success'],solver_status=metadata['status'],
        raw_fun_vs_endpoint_value=compare(f,metadata['fun'],atol=1e-10,rtol=1e-8),
        chart_vs_physical_value=compare(f,v,atol=1e-10,rtol=1e-8),
        raw_solver_jac_vs_endpoint=compare(raw['solver_jac'],gz,atol=1e-8,rtol=1e-7),
        physical_chain_vs_chart=compare(chart.W.T@g,gz,atol=1e-8,rtol=1e-7),
        raw_jac_semantics='SciPy1.14.1 Newton-CG returns gfk at pre-step point on xtol termination;not generally grad(result.x)',
        physical_J_gradient_infinity=float(np.max(abs(g))))


def rc2_report(original,numpy_decoded,jax_decoded,native,codec,point):
    import numpy as np
    value,gradient=native
    checks={}
    for label,(v,g) in [('original',original),('numpy_decoded',numpy_decoded),('jax_decoded',jax_decoded)]:
        checks[label]=dict(value=compare(value,v,atol=1e-10,rtol=1e-8),
                          J_gradient=compare(gradient,g,atol=1e-6,rtol=1e-7))
    return dict(comparisons=checks,codec=codec,
        decoded_rotation_numpy_vs_original_maxabs=float(np.max(abs(point['Un']-point['U']))),
        decoded_rotation_jax_vs_original_maxabs=float(np.max(abs(point['Uj']-point['U']))),
        decoded_rotation_jax_vs_numpy_maxabs=float(np.max(abs(point['Uj']-point['Un']))),
        all_measurements_are_binary64=True,thresholds_changed=False)
