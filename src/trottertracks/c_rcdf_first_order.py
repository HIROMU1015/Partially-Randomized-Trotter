"""Dormant saved-point diagnostic. Reuse both objectives; never call a fitter.

Import/preflight are passive. Evaluating molecular snapshots needs separate
scientific permission. No strength selection or optimization is performed.
"""
from pathlib import Path
import importlib.metadata
import time

from . import c_rcdf_comparator as rc
from . import c1_stage1_execution as io

SCOPE = "C_RC2_PR_SAVED_FIRST_ORDER_V1"
FAMILY = "artifacts/research_c_regularization_first_order/2026-11-01"
LIMITS = dict(snapshots_load=1, parameter_fidelity=3,
              objective_gradient=9, component_diagnostic=1)
MAX_CALLS = 14
POLICY = dict(rank=8, norb=4, ld=2,
    loss_scale=2.1839632053132894e-9, l2_scale=4.410884790420646,
    mass_scale=2.2093719370722074, strength=.1, gamma=.1, epsilon=1e-8,
    rotation_atol=1e-10, gradient_atol=1e-8, gradient_rtol=1e-7,
    first_order_gtol=1e-5, DF_budget_Ha=.00019875,
    new_fits=0, optimizer_iterations=0, retry=0, resume=0)
POINTS = ("initial", "rc2_final", "pr_final")


def preflight(root, path):
    """Hash/JSON/package metadata only; no NumPy imports or NPZ decode."""
    root = Path(root).resolve()
    c = io.read(path)
    if (c["scope"] != SCOPE or c["policy"] != POLICY
            or c["call_limits"] != LIMITS or c["maximum_logical_units"] != MAX_CALLS
            or c["fresh_output"] != FAMILY+"/run01"
            or tuple(c["points"]) != POINTS):
        raise ValueError("FIXED_FIRST_ORDER_CONTRACT")
    if io.sha(root/c["source_manifest"]) != c["source_manifest_sha256"]:
        raise ValueError("SOURCE_MANIFEST")
    for group in (io.read(root/c["source_manifest"])["files"], c["saved_inputs"]):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != digest:
                raise ValueError("SOURCE_OR_INPUT: "+name)
    if any(c[k] not in c["saved_inputs"] for k in ("input_snapshot",)):
        raise ValueError("UNPINNED_INPUT")
    if any(p not in c["saved_inputs"] for p in c["points"].values()):
        raise ValueError("UNPINNED_POINT")
    for name, version in c["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("ENVIRONMENT: "+name)
    dist = importlib.metadata.distribution("ffsim")
    for name, digest in c["ffsim_sources"].items():
        if io.sha(dist.locate_file("ffsim/"+name)) != digest:
            raise ValueError("FFSIM_SOURCE: "+name)
    return c


def validate_point(data, rank, norb, *, rotation_atol):
    import numpy as np
    from ffsim.linalg.util import df_tensors_from_params
    x = data["x0"] if "x0" in data else data["x"]
    J, U = data["J"], data["U"]
    if (x.shape != (rank*norb*norb,) or J.shape != (rank,norb,norb)
            or U.shape != J.shape or any(a.dtype != np.float64 or
            not np.isfinite(a).all() for a in (x,J,U))):
        raise ValueError("SAVED_POINT_SCHEMA")
    a, u = df_tensors_from_params(x, rank, norb, None, real=True)
    err = float(np.max(abs(u-U)))
    if (not np.array_equal(J,J.transpose(0,2,1)) or not np.array_equal(J,a)
            or not np.isfinite(u).all() or err > rotation_atol
            or np.max(abs(U.transpose(0,2,1)@U-np.eye(norb))) > rotation_atol):
        raise ValueError("SAVED_PARAMETER_FIDELITY")
    return x, dict(rotation_maxabs=err, J_exact=True)


def evaluate_points(g, points, rc_spec, pr_spec, call, emit=None):
    """Exactly nine existing value/gradient calls for three saved points."""
    import numpy as np
    functions = dict(base=rc.pr.make_value_and_grad(rc_spec.baseline()),
                     rc2=rc.make_value_and_grad(rc_spec),
                     pr=rc.pr.make_value_and_grad(pr_spec))
    result = {}
    for label in POINTS:
        result[label] = {}
        for name, fun in functions.items():
            def evaluate():
                value, grad = fun(points[label]["x"],g)
                grad = np.asarray(grad,dtype=np.float64)
                if (not np.isfinite(value) or grad.shape != points[label]["x"].shape
                        or not np.isfinite(grad).all()):
                    raise ValueError("NONFINITE_VALUE_OR_GRADIENT")
                if emit is not None: emit(label,name,float(value),grad)
                return float(value),grad
            result[label][name] = call("objective_gradient",label+"/"+name,evaluate)
    return result


def component_diagnostic(J, values, spec, *, gtol, atol, rtol):
    """First-order geometry only. No objective calls, tensor or sector work."""
    import numpy as np
    n, rank = spec.norb, spec.rank
    split = rank*n*(n-1)//2
    b, c, p = (values[k][1] for k in ("base","rc2","pr"))
    cpen, ppen = c-b, p-b
    rows, cols = np.triu_indices(n)
    analytic = np.r_[np.zeros(split), (spec.strength/spec.l2_scale
        *np.where(rows==cols,2.,4.)*J[:,rows,cols]).ravel()]
    if not np.allclose(cpen,analytic,atol=atol,rtol=rtol):
        raise ValueError("L2_COMPONENT_GRADIENT_IDENTITY")
    if any(np.max(abs(q[:split])) > atol for q in (cpen,ppen)):
        raise ValueError("DIRECT_U_PENALTY_NONZERO")
    # Sum a_t=0: identity-tensor transfer, not a step or a new candidate.
    s = J.sum(axis=(1,2)); a = -(s-s.mean())
    direction = np.r_[np.zeros(split), np.repeat(a,len(rows))]

    def norms(v):
        return dict(l2=float(np.linalg.norm(v)), infinity=float(np.max(abs(v))),
            U_l2=float(np.linalg.norm(v[:split])), J_l2=float(np.linalg.norm(v[split:])))

    def relation(v):
        den = float(np.linalg.norm(b)*np.linalg.norm(v))
        return dict(dot=float(b@v), cosine=None if den==0 else float(b@v/den),
            penalty_to_loss_l2=None if np.linalg.norm(b)==0 else
                float(np.linalg.norm(v)/np.linalg.norm(b)))

    def qualification(v):
        size = float(np.max(abs(v))); margin = atol+rtol*max(size,gtol)
        return ("WITHIN_ENGINEERING_GTOL" if size+margin <= gtol else
                "OUTSIDE_ENGINEERING_GTOL" if size-margin > gtol else
                "UNRESOLVED_THRESHOLD_MARGIN")

    return dict(gradients={k:norms(v) for k,v in dict(
        base=b,rc2=c,pr=p,rc2_penalty=cpen,pr_penalty=ppen).items()},
        competition=dict(rc2=relation(cpen),pr=relation(ppen)),
        first_order={"rc2":qualification(c),"pr":qualification(p)},
        identity_null_direction=dict(norm=float(np.linalg.norm(direction)),
            loss_slope=float(b@direction), rc2_penalty_slope=float(cpen@direction),
            pr_penalty_slope=float(ppen@direction),
            zero_direction=bool(np.linalg.norm(direction)==0)),
        rigorous_stationarity=False, historical_stop_cause="NOT_RECONSTRUCTED",
        optimal_strength=None, physical_sector_DF_error_recomputed=False,
        PF_error=None,RTE_error=None,CA_achieved=None,total_cost=None)


def run(root, path, output):
    """Only after separate permission; fresh output and mandatory STOP."""
    from . import c_pr_aware_df_pilot as old
    root, output = Path(root).resolve(), Path(output).resolve()
    c = preflight(root,path)
    if output != (root/c["fresh_output"]).resolve() or output.exists():
        raise ValueError("FRESH_RUN01_NO_RETRY_RESUME")
    output.mkdir(parents=True)
    calls = old.old.Calls(output,LIMITS,MAX_CALLS)
    start = time.monotonic()
    try:
        io.save(output/"contract.json",c)
        io.save(output/"SOURCE.json",io.read(root/c["source_manifest"]))
        def load():
            packet = old.old_input(root/c["input_snapshot"])
            return packet["g"], {k:old.old_input(root/p) for k,p in c["points"].items()}
        g, data = calls("snapshots_load","three saved points",load)
        import numpy as np
        if g.dtype != np.float64 or g.shape != (4,)*4 or not np.isfinite(g).all():
            raise ValueError("TARGET_SCHEMA")
        for label in POINTS:
            x, fidelity = calls("parameter_fidelity",label,
                lambda k=label: validate_point(data[k],8,4,rotation_atol=POLICY["rotation_atol"]))
            data[label]["x"] = x
            io.save(output/(label+"_fidelity.json"),fidelity)
        r = rc.Objective(8,4,2,POLICY["loss_scale"],POLICY["l2_scale"],.1)
        p = rc.pr.Objective(8,4,2,POLICY["loss_scale"],POLICY["mass_scale"],.1)
        def emit(label,name,value,grad):
            old.arrays(output/(label+"_"+name+".npz"),gradient=grad)
            io.save(output/(label+"_"+name+".json"),dict(value=value))
        values = evaluate_points(g,data,r,p,calls,emit)
        def diagnose():
            return {k:component_diagnostic(data[k]["J"],values[k],r,
                gtol=POLICY["first_order_gtol"],atol=POLICY["gradient_atol"],
                rtol=POLICY["gradient_rtol"]) for k in POINTS}
        io.save(output/"result.json",calls("component_diagnostic","both arms",diagnose))
    except BaseException as error:
        calls.stop.set()
        io.save(output/"failure.json",dict(error=str(error),first_failure=calls.first_failure))
        raise
    finally:
        old.finish(output,calls,start)
