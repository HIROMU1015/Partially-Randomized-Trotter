"""Dormant one-fit objective comparison; reuse the existing scientific kernels."""
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess
import time

from . import c_pr_aware_df_pilot as existing
from . import c_rcdf_comparator as objective

io, old, molecule = existing.io, existing.old, existing.molecule
SCOPE = "C_RCDF_COMMON_OPTIMIZER_STAGE_A_V1"
FAMILY = "artifacts/research_c_regularization_comparator/2026-11-01"
PARENT = dict(saved_snapshots_load=1, initial_tensor_assessment=1,
              sector_context=1, physical_sector=1, sector_assessment=2,
              representation_assessment=2, initial_gate=1,
              final_snapshot_load=1, comparison=1)
FIT = dict(fit_input_load=1, objective_setup=1, optimized_fitter=1,
           fit_initial_diagnostic=1, fit_final_diagnostic=1,
           final_tensor_assessment=1)
LIMITS = {**PARENT, **FIT}
MAX_CALLS = sum(LIMITS.values())
CONDITIONS = dict(atoms=4, adjacent_Angstrom=1., basis="STO-3G",
    Nalpha=2, Nbeta=2, sector_dimension=36, rank=8, ld=2,
    representation="number", optimizer="L-BFGS-B", options=None,
    tol=1e-8, strength=0.1, epsilon_Ha=1e-8,
    loss_scale_Ha_squared=2.1839632053132894e-9,
    DF_budget_Ha=.00019875, kappa="original_g_corrected_fixed",
    scalar="original_once", original_frame_order=True,
    mass_comparison_allowance_Ha=1e-10, new_fits=1, retry=0, resume=0)


def preflight(root, path, *, fit=False):
    """Bytes, JSON and package metadata only; no NPZ or numerical imports."""
    root = Path(root).resolve()
    c = io.read(path)
    if (c["scope"] != SCOPE or c["status"] != "UNAPPROVED_PROPOSAL"
            or c["conditions"] != CONDITIONS or c["call_limits"] != LIMITS
            or c["maximum_logical_units"] != MAX_CALLS
            or c["objective"] != objective.VERSION
            or c["stages_B_C_authorized"] is not False
            or c["fresh_output"] != FAMILY+"/run01"
            or c["initialization"] != objective.pr.INITIALIZATION):
        raise ValueError("FIXED_COMPARISON_CONTRACT")
    if io.sha(root/c["source_manifest"]) != c["source_manifest_sha256"]:
        raise ValueError("SOURCE_MANIFEST")
    for group in (io.read(root/c["source_manifest"])["files"], c["saved_inputs"]):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != digest:
                raise ValueError("SOURCE_OR_INPUT: "+name)
    for name in ("input_snapshot", "initial_snapshot", "initial_scales"):
        if c[name] not in c["saved_inputs"]:
            raise ValueError("UNPINNED_INPUT")
    for paths in c["comparators"].values():
        if any(p not in c["saved_inputs"] for p in paths.values()):
            raise ValueError("UNPINNED_COMPARATOR")
    for name, version in c["fit_packages" if fit else "evaluation_packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("ENVIRONMENT: "+name)
    if fit:
        dist = importlib.metadata.distribution("ffsim")
        for name, digest in c["ffsim_sources"].items():
            if io.sha(dist.locate_file("ffsim/"+name)) != digest:
                raise ValueError("FFSIM_SOURCE: "+name)
    return c


def load(root, c):
    import numpy as np
    p = existing.old_input(root/c["input_snapshot"])
    a = existing.old_input(root/c["initial_snapshot"])
    info = io.read(root/c["initial_scales"])
    if (p["h"].dtype != np.float64 or p["h"].shape != (4, 4)
            or not np.isfinite(p["h"]).all()
            or not np.isfinite(p["constant"]).all()
            or a["J"].shape != (8, 4, 4) or a["U"].shape != (8, 4, 4)
            or info["spec"]["rank"] != 8 or info["spec"]["norb"] != 4
            or info["spec"]["ld"] != 2 or info["spec"]["epsilon"] != 1e-8
            or info["spec"]["loss_scale"] != CONDITIONS["loss_scale_Ha_squared"]):
        raise ValueError("QUALIFIED_SNAPSHOT_SCHEMA_OR_SCALE")
    return p, a, info


def fit_worker(root, output, c):
    """A dedicated ffsim process; no evaluations of the old final J/U."""
    import numpy as np
    root, output = Path(root), Path(output)
    output.mkdir()
    calls = old.Calls(output, FIT, sum(FIT.values()))
    start = time.monotonic()
    try:
        if io.read(output.parent/"initial_gate.json")["status"] != "PASS":
            raise ValueError("INITIAL_SECTOR_GATE_REQUIRED")
        p, a, info = calls("fit_input_load", "same qualified Stage A x0",
                           lambda: load(root, c))

        def setup():
            spec = objective.initial_spec(
                a["J"], ld=2, loss_scale=info["spec"]["loss_scale"],
                strength=CONDITIONS["strength"])
            value = objective.components(p["g"], a["x0"], spec)
            if abs(value["normalized_tensor_objective"]-1.) > 1e-8:
                raise ValueError("SAVED_INITIAL_LOSS_SCALE_MISMATCH")
            io.save(output/"objective.json", dict(
                spec=asdict(spec), raw_rho=value["raw_rho"],
                rule="equal_initial_normalized_penalty_not_equal_error_or_gradient"))
            return spec
        spec = calls("objective_setup", "P0=sum ordered initial J squared", setup)

        def diagnostic(fun, x, name):
            v, dg = fun(x)
            value = objective.components(p["g"], x, spec)
            if abs(float(v)-value["combined_objective"]) > 1e-10+1e-8*abs(float(v)):
                raise ValueError("OBJECTIVE_EXPORT_MISMATCH")
            existing.arrays(output/(name+"_parameters.npz"), x=x, gradient=dg)
            io.save(output/(name+"_diagnostic.json"), dict(
                value, gradient_l2=float(np.linalg.norm(dg)),
                gradient_infinity=float(np.max(abs(dg)))))
        before = lambda fun, x, state: calls(
            "fit_initial_diagnostic", "before first optimizer iteration",
            lambda: diagnostic(fun, x, "initial"))
        after = lambda fun, res: calls(
            "fit_final_diagnostic", "finite terminal point even if success=false",
            lambda: diagnostic(fun, res.x, "final"))
        J, U, res = calls("optimized_fitter", "one unchanged ffsim invocation",
            lambda: objective.fit(
                p["g"], a["J"], a["U"], a["x0"], spec,
                before_optimizer=before, after_optimizer=after))
        if not np.isfinite(res.jac).all() or not np.isfinite(res.fun):
            raise ValueError("NONFINITE_RAW_OPTIMIZER_RESULT")
        existing.arrays(output/"final.npz", J=J, U=U, x=res.x, jac=res.jac)
        io.save(output/"optimizer.json", dict(
            success=bool(res.success), status=int(res.status),
            message=str(res.message), fun=float(res.fun), nit=int(res.nit),
            nfev=int(res.nfev), njev=int(res.njev)))
        io.save(output/"tensor.json", calls(
            "final_tensor_assessment", "independent saved-final reconstruction",
            lambda: old.tensor_metrics(p["g"], J, U)))
    except BaseException as error:
        calls.stop.set()
        io.save(output/"failure.json", dict(error=str(error),
                                            first_failure=calls.first_failure))
        raise
    finally:
        existing.finish(output, calls, start)


def decide(initial, final, report, references):
    success = bool(report["optimizer"]["success"])
    budget = final["sector_error"]+final["numerical_guard"] <= CONDITIONS["DF_budget_Ha"]
    mass = report["diagnostic"]["exact_tail_mass_Ha"]
    previous = references["proposed"]["diagnostic"]["exact_tail_mass_Ha"]
    difference = mass-previous
    tolerance = CONDITIONS["mass_comparison_allowance_Ha"]
    relation = ("UNRESOLVED_WITHIN_ENGINEERING_ALLOWANCE" if abs(difference) <= tolerance
                else "LOWER_NUMERICAL_POINT" if difference < 0 else "HIGHER_NUMERICAL_POINT")
    return dict(
        status="STAGE_A_ELIGIBLE_FOR_RESEARCH_DECISION" if success and budget
               else "STAGE_A_NOT_ELIGIBLE",
        optimizer_success=success, DF_budget="PASS" if budget else "NOT_MET",
        initial_sector=initial, final_sector=final, candidate=report,
        saved_references=references, tail_vs_saved_proposed=relation,
        exact_tail_mass_difference_Ha=difference,
        interpretation="One common-optimizer, initial-value-matched pilot; not an error-matched or globally optimized comparison",
        stage_B_requires_separate_decision_and_permission=True,
        stages_B_C_authorized=False, PF_error=None, RTE_error=None,
        attenuation=None, shots=None, native_cost=None, total_cost=None,
        CA_achieved=None, independent_novelty=None, mandatory_stop=True)


def run(root, path, output, fit_python):
    root, output = Path(root).resolve(), Path(output).resolve()
    c = preflight(root, path)
    if output != (root/c["fresh_output"]).resolve() or output.exists():
        raise ValueError("FRESH_RUN01_NO_RETRY_RESUME")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    calls = old.Calls(output, PARENT, sum(PARENT.values()))
    start = time.monotonic()
    try:
        io.save(output/"contract.json", c)
        io.save(output/"SOURCE.json", dict(
            io.read(root/c["source_manifest"]),
            execution_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()))
        io.save(output/"resource_plan.json",
                old.resource_plan(task_count=1, full_dimension=36))
        p, a, _ = calls("saved_snapshots_load", "saved input and qualified initial",
                         lambda: load(root, c))
        io.save(output/"initial_tensor.json", calls(
            "initial_tensor_assessment", "qualified initial, no new decomposition",
            lambda: old.tensor_metrics(p["g"], a["J"], a["U"])))
        ctx = calls("sector_context", "Nalpha=Nbeta=2", lambda: molecule.SectorCAR(4))
        physical = calls("physical_sector", "original-g physical Hamiltonian",
                          lambda: ctx.physical(p["h"], p["g"], p["constant"]))

        def assess(label, data):
            row, G = calls("sector_assessment", label, lambda:
                molecule.rank_assessment(ctx, p, data["J"], data["U"], physical, 8))
            representation = calls("representation_assessment", label, lambda:
                molecule.representation_gate(ctx, p, data["J"], data["U"], G))
            io.save(output/(label+"_sector.json"), dict(row, representation=representation))
            return row
        initial = assess("initial", a)

        def gate():
            if initial["sector_error"]+initial["numerical_guard"] > CONDITIONS["DF_budget_Ha"]:
                raise ValueError("INITIAL_DF_BUDGET_NOT_MET")
            return dict(status="PASS")
        io.save(output/"initial_gate.json", calls("initial_gate", "before fit", gate))
        subprocess.run([
            str(Path(fit_python).absolute()), str(root/c["entry"]),
            "--contract", str(Path(path).absolute()), "--fit-worker",
            "--output", str(output/"fit")], check=True)

        def final_load():
            data = existing.old_input(output/"fit/final.npz")
            report = dict(
                optimizer=io.read(output/"fit/optimizer.json"),
                diagnostic=io.read(output/"fit/final_diagnostic.json"),
                tensor=io.read(output/"fit/tensor.json"))
            references = {role: {name: io.read(root/p) for name, p in files.items()}
                          for role, files in c["comparators"].items()}
            return data, report, references
        final, report, references = calls(
            "final_snapshot_load", "new final plus saved scalar references only",
            final_load)
        row = assess("final", final)
        io.save(output/"result.json", calls("comparison", "Stage A only",
            lambda: decide(initial, row, report, references)))
    except BaseException as error:
        calls.stop.set()
        io.save(output/"failure.json", dict(error=str(error),
                                            first_failure=calls.first_failure))
        raise
    finally:
        existing.finish(output, calls, start, ("fit",))
