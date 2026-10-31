"""Dormant paired fits: same saved x0, fixed objectives, common convergence.

Reuse existing snapshot, CAR/sector, fitter, objective and call machinery.
No molecular work at import/preflight; separate scientific permission required.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess
import time

from . import c_common_convergence as policy
from . import c_rcdf_comparator_pilot as previous
from . import c_rcdf_first_order as saved

io,old,existing,molecule = previous.io,previous.old,previous.existing,previous.molecule
SCOPE = "C_RC2_PR_COMMON_CONVERGENCE_PAIRED_V1"
FAMILY = "artifacts/research_c_common_convergence/2026-11-01"
ROLES = ("rc2","pr")
PARENT = dict(saved_snapshots_load=1,initial_qualification_gate=1,
    sector_context=1,physical_sector=1,final_snapshot_load=2,
    sector_assessment=2,representation_assessment=2,comparison=1)
FIT = dict(fit_input_load=1,objective_setup=1,optimized_fitter=1,
    fit_initial_diagnostic=1,optimizer_terminal_export=1,
    independent_terminal_diagnostic=1,final_tensor_assessment=1)
LIMITS = {**PARENT,**{k:2*v for k,v in FIT.items()}}
MAX_CALLS = sum(LIMITS.values())
CONDITIONS = dict(atoms=4,adjacent_Angstrom=1.,basis="STO-3G",Nalpha=2,Nbeta=2,
    sector_dimension=36,rank=8,norb=4,ld=2,strength=.1,gamma=.1,epsilon=1e-8,
    loss_scale=2.1839632053132894e-9,l2_scale=4.410884790420646,
    mass_scale=2.2093719370722074,DF_budget_Ha=.00019875,
    kappa="original_g_corrected_fixed",scalar="original_once",original_frame_order=True,
    new_fit_maximum_per_arm=1,new_fits_maximum=2,warm_start=False,retry=0,resume=0,
    mass_comparison_allowance_Ha=1e-10)


def preflight(root,path,*,fit=False):
    root = Path(root).resolve(); c = io.read(path)
    if (c["scope"]!=SCOPE or c["conditions"]!=CONDITIONS or c["options"]!=policy.OPTIONS
            or c["criteria"]!=policy.CRITERIA or c["call_limits"]!=LIMITS
            or c["maximum_logical_units"]!=MAX_CALLS or c["fresh_output"]!=FAMILY+"/run01"):
        raise ValueError("FIXED_COMMON_CONVERGENCE_CONTRACT")
    if io.sha(root/c["source_manifest"])!=c["source_manifest_sha256"]:
        raise ValueError("SOURCE_MANIFEST")
    for group in (io.read(root/c["source_manifest"])["files"],c["saved_inputs"]):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest:
                raise ValueError("SOURCE_OR_INPUT: "+name)
    names=[c[k] for k in ("input_snapshot","initial_snapshot","initial_scales","initial_sector")]
    names += [p for row in c["comparators"].values() for p in row.values()]
    if any(p not in c["saved_inputs"] for p in names):raise ValueError("UNPINNED_INPUT")
    for name,version in c["fit_packages" if fit else "evaluation_packages"].items():
        if importlib.metadata.version(name)!=version:raise ValueError("ENVIRONMENT: "+name)
    if fit:
        dist=importlib.metadata.distribution("ffsim")
        for name,digest in c["ffsim_sources"].items():
            if io.sha(dist.locate_file("ffsim/"+name))!=digest:raise ValueError("FFSIM_SOURCE")
    return c


def setup(role,info):
    spec=info["spec"]
    for key in ("rank","norb","ld","loss_scale","mass_scale","epsilon"):
        if spec[key]!=CONDITIONS[key]:raise ValueError("FIXED_INITIAL_SCALE")
    if role=="rc2":
        return previous.objective,previous.objective.Objective(8,4,2,
            CONDITIONS["loss_scale"],CONDITIONS["l2_scale"],.1)
    if role=="pr":
        return previous.objective.pr,previous.objective.pr.Objective(8,4,2,
            CONDITIONS["loss_scale"],CONDITIONS["mass_scale"],.1)
    raise ValueError("REGISTERED_ARM_REQUIRED")


def fit_worker(root,output,c,role):
    import numpy as np
    root,output=Path(root),Path(output); output.mkdir()
    calls=old.Calls(output,FIT,sum(FIT.values())); start=time.monotonic()
    try:
        if io.read(output.parent/"initial_gate.json")["status"]!="PASS":
            raise ValueError("INITIAL_QUALIFICATION_REQUIRED")
        p,a,info=calls("fit_input_load",role,lambda:previous.load(root,c))
        def prepare():
            module,spec=setup(role,info)
            if not np.isclose(previous.objective.full_l2(a["J"],np),CONDITIONS["l2_scale"],
                              atol=0.,rtol=1e-12):raise ValueError("INITIAL_P0_CHANGED")
            io.save(output/"objective.json",dict(spec=asdict(spec),options=policy.OPTIONS,
                                                 criteria=policy.CRITERIA))
            return module,spec
        module,spec=calls("objective_setup",role,prepare)
        def before(fun,x,state):
            def assess():
                v,dg=fun(x); components=module.components(p["g"],x,spec)
                if abs(v-components["combined_objective"])>1e-10+1e-8*abs(v):
                    raise ValueError("INITIAL_OBJECTIVE_MISMATCH")
                existing.arrays(output/"initial_parameters.npz",x=x,gradient=dg)
                io.save(output/"initial_diagnostic.json",dict(components=components,
                    first_order=policy.first_order(np.asarray(dg),8,4)))
            return calls("fit_initial_diagnostic",role,assess)
        def after(fun,result):
            def export():
                existing.arrays(output/"optimizer_raw.npz",x=result.x,jac=result.jac,
                                fun=np.asarray(result.fun))
                finite=bool(np.isfinite(result.fun))
                io.save(output/"optimizer.json",dict(success=bool(result.success),
                    status=int(result.status),message=str(result.message),
                    fun=float(result.fun) if finite else None,raw_fun_repr=repr(result.fun),
                    nit=int(result.nit),nfev=int(result.nfev),njev=int(result.njev)))
                if not finite:raise ValueError("NONFINITE_OPTIMIZER_TERMINAL")
            calls("optimizer_terminal_export",role,export)
            def diagnose():
                value,gradient=module.make_value_and_grad(spec)(np.asarray(result.x),p["g"])
                assessment=policy.terminal(value,np.asarray(gradient),result,8,4)
                components=module.components(p["g"],result.x,spec)
                if abs(float(value)-components["combined_objective"])>1e-10+1e-8*abs(float(value)):
                    raise ValueError("TERMINAL_COMPONENT_OBJECTIVE_MISMATCH")
                existing.arrays(output/"terminal_independent.npz",x=result.x,gradient=gradient)
                io.save(output/"final_diagnostic.json",dict(components=components,assessment=assessment))
            calls("independent_terminal_diagnostic",role,diagnose)
        J,U,res=calls("optimized_fitter",role,lambda:policy.fit(module,p["g"],a["J"],
            a["U"],a["x0"],spec,before_optimizer=before,after_optimizer=after))
        existing.arrays(output/"final.npz",J=J,U=U,x=res.x,jac=res.jac)
        def tensor():
            saved.validate_point(dict(J=J,U=U,x=res.x),8,4,rotation_atol=1e-10)
            return old.tensor_metrics(p["g"],J,U)
        io.save(output/"tensor.json",calls("final_tensor_assessment",role,tensor))
    except BaseException as error:
        calls.stop.set();io.save(output/"failure.json",dict(error=str(error),first_failure=calls.first_failure))
        raise
    finally:existing.finish(output,calls,start)


def decide(rows,reports,references):
    budget={k:rows[k]["sector_error"]+rows[k]["numerical_guard"]<=CONDITIONS["DF_budget_Ha"] for k in ROLES}
    qualified={k:reports[k]["diagnostic"]["assessment"]["qualified"] for k in ROLES}
    status=("DF_BUDGET_NOT_MET" if not all(budget.values()) else
        "ONE_OR_BOTH_UNQUALIFIED" if not all(qualified.values()) else "BOTH_CONVERGENCE_CRITERIA_MET")
    masses={k:reports[k]["diagnostic"]["components"]["exact_tail_mass_Ha"] for k in ROLES}
    delta=masses["pr"]-masses["rc2"]
    relation=None
    if status=="BOTH_CONVERGENCE_CRITERIA_MET":
        relation=("UNRESOLVED_WITHIN_ENGINEERING_ALLOWANCE" if abs(delta)<=CONDITIONS["mass_comparison_allowance_Ha"]
                  else "PR_LOWER_NUMERICAL_TAIL" if delta<0 else "RC2_LOWER_NUMERICAL_TAIL")
    return dict(status=status,DF_budget=budget,qualified=qualified,reports=reports,
        sectors=rows,saved_scalar_references=references,exact_tail_mass_Ha=masses,
        descriptive_PR_minus_RC2_Ha=delta,qualified_tail_comparison=relation,
        rigorous_stationarity=False,optimal_strength=None,PF_error=None,RTE_error=None,
        attenuation=None,shots=None,native_cost=None,total_cost=None,CA_achieved=None,
        stage_B_requires_separate_permission=True,mandatory_stop=True)


def launch_workers(root,path,output,c,fit_python,workers,calls):
    def launch(role):
        if calls.stop.is_set():raise RuntimeError("DEPENDENT_LAUNCH_STOPPED")
        try:
            with (output/(role+"_worker.txt")).open("x") as log:
                subprocess.run([str(Path(fit_python).absolute()),str(root/c["entry"]),
                    "--contract",str(Path(path).absolute()),"--fit-worker",role,
                    "--output",str(output/role)],stdout=log,stderr=log,check=True)
        except BaseException:
            calls.stop.set();raise
    failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(launch,k) for k in ROLES]):
            try:future.result()
            except BaseException as error:failures.append(str(error))
    if failures:raise RuntimeError("TECHNICAL_WORKER_FAILURE: "+repr(failures))


def run(root,path,output,fit_python):
    root,output=Path(root).resolve(),Path(output).resolve();c=preflight(root,path)
    if output!=(root/c["fresh_output"]).resolve() or output.exists():raise ValueError("FRESH_RUN01_NO_RETRY_RESUME")
    output.mkdir(parents=True);calls=old.Calls(output,PARENT,sum(PARENT.values()));start=time.monotonic()
    try:
        io.save(output/"contract.json",c);io.save(output/"SOURCE.json",io.read(root/c["source_manifest"]))
        plan=old.resource_plan(task_count=2,full_dimension=36)
        # Twice the observed 387852 KiB saved-diagnostic peak: an estimate,
        # not a memory quota. Factor accounts for two live AD graphs/overhead.
        estimate=2*387852*1024
        if plan["available_memory_bytes"] is not None:
            plan["workers"]=min(plan["workers"],max(1,plan["available_memory_bytes"]//estimate))
        plan["estimated_fit_bytes_per_worker"]=estimate
        io.save(output/"resource_plan.json",plan)
        p,a,_=calls("saved_snapshots_load","common original x0",lambda:previous.load(root,c))
        def initial():
            row=io.read(root/c["initial_sector"])
            if row["sector_error"]+row["numerical_guard"]>CONDITIONS["DF_budget_Ha"]:
                raise ValueError("SAVED_INITIAL_DF_BUDGET")
            # Parent evaluation venv has no ffsim. Worker codec owns x0 unpack.
            # Bytes are the same qualified snapshot; do not regenerate/logm.
            return dict(status="PASS",sector=row,source="saved_same_qualified_x0")
        io.save(output/"initial_gate.json",calls("initial_qualification_gate","reuse initial sector PASS",initial))
        ctx=calls("sector_context","sector36",lambda:molecule.SectorCAR(4))
        physical=calls("physical_sector","original g",lambda:ctx.physical(p["h"],p["g"],p["constant"]))
        launch_workers(root,path,output,c,fit_python,plan["workers"],calls)
        rows,reports={},{}
        for role in ROLES:
            def load(k=role):
                data=existing.old_input(output/k/"final.npz")
                # Final pack/unpack fidelity was checked in the fit venv.
                report={name:io.read(output/k/file) for name,file in dict(
                    optimizer="optimizer.json",diagnostic="final_diagnostic.json",tensor="tensor.json").items()}
                return data,report
            final,reports[role]=calls("final_snapshot_load",role,load)
            row,G=calls("sector_assessment",role,lambda:molecule.rank_assessment(ctx,p,final["J"],final["U"],physical,8))
            representation=calls("representation_assessment",role,lambda:molecule.representation_gate(ctx,p,final["J"],final["U"],G))
            rows[role]=dict(row,representation=representation);io.save(output/(role+"_sector.json"),rows[role])
        def compare():
            refs={k:{label:io.read(root/p) for label,p in files.items()} for k,files in c["comparators"].items()}
            return decide(rows,reports,refs)
        io.save(output/"result.json",calls("comparison","common policy; no Stage B",compare))
    except BaseException as error:
        calls.stop.set();io.save(output/"failure.json",dict(error=str(error),first_failure=calls.first_failure))
        io.save(output/"result.json",dict(status="TECHNICAL_FAILURE",error=str(error),total_cost=None,mandatory_stop=True))
        raise
    finally:existing.finish(output,calls,start,ROLES)
