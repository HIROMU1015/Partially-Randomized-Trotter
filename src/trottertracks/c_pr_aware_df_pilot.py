"""Dormant paired Stage-A pilot; no PF/RTE/circuit/shot/resource execution.

Uses existing corrected ffsim fitter and SectorCAR/rank_assessment. The new
contract remains a proposal until the user explicitly permits its conditions.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time

from . import c_pr_aware_df as objective
from . import c1_h6_initialization as old

io, molecule = old.io, old.molecule
SCOPE = 'C_PR_AWARE_DF_PAIRED_STAGE_A_V1'
FAMILY = 'artifacts/research_c_pr_aware_df'
PREPARE = dict(original_input_load=1, saved_initial_load=1, corrected_codec=1,
               tensor_assessment=2, objective_setup=1)
FIT = dict(fit_input_load=1, optimized_fitter=1, fit_initial_diagnostic=1,
           fit_final_diagnostic=1, tensor_assessment=1)
EVALUATE = dict(snapshots_load=2, sector_context=1, physical_sector=1,
    sector_assessment=4, representation_assessment=4, initial_gate=1, comparison=1)
LIMITS = dict(PREPARE)
for k, n in FIT.items(): LIMITS[k] = LIMITS.get(k, 0) + 2*n
for k, n in EVALUATE.items(): LIMITS[k] = LIMITS.get(k, 0) + n
MAX_CALLS = sum(LIMITS.values())
CONDITIONS = dict(atoms=4, adjacent_Angstrom=1., basis='STO-3G', Nalpha=2, Nbeta=2,
    sector_dimension=36, rank=8, ld=2, gamma_baseline=0., gamma_proposed=.1,
    epsilon_Ha=1e-8, loss_floor_Ha_squared=1e-14, mass_reduction_target=.01,
    DF_budget_Ha=.00159/8, kappa='original_g_corrected_fixed', scalar='original_once',
    original_frame_order=True, representation='number', optimizer='L-BFGS-B',
    options=None, tol=1e-8, new_fits=2, retry=0, resume=0)


def preflight(root, path, input_path, *, fit=False):
    """Byte/policy/version checks; no molecular values are decoded."""
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_PROPOSAL'
            or c['conditions'] != CONDITIONS or c['call_limits'] != LIMITS
            or c['maximum_logical_units'] != MAX_CALLS or c['stages_B_C_authorized']
            or c['objective'] != objective.VERSION): raise ValueError('PILOT_SCOPE')
    if io.sha(input_path) != c['original_input']['sha256']: raise ValueError('ORIGINAL_INPUT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for p, h in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h: raise ValueError('SOURCE_OR_INPUT: '+p)
    if c['initial_export'] not in c['saved_inputs']: raise ValueError('UNPINNED_INITIAL')
    for name, want in c['fit_packages' if fit else 'evaluation_packages'].items():
        if importlib.metadata.version(name)!=want: raise ValueError('ENVIRONMENT: '+name)
    if fit:
        dist=importlib.metadata.distribution('ffsim')
        for p,h in c['ffsim_sources'].items():
            if io.sha(dist.locate_file('ffsim/'+p))!=h: raise ValueError('FFSIM_SOURCE')
    return c


def finish(output, calls, start, children=()):
    counts=dict(calls.counts); failed=calls.stop.is_set()
    for name in children:
        p=output/name/'STOP.json'
        if p.exists():
            child=io.read(p);failed |= child['failed']
            for k,n in child['call_counts'].items(): counts[k]=counts.get(k,0)+n
    io.save(output/'STOP.json',dict(failed=failed,scientific_units=sum(counts.values()),
        call_counts=counts,retry=0,resume=0,mandatory_stop=True))
    io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,
        process_CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def arrays(path, **values):
    import numpy as np
    with Path(path).open('xb') as f: np.savez(f, **values)


def prepare_worker(root, output, c, input_path):
    import numpy as np
    from .c1_stage1_execution import load_input
    output.mkdir(); call=old.Calls(output,PREPARE,sum(PREPARE.values()));start=time.monotonic()
    try:
        p=call('original_input_load','saved P-H4 h/g/c only',lambda:load_input(input_path))
        def initial():
            with np.load(root/c['initial_export'],allow_pickle=False) as a:
                return a['diag_coulomb_mats'],a['orbital_rotations']
        J,U=call('saved_initial_load','same saved CDF for both fits',initial)
        if J.shape!=(8,4,4) or U.shape!=J.shape: raise ValueError('FIXED_INITIAL_RANK_SHAPE')
        g=p['eri_chemist'];arrays(output/'input.npz',h=p['one_mo'],g=g,constant=p['nuclear_repulsion'])
        arrays(output/'before.npz',J=J,U=U)
        a,u,x,report,_=call('corrected_codec','faithful initial parameterization',lambda:objective.corrected.codec.roundtrip(J,U))
        arrays(output/'initial.npz',J=a,U=u,x0=x)
        for name,aa,uu in [('before',J,U),('initial',a,u)]:
            io.save(output/(name+'_tensor.json'),call('tensor_assessment',name,lambda aa=aa,uu=uu:old.tensor_metrics(g,aa,uu)))
        def setup():
            return objective.scales(g,a,x,ld=2,epsilon=CONDITIONS['epsilon_Ha'])
        io.save(output/'objective.json',call('objective_setup','shared initial scales and codec',setup))
        io.save(output/'codec.json',report)
    except BaseException:
        call.stop.set();raise
    finally: finish(output,call,start)


def fit_worker(root, output, c, role):
    import numpy as np
    output.mkdir();call=old.Calls(output,FIT,sum(FIT.values()));start=time.monotonic()
    try:
        def load():
            return (old_input(output.parent/'prepare/input.npz'),old_input(output.parent/'prepare/initial.npz'),
                    io.read(output.parent/'prepare/objective.json'))
        p,a,info=call('fit_input_load',role,load)
        if io.read(output.parent/'initial_gate.json')['status']!='PASS': raise ValueError('INITIAL_GATE_REQUIRED')
        spec=objective.Objective(**{**info['spec'],'gamma':CONDITIONS['gamma_'+role]})
        def diagnostic(fun,x,name):
            v,dg=fun(x);molecule.stage1.finite_arrays(v,dg)
            value=objective.components(p['g'],x,spec)
            if abs(v-value['combined_objective'])>1e-10+1e-8*abs(v): raise ValueError('OBJECTIVE_EXPORT_MISMATCH')
            arrays(output/(name+'_parameters.npz'),x=x,gradient=dg)
            io.save(output/(name+'_diagnostic.json'),dict(value,gradient_l2=float(np.linalg.norm(dg)),gradient_infinity=float(np.max(abs(dg)))))
        before=lambda fun,x,state:call('fit_initial_diagnostic',role,lambda:diagnostic(fun,x,'initial'))
        after=lambda fun,res:call('fit_final_diagnostic',role,lambda:diagnostic(fun,res.x,'final'))
        J,U,res=call('optimized_fitter',role,lambda:objective.fit(p['g'],a['J'],a['U'],a['x0'],spec,before_optimizer=before,after_optimizer=after))
        arrays(output/'final.npz',J=J,U=U,x=res.x,jac=res.jac)
        io.save(output/'optimizer.json',dict(success=bool(res.success),status=int(res.status),
            message=str(res.message),fun=float(res.fun),nit=int(res.nit),nfev=int(res.nfev),njev=int(res.njev)))
        io.save(output/'tensor.json',call('tensor_assessment',role,lambda:old.tensor_metrics(p['g'],J,U)))
    except BaseException:
        call.stop.set();raise
    finally: finish(output,call,start)


def old_input(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def decide(rows, reports):
    eligible={k:bool(reports[k]['optimizer']['success'] and rows[k]['sector_error']+rows[k]['numerical_guard']<=CONDITIONS['DF_budget_Ha'])
              for k in ('baseline','proposed')}
    a,b=(reports[k]['diagnostic']['exact_tail_mass_Ha'] for k in ('baseline','proposed'))
    reduction=None if a<=0 else (a-b)/a
    passed=all(eligible.values()) and reduction is not None and reduction>=CONDITIONS['mass_reduction_target']
    return dict(status='STAGE_A_PROXY_AND_DF_PASS' if passed else 'STAGE_A_NOT_MET_OR_UNRESOLVED',
        optimizer_success={k:reports[k]['optimizer']['success'] for k in eligible},eligible=eligible,
        exact_tail_mass_reduction=reduction,sector=rows,fit_diagnostics=reports,
        interpretation='Only registered DF/proxy conditions; no PR or total-resource conclusion',
        attenuation=None,PF_error=None,RTE_error=None,synthesis_error=None,shots=None,total_cost=None,
        CA_achieved=None,independent_novelty=None,stages_B_C_authorized=False,mandatory_stop=True)


def run(root, path, output, input_path, fit_python):
    import numpy as np
    from threadpoolctl import threadpool_limits
    root=Path(root).resolve();c=preflight(root,path,input_path)
    output=Path(output).resolve()
    if output!=(root/c['fresh_output']).resolve() or output.exists(): raise ValueError('FRESH_RUN01_NO_RETRY_RESUME')
    output.mkdir();call=old.Calls(output,EVALUATE,sum(EVALUATE.values()));start=time.monotonic()
    children=('prepare','baseline','proposed')
    def worker(role):
        subprocess.run([str(Path(fit_python).absolute()),str(root/c['entry']),'--contract',str(Path(path).absolute()),
            '--input',str(Path(input_path).absolute()),'--worker',role,'--output',str(output/role)],check=True)
    try:
        io.save(output/'contract.json',c);io.save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        worker('prepare')
        p,states=call('snapshots_load','original plus before/initial',lambda:(old_input(output/'prepare/input.npz'),
            {k:old_input(output/'prepare'/(k+'.npz')) for k in ('before','initial')}))
        ctx=call('sector_context','sector36',lambda:molecule.SectorCAR(4))
        physical=call('physical_sector','original-g Hamiltonian',lambda:ctx.physical(p['h'],p['g'],p['constant']))
        plan=old.resource_plan(task_count=2,full_dimension=36);io.save(output/'resource_plan.json',plan)
        rows={};totals={}
        def assess(name,data):
            row,G=call('sector_assessment',name,lambda:molecule.rank_assessment(ctx,p,data['J'],data['U'],physical,8))
            row['status']='SECTOR_EVALUATED'
            rep=call('representation_assessment',name,lambda:molecule.representation_gate(ctx,p,data['J'],data['U'],G))
            io.save(output/(name+'_sector.json'),dict(row,representation=rep));return row,sum(G)
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            for name,data in states.items():rows[name],totals[name]=assess(name,data)
        def gate():
            old.molecule.component_check(totals['before'],totals['initial'],1e-10,label='codec sector fidelity')
            if any(rows[k]['sector_error']+rows[k]['numerical_guard']>CONDITIONS['DF_budget_Ha'] for k in states):raise ValueError('INITIAL_DF_BUDGET_NOT_MET')
            return dict(status='PASS')
        io.save(output/'initial_gate.json',call('initial_gate','before any optimizer',gate))
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:list(pool.map(worker,('baseline','proposed')))
        finals=call('snapshots_load','two final states',lambda:{k:old_input(output/k/'final.npz') for k in ('baseline','proposed')})
        reports={}
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            for k,data in finals.items():
                rows[k],_=assess(k,data)
                reports[k]=dict(optimizer=io.read(output/k/'optimizer.json'),diagnostic=io.read(output/k/'final_diagnostic.json'),tensor=io.read(output/k/'tensor.json'))
        io.save(output/'result.json',call('comparison','registered pair only',lambda:decide(rows,reports)))
    except BaseException as error:
        call.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:finish(output,call,start,children)
