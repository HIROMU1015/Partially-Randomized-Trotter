"""H4 rank7 one initializer/one fit. Existing loss, codec and sector engines.

Initial correctness is required; initial DF budget is diagnostic, not admission.
Permission is supplied in chat, not authenticated by this scientific contract.
Import/default preflight does not decode molecular arrays or start optimization.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import subprocess
import time

from . import c_pr_aware_df_pilot as pilot
from .c1_stage3_molecular import component_check

io, old, objective, molecule = pilot.io, pilot.old, pilot.objective, pilot.molecule
SCOPE='C_H4_CDF7_CHOLESKY_ONE_FIT_DF_V1'
FAMILY='artifacts/research_c_cdf7_onefit'
FRESH=FAMILY+'/2026-10-30/run01'
PREPARE=dict(original_input_load=1,rank7_initializer=1,corrected_codec=1,
             tensor_assessment=2,objective_setup=1)
FIT=dict(fit_input_load=1,optimized_fitter=1,fit_initial_diagnostic=1,
         fit_final_diagnostic=1,tensor_assessment=1)
EVALUATE=dict(snapshots_load=2,sector_context=1,physical_sector=1,
    sector_assessment=3,representation_assessment=3,initial_correctness_gate=1,
    saved_comparison_load=1,comparison=1,tail_diagnostic=1)
LIMITS={}
for group in (PREPARE,FIT,EVALUATE):
    for k,v in group.items():LIMITS[k]=LIMITS.get(k,0)+v
MAX_CALLS=sum(LIMITS.values())
CONDITIONS=dict(atoms=4,adjacent_Angstrom=1.,basis='STO-3G',Nalpha=2,Nbeta=2,
    sector_dimension=36,rank=7,ld=2,gamma=0.,epsilon_Ha=1e-8,
    loss_floor_Ha_squared=1e-14,DF_budget_Ha=.00159/8,
    representation='number',kappa='original_g_corrected_fixed',scalar='original_once',
    original_frame_order=True,optimizer='L-BFGS-B',options=None,tol=1e-8,
    initial_budget_required=False,new_initializers=1,new_fits=1,retry=0,resume=0)
INITIALIZER=dict(max_vecs=7,tol=1e-8,cholesky=True,optimize=False)
ALLOWANCES=dict(component_maxabs=1e-10,rtol=0.,objective_absolute=1e-12,
    objective_relative=1e-8,codec='unchanged real_schur_column_gauge_v1 guards',
    qualified_x0='exact equality;unpack tensor/rotation checks in existing adapter')


def preflight(root,path,*,fit=False):
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['conditions']!=CONDITIONS or c['initializer']!=INITIALIZER
            or c['call_limits']!=LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or c['fresh_output']!=FRESH or c['allowances']!=ALLOWANCES
            or c['guard_policy']!=molecule.GUARD_POLICY
            or c['codec']!=objective.corrected.codec.VERSION
            or c['optimizer_settings']!=dict(objective.corrected.SETTINGS,max_vecs=7)
            or c['objective_policy']!='StageA_gamma0_f_over_max_f0_1e-14_no_penalty'
            or c['subsequent_PR_execution_authorized']):
        raise ValueError('CDF7_FIXED_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('SOURCE_OR_INPUT: '+p)
    if c['original_input'] not in c['saved_inputs']:
        raise ValueError('UNPINNED_ORIGINAL_INPUT')
    if c['saved_inputs'][c['original_input']]!=io.P_H4_SHA256:
        raise ValueError('ORIGINAL_P_H4_IDENTITY')
    if any(p not in c['saved_inputs'] for p in c['comparison_records'].values()):
        raise ValueError('UNPINNED_COMPARISON')
    for name,want in c['fit_packages' if fit else 'evaluation_packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT: '+name)
    if fit:
        dist=importlib.metadata.distribution('ffsim')
        for p,want in c['ffsim_sources'].items():
            if io.sha(dist.locate_file('ffsim/'+p))!=want:raise ValueError('FFSIM_SOURCE: '+p)
    return c


def structure(J,U):
    import numpy as np
    if (J.dtype!=np.float64 or U.dtype!=np.float64 or J.shape!=(7,4,4) or U.shape!=J.shape
            or not np.isfinite(J).all() or not np.isfinite(U).all()
            or not np.array_equal(J,J.transpose(0,2,1))
            or np.max(abs(U.transpose(0,2,1)@U-np.eye(4)))>1e-10):
        raise ValueError('CDF7_ACTUAL_RANK_SYMMETRY_ORTHOGONALITY')


def prepare_worker(root,out,c):
    root,out=Path(root),Path(out);out.mkdir()
    call=old.Calls(out,PREPARE,sum(PREPARE.values()));start=time.monotonic()
    try:
        from .c1_stage1_execution import load_input
        p=call('original_input_load','saved original h/g/c;no generation',lambda:load_input(root/c['original_input']))
        g=p['eri_chemist']
        pilot.arrays(out/'input.npz',h=p['one_mo'],g=g,constant=p['nuclear_repulsion'])
        J,U=call('rank7_initializer','one ffsim explicit Cholesky only',lambda:molecule.factor(g,7,optimize=False))
        structure(J,U);pilot.arrays(out/'before.npz',J=J,U=U)
        a,u,x,report,raw=call('corrected_codec','same initializer through real Schur',
            lambda:objective.corrected.codec.roundtrip(J,U))
        structure(a,u);pilot.arrays(out/'initial.npz',J=a,U=u,x0=x)
        pilot.arrays(out/'codec_parameters.npz',x=x,**raw);io.save(out/'codec.json',report)
        for name,j,v in [('before',J,U),('initial',a,u)]:
            io.save(out/(name+'_tensor.json'),call('tensor_assessment',name,
                lambda j=j,v=v:old.tensor_metrics(g,j,v)))
        info=call('objective_setup','existing StageA gamma0 scale frozen once',
            lambda:objective.scales(g,a,x,ld=2,epsilon=CONDITIONS['epsilon_Ha']))
        io.save(out/'objective.json',info)
    except BaseException as e:
        call.stop.set();io.save(out/'failure.json',dict(error=str(e),first_failure=call.first_failure));raise
    finally:pilot.finish(out,call,start)


def fit_worker(root,out,c):
    import numpy as np
    out=Path(out);out.mkdir();call=old.Calls(out,FIT,sum(FIT.values()));start=time.monotonic()
    try:
        def load():
            return (pilot.old_input(out.parent/'prepare/input.npz'),
                pilot.old_input(out.parent/'prepare/initial.npz'),io.read(out.parent/'prepare/objective.json'))
        p,a,info=call('fit_input_load','same original g and qualified rank7 x0',load)
        gate=io.read(out.parent/'initial_gate.json')
        if gate['status']!='PASS' or gate['initial_budget_is_admission']:
            raise ValueError('INITIAL_CORRECTNESS_GATE')
        structure(a['J'],a['U']);spec=objective.Objective(**info['spec'])
        if (spec.rank,spec.norb,spec.ld,spec.gamma)!=(7,4,2,0.):raise ValueError('CDF7_OBJECTIVE')
        def diagnostic(fun,x,name):
            value,grad=fun(x);molecule.stage1.finite_arrays(value,grad)
            pilot.arrays(out/(name+'_parameters.npz'),x=x,gradient=grad,raw_gradient=np.asarray(grad)*spec.loss_scale)
            d=dict(normalized_objective=float(value),raw_tensor_objective=float(value)*spec.loss_scale,
                loss_scale=spec.loss_scale,gradient_l2=float(np.linalg.norm(grad)),
                gradient_infinity=float(np.max(abs(grad))),
                raw_gradient_l2=float(np.linalg.norm(grad))*spec.loss_scale,
                raw_gradient_infinity=float(np.max(abs(grad)))*spec.loss_scale)
            io.save(out/(name+'_diagnostic.json'),d);return value,grad
        def before(fun,x,state):
            def check():
                value,_=diagnostic(fun,x,'initial')
                expected=io.read(out.parent/'prepare/initial_tensor.json')['objective_from_tensor']/spec.loss_scale
                old.check_objective(float(value),expected);io.save(out/'fitter_codec.json',state['codec'])
            call('fit_initial_diagnostic','actual normalized x0/gradient;no budget admission',check)
        def after(fun,res):
            # Preserve the optimizer's original x/jac even if subsequent checks fail.
            pilot.arrays(out/'optimizer_parameters.npz',x=res.x,jac=res.jac)
            io.save(out/'optimizer.json',dict(success=bool(res.success),status=int(res.status),
                message=str(res.message),fun=float(res.fun),nit=int(res.nit),nfev=int(res.nfev),njev=int(res.njev)))
            def check():
                value,gradient=diagnostic(fun,res.x,'final');molecule.stage1.finite_arrays(res.jac)
                old.check_objective(float(value),float(res.fun))
                component_check(np.asarray(gradient).reshape(1,-1),np.asarray(res.jac).reshape(1,-1),
                    1e-10,label='actual final gradient vs optimizer jac')
            call('fit_final_diagnostic','returned optimizer state;no additional iteration',check)
        J,U,res=call('optimized_fitter','one existing normalized tensor-only rank7 fit',lambda:objective.fit(
            p['g'],a['J'],a['U'],a['x0'],spec,before_optimizer=before,after_optimizer=after))
        structure(J,U);pilot.arrays(out/'final.npz',J=J,U=U,x=res.x,jac=res.jac)
        def tensor():
            value=old.tensor_metrics(p['g'],J,U)
            old.check_objective(float(res.fun),value['objective_from_tensor']/spec.loss_scale)
            return value
        io.save(out/'tensor.json',call('tensor_assessment','independent final tensor reconstruction',tensor))
    except BaseException as e:
        call.stop.set();io.save(out/'failure.json',dict(error=str(e),first_failure=call.first_failure));raise
    finally:pilot.finish(out,call,start)


def initial_gate(before,initial,rows):
    import numpy as np
    r=component_check(before,initial,1e-10,label='codec sector operator fidelity')
    return dict(status='PASS',operator=r,operator_difference_norm=float(np.linalg.norm(before-initial,2)),
        initial_budget_is_admission=False,initial_budget_diagnostic={
            k:bool(v['sector_error']+v['numerical_guard']<=CONDITIONS['DF_budget_Ha']) for k,v in rows.items()})


def compare(rows,tensors,optimizer,saved):
    passed=rows['final']['sector_error']+rows['final']['numerical_guard']<=CONDITIONS['DF_budget_Ha']
    return dict(status='CDF7_DF_ELIGIBLE' if optimizer['success'] and passed else 'CDF7_DF_NOT_MET',
        optimizer_success=optimizer['success'],DF_budget_status='PASS' if passed else 'NOT_MET',
        sector=rows,tensor=tensors,optimizer=optimizer,saved_CDF8=saved,
        tensor_objective_change=tensors['final']['objective_from_tensor']-tensors['initial']['objective_from_tensor'],
        sector_error_change_Ha=rows['final']['sector_error']-rows['initial']['sector_error'],
        initial_budget_is_admission=False,rank=7,ld=2,frame_order=list(range(7)),
        same_initialization_as_CDF8=False,rank_causal_effect=None,global_optimality=None,
        minimal_attainable_rank7_error=None,PF_error=None,RTE_error=None,attenuation=None,
        synthesis_error=None,shots=None,native_price=None,total_cost=None,CA_achieved=None,
        actual_initializer_calls=1,fit_invocations=1,retry=0,resume=0,
        selected_initial_as_fallback=False,subsequent_PR_execution_authorized=False,mandatory_stop=True)


def saved_comparison(root,c):
    records={k:io.read(Path(root)/p) for k,p in c['comparison_records'].items()}
    if records['STOP']['failed'] or records['STOP']['scientific_units']!=30:
        raise ValueError('SAVED_CDF8_STATUS')
    if records['SOURCE']['execution_commit']!='e1085c95f5c01b68566e17d543ace5832cb94061':
        raise ValueError('SAVED_CDF8_SOURCE')
    result=records['result']
    return {role:dict(sector=result['sector'][role],tensor=records[role+'_tensor'],
        optimizer=records[role+'_optimizer'],
        initialization='saved rank8 CDF plus shared StageA x0;not new rank7 Cholesky')
        for role in ('baseline','proposed')}


def run(root,path,out,fit_python):
    import numpy as np
    from threadpoolctl import threadpool_limits
    root=Path(root).resolve();out=Path(out).resolve();c=preflight(root,path)
    if out!=root/FRESH or out.exists():raise ValueError('FRESH_RUN01_ONLY_NO_RETRY_RESUME')
    out.mkdir(parents=True);call=old.Calls(out,EVALUATE,sum(EVALUATE.values()));start=time.monotonic()
    def worker(mode):
        subprocess.run([str(Path(fit_python).absolute()),'-B',str(root/c['entry']),
            '--contract',str(Path(path).absolute()),'--worker',mode,'--output',str(out/mode)],check=True)
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        worker('prepare')
        p,states=call('snapshots_load','original input and before/initial snapshots',lambda:(
            pilot.old_input(out/'prepare/input.npz'),{k:pilot.old_input(out/'prepare'/(k+'.npz')) for k in ('before','initial')}))
        ctx=call('sector_context','physical Nalpha=Nbeta2/sector36',lambda:molecule.SectorCAR(4))
        if ctx.d!=36:raise ValueError('SECTOR36_REQUIRED')
        physical=call('physical_sector','single original-g physical reference',lambda:ctx.physical(p['h'],p['g'],p['constant']))
        plan=old.resource_plan(task_count=2,full_dimension=36);io.save(out/'resource_plan.json',plan)
        def assess(item):
            name,data=item;structure(data['J'],data['U'])
            row,G=call('sector_assessment',name,lambda:molecule.rank_assessment(ctx,p,data['J'],data['U'],physical,7))
            row['status']='SECTOR_EVALUATED'
            rep=call('representation_assessment',name,lambda:molecule.representation_gate(ctx,p,data['J'],data['U'],G))
            io.save(out/(name+'_sector.json'),dict(row,representation=rep));return name,(row,sum(G))
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                assessed=dict(pool.map(assess,states.items()))
        rows={k:v[0] for k,v in assessed.items()}
        io.save(out/'initial_gate.json',call('initial_correctness_gate','codec fidelity only;budget diagnostic',
            lambda:initial_gate(assessed['before'][1],assessed['initial'][1],rows)))
        worker('fit')
        final=call('snapshots_load','single final rank7',lambda:pilot.old_input(out/'fit/final.npz'))
        rows['final']=assess(('final',final))[1][0]
        saved=call('saved_comparison_load','published CDF8 values only;no new evaluation',lambda:saved_comparison(root,c))
        tensors={k:io.read(out/'prepare'/(k+'_tensor.json')) for k in ('before','initial')}
        tensors['final']=io.read(out/'fit/tensor.json')
        result=call('comparison','independent optimizer/DF eligibility',lambda:compare(rows,tensors,io.read(out/'fit/optimizer.json'),saved))
        if result['status']=='CDF7_DF_ELIGIBLE':
            def tail():
                from .c_df_basis_switch import exact_keys
                coefficients=objective.coefficients(final['J'][2:],np)
                keys=exact_keys(final['U'],2)
                return dict(status='AUXILIARY_ONLY',lambda_R=float(np.sum(abs(coefficients))),
                    tail_frames=list(range(2,7)),exact_basis_keys=keys,distinct_exact_tail_keys=len(set(keys)),
                    PF_error=None,RTE_error=None,attenuation=None,native_price=None,shots=None,total_cost=None)
            result['tail_auxiliary']=call('tail_diagnostic','eligible final only;no RTE or circuit',tail)
        io.save(out/'result.json',result)
    except BaseException as e:
        call.stop.set();io.save(out/'failure.json',dict(status='CDF7_TECHNICAL_STOP',error=str(e),first_failure=call.first_failure));raise
    finally:pilot.finish(out,call,start,children=('prepare','fit'))
