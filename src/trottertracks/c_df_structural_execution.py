"""Dormant H4 support-restricted one-fit experiment. No downstream science.

Technical imports/preflight are passive. Child fit processes reuse the old
ffsim objective/codec and the parent's old sector/physical/representation code.
"""
from pathlib import Path
import importlib.metadata
import subprocess
import time
from . import c_df_structural_support as support
from . import c_pr_aware_df_pilot as old
io=old.io
SCOPE='C_DF_TAIL_PATH_SUPPORT_STAGE_DF_V1'
FRESH='artifacts/research_c_df_structural_support/2026-10-28/run01'
PREPARE=dict(saved_initial_load=1,support_projection=1,structure_assessment=1,tensor_assessment=1)
FIT=dict(fit_input_load=1,optimized_fitter=1,fit_initial_diagnostic=1,fit_final_diagnostic=1,tensor_assessment=1,structure_assessment=1)
EVAL=dict(saved_evidence_load=1,snapshots_load=2,sector_context=1,physical_sector=1,sector_assessment=2,representation_assessment=2,initial_gate=1,comparison=1)
LIMITS={}
for group in (PREPARE,FIT,EVAL):
    for k,v in group.items():LIMITS[k]=LIMITS.get(k,0)+v
MAX_CALLS=sum(LIMITS.values())
CONDITIONS=dict(atoms=4,adjacent_Angstrom=1.,basis='STO-3G',Nalpha=2,Nbeta=2,sector_dimension=36,rank=8,ld=2,
    representation='number',tail_J_support='diagonal_plus_path_01_12_23',head_J_support='full_symmetric',gamma=.1,epsilon=1e-8,
    original_frame_order=True,kappa='original_g_corrected_fixed',scalar='original_once',optimizer='L-BFGS-B',options=None,
    DF_budget_Ha=.00019875,new_fits=1,retry=0,resume=0)


def preflight(root,path,*,fit=False):
    root=Path(root).resolve();c=io.read(path)
    for k,v in dict(scope=SCOPE,implementation=support.VERSION,conditions=CONDITIONS,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,fresh_output=FRESH).items():
        if c[k]!=v:raise ValueError('STRUCTURAL_CONTRACT:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('STRUCTURAL_SOURCE')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            file=(root/p).resolve()
            if not file.is_relative_to(root) or io.sha(file)!=h:raise ValueError('STRUCTURAL_INPUT_SOURCE:'+p)
    for p,v in c['fit_packages' if fit else 'evaluation_packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('STRUCTURAL_ENVIRONMENT:'+p)
    if fit:
        dist=importlib.metadata.distribution('ffsim')
        for p,h in c['ffsim_sources'].items():
            if io.sha(dist.locate_file('ffsim/'+p))!=h:raise ValueError('FFSIM_SOURCE')
    return c


def worker(root,c,out,role):
    import numpy as np
    from ffsim.linalg.util import df_tensors_from_params
    from . import c_pr_aware_df as objective
    out=Path(out);out.mkdir();limits=PREPARE if role=='prepare' else FIT
    call=old.old.Calls(out,limits,sum(limits.values()));start=time.monotonic();layout=support.Support()
    try:
        if role=='prepare':
            p,initial,scale=call('saved_initial_load','same StageA input/x0/shared scales',lambda:(old.old_input(Path(root)/c['input']),old.old_input(Path(root)/c['initial']),io.read(Path(root)/c['scales'])))
            x=call('support_projection','explicit changed starting tensor,not a codec repair',lambda:layout.project(initial['x0']))
            J,U=df_tensors_from_params(x,8,4,None,real=True)
            io.save(out/'structure.json',call('structure_assessment','projected initial',lambda:layout.assess(J)))
            old.arrays(out/'initial.npz',J=J,U=U,x0=x)
            io.save(out/'tensor.json',call('tensor_assessment','projected initial',lambda:old.old.tensor_metrics(p['g'],J,U)))
            io.save(out/'objective.json',scale)
        elif role=='fit':
            if io.read(out.parent/'initial_gate.json')['status']!='SUPPORT_PASS':raise ValueError('STRUCTURAL_INITIAL_GATE_REQUIRED')
            p,a,scale=call('fit_input_load','one new constrained fit',lambda:(old.old_input(Path(root)/c['input']),old.old_input(out.parent/'prepare/initial.npz'),io.read(out.parent/'prepare/objective.json')))
            spec=objective.Objective(**{**scale['spec'],'gamma':.1})
            def diagnostic(fun,x,label):
                value,grad=fun(x);old.arrays(out/(label+'_parameters.npz'),x=x,unrestricted_gradient=grad,compact_gradient=np.asarray(grad)[list(layout.kept)])
                io.save(out/(label+'_diagnostic.json'),dict(objective.components(p['g'],x,spec),unrestricted_gradient_L2=float(np.linalg.norm(grad)),projected_gradient_L2=float(np.linalg.norm(np.asarray(grad)[list(layout.kept)]))))
            before=lambda fun,x,state:call('fit_initial_diagnostic','full and constrained gradient',lambda:diagnostic(fun,x,'initial'))
            after=lambda fun,res:call('fit_final_diagnostic','full and constrained gradient',lambda:diagnostic(fun,res.x,'final'))
            J,U,res=call('optimized_fitter','one existing ffsim/SciPy solve',lambda:support.fit(p['g'],a['J'],a['U'],a['x0'],spec,layout,before_optimizer=before,after_optimizer=after))
            old.arrays(out/'final.npz',J=J,U=U,x=res.x,projected_gradient=res.jac,raw_optimizer_x=res.restricted_optimizer_x,raw_optimizer_jac=res.restricted_optimizer_jac)
            io.save(out/'optimizer.json',dict(success=bool(res.success),status=int(res.status),message=str(res.message),fun=float(res.fun),nit=int(res.nit),nfev=int(res.nfev),njev=int(res.njev),gradient_convention=res.gradient_convention))
            io.save(out/'tensor.json',call('tensor_assessment','final',lambda:old.old.tensor_metrics(p['g'],J,U)))
            io.save(out/'structure.json',call('structure_assessment','final',lambda:layout.assess(J)))
        else:raise ValueError('WORKER_ROLE')
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error)));raise
    finally:old.finish(out,call,start)


def decision(rows,optimizer,initial,final,saved):
    df=bool(rows['final']['sector_error']+rows['final']['numerical_guard']<=CONDITIONS['DF_budget_Ha'])
    return dict(status='DF_ELIGIBLE_NOT_PERFORMANCE_GO' if df and optimizer['success'] else 'DF_NOT_MET_OR_OPTIMIZER_UNSUCCESSFUL',optimizer_success=optimizer['success'],final_DF_budget_PASS=df,
        initial_DF_budget_PASS=bool(rows['initial']['sector_error']+rows['initial']['numerical_guard']<=CONDITIONS['DF_budget_Ha']),sector=rows,initial_diagnostic=initial,final_diagnostic=final,saved_controls=saved,
        tensor_and_sector_separate=True,mask_is_not_same_H_transformation=True,new_fits=1,PF_RTE_shots_native=0,performance_advantage=None,actual_CA_achieved=None,retry=0,resume=0,mandatory_stop=True)


def run(root,path,fit_python):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH
    if out.exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    out.mkdir(parents=True);call=old.old.Calls(out,EVAL,sum(EVAL.values()));start=time.monotonic()
    def launch(role):
        subprocess.run([str(Path(fit_python).absolute()),'-B',str(root/c['entry']),'--contract',str(Path(path).resolve()),'--worker',role,'--output',str(out/role)],check=True)
    try:
        io.save(out/'resource_plan.json',old.old.resource_plan(task_count=1,full_dimension=36))
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        saved=call('saved_evidence_load','saved StageA controls only,no reanalysis',lambda:io.read(root/c['controls']))
        if saved['status']!='STAGE_A_PROXY_AND_DF_PASS':raise ValueError('SAVED_CONTROL_STATUS')
        launch('prepare')
        p,a=call('snapshots_load','original packet and constrained initial',lambda:(old.old_input(root/c['input']),old.old_input(out/'prepare/initial.npz')))
        ctx=call('sector_context','complete sector36',lambda:old.molecule.SectorCAR(4));physical=call('physical_sector','same original h/g/c',lambda:ctx.physical(p['h'],p['g'],p['constant']))
        def assess(label,data):
            row,G=call('sector_assessment',label,lambda:old.molecule.rank_assessment(ctx,p,data['J'],data['U'],physical,8))
            rep=call('representation_assessment',label,lambda:old.molecule.representation_gate(ctx,p,data['J'],data['U'],G));row['status']='SECTOR_EVALUATED';row['representation']=rep;io.save(out/(label+'_sector.json'),row);return row
        rows={'initial':assess('initial',a)}
        io.save(out/'initial_gate.json',call('initial_gate','support/representation only;initial budget is diagnostic',lambda:support.Support().assess(a['J'])))
        launch('fit')
        a=call('snapshots_load','one final candidate',lambda:old.old_input(out/'fit/final.npz'));rows['final']=assess('final',a)
        result=call('comparison','structural/DF criteria,not cost ranking',lambda:decision(rows,io.read(out/'fit/optimizer.json'),io.read(out/'fit/initial_diagnostic.json'),io.read(out/'fit/final_diagnostic.json'),saved))
        io.save(out/'result.json',result);return result
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:old.finish(out,call,start,('prepare','fit'))
