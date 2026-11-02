"""Integrated A/B/conditional C. No approval-proof machinery or old-run resume."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata,json,subprocess,sys,time,resource
from . import c_pr_integrated as algorithm
from . import c_pr_small_step_pilot as prior
from . import c_pr_integrated_performance as performance
from .c1_stage3_integrated import Calls
io=prior.io
BASE='b8f1429887af5e1255c0801908a7bc31123d15dd'
SCOPE='C_PR_INTEGRATED_H4_REDUCED_PERFORMANCE_V1'
ENTRY='scripts/run_research_c_pr_integrated.py'
FAMILY='artifacts/research_c_pr_integrated/2026-11-02'
FRESH=FAMILY+'/run01'
LIMITS=dict(saved_load=1,initial_identity=1,physical_context=1,phase_A_assessment=1,
    objective_setup=1,optimization_driver=1,inner_J_solve=579,precision_state=386,
    reduced_admission=193,independent_terminal_assessment=64,physical_assessment=65,
    selection=1,reference_comparison=1,performance_driver=1,export=1)
MAX_CALLS=sum(LIMITS.values())+performance.MAX_CALLS


def preflight(root,path,*,evaluation=False):
    root=Path(root).resolve();c=io.read(path)
    for key,want in dict(base_commit=BASE,scope=SCOPE,entry=ENTRY,fresh_output=FRESH,
        policy=asdict(algorithm.DEFAULT),fractions=list(algorithm.FRACTIONS),max_updates=64,
        objective=prior.OBJECTIVE,DF_budget_Ha=.00019875,target=performance.m4.TARGET,sampling=performance.SAMPLING,
        parent_call_limits=LIMITS,performance_call_limits=performance.LIMITS,
        maximum_logical_units=MAX_CALLS,retry=0,resume=0).items():
        if c[key]!=want:raise ValueError('FIXED_INTEGRATED:'+key)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,h in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=h:raise ValueError('SOURCE_INPUT:'+name)
    for name,v in c['evaluation_packages' if evaluation else 'fit_packages'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('ENVIRONMENT:'+name)
    if not evaluation:
        for p,h in c['ffsim_sources'].items():
            if io.sha(importlib.metadata.distribution('ffsim').locate_file('ffsim/'+p))!=h:raise ValueError('FFSIM_SOURCE')
        algorithm.stable.precision.require_extended()
    return c


class Service:
    def __init__(self,root,path,c,out):
        self.log=(out/'evaluation_worker.log').open('x')
        self.process=subprocess.Popen([c['evaluation_python'],'-B',str(root/ENTRY),'--contract',str(path),
            '--evaluation-worker'],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
    def request(self,operation,**values):
        self.process.stdin.write(json.dumps(dict(operation=operation,**values))+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('EVALUATION_WORKER_EXIT:see evaluation_worker.log')
        response=json.loads(line)
        if not response['ok']:raise RuntimeError('EVALUATION_FAIL:'+response['error'])
        return response['value']
    def close(self):
        try:
            if self.process.poll() is None:self.process.stdin.write('{"operation":"close"}\n');self.process.stdin.flush()
        except BrokenPipeError:pass
        self.process.wait();self.log.close()


def run(root,path,*,service_factory=Service):
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load,save_arrays
    root=Path(root).resolve();path=Path(path).resolve();c=preflight(root,path);out=root/FRESH
    if out.exists():raise ValueError('FRESH_INTEGRATED_NO_RETRY_RESUME')
    out.mkdir(parents=True);call=Calls(out,LIMITS,sum(LIMITS.values()));start=time.monotonic();service=None
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        def inputs():
            packet=load(root/c['input_paths']['input'],('h','g','constant'))
            cold=load(root/c['input_paths']['initial'],('J','U'))
            accepted=load(root/c['accepted']['terminal'],('J','U'))
            audit=load(root/c['accepted']['independent'],('J','U','eta'))
            event=load(root/c['accepted']['event'],('eta',))
            return packet,cold,accepted,audit,event
        packet,cold,accepted,audit,event=call('saved_load','fixed input and accepted candidate',inputs)
        def identity():
            for key in ('J','U'):
                if not np.array_equal(accepted[key],audit[key]):raise ValueError('ACCEPTED_JU_IDENTITY')
            if not np.array_equal(event['eta'],audit['eta']):raise ValueError('ACCEPTED_ETA_IDENTITY')
            if not np.array_equal(algorithm.stable.precision.geometry(cold['U'],event['eta']),accepted['U']):raise ValueError('ACCEPTED_GEOMETRY')
            if accepted['J'].shape!=(8,4,4) or packet['g'].shape!=(4,)*4:raise ValueError('FIXED_H4_SHAPE')
            old=io.read(root/c['accepted']['result']);stop=io.read(root/c['accepted']['STOP'])
            if old['accepted_U_updates']!=1 or old['terminal_label']!='1' or stop['failed'] or stop['scientific_units']!=19:raise ValueError('ACCEPTED_PROVENANCE')
            return dict(status='PASS',event_contains_eta_only=True,JU_from_bound_terminal_and_independent_state=True)
        io.save(out/'input_identity.json',call('initial_identity','eta/JU/SOURCE binding',identity))
        service=service_factory(root,path,c,out)
        call('physical_context','original Hamiltonian and sector36',lambda:service.request('initialize'))
        def physical(label,J,U):
            p=out/(label+'_candidate.npz');save_arrays(p,J=J,U=U)
            return service.request('assess',label=label,path=str(p))
        A=call('phase_A_assessment','saved19-call accepted J/U',lambda:physical('phase_A',accepted['J'],accepted['U']))
        io.save(out/'phase_A.json',A)
        if not A['budget_PASS']:
            result=dict(status='PHASE_A_DF_BUDGET_NOT_MET',phase_A=A,trajectory=None,performance=None)
        else:
            spec=call('objective_setup','unchanged PR objective',lambda:algorithm.stable.pr.Objective(**c['objective']))
            def record(kind,label,data,arrays):
                if arrays is not None:save_arrays(out/(kind+'_'+label+'.npz'),**arrays)
                with (out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,label=label,**data),allow_nan=False)+'\n')
            t=call('optimization_driver','one recentered trajectory,max64 updates',lambda:algorithm.trajectory(
                packet['g'],cold['J'],accepted['U'],spec,physical,call=call,record=record))
            def selection():
                best=t['best']
                if best is None:return None
                save_arrays(out/'selected.npz',J=best['J'],U=best['U'])
                return dict(**best['row'],selection_rule='minimum exact lambda;ties F then earliest index',
                    unconverged=not best['row']['state']['first_order_met'])
            chosen=call('selection','registered physical-qualified minimum lambda',selection)
            metrics={k:v for k,v in t.items() if k not in ('best','terminal')};io.save(out/'trajectory.json',metrics)
            def reference_comparison():
                refs={k:io.read(root/v) for k,v in c['optimization_references'].items()}
                report=dict(provenance=c['optimization_references'],scope='saved finite endpoints,not paired convergence comparison')
                if chosen is not None and 'StageA' in refs:
                    old=refs['StageA']['fit_diagnostics']['proposed']['diagnostic']['exact_tail_mass_Ha']
                    new=chosen['physical']['exact_tail_lambda_Ha']
                    report.update(old_PR_exact_lambda_Ha=old,new_exact_lambda_Ha=new,lambda_decrease_Ha=old-new,new_over_old_lambda=new/old)
                if 'RC2_PR' in refs:report['RC2_PR']={k:v for k,v in refs['RC2_PR'].items() if k in ('status','DF_budget','qualified','sectors')}
                return report
            refs=call('reference_comparison','fixed same-definition exact lambda and evidence',reference_comparison)
            perf=None
            if t['can_performance'] and chosen is not None and chosen['physical']['exact_tail_lambda_Ha']>0:
                perf=call('performance_driver','new candidate PF/RTE/RPE/native',lambda:service.request('performance',label=chosen['id']))
            result=dict(status=t['status'],phase_A=A,trajectory=metrics,selected=chosen,performance=perf,references=refs,
                true_reduced_gradient_qualification=None,certified_descent=None,actual_CA_achieved=None)
        result.update(retry=0,resume=0,mandatory_stop=True)
        call('export','result JSON and qualified selected export',lambda:io.save(out/'result.json',result))
        return result
    except BaseException as e:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(e),first_failure=call.first_failure));raise
    finally:
        if service is not None:service.close()
        cstop=io.read(out/'C/STOP.json') if (out/'C/STOP.json').exists() else dict(scientific_units=0,call_counts={},failed=False)
        counts=dict(call.counts);counts.update({'C.'+k:v for k,v in cstop['call_counts'].items()})
        io.save(out/'STOP.json',dict(failed=call.stop.is_set() or cstop['failed'],scientific_units=sum(counts.values()),
            call_counts=counts,maximum_logical_units=MAX_CALLS,stop_reason=result['status'] if 'result' in locals() else 'CORRECTNESS_OR_EXECUTION_FAILURE',retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
