"""Candidate-bound H6 orchestration over existing numerical engines.

Import is passive. Real arrays/circuits enter only through the separately
permitted scientific runner. Synthetic tests inject invented packets.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import threading
import time

from . import c1_stage1_execution as io
from . import c_h6_validation_preparation as design

UNTESTED = 'NUMERICAL_EQUIVALENCE_NOT_CHECKED'
QUALIFICATION = dict(numerical_coverage='registered sample0 only; ancilla x sector inputs, full outputs',
    unchecked_price_circuit_equivalence_assumed=True, full_Fock_equality_verified=False,
    implemented_PR_law_error=None, actual_CA_achieved=None)


def seed_policy(source, task, candidate, r, role):
    if role not in ('selection', 'final'): raise ValueError('FIXED_COHORT_ROLE')
    return dict(master=2026110606, cohort=f'H6_{task["arm"]}_{candidate}_r{r}_{role}',
        SOURCE=source, role=role, arm=task['arm'], candidate=candidate, r=r,
        samples=32 if role == 'selection' else 128)


class Backend:
    def __init__(self, root, contract, out, call, common, *, threads=1, batch=16):
        self.root, self.c, self.out = Path(root), contract, Path(out)
        self.call, self.common, self.threads, self.batch = call, common, threads, batch
        self.serial = threading.RLock()
        self.shells = common.get('h6_shells')

    def unit(self, stage, label, fn, *, count=1):
        marker=self.c.get('_dependency_stop')
        if marker and (self.root/marker).exists():raise RuntimeError('DEPENDENT_WORK_STOPPED')
        return self.call(stage,label,fn,count=count)

    def prepare(self, task, checkpoint):
        import numpy as np
        from . import c_pr_aware_df_stage_b as b
        from .c_pr_aware_df_stage_b_runner import load
        from .c1_h6_saved_pf import ground_alias
        from .c1_h4_strong_baseline import StrongBuilder
        stage = 'endpoint_reference'
        candidate = checkpoint['candidate_sha256']; df = checkpoint['physical']
        path = self.root/checkpoint['candidate_path']
        if io.sha(path) != candidate or df['candidate_sha256'] != candidate:
            raise ValueError('CANDIDATE_IDENTITY')
        if df['input_sha256'] != self.common['input_identity']:
            raise ValueError('CANDIDATE_INPUT_IDENTITY')
        if not df['budget_PASS']: return None
        folder = self.out/task['id']; folder.mkdir(parents=True, exist_ok=False)
        values = self.unit(stage, task['id']+':load', lambda:load(path, ('J','U')))
        ref = self.unit(stage, task['id']+':reference', lambda:load(
            self.root/df['reference_path'], ('J','U','generators','H')))
        n, rank, ld = (self.common['system'][k] for k in ('norb','rank','ld'))
        d = self.common['ctx'].d
        if (values['J'].shape != (rank,n,n) or
                ref['generators'].shape != (rank+1,d,d) or ref['H'].shape != (d,d)
                or any(not np.array_equal(values[k], ref[k]) for k in ('J','U'))):
            raise ValueError('CANDIDATE_REFERENCE_SHAPE_OR_ARRAY_IDENTITY')
        arm = self.unit(stage, task['id']+':model', lambda:b.prepare_arm(**self.common['p'],
            **values, role='proposed', source_identity=candidate,
            input_identity=self.common['input_identity'], ld=ld))
        for i,g in enumerate(ref['generators']):
            self.unit(stage, task['id']+f':factor{i}', lambda i=i,g=g:b.component_check(
                b.candidate_factor(self.common['ctx'],arm,i),g,1e-10,label='same candidate sector CAR'))
        def ground_call():
            try:return ground_alias(ref['H'],folder,'candidate',distinguish_alias=True)
            except ValueError as error:
                if str(error)!='UNRESOLVED_GROUND_ALIAS':raise
                return None,dict(status='UNRESOLVED_GROUND_ALIAS')
        pair, ground = self.unit(stage, task['id']+':ground',ground_call)
        if pair is None:
            io.save(folder/'ineligible.json',ground);return None
        return dict(task=task,candidate=candidate,df=df,arm=arm,ref=ref,pair=pair,
            ground=pair[1][:,0],ground_report=ground,out=folder,policies={},
            atoms=None,reference_context=None,exact={},pf_errors=None,exponential_cache={},
            checked_tail=None,tail_report=None,
            builder=StrongBuilder(relative_fusion=True))

    def references(self, a, r):
        if type(r) is not int or r not in design.R_VALUES:raise ValueError('FIXED_H6_R_GRID')
        import numpy as np
        from scipy.linalg import expm
        from . import c_pr_aware_df_stage_b as b
        from . import c_pr_aware_df_m4 as m4
        from . import c_rte_r_retuning as ret
        from . import c1_stage3_feasibility as feasible
        from .c_rc2_pr_stage_b import distribution_gate
        stage='endpoint_reference'; tag=a['task']['id']+f':r{r}'
        target=ret.m4.accuracy.AccuracyTarget(**ret.m4.TARGET)
        tail,config,dist=self.unit(stage,tag+':law',lambda:b.configure(a['arm'],Fraction(1,10),r,0))
        if a['atoms'] is None:
            a['atoms']=self.unit(stage,tag+':atoms',lambda:b.atom_operators(self.common['ctx'],a['arm']))
        tail_identity=(tail.tail_hash,tail.lambda_r,tail.components)
        def check_tail():
            if a['checked_tail'] is None:
                a['tail_report']=b.tail_gate(self.common['ctx'],a['arm'],a['ref']['generators'],tail,operators=a['atoms'])
                a['checked_tail']=tail_identity
            elif tail_identity!=a['checked_tail']:
                raise ValueError('R_SEARCH_CHANGED_CANDIDATE_TAIL')
            return a['tail_report']
        self.unit(stage,tag+':tail',check_tail)
        self.unit(stage,tag+':distribution',lambda:distribution_gate(tail,config,dist,expected_steps=r))
        self.unit(stage,tag+':normalization',lambda:ret.normalization(tail,config,r))
        first=a['pf_errors'] is None
        kernels=self.unit(stage,tag+':kernels',lambda:b.reference.reference_kernels(a['ref']['generators'],Fraction(1,10),
                ld=a['arm'].plan.ld,identity_tail=a['arm'].plan.extracted_identity,rte_config=config,
                exponential_cache=a['exponential_cache']) if first else
            m4.corrected_kernel(a['ref']['generators'],config,ld=a['arm'].plan.ld,
                identity_tail=a['arm'].plan.extracted_identity,saved_pf=a['pf'],expected_steps=r,
                exponential_cache=a['exponential_cache']))
        rounds=[]
        for m in range(13):
            if first:
                a['exact'][m]=self.unit(stage,tag+f':exact{m}',lambda m=m:expm(-.1j*(1<<m)*a['ref']['H']))
                pf_only=dict(kernels,config=None,corrected=kernels['pf'])
                pf_ref=self.unit(stage,tag+f':PF_round{m}',lambda m=m:b.reference.reference_round(pf_only,
                    (a['ref']['H'],Fraction(1,10)*(1<<m)),repetitions=1<<m,
                    exact_operator=a['exact'][m]))
                if a['pf_errors'] is None:a['pf_errors']=[]
                a['pf_errors'].append(pf_ref['pf_operator_error'])
            ref=self.unit(stage,tag+f':round{m}',lambda m=m:m4.rte_round(kernels,m,a['pf_errors'][m],state=a['ground']))
            decision=self.unit(stage,tag+f':phase{m}',lambda ref=ref:feasible.classify_round(ref,target))
            rounds.append(dict(m=m,reference=ref,feasibility=decision))
        if first:
            a['pf']=kernels['pf']
        packet=dict(target_id=f"H6:{a['candidate']}:r{r}",rounds=rounds,
            df_bound_Ha=a['df']['sector_error']+a['df']['numerical_guard'],error_evidence=None,
            shells=self.ensure_shells()['reports'])
        lanes=ret.assess_lanes(packet,target,lambda k,l,fn:self.unit(stage,tag+':'+k+':'+l,fn))
        feasible_summary=self.unit(stage,tag+':feasibility',lambda:feasible.summarize(
            [row['feasibility'] for row in rounds],packet['df_bound_Ha'],target))
        eligible=feasible_summary['status']=='POTENTIALLY_FEASIBLE' and lanes['ideal_exact_native']['energy']['same_accuracy']
        value=dict(candidate=a['candidate'],input_sha256=self.common['input_identity'],r=r,
            tail=tail,config=config,dist=dist,kernels=kernels,packet=packet,lanes=lanes,eligible=bool(eligible))
        a['policies'][r]=value
        io.save(a['out']/f'r{r}_reference.json',dict(candidate=a['candidate'],r=r,
            rounds=rounds,lanes=lanes,eligible=bool(eligible),qualification=QUALIFICATION))
        return value

    def ensure_shells(self):
        if self.shells is not None:return self.shells
        from . import c_pr_aware_df_stage_b2 as b2
        from .c1_h6_native import shell_unitary
        from .c1_h6_compiler_precision import compiled_wrapper
        from .pr_dictionary_validation.fixture import CompilerProfile
        from .pr_dictionary_validation.f2_execution import circuit_metrics
        stage='input_neutral';circuits={};matrices={};reports={}
        folder=self.out/'shells';folder.mkdir(exist_ok=False)
        for label in ('head','X','Y'):
            raw=self.unit(stage,'shell:'+label,lambda label=label:b2.shell_build(2*self.common['system']['norb'],label))
            compiled=self.unit(stage,'shell_compile:'+label,lambda:compiled_wrapper(raw,CompilerProfile()))
            self.qsave(folder/label,dict(raw=raw,compiled=compiled))
            observed={k:shell_unitary(q) for k,q in (('raw',raw),('compiled',compiled))}
            # Existing shell_structure is independent of system dimension.
            from .c1_stage3_molecular import component_check
            self.unit(stage,'shell_structure:'+label,lambda:component_check(observed['compiled'],observed['raw'],1e-10,label=label))
            import numpy as np
            h=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
            expected=h if label in ('head','X') else h@np.diag([1.,-1j])
            component_check(observed['raw'],expected,1e-10,label='exact shell '+label)
            circuits[label]=dict(raw=raw,compiled=compiled);matrices[label]=observed
            reports[label]=dict(metrics=circuit_metrics(compiled))
        self.shells=dict(circuits=circuits,matrices=matrices,reports=reports)
        io.save(folder/'reports.json',reports);return self.shells

    @staticmethod
    def qsave(prefix,circuits):
        from qiskit import qpy
        for name,qc in circuits.items():
            with Path(str(prefix)+'_'+name+'.qpy').open('xb') as stream:qpy.dump(qc,stream)

    def native(self,a,policy,role,count,*,deterministic=False):
        expected=1 if deterministic else 32 if role=='selection' else 128 if role=='final' else None
        if type(count) is not int or count!=expected:raise ValueError('REGISTERED_NATIVE_COHORT_SIZE')
        import numpy as np
        from . import c_pr_aware_df_stage_b as b, c_pr_aware_df_stage_b2 as b2
        from .c1_h6_synthesis_bound import circuit_identity
        from .c1_h6_compiler_precision import compiled_wrapper
        from .c1_stage2_native import actual_adjoint
        from .pr_dictionary_validation.fixture import CompilerProfile
        from .pr_dictionary_validation.f2_execution import circuit_metrics
        stage='native_build_compile_and_reports'
        r=0 if deterministic else policy['r']
        folder=a['out']/(f'D_native' if deterministic else f'r{r}_{role}');folder.mkdir(exist_ok=False)
        shells=self.ensure_shells();started=time.monotonic();rows=[];labels=[]
        seed=None if deterministic else seed_policy(self.c['source_sha256'],a['task'],a['candidate'],r,role)
        for i in range(count):
            tag=f"{a['task']['id']}:{role}:r{r}:sample{i}"
            def draw():
                if deterministic:return ()
                master=(seed['master'],seed['cohort'],seed['SOURCE'],self.common['input_identity'])
                events=b.independent_occurrence(a['arm'],policy['tail'],policy['dist'],policy['config'],
                    master=master,round_index=-1,axis=seed['cohort'],shot=i,occurrence=0)
                if len(events)!=r or any(e.taylor_order!=0 for e in events):raise ValueError('H6_K0_EVENT_COUNT')
                return events
            events=self.unit(stage,tag+':events',draw)
            sample_seed=None if deterministic else b.connection.trajectory_seed(
                (seed['master'],seed['cohort'],seed['SOURCE'],self.common['input_identity']),
                a['arm'].role,a['candidate'],-1,seed['cohort'],i,0)
            with self.serial:
                raw=self.unit(stage,tag+':build',lambda:b.build_kernel(a['arm'],Fraction(1,10),events,
                    None if deterministic else policy['config'],builder=a['builder']))
                compiled=self.unit(stage,tag+':compile',lambda:compiled_wrapper(raw,CompilerProfile()))
            structure=self.unit(stage,tag+':P0',lambda:circuit_identity(compiled))
            self.qsave(folder/f'sample{i:03d}',dict(raw=raw,compiled=compiled))
            io.save(folder/f'sample{i:03d}_attempt.json',dict(sample=i,candidate=a['candidate'],r=r,
                input_sha256=self.common['input_identity'],cohort=seed,seed=sample_seed,events=[e.to_dict() for e in events],
                numerical_equivalence='NUMERICAL_CHECK_PENDING' if i==0 else UNTESTED,structure=structure))
            checked=i==0
            numerical=UNTESTED
            circuits=dict(raw=raw,compiled=compiled)
            if checked:
                if deterministic: reference=policy['kernels']['pf']
                else:
                    if a['reference_context'] is None:
                        a['reference_context']=self.unit(stage,tag+':reference_context',lambda:design.sector_reference_context(
                            a['arm'],a['ref']['generators'],self.common['ctx'],Fraction(1,10),operators=a['atoms']))
                    a['reference_context']['event_matrices'].clear()
                    reference=self.unit(stage,tag+':trajectory_reference',lambda:b2.trajectory_reference(
                        a['arm'],a['reference_context'],Fraction(1,10),events))
                with self.serial:
                    inverses={k:self.unit(stage,tag+':inverse:'+k,lambda q=q:actual_adjoint(q)) for k,q in circuits.items()}
                self.qsave(folder/f'sample{i:03d}',{k+'_inverse':q for k,q in inverses.items()})
                from .c_pr_aware_df_stage_b_runner import save_arrays
                save_arrays(folder/f'sample{i:03d}_reference.npz',reference=reference)
                states={k:np.zeros(2*(1<<(2*self.common['system']['norb'])),complex) for k in ('raw','compiled')}
                weights={k:np.concatenate([shells['matrices']['head'][k][control,0]*a['ground'] for control in (0,1)]) for k in states}
                def observe(ids,raw_columns,compiled_columns):
                    for k,columns in (('raw',raw_columns),('compiled',compiled_columns)):
                        states[k][:]+=weights[k][ids]@columns
                    import json
                    with (folder/f'sample{i:03d}_columns.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(columns=ids,status='ACTIONS_OBSERVED_NOT_YET_QUALIFIED'))+'\n')
                inverse_iter=iter((inverses['raw'],inverses['compiled']))
                report=self.unit(stage,tag+':correctness',lambda:design.check_kernel(raw,compiled,reference,
                    self.common['basis'],batch_columns=self.batch,threads=self.threads,
                    inverse_factory=lambda qc:next(inverse_iter),column_observer=observe,
                    action_call=lambda n,fn:self.unit('native_state_actions',tag,fn,count=n)))
                report['complex_XY']=self.unit(stage,tag+':XY',lambda:self.xy(states,reference,a['ground'],shells))
                io.save(folder/f'sample{i:03d}_correctness.json',report)
                numerical='COMPLETE_SECTOR_NUMERICAL_CHECK_PASS'
            metrics=self.unit(stage,tag+':metrics',lambda:circuit_metrics(compiled));rows.append(metrics)
            labels.append(numerical)
            io.save(folder/f'sample{i:03d}.json',dict(sample=i,candidate=a['candidate'],
                input_sha256=self.common['input_identity'],r=r,cohort=seed,seed=sample_seed,
                events=[e.to_dict() for e in events],structure=structure,
                numerical_equivalence=numerical,metrics=metrics,qualification=QUALIFICATION))
            if i==0:
                io.save(folder/'first_registered_batch_timing.json',dict(samples=1,
                    wall_seconds=time.monotonic()-started,remaining_samples=count-1,
                    remaining_seconds_same_sample_model=(time.monotonic()-started)*(count-1),
                    model_caveat='sample0 includes correctness; unchecked samples differ; not a confirmed ETA',
                    qpy_bytes=sum(p.stat().st_size for p in folder.glob('*.qpy'))))
        io.save(folder/'timing.json',dict(samples=count,checked=1,
            total_seconds=time.monotonic()-started,qpy_bytes=sum(p.stat().st_size for p in folder.glob('*.qpy'))))
        io.save(folder/'gaussian_precision.json',a['builder'].basis_reports)
        return rows,labels

    @staticmethod
    def xy(states,reference,psi,shells):
        import numpy as np
        from .c1_h6_state_action import ancilla_z
        from .pr_dictionary_validation.gates import signal_gate
        target=complex(np.vdot(psi,reference@psi));report={}
        for kind,state in states.items():
            values={axis:float(ancilla_z((shells['matrices'][axis][kind]@state.reshape(2,-1)).reshape(1,-1))[0]) for axis in ('X','Y')}
            report[kind]=signal_gate(values['X'],values['Y'],target,2e-10)
        return report

    def account(self,a,policy,rows,labels):
        from . import c_pr_aware_df_stage_b2 as b2
        from . import c_pr_aware_df_stage_c as accounting
        from . import c_pr_aware_df_m3 as m3
        from . import c_near_r32_statistics as stats
        packet=deepcopy(policy['packet']);lanes=deepcopy(policy['lanes'])
        cal=b2.calibration(rows);packet.update(kernel_samples=rows,saved_calibration=cal)
        kernel=accounting.cost_calibration(packet,expected_n=len(rows))
        costs=[m3.serial_cost(packet,kernel,m,axis) for m in range(13) for axis in ('X','Y')]
        for value in lanes.values():
            total=m3.accuracy.aggregate_totals(value['energy'],value['rounds'],costs,cost_scope=accounting.COST_SCOPE)
            value['conditional_totals']=total
            value['weighted_SE']=m3.calibration_uncertainty(packet,value['rounds'],total,expected_n=len(rows))
        result=dict(lanes=lanes,calibration=cal,qualification=QUALIFICATION,numerical_labels=labels)
        bindings=stats.affine(packet,result)
        return dict(rows=rows,result=result,bindings=bindings,calibration=cal,
            candidate=a['candidate'],input_sha256=self.common['input_identity'],qualification=QUALIFICATION)

    def deterministic(self,a):
        from . import c_pr_aware_df_stage_b as b
        from .c_same_ju_strong_d import ideal_plans
        from .c1_h6_ca_resources import serial_round_cost
        from .c1_stage3_accuracy import aggregate_totals
        stage='endpoint_reference';tag=a['task']['id']+':D'
        kernels=self.unit(stage,tag+':kernel',lambda:b.reference.reference_kernels(a['ref']['generators'],Fraction(1,10),
            ld=len(a['ref']['generators'])-1,exponential_cache=a['exponential_cache']))
        rounds=[dict(m=m,**self.unit(stage,tag+f':round{m}',lambda m=m:b.reference.reference_round(kernels,
            (a['ref']['H'],Fraction(1,10)*(1<<m)),repetitions=1<<m,exact_operator=a['exact'][m]))) for m in range(13)]
        packet=dict(id='H6_D:'+a['candidate'],state_model='own_DF_ground_state_oracle',alias_verified=True,
            df_error_with_guard=a['df']['sector_error']+a['df']['numerical_guard'])
        plan=ideal_plans(rounds,packet,lambda k,l,fn:self.unit(stage,tag+':'+k+':'+l,fn))
        if not plan['energy']['same_accuracy']:
            return dict(candidate=a['candidate'],input_sha256=self.common['input_identity'],ideal_exact_native=dict(cost=dict(totals=None)),reason='D_INFEASIBLE_REGISTERED_POLICY')
        rows,labels=self.native(a,dict(kernels=kernels),'D',1,deterministic=True)
        cost_packet=dict(id=packet['id'],s=1,kernel_metrics=rows[0],
            shell_metrics={k:v['metrics'] for k,v in self.ensure_shells()['reports'].items()})
        costs=[item for m in range(13) for item in serial_round_cost(cost_packet,m)]
        cost=aggregate_totals(plan['energy'],plan['rounds'],costs,cost_scope='serial_native_kernel_composition_without_state_preparation')
        return dict(candidate=a['candidate'],input_sha256=self.common['input_identity'],
            ideal_exact_native=dict(cost=cost,rounds=plan['rounds'],energy=plan['energy']),
            qualification=QUALIFICATION,numerical_labels=labels)
