"""Saved molecular backend for the dormant integrated Stage3 runner.

Construction is passive. prepare() decodes scientific inputs and must only be
called after separate execution permission. Tests inject artificial arrays.
"""
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor,as_completed
import math
from pathlib import Path
import threading

from . import c1_stage1 as stage1
from . import c1_stage1_execution as io
from . import c1_stage2_native as native
from . import c1_stage3_connection as connection
from . import c1_stage3_reference as reference
from . import c1_stage3_qualification as qualification
from .c1_stage3_accuracy import METRICS
from .c1_stage3_native_bridge import partial_kernel


def component_check(actual,expected,tolerance,*,label):
    """The existing maximum-entry acceptance rule without diagnostic SVDs.

    Stage2's matrix_gate also computes a spectral diagnostic. Calibration here
    needs no such norm certificate: synthesis remains explicitly UNRESOLVED.
    Avoid three full512 SVDs for each raw/compiled/adjoint correctness test.
    """
    import numpy as np
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(label+': shape/nonfinite')
    error=actual-expected;maximum=float(np.max(np.abs(error)))
    result=dict(label=label,max_entry=maximum,frobenius=float(np.linalg.norm(error,'fro')),tolerance=tolerance,rtol=0)
    if maximum>tolerance:raise ValueError(str(result))
    return result


class MolecularBackend:
    def __init__(self, verified, output):
        self.v, self.output = verified, Path(output)
        self.c, self.p = verified['contract'], verified['contract']['proposal']
        self.models, self.generators, self.full, self.states = {}, {}, {}, {}
        self.plans, self.atom_operators, self.exact, self.reused = {}, {}, {}, {}
        self.cache = {'CDF':{},'XDF':{}}
        self.bases_ready=set()
        self.cache_locks = {arm:threading.Lock() for arm in self.cache}
        self.shell_lock, self.shells = threading.Lock(), None

    def prepare(self, call):
        import numpy as np
        self.call = call
        root, c = self.v['root'], self.c
        physical = c['physical_conditions']
        self.modes = physical['system_qubits']; self.dimension = 1<<self.modes
        self.tol = self.p['correctness_tolerance']
        packet = call('input_load','saved XDF whitelist',self._load_xdf)
        parent = (root/c['saved_fit_export']).parent
        exported, metadata = call('fit_export_load','saved converged fit',lambda:
            io.load_export(parent,dict(input=self.v['input'],source_path=parent/'SOURCE.json')))
        if metadata['actual_rank'] != physical['CDF_rank']: raise ValueError('SAVED_CDF_RANK')
        if float(exported['constant']) != float(packet['nuclear_repulsion']): raise ValueError('SCALAR_OWNERSHIP')
        from .c1_general_k_adapter import from_ffsim_number_arrays
        model = from_ffsim_number_arrays(one_body_tensor=exported['kappa'],
            diag_coulomb_mats=exported['diag_coulomb_mats'],orbital_rotations=exported['orbital_rotations'],
            constant=float(exported['constant']),z_representation=False,
            frame_ids=metadata['frame_ids'],group_ids=metadata['group_ids'],
            source_identity=c['saved_fit_export_sha256'],input_identity=c['input_sha256'],
            one_body_role='ffsim_corrected_kappa',enable_general_k=True)
        self.models['CDF'] = call('model_prepare','CDF',lambda:native.prepare_cdf(model))
        self.models['XDF'] = call('model_prepare','XDF',lambda:native.prepare_xdf(packet['df_one_body'],
            packet['df_lambdas'],packet['df_g_matrices'],float(exported['constant']),
            source_identity=c['input_sha256']+':XDF',input_identity=c['input_sha256']))
        for arm in self.models:
            if self.models[arm].modes != self.modes or len(self.models[arm].blocks)-1 != physical[arm+'_rank']:
                raise ValueError('NATIVE_MODEL_DIMENSION_RANK')
        s1 = root/c['stage1_operator_directory']
        matrices = {name:call('saved_operator_load',name,lambda name=name:
                    self._load_matrix(s1/(name+'.npz'),sector=name.endswith('_sector')))
                    for name in ['physical_sector','CDF_sector','XDF_sector','raw_base','raw_total']+
                                [f'raw_frame_{i}' for i in range(physical['CDF_rank'])]}
        self.generators['CDF'] = [matrices['raw_base']]+[matrices[f'raw_frame_{i}'] for i in range(physical['CDF_rank'])]
        self.full['CDF'] = matrices['raw_total']
        self.car = call('car_context',str(self.modes)+' modes',lambda:stage1.car_bilinears(self.modes))
        self.generators['XDF'] = [call('xdf_full_generator','one-body+c',lambda:
            float(exported['constant'])*np.eye(self.dimension)+native.xdf_car_generator(packet['df_one_body'],None,self.car))]
        for i in range(physical['XDF_rank']):
            self.generators['XDF'].append(call('xdf_full_generator',f'F{i}',lambda i=i:
                native.xdf_car_generator(packet['df_g_matrices'][i],packet['df_lambdas'][i],self.car)))
        self.full['XDF'] = call('xdf_full_sum','XDF',lambda:sum(self.generators['XDF']))
        ground = {arm:call('ground_eigenpair',arm,lambda arm=arm:self._ground(matrices[arm+'_sector'],arm))
                  for arm in ('physical','CDF','XDF')}
        self.energies = {arm:float(pair[0][0]) for arm,pair in ground.items()}
        old = io.read(root/c['stage1_result'])
        records = {r['id']:r for r in old['records']}
        # Reuse the already measured physical operator errors; no repeat norm.
        self.df_errors = {arm:records['sector_DF_'+arm]['error']+records['sector_DF_'+arm]['allowance']
                          for arm in ('CDF','XDF')}
        for arm in ('CDF','XDF'):
            self.states[arm] = call('state_alias_sector_checks',arm,
                lambda arm=arm:self._state(arm,ground[arm],matrices[arm+'_sector']))
            def split(arm=arm):
                plan=connection.split_native_model(self.models[arm],self.p['PR_LD'])
                self.atom_operators[arm]=self._original_atoms(plan)
                return plan
            self.plans[arm] = call('split_native_model',arm,split)
        self.builder=native.NativeBuilder()
        from qiskit import qpy
        for arm in ('CDF','XDF'):
            items={}
            for kind,path in self.c['saved_native_kernels'][arm].items():
                def load(path=path):
                    with (root/path).open('rb') as stream:circuits=qpy.load(stream)
                    if len(circuits)!=1 or circuits[0].num_qubits!=self.modes+1 or circuits[0].num_clbits:
                        raise ValueError('SAVED_NATIVE_KERNEL_SCHEMA')
                    return circuits[0]
                items[kind]=call('saved_native_kernel_load',arm+':'+kind,load)
            self.reused[arm]=items
        old2=io.read(root/c['stage2_result'])
        self.saved_metrics=old2['metrics']
        from scipy.linalg import expm
        from threadpoolctl import threadpool_limits
        workers=getattr(self,'execution_plan',{}).get('workers',1)
        with threadpool_limits(limits=1 if workers>1 else None):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(call,'exact_model_evolution',f'{arm}:m{m}',lambda arm=arm,m=m:
                    expm(-1j*float(Fraction(self.p['base_time']))*(1<<m)*self.full[arm])):(arm,m)
                    for arm in ('CDF','XDF') for m in self.p['rounds']}
                for future in as_completed(futures):self.exact[futures[future]]=future.result()

    def _load_xdf(self):
        import numpy as np
        with np.load(self.v['input'],allow_pickle=False) as archive:
            packet={k:archive[k] for k in ('df_one_body','df_lambdas','df_g_matrices','nuclear_repulsion')}
        shapes={'df_one_body':(self.modes,self.modes),'df_lambdas':(self.c['physical_conditions']['XDF_rank'],),
                'df_g_matrices':(self.c['physical_conditions']['XDF_rank'],self.modes,self.modes),'nuclear_repulsion':()}
        if any(a.shape!=shapes[k] for k,a in packet.items()):raise ValueError('SAVED_XDF_SCHEMA')
        stage1.finite_arrays(*packet.values())
        return packet

    def _load_matrix(self,path,*,sector=False):
        import numpy as np
        with np.load(path,allow_pickle=False) as archive:a=archive['matrix']
        d=self.c['physical_conditions']['sector_dimension'] if sector else self.dimension
        if a.shape!=(d,d) or a.dtype!=np.dtype('complex128') or not np.isfinite(a).all():
            raise ValueError('SAVED_OPERATOR_SCHEMA: '+str(path))
        return a

    def _ground(self,matrix,arm):
        import numpy as np
        if np.max(np.abs(matrix-matrix.conj().T))>self.tol['operator_component_abs']:
            raise ValueError('NONHERMITIAN_SAVED_SECTOR')
        pair=np.linalg.eigh(matrix)
        np.savez(self.output/('ground_'+arm+'.npz'),energy=pair[0][0],sector_state=pair[1][:,0])
        return pair

    def _state(self,arm,pair,sector):
        import numpy as np
        p=self.c['physical_conditions']
        indices=stage1.sector_indices(self.modes//2,p['Nalpha'],p['Nbeta'])
        psi=np.zeros(self.dimension,complex);psi[indices]=pair[1][:,0]
        if (len(indices)!=len(sector) or np.max(np.abs(self.full[arm][np.ix_(indices,indices)]-sector))>
                self.tol['operator_component_abs'] or np.linalg.norm(self.full[arm]@psi-pair[0][0]*psi)>
                self.tol['operator_component_abs']):raise ValueError('GROUND_SECTOR_EMBEDDING')
        lo,hi=self.p['alias_energy_window_ha']
        if not lo<=pair[0][0]<=hi:raise ValueError('ALIAS_ENERGY_WINDOW')
        if max(abs(lo),abs(hi))*float(Fraction(self.p['base_time']))+self.p['accuracy_target']['beta_total']>=math.pi:
            raise ValueError('ALIAS_BRANCH_MARGIN')
        return psi

    def _original_atoms(self,plan):
        """Independent CAR densities, not the native Gaussian or K emitter."""
        import numpy as np
        eye=np.eye(self.dimension);densities={};result={}
        for atom in plan.atoms:
            if atom.frame_id not in densities:
                U=np.asarray(plan.frames[atom.frame_id]);n=len(U)
                coefficients=(U.T[:,:,None]*U.conj().T[:,None,:]).reshape(n,n*n)
                densities[atom.frame_id]=(coefficients@self.car.reshape(n*n,-1)).reshape(n,self.dimension,self.dimension)
            q=eye
            for i in atom.support:q=q@(eye-2*densities[atom.frame_id][i])
            result[atom.atom_id]=(1 if atom.coefficient>0 else -1)*q
        return result

    def prepared_summary(self):
        return dict(ground_energies_ha=self.energies,df_error_bounds_ha=self.df_errors,
            df_bound_kind='Saved Stage1 numerical norm plus its original engineering allowance; not an interval proof',
            state_preparation='ORACLE_EXCLUDED',reference_scope='full system; no intermediate sector projection',
            new_fit=0,physical_DF_norm_recomputed=False,
            sampling_law=dict(model='ideal_finite_RTE_IID',reference_model_error=0,
                              actual_finite_bit_implementation_error=None))

    def kernel(self,candidate):
        arm=candidate.arm;h=Fraction(self.p['base_time'])/candidate.s
        plan=self.plans[arm];tail=config=distribution=None;support=None
        if candidate.family=='PR':
            tail,config,distribution=self.call('rte_configuration',candidate.id,
                lambda:connection.configure_rte(plan,h,candidate.r,candidate.K))
            if config is not None:
                support=qualification.exhaustive_trajectories(tail,distribution,candidate.r,
                    budget=self.p['cost_samples_per_PR_candidate'])
            else:support=((),)
        with self.cache_locks[arm]:
            kernels=reference.reference_kernels(self.generators[arm],h,
                ld=self.p['PR_LD'] if config is not None else len(self.generators[arm])-1,
                identity_tail=plan.extracted_identity,rte_config=config,exponential_cache=self.cache[arm])
        return dict(reference=kernels,tail=tail,config=config,distribution=distribution,
                    support=support,h=h,plan=plan)

    def reference(self,candidate,kernel,m):
        arm=candidate.arm
        row=reference.reference_round(kernel['reference'],(self.full[arm],Fraction(self.p['base_time'])*(1<<m)),
            repetitions=(1<<m)*candidate.s,state=self.states[arm],exact_operator=self.exact[arm,m])
        signal=row.pop('raw_signal',None)
        row['raw_signal_parts']=None if signal is None else [float(signal.real),float(signal.imag)]
        return row

    def _provider(self,plan):
        from .pr_dictionary_validation.circuits import BasisProvider
        parent=self
        class CachedProvider(BasisProvider):
            def basis(self,atom,singleton=False):
                # Prebuilt full-frame bases, identical policy for both arms.
                for block in plan.model.blocks:
                    if block.rotation==plan.frames[atom.frame_id]:
                        return parent.builder._basis(plan.model,block).copy()
                raise ValueError('Unregistered Gaussian basis')
        return CachedProvider(plan.model.modes,plan.frames)

    def _ensure_bases(self,arm):
        # Do not construct new Gaussian circuits for an arm whose registered
        # candidates all fail the PF/RTE/DF feasibility conditions.
        with self.cache_locks[arm]:
            if arm in self.bases_ready:return
            model=self.models[arm]
            for block in model.blocks:
                self.call('gaussian_build',arm+':'+block.block_id,
                          lambda block=block:self.builder._basis(model,block))
            self.bases_ready.add(arm)

    def _trajectory_reference(self,candidate,kernel,events):
        import numpy as np
        if kernel['config'] is None:return kernel['reference']['pf']
        from trotterlib.rte import event_unitary
        middle=np.eye(self.dimension,dtype=complex)
        for event in events:middle=event_unitary(event,self.atom_operators[candidate.arm])@middle
        half=tuple((i,kernel['h']/2) for i in range(self.p['PR_LD']+1))
        with self.cache_locks[candidate.arm]:
            left=reference._ordered_exponentials(self.generators[candidate.arm],half,self.cache[candidate.arm])
            right=reference._ordered_exponentials(self.generators[candidate.arm],tuple(reversed(half)),self.cache[candidate.arm])
        return np.exp(-1j*float(kernel['h'])*kernel['plan'].extracted_identity)*right@middle@left

    def _write_circuit(self,label,kind,qc):
        from qiskit import qpy
        path=self.output/'circuits';path.mkdir(exist_ok=True)
        with (path/(label+'_'+kind+'.qpy')).open('xb') as f:qpy.dump(qc,f)

    def _validate_compile(self,label,raw,target,*,controlled,arm=None):
        import numpy as np
        from qiskit.quantum_info import Operator
        from .pr_dictionary_validation.gates import compiled_wrapper,signal_gate
        from .pr_dictionary_validation.fixture import CompilerProfile
        from .pr_dictionary_validation.f2_execution import circuit_metrics
        measured=bool(raw.num_clbits)
        clean=raw.remove_final_measurements(inplace=False) if measured else raw
        self._write_circuit(label,'raw',raw)
        def rawcheck():
            value=target() if callable(target) else target
            expected=np.block([[np.eye(self.dimension),np.zeros_like(value)],
                               [np.zeros_like(value),value]]) if controlled else value
            matrix=Operator(clean).data
            report=component_check(matrix,expected,self.tol['operator_component_abs'],label='raw:'+label)
            return matrix,report,expected,value
        raw_matrix,raw_report,expected,system_target=self.call('raw_reference_check',label,rawcheck)
        def adjoint():
            qc=native.actual_adjoint(clean)
            return component_check(Operator(qc).data,expected.conj().T,self.tol['operator_component_abs'],label='actual adjoint:'+label)
        adj=self.call('actual_adjoint_check',label,adjoint)
        profile=CompilerProfile(**{**self.p['compiler'],'basis_gates':tuple(self.p['compiler']['basis_gates'])})
        compiled=self.call('compile',label,lambda:compiled_wrapper(raw,profile))
        self._write_circuit(label,'compiled',compiled)
        def compiledcheck():
            if any(x.operation.name not in {'rz','sx','x','cx','measure','barrier'} for x in compiled.data):
                raise ValueError('NONNATIVE_COMPILED_GATE')
            qc=compiled.remove_final_measurements(inplace=False) if measured else compiled
            matrix=Operator(qc).data
            report=component_check(matrix,expected,self.tol['operator_component_abs'],label='compiled:'+label)
            report['unitarity']=component_check(matrix.conj().T@matrix,np.eye(len(matrix)),
                self.tol['unitarity_component_abs'],label='unitarity:'+label)
            return matrix,report
        compiled_matrix,compiled_report=self.call('compiled_reference_check',label,compiledcheck)
        metrics=self.call('metrics',label,lambda:circuit_metrics(compiled))
        if controlled:
            def xy():
                init=np.concatenate((self.states[arm],np.zeros(self.dimension)))
                values=[]
                for matrix,compiled_flag in ((raw_matrix,False),(compiled_matrix,True)):
                    head=self.shells['head']['compiled_matrix' if compiled_flag else 'raw_matrix']
                    prefix=matrix@(head@init);axes=[]
                    for axis in ('X','Y'):
                        tail=self.shells[axis]['compiled_matrix' if compiled_flag else 'raw_matrix']
                        sv=tail@prefix
                        axes.append(float(np.sum(abs(sv[:self.dimension])**2)-np.sum(abs(sv[self.dimension:])**2)))
                    values.append(signal_gate(*axes,np.vdot(self.states[arm],system_target@self.states[arm]),self.tol['xy_abs']))
                return values
            signals=self.call('xy_pair_check',label,xy)
        else:signals=None
        return dict(metrics=metrics,diagnostics=dict(raw=raw_report,compiled=compiled_report,adjoint=adj,xy=signals),
                    raw_matrix=raw_matrix,compiled_matrix=compiled_matrix)

    def _shells(self):
        import numpy as np
        with self.shell_lock:
            if self.shells is not None:return
            H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2);Sdg=np.diag([1,-1j]);eye=np.eye(self.dimension)
            self.shells={}
            for kind in ('head','X','Y'):
                def build(kind=kind):
                    head,tail=connection.readout_segments(self.modes,'Y' if kind=='Y' else 'X')
                    return head if kind=='head' else tail
                qc=self.call('circuit_build','shell:'+kind,build)
                target=np.kron(H@Sdg if kind=='Y' else H,eye)
                self.shells[kind]=self._validate_compile('shell_'+kind,qc,target,controlled=False)

    def calibrate(self,candidate,kernel,call):
        import numpy as np
        self._ensure_bases(candidate.arm)
        self._shells()
        arm=candidate.arm;records={};event_records=[]
        budget=self.p['cost_samples_per_PR_candidate']
        def build_check(events,index):
            label=candidate.id+':'+str(index)
            from .c1_stage3_integrated import progress
            progress(self.output,candidate.id,'native_trajectory',
                dict(index=index,events=[e.to_dict() for e in events]))
            def build():
                if candidate.family=='D' or kernel['config'] is None:
                    return self.builder.ordered(self.models[arm],kernel['h'],optimized=True)
                return partial_kernel(kernel['plan'],kernel['h'],events,rte_config=kernel['config'],
                    builder=self.builder,provider=self._provider(kernel['plan']))
            qc=call('circuit_build',label,build)
            target=lambda:self._trajectory_reference(candidate,kernel,events)
            record=self._validate_compile(label.replace(':','_'),qc,target,controlled=True,arm=arm)
            progress(self.output,candidate.id,'native_metrics',dict(index=index,metrics=record['metrics'],
                                                                   diagnostics=record['diagnostics']))
            # Large matrices are discarded after correctness/X/Y checks.
            return dict(metrics=record['metrics'],diagnostics=record['diagnostics'])
        if candidate.family=='D' and candidate.s==1:
            key=arm+'::shared'
            record=dict(metrics=self.saved_metrics[key],diagnostics=dict(reused_stage2=True,
                source=self.c['stage2_source'],raw_path=self.c['saved_native_kernels'][arm]['raw'],
                compiled_path=self.c['saved_native_kernels'][arm]['compiled']))
            records[()]=record;samples=[record['metrics']];covered=True
        elif candidate.family=='D':
            record=build_check((),0);records[()]=record;samples=[record['metrics']];covered=True
        else:
            if kernel['support'] is not None:
                for i,events in enumerate(kernel['support']):
                    records[qualification.trajectory_key(events)]=build_check(events,i)
            samples=[]
            for i in range(budget):
                def sample(i=i):
                    if kernel['config'] is None:return ()
                    seed=connection.trajectory_seed(self.p['cost_seed_master'],arm,candidate.id,0,'cost',i,0)
                    return connection.sample_occurrence(kernel['tail'],kernel['distribution'],candidate.r,seed)
                events=call('cost_kernel_sample',candidate.id+':'+str(i),sample)
                key=qualification.trajectory_key(events)
                if key not in records:records[key]=build_check(events,len(records))
                samples.append(records[key]['metrics'])
                event_records.append([e.to_dict() for e in events])
            covered=kernel['support'] is not None
        mean={k:math.fsum(r[k] for r in samples)/len(samples) for k in METRICS}
        def envelope():return qualification.support_cost_envelope([r['metrics'] for r in records.values()],complete_support=covered)
        bound=call('cost_interval_assessment',candidate.id,envelope) if candidate.family=='PR' else envelope()
        cost=dict(target_id=candidate.id,policy='native_kernel_repeat_with_fences',mean=mean,
            lower=bound['lower'],upper=bound['upper'],simultaneous_interval_qualified=bound['simultaneous_interval_qualified'],
            classical_cost_samples=len(samples),support_qualification=bound,
            point_estimate_law='finite-bit calibration; no unproved ideal-mean transfer',complete_support=covered)
        shells={}
        for axis in ('X','Y'):
            metrics={k:self.shells['head']['metrics'][k]+self.shells[axis]['metrics'][k] for k in METRICS}
            shells[axis]=dict(axis=axis,policy='separate_native_readout',mean=metrics,
                lower=metrics,upper=metrics,simultaneous_interval_qualified=True)
        return dict(kernel=cost,shells=shells,diagnostics=list(records.values()),events=event_records,
                    synthesis_bound=None,whole_support_covered=covered)

    def qualify(self,candidate,calibration):
        diagnostics=calibration['diagnostics']
        values=[r['diagnostics'].get('compiled',{}).get('frobenius') for r in diagnostics]
        observed=max((x for x in values if x is not None),default=None)
        # Numeric residuals are never promoted to a whole-law enclosure.
        return qualification.synthesis_envelope(whole_support_covered=calibration['whole_support_covered'],
                                               numerical_diagnostic_max=observed)

    def virtual(self,chosen):
        import numpy as np
        arm=chosen['candidate']['arm']
        signals=[complex(*parts) for parts in chosen['raw_signals']]
        shots=[p['shots'] for p in chosen['evaluation']['rounds']]
        seed=connection.trajectory_seed(self.p['virtual_rpe_seed_master'],arm,chosen['id'],0,'virtual',0,0)
        result=reference.sample_rpe_phases(signals,shots,trials=self.p['virtual_trials_per_selected_protocol'],seed=seed)
        if 'phase_estimates' in result:
            estimates=-np.asarray(result['phase_estimates'])/float(Fraction(self.p['base_time']))
            result['physical_energy_errors_ha']=abs(estimates-self.energies['physical']).tolist()
            result['CA_failures']=int(np.count_nonzero(abs(estimates-self.energies['physical'])>
                                                     self.p['accuracy_target']['energy_ha']))
            result['guarantee']='diagnostic marginal trials; not a replacement for the analytic confidence bound'
        return result
