"""Lazy production backend. Importing this file launches no scientific kernel."""
from functools import partial
import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import h2_common_production_contract as c


class Backend:
    def __init__(self, session):
        c.require(type(session) is c.Session and session.used['bootstrap']==1, 'AUTHORIZED_SESSION_REQUIRED')
        session.verify();self.s=session;self.preps={};self.operations={}
        for root in (Path(session.cfg['acquisition_root'])/'scripts',Path(session.cfg['provider_root'])/'src'):
            sys.path.insert(0,str(root))
        self.np=importlib.import_module('numpy');self.qiskit=importlib.import_module('qiskit')
        c.require(self.np.__version__=='1.26.4' and self.qiskit.__version__=='1.3.0','PINNED_RUNTIME_VERSIONS')
        self.gaussian=importlib.import_module('h46_new2_native')
        self.core=importlib.import_module('h46_new2_core')
        self.native_kernel=importlib.import_module('h46_m5_native')
        self.plan=self.native_kernel.c.core  # acquisition SOURCE; absent in provider
        self.native_api=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
        self.cost_api=importlib.import_module('trotterlib.rte_compiled_cost')
        self.rte=importlib.import_module('trotterlib.rte')
        self.guard=importlib.import_module('h2_m5_compiler_phase_guard')
        self.df=importlib.import_module('trotterlib.df_hamiltonian')
        self.partial=importlib.import_module('trotterlib.df_partial_s2')
        self.state_hash=importlib.import_module('trotterlib.pr2_s0_s1_validation')._state_hash
        self.fp=importlib.import_module('trottertracks.resource_applicability.ax2b_stream_fingerprint_v5').canonical_qiskit_circuit_fingerprint
        self.simulator=importlib.import_module('trotterlib.df_trotter.circuit').simulate_statevector
        self.distribution=self.rte.finite_rte_distribution
        self.accounting=importlib.import_module('trottertracks.resource_applicability.h6_matched_accounting_v1')
        self.shots=importlib.import_module('trottertracks.resource_applicability.ax2b_numerical_accounting').u_aware_shots
        provider=Path(session.cfg['provider_root'])/'src'
        for name,module in tuple(sys.modules.items()):
            if name.split('.')[0] in ('trotterlib','trottertracks') and getattr(module,'__file__',None):
                c.require(Path(module.__file__).resolve().is_relative_to(provider.resolve()),'LOADED_PROVIDER_ORIGIN')
        session.loaded_closure()

    def setup(self):
        np=self.np;cfg=self.s.cfg
        record=c.read(cfg['inputs']['snapshot_receipt'])
        c.require(c.sha(cfg['inputs']['snapshot'])==record['sha256']==cfg['binding']['snapshot'],'SNAPSHOT_SHA')
        with np.load(cfg['inputs']['snapshot'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        for name,value in arrays.items():
            a=np.ascontiguousarray(value);saved=record['arrays'][name]
            actual=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
            c.require(all(actual[k]==saved[k] for k in actual),'SNAPSHOT_ARRAY:'+name)
        ham=self.df.DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],
                                tuple(arrays['g_matrices']),record['hamiltonian_metadata'])
        sector=self.df.PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
        full,state=arrays['state_vector'],arrays['sector_state_vector']
        c.require(np.array_equal(arrays['sector_basis_indices'],sector.basis_indices) and
                  np.array_equal(full[sector.basis_indices],state),'SAVED_SECTOR_STATE')
        c.require(self.partial.df_hamiltonian_hash(ham)==cfg['binding']['hamiltonian'] and
            self.state_hash(full,state)==cfg['binding']['state'] and ham.n_blocks==3 and ham.n_qubits==4,'SAVED_H2_BINDING')
        oracle=self.gaussian.OccupationOracle(ham,full,prefix=3)
        qstate=np.zeros(16,complex);qstate[oracle.qindices]=oracle.initial
        c.require(abs(np.linalg.norm(qstate)-1)<1e-12,'SAVED_STATE_NORMALIZATION')
        matrices=tuple((s[1]*s[0])@s[2] for s in oracle.specs)
        scalar,lam=self.core.matrix_only_tail_coefficients(ham,1)
        tail=sum(matrices[2:],np.zeros_like(matrices[0]))-scalar*np.eye(len(oracle.initial))
        self.ctx=SimpleNamespace(ham=ham,oracle=oracle,qstate=qstate,matrices=matrices,
            tail_identity=scalar,lambda_R=lam,tail=tail,tail_spec=self.core.spectrum(tail))
        ref=c.read(cfg['inputs']['old_reference'])['old_reference']
        self.reference_scalar=complex(ref['reference']['real'],ref['reference']['imag'])
        self.reference_error=ref['error'];self.reference_id='OLD_FP:'+c.sha(cfg['inputs']['old_reference'])
        self.input_digest=c.policy.digest(dict(binding=cfg['binding'],files=cfg['files'],series=c.SERIES))

    def prepare(self,prefix):
        rows=[];original=self.gaussian.gaussian_operations
        def observe(rotation):
            ops=original(rotation)
            rows.append(dict(rotation_sha256=hashlib.sha256(self.np.ascontiguousarray(rotation).tobytes()).hexdigest(),
                operations=[dict(gate=g.name,params=[float(p).hex() for p in g.params],wires=list(w)) for g,w in ops]))
            return ops
        self.gaussian.gaussian_operations=observe
        try:prep=self.gaussian.prepare(self.ctx.ham,prefix)
        finally:self.gaussian.gaussian_operations=original
        c.require(tuple(prep.deterministic_fragment_indices)==tuple(range(prefix)),'PREPARATION_FRAGMENT_ORDER')
        if prefix==1:
            c.require(abs(prep.extracted_identity_coefficient-self.ctx.tail_identity)<1e-9 and
                      abs(prep.exact_rte_lambda_r-self.ctx.lambda_R)<1e-9,'TAIL_SCALAR_LAMBDA')
        self.preps[prefix]=prep;self.operations[prefix]=c.policy.digest(rows)
        c.save(self.s.root/'results'/('preparation_'+str(prefix)+'.json'),dict(prefix=prefix,
            preparation_sha256=prep.preparation_hash,operation_stream=rows,operations_sha256=self.operations[prefix],
            native_policy=c.policy.NATIVE_POLICY,series=c.SERIES))

    def group(self,candidate):
        return dict(method=candidate.method,formula=candidate.formula,prefix=candidate.prefix,
                    q=candidate.q,policy=candidate.r_policy,r_value=candidate.r_value)

    def steps(self,candidate):
        # Raw Full schedule has unused rte_steps=1. It never creates a request.
        return self.plan.schedule(candidate.formula,candidate.q,candidate.r_policy or 'PER_PIECE',candidate.r_value or 1)

    def diagnose(self,candidate):
        from h2_common_production_diagnostics import path,join
        a=self.s.call('path_signal',path,self,candidate,False)
        b=self.s.call('path_signal',path,self,candidate,True)
        return join(self,candidate,a,b)

    def reference(self,candidate,corrected):
        def rebind():
            value=corrected-self.reference_scalar
            return dict(reference_id=self.reference_id,reference_error=self.reference_error,
                bias=dict(real=value.real,imag=value.imag),definition='ordinary_signed_corrected_minus_saved_full_H_reference',
                reference_recomputed=False,reference_error_certified=False)
        return self.s.call('reference_rebind',rebind)

    def native(self,candidate,diagnostic):
        np=self.np;prep=self.preps[candidate.prefix];group=self.group(candidate);requests=[]
        if candidate.method=='PR':
            seeds=c.read(self.s.cfg['inputs']['control_seeds'])[candidate.id]
            factory=self.partial.make_df_partial_s2_step_request
            cutoff=importlib.import_module('trotterlib.pr2_matched_accuracy_m1_execution')._explicit_cutoff_tolerance
            def sampled(*a,**kw):
                self.s.take('rte_event',kw['rte_steps'])
                return self.s.call('request_factory',factory,*a,**kw)
            requests=self.native_kernel.make_requests(group,prep,seeds,self.ctx.lambda_R,
                factory=sampled,tolerance=cutoff)
            expected=self.s.call('occupation_trajectory',self.ctx.oracle.trajectory,requests)
            c.save(self.s.root/'results'/(candidate.id+'_events_PRIVATE.json'),
                dict(events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests],conditional=False))
        else:
            c.require(diagnostic is not None,'FULL_INDEPENDENT_ACTION_REQUIRED')
            expected=np.zeros(16,complex);expected[self.ctx.oracle.qindices]=diagnostic['independent']['vectors'][:,0]
        api=SimpleNamespace(**{k:getattr(self.native_api,k) for k in ('block_instruction_bound','NativeEvolution')})
        for method in ('build_partial_native','build_deterministic_native'):
            fn=getattr(self.native_api,method)
            setattr(api,method,lambda *a,_fn=fn,**kw:self.s.call('lowering',_fn,*a,**kw))
        scalar=self.ctx.ham.constant+(self.ctx.tail_identity if candidate.method=='PR' else 0.)
        evolution=self.s.call('compose',self.native_kernel.compose,group,prep,requests,scalar,api,self.qiskit.QuantumCircuit,self.fp)
        low=np.r_[self.ctx.qstate,np.zeros(16,complex)];high=np.r_[np.zeros(16,complex),self.ctx.qstate]
        checks={}
        def gate(name,value,limit):
            value=float(value);c.policy.number(value,name);c.require(value<limit,'CORRECTNESS:'+name);checks[name]=value
        a=self.s.call('simulate',self.simulator,evolution.circuit,low)
        b=self.s.call('simulate',self.simulator,evolution.circuit,high)
        gate('native_low',np.linalg.norm(a-low),1e-9)
        gate('native_leakage',np.linalg.norm(b[:16]),1e-9)
        gate('occupation_vector',np.linalg.norm(b[16:]-expected),1e-9)
        z=complex(np.vdot(self.ctx.qstate,expected));costs={};rows=[]
        settings=self.rte.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
        for axis in ('cosine','sine'):
            wrapper=self.s.call('wrapper',self.native_api.build_native_hadamard_wrapper,evolution,
                axis=axis,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
            before=self.s.call('simulate',self.simulator,wrapper.remove_final_measurements(inplace=False),low)
            observed=float(np.vdot(before[:16],before[:16]).real-np.vdot(before[16:],before[16:]).real)
            gate('hadamard_'+axis,abs(observed-(z.real if axis=='cosine' else z.imag)),1e-9)
            previous=self.cost_api.transpile
            def guarded(*a,**kw):return self.s.call('transpile',self.guard.transpile,*a,**dict(kw,num_processes=1))
            self.cost_api.transpile=guarded
            try:cost=self.s.call('compile',self.cost_api.transpile_and_measure_cost,wrapper,settings,actual_circuit_fingerprint=self.fp(wrapper))
            finally:self.cost_api.transpile=previous
            c.require(cost.actual_circuit_fingerprint==self.fp(wrapper) and
                cost.compiler_settings_hash==self.native_kernel.c.COMPILER_HASH,'COMPILED_IDENTITY')
            after=self.s.call('simulate',self.simulator,cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
            gate('compiled_vector_'+axis,np.linalg.norm(after-before),1e-7)
            receipt=cost.transpiled_circuit.metadata['H2_phase_guard']
            c.require(receipt['unchanged_gate_stream'] and receipt['state_fit'] is False,'GUARD_RECEIPT')
            c.policy.integer(cost.rz_count,'RZ',minimum=1);costs[axis]=cost.rz_count
            rows.append(dict(axis=axis,RZ=cost.rz_count,native_fingerprint=self.fp(wrapper),
                compiled_fingerprint=self.fp(cost.transpiled_circuit),compiler_settings_hash=cost.compiler_settings_hash,guard=receipt))
        c.save(self.s.root/'results'/(candidate.id+'_native.json'),dict(series=c.SERIES,candidate=c.asdict(candidate),
            prep=prep.preparation_hash,operations=self.operations[candidate.prefix],schedule=self.steps(candidate),
            scalar=scalar,checks=checks,axes=rows,phase_alignment=False,state_fit=False,
            vector_gate_scope='SAVED_STATE_NOT_ALL_STATE_CERTIFICATE',contract_sha256=c.policy.digest(self.s.cfg)))
        return [dict(sample=0,**costs)]

    def reuse(self,candidate,witness):
        # Explicit witness must bind to new preparation, schedule, runtime and events.
        proof=witness['proof'];cfg=self.s.cfg
        c.require(set(proof)=={'events','correctness','cohort'},'REUSE_PROOF_PATHS')
        sealed={r['path'] for r in cfg['files']}
        for p in proof.values():c.require(p in sealed,'REUSE_PROOF_NOT_SEALED')
        reference=c.read(cfg['inputs']['snapshot_receipt'])
        current=dict(hamiltonian=cfg['binding']['hamiltonian'],state=cfg['binding']['state'],
            basis=reference['arrays']['sector_basis_indices']['sha256'],
            preparation=self.preps[1].preparation_hash,rotation=self.operations[1],
            schedule=c.policy.digest(self.steps(candidate)),runtime_closure=c.policy.digest(self.s.cfg['runtime']),
            compiler_settings=c.policy.digest(c.policy.COMPILER),
            deterministic_lowering=c.sha(self.native_api.__file__),
            event_law=c.policy.digest(dict(source=c.sha(self.rte.__file__),K=4,policy='ABS_TIME_PER_OUTER',value=32)),
            events=c.sha(proof['events']),wrapper=c.sha(self.native_api.__file__),
            scalar_phase=c.policy.digest(dict(scalar=self.ctx.ham.constant+self.ctx.tail_identity,T=.8,scalar_once=True)),
            compiler_source=c.policy.digest(dict(guard=c.sha(self.guard.__file__),cost=c.sha(self.cost_api.__file__))),
            correctness=c.sha(proof['correctness']),cohort=c.sha(proof['cohort']))
        cohort=c.read(proof['cohort'])
        c.require(cohort['candidate']==candidate.id and cohort['conditional'] is False and cohort['pooling'] is False,'REUSE_COHORT_LAW')
        c.require(cohort['tasks']==witness['tasks'],'REUSE_TASK_PROJECTION')
        accepted=c.read(proof['correctness'])
        c.require(accepted['candidate']==candidate.id and accepted['accepted_tasks']==[t['receipt_sha256'] for t in witness['tasks']], 'REUSE_CORRECTNESS_CLOSURE')
        events=c.read(proof['events'])
        c.require(events['candidate']==candidate.id and len(events['task_event_hashes'])==len(witness['tasks']), 'REUSE_EVENT_CLOSURE')
        for value in events['task_event_hashes']:c.policy.sha(value,'saved_event_hash')
        return c.reuse_costs(witness,current,candidate)

    def allowance(self,candidate,diagnostic):
        inputs=diagnostic['inputs'];inputs.validate(self.reference_id)
        result=self.accounting.empirical_allowance(reference_error=inputs.reference_error.value,
            path_errors=[x.value for x in inputs.path_signal_errors+inputs.path_vector_errors],
            normalization_error=inputs.log_B_discrepancy.value,raw_closure=inputs.corrected_signal_closure.value,
            maximum_norm=inputs.maximum_norm.value,
            work=sum(inputs.work_primary.values())+sum(inputs.work_independent.values()),
            settings=dict(roundoff_multiplier=64,safety_factor=10,floor=1e-12,scope=c.SERIES))
        return result

    def account(self,candidate,diagnostic,allowance,costs):
        shots=self.shots({k:abs(v) for k,v in diagnostic['bias'].items()},
            {k:allowance[k] for k in ('real','imag')},epsilon=.0005,
            log_B_upper=diagnostic['primary']['log_B']+allowance['log_B_margin'],
            evidence_kind='EMPIRICAL',evidence_ref=candidate.id+'_diagnostics.json',alpha_axis=.025)
        usable=shots['N_total'] is not None and all(r['u_over_headroom'] is not None and r['u_over_headroom']<=.01 for r in shots['axes'].values())
        G=None
        if usable:
            G=sum(shots['axes'][channel]['shots']*sum(t[axis] for t in costs)/len(costs)
                  for axis,channel in (('cosine','real'),('sine','imag')))
            c.policy.number(G,'G')
        row=dict(candidate=candidate.id,series=c.SERIES,allowance_policy=c.policy.ALLOWANCE_POLICY,
                 allowance=allowance,shots=shots,G=G,eligible=usable,conditional_empirical=True)
        c.save(self.s.root/'results'/(candidate.id+'_accounting.json'),row)
        return row

    def rank(self,rows):
        c.require([r['candidate'] for r in rows]==[x.id for x in c.policy.CANDIDATES], 'FOUR_RANK_ROWS')
        if not all(r['eligible'] for r in rows):return dict(status='INCOMPLETE_OR_INELIGIBLE_NO_C_DECISION',rows=rows,science_GO=False)
        full=rows[1]['G']<rows[0]['G'];pr=rows[3]['G']<rows[2]['G']
        return dict(status='CONDITIONAL_EMPIRICAL_COMPARISON',rows=rows,
                    full_Y4_cheaper=full,PR_Y4_cheaper=pr,reversal_candidate=(not full and pr),
                    causal_attribution=None,science_GO=False)
