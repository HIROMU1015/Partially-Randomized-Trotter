"""UNEXECUTED future one-shot worker, deterministic coherent means only."""
from fractions import Fraction
import hashlib,json,math,os
from pathlib import Path
import resource,signal,time
from .control import POLICY,require_permit,planned_rows


def _write(path,raw,cap):
    if len(raw)>cap:raise RuntimeError('output byte cap')
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
def _json(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
def _rational(x):return {'numerator_hex':hex(x.numerator),'denominator_hex':hex(x.denominator)}


class Pilot:
    def __init__(self,root,permit,report):
        require_permit(permit)
        self.root,self.permit,self.report=root,permit,report
        from .numerics import Kernels
        self.k=Kernels(permit)
        self.plan=planned_rows(permit.config)
        self.exact_events=0

    def phase_metrics(self,z,e,reference,ref_error):
        k=self.k
        visibility=abs(z);vr=k.up(e+k.gamma(8)*visibility)
        q=z*reference.conjugate()
        qr=k.up(abs(reference)*e+abs(z)*ref_error+e*ref_error+k.gamma(8)*abs(z)*abs(reference))
        floor=float(Fraction(POLICY['precision']['phase_visibility_floor']))
        out={'z':[z.real,z.imag],'signal_error_radius':e,'visibility':visibility,'visibility_error_radius':vr,
             'loss_from_one':1-visibility,'reference_visibility':abs(reference),
             'total_visibility_loss_relative_to_raw_H':abs(reference)-visibility,
             'total_complex_discrepancy_to_raw_H':[ (z-reference).real,(z-reference).imag ],'total_discrepancy_numerical_radius':k.up(e+ref_error+k.gamma(2)*(abs(z)+abs(reference))),
             'phase_interpretation':'TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H','phase_status':'UNDETERMINED_PHASE','phase_bias_rad':None,'phase_error_radius_rad':None,
             'negative_log_visibility':None}
        if visibility-vr<=floor or abs(reference)-ref_error<=floor:
            out['phase_reason']='visibility floor';return out
        if q.real<=qr and abs(q.imag)<=qr:
            out['phase_reason']='principal-angle branch cut';return out
        if qr>=abs(q):out['phase_reason']='phase ball includes zero';return out
        phase_error=k.up(math.asin(qr/abs(q))+2.0**-51)
        if phase_error>float(Fraction(POLICY['precision']['eps_phase_rad'])):
            out['phase_reason']='phase precision unresolved';return out
        out.update(phase_status='RESOLVED_PRINCIPAL_PHASE',phase_bias_rad=math.atan2(q.imag,q.real),phase_error_radius_rad=phase_error,
                   negative_log_visibility=-math.log(visibility),log_visibility_error_radius=k.up(vr/(visibility-vr)+2.0**-51))
        return out

    def absolute_interval(self,z):
        # Exact scalar arithmetic on stored binary64 values; same unverified
        # |fl(hypot)-hypot| <= 2u*hypot assumption as the unchanged precision model.
        a=Fraction(abs(z));relative=Fraction(1,1<<52)
        return a/(1+relative),a/(1-relative)

    def model_accounting(self,descriptor,z,reference,metrics):
        k=self.k
        bound=self.model_bounds[descriptor['split']]
        eta=abs(Fraction(descriptor['T']))*bound
        ref,ref_error=reference
        lo,hi=self.absolute_interval(z-ref)
        radius=Fraction(metrics['total_discrepancy_numerical_radius'])
        enlarged=radius+eta
        # These are enclosures of |z_method-z_tilde_ideal|, not a measured
        # algorithm-only point estimate. All reported endpoints remain rational.
        lower=max(Fraction(0),lo-enlarged);upper=hi+enlarged
        model={'operator_bound_Ha_exact':_rational(bound),
               'signal_bound_abs_exact':_rational(eta),
               'signal_bound_abs_upper':k.rational_up(eta),
               'bound_formula':'abs(T)*B_j','reference':'H_full',
               'method_target':'H_tilde_j=HD_j+HR_saved_j',
               'attribution_status':'MODEL_TARGET_CONFOUNDED' if eta else 'MODEL_TARGET_EQUAL_AT_THIS_TIME',
               'model_phase_status':'UNDETERMINED','model_phase_bound_rad':None,
               'model_phase_reason':'visibility/zero/branch conditions not established'}
        rlo,rhi=self.absolute_interval(ref)
        L=rlo-Fraction(ref_error)
        floor=Fraction(POLICY['precision']['phase_visibility_floor'])
        disk=Fraction(ref_error)+eta
        # Conservative rectangle encloses the disk: reject whenever it can
        # intersect the principal Arg cut. No phase unwrapping is inferred.
        cut=Fraction(ref.real)<=disk and abs(Fraction(ref.imag))<=disk
        if L>floor and L-eta>floor and eta<L and not cut:
            ratio=k.rational_up(eta/L)
            if ratio<1:
                model.update(model_phase_status='CONDITIONAL_PRINCIPAL_MODEL_PHASE_BOUND',
                  model_phase_bound_rad=k.up(math.asin(ratio)+2.0**-51),
                  model_phase_reason='raw and split visibility lower bounds exceed floor; union disk excludes zero and principal cut')
        return {'numerical_rounding':{'signal_radius':metrics['signal_error_radius'],
                  'raw_reference_radius':ref_error,'total_discrepancy_radius':metrics['total_discrepancy_numerical_radius'],
                  'precision_gate_excludes_model_drift':True},
                'Hamiltonian_model_difference':model,
                'finite_step_algorithm_error':{'target':'H_tilde_j',
                  'magnitude_interval_exact':{'lower':_rational(lower),'upper':_rational(upper)},
                  'point_value':None,'algorithm_only_error_measured':False,
                  'interpretation':'TRIANGLE_INEQUALITY_ENCLOSURE_ONLY_NOT_EXACT_SEPARATION'},
                'total_discrepancy_interpretation':'TOTAL_COMPLEX_DISCREPANCY_TO_RAW_H'}

    def store(self,descriptor,a,state,norm,reference):
        z,error=self.k.signal(a,state,norm)
        metrics=self.phase_metrics(z,error,*reference)
        accounting=self.model_accounting(descriptor,z,reference,metrics)
        row=dict(descriptor,**metrics,error_ledger=accounting,status='NUMERIC_DIAGNOSTIC_ONLY',evidence='CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
        self.report['rows'].append(row)
        if error>float(Fraction(POLICY['precision']['eps_signal_abs'])):
            row['status']='PRECISION_UNRESOLVED'
            raise RuntimeError('signal precision budget exceeded; no tolerance change or retry')
        return row

    def run(self):
        k=self.k
        from .word_reference import verify_future
        self.report['numerical_correctness']=verify_future(self.permit,k)
        from .exact_input import ExactInput
        exact=ExactInput(self.permit);target=exact.targets(self.root);self.exact_events=exact.events
        self.model_bounds={split:data['model_difference_operator_bound_Ha'] for split,data in target['splits'].items()}
        self.report['model_ledger']={'raw_reference':'H_full','method_target':'H_tilde_j=HD_j+HR_saved_j',
          'full':target['full_matrix_identity'],'splits':{split:{
            'model_difference_operator_bound_Ha':_rational(B),
            'model_signal_bound_by_T':{T:_rational(abs(Fraction(T))*B) for T in ['0',*POLICY['grid']['T_rel_Ha_inverse']]},
            'B_unit':'Ha','T_unit':'Ha^-1','signal_unit':'dimensionless',
            'exact_split_model_Hermiticity_checked':True} for split,B in self.model_bounds.items()}}
        full=k.exact_dense(target['full'])
        state=k.np.asarray(target['state'],dtype=k.np.complex128);norm=target['norm2']
        reference={'0':(1+0j,0.0)}
        for T in POLICY['grid']['T_rel_Ha_inverse']:
            reference[T]=k.signal(k.exp_i(full,Fraction(T)),state,norm)
            if reference[T][1]>float(Fraction(POLICY['precision']['eps_signal_abs'])):
                raise RuntimeError('ideal signal precision unresolved')
        self.report['reference_signals']={T:{'z':[v[0].real,v[0].imag],'error_radius':v[1]} for T,v in reference.items()}
        for split in ['G3','G6','G9']:
            data=target['splits'][split]
            hd=k.exact_dense(data['HD']);h=k.exact_dense(data['HR'],data['lambda'],known_op_bound=1.0)
            twirl=k.twirl_plan(data['coefficients'],256)
            self.report.setdefault('full_size_twirl_correctness',{})[split]=k.verify_twirl_probe(data['coefficients'],twirl)
            self.report['model_ledger']['splits'][split].update({'HD':exact.matrix_digest(data['HD']),'HR_saved_law':exact.matrix_digest(data['HR']),
              'lambda_exact_Ha':_rational(data['lambda']),'model_difference_operator_bound_Ha':_rational(data['model_difference_operator_bound_Ha']),
              'saved_table_sha256':data['saved_table_sha256'],'law_target':'exact abs(saved signed binary64 coeff)/exact sum(abs(coeff))',
              'legacy_full_HD_direct_comparison':'NOT_SAVED_BY_V5_UNDETERMINED; this is a new pilot construction'})
            d_cache={delta:(k.exp_i(hd,Fraction(delta)/4),k.exp_i(hd,Fraction(delta)/2)) for delta in POLICY['grid']['delta_rel_Ha_inverse']}
            for R in POLICY['grid']['R_total_labels_per_step']:
                for delta in POLICY['grid']['delta_rel_Ha_inverse']:
                    D,Do=d_cache[delta];angle=data['lambda']*Fraction(delta)/R
                    O=k.mul(k.mul(Do,k.power(k.mean(h,angle),R)),Do)
                    V=k.mul(k.mul(D,k.power(k.mean(h,angle),R//2)),D)
                    checkpoints={int(Fraction(T)/Fraction(delta)):T for T in POLICY['grid']['T_rel_Ha_inverse']}
                    shared=k.eye(256)
                    if POLICY['grid']['s_zero_control']:
                        for descriptor in self.plan:
                            if descriptor['split']==split and descriptor['R']==R and descriptor['delta']==delta and descriptor['T']=='0':
                                self.store(descriptor,shared,state,norm,reference['0'])
                    for s in range(1,max(checkpoints)+1):
                        shared=k.shared(shared,D,h,angle/2,R,twirl)
                        if s not in checkpoints:continue
                        T=checkpoints[s]
                        vp=k.power(V,s)
                        matrices={'shared_pm':shared,'ordinary':k.power(O,s),'independent_pm':k.mul(vp,vp)}
                        rows={}
                        for descriptor in self.plan:
                            if descriptor['split']==split and descriptor['R']==R and descriptor['delta']==delta and descriptor['T']==T:
                                rows[descriptor['method']]=self.store(descriptor,matrices[descriptor['method']],state,norm,reference[T])
                        for comparator in ['ordinary','independent_pm']:
                            a,b=rows['shared_pm'],rows[comparator]
                            difference=a['visibility']-b['visibility'];radius=k.up(a['visibility_error_radius']+b['visibility_error_radius']+k.gamma(2)*(abs(a['visibility'])+abs(b['visibility'])))
                            contrast='UNRESOLVED_ABSOLUTE_CONTRAST' if abs(difference)<=radius else 'RESOLVED_HIGHER_VISIBILITY' if difference>0 else 'RESOLVED_LOWER_VISIBILITY'
                            beta=data['lambda']**2/Fraction(4*R);u=Fraction(s)*Fraction(delta)**2
                            old=data['old_W_v_rho']
                            self.report['comparisons'].append({'split':split,'R':R,'delta':delta,'T':T,'comparator':comparator,
                              'comparison_target':'COMMON_H_TILDE_WITHIN_SPLIT','model_drift_in_numerical_radius':False,
                              'visibility_difference_shared_minus_comparator':difference,'absolute_error_radius':radius,'contrast_status':contrast,
                              'phase_status_shared':a['phase_status'],'phase_status_comparator':b['phase_status'],
                              'beta_Ha2':_rational(beta),'u_Ha_inverse_squared':_rational(u),
                              'leading_kappa_sh_hypothesis_only':float(beta*u)*(old['W']+old['v']),
                              'leading_kappa_ordinary_hypothesis_only':float(2*beta*u)*old['W'],
                              'T1b_scalar_domain_certified':False,'relative_visibility_improvement':None,
                              'relative_ratio_reason':'absolute differences primary; no division by tiny or unresolved losses',
                              'scientific_GO':None})
        self.report['status']='PILOT_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'


def run_authorized(root,permit):
    require_permit(permit)
    root=Path(root).resolve();limits=permit.config['resources'];started=time.monotonic()
    for kind,cap in [(resource.RLIMIT_AS,limits['memory_bytes']),(resource.RLIMIT_FSIZE,limits['result_bytes'])]:
        soft,hard=resource.getrlimit(kind);value=min([cap]+[x for x in [soft,hard] if x!=resource.RLIM_INFINITY]);resource.setrlimit(kind,(value,hard))
    output=root/permit.audit['output_directory'];output.mkdir(parents=True,exist_ok=False)
    _write(output/'execution_context.json',_json({'audit':permit.audit,'config':permit.config,'one_shot_consumed':True,'automatic_retry':False}),limits['context_bytes'])
    logfd=os.open(output/'runner.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    saved=(os.dup(1),os.dup(2));os.dup2(logfd,1);os.dup2(logfd,2);os.close(logfd)
    report={'schema':'m1s_finite_signal_result_v2','status':'TECHNICAL_STOP','audit':permit.audit,'config':permit.config,
       'model_accounting_policy':permit.config['model_accounting'],'auxiliary_split_ideal_signal_evaluated':False,
       'rows':[],'comparisons':[],'scientific_GO_or_no_GO_determined':False,'platform_certified':False,'automatic_retry':False,'mandatory_STOP':True}
    engine=None;handlers={}
    def interrupted(signum,frame):raise InterruptedError('signal '+str(signum))
    try:
        for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,interrupted)
        engine=Pilot(root,permit,report);engine.run()
    except (Exception,KeyboardInterrupt) as exc:
        report['failure']={'type':type(exc).__name__,'reason':str(exc)}
    finally:
        for sig,old in handlers.items():signal.signal(sig,old)
        import sys
        sys.stdout.flush();sys.stderr.flush();os.dup2(saved[0],1);os.dup2(saved[1],2);os.close(saved[0]);os.close(saved[1])
    completed={(r['split'],r['R'],r['delta'],r['T'],r['method']) for r in report['rows']}
    for d in planned_rows(permit.config):
        if (d['split'],d['R'],d['delta'],d['T'],d['method']) not in completed:report['rows'].append(dict(d,status='UNDETERMINED_NOT_EVALUATED_AFTER_STOP'))
    report['resources']={'elapsed_seconds':time.monotonic()-started,'maxrss_linux_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'matrix_products':engine.k.products if engine else 0,'twirl_applications':engine.k.twirls if engine else 0,'exact_events':engine.exact_events if engine else 0,
      'actual_address_space_cap':resource.getrlimit(resource.RLIMIT_AS)[0],'actual_file_cap':resource.getrlimit(resource.RLIMIT_FSIZE)[0]}
    raw=_json(report)
    if len(raw)>limits['result_bytes']:
        report={'schema':'m1s_finite_signal_result_v2','status':'TECHNICAL_STOP','reason':'result byte cap','oversize_payload_sha256':hashlib.sha256(raw).hexdigest(),
          'audit':permit.audit,'rows':[dict(d,status='UNDETERMINED_OUTPUT_LIMIT') for d in planned_rows(permit.config)],'mandatory_STOP':True}
        raw=_json(report)
    _write(output/'result.json',raw,limits['result_bytes']);_write(output/'result.sha256',(hashlib.sha256(raw).hexdigest()+'\n').encode(),128)
    return {'status':report['status'],'output_directory':str(output),'mandatory_STOP':True}
