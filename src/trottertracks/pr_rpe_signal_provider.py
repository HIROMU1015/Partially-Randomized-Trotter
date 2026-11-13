"""Direct-head, paired finite-RTE mean provider. Explicit numerical calls only.

No candidate generation, N/decoder/price optimizer, sampler or reference action.
Real calls require their own scientific execution scope. Import is passive.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from . import pr_rpe_signal_plan as metadata
common = metadata.common
MODULE = 'src/trottertracks/pr_rpe_signal_provider.py'
RECIPE = 'DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1'


class NumericalFailure(ValueError):
    def __init__(self,message,observed):
        super().__init__(message)
        self.observed=observed


def numerical_recipe(source):
    return dict(source=deepcopy(source),deterministic_PF='DIRECT_EXPM',
        finite_RTE=RECIPE,sampling=False)


def make_context(matrices, state, constant, tails, *, binding, source):
    """Supplied saved/fictional arrays only. Never generate H/state/tail norms."""
    import numpy as np
    if not source or len(matrices)!=binding['rank']+1:
        raise ValueError('Source-bound one-body plus saved-rank fragments required')
    state=np.array(state,dtype=np.complex128,copy=True)
    if state.ndim!=1 or not np.isfinite(state).all() or abs(np.linalg.norm(state)-1)>1e-12:
        raise ValueError('Finite normalized supplied state required; no renormalization')
    blocks=tuple(np.array(a,dtype=np.complex128,copy=True) for a in matrices)
    if not math.isfinite(constant):raise ValueError('Finite saved scalar required')
    for a in blocks:
        if (a.shape!=(len(state),len(state)) or not np.isfinite(a).all()
                or np.linalg.norm(a-a.conj().T,'fro')>1e-9):
            raise ValueError('Finite saved Hermitian blocks required; no symmetrization')
        a.setflags(write=False)
    state.setflags(write=False)
    descriptors=deepcopy(tails)
    for prefix,t in descriptors.items():
        if (type(prefix) is not int or not 1<=prefix<binding['rank'] or not t.get('source')
                or not math.isfinite(t['scalar']) or not math.isfinite(t['lambda_R']) or t['lambda_R']<=0):
            raise ValueError('Explicit prefix-specific scalar/lambda SOURCE required')
        t['matrix']=sum(blocks[prefix+1:],np.zeros_like(blocks[0]))-t['scalar']*np.eye(len(state))
        t['matrix'].setflags(write=False)
    return SimpleNamespace(matrices=blocks,state=state,constant=float(constant),tails=descriptors,
        binding=deepcopy(binding),source=deepcopy(source),heads={},exact_tails={},macros={})


def _reference(job, ref):
    c=job['physical_context']['physical_candidate']
    if (not ref.get('source') or not common.same(ref['binding'],job['physical_context']['binding'])
            or ref['T']!=c['T']):raise ValueError('Reference binding/time/SOURCE mismatch')
    e=ref['evidence'];a=complex(e['reference']['real'],e['reference']['imag'])
    b=complex(e['independent_reference']['real'],e['independent_reference']['imag'])
    pair=e['signal_difference']
    if (not all(math.isfinite(x) for x in (a.real,a.imag,b.real,b.imag,pair))
            or not 0<=pair<=1e-10 or abs(a-b)>1e-10):raise ValueError('Saved reference pair gate')
    return a,max(pair,abs(a-b))


def _macros(ctx,c,st):
    import numpy as np
    from scipy.linalg import expm
    from trotterlib.rte import finite_taylor_operator,finite_rte_distribution
    prefix=c['prefix'];tail=ctx.tails[prefix]
    key=(prefix,c['formula'],c['T']/c['q'],c['R_outer'],c['K'],tuple(st['signed_weights']),tuple(st['allocation']))
    if key in ctx.macros:return ctx.macros[key],True
    identity=np.eye(len(ctx.state),dtype=np.complex128)
    ops=[identity.copy() for _ in range(3)];logs=[];sensitivities=[];pieces=[];head_errors=[]
    for w,count in zip(st['signed_weights'],st['allocation']):
        time=c['T']*w/c['q'];dt=time/count;head=[]
        for i in range(prefix+1):
            hkey=(i,time)
            if hkey not in ctx.heads:ctx.heads[hkey]=expm(-.5j*time*ctx.matrices[i])
            factor=ctx.heads[hkey];head.append(factor)
            head_errors.append(float(np.linalg.norm(factor.conj().T@factor-identity,'fro')))
        tau=tail['lambda_R']*dt
        # Optional caller-owned cache lets the same full/sector diagnostic
        # share a normalizer. Its value depends only on the identical tau/K;
        # no operator/state/event law is changed or sampled here.
        cache=getattr(ctx,'normalizers',None);nkey=(tau.hex(),c['K'])
        if cache is not None and nkey in cache:
            b=cache[nkey]
        else:
            if cache is not None and len(cache)>=getattr(ctx,'normalizer_budget',math.inf):
                raise NumericalFailure('Finite-normalizer scope exhausted',dict(tau=tau,K=c['K']))
            b=finite_rte_distribution(tau,c['K']).exact_finite_distribution
            if cache is not None:cache[nkey]=b
        if not math.isfinite(b) or b<=1:
            raise NumericalFailure('Finite-normalizer precision unresolved',
                dict(piece_index=len(pieces),signed_time=time,events=count,event_time=dt,tau=tau,
                    normalizer=None if not math.isfinite(b) else b))
        # Existing paired-Taylor primitive, with physical A and dt. The
        # polynomial depends on dt*A; lambda is used by its sampling law.
        polynomial=finite_taylor_operator(tail['matrix'],dt,c['K'])
        corrected=np.linalg.matrix_power(polynomial,count)
        raw=np.linalg.matrix_power(polynomial/b,count)
        ekey=(prefix,time)
        if ekey not in ctx.exact_tails:ctx.exact_tails[ekey]=expm(-1j*time*tail['matrix'])
        for index,update in enumerate((ctx.exact_tails[ekey],corrected,raw)):
            for factor in head:ops[index]=factor@ops[index]
            ops[index]=update@ops[index]
            for factor in reversed(head):ops[index]=factor@ops[index]
        logs.append(count*math.log(b))
        sensitivities.append(count*(math.log(math.nextafter(b,math.inf))-math.log(math.nextafter(b,0.))))
        pieces.append(dict(signed_time=time,events=count,event_time=dt,tau=tau,b=b))
    packed=dict(operators=ops,log_B_per_outer=math.fsum(logs),pieces=pieces,
        log_sensitivity_per_outer=math.fsum(sensitivities),maximum_head_unitarity=max(head_errors))
    ctx.macros[key]=packed
    return packed,False


def evaluate(ctx,job,reference,*,root):
    """One explicit mean call. Numerical FAIL returns observed diagnostics."""
    import numpy as np
    from .pf_power_stability import power_diagnostics
    from .h4_four_method_screen_mean import allowance_formula
    context=job['physical_context'];c=context['physical_candidate'];st=context['structure']
    dictionary=context.get('dictionary_identity')
    if dictionary is not None:
        # B owns this canonical ID. Do not reuse the legacy recipe hash for a
        # Pauli dictionary, even when q/R happen to match a DF candidate.
        tail=ctx.tails.get(c['prefix'],{})
        if (dictionary.get('kind')!='GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL'
                or not dictionary.get('source') or not dictionary.get('dictionary_sha256')
                or not common.same(dictionary,context['split'].get('tail_dictionary'))
                or not common.same(dictionary,context['law'].get('tail_dictionary'))
                or not common.same(dictionary,tail.get('dictionary_identity'))):
            raise ValueError('Pauli dictionary/SOURCE/law identity mismatch')
        expected_candidate_id=job['B_candidate_id']
    else:
        expected_candidate_id=common.candidate_id(c)
    if (not common.same(context['binding'],ctx.binding) or c['id']!=expected_candidate_id
            or c['method']!='PR' or c['system']!=ctx.binding['system']
            or not common.same(st,common.structure(c)) or c['formula'] not in ('2nd','4th(m5_best)')
            or c['K'] not in (2,4,6) or c['prefix'] not in ctx.tails):
        raise ValueError('PR candidate/context/structure outside supported scope')
    identity=job['B_identity'];bc=identity['candidate']
    if (bc['id']!=job['B_candidate_id'] or identity['configuration']['configuration_id']!=job['B_configuration_id']
            or identity['m']!=job['m'] or identity['q_PF']!=c['q'] or identity['q_RPE']!=job['q_RPE']
            or not common.same(identity['binding'],ctx.binding) or not common.same(identity['structure'],st)
            or not common.same(identity['profile'],context['profile'])
            or not common.same(identity['configuration']['spec']['law'],context['law'])
            or not common.same(identity['configuration']['spec']['split'],context['split'])
            or (bc['prefix'],bc['formula'],bc['K'],bc['R_outer'])!=(c['prefix'],c['formula'],c['K'],c['R_outer'])
            or _scalar(bc['T'])!=_scalar(c['T']) or _scalar(bc['delta'])!=_scalar(c['delta'])):
        raise ValueError('Foreign B whole-schedule identity')
    if (c['prefix']!=context['split']['L_D'] or context['law']['model']!='ideal_finite_RTE_law'
            or context['split'].get('one_body_policy') not in (None,'DETERMINISTIC')
            or context['split']['randomized_scope']!='TAIL_AFTER_PREFIX'
            or context['split']['deterministic_scope']!='PREFIX'
            or context['numerical_recipe']['finite_RTE']!=RECIPE
            or context['numerical_recipe']['deterministic_PF']!='DIRECT_EXPM'
            or context['numerical_recipe']['sampling'] is not False):
        raise ValueError('Split/law/numerical recipe mismatch')
    tail=ctx.tails[c['prefix']]
    split=context['split']
    if (not common.same(tail['source'],split['lambda_source']) or
            tail['lambda_R']!=float(_scalar(split['lambda_R']))):
        raise ValueError('Prefix-specific tail lambda/SOURCE mismatch')
    ref,pair=_reference(job,reference)
    observed={}
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            macro,cache_hit=_macros(ctx,c,st)
            exact,gate=power_diagnostics(macro['operators'][0],c['q'],ctx.state)
            observed.update(gate)
            corrected=np.linalg.matrix_power(macro['operators'][1],c['q'])
            raw=np.linalg.matrix_power(macro['operators'][2],c['q'])
            columns=np.column_stack([exact@ctx.state,corrected@ctx.state,raw@ctx.state])
            columns*=np.exp(-1j*c['T']*(ctx.constant+tail['scalar']))
            logB=c['q']*macro['log_B_per_outer'];sensitivity=c['q']*macro['log_sensitivity_per_outer']
            attenuation=math.exp(-logB)
            closure=float(np.linalg.norm(columns[:,2]-attenuation*columns[:,1]))
            maximum=float(max(1.,np.linalg.norm(columns,axis=0).max(),
                *[np.linalg.norm(op@ctx.state) for op in macro['operators']]))
            means=[complex(np.vdot(ctx.state,columns[:,i])) for i in range(3)]
            if not all(math.isfinite(x) for x in (logB,sensitivity,closure,maximum,*[v for z in means for v in (z.real,z.imag)])):
                raise ValueError('Nonfinite finite-mean outputs')
        exact_norm_after_scalar=abs(float(np.linalg.norm(columns[:,0]))-1.)
        observed.update(closure=closure,maximum_head_unitarity=macro['maximum_head_unitarity'],
            exact_tail_norm_error_after_scalar=exact_norm_after_scalar,
            normalization_log_sensitivity=sensitivity,reference_pair_difference=pair)
        ev=dict(maximum_intermediate_norm=maximum,raw_corrected_vector_closure=closure,
            exact_tail_norm_error=max(gate['norm_error'],exact_norm_after_scalar),
            normalizer_one_ulp_log_sensitivity=sensitivity,
            maximum_norm_domain='MACROSTEP_AND_POWER_OUTPUT_BOUNDARIES')
        allow=allowance_formula(str(root))(dict(candidate=c,structure=st),
            dict(signal=dict(allowance_evidence=ev,log_B=logB)),dict(signal_difference=pair))
        allow.update(log_B_margin=max(allow['log_B_margin'],sensitivity),rigorous_bound=False,
            source=dict(commit='7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e',
                path='src/trottertracks/pr_r_dependent_n.py',function='analyze:allowance_prefix'),
            stage_count_is_logical_not_executed_matrix_operation_count=True)
        reasons=[]
        if not gate['gate_pass'] or exact_norm_after_scalar>1e-9:reasons.append('EXACT_TAIL_NORM_OR_UNITARITY_GATE')
        if macro['maximum_head_unitarity']>1e-9:reasons.append('HEAD_UNITARITY_GATE')
        if closure>1e-9:reasons.append('RAW_CORRECTED_CLOSURE_GATE')
        if attenuation==0:reasons.append('ATTENUATION_UNDERFLOW')
        if logB>math.log(float.fromhex('0x1.fffffffffffffp+1023')):reasons.append('B_OVERFLOW')
        if max(abs(means[2].real),abs(means[2].imag))>1+allow['u']:reasons.append('RAW_OUTCOME_RANGE')
        observed.update(exact_tail_gate_pass=gate['gate_pass'],gate_pass=not reasons)
        z=lambda v:dict(real=v.real,imag=v.imag)
        status='NUMERICAL_FAILURE' if reasons else 'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'
        return dict(status=status,physical_context=deepcopy(context),candidate=deepcopy(c),
            B_identity=deepcopy(job['B_identity']),B_law_binding=deepcopy(job['B_law_binding']),
            B_configuration_id=job['B_configuration_id'],B_candidate_id=job['B_candidate_id'],m=job['m'],
            signed_weights=deepcopy(st['signed_weights']),allocation=deepcopy(st['allocation']),event_count=st['event_count'],
            signal=dict(reference=z(ref),exact_tail=z(means[0]),corrected=z(means[1]),raw=z(means[2]),
                total_signed_bias=z(means[1]-ref),outer_PF_or_discard_bias=z(means[0]-ref),
                finite_RTE_bias=z(means[1]-means[0]),log_B=logB,B=None if 'B_OVERFLOW' in reasons else math.exp(logB),
                attenuation=attenuation,allowance_evidence=ev),allowance=allow,numerical=observed,
            numerical_failure_reasons=reasons,piece_diagnostics=deepcopy(macro['pieces']),macro_cache_hit=cache_hit,
            source=deepcopy(context['numerical_recipe']['source']),input_source=deepcopy(ctx.source),
            reference_source=deepcopy(reference['source']),reference_evidence=deepcopy(reference['evidence']),
            state_evidence=deepcopy(getattr(ctx,'state_metadata',None)),
            scalar_once=True,raw_corrected_independent_propagation=True,
            N_X=None,N_Y=None,C=None,G=None,global_RMSE_bound=None,scientific_PASS_FAIL=None,
            qualification=dict(ground=None,actual_sampler_law=None,finite_K_RMSE_transfer=None),automatic_retry=False)
    except (ValueError,OverflowError,FloatingPointError) as error:
        if isinstance(error,NumericalFailure):observed.update(error.observed)
        observed['gate_pass']=False
        return dict(status='NUMERICAL_FAILURE',physical_context=deepcopy(context),candidate=deepcopy(c),m=job['m'],
            B_identity=deepcopy(job['B_identity']),B_law_binding=deepcopy(job['B_law_binding']),
            B_configuration_id=job['B_configuration_id'],B_candidate_id=job['B_candidate_id'],
            source=deepcopy(context['numerical_recipe']['source']),input_source=deepcopy(ctx.source),
            reference_source=deepcopy(reference['source']),signal=None,numerical=observed,
            error=dict(type=type(error).__name__,message=str(error)),N_X=None,N_Y=None,C=None,G=None,
            global_RMSE_bound=None,scientific_PASS_FAIL=None,automatic_retry=False)


def _scalar(value):
    from fractions import Fraction
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


class SignalBatch:
    """B-validated one-configuration batch. run() is an explicit scientific leaf.

    Factories must supply bound arrays and saved references; none are implicit.
    Saved evidence needs its existing acceptance adapter, never a fresh retry.
    """
    def __init__(self,config,physical_by_round,*,b_validate,b_exact_scalar,source,
                 context_factory,reference_provider,root,saved_inventory=()):
        if not source or context_factory is None or reference_provider is None:
            raise ValueError('Explicit source/context factory/saved reference provider required')
        self.config=deepcopy(config);self.physical=deepcopy(physical_by_round)
        self.source=deepcopy(source);self.root=Path(root);self.validate=b_validate;self.exact=b_exact_scalar
        self.factory=context_factory;self.reference=reference_provider;self.inventory=deepcopy(saved_inventory)
        self.pending=self._plan()

    def _plan(self):
        return metadata.plan(self.config,self.physical,b_validate=self.validate,b_exact_scalar=self.exact,
            numerical_recipe=numerical_recipe(self.source),saved_inventory=self.inventory)

    def run(self,out):
        out=Path(out)
        if out.exists():raise ValueError('Fresh output required; no retry/resume')
        if not common.same(self.pending,self._plan()):raise ValueError('Plan mutated after B validation')
        out.mkdir(parents=True)
        def save(name,value):
            with (out/name).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        save('PLAN.json',self.pending)
        result=dict(status='STARTED',configuration_id=self.config['id'],source=deepcopy(self.source),
            records=[],errors=[],not_started_rounds=[],reference_actions=0,automatic_retry=False,
            sufficient_N=None,global_RMSE_bound=None,C=None,G=None,winner=None,mandatory_stop=True)
        current_m=None
        try:
            if any(j['status']!='MISSING_MEAN_REQUIRES_PERMISSION' for j in self.pending['jobs']):
                raise ValueError('Saved evidence requires existing acceptance; no reanalysis or retry fallback')
            ctx=self.factory(self.config)
            for job in self.pending['jobs']:
                current_m=job['m']
                record=evaluate(ctx,job,self.reference(job),root=self.root)
                save('m'+str(job['m'])+'.json',record);result['records'].append(record)
                if record['status']=='NUMERICAL_FAILURE':break
            result['status']=('STOP_NUMERICAL_FAILURE' if result['records'] and result['records'][-1]['status']=='NUMERICAL_FAILURE'
                else 'FINITE_MEAN_SCHEDULE_COMPLETE_NOT_RMSE_QUALIFIED')
        except Exception as error:
            result['status']='STOP_INPUT_OR_SOURCE_FAILURE'
            result['errors'].append(dict(m=current_m,type=type(error).__name__,message=str(error)))
        completed={r['m'] for r in result['records']}
        result['not_started_rounds']=[j['m'] for j in self.pending['jobs'] if j['m'] not in completed]
        save('result.json',result)
        return result


def load_h4_context(root,config):
    """Explicit saved-H4 factory, no SCF/DF/state generation or lambda fit.

    Tail lambda_source in B must match the returned source descriptor exactly.
    The caller fixes that descriptor in its configuration before run().
    """
    import sys
    from .h4_pr_rmse_completion import load_context,INDEX
    from .h4_four_method_screen import PHYS
    root=Path(root);store=common.SavedArtifacts(root)
    base=store.read(PHYS)['binding'];binding=config['spec']['binding']
    if any(binding.get(k)!=base[k] for k in ('system','rank','rank_policy','hamiltonian','state','snapshot')):
        raise ValueError('Saved H4 DF10 input binding required')
    index=store.read(INDEX);spec=index['T20_reference_files']['H4']
    data=Path(spec['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Saved H4 tail descriptor changed')
    info=json.loads(data)
    if not common.same(info['binding'],base):raise ValueError('Saved tail H/state binding differs')
    prefix=str(config['spec']['L_D'])
    if prefix not in info['prefixes']:raise ValueError('No saved prefix-specific lambda/scalar descriptor')
    tails={int(prefix):dict(info['prefixes'][prefix],source=dict(index=deepcopy(INDEX),
        context_sha256=spec['sha256'],selector=['prefixes',prefix]))}
    tail=tails[int(prefix)];split=config['spec']['split']
    if (not common.same(tail['source'],split['lambda_source'])
            or tail['lambda_R']!=float(_scalar(split['lambda_R']))):
        raise ValueError('B split must identify the exact saved tail descriptor')
    scripts=str(root/'scripts')
    if scripts not in sys.path:sys.path.insert(0,scripts)
    saved,state_metadata=load_context(root,base)
    ctx=make_context(saved.matrices,saved.state,saved.ham.constant,tails,binding=binding,source=INDEX)
    ctx.state_metadata=state_metadata
    return ctx


def saved_h4_reference(root,job,*,store=None):
    """Saved pair lookup only, with no new action fallback."""
    from .h4_four_method_screen import PHYS
    data=(store if store is not None else common.SavedArtifacts(root)).read(PHYS)
    binding=job['physical_context']['binding'];c=job['physical_context']['physical_candidate']
    if any(binding.get(k)!=data['binding'][k] for k in ('system','rank','rank_policy','hamiltonian','state','snapshot')):
        raise ValueError('Saved reference H/state mismatch')
    row=data['rounds'][job['m']]
    if row['physical_candidate']['T']!=c['T']:raise ValueError('Saved reference time mismatch')
    return dict(binding=deepcopy(binding),T=c['T'],source=dict(**PHYS,selector=['rounds',job['m'],'diagnostics','reference_pair']),
        evidence=deepcopy(row['diagnostics']['reference_pair']))
