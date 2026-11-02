"""F1-only registered-case executor. Import only after the authorization gate.

No automatic prerequisites, pilots, retries, sampling, F2, or condition search.
Case IDs/counts and all payloads must come from a future separately approved B
binding and final F1 packet. No such packet exists in stage I.
"""
from __future__ import annotations
from dataclasses import asdict
from fractions import Fraction
import json
import math
from .fixture import specification_from_record, EvolutionContext, CompilerProfile
from .prototype import MixedAtom,make_tail,configure,ordered_event
from .circuits import (BasisProvider,component_probe,build_evolution,
                       actual_adjoint,hadamard_wrapper,build_event_sequence)
from . import reference as ref
from . import gates
from . import pricing
from .authorization import file_sha,_under


def _atom(record):
    return MixedAtom(record['atom_id'],record['kind'],Fraction(record['coefficient']),
        tuple(record['origins']),tuple(record['word']),record['basis_key'],record.get('frame_id'))


class RegisteredF1:
    def __init__(self,verified):
        self.verified=verified; self.contract=verified['contract']; self.cache={}; self.contexts={}; self.compile_cache={}; self.law_coverage={}
        name=self.contract['fixture_bundle_path']
        bundle=json.loads(verified['inputs'][name])
        self.spec=specification_from_record(bundle['specification'])
        self.context=EvolutionContext('1/5',2,2,2)
        self.atoms={a:tuple(_atom(r) for r in bundle['arms'][a]) for a in ('SS','SP','PS','PP')}
        self.atoms['catalog']=tuple(_atom(r) for r in bundle['catalog_atoms'])
        self.provider=BasisProvider(self.spec.modes,
            {key:tuple(tuple(complex(Fraction(x)) for x in row) for row in matrix)
             for key,matrix in bundle['rational_frames'].items()})
        self.price_tables={}
        for arm,table in bundle.get('literal_prices',{}).items():
            self.price_tables[arm]={k:pricing.AtomPrice(**v) for k,v in table.items()}
        self.score_records=bundle.get('score_intervals',{})

    def law(self,arm):
        if arm not in self.contexts:
            tail=make_tail(self.atoms[arm],arm)
            tolerance=self.contract['truncation_tolerance'][arm]
            if arm=='catalog':
                from trotterlib.rte import make_rte_config
                config,distribution=make_rte_config(tail,evolution_time=1/20,rte_steps=1,
                    finite_taylor_order=2,truncation_tolerance=tolerance,seed=0)
            else: config,distribution=configure(tail,self.context,truncation_tolerance=tolerance)
            if config is None: raise ValueError('No randomized coverage for fixed F1 fixture')
            self.contexts[arm]=(tail,config,distribution)
        return self.contexts[arm]

    def event(self,arm,record):
        tail,_config,law=self.law(arm)
        return ordered_event(tail,law,len(record['products']),record['products'],record['child'])

    def macros(self,arm,records):
        return tuple(tuple(self.event(arm,e) for e in macro) for macro in records)

    def reference_case(self,case):
        ctx=ref.precision_context(case['digits']); arm=case['arm']; atoms=self.atoms[arm]
        kind=case['reference_kind']; h,d,r,fs,frames=ref.original_operators(ctx,self.spec)
        result=dict(ctx=ctx,h=h,d=d,r=r,frames=frames)
        if kind=='target':
            result['operator']=ref.finite_target(ctx,self.spec,self.context)
            result['step']=ref.polynomial(ctx,r,ref.exact(ctx,str(self.context.micro_time)),3)
        elif kind=='motif':
            macros=self.macros(arm,case['macros'])
            u=ref.trajectory_operator(ctx,self.spec,self.context,macros,atoms)
            result.update(operator=u,signal=ref.signal(ctx,u,self.spec.occupied),
                wrapper_X=ref.wrapper_operator(ctx,u,self.spec.occupied,'X'),
                wrapper_Y=ref.wrapper_operator(ctx,u,self.spec.occupied,'Y'))
        elif kind in ('atom','catalog'):
            micro=Fraction(1,20)
            _lam,tau,_orders,_weights,_norm,_probs=ref.independent_law(ctx,atoms,micro,2)
            by_id={a.atom_id:a for a in atoms}
            if kind=='atom':
                atom=by_id[case['atom_id']]; q=ref.unsigned_atom_operator(ctx,atom,frames)
                sign=1 if atom.coefficient>0 else -1
                result['product']=sign*q
                for n in (0,2):
                    angle=sign*ctx.atan(tau/(n+1))
                    result['rotation'+str(n)]=ctx.cos(angle)*ctx.eye(q.rows)-ctx.j*ctx.sin(angle)*q
            else:
                event=self.event(arm,case['event'])
                phase=ctx.exp(ctx.j*ref.exact(ctx,'1/31')*ref.exact(ctx,'1/20'))
                result['operator']=phase*ref.event_operator(ctx,event,by_id,frames,tau)
        else: raise ValueError('Unknown reference kind')
        if case.get('paired_reference'):
            other=self.cache[case['paired_reference']]
            for name in ('operator','step','product','rotation0','rotation2','wrapper_X','wrapper_Y'):
                if name in result and name in other:
                    gates.high_precision_agreement(other[name],result[name])
        return result,dict(reference_kind=kind,digits=case['digits'],status='REFERENCE_COMPUTED')

    def run_case(self,case):
        handler=case['handler']; arm=case.get('arm'); report={}
        if handler=='REFERENCE': return self.reference_case(case)
        if handler=='H_ID':
            reference=self.cache[case['reference']]; ctx=reference['ctx']
            r=ref.dictionary_operator(ctx,self.atoms[arm],reference['frames'])
            h=r+reference['d']+ref.exact(ctx,self.spec.scalar)*ctx.eye(r.rows)
            report=gates.matrix_gate(ref.to_numpy(h),ref.to_numpy(reference['h']),1e-11,label='H_ID')
            # Separate sector report; no projection during intermediate actions.
            sector=[i for i in range(h.rows) if i.bit_count()==self.spec.particles]
            sub=lambda m: ref.to_numpy(m)[sector,:][:,sector]
            report['sector']=gates.matrix_gate(sub(h),sub(reference['h']),1e-11,label='H_SECTOR')
            return h,report
        if handler=='LAW':
            tail,config,law=self.law(arm); e=self.event(arm,case['event'])
            reference=self.cache[case['reference']]; ctx=reference['ctx']
            lam,tau,orders,weights,norm,p=ref.independent_law(ctx,tail.atoms,Fraction(1,20),2)
            indices={a.atom_id:i for i,a in enumerate(tail.atoms)}
            weight=weights[orders.index(e.taylor_order)]
            for aid in (*e.product_component_ids,e.rotation_component_id): weight*=p[indices[aid]]
            error=max(abs(e.event_probability-float(weight/norm)),abs(e.event_coefficient-float(weight)))
            if error>1e-12: raise ValueError('LAW_COEFFICIENT_OR_PROBABILITY_MISMATCH')
            if abs(config.lambda_r-float(lam))>1e-12*max(1,float(lam)):
                raise ValueError('LAW_LAMBDA_MISMATCH')
            if abs(law.exact_finite_distribution-float(norm))>1e-12*max(1,float(norm)):
                raise ValueError('LAW_NORMALIZER_MISMATCH')
            if abs(e.rotation_angle-float(ctx.atan(tau/(e.taylor_order+1))))>1e-12:
                raise ValueError('LAW_ANGLE_MISMATCH')
            state=self.law_coverage.setdefault(arm,dict(keys=set(),probabilities=[]))
            key=(e.taylor_order,e.product_component_ids,e.rotation_component_id)
            if key in state['keys']: raise ValueError('Duplicate finite event')
            state['keys'].add(key); state['probabilities'].append(e.event_probability)
            mass=math.fsum(state['probabilities'])
            total=len(tail.atoms)+len(tail.atoms)**3
            if len(state['keys'])==total and abs(mass-1)>1e-12:
                raise ValueError('FINITE_EVENT_PROBABILITY_MASS_MISMATCH')
            return e,dict(error=error,probability=e.event_probability,coefficient=e.event_coefficient,
                probability_mass=mass,enumerated_events=len(state['keys']),expected_events=total)
        if handler=='COMPONENT':
            _tail,_config,law=self.law(arm)
            atom=next(a for a in self.atoms[arm] if a.atom_id==case['atom_id'])
            role=case['role']; order=case.get('order',0)
            theta=None if role=='product' else math.atan(law.dimensionless_step_time/(order+1))
            circuit=component_probe(atom,self.provider,angle=theta)
            reference=self.cache[case['reference']]
            expected=ref.to_numpy(reference['product' if role=='product' else 'rotation'+str(order)])
            report=gates.controlled_gate(circuit,expected)
            from qiskit.quantum_info import Operator
            n=1<<self.spec.modes; block=Operator(circuit).data[n:,n:]
            return dict(circuit=circuit,block=block,atom_id=atom.atom_id,role=role,order=order),report
        if handler=='MOTIF':
            macros=self.macros(arm,case['macros'])
            evolution=build_evolution(self.spec,self.context,macros,self.atoms[arm],self.provider,
                input_identity=self.contract['fixture_input_sha256'],
                source_identity=self.contract['source_closure_sha256'],
                optimized=case['variant']!='raw')
            circuit=actual_adjoint(evolution) if case['variant']=='adjoint' else evolution.circuit
            expected=ref.to_numpy(self.cache[case['reference']]['operator'])
            if case['variant']=='adjoint': expected=expected.conj().T
            report=gates.controlled_gate(circuit,expected)
            return dict(circuit=circuit,evolution=evolution),report
        if handler=='WRAPPER':
            result=self.cache[case['evolution']]['evolution']; axis=case['axis']
            circuit=hadamard_wrapper(result,self.spec,axis)
            expected=self.cache[case['reference']]
            value=gates.wrapper_signal(circuit,axis)
            wanted=complex(expected['signal']).real if axis=='X' else complex(expected['signal']).imag
            if abs(value-wanted)>2e-10: raise ValueError('WRAPPER_SIGNAL_MISMATCH')
            return dict(circuit=circuit,axis=axis),dict(axis=axis,error=abs(value-wanted))
        if handler=='COMPILE':
            from qiskit.quantum_info import Operator
            raw=self.cache[case['wrapper']]; axis=raw['axis']
            circuit=gates.compiled_wrapper(raw['circuit'],CompilerProfile(),cache=self.compile_cache,
                identity_context=dict(input=self.contract['fixture_input_sha256'],
                    source=self.contract['source_closure_sha256'],
                    environment=self.contract['environment_lock_sha256']))
            unitary=circuit.remove_final_measurements(inplace=False)
            expected=self.cache[case['reference']]
            report=gates.matrix_gate(Operator(unitary).data,ref.to_numpy(expected['wrapper_'+axis]),
                                     5e-10,label='COMPILED_WRAPPER')
            value=gates.wrapper_signal(circuit,axis)
            wanted=complex(expected['signal']).real if axis=='X' else complex(expected['signal']).imag
            if abs(value-wanted)>5e-10: raise ValueError('COMPILED_SIGNAL_MISMATCH')
            report['signal_error']=abs(value-wanted)
            report['counts']={str(k):int(v) for k,v in circuit.count_ops().items()}
            return dict(circuit=circuit),report
        if handler=='MEAN':
            tail,_config,law=self.law(arm); blocks={}
            for cid in case['components']:
                record=self.cache[cid]
                key=(record['atom_id'],'product') if record['role']=='product' else (
                    record['atom_id'],'rotation',record['order'])
                if key in blocks: raise ValueError('Duplicate component block')
                blocks[key]=record['block']
            one=gates.component_derived_mean(tail,law,blocks)
            total=gates.compose_corrected_mean(one,self.spec,self.context)
            expected=self.cache[case['reference']]
            return total,dict(step=gates.matrix_gate(one,ref.to_numpy(expected['step']),2e-10,label='STEP_MEAN'),
                full=gates.matrix_gate(total,ref.to_numpy(expected['operator']),2e-10,label='FULL_MEAN'))
        if handler=='CATALOG_EVENT':
            e=self.event('catalog',case['event'])
            circuit=build_event_sequence((e,),self.atoms['catalog'],self.provider,
                                         optimized=case['variant']=='optimized')
            circuit.p(1/(31*20),self.spec.modes)
            expected=ref.to_numpy(self.cache[case['reference']]['operator'])
            report=gates.controlled_gate(circuit,expected)
            from qiskit.quantum_info import Operator
            n=1<<self.spec.modes
            return dict(circuit=circuit,block=Operator(circuit).data[n:,n:],event=e),report
        if handler=='CATALOG_MEAN':
            import numpy as np
            expected=self.cache[case['reference']]; ctx=expected['ctx']
            atoms=self.atoms['catalog']; frames=expected['frames']
            r=ref.dictionary_operator(ctx,atoms,frames)
            target=ctx.exp(ctx.j*ref.exact(ctx,'1/31')*ref.exact(ctx,'1/20'))*ref.polynomial(
                ctx,r,ref.exact(ctx,'1/20'),3)
            mean=np.zeros((16,16),complex)
            if len(case['events'])!=68: raise ValueError('Catalog must contain all68 events')
            for cid in case['events']:
                record=self.cache[cid]; mean+=record['event'].event_coefficient*record['block']
            return mean,gates.matrix_gate(mean,ref.to_numpy(target),2e-10,label='CATALOG_MEAN')
        if handler=='PRICE_DIRECT':
            event=self.cache[case['event']]['event']
            apps=[(x.component_id,x.role) for x in event.application_sequence]
            price=pricing.direct_literal_price(apps,self.price_tables['catalog'],
                                               event_count=1,scalar_reserves=1)
            return price,dict(literal_rz_tokens=price,probability=event.event_probability)
        if handler=='PRICE_DP':
            tail,_config,law=self.law('catalog')
            mass,value=pricing.expected_literal_price(self.price_tables['catalog'],
                {a.component_id:a.probability for a in tail.components},
                dict(zip(law.orders,law.order_probabilities,strict=True)),q=1,r=1,fixed_price=0)
            if abs(mass-1)>1e-12: raise ValueError('Price law mass')
            result={'double':value}
            for digits in (90,128):
                ctx=ref.precision_context(digits)
                _,_,orders,weights,norm,probs=ref.independent_law(ctx,tail.atoms,Fraction(1,20),2)
                pm={a.atom_id:p for a,p in zip(tail.atoms,probs,strict=True)}
                om={n:w/norm for n,w in zip(orders,weights,strict=True)}
                m,v=pricing.expected_literal_price(self.price_tables['catalog'],pm,om,q=1,r=1,fixed_price=0)
                if abs(m-1)>ctx.mpf('1e-60'): raise ValueError('High-precision price mass')
                result[str(digits)]=v
            return result,dict(expected_literal_rz_tokens=value,mass=mass)
        if handler=='PRICE_COMPARE':
            direct=sum(self.cache[cid]*self.cache[case['event_by_price'][cid]]['event'].event_probability
                       for cid in case['direct_prices'])
            dp=self.cache[case['dp']]; error=abs(direct-dp['double'])
            if error>1e-10*max(1,abs(direct)): raise ValueError('LITERAL_PRICE_DP_MISMATCH')
            high_precision={}
            tail,_,_=self.law('catalog'); indices={a.atom_id:i for i,a in enumerate(tail.atoms)}
            for digits in (90,128):
                ctx=ref.precision_context(digits)
                _,_,orders,weights,norm,probs=ref.independent_law(ctx,tail.atoms,Fraction(1,20),2)
                expected=ctx.mpf(0)
                for cid in case['direct_prices']:
                    e=self.cache[case['event_by_price'][cid]]['event']
                    weight=weights[orders.index(e.taylor_order)]/norm
                    for aid in (*e.product_component_ids,e.rotation_component_id): weight*=probs[indices[aid]]
                    expected+=weight*self.cache[cid]
                residual=abs(expected-dp[str(digits)])
                if residual>ctx.mpf('1e-40')*max(1,abs(expected)):
                    raise ValueError('HIGH_PRECISION_LITERAL_PRICE_DP_MISMATCH')
                high_precision[str(digits)]=str(residual)
            return direct,dict(expected_literal_rz_tokens=direct,error=error,
                               high_precision_residuals=high_precision)
        if handler=='ARM_PRICE':
            tail,_config,law=self.law(arm)
            # D boundaries fused: q+1 diagonal sweeps; literal CP is three RZ tokens.
            dcount=sum(Fraction(x)!=0 for x in self.spec.deterministic)
            fixed_d=3*dcount*(self.context.q+1)
            values={}
            for axis,readout in (('X',4),('Y',5)):
                mass,phi=pricing.expected_literal_price(self.price_tables[arm],
                    {a.component_id:a.probability for a in tail.components},
                    dict(zip(law.orders,law.order_probabilities,strict=True)),
                    q=self.context.q,r=self.context.r,fixed_price=fixed_d+readout)
                if abs(mass-1)>1e-12: raise ValueError('Price mass')
                values[axis]=phi
            from .intervals import score_enclosure
            qualification=self.contract.get('interval_provider_qualification',{})
            try:
                score,details=score_enclosure(tail.atoms,self.price_tables[arm],self.context,
                    fixed_x=fixed_d+4,fixed_y=fixed_d+5,
                    provider_identity=qualification.get('source_sha256',''),
                    qualification=qualification.get('status'))
                return score,dict(Phi=values,status='INTERVAL_PRICED',**details)
            except pricing.UnpricedError as error:
                return None,dict(Phi=values,status='UNPRICED_FALLBACK',reason=str(error))
        if handler in ('SELECTOR','EXHAUSTIVE'):
            def evaluate(a):
                value=self.cache[case['prices'][a]]
                if value is None: raise pricing.UnpricedError('Score interval not certified')
                return value
            result=pricing.select_dictionary(evaluate) if handler=='SELECTOR' else pricing.exhaustive_four(evaluate)
            return result,dict(status=result['status'],arm=result['arm'])
        raise ValueError('Unknown registered handler')


def execute_registered_f1(verified):
    engine=RegisteredF1(verified); output=verified['output']; done=set()
    uses={c['id']:0 for c in verified['inventory']['cases']}
    for item in verified['inventory']['cases']:
        for dep in item.get('requires',[]): uses[dep]+=1
    for case in verified['inventory']['cases']:
        # No automatic prerequisites or retries; preserve first failed attempt.
        if not set(case.get('requires',[])).issubset(done): raise ValueError('Missing prerequisite')
        for path,sha in verified['contract']['source_files'].items():
            if file_sha(_under(verified['source_root'],path).read_bytes())!=sha:
                raise ValueError('SOURCE changed during run')
        path=output/(case['id']+'.json')
        if path.exists(): raise ValueError('Duplicate output/case attempt')
        with path.open('x') as stream:
            stream.write(json.dumps(dict(status='ATTEMPTED',id=case['id']))+'\n')
        value,report=engine.run_case(case)
        engine.cache[case['id']]=value; done.add(case['id'])
        if isinstance(value,dict) and 'circuit' in value:
            from qiskit import qpy
            with (output/(case['id']+'.qpy')).open('xb') as stream:
                qpy.dump(value['circuit'],stream)
        path.write_text(json.dumps(dict(status='COMPLETED',id=case['id'],report=report),
                                   ensure_ascii=False,allow_nan=False)+'\n')
        # Release large reference/circuit objects after their final registered use.
        for dependency in case.get('requires',[]):
            uses[dependency]-=1
            if uses[dependency]==0: engine.cache.pop(dependency,None)
        if uses[case['id']]==0: engine.cache.pop(case['id'],None)
    for arm in ('SS','SP','PS','PP'):
        law=engine.law_coverage.get(arm)
        n=len(engine.atoms[arm])
        if law is None or len(law['keys'])!=n+n**3 or abs(math.fsum(law['probabilities'])-1)>1e-12:
            raise ValueError('Incomplete finite-event coverage: '+arm)
    return dict(completed_cases=len(done),status='REGISTERED_CASES_COMPLETED_REVIEW_REQUIRED',
                scientific_go=False,F2_authorized=False)
