"""Execution-only adapter; unexecuted scientific code in F1-PRE.

Reuses I gates/router/pricing and independent CAR reference primitives.
Restores the B law byte-for-byte at binary64, without rebinding or sampling.
Five independent input partitions share immutable inputs, not mutable contexts.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import multiprocessing
import os
import time
from .execution import RegisteredF1
from .prototype import MixedTailView, atom_record
from . import reference as ref, gates, pricing
from .fixture import digest
from .f1_saved_plan import ARMS


def decode_saved(value):
    if isinstance(value,dict):
        if set(value)=={'encoding','value'} and value['encoding']=='ieee754-binary64-hex':
            result=float.fromhex(value['value'])
            if not math.isfinite(result): raise ValueError('Nonfinite B value')
            return result
        return {k:decode_saved(v) for k,v in value.items()}
    if isinstance(value,list): return tuple(decode_saved(v) for v in value)
    return value


class SavedF1(RegisteredF1):
    def __init__(self,verified):
        super().__init__(verified)
        self.reference_contexts={}; self.reference_originals={}; self.independent_laws={}
        self.price_binding_reports={}; self.motif_cache={}; self.macro_cache={}; self.catalog_events={}

    def event(self,arm,record):
        if arm!='catalog': return super().event(arm,record)
        key=(tuple(record['products']),record['child'])
        if key not in self.catalog_events: self.catalog_events[key]=super().event(arm,record)
        return self.catalog_events[key]

    def macros(self,arm,records):
        key=(arm,tuple(tuple((tuple(e['products']),e['child']) for e in macro) for macro in records))
        if key not in self.macro_cache: self.macro_cache[key]=super().macros(arm,records)
        return self.macro_cache[key]

    def law(self,arm):
        if arm not in self.contexts:
            from trotterlib.rte import RTEComponent, RTEConfig, RTEFiniteDistribution
            payload=decode_saved(json.loads(self.verified['inputs'][f'laws/{arm}.json']))['payload']
            config=RTEConfig(**payload['config'])
            distribution=RTEFiniteDistribution(**payload['distribution'])
            components=tuple(RTEComponent(**c) for c in payload['components'])
            atoms=self.atoms[arm]
            if tuple(c.component_id for c in components)!=tuple(a.atom_id for a in atoms):
                raise ValueError('B atom/component order mismatch')
            if digest([atom_record(a) for a in atoms])!=config.tail_hash:
                raise ValueError('B dictionary identity mismatch')
            tail=MixedTailView(config.tail_id,config.tail_hash,config.lambda_r,
                               components,atoms,Fraction(payload['exact_lambda']))
            self.contexts[arm]=(tail,config,distribution)
        return self.contexts[arm]

    def event_record(self,arm,ordinal):
        ids=tuple(a.atom_id for a in self.atoms[arm]); n=len(ids)
        if type(ordinal) is not int or not 0<=ordinal<n+n**3:
            raise ValueError('Event ordinal outside registered law')
        if ordinal<n: return dict(products=(),child=ids[ordinal])
        child,remainder=divmod(ordinal-n,n*n); p1,p2=divmod(remainder,n)
        return dict(products=(ids[p1],ids[p2]),child=ids[child])

    def resolve_case(self,record):
        case=dict(record); arm=case.get('arm')
        if 'event_ordinal' in case: case['event']=self.event_record(arm,case['event_ordinal'])
        if 'atom_index' in case: case['atom_id']=self.atoms[arm][case['atom_index']].atom_id
        if 'motif_index' in case:
            from .cases import motif_events
            key=(arm,case['motif_index'])
            if key not in self.motif_cache:
                tail,_,law=self.law(arm)
                self.motif_cache[key]=motif_events(tail,law,case['motif_index'])
            events=self.motif_cache[key]
            case['macros']=tuple(tuple(dict(products=e.product_component_ids,
                child=e.rotation_component_id) for e in macro) for macro in events)
            macrokey=(arm,tuple(tuple((tuple(e['products']),e['child']) for e in macro)
                                for macro in case['macros']))
            self.macro_cache[macrokey]=events
        return case

    def reference_case(self,case):
        digits=case['digits']; arm=case['arm']; atoms=self.atoms[arm]
        if digits not in self.reference_contexts:
            self.reference_contexts[digits]=ref.precision_context(digits)
            self.reference_originals[digits]=ref.original_operators(self.reference_contexts[digits],self.spec)
        ctx=self.reference_contexts[digits]
        h,d,r,fs,frames=self.reference_originals[digits]
        result=dict(ctx=ctx,h=h,d=d,r=r,fs=fs,frames=frames)
        kind=case['reference_kind']; method='taylor' if digits==90 else 'pade'
        if kind=='target':
            half=ctx.expm(-ctx.j*ref.exact(ctx,str(self.context.delta))*d/2,method=method)
            step=ref.polynomial(ctx,r,ref.exact(ctx,str(self.context.micro_time)),3)
            one=half*(step**self.context.r)*half
            result.update(step=step,operator=ctx.exp(-ctx.j*ref.exact(ctx,self.spec.scalar)*
                ref.exact(ctx,self.context.total_time))*(one**self.context.q))
        else:
            lawkey=(arm,digits)
            if lawkey not in self.independent_laws:
                self.independent_laws[lawkey]=ref.independent_law(ctx,atoms,Fraction(1,20),2)
            _,tau,_,_,_,_=self.independent_laws[lawkey]
            by_id={a.atom_id:a for a in atoms}
            if kind=='motif':
                macros=self.macros(arm,case['macros'])
                half=ctx.expm(-ctx.j*ref.exact(ctx,str(self.context.delta))*d/2,method=method)
                u=ctx.eye(1<<self.spec.modes)
                for macro in macros:
                    step=ctx.eye(u.rows)
                    for e in macro: step=ref.event_operator(ctx,e,by_id,frames,tau)*step
                    u=half*step*half*u
                u=ctx.exp(-ctx.j*ref.exact(ctx,self.spec.scalar)*ref.exact(ctx,self.context.total_time))*u
                result.update(operator=u,signal=ref.signal(ctx,u,self.spec.occupied),
                    wrapper_X=ref.wrapper_operator(ctx,u,self.spec.occupied,'X'),
                    wrapper_Y=ref.wrapper_operator(ctx,u,self.spec.occupied,'Y'))
            elif kind=='atom':
                atom=by_id[case['atom_id']]; q=ref.unsigned_atom_operator(ctx,atom,frames)
                sign=1 if atom.coefficient>0 else -1
                result['product']=sign*q
                for n in (0,2):
                    angle=sign*ctx.atan(tau/(n+1))
                    result['rotation'+str(n)]=ctx.cos(angle)*ctx.eye(q.rows)-ctx.j*ctx.sin(angle)*q
            elif kind=='catalog':
                e=self.event(arm,case['event'])
                result['operator']=ctx.exp(ctx.j*ref.exact(ctx,'1/620'))*ref.event_operator(ctx,e,by_id,frames,tau)
            else: raise ValueError('Unknown reference kind')
        agreement={}
        if case.get('paired_reference'):
            other=self.cache[case['paired_reference']]
            for key in ('h','d','r','operator','step','product','rotation0','rotation2','wrapper_X','wrapper_Y'):
                if key in result and key in other:
                    agreement[key]=gates.high_precision_agreement(other[key],result[key])
            if 'signal' in result:
                error=abs(ctx.mpc(other['signal'])-result['signal'])
                if error>ctx.mpf('1e-60'): raise ValueError('REFERENCE_SIGNAL_PRECISION_DISAGREEMENT')
                agreement['signal']=str(error)
        return result,dict(reference_kind=kind,digits=digits,
                           agreement=agreement,status='REFERENCE_COMPUTED')

    def law_case(self,case):
        arm=case['arm']; tail,config,law=self.law(arm)
        reference=self.cache[case['reference']]; ctx=reference['ctx']
        key=(arm,ctx.dps)
        if key not in self.independent_laws:
            self.independent_laws[key]=ref.independent_law(ctx,tail.atoms,Fraction(1,20),2)
        lam,tau,orders,weights,norm,p=self.independent_laws[key]
        state=self.law_coverage.get(arm)
        if state is None:
            raw=[c.probability for c in tail.components]; mass=math.fsum(raw)
            if any(x<=0 or not math.isfinite(x) for x in raw): raise ValueError('Invalid saved component probability')
            l1=math.fsum(abs(x/mass-float(y)) for x,y in zip(raw,p,strict=True))
            if abs(mass-1)>1e-12 or l1>1e-12: raise ValueError('COMPONENT_PROBABILITY_NORMALIZATION')
            if abs(config.lambda_r-float(lam))>1e-12*max(1,float(lam)):
                raise ValueError('LAW_LAMBDA_MISMATCH')
            if abs(law.exact_finite_distribution-float(norm))>1e-12*max(1,float(norm)):
                raise ValueError('LAW_NORMALIZER_MISMATCH')
            for n,w,prob in zip(orders,weights,law.order_probabilities,strict=True):
                if not math.isfinite(prob) or prob<0 or abs(prob-float(w/norm))>1e-12:
                    raise ValueError('LAW_ORDER_PROBABILITY')
            state=dict(keys=set(),probabilities=[],probability_errors=[],indices={a.atom_id:i for i,a in enumerate(tail.atoms)},
                       component_mass=mass,normalization_l1=l1)
            self.law_coverage[arm]=state
        e=self.event(arm,case['event'])
        weight=weights[orders.index(e.taylor_order)]
        for aid in (*e.product_component_ids,e.rotation_component_id): weight*=p[state['indices'][aid]]
        if not all(math.isfinite(x) and x>=0 for x in (e.event_probability,e.event_coefficient)):
            raise ValueError('Nonfinite/negative event law')
        ep=abs(e.event_probability-float(weight/norm)); ec=abs(e.event_coefficient-float(weight))
        if ep>1e-12 or ec>1e-12*max(1,abs(float(weight))): raise ValueError('LAW_WEIGHT_MISMATCH')
        if abs(e.rotation_angle-float(ctx.atan(tau/(e.taylor_order+1))))>1e-12:
            raise ValueError('LAW_ANGLE_MISMATCH')
        if e.phase!=complex((-1)**(e.taylor_order//2)):
            raise ValueError('LAW_PAIRED_PHASE')
        expected_ids=(*e.product_component_ids,e.rotation_component_id)
        if e.selected_component_ids!=expected_ids or tuple(a.component_id for a in e.application_sequence)!=expected_ids:
            raise ValueError('LAW_APPLICATION_ORDER')
        k=(e.taylor_order,e.product_component_ids,e.rotation_component_id)
        if k in state['keys']: raise ValueError('Duplicate finite event')
        state['keys'].add(k); state['probabilities'].append(e.event_probability)
        state['probability_errors'].append(ep)
        # O(E) total summation, replacing I's O(E^2) repeated prefix fsum.
        return None,dict(probability_error=ep,coefficient_error=ec,
            probability=e.event_probability,coefficient=e.event_coefficient,
            products=e.product_component_ids,child=e.rotation_component_id,
            order=e.taylor_order,angle=e.rotation_angle,phase=[e.phase.real,e.phase.imag])

    def bind_prices(self,arm):
        if arm not in self.price_tables:
            before=len(self.provider._cache)
            self.price_tables[arm]=pricing.recipe_prices(self.atoms[arm],self.provider)
            for price in self.price_tables[arm].values(): price.validate()
            from dataclasses import asdict
            self.price_binding_reports[arm]=dict(status='LITERAL_RECIPE_BOUND',
                tables={k:asdict(v) for k,v in self.price_tables[arm].items()},
                basis_cache_entries_before=before,basis_cache_entries_after=len(self.provider._cache))

    def run_case(self,record):
        case=self.resolve_case(record); handler=case['handler']; arm=case.get('arm')
        if handler=='LAW': return self.law_case(case)
        if handler in ('PRICE_DIRECT','PRICE_DP','PRICE_COMPARE','ARM_PRICE'):
            self.bind_prices(arm)
        compile_entries_before=len(self.compile_cache)
        value,report=super().run_case(case)
        if handler=='COMPILE':
            report['compile_cache_hit']=len(self.compile_cache)==compile_entries_before
            report['full_IR_cache_keys']=sorted(self.compile_cache)
        if handler in ('MOTIF','WRAPPER','COMPILE','CATALOG_EVENT'):
            report['literal_gate_counts']={str(k):int(v) for k,v in value['circuit'].count_ops().items()}
            report['depth']=value['circuit'].depth()
        if handler=='COMPONENT' and case['role']=='product':
            import numpy as np
            q=value['block']
            report['Hermiticity']=gates.matrix_gate(q,q.conj().T,1e-11,label='COMPONENT_HERMITICITY')
            report['involution']=gates.matrix_gate(q@q,np.eye(len(q)),1e-11,label='COMPONENT_INVOLUTION')
        if handler=='H_ID':
            import numpy as np
            actual=ref.to_numpy(value)
            report['Hermiticity']=gates.matrix_gate(actual,actual.conj().T,1e-11,label='H_HERMITICITY')
            reference=self.cache[case['reference']]; ctx=reference['ctx']; fragments={}
            physical=json.loads(self.verified['inputs']['inputs/physical.json'])['payload']
            from .execution import _atom
            alternatives={item['fragment']:item for item in physical['typed_alternatives']}
            for idx,f in enumerate(self.spec.fragments):
                # Compare each saved S/P fragment to original independent CAR f^2.
                for representation in ('S','P'):
                    records=alternatives[f.fragment_id][representation]
                    atoms=tuple(_atom(x) for x in records)
                    operator=ref.dictionary_operator(ctx,atoms,reference['frames'])
                    fragments[f.fragment_id+'_'+representation]=gates.matrix_gate(
                        ref.to_numpy(operator),ref.to_numpy(reference['fs'][idx]),1e-11,label='FRAGMENT_ID')
            report['fragments']=fragments
        if handler in ('PRICE_DIRECT','PRICE_DP','ARM_PRICE'):
            report['price_binding']=self.price_binding_reports.get(arm)
        return value,report


def check_source(verified):
    root=verified['source_root']
    for path,sha in verified['contract']['source_files'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=sha:
            raise ValueError('SOURCE changed: '+path)


def execute_partition(verified):
    """One input partition; all attempts preserved in one append-only journal."""
    check_source(verified); engine=SavedF1(verified); output=verified['output']
    output.mkdir(exist_ok=False); done=set(); cases=verified['inventory']['cases']
    uses={c['id']:0 for c in cases}
    for c in cases:
        for dep in c['requires']: uses[dep]+=1
    started=time.monotonic(); cpu_started=time.process_time(); family=None; prices={}; completed={}
    with (output/'journal.jsonl').open('x') as journal:
        def log(record):
            journal.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); journal.flush()
        try:
            for case in cases:
                if (output.parent/'STOP_REQUESTED.json').exists():
                    raise RuntimeError('Another partition failed; no further scientific attempt')
                if family!=case['handler']:
                    check_source(verified); family=case['handler']
                if not set(case['requires']).issubset(done): raise ValueError('Missing prerequisite')
                log(dict(id=case['id'],status='ATTEMPTED')); tick=time.monotonic()
                value,report=engine.run_case(case)
                if isinstance(value,dict) and 'circuit' in value:
                    from qiskit import qpy
                    with (output/(case['id']+'.qpy')).open('xb') as stream: qpy.dump(value['circuit'],stream)
                if case['handler']=='REFERENCE':
                    ctx=value['ctx']; digits=case['digits']
                    matrices={key:[[ctx.nstr(matrix[i,j],digits) for j in range(matrix.cols)]
                        for i in range(matrix.rows)] for key,matrix in value.items()
                        if hasattr(matrix,'rows') and hasattr(matrix,'cols')}
                    with (output/(case['id']+'.reference.json')).open('x') as f:
                        json.dump(dict(digits=digits,matrices=matrices),f)
                if case['handler'] in ('H_ID','MEAN','CATALOG_MEAN','COMPONENT','CATALOG_EVENT'):
                    import numpy as np
                    matrix=value['block'] if isinstance(value,dict) else (
                        ref.to_numpy(value) if hasattr(value,'rows') else value)
                    with (output/(case['id']+'.npy')).open('xb') as f: np.save(f,matrix,allow_pickle=False)
                if case['handler']=='ARM_PRICE': prices[case['id']]=value
                engine.cache[case['id']]=value; done.add(case['id'])
                completed[case['call_group']]=completed.get(case['call_group'],0)+1
                log(dict(id=case['id'],status='COMPLETED',seconds=time.monotonic()-tick,report=report))
                for dep in case['requires']:
                    uses[dep]-=1
                    if uses[dep]==0: engine.cache.pop(dep,None)
                if uses[case['id']]==0: engine.cache.pop(case['id'],None)
            coverage={}
            for arm,state in engine.law_coverage.items():
                n=len(engine.atoms[arm]); mass=math.fsum(state['probabilities'])
                joint_l1=math.fsum(state['probability_errors'])
                if len(state['keys'])!=n+n**3 or abs(mass-1)>1e-12 or joint_l1>1e-12:
                    raise ValueError('Incomplete finite event law: '+arm)
                coverage[arm]=dict(count=len(state['keys']),mass=mass,
                    component_mass=state['component_mass'],normalization_l1=state['normalization_l1'],
                    joint_probability_L1=joint_l1)
            check_source(verified)
            import resource
            result=dict(status='PARTITION_COMPLETED_REVIEW_REQUIRED',counts=completed,
                coverage=coverage,seconds=time.monotonic()-started,
                cpu_seconds=time.process_time()-cpu_started,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prices=prices)
            log(dict(status='PARTITION_COMPLETE',**{k:v for k,v in result.items() if k!='status'}))
            return result
        except BaseException as error:
            log(dict(status='FAILED_NO_RETRY',error_type=type(error).__name__,error=str(error)))
            try:
                with (output.parent/'STOP_REQUESTED.json').open('x') as f:
                    json.dump(dict(partition=output.name,error_type=type(error).__name__,
                                   error=str(error)),f)
            except FileExistsError:
                pass
            raise


def execute_saved_f1(verified):
    from .f1_saved_plan import iter_cases
    if verified['inventory']['cases']!=list(iter_cases()): raise ValueError('Unregistered case plan')
    # Five real independent datasets, not an arbitrary CPU cap. BLAS/Rayon are
    # coordinated with process parallelism; no scientific time/memory cap.
    partitions={a:[] for a in (*ARMS,'catalog')}; final=[]
    for case in verified['inventory']['cases']:
        (partitions[case['arm']] if case.get('arm') in partitions else final).append(case)
    affinity=len(os.sched_getaffinity(0))
    workers=min(len(partitions),affinity)
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                      NUMBA_NUM_THREADS='1',QISKIT_NUM_PROCS='1',
                      RAYON_NUM_THREADS=str(max(1,affinity//workers)))
    resource=dict(cpu_affinity=affinity,independent_partitions=len(partitions),workers=workers,
        blas_threads_per_worker=1,rayon_threads_per_worker=max(1,affinity//workers),
        wall_time_cap=None,memory_cap=None,worker_rule='min(independent_datasets,available_cpu_affinity)')
    with (verified['output']/'resources.json').open('x') as f: json.dump(resource,f)
    outputs={}; allcounts={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={}
        for arm,cases in partitions.items():
            child=dict(verified,inventory=dict(cases=cases),output=verified['output']/arm)
            futures[pool.submit(execute_partition,child)]=arm
        for future in as_completed(futures):
            try: outputs[futures[future]]=future.result()
            except BaseException:
                for pending in futures: pending.cancel()
                raise
    # Unqualified scalar provider is deliberately explicit; no certified score
    # crosses process boundaries. A later qualified provider needs new approval.
    engine=SavedF1(verified)
    for result in outputs.values():
        engine.cache.update(result['prices'])
        for group,count in result['counts'].items(): allcounts[group]=allcounts.get(group,0)+count
    with (verified['output']/'selection.jsonl').open('x') as f:
        for case in final:
            f.write(json.dumps(dict(id=case['id'],status='ATTEMPTED'))+'\n'); f.flush()
            value,report=engine.run_case(case)
            f.write(json.dumps(dict(id=case['id'],status='COMPLETED',report=report))+'\n'); f.flush()
            allcounts[case['call_group']]=allcounts.get(case['call_group'],0)+1
    if allcounts!=verified['contract']['logical_case_counts']: raise ValueError('Final call accounting mismatch')
    check_source(verified)
    result=dict(status='COMPLETED_MANDATORY_STOP_F1_REVIEW_REQUIRED',counts=allcounts,
        scientific_go=False,F2_authorized=False,interval_status='UNPRICED_FALLBACK',
        shot_status='SHOT_COUNT_UNRESOLVED',full_P0_price_qualification=False)
    with (verified['output']/'summary.json').open('x') as f: json.dump(result,f)
    return result
