"""Dormant F2 pipeline. Science is imported only after a fresh F2 grant.

Build/reference -> exact full-IR deduplication -> unique full-wrapper compile
-> per-logical-case checks -> weighted finite-target resource accounting.
No F1 law, reference result, price, compile cache or authorization is reused.
"""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import multiprocessing
import os
import resource
import shutil
import time
from .f2_plan import ARMS,CONTEXT,case_id,iter_jobs,compile_groups,counts


def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False); f.write('\n')


def check_source(verified):
    for p,h in verified['contract']['source_files'].items():
        if hashlib.sha256((verified['source_root']/p).read_bytes()).hexdigest()!=h:
            raise ValueError('Frozen SOURCE changed: '+p)


def circuit_metrics(circuit):
    # Reuse existing metric definitions, but preserve the fixed P0 transpiler.
    from trotterlib.rte_compiled_cost import _gate_counts,_filtered_depth
    ops=dict(_gate_counts(circuit))
    return dict(rz_count=ops.get('rz',0),cx_count=ops.get('cx',0),
        rz_depth=_filtered_depth(circuit,'rz'),cx_depth=_filtered_depth(circuit,'cx'),
        total_depth=int(circuit.depth() or 0),circuit_size=int(circuit.size()),
        measurement_count=ops.get('measure',0),quantum_only_depth=int(
            circuit.remove_final_measurements(inplace=False).depth() or 0))


def _attempt(stage,identifier,function):
    out=_WORKER.out
    if (out/'STOP_REQUESTED.json').exists(): raise RuntimeError('Cooperative STOP')
    path=out/'attempts'/f'{stage}_{identifier}.jsonl'
    with path.open('x') as f:
        f.write(json.dumps(dict(status='ATTEMPTED',stage=stage,id=identifier))+'\n');f.flush()
        tick=time.monotonic();cpu=time.process_time()
        try:
            result=function()
            f.write(json.dumps(dict(status='COMPLETED',seconds=time.monotonic()-tick,
                cpu_seconds=time.process_time()-cpu,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
            return result
        except BaseException as error:
            f.write(json.dumps(dict(status='FAILED',error_type=type(error).__name__,error=str(error)))+'\n')
            try: write_json(out/'STOP_REQUESTED.json',dict(stage=stage,id=identifier,error=str(error)))
            except FileExistsError: pass
            raise


class Worker:
    def __init__(self,payload):
        from .circuits import BasisProvider
        self.__dict__.update(payload);self.contexts={};self.references={};self.taus={}
        self.provider=BasisProvider(self.spec.modes,self.frames)

    def reference_state(self,digits):
        from . import reference as ref
        if digits not in self.references:
            ctx=ref.precision_context(digits)
            h,d,r,fs,frames=ref.original_operators(ctx,self.spec)
            half=ctx.expm(-ctx.j*ref.exact(ctx,'1/5')*d/2,
                method='taylor' if digits==90 else 'pade')
            self.references[digits]=(ctx,frames,half)
        return self.references[digits]

    def trajectory_reference(self,arm,i,j,digits):
        from . import reference as ref
        ctx,frames,half=self.reference_state(digits)
        if arm=='diagnostic': return ctx,ctx.eye(16)
        key=(arm,digits)
        if key not in self.taus:
            self.taus[key]=ref.independent_law(ctx,self.atoms[arm],Fraction(1,10),0)[1]
            if any(abs(e.rotation_angle-float(ctx.atan(self.taus[key])))>1e-12 for e in self.events[arm]):
                raise ValueError('F2_ANGLE_MISMATCH')
        by_id={a.atom_id:a for a in self.atoms[arm]}
        step=ctx.eye(16)
        for k in (i,j):
            step=ref.event_operator(ctx,self.events[arm][k],by_id,frames,self.taus[key])*step
        u=ctx.exp(-ctx.j*ref.exact(ctx,self.spec.scalar)*ref.exact(ctx,'1/5'))*half*step*half
        return ctx,u

    def build(self,job):
        import numpy as np
        from qiskit import QuantumCircuit,qpy
        from qiskit.quantum_info import Operator
        from . import reference as ref,gates
        from .fixture import CompilerProfile,digest
        from .circuits import build_evolution,hadamard_wrapper,MixedEvolutionResult
        arm,i,j=job
        if arm=='diagnostic':
            evolution=MixedEvolutionResult(QuantumCircuit(5),4,4,self.context,((),),
                self.identity['input'],self.identity['source'],digest('prep_readout_only'),'diagnostic')
        else:
            evolution=build_evolution(self.spec,self.context,((self.events[arm][i],self.events[arm][j]),),
                self.atoms[arm],self.provider,input_identity=self.identity['input'],
                source_identity=self.identity['source'],optimized=True)
        refs={digits:self.trajectory_reference(arm,i,j,digits) for digits in (90,128)}
        rows=[]
        for axis in ('X','Y'):
            cid=case_id(arm,i,j,axis); circuit=hadamard_wrapper(evolution,self.spec,axis)
            matrices={d:ref.wrapper_operator(ctx,u,self.spec.occupied,axis) for d,(ctx,u) in refs.items()}
            agreement=gates.high_precision_agreement(matrices[90],matrices[128])
            zs={d:ref.signal(ctx,u,self.spec.occupied) for d,(ctx,u) in refs.items()}
            ctx=refs[128][0]; signal_agreement=abs(ctx.mpc(zs[90])-zs[128])
            if signal_agreement>ctx.mpf('1e-60'): raise ValueError('REFERENCE_SIGNAL_PRECISION_DISAGREEMENT')
            target=ref.to_numpy(matrices[128]); wanted=complex(zs[128]).real if axis=='X' else complex(zs[128]).imag
            raw=Operator(circuit.remove_final_measurements(inplace=False)).data
            report=gates.matrix_gate(raw,target,1e-10,label='RAW_FULL_WRAPPER')
            report['unitarity']=gates.matrix_gate(raw.conj().T@raw,np.eye(32),1e-10,label='RAW_UNITARITY')
            signal=gates.wrapper_signal(circuit,axis); error=abs(signal-wanted)
            if error>2e-10: raise ValueError('RAW_SIGNAL_MISMATCH')
            with (self.out/'build'/f'{cid}.qpy').open('xb') as f:qpy.dump(circuit,f)
            np.save(self.out/'build'/f'{cid}.reference128.npy',target,allow_pickle=False)
            if arm!='diagnostic' and axis=='X':
                # W_X=H C H P. Undo only the known prep/readout of the actual
                # checked full wrapper; no extra component/evolution build.
                h=np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2),np.eye(16))
                p=np.zeros((32,32),complex)
                mask=sum(1<<k for k in self.spec.occupied)
                for k in range(32):p[k^mask,k]=1
                control=h@raw@p@h
                np.save(self.out/'build'/f'{arm}_{i:03}_{j:03}.block.npy',control[16:,16:],allow_pickle=False)
            row=dict(id=cid,arm=arm,i=i,j=j,axis=axis,raw_operator=report,raw_signal_error=error,
                reference_precision_evaluations=2,reference_agreement=agreement,
                reference_signal_agreement=str(signal_agreement),reference_signal_component=wanted,
                raw_metrics=circuit_metrics(circuit),raw_qpy_sha256=hashlib.sha256(
                    (self.out/'build'/f'{cid}.qpy').read_bytes()).hexdigest(),compile_key=gates.compile_cache_key(
                    circuit,CompilerProfile(),self.identity))
            write_json(self.out/'build'/f'{cid}.json',row);rows.append(row)
        return rows

    def compile(self,group):
        import numpy as np
        from qiskit import qpy
        from qiskit.quantum_info import Operator
        from . import gates
        from .fixture import CompilerProfile
        key,ids=group
        record=json.loads((self.out/'build'/f'{ids[0]}.json').read_text())
        path=self.out/'build'/f'{ids[0]}.qpy'
        if record['compile_key']!=key or hashlib.sha256(path.read_bytes()).hexdigest()!=record['raw_qpy_sha256']:
            raise ValueError('Saved full IR integrity mismatch')
        # Preserve the original exact-IR key. QPY reserialization after loading
        # need not be bitwise canonical; actual operator/signal gates below are
        # the scientific checks on the loaded and compiled circuit.
        with path.open('rb') as f:raw=qpy.load(f)[0]
        started=time.monotonic()
        circuit=gates.compiled_wrapper(raw,CompilerProfile())
        compile_seconds=time.monotonic()-started
        with (self.out/'compiled'/f'{key}.qpy').open('xb') as f:qpy.dump(circuit,f)
        matrix=Operator(circuit.remove_final_measurements(inplace=False)).data
        metrics=circuit_metrics(circuit);rows=[]
        for index,cid in enumerate(ids):
            raw_record=json.loads((self.out/'build'/f'{cid}.json').read_text())
            # Every logical alias is independently checked against its saved
            # reference; matching IR is never used to skip a scientific gate.
            target=np.load(self.out/'build'/f'{cid}.reference128.npy',allow_pickle=False)
            report=gates.matrix_gate(matrix,target,5e-10,label='COMPILED_FULL_WRAPPER')
            report['unitarity']=gates.matrix_gate(matrix.conj().T@matrix,np.eye(32),1e-10,label='COMPILED_UNITARITY')
            signal=gates.wrapper_signal(circuit,raw_record['axis'])
            error=abs(signal-raw_record['reference_signal_component'])
            if error>5e-10: raise ValueError('COMPILED_SIGNAL_MISMATCH')
            row=dict(id=cid,arm=raw_record['arm'],axis=raw_record['axis'],i=raw_record['i'],j=raw_record['j'],
                compile_key=key,cache_alias=index>0,compile_seconds=compile_seconds if index==0 else 0,
                operator=report,signal_error=error,metrics=metrics)
            write_json(self.out/'compiled'/f'{cid}.json',row);rows.append(row)
        return rows


_WORKER=None
def initialize_worker(payload):
    global _WORKER
    _WORKER=Worker(payload)


def build_job(job):
    return _attempt('build',case_id(*job,'X')+'_pair',lambda:_WORKER.build(job))


def compile_job(group):
    return _attempt('compile',group[0],lambda:_WORKER.compile(group))


def run_parallel(function,jobs,payload,workers):
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(payload,)) as pool:
        # A worker journals actual attempts. Pending work observes STOP_REQUESTED.
        for rows in pool.map(function,jobs,chunksize=1):
            yield from rows


def aggregate_metrics(records,weights):
    """Exact weighted sum of integer saved costs; also exercised with mock data."""
    result={a:{b:{} for b in ('X','Y')} for a in ARMS};mass={a:{b:Fraction(0) for b in ('X','Y')} for a in ARMS}
    seen=set()
    for row in records:
        if row['id'] in seen: raise ValueError('Duplicate resource record')
        seen.add(row['id']);a=row['arm']
        if a=='diagnostic': continue
        b=row['axis'];weight=weights[a][row['i']]*weights[a][row['j']]
        mass[a][b]+=weight
        for name,count in row['metrics'].items():
            if type(count) is not int or count<0: raise ValueError('Invalid gate metric')
            result[a][b][name]=result[a][b].get(name,Fraction(0))+weight*count
    if any(m!=1 for axes in mass.values() for m in axes.values()): raise ValueError('Incomplete trajectory resource mass')
    return result


def execute_f2(verified):
    """Called by the runner only after verified, atomically consumed F2 approval."""
    import numpy as np
    from .fixture import specification_from_record,EvolutionContext
    from .execution import _atom
    from .prototype import make_tail,ordered_event
    from . import reference as ref,gates
    from .f2_statistics import qualified_shots
    from .pricing import ScoreInterval,select_dictionary,exhaustive_four
    from trotterlib.rte import finite_rte_distribution
    c=verified['contract'];out=verified['output'];check_source(verified)
    for name in ('attempts','build','compiled'): (out/name).mkdir()
    start=time.monotonic();cpu=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN);timings={}
    bundle=json.loads(verified['inputs']['handoff/f1_bundle.json'])
    spec=specification_from_record(bundle['specification']);context=EvolutionContext('1/5',1,2,0)
    atoms={a:tuple(_atom(x) for x in bundle['arms'][a]) for a in ARMS}
    frames={k:tuple(tuple(complex(Fraction(x)) for x in row) for row in v) for k,v in bundle['rational_frames'].items()}
    tails={};laws={};events={};weights={};ranges={};law_reports={}
    tick=time.monotonic()
    for a in ARMS:
        tail=make_tail(atoms[a],a);tails[a]=tail
        # Avoid configure(): its physical Taylor truncation acceptance budget is
        # not part of the adopted finite-target comparison. K=0 is explicit.
        law=finite_rte_distribution(tail.lambda_r*float(context.micro_time),0);laws[a]=law
        events[a]=tuple(ordered_event(tail,law,0,(),x.atom_id) for x in atoms[a])
        bits=tuple(Fraction.from_float(e.event_probability) for e in events[a]);mass=sum(bits)
        if any(x<=0 for x in bits) or abs(float(mass)-1)>1e-12:raise ValueError('F2 law mass')
        weights[a]=tuple(x/mass for x in bits)
        # Existing K=0 relation B_tot=B^2=1+(lambda*h)^2, no numerical evaluation
        # is performed during preparation. Actual outcomes are scaled by this.
        ranges[a]=1+(tail.exact_lambda*context.micro_time)**2
        if abs(law.exact_finite_distribution**2-float(ranges[a]))>1e-12*max(1,float(ranges[a])):
            raise ValueError('Btot finite-bit mismatch')
        l1=sum(abs(p-abs(x.coefficient)/tail.exact_lambda) for p,x in zip(weights[a],atoms[a],strict=True))
        if l1>Fraction('1e-12'):raise ValueError('F2_PROBABILITY_NORMALIZATION_L1')
        law_reports[a]=dict(lambda_exact=str(tail.exact_lambda),B_binary64=law.exact_finite_distribution,
            Btot_exact=str(ranges[a]),raw_probability_mass=str(mass),ideal_probability_L1=str(l1),
            finite_bit_probabilities=[str(x) for x in weights[a]],qualification='DIAGNOSTICS_NOT_A_CERTIFICATE')
    timings['law_and_finite_bit_setup']=time.monotonic()-tick
    write_json(out/'F2_laws.json',law_reports)
    tick=time.monotonic();common={}
    for d in (90,128):
        ctx=ref.precision_context(d);finite=ref.finite_target(ctx,spec,context)
        physical=ref.physical_target(ctx,spec,'1/5')
        common[d]=dict(ctx=ctx,finite=finite,physical=physical)
    reference_agreement={k:gates.high_precision_agreement(common[90][k],common[128][k]) for k in ('finite','physical')}
    finite=common[128]['finite'];ctx=common[128]['ctx']
    zf=ref.signal(ctx,finite,spec.occupied);zp=ref.signal(ctx,common[128]['physical'],spec.occupied)
    write_json(out/'common_reference.json',dict(precision_evaluations=4,agreement=reference_agreement,
        physical_bias=str(abs(zf-zp)),physical_bias_status='NUMERICAL_DIAGNOSTIC_ONLY',physical_same_accuracy_claim=False))
    timings['common_references']=time.monotonic()-tick
    affinity=len(os.sched_getaffinity(0))
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    workers=min(affinity,len(tuple(iter_jobs())),max(1,available//(595704*1024)))
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1',QISKIT_NUM_PROCS='1',RAYON_NUM_THREADS='1')
    write_json(out/'resources.json',dict(cpu_affinity=affinity,MemAvailable_bytes=available,
        workers=workers,worker_memory_estimate_from_F1_RSS_KiB=595704,threads_per_worker=1,
        wall_time_cap=None,memory_cap=None,disk_free_bytes=shutil.disk_usage(out).free))
    payload=dict(spec=spec,context=context,atoms=atoms,events=events,frames=frames,out=out,
        identity=dict(input=c['input_files']['handoff/f1_bundle.json'],source=c['source_closure_sha256'],
            environment=c['input_files']['requirements.lock']))
    check_source(verified);tick=time.monotonic()
    built=list(run_parallel(build_job,iter_jobs(),payload,workers))
    if (len(built)!=counts()['full_wrapper_build_requests'] or
            sum(x['reference_precision_evaluations'] for x in built)+4!=counts()['reference_precision_evaluations']):
        raise ValueError('Build/reference accounting mismatch')
    timings['build_and_wrapper_references']=time.monotonic()-tick
    groups=compile_groups(built);check_source(verified)
    write_json(out/'compile_inventory.json',dict(logical_requests=len(built),unique_misses=len(groups),
        groups=[dict(key=k,logical_ids=ids) for k,ids in groups]))
    tick=time.monotonic();compiled=list(run_parallel(compile_job,groups,payload,workers))
    if len(compiled)!=counts()['compile_requests']:raise ValueError('Compile accounting mismatch')
    timings['unique_compile_and_logical_checks']=time.monotonic()-tick
    means={};tick=time.monotonic()
    for a in ARMS:
        mean=np.zeros((16,16),complex)
        for i in range(len(atoms[a])):
            for j in range(len(atoms[a])):
                block=np.load(out/'build'/f'{a}_{i:03}_{j:03}.block.npy',allow_pickle=False)
                mean+=float(weights[a][i]*weights[a][j])*block
        mean*=float(ranges[a])
        means[a]=gates.matrix_gate(mean,ref.to_numpy(finite),2e-10,label='F2_FINITE_CORRECTED_MEAN')
        np.save(out/f'{a}_corrected_mean.npy',mean,allow_pickle=False)
    timings['four_actual_block_means']=time.monotonic()-tick
    write_json(out/'mean_checks.json',means)
    tick=time.monotonic();phis=aggregate_metrics(compiled,weights)
    shots={a:qualified_shots(ranges[a],c['statistics']) for a in ARMS}
    costs={a:{metric:shots[a]['N_X']*phis[a]['X'][metric]+shots[a]['N_Y']*phis[a]['Y'][metric]
        for metric in ('rz_count','cx_count')} for a in ARMS}
    timings['four_arm_statistical_and_resource_aggregation']=time.monotonic()-tick
    tick=time.monotonic()
    scores={a:ScoreInterval(costs[a]['rz_count'],costs[a]['rz_count'],c['source_closure_sha256'],True) for a in ARMS}
    selected=select_dictionary(scores.__getitem__)
    timings['conditional_selector']=time.monotonic()-tick;tick=time.monotonic()
    exhaustive=exhaustive_four(scores.__getitem__)
    timings['known_four_candidate_search']=time.monotonic()-tick
    check_source(verified)
    result=dict(status='COMPLETED_F2_FINITE_TARGET_CONDITIONAL_REVIEW_REQUIRED_MANDATORY_STOP',
        counts=counts(),actual_compile_misses=len(groups),compile_aliases=len(built)-len(groups),
        Phi={a:{axis:{m:str(v) for m,v in values.items()} for axis,values in axes.items()} for a,axes in phis.items()},
        N=shots,weighted_costs={a:{m:str(v) for m,v in x.items()} for a,x in costs.items()},
        selector=dict(status=selected['status'],arm=selected['arm']),
        known_four=dict(status=exhaustive['status'],arm=exhaustive['arm']),
        comparison_scope='FINITE_TARGET_CONDITIONAL_ON_ADOPTED_QUALIFICATIONS',
        physical_same_accuracy_claim=False,novelty_status='INDEPENDENT_METHOD_DELTA_UNESTABLISHED',
        dictionary_generation='NOT_REEXECUTED_SAVED_B_INPUT_REUSED',timings=timings,
        main_process_CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        worker_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-children.ru_utime-children.ru_stime,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        quantum_shots_drawn=0,scientific_go=False,next_owner='USER_GPT',mandatory_stop=True)
    write_json(out/'summary.json',result)
    return result
