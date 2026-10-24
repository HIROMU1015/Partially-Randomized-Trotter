"""Dormant PS48 executor; call only after the new scope's grant is claimed.

No law/Hamiltonian/reference regeneration. Saved QPY supplies the original
central angles; saved independently validated 32x32 targets supply the gates.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os
import resource
import shutil
import time
from . import f2_execution as shared
from .f2_plan import case_id, compile_groups
from .rectangular_preflight import cases


def bind_saved_angles(operations, ordered_atoms, scalar_phase):
    """Match the unchanged raw append_central recipe, not arbitrary RZ gates.

    Input consists of (name, physical_qubit_indices, parameters) tokens. Tests
    use artificial tokens; QPY is decoded only during approved execution.
    """
    found=[]
    for k in range(len(operations)-3):
        a,b,c,d=operations[k:k+4]
        if (a[0],b[0],c[0],d[0])!=('rz','cx','rz','cx'): continue
        target=a[1]
        if len(target)!=1 or b[1]!=(4,target[0]) or c[1]!=target or d[1]!=b[1]: continue
        angle=float(a[2][0])
        if float(c[2][0])!=-angle: raise ValueError('Changed central inverse angle')
        found.append((target[0],angle))
    nonidentity=[a for a in ordered_atoms if a.kind!='IDENTITY']
    if len(found)!=len(nonidentity): raise ValueError('Ambiguous saved central-angle binding')
    angles=[]
    for (target,value),atom in zip(found,nonidentity,strict=True):
        if target!=atom.support[-1]: raise ValueError('Saved central support/order mismatch')
        theta=value*(1 if atom.coefficient>0 else -1)
        if not math.isfinite(theta) or theta<=0: raise ValueError('Saved signed angle mismatch')
        angles.append(theta)
    if not angles or len({x.hex() for x in angles})!=1:
        raise ValueError('K0 rotations must bind the same saved positive angle')
    theta=angles[0]
    phases=[float(p[0]) for name,qubits,p in operations if name=='p' and qubits==(4,)]
    identity=-sum((1 if a.coefficient>0 else -1)*theta for a in ordered_atoms if a.kind=='IDENTITY')
    wanted=([identity] if any(a.kind=='IDENTITY' for a in ordered_atoms) and identity else [])+[scalar_phase]
    if len(phases)!=len(wanted) or any(abs(x-y)>1e-12 for x,y in zip(phases,wanted,strict=True)):
        raise ValueError('Saved identity/scalar phase mismatch')
    return theta


@dataclass(frozen=True)
class BoundK0Rotation:
    """Circuit-facing view of the fixed K0 event, not a regenerated distribution."""
    atom: object
    rotation_angle: float
    law_sha256: str

    @property
    def selected_component_ids(self): return (self.atom.atom_id,)

    @property
    def phase(self): return complex(1,0)

    @property
    def unsigned_rotation_angle(self):
        return self.rotation_angle*(1 if self.atom.coefficient>0 else -1)

    @property
    def application_sequence(self):
        return (SimpleNamespace(component_id=self.atom.atom_id,role='rotation',
            coefficient_sign=1 if self.atom.coefficient>0 else -1,
            is_identity=self.atom.kind=='IDENTITY',basis_id=self.atom.basis_key),)

    def to_dict(self):
        return dict(taylor_order=0,rotation_component_id=self.atom.atom_id,
            rotation_angle_hex=self.rotation_angle.hex(),law_sha256=self.law_sha256,
            coefficient_sign=self.application_sequence[0].coefficient_sign,phase=dict(real=1,imag=0))


class RectangularWorker(shared.Worker):
    def __init__(self,payload):
        from .rectangular import RectangularBasisProvider
        self.__dict__.update(payload)
        self.provider=RectangularBasisProvider(4,self.frames)
        self.provider._rectangular_cache=self.bases

    def build(self,job):
        import numpy as np
        from qiskit import qpy
        from qiskit.quantum_info import Operator
        from . import gates
        from .fixture import CompilerProfile
        from .circuits import build_evolution,hadamard_wrapper
        arm,i,j=job
        events=tuple(BoundK0Rotation(self.atoms[k],self.theta,self.law_sha256) for k in (i,j))
        evolution=build_evolution(self.spec,self.context,(events,),self.atoms,self.provider,
            input_identity=self.identity['input'],source_identity=self.identity['source'])
        rows=[]
        for axis in ('X','Y'):
            cid=case_id(arm,i,j,axis)
            baseline=json.loads((self.baseline/'build'/f'{cid}.json').read_text())
            circuit=hadamard_wrapper(evolution,self.spec,axis)
            target_path=self.baseline/'build'/f'{cid}.reference128.npy'
            target=np.load(target_path,allow_pickle=False)
            matrix=Operator(circuit.remove_final_measurements(inplace=False)).data
            report=gates.matrix_gate(matrix,target,1e-10,label='RAW_FULL_WRAPPER_NO_PHASE_FIT')
            report['unitarity']=gates.matrix_gate(matrix.conj().T@matrix,np.eye(32),1e-10,label='RAW_UNITARITY')
            error=abs(gates.wrapper_signal(circuit,axis)-baseline['reference_signal_component'])
            if error>2e-10: raise ValueError('RAW_SIGNAL_MISMATCH')
            with (self.out/'build'/f'{cid}.qpy').open('xb') as f:qpy.dump(circuit,f)
            shutil.copyfile(target_path,self.out/'build'/target_path.name)
            row=dict(id=cid,arm=arm,i=i,j=j,axis=axis,raw_operator=report,
                raw_signal_error=error,reference_precision_evaluations=0,
                reference_signal_component=baseline['reference_signal_component'],
                baseline_reference_agreement=baseline['reference_agreement'],
                raw_metrics=shared.circuit_metrics(circuit),
                raw_qpy_sha256=shared.hashlib.sha256((self.out/'build'/f'{cid}.qpy').read_bytes()).hexdigest(),
                compile_key=gates.compile_cache_key(circuit,CompilerProfile(),self.identity))
            shared.write_json(self.out/'build'/f'{cid}.json',row);rows.append(row)
        return rows


def initialize_worker(payload):
    # Reuse existing journal/STOP, build dispatch and compile implementation.
    shared._WORKER=RectangularWorker(payload)


def parallel(function,jobs,payload,workers):
    with shared.ProcessPoolExecutor(max_workers=workers,
            mp_context=shared.multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(payload,)) as pool:
        for rows in pool.map(function,jobs,chunksize=1): yield from rows


def execute(verified):
    """Future one-shot. Failure propagates; runner preserves partial results/STOP."""
    import numpy as np
    from qiskit import qpy
    from qiskit.quantum_info import Operator
    from .execution import _atom
    from .fixture import specification_from_record,EvolutionContext
    from .rectangular import RectangularBasisProvider
    from . import gates
    out=verified['output'];baseline=verified['baseline_root'];plan=verified['plan']
    start=time.monotonic();cpu=time.process_time()
    for name in ('attempts','build','compiled','basis'): (out/name).mkdir()
    bundle=json.loads((verified['saved_input_root']/'handoff/f1_bundle.json').read_bytes())
    spec=specification_from_record(bundle['specification']);context=EvolutionContext('1/5',1,2,0)
    atoms=tuple(_atom(a) for a in bundle['arms']['PS'])
    frames={k:tuple(tuple(complex(Fraction(x)) for x in row) for row in v)
        for k,v in bundle['rational_frames'].items()}
    angles={}
    for row in cases():
        cid=row['id']
        with (baseline/'build'/f'{cid}.qpy').open('rb') as f:loaded=qpy.load(f)
        if len(loaded)!=1: raise ValueError('One saved raw wrapper per QPY required')
        raw=loaded[0]
        if raw.num_qubits!=5 or raw.num_clbits!=1 or raw.metadata['axis']!=row['axis']:
            raise ValueError('Saved wrapper layout/axis changed')
        tokens=[(x.operation.name,tuple(raw.find_bit(q).index for q in x.qubits),
            tuple(x.operation.params)) for x in raw.data]
        angles[cid]=bind_saved_angles(tokens,[atoms[row['i']],atoms[row['j']]],
            -float(Fraction(spec.scalar))*float(Fraction(context.total_time)))
    if len({x.hex() for x in angles.values()})!=1: raise ValueError('Cross-case angle mismatch')
    theta=next(iter(angles.values()))
    shared.write_json(out/'angle_binding.json',dict(angle_hex=theta.hex(),
        cases={k:v.hex() for k,v in angles.items()},law_unchanged=True))
    provider=RectangularBasisProvider(4,frames);basis_reports={};indices=[1<<p for p in range(4)]
    for index in range(20,26):
        atom=atoms[index];g=provider.basis(atom,True)
        matrix=Operator(g).data;inverse=Operator(g.inverse()).data
        w=matrix[np.ix_(indices,indices)];v=np.asarray(frames[atom.frame_id])[:,list(atom.support)]
        t=provider.reference_support(atom,True);p=w[:,list(t)]@w[:,list(t)].conj().T
        report=dict(projector=gates.matrix_gate(p,v@v.conj().T,1e-10,label='SELECTED_PROJECTOR'),
            actual_adjoint=gates.matrix_gate(inverse,matrix.conj().T,1e-10,label='ACTUAL_ADJOINT'),
            unitarity=gates.matrix_gate(matrix.conj().T@matrix,np.eye(16),1e-10,label='BASIS_UNITARITY'))
        basis_reports[atom.atom_id]=report
        with (out/'basis'/f'{index}.qpy').open('xb') as f:qpy.dump(g,f)
        shared.write_json(out/'basis'/f'{index}.json',report)
    # Use available affinity and measured prior worker RSS, not a fixed small cap.
    affinity=len(os.sched_getaffinity(0))
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
        if x.startswith('MemAvailable:')))*1024
    capacity=min(affinity,max(1,available//(595704*1024)))
    identity=dict(input=plan['saved_B_inputs']['handoff/f1_bundle.json'],
        source=verified['authorization']['source_closure_sha256'],
        environment=plan['saved_B_inputs']['requirements.lock'])
    law_path=next(p for p in plan['public_files'] if p.endswith('/F2_laws.json'))
    payload=dict(spec=spec,context=context,atoms=atoms,frames=frames,baseline=baseline,
        out=out,identity=identity,theta=theta,law_sha256=plan['public_files'][law_path],
        bases=provider._rectangular_cache)
    jobs=tuple(dict.fromkeys(('PS',r['i'],r['j']) for r in cases()))
    build_workers=min(capacity,len(jobs))
    built=list(parallel(shared.build_job,jobs,payload,build_workers))
    groups=compile_groups(built);compile_workers=min(capacity,len(groups))
    shared.write_json(out/'resources.json',dict(cpu_affinity=affinity,MemAvailable_bytes=available,
        build_workers=build_workers,compile_workers=compile_workers,
        worker_RSS_estimate_KiB=595704,threads_per_worker=1,wall_time_cap=None,memory_cap=None))
    compiled=list(parallel(shared.compile_job,groups,payload,compile_workers))
    if {r['id'] for r in built}!={r['id'] for r in cases()} or len(compiled)!=48:
        raise ValueError('PS48 accounting mismatch')
    comparisons=[]
    for row in compiled:
        old=json.loads((baseline/'compiled'/f'{row["id"]}.json').read_text())
        comparisons.append(dict(id=row['id'],baseline=old['metrics'],candidate=row['metrics'],
            delta={k:row['metrics'][k]-old['metrics'][k] for k in old['metrics']}))
    shared.write_json(out/'CASE_COMPARISONS.json',comparisons)
    summary=dict(status='PS48_CORRECTNESS_PASS_LIMITED_CASE_COMPARISON_MANDATORY_STOP',
        completed_wrappers=len(built),compile_requests=len(compiled),
        actual_transpiles=sum(not r['cache_alias'] for r in compiled),
        new_reference_precision_evaluations=0,candidate_basis=len(basis_reports),
        basis_reports=basis_reports,wall_seconds=time.monotonic()-start,
        parent_cpu_seconds=time.process_time()-cpu,
        children_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_parent_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Selected cases only; no full-F2 expectation, certified N or superiority claim')
    shared.write_json(out/'summary.json',summary)
    return summary
